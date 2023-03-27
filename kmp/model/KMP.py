#!/usr/bin/env python3

import logging
import math
import numpy as np

from scipy.linalg import block_diag

class KMP:
    """Trajectory imitation and adaptation using Kernelized Movement Primitives.

    Parameters
    ----------
    l : int, default=1
        Lambda regularization factor for the mean minimization problem.
    lc : int, default=60
        Lambda_c regularization factor for the covariance minimization problem.
    tol : float, default=2e-3
        Tolerance for the discrimination of conflicting points.
    kernel_delta : float, default=0.001
        Coefficient for the discrete derivative.
    kernel_gamma : int, default=6
        Coefficient for the rbf kernel. 
    """
    def __init__(self, l=1, lc=100, tol=0.002, kernel_delta=0.001, kernel_gamma=6) -> None:
        self.__logger = logging.getLogger(__name__)
        self.trained = False
        self.l = l
        self.lc = lc
        self.tol = tol
        self.kernel_delta = kernel_delta
        self.kernel_gamma = kernel_gamma

    def set_waypoint(self, s, xi, sigma):
        """Adds a waypoint to the database, checking for conflicts.

        Parameters
        ----------
        s : array-like of shape (n_input_features,n_samples)
            Array of input vectors.
        xi : array-like of shape (n_output_features,n_samples)
            Array of output vectors
        sigma : array-like of shape (n_output_features,n_output_features,n_samples)
            Array of covariance matrices
        """
        # Loop over the reference database to find any conflicts
        min_dist = math.inf
        for i in range(self.N):
            dist = np.linalg.norm(self.s[:,i]-s)
            if  dist < min_dist:
                min_dist = dist
                id = i
        if min_dist < self.tol:
            # Replace the conflicting point
            self.s[:,id] = s
            self.xi[:,id] = xi
            self.sigma[:,:,id] = sigma
        else:
            # Add the new point to the database
            self.s = np.append(self.s, s)
            self.xi = np.append(self.xi, xi)
            self.sigma = np.append(self.sigma, sigma)
        self.N = self.s.shape[1]
        # Refit the model with the new data
        self.fit(self.s, self.xi, self.sigma)

    def __kernel_matrix(self, s1, s2):
        """Computes the kernel matrix for the given input pair.

        Parameters
        ----------
        s1 : array-like of shape (n_features)
            The first input.
        s2 : array-like of shape (n_features)
            The second input.

        Returns
        -------
        kernel : array-like of shape (n_features,n_features)
            The kernel matrix evaluated in the provided input pair.
        """
        # Radial basis function kernel
        s1dt = s1 + self.kernel_delta
        s2dt = s2 + self.kernel_delta
        kt_t = np.exp(-self.kernel_gamma*(s1-s2)**2)
        kt_dt_tmp = np.exp(-self.kernel_gamma*(s1-s2dt)**2)
        kdt_t_tmp = np.exp(-self.kernel_gamma*(s1dt-s2)**2)
        kdt_dt_tmp = np.exp(-self.kernel_gamma*(s1dt-s2dt)**2)
        kt_dt = (kt_dt_tmp - kt_t)/self.kernel_delta
        kdt_t = (kdt_t_tmp - kt_t)/self.kernel_delta
        kdt_dt = (kdt_dt_tmp - kdt_t_tmp - kt_dt_tmp + kt_t)/self.kernel_delta**2       
        kernel = np.zeros((self.O,self.O))
        dim2 = int(self.O/2)
        for i in range(dim2):
            kernel[i,i] = kt_t
            kernel[i,i+dim2] = 0#kt_dt
            kernel[i+dim2,i] = 0#kdt_t
            kernel[i+dim2,i+dim2] = kdt_dt
        return kernel
                
    def fit(self, X, Y, var):
        """Train the model by computing the estimator matrices for the mean (K+lambda*sigma)^-1 and 
        for the covariance (K+lambda_c*sigma)^-1.

        Parameters
        ----------
        X : array-like of shape (n_input_features,n_samples)
            Array of input vectors.
        Y : array-like of shape (n_output_features,n_samples)
            Array of output vectors
        var : array-like of shape (n_output_features,n_output_features,n_samples)
            Array of covariance matrices
        """
        self.s = X
        self.xi = Y
        self.sigma = var
        self.O = self.xi.shape[0]
        self.N = self.xi.shape[1]
        k_mean = np.zeros((self.N*self.O,self.N*self.O))
        k_covariance = np.zeros((self.N*self.O,self.N*self.O))
        for i in range(self.N):
            for j in range(i+1):
                kernel = self.__kernel_matrix(self.s[:,i],self.s[:,j])
                k_mean[i*self.O:(i+1)*self.O,j*self.O:(j+1)*self.O] = kernel
                k_covariance[i*self.O:(i+1)*self.O,j*self.O:(j+1)*self.O] = kernel
                if i != j:
                    k_mean[j*self.O:(j+1)*self.O,i*self.O:(i+1)*self.O] = kernel
                    k_covariance[j*self.O:(j+1)*self.O,i*self.O:(i+1)*self.O] = kernel
        sigma_diag = block_diag(*[self.sigma[:,:,i] for i in range(self.sigma.shape[2])])
        k_mean += self.l*sigma_diag
        k_covariance += self.lc*sigma_diag
        self.__mean_estimator = np.linalg.inv(k_mean)
        self.__covariance_estimator = np.linalg.inv(k_covariance)
    
    def predict(self, s):
        """Carry out a prediction on the mean and covariance associated to the given input.

        Parameters
        ----------
        s : array-like of shape (n_features,n_samples)
            The set of inputs to make a prediction of.

        Returns
        -------
        xi : array-like of shape (n_features,n_samples)
            The array of predicted means.

        sigma : array-like of shape (n_features,n_features,n_samples)
            The array of predicted covariance matrices.
        """
        xi = np.zeros((self.O,s.shape[1]))
        sigma = np.zeros((self.O,self.O,s.shape[1]))
        for j in range(s.shape[1]):
            k = np.zeros((self.O,self.N*self.O))
            Y = np.zeros(self.N*self.O)
            for i in range(self.N):
                k[:,i*self.O:(i+1)*self.O] = self.__kernel_matrix(s[:,j],self.s[:,i])
                Y[i*self.O:i*self.O+self.O] = self.xi[:self.O,i]
            xi[:,j] = k@self.__mean_estimator@Y
            sigma[:,:,j] = (self.N/self.lc)*(self.__kernel_matrix(s[:,j],s[:,j]) - k@self.__covariance_estimator@k.T)
        self.__logger.info('KMP Done.')
        return xi, sigma