#!/usr/bin/env python3

import logging
import math
import numpy as np

class KMP:
    """Trajectory imitation and adaptation using Kernelized Movement Primitives.

    Parameters
    ----------
    l : int, default=0.5
        Lambda regularization factor for the mean minimization problem.
    lc : int, default=100
        Lambda_c regularization factor for the covariance minimization problem.
    tol : float, default=0.0005
        Tolerance for the discrimination of conflicting points.
    kernel_gamma : int, default=6
        Coefficient for the rbf kernel. 
    priorities : array-like of shape (n_trajectories,), default=None
        Functions that map the input space into a priority value for trajectory superposition.
    """
    def __init__(self, l=0.5, lc=100, tol=0.0005, kernel_gamma=6, priorities=None) -> None:
        self.__logger = logging.getLogger(__name__)
        self.trained = False
        self.l = l
        self.lc = lc
        self.tol = tol
        self.kernel_gamma = kernel_gamma
        self.priorities = priorities

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
        for j in range(len(s)):
            # Loop over the reference database to find any conflicts
            min_dist = math.inf
            for i in range(self.N):
                dist = np.linalg.norm(self.s[:,i]-s[j])
                if  dist < min_dist:
                    min_dist = dist
                    id = i
            if min_dist < self.tol:
                # Replace the conflicting point
                self.s[:,id] = s[j]
                self.xi[:,id] = xi[j]
                self.sigma[:,:,id] = sigma[j]
            else:
                # Add the new point to the database
                self.s = np.append(self.s, s[j])
                self.xi = np.append(self.xi, xi[j])
                self.sigma = np.append(self.sigma, sigma[j])
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
        # Note that we use only the upper 2x2 part of the kernel matrix defined in the paper, because
        # we only consider the position as output, in order to dynamically set waypoints. If velocity
        # adaptation is needed, this function should be rewritten accordingly.
        kernel = np.eye(self.O)*np.exp(-self.kernel_gamma*(s1-s2)**2)
        return kernel
                
    def fit(self, X, Y, var):
        """"Train" the model by computing the estimator matrices for the mean (K+lambda*sigma)^-1 and 
        for the covariance (K+lambda_c*sigma)^-1. The n_trajectories axis of the arguments is 
        considered only if the `self.priorities` parameter is not None.

        Parameters
        ----------
        X : array-like of shape (n_input_features,n_samples)
            Array of input vectors.
        Y : array-like of shape (n_trajectories,n_output_features,n_samples)
            Array of output vectors
        var : array-like of shape (n_trajectories,n_output_features,n_output_features,n_samples)
            Array of covariance matrices
        """
        if self.priorities is None:
            # Single trajectory
            self.s = X.copy()
            self.xi = Y.copy()
            self.sigma = var.copy()
            self.O = self.xi.shape[0]
            self.N = self.xi.shape[1]
        else:
            # Trajectory superposition
            L = len(Y)
            self.s = X.copy()
            self.xi = np.zeros_like(Y[0])
            self.sigma = np.zeros_like(var[0])
            self.O = self.xi.shape[0]
            self.N = self.xi.shape[1]
            # Compute covariances
            for n in range(self.N):
                for l in range(L):
                    self.sigma[:,:,n] += np.linalg.inv(var[l][:,:,n]/self.priorities[l](self.s[:,n]))
                # Covariance = precision^-1
                self.sigma[:,:,n] = np.linalg.inv(self.sigma[:,:,n])
            # Compute means
            for n in range(self.N):
                for l in range(L):
                    self.xi[:,n] += np.linalg.inv(var[l][:,:,n]/self.priorities[l](self.s[:,n]))@Y[l][:,n]
                self.xi[:,n] = self.sigma[:,:,n]@self.xi[:,n]
        k_mean = np.zeros((self.N*self.O,self.N*self.O))
        k_covariance = np.zeros((self.N*self.O,self.N*self.O))
        # Construct the estimators
        for i in range(self.N):
            for j in range(i,self.N):
                kernel = self.__kernel_matrix(self.s[:,i],self.s[:,j])
                k_mean[i*self.O:(i+1)*self.O,j*self.O:(j+1)*self.O] = kernel
                k_covariance[i*self.O:(i+1)*self.O,j*self.O:(j+1)*self.O] = kernel
                if i == j:
                    k_mean[j*self.O:(j+1)*self.O,i*self.O:(i+1)*self.O] += self.l*self.sigma[:,:,i]
                    k_covariance[j*self.O:(j+1)*self.O,i*self.O:(i+1)*self.O] += self.lc*self.sigma[:,:,i]
                else:
                    k_mean[j*self.O:(j+1)*self.O,i*self.O:(i+1)*self.O] = kernel
                    k_covariance[j*self.O:(j+1)*self.O,i*self.O:(i+1)*self.O] = kernel
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