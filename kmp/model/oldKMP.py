#!/usr/bin/env python3

import logging
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
    datefmt='%Y-%m-%d,%H:%M:%S'
)
import math
import numpy as np
from numpy.typing import ArrayLike
from typing import Type, Tuple

class KMP:
    """Trajectory imitation and adaptation using Kernelized Movement Primitives.

    Parameters
    ----------
    l : int, default=0.5
        Lambda regularization factor for the mean minimization problem.
    lc : int, default=10
        Lambda_c regularization factor for the covariance minimization problem.
    tol : float, default=0.0005
        Tolerance for the discrimination of conflicting points.
    kernel_gamma : int, default=6
        Coefficient for the rbf kernel. 
    priorities : array-like of shape (n_trajectories,), default=None
        Functions that map the input space into a priority value for trajectory superposition. The 
        sum of all priority functions evaluated in the same (any) input must be one.
    """
    def __init__(self: Type['KMP'], 
                 l: float=0.5, 
                 lc: int=10, 
                 tol: float=0.0005, 
                 kernel_gamma: int=6, 
                 priorities: ArrayLike=None) -> None:
        if l <= 0:
            raise ValueError('l must be strictly positive.')
        if lc <= 0:
            raise ValueError('lc must be strictly positive.')
        if tol <= 0:
            raise ValueError('tol must be strictly positive.')
        if kernel_gamma <= 0:
            raise ValueError('kernel_gamma must be strictly positive.')
        self.trained = False
        self.l = l
        self.lc = lc
        self.tol = tol
        self.kernel_gamma = kernel_gamma
        self.priorities = priorities
        self.__logger = logging.getLogger(__name__)

    def set_waypoint(self: Type['KMP'], 
                     s: ArrayLike, 
                     xi: ArrayLike, 
                     sigma: ArrayLike) -> None:
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
                self.s = np.append(self.s, np.array(s[j]).reshape(1,-1))
                self.xi = np.append(self.xi, xi[j])
                self.sigma = np.append(self.sigma, sigma[j])
        # Refit the model with the new data
        self.fit(self.s, self.xi, self.sigma)

    def __kernel(self: Type['KMP'], 
                 t1: float, 
                 t2: float, 
                 gamma: float) -> float:
        """Computes the Gaussian kernel function for the given input pair.

        Parameters
        ----------
        t1 : float
            The first input.
        t2 : float
            The second input.
        gamma : float
            l term in the exponential. Must be strictly positive.

        Returns
        -------
        kernel : float
                The result of the evaluation.
        """
        if gamma <= 0:
            raise ValueError('gamma must be strictly positive.')
        return np.exp(-gamma*(t1-t2)**2)[0]

    def __kernel_matrix(self: Type['KMP'], 
                        t1: float, 
                        t2: float) -> ArrayLike:
        """Computes the kernel matrix for the given input pair.

        Parameters
        ----------
        t1 : float
            The first input.
        t2 : float
            The second input.

        Returns
        -------
        kernel : array-like of shape (n_features,n_features)
            The kernel matrix evaluated in the provided input pair.
        """
        dt = 0.001
        t1dt = t1 + dt
        t2dt = t2 + dt
        # Half of the output features are position, the other half velocity
        O = int(self.O/2)
        # Compute the kernel blocks
        ktt = self.__kernel(t1, t2, self.kernel_gamma)
        if self.O > 3:
            ktd_tmp = self.__kernel(t1,t2dt,self.kernel_gamma)
            ktd = (ktd_tmp - ktt)/dt
            kdt_tmp = self.__kernel(t1dt,t2,self.kernel_gamma)
            kdt = (kdt_tmp - ktt)/dt
            kdd_tmp = self.__kernel(t1dt,t2dt,self.kernel_gamma)
            kdd = (kdd_tmp - ktd_tmp - kdt_tmp + ktt)/dt**2
            # Construct the kernel
            kernel = np.block([[ktt*np.eye(O), ktd*np.eye(O)],[kdt*np.eye(O), kdd*np.eye(O)]])
        else: # Position only output case
            kernel = ktt*np.eye(self.O)
        return kernel
                
    def fit(self: Type['KMP'], 
            X: ArrayLike, 
            Y: ArrayLike, 
            var: ArrayLike) -> None:
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
            for j in range(self.N):
                kernel = self.__kernel_matrix(self.s[:,i],self.s[:,j])
                k_mean[i*self.O:(i+1)*self.O,j*self.O:(j+1)*self.O] = kernel
                k_covariance[i*self.O:(i+1)*self.O,j*self.O:(j+1)*self.O] = kernel
                if i == j:
                    # Add the regularization terms on the diagonal
                    k_mean[j*self.O:(j+1)*self.O,i*self.O:(i+1)*self.O] = kernel + self.l*self.sigma[:,:,i]
                    k_covariance[j*self.O:(j+1)*self.O,i*self.O:(i+1)*self.O] = kernel + self.lc*self.sigma[:,:,i]
        self.__mean_estimator = np.linalg.inv(k_mean)
        self.__covariance_estimator = np.linalg.inv(k_covariance)

    def predict(self: Type['KMP'], s: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
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
                for h in range(self.O):
                    Y[i*self.O+h] = self.xi[h,i]
            xi[:,j] = k@self.__mean_estimator@Y
            sigma[:,:,j] = (self.N/self.lc)*(self.__kernel_matrix(s[:,j],s[:,j]) - k@self.__covariance_estimator@k.T)
        self.__logger.info('KMP Done.')
        return xi, sigma