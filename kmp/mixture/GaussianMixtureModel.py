#!/usr/bin/env python3

import logging
import numpy as np
REALMIN = np.finfo(np.float64).tiny # To avoid division by 0
import warnings

from numpy.linalg import inv, det

class GaussianMixtureModel:
    """A Gaussian Mixture Model representation that allows for the extraction of the probabilistic 
    parameters of the underlying Gaussian mixture distribution to a set of demonstrations.

    based on: "On Learning, Representing and Generalizing a Task in a Humanoid Robot", 
    Calinon et al., 2007

    Parameters
    ----------
    n_components : int, default=8
        The number of mixture components.

    n_demos : int, default=5
        The number of demonstrations in the database.

    max_it : int, default=100
        Maximum number of iterations for the EM algorithm.
        
    tol : _type_, default=1e-4
        Convergence criterion (lower bound for the average log-likelihood) for the EM algorithm.

    reg_factor : _type_, default=1e-4
        Regularization factor added to the diagonal of the covariance matrices to ensure they are 
        positive definite.

    Attributes
    ----------
    priors_ : array-like of shape (n_components,)
        The priors of each component of the mixture.

    means_ : array-like of shape (n_features, n_components)
        The mean of each mixture component.

    covariances_ : array-like of shape (n_features, n_features, n_components)
        The covariance of each mixture component.

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.

    n_iter_ : int
        Number of steps of EM to reach convergence.

    lower_bound_ : float
        The lowest value of the average log-lokelihood seen during fit().

    n_input_features_ : int
        The input space dimension seen during fit().

    n_output_features_ : int
        The output space dimension seen during fit().

    n_features_ : int
        The sum of n_input_features_ and n_output_features_.

    n_components_ : int
        The number of Gaussian components used during fit().

    n_demos_ : int
        The number of demonstrations seen during fit().

    n_samples_ : int
        The number of samples of each demonstration seen during fit().
    """
    def __init__(self, n_components=8, 
                       n_demos=5,
                       max_it=100,
                       tol=1e-4,
                       reg_factor=1e-4) -> None:
        # Class attributes
        self.n_components_ = n_components
        self.n_demos_ = n_demos
        self.max_it = max_it
        self.tol = tol
        self.reg_factor = reg_factor
        self.__initialized = False
        self.__logger = logging.getLogger(__name__)
        self.__logger.info('Instantiated GaussianMixtureModel')

    def __initialize(self, X, Y):
        """Performs the initial guess of the priors, means and covariances.

        Parameters
        ----------
        X : array-like of shape (n_input_features, n_samples)
            The array of input vectors.
        Y : array-like of shape (n_output_features, n_samples)
            The array of output vectors.
        """
        # Set the relevant attributes
        self.n_input_features_ = X.shape[0]
        self.n_output_features_ = Y.shape[0]
        self.n_features_ = self.n_input_features_ + self.n_output_features_
        self.n_samples_ = int(Y.shape[1]/self.n_demos_)
        # Setup the arrays for the priors, means and covariance matrices
        self.priors_ = np.zeros((self.n_components_)) 
        self.means_ = np.zeros((self.n_features_,self.n_components_)) 
        self.covariances_ = np.zeros((self.n_features_,self.n_features_,self.n_components_))
        # Setup the dataset
        X = np.vstack([X,Y])
        # Evenly assign data points to each Gaussian component
        ids_full = np.tile((np.arange(self.n_samples_)*self.n_components_/self.n_samples_).astype(int),self.n_demos_)
        for c in range(self.n_components_):
            ids = [i for i,val in enumerate(ids_full) if val == c]
            # Compute priors, mean and the covariance matrix
            self.priors_[c] = len(ids)
            self.means_[:,c] = np.mean(X[:,ids].T,axis=0)
            self.covariances_[:,:,c] = np.cov(X[:,ids]) + np.eye(self.n_features_)*self.reg_factor
        # Normalize the prior probabilities
        self.priors_ = self.priors_/np.sum(self.priors_)
        self.__initialized = True
        self.__logger.info('Initialized GaussianMixtureModel')

    def __pdf(self, X, sigma, mu):
        """Computes the Gaussian probability density function defined by the given mean and covariance,
        evaluated in the given points.

        Parameters
        ----------
        X : int or array-like of shape (n_features,n_samples)
            The data points in which the pdf is evaluated.
        sigma : int or array-like of shape (n_features,n_features)
            The variance or covariance matrix that defines the distribution.
        mu : int or array-like of shape (n_features)
            The mean vector that defines the distribution.

        Returns
        -------
        pdf : int or array-like of shape (n_features,n_samples)
            The computed probability density function values.
        """
        if X.ndim <= 1:
            # Univariate case
            X_cntr = X - mu
            pdf = X_cntr**2/sigma
            try:
                pdf = np.exp(-0.5*pdf)/np.sqrt(2*np.pi*np.abs(sigma))
            except ZeroDivisionError:
                pdf = np.exp(-0.5*pdf)/np.sqrt(2*np.pi*np.abs(sigma)+REALMIN)
        else:
            # Multivariate case
            X_cntr = np.asarray(X).T - np.tile(np.asarray(mu).T,(self.n_demos_*self.n_samples_,1))
            pdf = np.sum(X_cntr@inv(sigma)*X_cntr, axis=1)
            try:
                pdf = np.exp(-0.5*pdf)/np.sqrt((2*np.pi)**(self.n_features_)*np.abs(det(sigma)))
            except ZeroDivisionError:
                pdf = np.exp(-0.5*pdf)/np.sqrt((2*np.pi)**(self.n_features_)*np.abs(det(sigma)) + REALMIN)
        return pdf

    def __likelihood(self, X, sigma, mu):
        """Computes the likelihood of the Gaussian distribution defined by the given mean and covariance,
        evaluated in the given points.

        Parameters
        ----------
        X : int or array-like of shape (n_features,n_samples)
            The data points in which the likelihood is evaluated.
        sigma : int or array-like of shape (n_features,n_features,n_components)
            The variance or covariance matrix that defines the distribution.
        mu : int or array-like of shape (n_features,n_components)
            The mean vector that defines the distribution.

        Returns
        -------
        likelihood : array-like of shape (n_samples,n_components)
            The likelihood of each sample with respect to each Gaussian component.
        """
        likelihood = [self.priors_[c]*self.__pdf(X, sigma[:,:,c], mu[:,c]) for c in range(self.n_components_)]
        return likelihood

    def fit(self, X, Y):
        """Fit the model using the Expectation-Maximization algorithm.

        Parameters
        ----------
        X : array-like of shape (n_input_features, n_samples)
            The array of input vectors.
        Y : array-like of shape (n_output_features, n_samples)
            The array of output vectors.
        """
        # Make the initial guess if it wasn't already made
        if not self.__initialized : self.__initialize(X, Y)
        # Construct the dataset
        X = np.vstack([X,Y])
        # Expectation-Maximization
        LL = np.zeros(self.max_it)
        for it in range(self.max_it):
            # Expectation step
            L = self.__likelihood(X, self.covariances_, self.means_)
            # Pseudo posterior: likelihood/evidence
            try:
                gamma = L/np.tile(np.sum(L,axis=0),(self.n_components_,1))
            except ZeroDivisionError:
                gamma = L/np.tile(np.sum(L,axis=0)+REALMIN,(self.n_components_,1))
            gamma_mean = gamma/np.tile(np.sum(gamma,axis=1),(self.n_samples_*self.n_demos_,1)).T
            # Maximization step
            for c in range(self.n_components_):
                # Update priors
                self.priors_[c] = np.sum(gamma[c])/(self.n_samples_*self.n_demos_)
                # Update mean
                self.means_[:,c] = X @ gamma_mean[c].T
                # Update covariance
                X_cntr = X.T - np.tile(self.means_[:,c],(self.n_samples_*self.n_demos_,1))
                sigma = X_cntr.T @ np.diag(gamma_mean[c]) @ X_cntr
                self.covariances_[:,:,c] = sigma + np.eye(self.n_features_)*self.reg_factor
            # Average log likelihood
            LL[it] = np.sum(np.log(np.sum(L,axis=1)))/(self.n_samples_*self.n_demos_)
            # Check convergence
            if it > 0 and (it >= self.max_it or np.abs(LL[it]-LL[it-1]) < self.tol):
                self.__logger.info(f'EM converged in {it} steps')
                self.lower_bound_ = LL[it]
                self.n_iter_ = it
                self.converged_ = True
                return        
        self.converged_ = False
        warnings.warn("ConvergenceWarning: EM algorithm did not converge")

    def predict(self, X):
        """Predict the expected output using Gaussian Mixture Regression.

        Parameters
        ----------
        X : array-like of shape (n_input_features, n_samples)
            The array of input vectors.

        Returns
        -------
        means : array-like of shape (n_input_features, n_samples)
            The array of mean vectors.

        covariances : array-like of shape (n_input_features, n_input_features, n_samples)
            The array of covariance matrices.
        """
        # Get the indexes of the input- and output-related quantities
        I = X.shape[0]
        N = X.shape[1]
        ts = list(range(I)) # Input indexes
        s = list(range(I,self.covariances_.shape[0])) # Output indexes
        # Setup the output arrays
        means = np.zeros((len(s),N))
        covariances = np.zeros((len(s),len(s),N))
        # Perform regression
        mu_tmp = np.zeros((len(s),self.n_components_))
        beta = np.zeros((self.n_components_,N))
        for i in range(N):
            for t in ts: # If the input is multidimensional, repeat the procedure for each dimension
                # Compute posteriors (Calinon eq. 11)
                for c in range(self.n_components_):
                    beta[c,i] = self.priors_[c]*self.__pdf(X[t,i],self.covariances_[t,t,c],self.means_[t,c])
                # Normalize
                try:
                    beta[:,i] = beta[:,i]/np.sum(beta[:,i])
                except ZeroDivisionError:
                    beta[:,i] = beta[:,i]/np.sum(beta[:,i]+REALMIN)
                # Compute conditional means (Calinon eq. 10)
                for c in range(self.n_components_):
                    mu_tmp[:,c] = self.means_[s,c] + self.covariances_[s,t,c]*(X[t,i]-self.means_[t,c])/self.covariances_[t,t,c]
                    means[:,i] = means[:,i] + beta[c,i]*mu_tmp[:,c]
                # Compute conditional covariances (Calinon eq. 10)
                for c in range(self.n_components_):
                    gmr_sigma = self.covariances_[np.ix_(s,s,[c])][:,:,0] - (np.reshape(self.covariances_[s,t,c], (len(s),1))/self.covariances_[t,t,c])*self.covariances_[t,s,c]
                    covariances[:,:,i] = covariances[:,:,i] + (beta[c,i]**2)*gmr_sigma
                reg_factor = np.eye(len(s))*self.reg_factor
                covariances[:,:,i] = covariances[:,:,i] + reg_factor
        self.__logger.info('GMR done')
        return means, covariances