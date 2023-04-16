#!/usr/bin/env python3

import logging
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
    datefmt='%Y-%m-%d,%H:%M:%S'
)
import numpy as np

from numpy.typing import ArrayLike
from typing import Type

class Uniform:
    """Clusterization by even distribution of data points.

    Parameters
    ----------
    n_components : int
        The number of clusters.
    n_demos : int
        The length of each demonstration.
    """
    def __init__(self: 'Uniform', n_components: int, n_demos: int) -> None:
        if n_components <= 0:
            raise ValueError('n_components must be strictly positive.')
        if n_demos <= 0:
            raise ValueError('n_demos must be strictly positive.')
        self.n_components = n_components
        self.n_demos = n_demos

    def fit_predict(self: 'Uniform', X: ArrayLike) -> ArrayLike:
        """Compute the cluster labels by evenly dividing the data.

        Parameters
        ----------
        X : array-like of shape (n_features,n_samples)
            The input data.

        Returns
        -------
        array-like of shape (n_samples,)
            The labels of the data.
        """
        n_samples = int(X.shape[1]/self.n_demos)
        return np.tile((np.arange(n_samples)*self.n_components/n_samples).astype(int),self.n_demos)
    
class KMeans:
    """Clusterization with the KMeans algorithm.

    Parameters
    ----------
    n_components : int
        The number of clusters.
    n_iter : int, default=100
        The maximum number of iterations of the algorithm.
    tol : float, default=1e-4
        The tolerance for convergence. The algorithm converges when the distance between two 
        consecutive iterations' centroids is less than tol.
    """
    def __init__(self: Type['KMeans'], n_components: int, n_iter: int=100, tol: float=1e-4) -> None:
        if n_components <= 0:
            raise ValueError('n_components must be strictly positive.')
        if n_iter <= 0:
            raise ValueError('n_iter must be strictly positive.')
        if tol <= 0:
            raise ValueError('tol must be strictly positive.')
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.__logger = logging.getLogger(__name__)

    def fit_predict(self: Type['KMeans'], X: ArrayLike) -> ArrayLike:
        """Compute the cluster labels using the KMeans algorithm.

        Parameters
        ----------
        X : array-like of shape (n_features,n_samples)
            The input data.

        Returns
        -------
        array-like of shape (n_samples,)
            The labels of the data.
        """
        # Randomly pick the initial centroids
        centroids = X[:, np.random.choice(X.shape[1], self.n_components, replace=False)].T
        for iteration in range(self.n_iter):
            # Calculate distances from centroids to samples
            distances = np.array([np.linalg.norm(X.T-centroid,axis=1) for centroid in centroids])
            # Assign each sample to the closest centroid
            labels = np.argmin(distances, axis=0)
            # Update centroids by taking the mean of samples in each cluster
            new_centroids = np.array([X[:, labels == i].mean(axis=1) for i in range(self.n_components)])                
            # Check for convergence
            if np.allclose(centroids,new_centroids,rtol=self.tol):
                self.__logger.info(f"KMeans converged after {iteration+1} iterations")
                break
            centroids = new_centroids
        # Final labels after convergence
        return np.argmin(distances, axis=0)