import logging
import numpy as np

from numpy.linalg import inv, det
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from typing import Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
    datefmt='%Y-%m-%d,%H:%M:%S'
)

REALMIN = np.finfo(np.float64).tiny  # To avoid division by 0


class GaussianMixtureModel():
    """Representation of a Gaussian Mixture Model probability distribution. The class allows for the estimation of the 
    parameters of a GMM, specifically in the case of learning from demonstration. The class is based on sklearn's GMM 
    implementation, expanding it to implement Gaussian Mixture Regression.

    Parameters
    ----------
    n_components : int, default = 10
        Number of Gaussian components of the model.
    n_demos : int, default = 5
        Number of demonstrations in the training dataset.
    diag_reg_factor : float, default = 1e-6
        Non negative regularization factor added to the diagonal of the covariances to ensure they are positive.
    """

    def __init__(self,
                 n_components: int = 10,
                 n_demos: int = 5,
                 diag_reg_factor: float = 1e-4) -> None:
        self.n_components = n_components
        self.n_demos = n_demos
        self.diag_reg_factor = diag_reg_factor
        self.model = GaussianMixture(
            n_components=n_components, reg_covar=diag_reg_factor, random_state=420)
        self.logger = logging.getLogger(__name__)

    def fit(self, data: np.ndarray) -> None:
        """Wrapper around sklearn's GMM fit implementation.

        Parameters
        ----------
        data : np.ndarray
            The dataset to fit the model on.
        """
        self.n_features = data.shape[1]
        self.model.fit(data)
        self.logger.info("GMM fit done.")
        self.priors = self.model.weights_
        self.means = self.model.means_.T # transpose to have shape (n_features, n_samples)
        self.covariances = np.transpose(self.model.covariances_, (1, 2, 0)) # transpose to have shape (n_features, n_features, n_samples)

    def predict(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Use Gaussian Mixture Regression to predict mean and covariance of the given inputs

        Parameters
        ----------
        data : ArrayLike of shape (n_input_features, n_samples)

        Returns
        -------
        means : ArrayLike of shape (n_output_features, n_samples)
            The mean vectors associated to each input point.
        covariances : ArrayLike of shape (n_output_features, n_output_features, n_samples)
            The covariance matrices associated to each input point.
        """
        # Dimensionality of the inputs, number of points
        I, N = data.shape
        # Dimensionality of the outputs
        O = self.n_features - I
        diag_reg_factor = np.eye(O)*self.diag_reg_factor
        # Initialize needed arrays
        mu_tmp = np.zeros((O, self.n_components))
        means = np.zeros((O, N))
        covariances = np.zeros((O, O, N))
        H = np.zeros((self.n_components, N))
        for t in range(N):
            # Activation weight
            for i in range(self.n_components):
                mu = self.means[:I, i]
                sigma = self.covariances[:I, :I, i]
                dist = multivariate_normal(mu, sigma)
                H[i, t] = self.priors[i] * dist.pdf(data[:,t])
            H[:, t] /= np.sum(H[:, t] + REALMIN)
            # Conditional means
            for i in range(self.n_components):
                sigma_tmp = self.covariances[I:, :I, i]@inv(self.covariances[:I, :I, i])
                mu_tmp[:, i] = self.means[I:, i] + \
                    sigma_tmp@(data[:, t]-self.means[:I, i])
                means[:, t] += H[i, t]*mu_tmp[:, i]
            # Conditional covariances
            for i in range(self.n_components):
                sigma_tmp = self.covariances[I:, I:, i] - \
                    self.covariances[I:, :I, i]@inv(
                        self.covariances[:I, :I, i])@self.covariances[:I, I:, i]
                covariances[:, :, t] += H[i, t] * \
                    (sigma_tmp + np.outer(mu_tmp[:, i], mu_tmp[:, i]))
            covariances[:, :, t] += diag_reg_factor - \
                np.outer(means[:, t], means[:, t])
        self.logger.info("GMR done.")
        return means, covariances

    def bic(self, data: np.ndarray) -> float:
        """Wrapper around sklearn's GMM BIC function.

        Parameters
        ----------
        data : np.ndarray
            The data to evaluate the BIC in.

        Returns
        -------
        float
            The computed BIC. The lower the better.
        """
        return self.model.bic(data)
