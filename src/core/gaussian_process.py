import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky, cho_solve


class GaussianProcess:
    """
    Gaussian Process with Squared Exponential (RBF) Kernel
    
    Parameters:
        length_scale: Kernel length scale parameter
        noise: Measurement noise variance
    """
    
    def __init__(self, length_scale=1.0, noise=1e-6):
        self.length_scale = length_scale
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.L = None
        
    def squared_exponential_kernel(self, X1, X2):
        """Compute RBF kernel between two sets of points"""
        dists = cdist(X1, X2, metric='euclidean')
        return np.exp(-0.5 * (dists / self.length_scale) ** 2)
    
    def fit(self, X, y):
        """
        Fit GP model to training data
        
        Args:
            X: Training inputs, shape (n_samples, n_features)
            y: Training outputs, shape (n_samples,)
        """
        self.X_train = np.atleast_2d(X)
        self.y_train = np.atleast_1d(y).reshape(-1, 1)
        
        K = self.squared_exponential_kernel(self.X_train, self.X_train)
        K += self.noise * np.eye(len(self.X_train))
        
        self.L = cholesky(K, lower=True)
        
    def predict(self, X, return_std=True):
        """
        Predict mean and standard deviation at test points
        
        Args:
            X: Test inputs, shape (n_test, n_features)
            return_std: Whether to return standard deviation
            
        Returns:
            mean: Predicted means, shape (n_test,)
            std: Predicted standard deviations (if return_std=True), shape (n_test,)
        """
        X = np.atleast_2d(X)
        
        K_s = self.squared_exponential_kernel(self.X_train, X)
        
        alpha = cho_solve((self.L, True), self.y_train)
        mean = K_s.T @ alpha
        mean = mean.flatten()
        
        if not return_std:
            return mean
        
        K_ss = self.squared_exponential_kernel(X, X)
        v = cho_solve((self.L, True), K_s)
        var = np.diag(K_ss) - np.sum(K_s * v, axis=0)
        var = np.maximum(var, 0)
        std = np.sqrt(var)
        
        return mean, std

