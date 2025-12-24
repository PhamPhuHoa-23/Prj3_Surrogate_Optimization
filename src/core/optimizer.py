import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

from .gaussian_process import GaussianProcess
from .acquisition import (
    probability_of_improvement,
    expected_improvement,
    lower_confidence_bound,
    prediction_based,
    error_based
)


class BayesianOptimizer:
    """
    Bayesian Optimization with Gaussian Process Surrogate
    
    Args:
        objective_func: Function to minimize
        bounds: Search space bounds [(min1, max1), (min2, max2), ...]
        acquisition: Acquisition function ('ei', 'pi', 'lcb', 'pbe', 'ebe')
        gp_params: GP hyperparameters dict
    """
    
    def __init__(self, objective_func, bounds, acquisition='ei', gp_params=None):
        self.objective_func = objective_func
        bounds = np.atleast_2d(bounds)
        if bounds.shape[1] != 2:
            bounds = bounds.T
        self.bounds = bounds
        self.acquisition = acquisition
        
        if gp_params is None:
            gp_params = {'length_scale': 1.0, 'noise': 1e-6}
        self.gp = GaussianProcess(**gp_params)
        
        self.X_samples = []
        self.y_samples = []
        
    def _normalize(self, X):
        """Normalize to [0, 1]"""
        return (X - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
    
    def _denormalize(self, X_norm):
        """Denormalize to original space"""
        return X_norm * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
    
    def _acquisition_function(self, X, y_best):
        """Compute acquisition function"""
        mean, std = self.gp.predict(X)
        
        if self.acquisition == 'ei':
            return expected_improvement(mean, std, y_best)
        elif self.acquisition == 'pi':
            return probability_of_improvement(mean, std, y_best)
        elif self.acquisition == 'lcb':
            return -lower_confidence_bound(mean, std, alpha=2.0)
        elif self.acquisition == 'pbe':
            return -prediction_based(mean, std)  # Negate for maximization
        elif self.acquisition == 'ebe':
            return error_based(mean, std)
        else:
            raise ValueError(f"Unknown acquisition: {self.acquisition}")
    
    def _optimize_acquisition(self, n_restarts=25):
        """Find next point by optimizing acquisition function"""
        dim = len(self.bounds)
        best_x = None
        best_acq = -np.inf
        
        y_best = np.min(self.y_samples)
        
        # Random search + local optimization
        X_random = np.random.uniform(0, 1, size=(1000 * dim, dim))
        acq_values = self._acquisition_function(X_random, y_best)
        top_indices = np.argsort(acq_values)[-n_restarts:]
        
        for idx in top_indices:
            x0 = X_random[idx]
            
            def neg_acquisition(x):
                return -self._acquisition_function(x.reshape(1, -1), y_best)[0]
            
            res = minimize(neg_acquisition, x0, method='L-BFGS-B', bounds=[(0, 1)] * dim)
            
            if -res.fun > best_acq:
                best_acq = -res.fun
                best_x = res.x
        
        return best_x.reshape(1, -1)
    
    def optimize(self, n_init=5, n_iter=20, random_state=None, verbose=True):
        """
        Run Bayesian Optimization
        
        Args:
            n_init: Number of random initialization points
            n_iter: Number of optimization iterations
            random_state: Random seed for reproducibility
            verbose: Print progress
            
        Returns:
            X_best: Best point found
            y_best: Best objective value
            history: Dict with optimization history
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Random initialization
        X_init_norm = np.random.uniform(0, 1, size=(n_init, len(self.bounds)))
        X_init = self._denormalize(X_init_norm)
        y_init = np.array([self.objective_func(x) for x in X_init])
        
        self.X_samples = X_init_norm.tolist()
        self.y_samples = y_init.tolist()
        self.gp.fit(np.array(self.X_samples), np.array(self.y_samples))
        
        history = {'X': [X_init], 'y': [y_init], 'X_norm': [X_init_norm]}
        
        if verbose:
            print(f"Bayesian Optimization ({self.acquisition.upper()})")
            print(f"Initialization: {n_init} random points")
        
        # Optimization loop
        for i in range(n_iter):
            X_next_norm = self._optimize_acquisition()
            X_next = self._denormalize(X_next_norm)
            y_next = self.objective_func(X_next.flatten())
            
            self.X_samples.append(X_next_norm.flatten())
            self.y_samples.append(y_next)
            self.gp.fit(np.array(self.X_samples), np.array(self.y_samples))
            
            history['X'].append(X_next)
            history['y'].append(np.array([y_next]))
            history['X_norm'].append(X_next_norm)
            
            if verbose:
                print(f"Iter {i+1}/{n_iter}: y = {float(y_next):7.4f}, best = {np.min(self.y_samples):7.4f}")
        
        # Return best result
        best_idx = np.argmin(self.y_samples)
        X_best = self._denormalize(np.array(self.X_samples[best_idx]).reshape(1, -1)).flatten()
        y_best = self.y_samples[best_idx]
        
        return X_best, y_best, history


class SafeOptimizer:
    """
    SafeOpt: Safe Bayesian Optimization with Safety Constraints
    
    Args:
        gp: Gaussian Process model
        bounds: Search space bounds
        y_threshold: Safety threshold (f(x) >= y_threshold)
        beta: Confidence parameter for UCB/LCB
        p_safe: Required safety probability
    """
    
    def __init__(self, gp, bounds, y_threshold, beta=3.0, p_safe=0.95):
        self.gp = gp
        self.bounds = np.array(bounds)
        self.y_threshold = y_threshold
        self.beta = beta
        self.p_safe = p_safe
        
        self.X_samples = []
        self.y_samples = []
        
    def _confidence_bounds(self, X):
        """Compute UCB and LCB"""
        mean, std = self.gp.predict(X)
        u = mean + np.sqrt(self.beta) * std
        l = mean - np.sqrt(self.beta) * std
        return u, l
    
    def _compute_safe_set(self, X):
        """Identify safe points based on probability threshold"""
        mean, std = self.gp.predict(X)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            z = (self.y_threshold - mean) / std
            p_safety = norm.cdf(z)
        
        return p_safety >= self.p_safe
    
    def select_next_point(self, n_candidates=1000):
        """
        Select next safe point to sample
        
        Args:
            n_candidates: Number of random candidates to consider
            
        Returns:
            X_next: Next point to sample (shape: (n_features,))
        """
        # Generate candidate points
        X_candidates = np.random.uniform(
            self.bounds[:, 0], 
            self.bounds[:, 1], 
            size=(n_candidates, len(self.bounds))
        )
        
        # Compute confidence bounds
        u, l = self._confidence_bounds(X_candidates)
        
        # Identify safe points (LCB above threshold)
        safe_mask = l >= self.y_threshold
        
        if not np.any(safe_mask):
            # No safe points: select safest option
            return X_candidates[np.argmax(l)]
        
        # Among safe points: select one with highest UCB (most optimistic)
        safe_u = u.copy()
        safe_u[~safe_mask] = -np.inf
        return X_candidates[np.argmax(safe_u)]
    
    def optimize(self, objective_func, X_init, y_init, n_iter=20, verbose=True):
        """
        Run SafeOpt algorithm
        
        Args:
            objective_func: Objective function to optimize
            X_init: Safe initialization points (shape: (n_init, n_features))
            y_init: Function values at X_init
            n_iter: Number of iterations
            verbose: Print progress
            
        Returns:
            X_best: Best safe point found
            y_best: Best value
            history: Optimization history
        """
        self.X_samples = X_init.tolist()
        self.y_samples = y_init.tolist()
        self.gp.fit(np.array(self.X_samples), np.array(self.y_samples))
        
        history = {'X': [X_init], 'y': [y_init], 'violations': 0}
        
        if verbose:
            print(f"SafeOpt (beta={self.beta}, threshold={self.y_threshold})")
            print(f"Initialization: {len(X_init)} safe points")
        
        for i in range(n_iter):
            X_next = self.select_next_point()
            y_next = objective_func(X_next)
            
            # Track violations
            if y_next < self.y_threshold:
                history['violations'] += 1
            
            self.X_samples.append(X_next)
            self.y_samples.append(y_next)
            self.gp.fit(np.array(self.X_samples), np.array(self.y_samples))
            
            history['X'].append(X_next.reshape(1, -1))
            history['y'].append(np.array([y_next]))
            
            if verbose:
                safe_str = "SAFE" if y_next >= self.y_threshold else "UNSAFE"
                print(f"Iter {i+1}/{n_iter}: y = {float(y_next):7.4f} [{safe_str}], best = {np.max(self.y_samples):7.4f}")
        
        # Return best safe result
        safe_indices = [i for i, y in enumerate(self.y_samples) if y >= self.y_threshold]
        if safe_indices:
            best_idx = safe_indices[np.argmax([self.y_samples[i] for i in safe_indices])]
        else:
            best_idx = np.argmax(self.y_samples)
        
        X_best = np.array(self.X_samples[best_idx])
        y_best = self.y_samples[best_idx]
        
        return X_best, y_best, history

