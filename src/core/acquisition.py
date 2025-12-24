import numpy as np
from scipy.stats import norm


def probability_of_improvement(mean, std, y_best, xi=0.01):
    """
    Probability of Improvement (PI)
    
    Args:
        mean: Predicted mean
        std: Predicted standard deviation
        y_best: Best observed value (for minimization)
        xi: Exploration parameter
        
    Returns:
        PI values at each point
    """
    with np.errstate(divide='warn'):
        z = (y_best - xi - mean) / std
        pi = norm.cdf(z)
        
    pi[std == 0.0] = 0.0
    return pi


def expected_improvement(mean, std, y_best, xi=0.01):
    """
    Expected Improvement (EI)
    
    Args:
        mean: Predicted mean
        std: Predicted standard deviation
        y_best: Best observed value (for minimization)
        xi: Exploration parameter
        
    Returns:
        EI values at each point
    """
    with np.errstate(divide='warn'):
        z = (y_best - xi - mean) / std
        ei = (y_best - xi - mean) * norm.cdf(z) + std * norm.pdf(z)
        
    ei[std == 0.0] = 0.0
    return ei


def lower_confidence_bound(mean, std, alpha=2.0):
    """
    Lower Confidence Bound (LCB) for minimization
    
    Args:
        mean: Predicted mean
        std: Predicted standard deviation
        alpha: Exploration-exploitation trade-off parameter
        
    Returns:
        LCB values at each point
    """
    return mean - alpha * std


def upper_confidence_bound(mean, std, alpha=2.0):
    """
    Upper Confidence Bound (UCB) for maximization or SafeOpt
    
    Args:
        mean: Predicted mean
        std: Predicted standard deviation
        alpha: Confidence parameter
        
    Returns:
        UCB values at each point
    """
    return mean + alpha * std


def prediction_based(mean, std, y_best=None):
    """
    Prediction-Based Exploration (PBE)
    Pure exploitation: select point with best predicted mean
    
    Args:
        mean: Predicted mean
        std: Predicted standard deviation (unused)
        y_best: Best observed value (unused)
        
    Returns:
        Predicted mean values (for minimization, lower is better)
    """
    return mean


def error_based(mean, std, y_best=None):
    """
    Error-Based Exploration (EBE)
    Pure exploration: select point with highest uncertainty
    
    Args:
        mean: Predicted mean (unused)
        std: Predicted standard deviation
        y_best: Best observed value (unused)
        
    Returns:
        Uncertainty values (higher is better)
    """
    return std

