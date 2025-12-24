import sys
sys.path.insert(0, '..')

import numpy as np
from src.core.acquisition import (
    probability_of_improvement,
    expected_improvement,
    lower_confidence_bound,
    upper_confidence_bound,
    prediction_based,
    error_based
)


def test_pi():
    """Test Probability of Improvement"""
    mean = np.array([1.0, 2.0, 3.0])
    std = np.array([0.1, 0.2, 0.3])
    y_best = 2.5
    
    pi = probability_of_improvement(mean, std, y_best)
    
    assert pi.shape == mean.shape
    assert np.all(pi >= 0) and np.all(pi <= 1)
    assert pi[0] > pi[2]  # Better mean should have higher PI


def test_ei():
    """Test Expected Improvement"""
    mean = np.array([1.0, 2.0, 3.0])
    std = np.array([0.1, 0.2, 0.3])
    y_best = 2.5
    
    ei = expected_improvement(mean, std, y_best)
    
    assert ei.shape == mean.shape
    assert np.all(ei >= 0)
    assert ei[0] > ei[2]  # Better mean should have higher EI


def test_lcb():
    """Test Lower Confidence Bound"""
    mean = np.array([1.0, 2.0, 3.0])
    std = np.array([0.1, 0.2, 0.3])
    
    lcb = lower_confidence_bound(mean, std, alpha=2.0)
    
    assert lcb.shape == mean.shape
    assert np.all(lcb < mean)  # LCB should be below mean
    assert lcb[0] < lcb[2]  # Lower mean → lower LCB


def test_ucb():
    """Test Upper Confidence Bound"""
    mean = np.array([1.0, 2.0, 3.0])
    std = np.array([0.1, 0.2, 0.3])
    
    ucb = upper_confidence_bound(mean, std, alpha=2.0)
    
    assert ucb.shape == mean.shape
    assert np.all(ucb > mean)  # UCB should be above mean
    assert ucb[0] < ucb[2]  # Higher mean → higher UCB


def test_pbe():
    """Test Prediction-Based Exploration"""
    mean = np.array([1.0, 2.0, 3.0])
    std = np.array([0.1, 0.2, 0.3])
    
    pbe = prediction_based(mean, std)
    
    assert pbe.shape == mean.shape
    assert np.array_equal(pbe, mean)  # PBE returns mean


def test_ebe():
    """Test Error-Based Exploration"""
    mean = np.array([1.0, 2.0, 3.0])
    std = np.array([0.1, 0.2, 0.3])
    
    ebe = error_based(mean, std)
    
    assert ebe.shape == std.shape
    assert np.array_equal(ebe, std)  # EBE returns std


def test_acquisition_zero_std():
    """Test acquisition functions with zero std (avoid division by zero)"""
    mean = np.array([1.0, 2.0])
    std = np.array([0.0, 0.1])
    y_best = 1.5
    
    pi = probability_of_improvement(mean, std, y_best)
    ei = expected_improvement(mean, std, y_best)
    
    assert not np.any(np.isnan(pi))
    assert not np.any(np.isnan(ei))
    assert pi[0] == 0.0  # Zero std → zero PI
    assert ei[0] == 0.0  # Zero std → zero EI


def run_all_tests():
    """Run all acquisition function tests"""
    print("\nTesting Acquisition Functions...")
    test_pi()
    test_ei()
    test_lcb()
    test_ucb()
    test_pbe()
    test_ebe()
    test_acquisition_zero_std()


if __name__ == "__main__":
    run_all_tests()

