import sys
sys.path.insert(0, '..')

import numpy as np
from src.core import GaussianProcess


def test_gp_initialization():
    """Test GP initialization"""
    gp = GaussianProcess(length_scale=1.0, noise=0.01)
    assert gp.length_scale == 1.0
    assert gp.noise == 0.01
    assert gp.X_train is None
    assert gp.y_train is None


def test_gp_kernel():
    """Test RBF kernel computation"""
    gp = GaussianProcess(length_scale=1.0)
    X1 = np.array([[0], [1], [2]])
    X2 = np.array([[0], [1]])
    
    K = gp.squared_exponential_kernel(X1, X2)
    
    assert K.shape == (3, 2)
    assert np.allclose(K[0, 0], 1.0)  # Same point
    assert K[0, 0] > K[0, 1]  # Closer points have higher correlation


def test_gp_fit_predict():
    """Test GP fit and predict"""
    np.random.seed(42)
    
    # Training data
    X_train = np.array([[0], [1], [2], [3]])
    y_train = np.sin(X_train).flatten()
    
    # Fit GP
    gp = GaussianProcess(length_scale=1.0, noise=0.01)
    gp.fit(X_train, y_train)
    
    assert gp.X_train is not None
    assert gp.y_train is not None
    assert gp.L is not None
    
    # Predict
    X_test = np.array([[1.5]])
    mean, std = gp.predict(X_test)
    
    assert mean.shape == (1,)
    assert std.shape == (1,)
    assert std[0] > 0  # Should have uncertainty
    
    # Predict at training point should be close to training value
    mean_train, std_train = gp.predict(X_train[:1])
    assert np.abs(mean_train[0] - y_train[0]) < 0.1
    assert std_train[0] < 0.1  # Low uncertainty at training point
    

def test_gp_prediction_uncertainty():
    """Test that uncertainty increases away from data"""
    np.random.seed(42)
    
    X_train = np.array([[0], [1]])
    y_train = np.array([0, 1])
    
    gp = GaussianProcess(length_scale=1.0, noise=0.01)
    gp.fit(X_train, y_train)
    
    # Uncertainty at interpolation vs extrapolation
    _, std_interp = gp.predict(np.array([[0.5]]))
    _, std_extrap = gp.predict(np.array([[5.0]]))
    
    assert std_extrap[0] > std_interp[0]


def run_all_tests():
    """Run all GP tests"""
    print("\nTesting Gaussian Process...")
    test_gp_initialization()
    test_gp_kernel()
    test_gp_fit_predict()
    test_gp_prediction_uncertainty()


if __name__ == "__main__":
    run_all_tests()

