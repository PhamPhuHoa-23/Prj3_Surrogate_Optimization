import sys
sys.path.insert(0, '..')

import numpy as np
from src.core import BayesianOptimizer, SafeOptimizer, GaussianProcess


def simple_quadratic(x):
    """Simple test function: (x - 0.7)^2, minimum at x=0.7"""
    x = np.atleast_1d(x).flatten()[0]
    return (x - 0.7)**2


def test_bayesian_optimizer_initialization():
    """Test BayesianOptimizer initialization"""
    opt = BayesianOptimizer(
        objective_func=simple_quadratic,
        bounds=[(0, 1)],
        acquisition='ei'
    )
    
    assert opt.objective_func is not None
    assert opt.bounds.shape == (1, 2)
    assert opt.acquisition == 'ei'
    assert len(opt.X_samples) == 0


def test_bayesian_optimizer_run():
    """Test running Bayesian Optimization"""
    np.random.seed(42)
    
    opt = BayesianOptimizer(
        objective_func=simple_quadratic,
        bounds=[(0, 1)],
        acquisition='ei',
        gp_params={'length_scale': 0.2, 'noise': 1e-6}
    )
    
    X_best, y_best, history = opt.optimize(
        n_init=3,
        n_iter=10,
        random_state=42,
        verbose=False
    )
    
    assert X_best.shape == (1,)
    assert isinstance(y_best, (int, float, np.number))
    assert len(opt.X_samples) == 13  # 3 init + 10 iter
    assert y_best < 0.1  # Should find near-optimal solution
    assert 0.5 < X_best[0] < 0.9  # Should be near 0.7


def test_multiple_acquisitions():
    """Test different acquisition functions"""
    np.random.seed(42)
    acquisitions = ['ei', 'pi', 'lcb', 'pbe', 'ebe']
    
    for acq in acquisitions:
        opt = BayesianOptimizer(
            objective_func=simple_quadratic,
            bounds=[(0, 1)],
            acquisition=acq
        )
        
        X_best, y_best, history = opt.optimize(
            n_init=3,
            n_iter=5,
            random_state=42,
            verbose=False
        )
        
        assert X_best is not None
        assert y_best is not None
    

def safe_function(x):
    """Test function for SafeOpt: sin(x), safe region where y >= -0.5"""
    x = np.atleast_1d(x).flatten()
    if x.ndim > 1:
        x = x[0]
    return np.sin(x * np.pi)


def test_safe_optimizer():
    """Test SafeOptimizer"""
    np.random.seed(42)
    
    # Safe initialization points
    X_init = np.array([[0.25], [0.75]])
    y_init = np.array([float(safe_function(x.flatten())) for x in X_init])
    
    gp = GaussianProcess(length_scale=0.3, noise=1e-6)
    safe_opt = SafeOptimizer(
        gp=gp,
        bounds=[(0, 1)],
        y_threshold=-0.5,
        beta=3.0
    )
    
    X_best, y_best, history = safe_opt.optimize(
        objective_func=lambda x: float(safe_function(x)),
        X_init=X_init,
        y_init=y_init,
        n_iter=10,
        verbose=False
    )
    
    assert X_best is not None
    assert isinstance(y_best, (int, float, np.number))
    assert 'violations' in history
    assert history['violations'] >= 0


def test_normalization():
    """Test bounds normalization/denormalization"""
    opt = BayesianOptimizer(
        objective_func=simple_quadratic,
        bounds=[(-5, 10)],
        acquisition='ei'
    )
    
    X = np.array([[0.0], [0.5], [1.0]])
    X_norm = opt._normalize(X)
    X_denorm = opt._denormalize(X_norm)
    
    assert np.allclose(X_denorm, X)
    assert np.all(X_norm >= 0) and np.all(X_norm <= 1)


def run_all_tests():
    """Run all optimizer tests"""
    print("\nTesting Optimizers...")
    test_bayesian_optimizer_initialization()
    test_bayesian_optimizer_run()
    test_multiple_acquisitions()
    test_safe_optimizer()
    test_normalization()


if __name__ == "__main__":
    run_all_tests()

