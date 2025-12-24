import sys
sys.path.insert(0, '..')

import numpy as np
from src.benchmarks import get_benchmark


def test_forrester():
    """Test Forrester function"""
    f = get_benchmark('forrester')
    
    assert f.name == 'Forrester'
    assert f.dim == 1
    assert f.bounds.shape == (1, 2)
    
    y = f(np.array([0.5]))
    assert isinstance(y, (int, float, np.number))


def test_gramacy_lee():
    """Test Gramacy & Lee function"""
    f = get_benchmark('gramacy_lee')
    
    assert f.name == 'Gramacy & Lee'
    assert f.dim == 1
    
    y = f(np.array([1.5]))
    assert isinstance(y, (int, float, np.number))


def test_ackley():
    """Test Ackley function"""
    f = get_benchmark('ackley', dim=2)
    
    assert f.name == 'Ackley'
    assert f.dim == 2
    assert f.optimum == 0.0
    
    # Test at optimum
    y = f(np.array([[0, 0]]))
    assert np.abs(y) < 1e-10  # Should be near zero at optimum
    
    # Test away from optimum
    y = f(np.array([[1, 1]]))
    assert y > 0


def test_rosenbrock():
    """Test Rosenbrock function"""
    f = get_benchmark('rosenbrock', dim=2)
    
    assert f.name == 'Rosenbrock'
    assert f.dim == 2
    assert f.optimum == 0.0
    
    # Test at optimum (1, 1)
    y = f(np.array([[1, 1]]))
    assert np.abs(y) < 1e-10
    
    # Test away from optimum
    y = f(np.array([[0, 0]]))
    assert y > 0


def test_rastrigin():
    """Test Rastrigin function"""
    f = get_benchmark('rastrigin', dim=2)
    
    assert f.name == 'Rastrigin'
    assert f.dim == 2
    assert f.optimum == 0.0
    
    # Test at optimum
    y = f(np.array([[0, 0]]))
    assert np.abs(y) < 1e-10
    
    # Test away from optimum
    y = f(np.array([[1, 1]]))
    assert y > 0


def test_flower():
    """Test Flower function"""
    f = get_benchmark('flower')
    
    assert f.name == 'Flower'
    assert f.dim == 2
    assert f.optimum == 1.0
    
    # Test at origin (optimum)
    y = f(np.array([[0, 0]]))
    assert np.abs(y) < 0.01  # Near zero at origin
    
    # Test various points
    y = f(np.array([[0.5, 0.0]]))
    assert isinstance(y, (int, float, np.number))


def test_benchmark_vectorization():
    """Test that benchmarks handle both 1D and 2D inputs"""
    f = get_benchmark('ackley', dim=2)
    
    # Single point (1D array)
    y1 = f(np.array([1, 1]))
    
    # Single point (2D array)
    y2 = f(np.array([[1, 1]]))
    
    # Multiple points
    y3 = f(np.array([[1, 1], [2, 2]]))
    
    assert isinstance(y1, (int, float, np.number))
    assert isinstance(y2, (int, float, np.number))
    assert isinstance(y3, (np.ndarray, int, float))


def test_get_benchmark_invalid():
    """Test getting invalid benchmark"""
    try:
        get_benchmark('nonexistent_function')
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert 'Unknown benchmark' in str(e)


def run_all_tests():
    """Run all benchmark tests"""
    print("\nTesting Benchmark Functions...")
    test_forrester()
    test_gramacy_lee()
    test_ackley()
    test_rosenbrock()
    test_rastrigin()
    test_flower()
    test_benchmark_vectorization()
    test_get_benchmark_invalid()


if __name__ == "__main__":
    run_all_tests()

