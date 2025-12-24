import numpy as np


class BenchmarkFunction:
    """Base class for benchmark optimization functions"""
    
    def __init__(self, name, bounds, optimum=None):
        self.name = name
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.optimum = optimum
        
    def __call__(self, x):
        raise NotImplementedError


class ForresterFunction(BenchmarkFunction):
    """Forrester function (1D smooth)"""
    
    def __init__(self):
        super().__init__('Forrester', [(0, 1)], optimum=0.757)
        
    def __call__(self, x):
        x = np.atleast_1d(x).flatten()[0]
        return (6*x - 2)**2 * np.sin(12*x - 4)


class GramacyLeeFunction(BenchmarkFunction):
    """Gramacy & Lee function (1D discontinuous)"""
    
    def __init__(self):
        super().__init__('Gramacy & Lee', [(0.5, 2.5)], optimum=-0.869)
        
    def __call__(self, x):
        x = np.atleast_1d(x).flatten()[0]
        return np.sin(10 * np.pi * x) / (2 * x) + (x - 1)**4


class AckleyFunction(BenchmarkFunction):
    """Ackley function (multimodal)"""
    
    def __init__(self, dim=2):
        bounds = [(-5, 5)] * dim
        super().__init__('Ackley', bounds, optimum=0.0)
        
    def __call__(self, x):
        x = np.atleast_2d(x)
        a, b, c = 20, 0.2, 2 * np.pi
        d = x.shape[1]
        
        sum1 = np.sum(x**2, axis=1)
        sum2 = np.sum(np.cos(c * x), axis=1)
        
        term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
        term2 = -np.exp(sum2 / d)
        
        result = term1 + term2 + a + np.e
        return result.item() if result.size == 1 else result


class RosenbrockFunction(BenchmarkFunction):
    """Rosenbrock function (narrow valley)"""
    
    def __init__(self, dim=2):
        bounds = [(-5, 10)] * dim
        super().__init__('Rosenbrock', bounds, optimum=0.0)
        
    def __call__(self, x):
        x = np.atleast_2d(x)
        result = np.sum(100 * (x[:, 1:] - x[:, :-1]**2)**2 + (1 - x[:, :-1])**2, axis=1)
        return result.item() if result.size == 1 else result


class RastriginFunction(BenchmarkFunction):
    """Rastrigin function (highly multimodal)"""
    
    def __init__(self, dim=2):
        bounds = [(-5.12, 5.12)] * dim
        super().__init__('Rastrigin', bounds, optimum=0.0)
        
    def __call__(self, x):
        x = np.atleast_2d(x)
        A = 10
        n = x.shape[1]
        result = A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=1)
        return result.item() if result.size == 1 else result


class FlowerFunction(BenchmarkFunction):
    """
    Flower function for safe optimization
    Has petal-like safe/unsafe regions
    """
    
    def __init__(self):
        super().__init__('Flower', [(-1, 1), (-1, 1)], optimum=1.0)
        
    def __call__(self, x):
        """
        Flower function: f(x1, x2) = r^3 * cos(4*theta)
        where r = sqrt(x1^2 + x2^2), theta = arctan2(x2, x1)
        Maximum at origin (0, 0)
        """
        x = np.atleast_1d(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        x1, x2 = x[:, 0], x[:, 1]
        r = np.sqrt(x1**2 + x2**2)
        theta = np.arctan2(x2, x1)
        
        result = (r ** 3) * np.cos(4 * theta)
        return result.item() if result.size == 1 else result


# Benchmark registry
BENCHMARK_SUITE = {
    'forrester': ForresterFunction,
    'gramacy_lee': GramacyLeeFunction,
    'ackley': AckleyFunction,
    'rosenbrock': RosenbrockFunction,
    'rastrigin': RastriginFunction,
    'flower': FlowerFunction,
}


def get_benchmark(name, **kwargs):
    """
    Get benchmark function by name
    
    Args:
        name: Benchmark name ('forrester', 'ackley', etc.)
        **kwargs: Additional arguments passed to benchmark constructor
        
    Returns:
        BenchmarkFunction instance
    """
    if name not in BENCHMARK_SUITE:
        available = ', '.join(BENCHMARK_SUITE.keys())
        raise ValueError(f"Unknown benchmark '{name}'. Available: {available}")
    return BENCHMARK_SUITE[name](**kwargs)

