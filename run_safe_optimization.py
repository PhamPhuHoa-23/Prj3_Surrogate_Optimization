import sys
sys.path.insert(0, '..')

import numpy as np
import json
import os
from datetime import datetime

from src.core import GaussianProcess, SafeOptimizer
from src.benchmarks import get_benchmark


def run_safeopt_experiment(n_runs=10, n_iter=40, beta=10.0, y_threshold=-0.2):
    """
    Run SafeOpt on Flower function with safety constraints
    
    Args:
        n_runs: Number of independent runs
        n_iter: Number of optimization iterations
        beta: Confidence parameter
        y_threshold: Safety threshold
    """
    
    benchmark = get_benchmark('flower')
    output_dir = 'output_safe'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_runs = []
    
    print("="*70)
    print("SAFEOPT EXPERIMENT - FLOWER FUNCTION")
    print("="*70)
    print(f"Safety threshold: y >= {y_threshold}")
    print(f"Beta parameter: {beta}")
    print(f"Iterations: {n_iter}")
    print(f"Runs: {n_runs}")
    print("="*70)
    
    for run in range(n_runs):
        print(f"\nRun {run+1}/{n_runs}...")
        
        # Random safe initialization
        np.random.seed(100 + run)
        n_init = 5
        X_init = []
        y_init = []
        
        # Sample safe points
        attempts = 0
        while len(X_init) < n_init and attempts < 100:
            r = np.random.uniform(0.3, 0.8)
            theta = np.random.uniform(0, 2*np.pi)
            x = np.array([r * np.cos(theta), r * np.sin(theta)])
            
            y = benchmark(x)
            if isinstance(y, np.ndarray):
                y = y.item()
            
            if y >= y_threshold:
                X_init.append(x)
                y_init.append(y)
            
            attempts += 1
        
        # Fallback to conservative points near origin
        while len(X_init) < n_init:
            angle = len(X_init) * (2*np.pi / n_init)
            x = np.array([0.4*np.cos(angle), 0.4*np.sin(angle)])
            y = benchmark(x)
            X_init.append(x)
            y_init.append(y.item() if isinstance(y, np.ndarray) else y)
        
        X_init = np.array(X_init)
        y_init = np.array(y_init)
        
        # Run SafeOpt
        gp = GaussianProcess(length_scale=0.5, noise=1e-6)
        safe_opt = SafeOptimizer(
            gp=gp,
            bounds=benchmark.bounds.tolist(),
            y_threshold=y_threshold,
            beta=beta,
            p_safe=0.95
        )
        
        X_best, y_best, history = safe_opt.optimize(
            objective_func=benchmark,
            X_init=X_init,
            y_init=y_init,
            n_iter=n_iter,
            verbose=False
        )
        
        # Record results
        all_runs.append({
            'X_best': X_best.tolist(),
            'y_best': float(y_best),
            'violations': int(history['violations']),
            'final_best': float(np.max(safe_opt.y_samples))
        })
        
        print(f"\tBest: {y_best:.6f}, Violations: {history['violations']}/{n_iter}")
    
    # Compute summary
    summary = {
        'mean_best': float(np.mean([r['final_best'] for r in all_runs])),
        'std_best': float(np.std([r['final_best'] for r in all_runs])),
        'total_violations': int(np.sum([r['violations'] for r in all_runs])),
        'violation_rate': float(np.sum([r['violations'] for r in all_runs]) / (n_runs * n_iter))
    }
    
    # Save results
    results = {
        'benchmark': 'flower',
        'timestamp': timestamp,
        'config': {
            'y_threshold': y_threshold,
            'n_iter': n_iter,
            'n_runs': n_runs,
            'beta': beta
        },
        'runs': all_runs,
        'summary': summary
    }
    
    output_file = os.path.join(output_dir, f'results_safe_{timestamp}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"SUMMARY:")
    print(f"Mean best: {summary['mean_best']:.6f} ± {summary['std_best']:.6f}")
    print(f"Total violations: {summary['total_violations']} / {n_runs * n_iter}")
    print(f"Violation rate: {summary['violation_rate']*100:.2f}%")
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run_safeopt_experiment(n_runs=10, n_iter=40, beta=10.0, y_threshold=-0.2)

