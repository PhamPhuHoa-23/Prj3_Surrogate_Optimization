import sys
sys.path.insert(0, '..')

import numpy as np
import json
import os
from datetime import datetime

from src.core import BayesianOptimizer
from src.benchmarks import get_benchmark


def run_single_experiment(benchmark_name, acquisition, n_runs=10, n_iter=30, n_init=5):
    """Run single benchmark with specific acquisition function"""
    
    benchmark = get_benchmark(benchmark_name)
    results = []
    
    print(f"\n{'='*70}")
    print(f"Benchmark: {benchmark.name} | Acquisition: {acquisition.upper()}")
    print(f"{'='*70}")
    
    for run in range(n_runs):
        print(f"Run {run+1}/{n_runs}...", end=' ')
        
        optimizer = BayesianOptimizer(
            objective_func=benchmark,
            bounds=benchmark.bounds,
            acquisition=acquisition,
            gp_params={'length_scale': 0.5, 'noise': 1e-6}
        )
        
        X_best, y_best, history = optimizer.optimize(
            n_init=n_init,
            n_iter=n_iter,
            random_state=42 + run,
            verbose=False
        )
        
        results.append({
            'X_best': X_best.tolist(),
            'y_best': float(y_best),
            'history_y': [float(y) for y in optimizer.y_samples]
        })
        
        print(f"Best: {y_best:.6f}")
    
    return results


def main():
    """Run full benchmark suite"""
    
    # Configuration
    benchmarks = ['forrester', 'gramacy_lee', 'ackley', 'rosenbrock', 'rastrigin']
    acquisitions = ['pbe', 'ebe', 'ei', 'pi', 'lcb']
    n_runs = 10
    n_iter = 30
    n_init = 5
    
    output_dir = 'output_benchmarks'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}
    
    print("="*70)
    print("BAYESIAN OPTIMIZATION BENCHMARK SUITE")
    print("="*70)
    print(f"Benchmarks: {len(benchmarks)}")
    print(f"Acquisitions: {len(acquisitions)}")
    print(f"Runs per experiment: {n_runs}")
    print(f"Iterations: {n_iter}")
    print(f"Init points: {n_init}")
    
    # Run experiments
    for benchmark_name in benchmarks:
        all_results[benchmark_name] = {}
        
        for acquisition in acquisitions:
            results = run_single_experiment(
                benchmark_name, acquisition, n_runs, n_iter, n_init
            )
            all_results[benchmark_name][acquisition] = results
    
    # Save results
    output_file = os.path.join(output_dir, f'results_{timestamp}.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}\n")
    
    # Print summary
    print("\nSummary (mean ± std over 10 runs):")
    print("-" * 70)
    
    for benchmark_name in benchmarks:
        print(f"\n{benchmark_name.upper()}:")
        for acquisition in acquisitions:
            results = all_results[benchmark_name][acquisition]
            best_values = [r['y_best'] for r in results]
            mean_best = np.mean(best_values)
            std_best = np.std(best_values)
            print(f"\t{acquisition.upper():6s}: {mean_best:8.4f} ± {std_best:6.4f}")


if __name__ == "__main__":
    main()

