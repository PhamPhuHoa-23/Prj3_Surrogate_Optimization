# Bayesian Optimization & SafeOpt Implementation

**Course Project: Algorithms for Optimization (2nd Edition)**  
**Chapter 19: Surrogate Optimization**

Implementation of surrogate-based optimization algorithms from Chapter 19 of *Algorithms for Optimization, 2nd Edition* (Kochenderfer & Wheeler, 2025).

---

## Project Description

**Group Assignment Requirements:**
- Select one chapter from *Algorithms for Optimization* (excluding Introduction, Bracketing, Direct)
- Re-present concepts and implement demonstrations in Python
- Encouraged to extend topics with recent research directions
- Deliverables: Report (PDF), Source code (Python), LaTeX source (if applicable)

**Our Focus:** Chapter 19 - Surrogate Optimization using Gaussian Processes for Bayesian Optimization and Safe Exploration.

---

## Project Structure

```
prj3/
├── src/
│   ├── core/
│   │   ├── gaussian_process.py    # GP regression with RBF kernel
│   │   ├── acquisition.py         # All acquisition functions (EI, PI, LCB, etc.)
│   │   └── optimizer.py           # BayesianOptimizer & SafeOptimizer classes
│   ├── benchmarks/
│   │   └── functions.py           # Test functions (Forrester, Ackley, Flower, etc.)
│   └── utils/
│       └── config.py              # Visualization settings
├── tests/
│   ├── test_gp.py
│   ├── test_acquisition.py
│   ├── test_optimizer.py
│   ├── test_benchmarks.py
│   └── run_all_tests.py
├── run_benchmarks.py              # Main experiment: compare acquisitions
├── run_safe_optimization.py       # SafeOpt experiment
├── README.md
└── requirements.txt
```

---

## Core Implementations

### 1. Gaussian Process (`src/core/gaussian_process.py`)

**Squared Exponential (RBF) Kernel:**

```python
K(x, x') = exp(-||x - x'||^2 / (2 * length_scale^2))
```

**Implementation approach:**
- Use `scipy.spatial.distance.cdist` for efficient distance computation
- Cholesky decomposition for numerical stability: `K = L·L^T`
- Predictions via `cho_solve` instead of matrix inversion

**Key methods:**
```python
def fit(self, X, y):
    # Build kernel matrix K
    K = self.squared_exponential_kernel(X, X)
    K += self.noise * I  # Add noise to diagonal
    self.L = cholesky(K)  # Cholesky factorization

def predict(self, X_test):
    K_s = self.squared_exponential_kernel(X_train, X_test)
    alpha = cho_solve((L, True), y_train)
    mean = K_s^T · alpha
    
    # Variance: var = K(x*,x*) - K(x*,X)^T · K^(-1) · K(x*,X)
    v = cho_solve((L, True), K_s)
    var = diag(K_ss) - sum(K_s * v, axis=0)
    std = sqrt(max(var, 0))
    
    return mean, std
```

---

### 2. Acquisition Functions (`src/core/acquisition.py`)

Each acquisition function balances **exploration** (sampling uncertain regions) vs **exploitation** (sampling near predicted optimum).

#### **2.1 Probability of Improvement (PI)** - Algorithm 19.1

**Concept:** Maximize probability that new sample will be better than current best.

**Mathematical formulation:**
```
P(f(x) < y_min) = Φ((y_min - μ(x)) / σ(x))
```
where Φ is the standard normal CDF.

**Implementation:**
```python
def probability_of_improvement(mean, std, y_best, xi=0.01):
    """
    Args:
        mean: GP predicted mean at candidate points
        std: GP predicted standard deviation
        y_best: Current best observed value (for minimization)
        xi: Exploration parameter (small positive value)
    
    Returns:
        PI values (higher = more likely to improve)
    """
    with np.errstate(divide='warn'):
        z = (y_best - xi - mean) / std
        pi = norm.cdf(z)  # Standard normal CDF
    
    # Handle zero variance (already sampled points)
    pi[std == 0.0] = 0.0
    return pi
```

**Key insight:** PI only cares about *whether* we improve, not *how much*.

---

#### **2.2 Expected Improvement (EI)** - Algorithm 19.2

**Concept:** Maximize expected amount of improvement over current best.

**Mathematical formulation:**
```
EI(x) = E[max(y_min - f(x), 0)]
      = (y_min - μ(x)) · Φ(z) + σ(x) · φ(z)
```
where z = (y_min - μ(x)) / σ(x), φ is standard normal PDF.

**Implementation:**
```python
def expected_improvement(mean, std, y_best, xi=0.01):
    """
    Computes expected improvement using closed-form solution.
    
    Derivation:
    - Improvement I(y) = max(y_min - y, 0)
    - Under GP, y ~ N(μ, σ²)
    - Integrate I(y) over this distribution
    - Results in closed form above
    """
    with np.errstate(divide='warn'):
        z = (y_best - xi - mean) / std
        ei = (y_best - xi - mean) * norm.cdf(z) + std * norm.pdf(z)
    
    ei[std == 0.0] = 0.0
    return ei
```

**Key insight:** EI balances probability of improvement AND magnitude of improvement.

---

#### **2.3 Lower Confidence Bound (LCB)** - Section 19.3

**Concept:** Trade off between predicted mean (exploitation) and uncertainty (exploration) via tunable parameter α.

**Mathematical formulation:**
```
LCB(x) = μ(x) - α·σ(x)
```

**Implementation:**
```python
def lower_confidence_bound(mean, std, alpha=2.0):
    """
    Args:
        alpha: Exploration-exploitation trade-off
               - α = 0: pure exploitation (minimize mean)
               - α → ∞: pure exploration (maximize uncertainty)
               - α = 2.0: common default (≈95% confidence interval)
    
    Returns:
        LCB values (lower = better for minimization)
    """
    return mean - alpha * std
```

**Selecting next point:** `x_next = argmin LCB(x)` (minimize the lower bound)

**Key insight:** 
- Small α: greedy, focuses on best predicted regions
- Large α: explores uncertain regions even if predicted value is poor

---

#### **2.4 Prediction-Based Exploration (PBE)** - Section 19.1

**Concept:** Pure exploitation - select point with best predicted mean.

**Mathematical formulation:**
```
x_next = argmin μ(x)
```

**Implementation:**
```python
def prediction_based(mean, std, y_best=None):
    """
    Simplest approach: trust the GP mean completely.
    Ignores uncertainty (std) and current best (y_best).
    
    Pros: Fast convergence when near optimum
    Cons: Can get stuck re-sampling same region
    """
    return mean  # Return as-is for minimization
```

**Key insight:** No exploration - can waste evaluations sampling near existing points.

---

#### **2.5 Error-Based Exploration (EBE)** - Section 19.2

**Concept:** Pure exploration - sample where uncertainty is highest.

**Mathematical formulation:**
```
x_next = argmax σ(x)
```

**Implementation:**
```python
def error_based(mean, std, y_best=None):
    """
    Opposite of PBE: ignore predicted values, only reduce uncertainty.
    
    Pros: Builds accurate global model
    Cons: Wastes evaluations in regions far from optimum
    """
    return std  # Higher std = more exploration
```

**Key insight:** Good for learning function shape, poor for optimization.

---

#### **Upper Confidence Bound (UCB)** - For SafeOpt

**Concept:** Used in SafeOpt for maximization/safety analysis.

**Mathematical formulation:**
```
UCB(x) = μ(x) + α·σ(x)
```

**Implementation:**
```python
def upper_confidence_bound(mean, std, alpha=2.0):
    """
    Optimistic estimate - used in SafeOpt to:
    1. Identify safe regions (where UCB might exceed threshold)
    2. Find potential maximizers (where UCB is highest)
    """
    return mean + alpha * std
```

---

### 3. Bayesian Optimizer (`src/core/optimizer.py`)

**Main optimization loop:**

```python
class BayesianOptimizer:
    def optimize(self, n_init=5, n_iter=20):
        # 1. Random initialization
        X_samples = random_uniform(bounds, n_init)
        y_samples = [f(x) for x in X_samples]
        
        # 2. Iterative optimization
        for i in range(n_iter):
            # Fit GP to current data
            gp.fit(X_samples, y_samples)
            
            # Find next point by optimizing acquisition
            x_next = argmax acquisition(gp.predict(X))
            y_next = f(x_next)
            
            # Add to dataset
            X_samples.append(x_next)
            y_samples.append(y_next)
        
        # 3. Return best observed point
        best_idx = argmin(y_samples)
        return X_samples[best_idx], y_samples[best_idx]
```

**Acquisition optimization approach:**
```python
def _optimize_acquisition(self):
    # 1. Generate many random candidates
    X_random = uniform(0, 1, size=(1000 * dim, dim))
    
    # 2. Evaluate acquisition at all candidates
    acq_values = acquisition_function(X_random)
    
    # 3. Take top-N candidates
    top_candidates = X_random[argsort(acq_values)[-25:]]
    
    # 4. Local optimization from each
    for x0 in top_candidates:
        x_opt = minimize(neg_acquisition, x0, method='L-BFGS-B')
        # Track best found
    
    return best_x
```

---

### 4. SafeOpt Algorithm (`src/core/optimizer.py`) - Section 19.6

**Problem:** Minimize `f(x)` while ensuring `f(x) ≥ y_threshold` (safety constraint).

**SafeOpt Strategy:**

1. **Safe Set S:** Points where lower confidence bound exceeds threshold
   ```
   S = {x : LCB(x) = μ(x) - √β·σ(x) ≥ y_threshold}
   ```

2. **Potential Minimizers M:** Safe points that might contain optimum
   ```
   M = {x ∈ S : LCB(x) ≤ min_{x'∈S} UCB(x')}
   ```

3. **Potential Expanders E:** Safe points that might expand safe region
   ```
   E = {x ∈ S : might lead to larger S if sampled at LCB(x)}
   ```

**Selection rule:**
```
x_next = argmax_{x ∈ M ∪ E} [UCB(x) - LCB(x)]
```
(Select point with highest uncertainty among minimizers and expanders)

**Implementation:**
```python
class SafeOptimizer:
    def select_next_point(self, n_candidates=1000):
        # Generate candidates
        X = random_uniform(bounds, n_candidates)
        
        # Compute confidence bounds
        u, l = self._confidence_bounds(X)  # UCB, LCB
        
        # Identify safe points (LCB above threshold)
        safe_mask = (l >= self.y_threshold)
        
        if not any(safe_mask):
            return X[argmax(l)]  # No safe points: pick safest
        
        # Among safe points: pick highest UCB (optimistic)
        safe_u = u.copy()
        safe_u[~safe_mask] = -inf
        return X[argmax(safe_u)]
    
    def _confidence_bounds(self, X):
        mean, std = self.gp.predict(X)
        u = mean + sqrt(self.beta) * std
        l = mean - sqrt(self.beta) * std
        return u, l
```

**Safety probability check:**
```python
def _compute_safe_set(self, X):
    mean, std = self.gp.predict(X)
    
    # Probability that f(x) ≥ threshold
    z = (self.y_threshold - mean) / std
    p_safe = norm.cdf(z)  # P(f(x) < threshold)
    p_safe = 1 - p_safe    # P(f(x) ≥ threshold)
    
    return p_safe >= self.p_safe  # e.g., 0.95
```

---

## Benchmark Functions

**Univariate (1D):**
- **Forrester:** `f(x) = (6x-2)^2 x sin(12x-4)` - Smooth, single global minimum
- **Gramacy & Lee:** `f(x) = sin(10πx)/(2x) + (x-1)^4` - Discontinuous

**Multivariate (2D+):**
- **Ackley:** Many local minima, single global minimum at origin
- **Rosenbrock:** Narrow curved valley, global minimum at (1,1,...,1)
- **Rastrigin:** Highly multimodal with regular grid of local minima

**Safe Optimization:**
- **Flower:** `f(x₁,x₂) = r^3 x cos(4θ)` - Petal-like safe/unsafe regions

---

## How to Run

### Installation
```bash
pip install -r requirements.txt
```

### 1. Run Tests
```bash
cd tests
python run_all_tests.py
```

### 2. Compare Acquisition Functions
```bash
python run_benchmarks.py
```

### 3. Run SafeOpt Experiment
```bash
python run_safe_optimization.py
```

---

## License

Academic/Research use for course project.
