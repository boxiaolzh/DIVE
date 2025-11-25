"""Fixed DIVE test"""
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.DIVE.main import DynamicDimBO

def sphere_function(x):
    return np.sum(np.array(x)**2)

def dive_objective(x):
    """Fixed: constraints now vary with x"""
    f_val = sphere_function(x)
    x_arr = np.array(x)

    # Varying constraints (have non-zero MI with x)
    x_sum = np.sum(x_arr)
    x_norm = np.linalg.norm(x_arr)

    gain = 65.0 + 5 * np.tanh(x_sum)        # 60-70 dB
    phase = 70.0 + 10 * np.tanh(x_norm)     # 60-80 deg
    gbw = 5e6 * (1 + 0.2 * np.tanh(x_arr[0])) # 4-6 MHz
    current = f_val

    return np.array([[gain, current, phase, gbw]])

# Setup
dim = 6
bounds = [(-5.0, 5.0) for _ in range(dim)]
constraints = {'gain': 60.0, 'phase': 60.0, 'gbw': 4e6}

print("Initializing DIVE...")
bo = DynamicDimBO(
    total_dim=dim,
    objective_function=dive_objective,
    bounds=bounds,
    constraints=constraints,
    seed=42,
    initial_samples=15
)

print("\nRunning optimization...")
bo.optimize(n_iter=30)

# Detailed results
print("\n" + "="*50)
print(f"Total evaluations: {len(bo.X)}")

feasible_count = len([y for y in bo.y if bo._is_feasible(y)])
print(f"Feasible solutions: {feasible_count}/{len(bo.y)}")

if bo.best_feasible_point is not None:
    best_val = sphere_function(bo.best_feasible_point)
    print(f"\n✓ Optimization successful!")
    print(f"  Best value: {best_val:.6f}")
    print(f"  Best point: {bo.best_feasible_point}")
    print(f"  Distance to optimum: {np.linalg.norm(bo.best_feasible_point):.6f}")
    print(f"  Active dims: {len(bo.active_dims)}/{dim}")
else:
    print(f"\n⚠ best_feasible_point is None")
    if hasattr(bo, 'pareto_set') and len(bo.pareto_set) > 0:
        print(f"  But Pareto set contains {len(bo.pareto_set)} solutions")

print("="*50)
