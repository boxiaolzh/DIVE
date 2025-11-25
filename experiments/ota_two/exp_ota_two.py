import os
import torch
import numpy as np
from src.DIVE.main import DynamicDimBO
from Simulation.simulation.ota_two.ota_two import OTA_two_simulation_all
from src.DIVE.utils import seed_set, get_bounds_from_config, load_config


def main(seed):
    """
    Optimize two-stage amplifier (180nm process)

    Parameters:
    seed: Random seed
    """
    # Set result save path
    base_dir = "./result"
    os.makedirs(base_dir, exist_ok=True)

    # Set filename
    prefix = "DIVE_OTA_two"
    file_path = os.path.join(base_dir, f'{prefix}_seed_{seed}.csv')

    # Load configuration
    experiment = 'experiment1'
    config = load_config("../config.yaml")
    bounds = get_bounds_from_config(config, experiment)
    constraints = {
        'gain': float(config[experiment]['constraints']['gain']),
        'phase': float(config[experiment]['constraints']['phase']),
        'gbw': float(config[experiment]['constraints']['gbw'])
    }

    # Use 180nm process simulation function
    objective_function = OTA_two_simulation_all

    # Set optimizer parameters (remove initial feasible point)
    bo_params = {
        'total_dim': len(bounds),
        'objective_function': objective_function,
        'bounds': bounds,
        'constraints': constraints,
        'seed': seed,
        'initial_samples': 20
    }

    print("Starting optimization of two-stage operational amplifier...")
    print(f"Optimization parameters:")
    print(f"  Total dimensions: {len(bounds)}")
    print(f"  Initial samples: 20")
    print(
        f"  Constraints: gain > {constraints['gain']}dB, phase > {constraints['phase']}°, gbw > {np.exp(constraints['gbw']):.0e}Hz")

    # Initialize optimizer
    bo = DynamicDimBO(**bo_params)

    # Run optimization
    bo.optimize(n_iter=380)

    # Output final results
    print(f"\nFinal optimization results:")
    if hasattr(bo, 'best_feasible_current') and bo.best_feasible_point is not None:
        print(f"Best feasible current found: {np.exp(bo.best_feasible_current):.2e} A")
        print(f"Best parameter point:")
        for i, val in enumerate(bo.best_feasible_point):
            print(f"  Parameter {i}: {val:.6e}")
    else:
        print("No feasible solution found")

    print(f"Final active dimensions: {bo.active_dims}")
    print(f"Dimension importance scores: {np.round(bo.dim_scores[bo.active_dims], 3)}")


if __name__ == "__main__":
    # Single run
    SEED = 1
    seed_set(seed=SEED)
    main(seed=SEED)
