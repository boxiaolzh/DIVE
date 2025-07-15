
# DIVE: Dynamic Information-Guided Variable Expansion for Analog Circuit Optimization

This repository contains the implementation of DIVE.

## Overview

DIVE addresses the "curse of dimensionality" in transistor sizing by progressively expanding the optimization space based on mutual information analysis, mimicking expert designers' workflow of focusing on key parameters first.

## Quick Start

```python
from src.DIVE.main import DynamicDimBO
from your_circuit_simulator import your_objective_function

# Initialize DIVE optimizer
bo = DynamicDimBO(
    total_dim=12,  # Total number of design parameters
    objective_function=your_objective_function,
    bounds=parameter_bounds,
    constraints={'gain': 60, 'phase': 60, 'gbw': 4e6},
    initial_samples=20,
    seed=42
)

# Run optimization
bo.optimize(n_iter=400)
```

## Citation

```bibtex
@article{dive2024,
  title={DIVE: Dynamic Information-Guided Variable Expansion for Deeper Analog Circuit Optimization},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```
