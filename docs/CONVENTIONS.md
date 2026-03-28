# Conventions

## Language

- Code: English (variable names, comments, docstrings)
- Documentation: English

## Python Style

### General
- Python 3.10+ (use type hints)
- Follow PEP 8
- Max line length: 100 characters
- Use f-strings for formatting

### Naming
- Classes: `PascalCase` → `KalmanFilter`, `MultiTargetTracker`
- Functions/methods: `snake_case` → `get_state`, `measure_batch`
- Variables: `snake_case` → `true_state`, `noise_std`
- Constants: `UPPER_SNAKE` → `DEFAULT_DT`, `MAX_MISSED`
- Private: prefix with `_` → `_compute_kalman_gain`
- Files: `snake_case.py` → `multi_target.py`

### Type Hints
```python
def step(self, z: np.ndarray) -> np.ndarray:
    ...

def rmse(true_states: np.ndarray, estimated_states: np.ndarray) -> float:
    ...
```

### Docstrings (Google style)
```python
def step(self, z: np.ndarray) -> np.ndarray:
    """Run one predict-update cycle.

    Args:
        z: Measurement vector [x, y], shape (2,).

    Returns:
        Updated state estimate [x, y, vx, vy], shape (4,).
    """
```

### Imports (order)
```python
# 1. Standard library
import os
from typing import Optional

# 2. Third party
import numpy as np
import matplotlib.pyplot as plt

# 3. Local
from radarsim.sim.target import Target
from radarsim.tracker.kf import KalmanFilter
```

## NumPy Conventions

- State vector: shape `(4,)` — flat 1D array `[x, y, vx, vy]`
- Measurement vector: shape `(2,)` — flat 1D array `[x, y]`
- Matrices (F, H, P, Q, R): 2D arrays `(n, m)`
- Trajectory: shape `(n_steps, 4)` — each row is a state
- Measurements array: shape `(n_steps, 2)` — each row is a measurement

**Important:** Always use flat 1D arrays `(4,)` for state and `(2,)` for measurements. Never use column vectors `(4, 1)` or `(2, 1)`. This avoids reshape headaches and makes indexing cleaner. The KalmanFilter class handles matrix math internally using 2D matrices but exposes flat arrays externally.

## File Organization

- One class per file (if class is substantial)
- Related utility functions can share a file
- Each module has `__init__.py` that exports public API
- Examples are standalone scripts that import from `radarsim`

## Testing

- Use pytest
- Test files mirror source: `radarsim/tracker/kf.py` → `tests/test_kf.py`
- Test names: `test_<what>_<condition>` → `test_predict_constant_velocity`, `test_update_reduces_uncertainty`
- Each test tests one thing

## Git

### Commit Messages
Format: `<type>: <short description>`

Types:
- `init` — project setup
- `feat` — new feature
- `fix` — bug fix
- `refactor` — code restructuring without behavior change
- `test` — adding or fixing tests
- `docs` — documentation only
- `style` — formatting, no logic change

Examples:
```
init: create project structure and requirements
feat: add linear target simulation
fix: correct Q matrix dt scaling
test: add Kalman filter convergence test
docs: add ECM analysis results to README
```

### Commit Granularity
- One logical change per commit
- Every commit should leave the project in a working state
- Never commit broken code
- Commit after each completed task in ROADMAP.md

## Output Files

- Generated plots go to `output/` (gitignored)
- Plots saved for README go to `docs/images/` (tracked)
- Use `plt.savefig()` not `plt.show()` in example scripts
- Save as PNG, 150 DPI minimum
