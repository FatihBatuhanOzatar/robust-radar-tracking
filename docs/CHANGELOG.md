# Changelog

All notable changes to this project will be documented in this file.

Format: Each entry includes the date, commit type, and description of what changed and why.

---

## 2026-03-28 — Project Initialization

- **init:** Created project documentation (PROJECT.md, ARCHITECTURE.md, ROADMAP.md, CONVENTIONS.md, CHANGELOG.md)
- **init:** Established architecture: modular design with sim/, tracker/, analysis/, viz/ packages
- **init:** Defined 6-phase roadmap from single-target CV tracking to optional advanced extensions
- **init:** Set coding conventions: flat (4,) state arrays, Google-style docstrings, conventional commits

## 2026-03-28 — Project Structure

- **init:** Created package directories: `radarsim/`, `radarsim/sim/`, `radarsim/tracker/`, `radarsim/analysis/`, `radarsim/viz/` with `__init__.py` files
- **init:** Created `examples/` and `tests/` directories
- **init:** Added `requirements.txt` (numpy, matplotlib, pytest)
- **init:** Added `.gitignore` (Python, IDE, OS files, `output/` directory)

## 2026-03-28 — Linear Target Simulation

- **feat:** Implemented `Target` class in `radarsim/sim/target.py` with constant velocity motion model
- CV model updates position via `x += vx*dt`, velocity unchanged — pure ground truth, no noise
- `step(dt)` advances one time step, returns flat `(4,)` state `[x, y, vx, vy]`
- `get_trajectory(dt, n_steps)` generates full trajectory `(n_steps, 4)` non-destructively
- Future models (`ct`, `random`) raise `NotImplementedError` until Phase 2
- Exported `Target` from `radarsim.sim` subpackage

## 2026-03-28 — Noisy Radar Measurement Generation

- **feat:** Implemented `Radar` class in `radarsim/sim/radar.py` with Gaussian noise model
- `measure(true_state)` extracts position from state `(4,)`, adds independent Gaussian noise, returns `(2,)`
- `measure_batch(true_states)` vectorized batch processing for full trajectories `(n_steps, 4)` → `(n_steps, 2)`
- Uses `np.random.default_rng()` with optional `seed` keyword for reproducible noise
- Exported `Radar` from `radarsim.sim` subpackage

## 2026-03-28 — Constant Velocity Kalman Filter

- **feat:** Implemented `KalmanFilter` class in `radarsim/tracker/kf.py` with constant-velocity model
- State vector `[x, y, vx, vy]` shape `(4,)` — flat arrays externally, 2D matrix math internally
- Physically-derived Q matrix from acceleration uncertainty (Bar-Shalom formulation), not arbitrary diagonal
- `init_state(z)` sets position from first measurement, velocity to zero, large initial P for velocity
- `predict()` projects state and covariance forward: `x = F @ x`, `P = F @ P @ F.T + Q`
- `update(z)` incorporates measurement via Kalman gain, uses Joseph form for numerical stability
- `step(z)` combines predict + update; `step_no_measurement()` predict-only for future ECM dropout
- R matrix: `diag(r_x², r_y²)` — separate configurable noise per axis
- Quick-test: ~42% RMSE improvement over raw radar on 50-step constant-velocity scenario
- Exported `KalmanFilter` from `radarsim.tracker` subpackage

## 2026-03-28 — RMSE and Error Metrics

- **feat:** Implemented `radarsim/analysis/metrics.py` with three functions
- `rmse(true_states, estimated_states)` — scalar RMSE on position across all time steps
- `position_error_over_time(true_states, estimated_states)` — per-step Euclidean position error, shape `(n_steps,)`
- `velocity_error_over_time(true_states, estimated_states)` — per-step Euclidean velocity error, shape `(n_steps,)`
- All functions take `(n_steps, 4)` arrays, use `np.linalg.norm` for Euclidean distance
- Exported all three functions from `radarsim.analysis` subpackage

## 2026-03-28 — Tracking Result Visualization

- **feat:** Implemented `radarsim/viz/plots.py` with two plotting functions
- `plot_tracking_result(true, measured, estimated, title)` — 2D plot comparing true trajectory (blue solid), radar measurements (red scatter), and KF estimate (green dashed), with start/end markers, grid, equal aspect ratio
- `plot_error_over_time(errors, title)` — error timeline with mean error reference line
- Both functions return `matplotlib.figure.Figure` for caller to save via `fig.savefig()`, no `plt.show()` calls
- Exported both functions from `radarsim.viz` subpackage

## 2026-03-28 — Single Target Tracking Example

- **feat:** Added `examples/single_target.py` demo for Phase 1
- Initialized tracking scenario with constant velocity target (60 steps, dt=1s)
- Configured 2D radar with 25m measurement noise vs KF with q=0.5
- Printed tracked metrics: raw radar RMSE (30.02m) and KF RMSE (16.51m) verifying ~45% improvement
- Hooked up `radarsim.viz.plots` to output standard `output/single_target_tracking.png` and `output/single_target_error.png`

