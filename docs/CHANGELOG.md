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

## 2026-03-28 — Kalman Filter Unit Tests

- **test:** Created `tests/test_kf.py` with 3 core unit tests
- `test_predict_constant_velocity`: Verifies position advances by exactly `v * dt` and velocity remains constant
- `test_update_reduces_uncertainty`: Verifies that incorporating a radar measurement strictly reduces the trace of the covariance matrix `P`
- `test_straight_line_convergence`: Simulates 50 steps of noisy Tracking tracking with `radar.seed=42`, asserting KF RMSE is lower than RAW RMSE and final position bounds
- Phase 1 completed successfully

## 2026-03-28 — Coordinated Turn Target Model

- **feat:** Added coordinated turn (CT) motion model to `Target` class in `radarsim/sim/target.py`
- CT model moves the target along a circular arc at constant speed, tracking heading internally
- New constructor parameter `turn_rate` (rad/s, required when `model="ct"`)
- Near-zero turn rate (`abs < 1e-10`) delegates to CV model — physically correct (zero turn = straight line) and avoids division by zero
- Exposed state stays `[x, y, vx, vy]` shape `(4,)` — heading is internal only
- `get_trajectory()` saves/restores both `state` and `_heading` for non-destructive operation
- **test:** Created `tests/test_target.py` with 8 tests covering CV basics and CT model:
  - CT step returns correct shape `(4,)`
  - CT preserves speed magnitude across 100 steps
  - Near-zero turn rate produces identical trajectory to CV
  - `get_trajectory()` is non-destructive for CT (state + heading restored)
  - CT actually changes velocity direction over time
  - CT requires `turn_rate` parameter (ValueError if missing)

## 2026-03-28 — Random Maneuver Target Model

- **feat:** Added random maneuver model to `Target` class in `radarsim/sim/target.py`
- Random acceleration perturbations drawn from `N(0, accel_std²)` independently in x and y each step
- Position update uses full kinematic equation: `x += v*dt + 0.5*a*dt²`
- New constructor parameters: `accel_std` (required for `model="random"`), `seed` (optional reproducibility)
- RNG state (`np.random.default_rng`) saved/restored in `get_trajectory()` for non-destructive + reproducible operation
- **test:** Added 5 tests for random model to `tests/test_target.py`:
  - Random step returns correct shape `(4,)`
  - Random model changes velocity between steps
  - Same seed produces identical trajectories
  - `get_trajectory()` restores state and RNG state
  - Random model requires `accel_std` parameter (ValueError if missing)

## 2026-03-28 — Maneuver Scenario with KF Breakdown Analysis

- **feat:** Created `examples/maneuver.py` — Phase 2 demo showing CV KF failure during coordinated turn
- 3-phase scenario: straight (30 steps) → coordinated turn at ω=0.05 rad/s (40 steps) → straight (30 steps)
- Trajectory built by stitching three Target instances, passing end state of each phase to the next
- Per-segment RMSE analysis: straight=16.77m, turn=77.22m (4.6x degradation), recovery=40.77m
- **feat:** Added optional `vlines` parameter to `plot_error_over_time()` in `radarsim/viz/plots.py`
  - Draws vertical annotation lines with configurable color, style, and label
  - Backward compatible — no change when `vlines` is not passed
- Outputs: `output/maneuver_tracking.png` (2D trajectory), `output/maneuver_error.png` (error timeline with maneuver window annotated)

## 2026-03-29 — ECM Simulation Model

- **feat:** Implemented `ECMModel` class in `radarsim/sim/ecm.py` with three electronic countermeasure modes
- **noise_spike:** Adds extra Gaussian noise scaled by `(noise_multiplier - 1) * noise_std` on top of existing radar noise, so total effective noise is `noise_multiplier * noise_std`
- **dropout:** Drops measurements with configurable probability `dropout_prob`, returning `(None, False)` for predict-only KF operation
- **bias:** Adds a fixed systematic offset vector `[bx, by]` to measurements — the hardest case for a KF since bias is indistinguishable from target motion
- `apply(measurement, t)` returns `(degraded_measurement_or_None, is_valid)` tuple; outside the ECM window `[ecm_start, ecm_end)`, measurements pass through unchanged
- ECM window boundaries: `ecm_start` is inclusive, `ecm_end` is exclusive
- Optional `seed` parameter for reproducible dropout decisions and noise generation
- Exported `ECMModel` from `radarsim.sim` subpackage
- **Note:** `step_no_measurement()` already existed in `KalmanFilter` since Phase 1 — ROADMAP task marked DONE
- **test:** Created `tests/test_ecm.py` with 13 tests covering all three modes:
  - noise_spike: returns (ndarray, True) with shape (2,), passthrough outside window, adds noise during window, requires positive noise_std
  - dropout: returns (None, False) during window, passthrough outside window, partial probability keeps some measurements
  - bias: adds exact offset during window, passthrough outside window, requires bias parameter
  - validation: invalid mode raises ValueError, window boundary semantics (start inclusive, end exclusive)
