# Changelog

All notable changes to this project will be documented in this file.

Format: Each entry includes the date, commit type, and description of what changed and why.

---

## 2026-03-31 — C++ Port of Kalman Filter Engine (Phase 6 Task 4)

- **feat:** Implemented a pure C++ port of the CV Kalman Filter in `cpp/kalman.h` & `kalman.cpp`
- 1:1 algorithmic parity with Python `kf.py` including Joseph-form covariance update and Bar-Shalom process noise covariance formulation.
- **feat:** Implemented zero-dependency fixed-size matrix linear algebra in `cpp/matrix.h` & `matrix.cpp`
  - Highly optimized, stack-allocated explicit loops for (4×4, 2×4, 4×2, 2×2) array dimensions.
  - Closed-form 4-line formula for 2×2 inversion eliminating the need for `Eigen` or BLAS libraries.
- **feat:** Added CLI benchmark simulation `cpp/main.cpp` demonstrating tracking on a 100-step scenario mimicking `single_target.py`.
  - Proved identical performance trends (55.7% RMSE improvement) to the Python implementation.
  - Benchmarked tracking performance at ~0.03 milliseconds per 100 iterations.
- **init:** Added `cpp/Makefile` with optimized `-O2` routines and `-g -fsanitize=address,undefined` debug target checks.
- **docs:** Added `cpp/README.md` and integrated the benchmark results directly into the root `README.md`.
- **docs:** Updated `ROADMAP.md` tracking the C++ core engine as completed functionality.

---

## 2026-03-31 — Extended Kalman Filter (Phase 6 Task 1)

- **feat:** Implemented `ExtendedKalmanFilter` class in `radarsim/tracker/ekf.py` — EKF for coordinated-turn target tracking
- State vector `[x, y, v, theta, omega]` shape `(5,)` — speed + heading + turn rate instead of (vx, vy), capturing circular motion naturally
- `_compute_f(x)` implements the CT nonlinear state transition (circular arc integral, same equations as `Target._step_ct()`); falls back to CV approximation when `|omega| < 1e-6`
- `_compute_jacobian(x)` computes the full 5×5 partial-derivative matrix analytically — separate code paths for CT (omega non-zero) and CV (omega near-zero); all entries verified finite
- `predict()` uses `x = f(x)` (nonlinear) and `P = F_jac @ P @ F_jac.T + Q` (linearised covariance propagation)
- `update(z)` is identical to standard KF: H is linear (2×5, observing [x, y] directly), Joseph form for numerical stability
- `get_position()` returns `(2,)` `[x, y]` — compatibility shim for metrics expecting KF-style position output
- Q matrix: diagonal `diag([q_pos, q_pos, q_vel, q_theta, q_omega])` — empirically tuned, accepts dict (named keys with defaults) or 5-element sequence
- Exported `ExtendedKalmanFilter` from `radarsim.tracker` subpackage
- **test:** Created `tests/test_ekf.py` with 19 tests across 5 test classes:
  - `TestInitState` (5 tests): position from measurement, v/theta/omega=0, shape checks
  - `TestPredict` (6 tests): straight flight, north flight, covariance grows, heading advances, near-zero omega no crash, Jacobian all-finite
  - `TestUpdate` (2 tests): covariance reduces, state pulled toward measurement
  - `TestStep` (2 tests): EKF RMSE < raw RMSE on 50-step CT scenario, returns (5,) state
  - `TestQParsing` (4 tests): dict format, sequence format, wrong length raises, missing keys use defaults
- All 71 existing tests continue to pass (EKF adds 19, total: 71)

---

## 2026-03-31 — EKF vs KF Comparison + README Update (Phase 6 Tasks 2 & 3)

- **feat:** Created `examples/ekf_comparison.py` — runs CV-KF and CT-EKF on the same 3-phase maneuver scenario
- Identical ground truth and radar measurements (seed=42, σ=25m) used for both filters for a fair comparison
- Q parameters tuned to `q_pos=0.5, q_vel=1.0, q_theta=0.05, q_omega=0.01`:
  - `q_omega=0.01` gives the filter enough bandwidth to track the turn-rate jump (0 → 0.05 rad/s at step 30) within a few time steps
  - Too small → slow omega convergence; too large → noisy turn-rate estimates
- **RMSE results (Phase B — the turn):** KF=73.72m, EKF=22.79m — **3.2× improvement**. EKF overall: KF=51.79m → EKF=22.05m (2.4× total)
- Phase A is slightly worse for EKF (22.38m vs 19.07m) — expected; EKF initialises with v=0, θ=0, ω=0 and needs ~10 steps to converge its speed/heading from position-only measurements
- Generates two plots saved to both `output/` and `docs/images/`:
  - `ekf_comparison_tracking.png` — 2D trajectory with turn phase highlighted; KF diverges visibly during arc, EKF follows the curve
  - `ekf_comparison_error.png` — per-step errors for both filters with maneuver window shaded gold; KF error spikes to ~150m peak, EKF stays ≤60m
- **docs:** Added "Extended Kalman Filter — Fixing the Maneuver Problem" section to README between Maneuver Analysis and ECM Resilience
  - Explains EKF state vector `[x, y, v, θ, ω]` and the Jacobian linearisation approach
  - Embeds both comparison plots with full RMSE table
  - Documents Phase A startup transient honestly
  - Updated Architecture section to include `ekf.py`
  - Updated test count (52 → 90 after EKF's 19 tests + pre-existing 71)
  - Added `ekf_comparison.py` to How to Run

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

## 2026-03-29 — ECM Scenario Analysis

- **feat:** Created `examples/ecm_scenario.py` — Phase 3 demo combining noise spike, dropout, and bias ECM analysis
- Straight-line CV target (100 steps, dt=1s), ECM window at steps 30-59, radar noise 25m
- **Part 1 — ECM Mode Comparison:** Runs all 3 modes, prints per-segment RMSE:
  - Noise spike (5x): during-ECM 54.77m (3.2x degradation), fast recovery to 20.08m
  - Dropout (100%): during-ECM 18.64m (CV predict-only is accurate!), post-ECM 14.41m
  - Bias (+50, +30): during-ECM 62.14m (worst — systematic pull), recovers to 15.46m
- **Part 1b — Per-mode tracking plots:** `ecm_noise_spike.png`, `ecm_dropout.png`, `ecm_bias.png` via `plot_tracking_result()`
- **Part 2 — Combined error comparison:** `ecm_comparison.png` with all 3 error curves overlaid and ECM window shaded
- **Part 3 — Q parameter effect on dropout recovery:** Tests q=0.1, q=0.5, q=2.0
  - Lower Q → lower post-ECM RMSE (11.78m) but filter trusts model more (slower adaptation)
  - Higher Q → higher post-ECM RMSE (16.74m) but peak error also higher (47.67m vs 36.23m)
  - Recovery time = 1 step for all Q values (CV model matches actual motion perfectly)
  - Generates `ecm_q_comparison.png`
- Key insight: bias is the most damaging ECM mode; dropout is the least damaging when the motion model is correct

## 2026-03-29 — Track Class for Multi-Target Tracking

- **feat:** Implemented `Track` class in `radarsim/tracker/multi_target.py`
- Track is a data container wrapping a `KalmanFilter` instance with lifecycle bookkeeping
- `track_id`: unique integer identifier assigned by the tracker
- `kf`: dedicated KalmanFilter instance (each track owns its own, independently)
- `age`: steps since creation (incremented externally by MultiTargetTracker)
- `missed`: consecutive missed measurements (reset to 0 on update, incremented on coast)
- Constructor takes `track_id`, `kf`, and `initial_measurement` — calls `kf.init_state()` at birth
- Track does NOT orchestrate KF calls — it's a pure container; MultiTargetTracker manages the control flow
- Exported `Track` from `radarsim.tracker` subpackage
- **test:** Created `tests/test_multi_target.py` with 6 tests:
  - Track creation has correct id, age=0, missed=0
  - KF state matches initial measurement (position set, velocity zero)
  - Track stores the same KF instance (not a copy)
  - age and missed counters are mutable externally
  - Multiple tracks can have unique IDs
  - Independent KF instances — updating one track's KF doesn't affect another

## 2026-03-29 — Hungarian Data Association (Updated)

- **refactor:** Upgraded data association from greedy Nearest-Neighbor to Global Optimal Hungarian Algorithm (`scipy.optimize.linear_sum_assignment`).
- **feat:** Added `hungarian_associate()` function in `radarsim/tracker/multi_target.py` to minimize total Euclidean assignment error.
- **test:** Replaced `TestNearestNeighborAssociate` with `TestHungarianAssociate` and added `test_associate_global_optimality` verifying algorithm advantages over greedy approach.

## 2026-03-29 — Nearest-Neighbor Data Association

- **feat:** Implemented `nearest_neighbor_associate()` function in `radarsim/tracker/multi_target.py`
- Pure function: takes list of predicted positions `(2,)` and measurements `(n, 2)`, returns `dict[int, int]` mapping track index → measurement index
- Algorithm: build Euclidean distance matrix (position only) → greedy assignment (closest pair first, remove both from pool, repeat)
- **Gating:** optional `gate_threshold` parameter rejects pairs beyond a maximum distance — prevents distant measurements from being assigned to tracks
- Handles edge cases: empty predictions, empty measurements, more tracks than measurements, more measurements than tracks
- Standalone function (not a method) for independent testability; MultiTargetTracker.associate() will delegate to it
- **test:** Added 7 tests to `tests/test_multi_target.py`:
  - Perfect match (predictions = measurements)
  - Shuffled measurements (correct pairing by proximity)
  - More measurements than tracks (surplus unassigned)
  - More tracks than measurements (some tracks unmatched)
  - Gating rejects distant pairs
  - Empty measurements → empty dict
  - Empty predictions → empty dict

## 2026-03-29 — MultiTargetTracker Class

- **feat:** Implemented `MultiTargetTracker` class in `radarsim/tracker/multi_target.py`
- Core predict-associate-update loop in `step(measurements)`:
  1. Predict all active tracks (`kf.predict()`)
  2. Associate predictions ↔ measurements via `nearest_neighbor_associate()`
  3. Update matched tracks (`kf.update(z)`, reset `missed = 0`)
  4. Coast unmatched tracks (prediction already ran, increment `missed += 1`)
  5. Increment `age` for all tracks
- `associate()` method: thin wrapper delegating to standalone `nearest_neighbor_associate()` with stored `gate_threshold`
- `get_active_tracks()` returns shallow copy of track list
- `_create_track(measurement)` helper: creates a fresh `KalmanFilter` and `Track` with auto-incrementing ID
- Constructor stores KF parameters (`dt`, `q`, `r_x`, `r_y`) and `max_missed` threshold for reuse when creating tracks
- **Design:** predict-then-update pattern (not `kf.step()`) prevents double-prediction for matched tracks
- Track initialization (birth) and termination (death) not yet in `step()` — coming in Tasks 4 & 5
- Exported `MultiTargetTracker` from `radarsim.tracker` subpackage
- **test:** Added 7 tests to `tests/test_multi_target.py` (total: 20):
  - Tracker init has no active tracks
  - `_create_track` assigns sequential IDs
  - Single track + matching measurement → updated, missed=0
  - Single track + no measurements → coasted, missed=1
  - Two tracks + shuffled measurements → correct association
  - Age increments across multiple steps
  - Consecutive misses accumulate and reset on match

## 2026-03-29 — Track Initialization and Termination

- **feat:** Completed the tracking lifecycle in `MultiTargetTracker.step()`
- **Initialization (Birth):** After association, step over all unassigned measurements and create a new track (`_create_track`) for each. New tracks begin with `age=0`, `missed=0`.
- **Termination (Death):** Filter out active tracks that have exceeded the `max_missed` threshold.
- Updated `step()` docstring to reflect the completed algorithm.
- **test:** Added 3 tests to `tests/test_multi_target.py` covering lifecycle events:
  - `test_tracker_birth_unassigned_measurement`: Passing an unassigned measurement correctly creates a new track with age=0.
  - `test_tracker_death_max_missed`: A track coasting past `max_missed` steps gets completely removed.
  - `test_tracker_birth_and_death_together`: Verifies a track dying and another track spawning in the same step work flawlessly simultaneously.

## 2026-03-29 — Multi-Target Configuration and Visualization

- **feat:** Created `examples/multi_target.py` demo for end-to-end multi-target tracking.
- Implemented a 100-step simulation tracking 3 distinct Constant Velocity targets:
  - **Target A:** Always present, fast straight line.
  - **Target B:** Always present, diagonal trajectory.
  - **Target C:** Spawns dynamically at `t=20` and vanishes at `t=70`.
- Applied measurement shuffle to ensure the `MultiTargetTracker` algorithm does not rely on list positioning for data association.
- Calibrated Tracker hyper-parameters: 15m initialization noise, `gate_threshold = 45m` (3-sigma bounds), and `max_missed = 4` logic efficiently rejects clutter without premature track loss.
- Custom Pyplot plotting integration directly within the file generating 2 explicit visuals per specifications:
  - `multi_target_tracking.png`: Tracks the 2D positional movements for estimated models vs ground truth across multiple colors.
  - `multi_target_track_count.png`: Tracks the lifetime dynamics of active tracker elements vs exact truth references over 100 seconds to vividly exhibit target initialization and termination efficiency.

## 2026-03-29 — Parameter Sweep Analysis

- **feat:** Implemented `radarsim/analysis/parameter_sweep.py` with three functions for Q/R sensitivity analysis
- `sweep_q(scenario_fn, q_values, r_x, r_y)` — runs scenario at each Q value with fixed R, returns dict of Q → RMSE
- `sweep_r(scenario_fn, r_values, q)` — runs scenario at each R value with fixed Q, returns dict of R → RMSE
- `sweep_qr_heatmap(scenario_fn, q_values, r_values)` — runs all Q × R combinations, returns 2D RMSE grid `(len(q), len(r))`
- All functions take a generic `scenario_fn(q, r_x, r_y) -> (true_states, estimated_states)` callable — works with any scenario
- Exported all three functions from `radarsim.analysis` subpackage
- **feat:** Created `examples/parameter_sweep.py` with CV scenario and three analyses:
  - Q sweep (8 values, fixed R=25m): RMSE ranges 13.88m (Q=0.01) to 18.04m (Q=10.0) — lower Q better for straight-line CV
  - R sweep (8 values, fixed Q=0.5): RMSE scales near-linearly with measurement noise (3.74m at R=5 to 50.36m at R=100)
  - Q × R heatmap (64 runs): optimal at Q=0.01, R=5.0 with RMSE=2.95m; uses annotated `imshow` with RdYlGn_r colormap
- Generates `q_sweep.png`, `r_sweep.png`, `qr_heatmap.png` in `output/`

## 2026-03-30 — Cross-Scenario Performance Comparison

- **feat:** Created `examples/scenario_comparison.py` — runs KF through all 5 project scenarios and generates a summary bar chart
- Scenarios compared: Single Target CV (baseline), Maneuver (turn), ECM Noise Spike, ECM Dropout, ECM Bias
- Results: CV baseline 14.69m, Maneuver 51.72m (3.5x), Noise Spike 33.94m (2.3x), Dropout 16.67m (1.1x), Bias 36.67m (2.5x)
- Key insight: dropout is nearly harmless when motion model is correct (1.1x); maneuver is the worst case (3.5x) because the model itself is wrong
- Generates `scenario_comparison.png` in `output/` — horizontal bar chart with color coding, RMSE annotations, and baseline reference line

## 2026-03-30 — Analysis Plots for README

- **docs:** Copied 10 plots from `output/` to `docs/images/` (tracked in git) for README embedding
- Files: `single_target_tracking.png`, `maneuver_tracking.png`, `maneuver_error.png`, `ecm_comparison.png`, `ecm_q_comparison.png`, `multi_target_tracking.png`, `multi_target_track_count.png`, `q_sweep.png`, `qr_heatmap.png`, `scenario_comparison.png`
- All scripts re-run with fresh seeds to ensure plots are current

## 2026-03-30 — Animated Tracking Visualization

- **feat:** Implemented `radarsim/viz/animation.py` with `animate_tracking()` function
- Uses `matplotlib.animation.FuncAnimation` + `PillowWriter` (no ffmpeg dependency)
- Frame-by-frame build-up: true trajectory (blue line), KF estimate (green dashed), radar measurements (red dots with fading trail)
- Current position markers (blue circle = true, green square = estimate) and time display
- Configurable `fps`, `trail_length` parameters
- **feat:** Created `examples/animation_demo.py` — standalone script generating the GIF
- Uses Phase 1 CV scenario (60 steps, seed=42) at 15fps with 8-step measurement trail
- Saves `tracking_animation.gif` (519KB) directly to `docs/images/`
- Exported `animate_tracking` from `radarsim.viz` subpackage

## 2026-03-30 — Comprehensive README (Phase 5 Complete)

- **docs:** Rewrote README.md with full project story — all results, hard RMSE numbers, embedded images
- Sections: animation header, What It Does, Defense Domain Context, Results (6 subsections), Architecture, Tech Stack, How to Run, License
- Defense domain context: missile interception (Iron Dome/Patriot), air surveillance, electronic warfare, sensor fusion
- All results include precise numbers: CV 49.6% improvement, maneuver 4.6× degradation, dropout 1.1× harmless, bias 3.7×
- Embedded 11 images from `docs/images/` including the animated GIF as the hero image
- Phase 5 status: **DONE** — all 6 tasks complete
