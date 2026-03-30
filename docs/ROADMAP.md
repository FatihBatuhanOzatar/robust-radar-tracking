# Roadmap

## Phase 1 — Core Tracker
**Goal:** Single target, 2D, constant velocity KF, basic visualization, RMSE.
**Status:** DONE

| Task | Status | Commit message |
|------|--------|---------------|
| Create project structure, requirements.txt, .gitignore | DONE | `init: create project structure and requirements` |
| Implement Target class with constant velocity model | DONE | `feat: add linear target simulation` |
| Implement Radar class with Gaussian noise | DONE | `feat: add noisy radar measurement generation` |
| Implement KalmanFilter class (CV model) | DONE | `feat: implement constant velocity Kalman filter` |
| Implement metrics.py (RMSE, position error) | DONE | `feat: add RMSE and error metrics` |
| Implement plots.py (tracking result plot, error plot) | DONE | `feat: add tracking result visualization` |
| Create examples/single_target.py demo | DONE | `feat: add single target tracking example` |
| Add unit tests for KF (predict, update, convergence) | DONE | `test: add Kalman filter unit tests` |

**Exit criteria:** `python examples/single_target.py` produces tracking plot with visible RMSE improvement over raw radar.

---

## Phase 2 — Maneuver / Model Breakdown
**Goal:** Show where CV model breaks during target maneuvers.
**Status:** DONE

| Task | Status | Commit message |
|------|--------|---------------|
| Add coordinated turn model to Target class | DONE | `feat: add coordinated turn target model` |
| Add random maneuver model to Target class | DONE | `feat: add random maneuver target model` |
| Create maneuver scenario showing CV KF degradation | DONE | `feat: add maneuver scenario with KF breakdown analysis` |
| Add error-vs-time plot during maneuver | DONE | `feat: add maneuver error analysis plots` |
| Optional: constant acceleration model comparison | SKIPPED | — |
| Create examples/maneuver.py demo | DONE | `feat: add maneuver tracking example` |

**Exit criteria:** Clear graph showing KF error spike during maneuver vs straight-line tracking.

---

## Phase 3 — ECM / Measurement Degradation
**Goal:** Analyze filter behavior under jamming and signal loss.
**Status:** DONE

| Task | Status | Commit message |
|------|--------|---------------|
| Implement ECMModel class (noise spike, dropout, bias) | DONE | `feat: add ECM simulation model` |
| Implement predict-only mode in KF (no measurement) | DONE | (already implemented in Phase 1 as `step_no_measurement()`) |
| Create ECM scenario with noise spike | DONE | `feat: add ECM scenario with noise spike, dropout, bias analysis` |
| Create ECM scenario with measurement dropout | DONE | (combined into ecm_scenario.py) |
| Analyze recovery time after ECM ends | DONE | (combined into ecm_scenario.py) |
| Analyze Q parameter effect on ECM resilience | DONE | (covered by Q comparison in ecm_scenario.py) |
| Create examples/ecm_scenario.py demo | DONE | (combined into ecm_scenario.py) |

**Exit criteria:** Graphs showing filter degradation during ECM, recovery after ECM ends, and Q parameter impact.

---

## Phase 4 — Multi-Target Tracking
**Goal:** Track multiple targets simultaneously with data association.
**Status:** DONE

| Task | Status | Commit message |
|------|--------|---------------|
| Implement Track class | DONE | `feat: add Track class for target management` |
| Implement nearest-neighbor data association | DONE | `feat: add nearest neighbor data association` |
| Implement MultiTargetTracker class | DONE | `feat: add multi-target tracker` |
| Add track initialization logic | DONE | `feat: add track initialization and termination logic` |
| Add track termination logic | DONE | `feat: add track initialization and termination logic` |
| Create multi-target scenario (3+ targets) | DONE | `feat: add multi-target tracking scenario and visualization` |
| Add multi-target visualization | DONE | `feat: add multi-target tracking scenario and visualization` |
| Create examples/multi_target.py demo | DONE | `feat: add multi-target tracking scenario and visualization` |

**Exit criteria:** 3+ targets tracked simultaneously with correct association, track init/termination working.

---

## Phase 5 — Analysis & Visualization
**Goal:** Make the project convincing with comprehensive analysis and polished visuals.
**Status:** DONE

| Task | Status | Commit message |
|------|--------|---------------|
| Implement Q/R parameter sweep | DONE | `feat: add parameter sweep analysis` |
| Generate RMSE heatmap (Q vs R) | DONE | `feat: add parameter sweep analysis` |
| Create KF performance comparison across scenarios | DONE | `feat: add cross-scenario performance comparison` |
| Implement matplotlib animation | DONE | `feat: add animated tracking visualization` |
| Write comprehensive README with analysis results | DONE | `docs: write comprehensive README with results` |
| Add all generated plots to docs/images/ | DONE | `docs: add analysis plots and figures` |

**Exit criteria:** README tells a complete story with graphs. Anyone reading it understands the project without running code.

---

## Phase 6 — Advanced Extensions (Optional)
**Goal:** Additional features if time permits.
**Status:** IN PROGRESS

| Task | Status | Commit message |
|------|--------|---------------|
| Implement ExtendedKalmanFilter class (CT model) | DONE | `feat: add Extended Kalman Filter for coordinated turn model` |
| Create KF vs EKF maneuver comparison | DONE | `feat: add KF vs EKF maneuver comparison` |
| Add EKF section to README | DONE | `docs: add EKF comparison results to README` |

**Status: DONE**

Other possible extensions (not started):
- Constant acceleration model
- C++ port of core Kalman engine
- Real-time scheduler integration (threat-based prioritization)
- Threat scoring system
