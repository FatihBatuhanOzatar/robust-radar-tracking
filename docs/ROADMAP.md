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
**Status:** TODO

| Task | Status | Commit message |
|------|--------|---------------|
| Add coordinated turn model to Target class | DONE | `feat: add coordinated turn target model` |
| Add random maneuver model to Target class | DONE | `feat: add random maneuver target model` |
| Create maneuver scenario showing CV KF degradation | TODO | `feat: add maneuver scenario with KF breakdown analysis` |
| Add error-vs-time plot during maneuver | TODO | `feat: add maneuver error analysis plots` |
| Optional: constant acceleration model comparison | TODO | `feat: add CA model comparison` |
| Create examples/maneuver.py demo | TODO | `feat: add maneuver tracking example` |

**Exit criteria:** Clear graph showing KF error spike during maneuver vs straight-line tracking.

---

## Phase 3 — ECM / Measurement Degradation
**Goal:** Analyze filter behavior under jamming and signal loss.
**Status:** BLOCKED (waiting for Phase 2)

| Task | Status | Commit message |
|------|--------|---------------|
| Implement ECMModel class (noise spike, dropout, bias) | TODO | `feat: add ECM simulation model` |
| Implement predict-only mode in KF (no measurement) | TODO | `feat: add predict-only mode for measurement dropout` |
| Create ECM scenario with noise spike | TODO | `feat: add noise spike ECM scenario` |
| Create ECM scenario with measurement dropout | TODO | `feat: add measurement dropout scenario` |
| Analyze recovery time after ECM ends | TODO | `feat: add ECM recovery analysis` |
| Analyze Q parameter effect on ECM resilience | TODO | `feat: add Q-tuning ECM resilience analysis` |
| Create examples/ecm_scenario.py demo | TODO | `feat: add ECM scenario example` |

**Exit criteria:** Graphs showing filter degradation during ECM, recovery after ECM ends, and Q parameter impact.

---

## Phase 4 — Multi-Target Tracking
**Goal:** Track multiple targets simultaneously with data association.
**Status:** BLOCKED (waiting for Phase 3)

| Task | Status | Commit message |
|------|--------|---------------|
| Implement Track class | TODO | `feat: add Track class for target management` |
| Implement nearest-neighbor data association | TODO | `feat: add nearest neighbor data association` |
| Implement MultiTargetTracker class | TODO | `feat: add multi-target tracker` |
| Add track initialization logic | TODO | `feat: add track initialization from new measurements` |
| Add track termination logic | TODO | `feat: add track termination for lost targets` |
| Create multi-target scenario (3+ targets) | TODO | `feat: add multi-target tracking scenario` |
| Add multi-target visualization | TODO | `feat: add multi-target tracking visualization` |
| Create examples/multi_target.py demo | TODO | `feat: add multi-target tracking example` |

**Exit criteria:** 3+ targets tracked simultaneously with correct association, track init/termination working.

---

## Phase 5 — Analysis & Visualization
**Goal:** Make the project convincing with comprehensive analysis and polished visuals.
**Status:** BLOCKED (waiting for Phase 4)

| Task | Status | Commit message |
|------|--------|---------------|
| Implement Q/R parameter sweep | TODO | `feat: add parameter sweep analysis` |
| Generate RMSE heatmap (Q vs R) | TODO | `feat: add Q/R sweep heatmap` |
| Create KF performance comparison across scenarios | TODO | `feat: add cross-scenario performance comparison` |
| Implement matplotlib animation | TODO | `feat: add animated tracking visualization` |
| Write comprehensive README with analysis results | TODO | `docs: write comprehensive README with results` |
| Add all generated plots to docs/images/ | TODO | `docs: add analysis plots and figures` |

**Exit criteria:** README tells a complete story with graphs. Anyone reading it understands the project without running code.

---

## Phase 6 — Advanced Extensions (Optional)
**Goal:** Additional features if time permits.
**Status:** BLOCKED (waiting for Phase 5)

Possible extensions (pick based on time):
- Extended Kalman Filter (nonlinear motion model)
- Constant acceleration model
- C++ port of core Kalman engine
- Real-time scheduler integration (threat-based prioritization)
- Threat scoring system
