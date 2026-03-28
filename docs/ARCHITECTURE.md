# Architecture

## Directory Structure

```
robust-radar-tracking/
├── radarsim/                      # Main Python package
│   ├── __init__.py
│   ├── sim/                       # Simulation engine
│   │   ├── __init__.py
│   │   ├── target.py              # Target motion models
│   │   ├── radar.py               # Radar measurement simulator
│   │   └── ecm.py                 # Electronic countermeasure models (Phase 3)
│   ├── tracker/                   # Tracking algorithms
│   │   ├── __init__.py
│   │   ├── kf.py                  # Standard Kalman Filter
│   │   └── multi_target.py        # Multi-target tracker (Phase 4)
│   ├── analysis/                  # Analysis tools
│   │   ├── __init__.py
│   │   ├── metrics.py             # RMSE, error calculations
│   │   └── parameter_sweep.py     # Q/R parameter analysis (Phase 5)
│   └── viz/                       # Visualization
│       ├── __init__.py
│       ├── plots.py               # Static plots
│       └── animation.py           # Animated tracking display (Phase 5)
├── examples/                      # Runnable demo scripts
│   ├── single_target.py           # Phase 1 demo
│   ├── maneuver.py                # Phase 2 demo
│   ├── ecm_scenario.py            # Phase 3 demo
│   └── multi_target.py            # Phase 4 demo
├── tests/                         # Unit tests
│   ├── test_kf.py
│   ├── test_target.py
│   └── test_radar.py
├── docs/                          # Project documentation
│   ├── PROJECT.md
│   ├── ARCHITECTURE.md
│   ├── ROADMAP.md
│   ├── CONVENTIONS.md
│   └── CHANGELOG.md
├── output/                        # Generated plots and results (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

## Module Responsibilities

### radarsim/sim/target.py
Generates true target trajectories. No noise, no estimation — just ground truth.

**Classes:**
- `Target`: Represents a single moving target
  - `__init__(self, x0, y0, vx0, vy0, model="cv")` — initial state and motion model
  - `step(self, dt) -> np.ndarray` — advance one time step, return true state [x, y, vx, vy]
  - `get_trajectory(self, dt, n_steps) -> np.ndarray` — generate full trajectory, shape (n_steps, 4)

**Motion models (string parameter):**
- `"cv"` — constant velocity (Phase 1)
- `"ct"` — coordinated turn with configurable turn rate (Phase 2)
- `"random"` — random acceleration perturbations (Phase 2)

### radarsim/sim/radar.py
Simulates noisy radar measurements from true target positions.

**Classes:**
- `Radar`: Radar sensor model
  - `__init__(self, noise_std_x, noise_std_y)` — measurement noise standard deviations
  - `measure(self, true_state) -> np.ndarray` — return noisy [x, y] measurement
  - `measure_batch(self, true_states) -> np.ndarray` — batch measurement for full trajectory

### radarsim/sim/ecm.py (Phase 3)
Models electronic countermeasures that degrade radar performance.

**Classes:**
- `ECMModel`: Configurable ECM effects
  - `__init__(self, noise_multiplier, dropout_prob, bias)` — ECM parameters
  - `apply(self, measurement, t) -> tuple[np.ndarray | None, bool]` — apply ECM to measurement, return (degraded_measurement_or_None, is_valid)

**ECM modes:**
- Noise spike: multiply radar noise by a factor during ECM window
- Dropout: measurement completely lost (returns None)
- Bias: systematic offset added to measurement

### radarsim/tracker/kf.py
Standard Kalman Filter implementation. State: [x, y, vx, vy].

**Classes:**
- `KalmanFilter`:
  - `__init__(self, dt, q, r_x, r_y)` — time step, process noise intensity, measurement noise
  - `init_state(self, z) -> None` — initialize state from first measurement
  - `predict(self) -> np.ndarray` — prediction step, return predicted state
  - `update(self, z) -> np.ndarray` — update step with measurement, return updated state
  - `step(self, z) -> np.ndarray` — predict + update combined, return estimated state
  - `step_no_measurement(self) -> np.ndarray` — predict only (for ECM dropout), return predicted state
  - `get_state(self) -> np.ndarray` — current state estimate
  - `get_covariance(self) -> np.ndarray` — current P matrix

**Internal attributes:**
- `x`: state vector (4,)
- `P`: covariance matrix (4,4)
- `F`: state transition matrix (4,4)
- `H`: measurement matrix (2,4)
- `Q`: process noise covariance (4,4)
- `R`: measurement noise covariance (2,2)

**Q matrix construction:** Uses physically-derived process noise from acceleration uncertainty:
```
q * [[dt⁴/4,  0,      dt³/2,  0     ],
     [0,      dt⁴/4,  0,      dt³/2 ],
     [dt³/2,  0,      dt²,    0     ],
     [0,      dt³/2,  0,      dt²   ]]
```

### radarsim/tracker/multi_target.py (Phase 4)
Manages multiple KalmanFilter instances with data association.

**Classes:**
- `Track`: Single target track
  - `kf`: KalmanFilter instance
  - `id`: unique track ID
  - `age`: number of steps since creation
  - `missed`: consecutive missed measurements
  
- `MultiTargetTracker`:
  - `__init__(self, dt, q, r_x, r_y, max_missed)` — parameters + track termination threshold
  - `step(self, measurements) -> list[Track]` — process measurements, return active tracks
  - `associate(self, predictions, measurements) -> dict` — nearest neighbor data association
  - `get_active_tracks(self) -> list[Track]` — return tracks that are still alive

### radarsim/analysis/metrics.py
Performance measurement functions.

**Functions:**
- `rmse(true_states, estimated_states) -> float` — root mean square error on position
- `position_error_over_time(true_states, estimated_states) -> np.ndarray` — per-step error
- `velocity_error_over_time(true_states, estimated_states) -> np.ndarray` — per-step velocity error

### radarsim/analysis/parameter_sweep.py (Phase 5)
Runs experiments varying Q and R parameters.

**Functions:**
- `sweep_q(scenario, q_values) -> dict` — run same scenario with different Q, return RMSE for each
- `sweep_r(scenario, r_values) -> dict` — same for R
- `sweep_qr_heatmap(scenario, q_values, r_values) -> np.ndarray` — 2D RMSE heatmap

### radarsim/viz/plots.py
Static matplotlib plots.

**Functions:**
- `plot_tracking_result(true, measured, estimated, title) -> fig` — main tracking plot
- `plot_error_over_time(errors, title) -> fig` — error timeline
- `plot_covariance_over_time(covariances, title) -> fig` — uncertainty timeline
- `plot_parameter_sweep(sweep_results, title) -> fig` — parameter analysis
- `plot_ecm_scenario(true, measured, estimated, ecm_windows, title) -> fig` — ECM visualization

### radarsim/viz/animation.py (Phase 5)
Animated matplotlib visualization.

**Functions:**
- `animate_tracking(true, measured, estimated, dt, save_path) -> None` — animated GIF/MP4

## Data Flow

```
Target.step()  →  true_state  →  Radar.measure()  →  noisy_measurement
                                         ↓
                                  ECM.apply() (optional)
                                         ↓
                              KalmanFilter.step()  →  estimated_state
                                         ↓
                              metrics.rmse()  →  performance numbers
                              plots.plot_tracking_result()  →  figures
```

## Design Principles

1. **Each module is independently testable** — Target doesn't know about Radar, Radar doesn't know about KF
2. **No global state** — all parameters passed explicitly
3. **NumPy arrays as interface** — state is always np.ndarray shape (4,) or (4,1)
4. **Consistent state format** — [x, y, vx, vy] everywhere, always flat `(4,)` shape
5. **Scheduler-ready** — KalmanFilter.step() is a pure function of (current_state, measurement), no side effects beyond updating internal state
