# robust-radar-tracking

2D radar target tracking system built from scratch with Kalman filtering. Starts with single-target constant velocity tracking and grows toward robust multi-target tracking under maneuver and measurement degradation scenarios.

## Why This Project?

In defense systems — missile interception, air surveillance, autonomous navigation — radar measurements are always noisy and sometimes deliberately degraded through electronic countermeasures (jamming). This project demonstrates how Kalman filtering extracts accurate target state from unreliable sensor data, and analyzes what happens when conditions deteriorate.

The core question: **how do you track something you can barely see?**

## What It Does

- **Simulates radar tracking scenarios** — targets move in 2D, a radar produces noisy position measurements, Kalman filter estimates the true state
- **Tests robustness** — what happens during target maneuvers, jamming, signal dropout?
- **Quantifies performance** — RMSE analysis, parameter sensitivity, recovery metrics
- **Tracks multiple targets** — data association, track initialization and termination

## Project Status

🔧 **Phase 1 — Core Tracker** (in progress)

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Single target, constant velocity KF, RMSE | 🔧 In Progress |
| 2 | Maneuver scenarios, model breakdown analysis | ⏳ Planned |
| 3 | ECM — jamming, dropout, bias, recovery analysis | ⏳ Planned |
| 4 | Multi-target tracking with data association | ⏳ Planned |
| 5 | Comprehensive analysis, visualization, documentation | ⏳ Planned |
| 6 | Optional extensions (EKF, C++ port, scheduler) | ⏳ Planned |

## Architecture

```
radarsim/
├── sim/           → Target motion models + radar sensor simulation
├── tracker/       → Kalman filter implementations
├── analysis/      → RMSE, parameter sweeps, robustness metrics
└── viz/           → Static plots and animations
```

Modular design — each component is independently testable. The tracker doesn't know about the simulator, the analyzer doesn't know about the tracker internals.

## Tech Stack

Python 3.10+, NumPy, Matplotlib. No ML frameworks, no heavy dependencies. Kalman filter implemented from scratch — no library calls.


## Defense Domain Context

The Kalman filter is foundational in defense systems:

- **Missile interception** — Iron Dome, Patriot, HISAR systems use Kalman-based tracking to predict intercept points from noisy radar data
- **Air surveillance** — tracking aircraft across radar sweeps with measurement gaps
- **Electronic warfare** — maintaining track when the enemy jams your radar
- **Sensor fusion** — combining radar, IR, and other sensors (each noisy) into one estimate

This project simulates these scenarios at a fundamental level. The algorithms are the same — the scale and sensor data differ.
## License

MIT
