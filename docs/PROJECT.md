# robust-radar-tracking

## Project Overview

**Name:** robust-radar-tracking
**One-liner:** 2D radar target tracking project that starts with single-target Kalman filtering and grows toward robust multi-target tracking under maneuver and measurement degradation.

## What This Project Does

Simulates a radar tracking scenario where:
1. Multiple targets move in a 2D plane (straight line, maneuvering, random)
2. A radar sensor produces noisy position measurements
3. Kalman filters estimate the true target positions from noisy data
4. The system is tested under adversarial conditions (jamming, signal loss, sudden maneuvers)

## What Problem It Solves

In defense systems (missile tracking, air surveillance, autonomous navigation), radar measurements are always noisy and sometimes degraded. This project demonstrates how Kalman filtering extracts accurate target state (position + velocity) from unreliable sensor data, and analyzes system behavior when conditions deteriorate.

## Tech Stack

- **Language:** Python 3.10+
- **Core libraries:** NumPy (matrix operations), Matplotlib (visualization)
- **Optional future:** C++ port for core Kalman engine
- **No ML frameworks, no heavy dependencies**

## Scope

### In Scope
- 2D target simulation (constant velocity, coordinated turn, random maneuver)
- Radar measurement simulation (Gaussian noise, configurable)
- Standard Kalman Filter (constant velocity model)
- Multi-target tracking with data association (nearest neighbor)
- ECM simulation (noise spikes, measurement dropout, bias injection)
- Performance analysis (RMSE, parameter sweeps, robustness metrics)
- Visualization (static plots + animation)

### Out of Scope (for now)
- 3D tracking (architecture supports it but not implemented)
- Extended Kalman Filter (optional future extension)
- Real sensor data ingestion
- Real-time scheduling integration
- GUI / web interface

## Target Audience

Defense industry internship recruiters and technical interviewers. This project demonstrates:
- Domain knowledge (radar, tracking, electronic warfare basics)
- Algorithm understanding (Kalman filter from scratch, not library calls)
- System thinking (modular architecture, multiple interacting components)
- Engineering rigor (parameter analysis, robustness testing, clean code)

## Success Criteria

- Single target tracking with <50% RMSE improvement over raw radar
- Visible performance degradation during maneuvers (KF limitation demonstrated)
- ECM resilience analysis with recovery time metrics
- Multi-target tracking with correct data association
- Clean, well-documented codebase with meaningful commit history
- Comprehensive README with analysis results and graphs
