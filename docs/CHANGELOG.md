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
