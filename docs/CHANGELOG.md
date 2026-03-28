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
