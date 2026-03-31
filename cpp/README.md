# C++ Kalman Filter Port

This directory contains a pure C++ port of the core constant-velocity 2D Kalman Filter engine from `radarsim/tracker/kf.py`.

The purpose of this port is to demonstrate a low-level understanding of the linear algebra involved, without relying on external libraries like Eigen.

## Architecture

- **`matrix.h` / `matrix.cpp`**: Fixed-size matrix operations (4x4, 2x4, 4x2, 2x2). Uses straightforward triple-nested loops entirely on the stack. Includes a closed-form 2x2 matrix inversion for the S matrix. No heap allocation, no templates.
- **`kalman.h` / `kalman.cpp`**: Direct 1:1 port of the Python `KalmanFilter` class. Uses Bar-Shalom physically-derived process noise covariance (Q) and Joseph-form covariance update for numerical stability.
- **`main.cpp`**: CLI demo that replicates the Python `single_target.py` scenario. Runs tracking for 100 steps and benchmarks the performance against 10,000 iterations.
- **`Makefile`**: Simple Makefile with optimized (`-O2`) and debug (`-g -fsanitize=address,undefined`) targets.

## Benchmark Results

```text
=== C++ Kalman Filter Benchmark ===
Steps: 100
Raw Radar RMSE:    34.28 m
KF Estimate RMSE:  15.18 m
Improvement: 55.7%

Timing (100 steps): 0.03 ms
Timing (10000 x 100 steps): 263 ms
```

*Note: The C++ version is drastically faster than the Python version due to zero-allocation stack arrays and compiled ahead-of-time code.*

## How to Build and Run

If you are using a Linux environment (or WSL):

```bash
# Build the optimized executable
make

# Run the benchmark
./kalman
```

If you don't have `make`, you can compile it directly with `g++` or `clang++`:

```bash
g++ -O2 -Wall -Wextra -std=c++17 -o kalman main.cpp kalman.cpp matrix.cpp
./kalman
```
