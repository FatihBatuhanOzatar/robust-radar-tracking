/**
 * @file main.cpp
 * @brief CLI demo and benchmark for the C++ Kalman Filter.
 *
 * Replicates the Python single_target.py scenario:
 * - Constant velocity target: start (0,0), velocity (15, 10) m/s
 * - Radar noise: 25m std dev in both axes
 * - KF parameters: q=0.5, dt=1.0
 *
 * Prints RMSE comparison (raw radar vs KF) and benchmark timing.
 */

#include "kalman.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>

// ---- Simulation parameters (matching Python single_target.py) ----
static constexpr int    N_STEPS   = 100;
static constexpr double DT        = 1.0;     // seconds
static constexpr double VX0       = 15.0;    // m/s
static constexpr double VY0       = 10.0;    // m/s
static constexpr double NOISE_STD = 25.0;    // meters
static constexpr double Q_VAR     = 0.5;     // process noise intensity
static constexpr int    SEED      = 42;

// Number of benchmark iterations
static constexpr int BENCH_RUNS = 10000;

/**
 * Run a single KF tracking scenario.
 *
 * @param seed        RNG seed for reproducible noise.
 * @param radar_rmse  Output: RMSE of raw radar measurements.
 * @param kf_rmse     Output: RMSE of Kalman Filter estimates.
 */
static void run_scenario(int seed, double& radar_rmse, double& kf_rmse) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, NOISE_STD);

    // Simulate constant-velocity target + noisy measurements
    double true_x[N_STEPS], true_y[N_STEPS];
    double meas_x[N_STEPS], meas_y[N_STEPS];

    for (int t = 0; t < N_STEPS; ++t) {
        true_x[t] = VX0 * DT * t;       // x = vx * t
        true_y[t] = VY0 * DT * t;       // y = vy * t
        meas_x[t] = true_x[t] + noise(rng);
        meas_y[t] = true_y[t] + noise(rng);
    }

    // Run Kalman Filter
    KalmanFilter kf(DT, Q_VAR, NOISE_STD, NOISE_STD);
    kf.init_state(meas_x[0], meas_y[0]);

    double radar_sse = 0.0;
    double kf_sse    = 0.0;

    // Step 0: already initialized — compute errors for first step
    {
        double dx_r = meas_x[0] - true_x[0];
        double dy_r = meas_y[0] - true_y[0];
        radar_sse += dx_r * dx_r + dy_r * dy_r;
        kf_sse += kf.get_position_error_sq(true_x[0], true_y[0]);
    }

    // Steps 1..N-1: predict + update
    for (int t = 1; t < N_STEPS; ++t) {
        kf.step(meas_x[t], meas_y[t]);

        double dx_r = meas_x[t] - true_x[t];
        double dy_r = meas_y[t] - true_y[t];
        radar_sse += dx_r * dx_r + dy_r * dy_r;
        kf_sse += kf.get_position_error_sq(true_x[t], true_y[t]);
    }

    radar_rmse = std::sqrt(radar_sse / N_STEPS);
    kf_rmse    = std::sqrt(kf_sse / N_STEPS);
}

int main() {
    // ---- Single run with results ----
    double radar_rmse = 0.0, kf_rmse = 0.0;
    run_scenario(SEED, radar_rmse, kf_rmse);

    double improvement = ((radar_rmse - kf_rmse) / radar_rmse) * 100.0;

    printf("=== C++ Kalman Filter Benchmark ===\n");
    printf("Steps: %d\n", N_STEPS);
    printf("Raw Radar RMSE:   %6.2f m\n", radar_rmse);
    printf("KF Estimate RMSE: %6.2f m\n", kf_rmse);
    printf("Improvement: %.1f%%\n", improvement);

    // ---- Benchmark: time a single 100-step run ----
    {
        auto start = std::chrono::high_resolution_clock::now();
        double r, k;
        run_scenario(SEED, r, k);
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        printf("\nTiming (%d steps): %.2f ms\n", N_STEPS, ms);
    }

    // ---- Benchmark: 10000 runs ----
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < BENCH_RUNS; ++i) {
            double r, k;
            run_scenario(SEED, r, k);
        }
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        printf("Timing (%d x %d steps): %.0f ms\n", BENCH_RUNS, N_STEPS, ms);
    }

    return 0;
}
