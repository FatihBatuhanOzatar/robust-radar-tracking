"""Phase 1 Demo: Single target constant-velocity tracking.

This script demonstrates tracking a single target moving in a straight line
using a standard constant-velocity Kalman Filter.
"""

import os
import sys
from pathlib import Path

# Add project root to path so we can import radarsim
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from radarsim.sim.target import Target
from radarsim.sim.radar import Radar
from radarsim.tracker.kf import KalmanFilter
from radarsim.analysis.metrics import rmse, position_error_over_time
from radarsim.viz.plots import plot_tracking_result, plot_error_over_time


def main():
    # 1. Configuration and Setup
    np.random.seed(42)  # constraint: fixed random seed for reproducibility
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)  # constraint: create output dir
    
    dt = 1.0         # 1 second update rate
    n_steps = 60     # 60 seconds
    
    noise_std = 25.0 # 25 meters measurement noise standard deviation
    q_var = 0.5      # Process noise variance parameter

    print("--- Single Target Tracking Demo ---")
    print(f"Setting up simulation ({n_steps} steps, dt={dt}s)...")

    # 2. Simulate Target Motion
    # Starts at (0, 0), moving at 15 m/s in X and 10 m/s in Y direction
    target = Target(x0=0.0, y0=0.0, vx0=15.0, vy0=10.0, model="cv")
    true_trajectory = target.get_trajectory(dt, n_steps)

    # 3. Simulate Radar Measurements
    radar = Radar(noise_std_x=noise_std, noise_std_y=noise_std)
    measurements = radar.measure_batch(true_trajectory)

    # 4. Kalman Filtering Tracking
    print("Running Kalman Filter tracking...")
    kf = KalmanFilter(dt=dt, q=q_var, r_x=noise_std, r_y=noise_std)
    
    estimated_trajectory = np.zeros((n_steps, 4))
    
    # Initialize from the very first measurement
    kf.init_state(measurements[0])
    estimated_trajectory[0] = kf.get_state()
    
    # Process sequentially
    for t in range(1, n_steps):
        estimated_trajectory[t] = kf.step(measurements[t])

    # 5. Analysis & Metrics
    print("Computing metrics...")
    # Calculate RMSE for raw radar measurements vs true position
    # Pad measurements with 0 velocity to reuse the rmse function which expects (n, 4)
    padded_measurements = np.zeros((n_steps, 4))
    padded_measurements[:, :2] = measurements
    
    radar_rmse = rmse(true_trajectory, padded_measurements)
    kf_rmse = rmse(true_trajectory, estimated_trajectory)
    improvement = ((radar_rmse - kf_rmse) / radar_rmse) * 100

    print("\n--- Results ---")
    print(f"Raw Radar RMSE: {radar_rmse:.2f} m")
    print(f"KF Estimate RMSE: {kf_rmse:.2f} m")
    print(f"Improvement: {improvement:.1f}%")

    # 6. Visualization
    print("\nGenerating plots...")
    
    fig_tracking = plot_tracking_result(
        true=true_trajectory, 
        measured=measurements, 
        estimated=estimated_trajectory, 
        title="Single Target CV Tracking"
    )
    tracking_path = output_dir / "single_target_tracking.png"
    fig_tracking.savefig(tracking_path, dpi=150)
    print(f"Saved: {tracking_path.relative_to(project_root)}")
    
    errors = position_error_over_time(true_trajectory, estimated_trajectory)
    fig_error = plot_error_over_time(
        errors=errors, 
        title="KF Position Error Over Time"
    )
    error_path = output_dir / "single_target_error.png"
    fig_error.savefig(error_path, dpi=150)
    print(f"Saved: {error_path.relative_to(project_root)}")
    
    print("Done!")


if __name__ == "__main__":
    main()
