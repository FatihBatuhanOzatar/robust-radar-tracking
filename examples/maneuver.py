"""Phase 2 Demo: CV Kalman Filter breakdown during coordinated turn.

This script demonstrates how a constant-velocity Kalman Filter fails
when the target maneuvers. The scenario has three phases:

  Phase A (steps 0-29):  Straight-line flight — KF tracks well
  Phase B (steps 30-69): Coordinated turn     — KF error spikes
  Phase C (steps 70-99): Straight-line flight  — KF recovers

The key insight: the CV model assumes the target always moves in a
straight line. During a turn, the prediction is systematically wrong,
causing the filter to lag behind the true trajectory.
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


def build_maneuver_trajectory(dt: float) -> np.ndarray:
    """Build a 3-phase trajectory: straight → turn → straight.

    Stitches together three Target instances to create a multi-phase
    ground truth trajectory.

    Args:
        dt: Time step duration (seconds).

    Returns:
        True trajectory array, shape (100, 4).
    """
    n_straight_a = 30
    n_turn = 40
    n_straight_c = 30

    # Phase A: straight-line flight
    target_a = Target(x0=0.0, y0=0.0, vx0=50.0, vy0=20.0, model="cv")
    traj_a = np.zeros((n_straight_a, 4))
    for i in range(n_straight_a):
        traj_a[i] = target_a.step(dt)

    # Phase B: coordinated turn — pick up from Phase A's end state
    end_a = target_a.state
    target_b = Target(
        x0=end_a[0], y0=end_a[1], vx0=end_a[2], vy0=end_a[3],
        model="ct", turn_rate=0.05,
    )
    traj_b = np.zeros((n_turn, 4))
    for i in range(n_turn):
        traj_b[i] = target_b.step(dt)

    # Phase C: straight-line again — pick up from Phase B's end state
    end_b = target_b.state
    target_c = Target(
        x0=end_b[0], y0=end_b[1], vx0=end_b[2], vy0=end_b[3],
        model="cv",
    )
    traj_c = np.zeros((n_straight_c, 4))
    for i in range(n_straight_c):
        traj_c[i] = target_c.step(dt)

    return np.vstack([traj_a, traj_b, traj_c])


def main():
    # --- Configuration ---
    np.random.seed(42)
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    dt = 1.0
    noise_std = 25.0
    q_var = 0.5

    # Phase boundaries (step indices)
    maneuver_start = 30
    maneuver_end = 70
    n_steps = 100

    print("--- Maneuver Scenario: CV KF Breakdown Demo ---")
    print(f"Scenario: {n_steps} steps, dt={dt}s")
    print(f"  Phase A (steps  0-{maneuver_start - 1}): Straight line")
    print(f"  Phase B (steps {maneuver_start}-{maneuver_end - 1}): "
          f"Coordinated turn (omega=0.05 rad/s)")
    print(f"  Phase C (steps {maneuver_end}-{n_steps - 1}): Straight line")

    # --- Build trajectory ---
    print("\nBuilding trajectory...")
    true_trajectory = build_maneuver_trajectory(dt)

    # --- Simulate radar ---
    radar = Radar(noise_std_x=noise_std, noise_std_y=noise_std)
    measurements = radar.measure_batch(true_trajectory)

    # --- Run KF (constant velocity — the whole point is it's wrong) ---
    print("Running CV Kalman Filter...")
    kf = KalmanFilter(dt=dt, q=q_var, r_x=noise_std, r_y=noise_std)

    estimated_trajectory = np.zeros((n_steps, 4))
    kf.init_state(measurements[0])
    estimated_trajectory[0] = kf.get_state()

    for t in range(1, n_steps):
        estimated_trajectory[t] = kf.step(measurements[t])

    # --- Analysis ---
    print("\nComputing metrics...")
    errors = position_error_over_time(true_trajectory, estimated_trajectory)

    # Per-segment RMSE
    rmse_a = rmse(
        true_trajectory[:maneuver_start],
        estimated_trajectory[:maneuver_start],
    )
    rmse_b = rmse(
        true_trajectory[maneuver_start:maneuver_end],
        estimated_trajectory[maneuver_start:maneuver_end],
    )
    rmse_c = rmse(
        true_trajectory[maneuver_end:],
        estimated_trajectory[maneuver_end:],
    )
    rmse_total = rmse(true_trajectory, estimated_trajectory)

    print("\n--- Results ---")
    print(f"Phase A (straight):  RMSE = {rmse_a:.2f} m")
    print(f"Phase B (turn):      RMSE = {rmse_b:.2f} m")
    print(f"Phase C (recovery):  RMSE = {rmse_c:.2f} m")
    print(f"Overall:             RMSE = {rmse_total:.2f} m")
    print(f"\nManeuver degradation: {rmse_b / rmse_a:.1f}x worse during turn")

    # --- Visualization ---
    print("\nGenerating plots...")

    # 1. Tracking plot
    fig_tracking = plot_tracking_result(
        true=true_trajectory,
        measured=measurements,
        estimated=estimated_trajectory,
        title="CV Kalman Filter During Coordinated Turn",
    )
    tracking_path = output_dir / "maneuver_tracking.png"
    fig_tracking.savefig(tracking_path, dpi=150)
    print(f"Saved: {tracking_path.relative_to(project_root)}")

    # 2. Error-over-time plot with maneuver window annotation
    fig_error = plot_error_over_time(
        errors=errors,
        title="KF Position Error — Maneuver Breakdown",
        vlines=[
            {"x": maneuver_start, "label": "Turn starts",
             "color": "orange", "linestyle": "--"},
            {"x": maneuver_end, "label": "Turn ends",
             "color": "green", "linestyle": "--"},
        ],
    )
    error_path = output_dir / "maneuver_error.png"
    fig_error.savefig(error_path, dpi=150)
    print(f"Saved: {error_path.relative_to(project_root)}")

    print("\nDone!")


if __name__ == "__main__":
    main()
