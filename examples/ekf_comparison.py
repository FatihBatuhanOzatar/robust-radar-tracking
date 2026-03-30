"""Phase 6 Demo: Extended Kalman Filter vs Standard KF during maneuver.

This script runs both filters on the same 3-phase maneuver scenario
(identical to examples/maneuver.py) and compares their performance.

Scenario:
  Phase A (steps  0-29): Straight-line flight  — both filters should do well
  Phase B (steps 30-69): Coordinated turn      — KF error spikes, EKF stays stable
  Phase C (steps 70-99): Straight-line flight  — both recover

Key result: the EKF uses a coordinated-turn motion model and estimates the
turn rate (omega) from the measurement stream. It does not need to know the
turn rate in advance. The KF, assuming CV, cannot represent the turn at all.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt

from radarsim.sim.target import Target
from radarsim.sim.radar import Radar
from radarsim.tracker.kf import KalmanFilter
from radarsim.tracker.ekf import ExtendedKalmanFilter
from radarsim.analysis.metrics import rmse, position_error_over_time


# ---------------------------------------------------------------------------
# Scenario helpers (mirror maneuver.py exactly)
# ---------------------------------------------------------------------------

def build_maneuver_trajectory(dt: float) -> np.ndarray:
    """Build the 3-phase ground-truth trajectory: straight → turn → straight.

    Returns:
        True trajectory, shape (100, 4).
    """
    n_straight_a = 30
    n_turn = 40
    n_straight_c = 30

    target_a = Target(x0=0.0, y0=0.0, vx0=50.0, vy0=20.0, model="cv")
    traj_a = np.zeros((n_straight_a, 4))
    for i in range(n_straight_a):
        traj_a[i] = target_a.step(dt)

    end_a = target_a.state
    target_b = Target(
        x0=end_a[0], y0=end_a[1], vx0=end_a[2], vy0=end_a[3],
        model="ct", turn_rate=0.05,
    )
    traj_b = np.zeros((n_turn, 4))
    for i in range(n_turn):
        traj_b[i] = target_b.step(dt)

    end_b = target_b.state
    target_c = Target(
        x0=end_b[0], y0=end_b[1], vx0=end_b[2], vy0=end_b[3],
        model="cv",
    )
    traj_c = np.zeros((n_straight_c, 4))
    for i in range(n_straight_c):
        traj_c[i] = target_c.step(dt)

    return np.vstack([traj_a, traj_b, traj_c])


def _pad_to_4col(positions: np.ndarray) -> np.ndarray:
    """Pad a (n, 2) position array to (n, 4) for compatibility with rmse()."""
    n = len(positions)
    return np.column_stack([positions, np.zeros((n, 2))])


# ---------------------------------------------------------------------------
# Run both filters
# ---------------------------------------------------------------------------

def run_kf(
    measurements: np.ndarray,
    dt: float,
    noise_std: float,
) -> np.ndarray:
    """Run the standard constant-velocity Kalman Filter.

    Args:
        measurements: Radar measurements, shape (n_steps, 2).
        dt: Time step (seconds).
        noise_std: Radar measurement noise std (meters).

    Returns:
        Estimated trajectory, shape (n_steps, 4) — [x, y, vx, vy].
    """
    n_steps = len(measurements)
    kf = KalmanFilter(dt=dt, q=0.5, r_x=noise_std, r_y=noise_std)
    estimated = np.zeros((n_steps, 4))

    kf.init_state(measurements[0])
    estimated[0] = kf.get_state()
    for t in range(1, n_steps):
        estimated[t] = kf.step(measurements[t])

    return estimated


def run_ekf(
    measurements: np.ndarray,
    dt: float,
    noise_std: float,
) -> np.ndarray:
    """Run the Extended Kalman Filter with coordinated-turn model.

    Q tuning rationale:
      q_pos=0.5   — small position noise; trust measurements for position
      q_vel=1.0   — speed convergence from v=0 init needs some flexibility
      q_theta=0.05 — heading updates by ~0.05 rad/step during turn; match this
      q_omega=0.01 — omega must jump from 0→0.05 at turn onset; needs bandwidth

    Args:
        measurements: Radar measurements, shape (n_steps, 2).
        dt: Time step (seconds).
        noise_std: Radar measurement noise std (meters).

    Returns:
        Estimated positions, shape (n_steps, 2) — [x, y].
    """
    n_steps = len(measurements)
    ekf = ExtendedKalmanFilter(
        dt=dt,
        q_params={"q_pos": 0.5, "q_vel": 1.0, "q_theta": 0.05, "q_omega": 0.01},
        r_x=noise_std,
        r_y=noise_std,
    )
    positions = np.zeros((n_steps, 2))

    ekf.init_state(measurements[0])
    positions[0] = ekf.get_position()
    for t in range(1, n_steps):
        ekf.step(measurements[t])
        positions[t] = ekf.get_position()

    return positions


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_trajectory_comparison(
    true: np.ndarray,
    measured: np.ndarray,
    kf_est: np.ndarray,
    ekf_pos: np.ndarray,
    maneuver_start: int,
    maneuver_end: int,
) -> plt.Figure:
    """2D comparison: true + measurements + KF + EKF on one axis.

    Args:
        true: True trajectory, shape (n, 4).
        measured: Raw radar measurements, shape (n, 2).
        kf_est: KF estimated trajectory, shape (n, 4).
        ekf_pos: EKF estimated positions, shape (n, 2).
        maneuver_start: Step index where turn begins.
        maneuver_end: Step index where turn ends.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(11, 8))

    # True trajectory with phase shading via scatter colour
    ax.plot(
        true[:, 0], true[:, 1],
        "b-", linewidth=2.0, label="True trajectory", zorder=4,
    )
    # Highlight the turn segment
    ax.plot(
        true[maneuver_start:maneuver_end, 0],
        true[maneuver_start:maneuver_end, 1],
        "b-", linewidth=3.5, alpha=0.35, label="Turn phase (true)",
    )
    ax.scatter(
        measured[:, 0], measured[:, 1],
        c="red", s=12, alpha=0.45, label="Radar measurements", zorder=2,
    )
    ax.plot(
        kf_est[:, 0], kf_est[:, 1],
        color="darkorange", linestyle="--", linewidth=1.8,
        label="CV Kalman Filter", zorder=5,
    )
    ax.plot(
        ekf_pos[:, 0], ekf_pos[:, 1],
        color="limegreen", linestyle="-", linewidth=2.0,
        label="Extended Kalman Filter (CT)", zorder=6,
    )

    # Start / end markers
    ax.plot(true[0, 0], true[0, 1], "ko", markersize=9, label="Start")
    ax.plot(true[-1, 0], true[-1, 1], "k^", markersize=9, label="End")

    ax.set_xlabel("X position (m)", fontsize=12)
    ax.set_ylabel("Y position (m)", fontsize=12)
    ax.set_title(
        "KF vs EKF — Coordinated Turn Maneuver\n"
        "(CV model breaks during turn; CT-EKF stays on target)",
        fontsize=13,
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    return fig


def plot_error_comparison(
    kf_errors: np.ndarray,
    ekf_errors: np.ndarray,
    maneuver_start: int,
    maneuver_end: int,
) -> plt.Figure:
    """Error-over-time plot with both filters and maneuver window shading.

    Args:
        kf_errors: Per-step KF position errors, shape (n,).
        ekf_errors: Per-step EKF position errors, shape (n,).
        maneuver_start: Step index where turn begins.
        maneuver_end: Step index where turn ends.

    Returns:
        Matplotlib Figure.
    """
    steps = np.arange(len(kf_errors))

    fig, ax = plt.subplots(figsize=(11, 5))

    # Shade the maneuver window
    ax.axvspan(
        maneuver_start, maneuver_end,
        alpha=0.12, color="gold", label="Turn phase",
    )

    ax.plot(
        steps, kf_errors,
        color="darkorange", linewidth=1.6, label="CV Kalman Filter",
    )
    ax.plot(
        steps, ekf_errors,
        color="limegreen", linewidth=1.8, label="Extended Kalman Filter (CT)",
    )

    # Phase boundary lines
    ax.axvline(
        maneuver_start, color="gray", linestyle="--",
        linewidth=1.2, label="Turn start/end",
    )
    ax.axvline(
        maneuver_end, color="gray", linestyle="--", linewidth=1.2,
    )

    ax.set_xlabel("Time step", fontsize=12)
    ax.set_ylabel("Position error (m)", fontsize=12)
    ax.set_title(
        "Position Error Over Time — KF vs EKF\n"
        "KF error spikes during turn; EKF remains stable",
        fontsize=13,
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    np.random.seed(42)
    output_dir = project_root / "output"
    docs_images_dir = project_root / "docs" / "images"
    output_dir.mkdir(exist_ok=True)
    docs_images_dir.mkdir(exist_ok=True)

    dt = 1.0
    noise_std = 25.0
    maneuver_start = 30
    maneuver_end = 70
    n_steps = 100

    print("=" * 60)
    print("  KF vs EKF — Maneuver Scenario Comparison")
    print("=" * 60)
    print(f"  {n_steps} steps, dt={dt}s, radar noise σ={noise_std}m")
    print(f"  Phase A (steps  0-{maneuver_start - 1}): Straight line (CV)")
    print(f"  Phase B (steps {maneuver_start}-{maneuver_end - 1}): CT turn  (ω=0.05 rad/s)")
    print(f"  Phase C (steps {maneuver_end}-{n_steps - 1}): Straight line (CV)")

    # --- Ground truth + measurements (identical data for both filters) ---
    print("\nBuilding trajectory and radar measurements...")
    true_trajectory = build_maneuver_trajectory(dt)
    radar = Radar(noise_std_x=noise_std, noise_std_y=noise_std)
    measurements = radar.measure_batch(true_trajectory)

    # --- Run filters ---
    print("Running CV Kalman Filter...")
    kf_est = run_kf(measurements, dt, noise_std)

    print("Running Extended Kalman Filter (CT model)...")
    ekf_pos = run_ekf(measurements, dt, noise_std)

    # Pad EKF positions to (n, 4) for rmse() compatibility
    ekf_est4 = _pad_to_4col(ekf_pos)

    # --- Per-phase RMSE ---
    def phase_rmse(est4: np.ndarray, start: int, stop: int) -> float:
        return rmse(true_trajectory[start:stop], est4[start:stop])

    kf_rmse_a = phase_rmse(kf_est, 0, maneuver_start)
    kf_rmse_b = phase_rmse(kf_est, maneuver_start, maneuver_end)
    kf_rmse_c = phase_rmse(kf_est, maneuver_end, n_steps)
    kf_rmse_total = rmse(true_trajectory, kf_est)

    ekf_rmse_a = phase_rmse(ekf_est4, 0, maneuver_start)
    ekf_rmse_b = phase_rmse(ekf_est4, maneuver_start, maneuver_end)
    ekf_rmse_c = phase_rmse(ekf_est4, maneuver_end, n_steps)
    ekf_rmse_total = rmse(true_trajectory, ekf_est4)

    print("\n" + "=" * 60)
    print("  RMSE Comparison")
    print("=" * 60)
    print(f"{'Phase':<28} {'CV-KF':>10} {'CT-EKF':>10} {'Improvement':>12}")
    print("-" * 60)
    print(
        f"{'A — Straight (steps 0-29)':<28} "
        f"{kf_rmse_a:>10.2f}m "
        f"{ekf_rmse_a:>10.2f}m "
        f"{kf_rmse_a / ekf_rmse_a:>10.2f}x"
    )
    print(
        f"{'B — Turn    (steps 30-69)':<28} "
        f"{kf_rmse_b:>10.2f}m "
        f"{ekf_rmse_b:>10.2f}m "
        f"{kf_rmse_b / ekf_rmse_b:>10.2f}x"
    )
    print(
        f"{'C — Recover (steps 70-99)':<28} "
        f"{kf_rmse_c:>10.2f}m "
        f"{ekf_rmse_c:>10.2f}m "
        f"{kf_rmse_c / ekf_rmse_c:>10.2f}x"
    )
    print("-" * 60)
    print(
        f"{'Overall':<28} "
        f"{kf_rmse_total:>10.2f}m "
        f"{ekf_rmse_total:>10.2f}m "
        f"{kf_rmse_total / ekf_rmse_total:>10.2f}x"
    )
    print("=" * 60)

    # Sanity check: EKF must be substantially better during turn
    if kf_rmse_b / ekf_rmse_b < 2.0:
        print("\n[WARNING] EKF turn improvement < 2x — check Q tuning or Jacobian!")
    else:
        print(
            f"\n[OK] EKF reduces turn RMSE by "
            f"{kf_rmse_b / ekf_rmse_b:.1f}x vs CV-KF."
        )

    # --- Per-step errors ---
    kf_errors = position_error_over_time(true_trajectory, kf_est)
    ekf_errors = position_error_over_time(true_trajectory, ekf_est4)

    # --- Generate plots ---
    print("\nGenerating plots...")

    fig_track = plot_trajectory_comparison(
        true=true_trajectory,
        measured=measurements,
        kf_est=kf_est,
        ekf_pos=ekf_pos,
        maneuver_start=maneuver_start,
        maneuver_end=maneuver_end,
    )
    fig_err = plot_error_comparison(
        kf_errors=kf_errors,
        ekf_errors=ekf_errors,
        maneuver_start=maneuver_start,
        maneuver_end=maneuver_end,
    )

    # Save to output/ (local, gitignored)
    track_out = output_dir / "ekf_comparison_tracking.png"
    err_out = output_dir / "ekf_comparison_error.png"
    fig_track.savefig(track_out, dpi=150, bbox_inches="tight")
    fig_err.savefig(err_out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {track_out.relative_to(project_root)}")
    print(f"  Saved: {err_out.relative_to(project_root)}")

    # Save to docs/images/ (tracked in git for README)
    track_docs = docs_images_dir / "ekf_comparison_tracking.png"
    err_docs = docs_images_dir / "ekf_comparison_error.png"
    fig_track.savefig(track_docs, dpi=150, bbox_inches="tight")
    fig_err.savefig(err_docs, dpi=150, bbox_inches="tight")
    print(f"  Saved: {track_docs.relative_to(project_root)}")
    print(f"  Saved: {err_docs.relative_to(project_root)}")

    plt.close("all")
    print("\nDone.")

    # Return numbers so README can embed them accurately
    return {
        "kf_a": kf_rmse_a, "kf_b": kf_rmse_b,
        "kf_c": kf_rmse_c, "kf_total": kf_rmse_total,
        "ekf_a": ekf_rmse_a, "ekf_b": ekf_rmse_b,
        "ekf_c": ekf_rmse_c, "ekf_total": ekf_rmse_total,
    }


if __name__ == "__main__":
    main()
