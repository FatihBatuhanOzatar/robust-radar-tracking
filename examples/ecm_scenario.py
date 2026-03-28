"""Phase 3 Demo: Kalman Filter behavior under electronic countermeasures.

This script tests a constant-velocity Kalman Filter against three
adversarial sensor conditions on a straight-line trajectory:

  1. Noise spike  — radar noise multiplied by 5x during ECM window
  2. Dropout       — all measurements lost during ECM window
  3. Bias          — systematic offset added during ECM window

The target moves in a straight line at constant velocity the entire
time. The challenge is NOT maneuver — it is degraded measurements.

Key analysis:
  - Per-mode RMSE breakdown (pre-ECM, during-ECM, post-ECM)
  - Combined error comparison across all three modes
  - Q parameter effect on dropout recovery time
"""

import os
import sys
from pathlib import Path

# Add project root to path so we can import radarsim
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np

from radarsim.analysis.metrics import position_error_over_time, rmse
from radarsim.sim.ecm import ECMModel
from radarsim.sim.radar import Radar
from radarsim.sim.target import Target
from radarsim.tracker.kf import KalmanFilter
from radarsim.viz.plots import plot_tracking_result


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DT = 1.0
N_STEPS = 100
NOISE_STD = 25.0
Q_DEFAULT = 0.5
ECM_START = 30
ECM_END = 60
RADAR_SEED = 42
ECM_SEED = 99


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def run_ecm_scenario(
    ecm: ECMModel,
    true_trajectory: np.ndarray,
    measurements: np.ndarray,
    q: float = Q_DEFAULT,
) -> np.ndarray:
    """Run KF tracking with ECM applied to measurements.

    Creates a fresh KalmanFilter, loops through each time step, and
    uses ecm.apply() to decide between step() and step_no_measurement().

    Args:
        ecm: Configured ECMModel instance.
        true_trajectory: Ground truth states, shape (n_steps, 4).
        measurements: Raw radar measurements, shape (n_steps, 2).
        q: Process noise intensity for the KalmanFilter.

    Returns:
        Estimated trajectory, shape (n_steps, 4).
    """
    n_steps = len(true_trajectory)
    kf = KalmanFilter(dt=DT, q=q, r_x=NOISE_STD, r_y=NOISE_STD)

    estimated = np.zeros((n_steps, 4))
    kf.init_state(measurements[0])
    estimated[0] = kf.get_state()

    for t in range(1, n_steps):
        degraded, is_valid = ecm.apply(measurements[t], t=t)
        if is_valid:
            estimated[t] = kf.step(degraded)
        else:
            estimated[t] = kf.step_no_measurement()

    return estimated


def compute_recovery_time(
    errors: np.ndarray,
    ecm_end: int,
    pre_ecm_rmse: float,
    threshold_factor: float = 2.0,
) -> int | None:
    """Count steps after ECM ends until error drops below threshold.

    Recovery is defined as the first step after ecm_end where the
    instantaneous position error falls below threshold_factor * pre_ecm_rmse.

    Args:
        errors: Per-step position error, shape (n_steps,).
        ecm_end: Time step where ECM ends (exclusive).
        pre_ecm_rmse: RMSE during pre-ECM period (baseline).
        threshold_factor: Recovery threshold as multiple of baseline.

    Returns:
        Number of steps to recover, or None if never recovered.
    """
    threshold = threshold_factor * pre_ecm_rmse
    for i in range(ecm_end, len(errors)):
        if errors[i] < threshold:
            return i - ecm_end
    return None


def print_segment_rmse(
    label: str,
    true_trajectory: np.ndarray,
    estimated: np.ndarray,
) -> tuple[float, float, float]:
    """Print and return per-segment RMSE for an ECM scenario.

    Args:
        label: ECM mode name for display.
        true_trajectory: Ground truth states, shape (n_steps, 4).
        estimated: Estimated states, shape (n_steps, 4).

    Returns:
        Tuple of (pre_ecm_rmse, during_ecm_rmse, post_ecm_rmse).
    """
    rmse_pre = rmse(true_trajectory[:ECM_START], estimated[:ECM_START])
    rmse_during = rmse(
        true_trajectory[ECM_START:ECM_END],
        estimated[ECM_START:ECM_END],
    )
    rmse_post = rmse(true_trajectory[ECM_END:], estimated[ECM_END:])

    print(f"  {label}:")
    print(f"    Pre-ECM  (steps  0-{ECM_START - 1:2d}):  RMSE = {rmse_pre:.2f} m")
    print(f"    During   (steps {ECM_START}-{ECM_END - 1}):  RMSE = {rmse_during:.2f} m")
    print(f"    Post-ECM (steps {ECM_END}-{N_STEPS - 1}):  RMSE = {rmse_post:.2f} m")

    return rmse_pre, rmse_during, rmse_post


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Phase 3 Demo: ECM Scenario Analysis")
    print("=" * 60)
    print(f"\nScenario: {N_STEPS} steps, dt={DT}s, straight-line CV target")
    print(f"Radar noise: {NOISE_STD} m (both axes)")
    print(f"KF process noise: q={Q_DEFAULT}")
    print(f"ECM window: steps {ECM_START}-{ECM_END - 1} (inclusive)")

    # ------------------------------------------------------------------
    # Build shared trajectory and measurements
    # ------------------------------------------------------------------
    target = Target(x0=0.0, y0=0.0, vx0=50.0, vy0=20.0, model="cv")
    true_trajectory = target.get_trajectory(dt=DT, n_steps=N_STEPS)

    radar = Radar(noise_std_x=NOISE_STD, noise_std_y=NOISE_STD, seed=RADAR_SEED)
    measurements = radar.measure_batch(true_trajectory)

    # ------------------------------------------------------------------
    # Part 1: Run all 3 ECM modes
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Part 1: ECM Mode Comparison")
    print("-" * 60)

    ecm_configs = {
        "noise_spike": ECMModel(
            mode="noise_spike",
            ecm_start=ECM_START,
            ecm_end=ECM_END,
            noise_multiplier=5.0,
            noise_std=NOISE_STD,
            seed=ECM_SEED,
        ),
        "dropout": ECMModel(
            mode="dropout",
            ecm_start=ECM_START,
            ecm_end=ECM_END,
            dropout_prob=1.0,
            seed=ECM_SEED,
        ),
        "bias": ECMModel(
            mode="bias",
            ecm_start=ECM_START,
            ecm_end=ECM_END,
            bias=np.array([50.0, 30.0]),
        ),
    }

    results = {}  # mode -> (estimated, errors, rmse_segments)

    for mode_name, ecm in ecm_configs.items():
        estimated = run_ecm_scenario(ecm, true_trajectory, measurements)
        errors = position_error_over_time(true_trajectory, estimated)
        rmse_segments = print_segment_rmse(mode_name, true_trajectory, estimated)
        results[mode_name] = (estimated, errors, rmse_segments)

    # ------------------------------------------------------------------
    # Part 1b: Per-mode tracking plots
    # ------------------------------------------------------------------
    print("\nGenerating per-mode tracking plots...")

    mode_titles = {
        "noise_spike": "Noise Spike (5x noise, steps 30-59)",
        "dropout": "Measurement Dropout (steps 30-59)",
        "bias": "Bias Injection (+50m, +30m, steps 30-59)",
    }

    for mode_name, (estimated, _, _) in results.items():
        fig = plot_tracking_result(
            true=true_trajectory,
            measured=measurements,
            estimated=estimated,
            title=f"ECM: {mode_titles[mode_name]}",
        )
        filename = f"ecm_{mode_name}.png"
        filepath = output_dir / filename
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        print(f"  Saved: output/{filename}")

    # ------------------------------------------------------------------
    # Part 2: Combined error comparison plot
    # ------------------------------------------------------------------
    print("\nGenerating combined error comparison plot...")

    fig_cmp, ax_cmp = plt.subplots(figsize=(12, 6))

    mode_colors = {
        "noise_spike": "#e74c3c",
        "dropout": "#2980b9",
        "bias": "#27ae60",
    }
    mode_labels = {
        "noise_spike": "Noise Spike (5x)",
        "dropout": "Dropout (100%)",
        "bias": "Bias (+50, +30)",
    }

    steps = np.arange(N_STEPS)
    for mode_name, (_, errors, _) in results.items():
        ax_cmp.plot(
            steps, errors,
            color=mode_colors[mode_name],
            linewidth=1.2,
            label=mode_labels[mode_name],
        )

    # Mark ECM window
    ax_cmp.axvspan(
        ECM_START, ECM_END,
        alpha=0.15, color="gray", label="ECM window",
    )
    ax_cmp.axvline(x=ECM_START, color="gray", linestyle="--", linewidth=0.8)
    ax_cmp.axvline(x=ECM_END, color="gray", linestyle="--", linewidth=0.8)

    ax_cmp.set_xlabel("Time step")
    ax_cmp.set_ylabel("Position error (m)")
    ax_cmp.set_title("KF Position Error — ECM Mode Comparison")
    ax_cmp.legend(loc="upper right")
    ax_cmp.grid(True, alpha=0.3)
    fig_cmp.tight_layout()
    fig_cmp.savefig(output_dir / "ecm_comparison.png", dpi=150)
    plt.close(fig_cmp)
    print("  Saved: output/ecm_comparison.png")

    # ------------------------------------------------------------------
    # Part 3: Q parameter comparison (dropout mode)
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Part 3: Q Parameter Effect on Dropout Recovery")
    print("-" * 60)

    q_values = [0.1, 0.5, 2.0]
    q_colors = {0.1: "#8e44ad", 0.5: "#2980b9", 2.0: "#e67e22"}

    fig_q, ax_q = plt.subplots(figsize=(12, 6))

    for q_val in q_values:
        ecm_dropout = ECMModel(
            mode="dropout",
            ecm_start=ECM_START,
            ecm_end=ECM_END,
            dropout_prob=1.0,
            seed=ECM_SEED,
        )
        estimated_q = run_ecm_scenario(
            ecm_dropout, true_trajectory, measurements, q=q_val,
        )
        errors_q = position_error_over_time(true_trajectory, estimated_q)

        # Compute metrics
        rmse_pre_q = rmse(
            true_trajectory[:ECM_START], estimated_q[:ECM_START],
        )
        rmse_post_q = rmse(
            true_trajectory[ECM_END:], estimated_q[ECM_END:],
        )
        peak_post = float(np.max(errors_q[ECM_END:])) if ECM_END < len(errors_q) else 0.0
        recovery = compute_recovery_time(
            errors_q, ECM_END, rmse_pre_q, threshold_factor=2.0,
        )

        recovery_str = (
            f"{recovery} steps" if recovery is not None else "did not recover"
        )
        print(f"  q={q_val:.1f}:")
        print(f"    Pre-ECM RMSE  = {rmse_pre_q:.2f} m")
        print(f"    Post-ECM RMSE = {rmse_post_q:.2f} m")
        print(f"    Peak error after ECM = {peak_post:.2f} m")
        print(f"    Recovery (< 2x pre-ECM) = {recovery_str}")

        ax_q.plot(
            steps, errors_q,
            color=q_colors[q_val],
            linewidth=1.2,
            label=f"q={q_val} (post RMSE: {rmse_post_q:.1f}m)",
        )

    # Mark ECM window
    ax_q.axvspan(
        ECM_START, ECM_END,
        alpha=0.15, color="gray", label="ECM window",
    )
    ax_q.axvline(x=ECM_START, color="gray", linestyle="--", linewidth=0.8)
    ax_q.axvline(x=ECM_END, color="gray", linestyle="--", linewidth=0.8)

    ax_q.set_xlabel("Time step")
    ax_q.set_ylabel("Position error (m)")
    ax_q.set_title("Dropout Recovery — Q Parameter Comparison")
    ax_q.legend(loc="upper right")
    ax_q.grid(True, alpha=0.3)
    fig_q.tight_layout()
    fig_q.savefig(output_dir / "ecm_q_comparison.png", dpi=150)
    plt.close(fig_q)
    print("\n  Saved: output/ecm_q_comparison.png")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
