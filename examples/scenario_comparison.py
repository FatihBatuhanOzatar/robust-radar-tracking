"""Phase 5 Demo: Cross-scenario KF performance comparison.

Runs the Kalman Filter through all project scenarios and produces
a single bar chart comparing RMSE across conditions:

  1. Single target CV  — baseline (Phase 1)
  2. Maneuver (turn)   — CV model breakdown (Phase 2)
  3. ECM noise spike   — 5x noise amplification (Phase 3)
  4. ECM dropout       — 100% measurement loss (Phase 3)
  5. ECM bias          — systematic offset (Phase 3)

This chart tells the core project story: a CV Kalman Filter works
well under ideal conditions, degrades during maneuvers, and shows
varying resilience to different ECM modes.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from radarsim.analysis.metrics import rmse
from radarsim.sim.ecm import ECMModel
from radarsim.sim.radar import Radar
from radarsim.sim.target import Target
from radarsim.tracker.kf import KalmanFilter


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
RADAR_SEED = 42
ECM_SEED = 99


# ---------------------------------------------------------------------------
# Scenario runners — each returns overall RMSE
# ---------------------------------------------------------------------------

def run_single_target_cv() -> float:
    """Phase 1 baseline: single target, constant velocity."""
    dt, n_steps, noise_std, q = 1.0, 60, 25.0, 0.5

    target = Target(x0=0.0, y0=0.0, vx0=15.0, vy0=10.0, model="cv")
    true_states = target.get_trajectory(dt, n_steps)

    radar = Radar(noise_std_x=noise_std, noise_std_y=noise_std, seed=RADAR_SEED)
    measurements = radar.measure_batch(true_states)

    kf = KalmanFilter(dt=dt, q=q, r_x=noise_std, r_y=noise_std)
    estimated = np.zeros((n_steps, 4))
    kf.init_state(measurements[0])
    estimated[0] = kf.get_state()
    for t in range(1, n_steps):
        estimated[t] = kf.step(measurements[t])

    return rmse(true_states, estimated)


def run_maneuver() -> float:
    """Phase 2: coordinated turn — CV model breakdown."""
    dt, noise_std, q = 1.0, 25.0, 0.5

    # Build 3-phase trajectory: straight(30) → turn(40) → straight(30)
    target_a = Target(x0=0.0, y0=0.0, vx0=50.0, vy0=20.0, model="cv")
    traj_a = np.zeros((30, 4))
    for i in range(30):
        traj_a[i] = target_a.step(dt)

    end_a = target_a.state
    target_b = Target(
        x0=end_a[0], y0=end_a[1], vx0=end_a[2], vy0=end_a[3],
        model="ct", turn_rate=0.05,
    )
    traj_b = np.zeros((40, 4))
    for i in range(40):
        traj_b[i] = target_b.step(dt)

    end_b = target_b.state
    target_c = Target(
        x0=end_b[0], y0=end_b[1], vx0=end_b[2], vy0=end_b[3],
        model="cv",
    )
    traj_c = np.zeros((30, 4))
    for i in range(30):
        traj_c[i] = target_c.step(dt)

    true_states = np.vstack([traj_a, traj_b, traj_c])
    n_steps = len(true_states)

    radar = Radar(noise_std_x=noise_std, noise_std_y=noise_std, seed=RADAR_SEED)
    measurements = radar.measure_batch(true_states)

    kf = KalmanFilter(dt=dt, q=q, r_x=noise_std, r_y=noise_std)
    estimated = np.zeros((n_steps, 4))
    kf.init_state(measurements[0])
    estimated[0] = kf.get_state()
    for t in range(1, n_steps):
        estimated[t] = kf.step(measurements[t])

    return rmse(true_states, estimated)


def _run_ecm_scenario(ecm: ECMModel) -> float:
    """Helper: run CV target through an ECM scenario."""
    dt, n_steps, noise_std, q = 1.0, 100, 25.0, 0.5

    target = Target(x0=0.0, y0=0.0, vx0=50.0, vy0=20.0, model="cv")
    true_states = target.get_trajectory(dt, n_steps)

    radar = Radar(noise_std_x=noise_std, noise_std_y=noise_std, seed=RADAR_SEED)
    measurements = radar.measure_batch(true_states)

    kf = KalmanFilter(dt=dt, q=q, r_x=noise_std, r_y=noise_std)
    estimated = np.zeros((n_steps, 4))
    kf.init_state(measurements[0])
    estimated[0] = kf.get_state()

    for t in range(1, n_steps):
        degraded, is_valid = ecm.apply(measurements[t], t=t)
        if is_valid:
            estimated[t] = kf.step(degraded)
        else:
            estimated[t] = kf.step_no_measurement()

    return rmse(true_states, estimated)


def run_ecm_noise_spike() -> float:
    """Phase 3: noise spike — 5x noise during steps 30-59."""
    ecm = ECMModel(
        mode="noise_spike", ecm_start=30, ecm_end=60,
        noise_multiplier=5.0, noise_std=25.0, seed=ECM_SEED,
    )
    return _run_ecm_scenario(ecm)


def run_ecm_dropout() -> float:
    """Phase 3: dropout — 100% measurement loss during steps 30-59."""
    ecm = ECMModel(
        mode="dropout", ecm_start=30, ecm_end=60,
        dropout_prob=1.0, seed=ECM_SEED,
    )
    return _run_ecm_scenario(ecm)


def run_ecm_bias() -> float:
    """Phase 3: bias — +50m, +30m offset during steps 30-59."""
    ecm = ECMModel(
        mode="bias", ecm_start=30, ecm_end=60,
        bias=np.array([50.0, 30.0]),
    )
    return _run_ecm_scenario(ecm)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Phase 5: Cross-Scenario Performance Comparison")
    print("=" * 60)

    # Run all scenarios
    scenarios = [
        ("Single Target CV",  run_single_target_cv),
        ("Maneuver (turn)",    run_maneuver),
        ("ECM: Noise Spike",   run_ecm_noise_spike),
        ("ECM: Dropout",       run_ecm_dropout),
        ("ECM: Bias",          run_ecm_bias),
    ]

    names: list[str] = []
    rmses: list[float] = []

    for name, runner in scenarios:
        result = runner()
        names.append(name)
        rmses.append(result)
        print(f"  {name:<20s}  RMSE = {result:.2f} m")

    # Find baseline for comparison
    baseline = rmses[0]
    print(f"\n  Baseline (CV): {baseline:.2f} m")
    for name, r in zip(names[1:], rmses[1:]):
        ratio = r / baseline
        print(f"  {name:<20s} → {ratio:.1f}x baseline")

    # ------------------------------------------------------------------
    # Generate bar chart
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#2ecc71", "#e67e22", "#e74c3c", "#3498db", "#9b59b6"]
    y_pos = np.arange(len(names))

    bars = ax.barh(y_pos, rmses, color=colors, edgecolor="white",
                   linewidth=1.5, height=0.6)

    # Annotate each bar with its RMSE value
    for bar, val in zip(bars, rmses):
        ax.text(
            bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f} m",
            va="center", ha="left", fontsize=12, fontweight="bold",
            color="#2c3e50",
        )

    # Baseline reference line
    ax.axvline(x=baseline, color="#2ecc71", linestyle="--", linewidth=1.5,
               alpha=0.6, label=f"CV baseline ({baseline:.1f} m)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=12)
    ax.set_xlabel("Position RMSE (m)", fontsize=12)
    ax.set_title("Kalman Filter Performance Across Scenarios", fontsize=14,
                 fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlim(0, max(rmses) * 1.25)

    # Invert y axis so first scenario is on top
    ax.invert_yaxis()

    fig.tight_layout()
    save_path = output_dir / "scenario_comparison.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {save_path.relative_to(project_root)}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
