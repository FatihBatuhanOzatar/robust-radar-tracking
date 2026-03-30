"""Phase 5: Cross-scenario KF performance comparison.

Runs all five core scenarios and produces a bar chart comparing
position RMSE across them. This is the "one plot that tells the
whole story" — showing where the CV Kalman Filter excels and where
it breaks down.

Scenarios:
  1. Single target CV       — baseline (Phase 1)
  2. Coordinated turn       — maneuver degradation (Phase 2)
  3. ECM noise spike (5x)   — sensor jamming (Phase 3)
  4. ECM dropout (100%)     — complete signal loss (Phase 3)
  5. ECM bias (+50, +30)    — systematic measurement error (Phase 3)
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
# Shared configuration
# ---------------------------------------------------------------------------
DT = 1.0
NOISE_STD = 25.0
Q_VAR = 0.5
RADAR_SEED = 42
ECM_SEED = 99


# ---------------------------------------------------------------------------
# Scenario runners — each returns (true_states, estimated_states)
# ---------------------------------------------------------------------------

def run_single_target_cv() -> tuple[np.ndarray, np.ndarray]:
    """Baseline: 60-step straight-line CV tracking."""
    n_steps = 60
    target = Target(x0=0.0, y0=0.0, vx0=15.0, vy0=10.0, model="cv")
    true_states = target.get_trajectory(DT, n_steps)

    radar = Radar(noise_std_x=NOISE_STD, noise_std_y=NOISE_STD, seed=RADAR_SEED)
    measurements = radar.measure_batch(true_states)

    kf = KalmanFilter(dt=DT, q=Q_VAR, r_x=NOISE_STD, r_y=NOISE_STD)
    estimated = np.zeros((n_steps, 4))
    kf.init_state(measurements[0])
    estimated[0] = kf.get_state()
    for t in range(1, n_steps):
        estimated[t] = kf.step(measurements[t])

    return true_states, estimated


def run_maneuver() -> tuple[np.ndarray, np.ndarray]:
    """100-step scenario: straight → coordinated turn → straight."""
    n_straight_a, n_turn, n_straight_c = 30, 40, 30
    n_steps = n_straight_a + n_turn + n_straight_c

    # Phase A: straight
    target_a = Target(x0=0.0, y0=0.0, vx0=50.0, vy0=20.0, model="cv")
    traj_a = np.zeros((n_straight_a, 4))
    for i in range(n_straight_a):
        traj_a[i] = target_a.step(DT)

    # Phase B: coordinated turn
    end_a = target_a.state
    target_b = Target(
        x0=end_a[0], y0=end_a[1], vx0=end_a[2], vy0=end_a[3],
        model="ct", turn_rate=0.05,
    )
    traj_b = np.zeros((n_turn, 4))
    for i in range(n_turn):
        traj_b[i] = target_b.step(DT)

    # Phase C: straight again
    end_b = target_b.state
    target_c = Target(
        x0=end_b[0], y0=end_b[1], vx0=end_b[2], vy0=end_b[3],
        model="cv",
    )
    traj_c = np.zeros((n_straight_c, 4))
    for i in range(n_straight_c):
        traj_c[i] = target_c.step(DT)

    true_states = np.vstack([traj_a, traj_b, traj_c])

    np.random.seed(42)
    radar = Radar(noise_std_x=NOISE_STD, noise_std_y=NOISE_STD, seed=RADAR_SEED)
    measurements = radar.measure_batch(true_states)

    kf = KalmanFilter(dt=DT, q=Q_VAR, r_x=NOISE_STD, r_y=NOISE_STD)
    estimated = np.zeros((n_steps, 4))
    kf.init_state(measurements[0])
    estimated[0] = kf.get_state()
    for t in range(1, n_steps):
        estimated[t] = kf.step(measurements[t])

    return true_states, estimated


def _run_ecm_mode(
    mode: str,
    ecm_kwargs: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a 100-step CV scenario with an ECM window at steps 30-59."""
    n_steps = 100
    ecm_start, ecm_end = 30, 60

    target = Target(x0=0.0, y0=0.0, vx0=50.0, vy0=20.0, model="cv")
    true_states = target.get_trajectory(DT, n_steps)

    radar = Radar(noise_std_x=NOISE_STD, noise_std_y=NOISE_STD, seed=RADAR_SEED)
    measurements = radar.measure_batch(true_states)

    ecm = ECMModel(
        mode=mode,
        ecm_start=ecm_start,
        ecm_end=ecm_end,
        seed=ECM_SEED,
        **ecm_kwargs,
    )

    kf = KalmanFilter(dt=DT, q=Q_VAR, r_x=NOISE_STD, r_y=NOISE_STD)
    estimated = np.zeros((n_steps, 4))
    kf.init_state(measurements[0])
    estimated[0] = kf.get_state()

    for t in range(1, n_steps):
        degraded, is_valid = ecm.apply(measurements[t], t=t)
        if is_valid:
            estimated[t] = kf.step(degraded)
        else:
            estimated[t] = kf.step_no_measurement()

    return true_states, estimated


def run_ecm_noise_spike() -> tuple[np.ndarray, np.ndarray]:
    return _run_ecm_mode("noise_spike", {
        "noise_multiplier": 5.0,
        "noise_std": NOISE_STD,
    })


def run_ecm_dropout() -> tuple[np.ndarray, np.ndarray]:
    return _run_ecm_mode("dropout", {"dropout_prob": 1.0})


def run_ecm_bias() -> tuple[np.ndarray, np.ndarray]:
    return _run_ecm_mode("bias", {"bias": np.array([50.0, 30.0])})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    images_dir = project_root / "docs" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 5: Cross-Scenario Performance Comparison")
    print("=" * 60)

    # Define all scenarios
    scenarios = [
        ("Single Target CV",     run_single_target_cv),
        ("Coordinated Turn",     run_maneuver),
        ("ECM: Noise Spike (5×)", run_ecm_noise_spike),
        ("ECM: Dropout (100%)",  run_ecm_dropout),
        ("ECM: Bias (+50, +30)", run_ecm_bias),
    ]

    # Run each scenario and collect RMSE
    names: list[str] = []
    rmses: list[float] = []

    for name, runner in scenarios:
        true_states, estimated = runner()
        r = rmse(true_states, estimated)
        names.append(name)
        rmses.append(r)
        print(f"  {name:<25s}  RMSE = {r:.2f} m")

    # Identify best and worst
    best_idx = int(np.argmin(rmses))
    worst_idx = int(np.argmax(rmses))
    print(f"\n  Best:  {names[best_idx]} ({rmses[best_idx]:.2f} m)")
    print(f"  Worst: {names[worst_idx]} ({rmses[worst_idx]:.2f} m)")
    print(f"  Degradation factor: {rmses[worst_idx] / rmses[best_idx]:.1f}×")

    # ------------------------------------------------------------------
    # Generate bar chart
    # ------------------------------------------------------------------
    colors = ["#2ecc71", "#e67e22", "#e74c3c", "#3498db", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(12, 6))

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, rmses, color=colors, edgecolor="white",
                   linewidth=1.2, height=0.6)

    # Annotate RMSE values on each bar
    for i, (bar, val) in enumerate(zip(bars, rmses)):
        ax.text(
            val + max(rmses) * 0.02, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f} m",
            va="center", ha="left", fontsize=12, fontweight="bold",
            color=colors[i],
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=12)
    ax.set_xlabel("Position RMSE (m)", fontsize=13)
    ax.set_title(
        "Kalman Filter Performance — Cross-Scenario Comparison",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlim(0, max(rmses) * 1.25)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()  # best scenario on top

    fig.tight_layout()

    # Save to both output/ and docs/images/
    for save_path in [
        output_dir / "scenario_comparison.png",
        images_dir / "scenario_comparison.png",
    ]:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path.relative_to(project_root)}")

    plt.close(fig)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
