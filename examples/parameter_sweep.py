"""Phase 5 Demo: Parameter sweep analysis — Q and R sensitivity.

This script demonstrates how tracking accuracy depends on the Kalman
Filter's tuning parameters:

  - Q (process noise intensity) — how much the filter trusts its
    motion model vs. incoming measurements.
  - R (measurement noise) — how much measurement uncertainty the
    filter expects.

Three analyses are produced:
  1. RMSE vs Q sweep (fixed R)
  2. RMSE vs R sweep (fixed Q)
  3. 2D RMSE heatmap across Q × R combinations
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from radarsim.analysis.parameter_sweep import sweep_q, sweep_r, sweep_qr_heatmap
from radarsim.sim.radar import Radar
from radarsim.sim.target import Target
from radarsim.tracker.kf import KalmanFilter


# ---------------------------------------------------------------------------
# Scenario function
# ---------------------------------------------------------------------------
# Fixed simulation parameters
DT = 1.0
N_STEPS = 60
RADAR_SEED = 42


def cv_scenario(q: float, r_x: float, r_y: float) -> tuple[np.ndarray, np.ndarray]:
    """Single-target constant-velocity tracking scenario.

    Creates a fresh target, radar, and KF for each call so that
    results are independent and reproducible (fixed radar seed).

    Args:
        q: Process noise intensity for KalmanFilter.
        r_x: Measurement noise std in x (meters).
        r_y: Measurement noise std in y (meters).

    Returns:
        Tuple of (true_states, estimated_states), both shape (n_steps, 4).
    """
    # Ground truth
    target = Target(x0=0.0, y0=0.0, vx0=15.0, vy0=10.0, model="cv")
    true_states = target.get_trajectory(DT, N_STEPS)

    # Measurements — same seed every call for consistent comparison
    radar = Radar(noise_std_x=r_x, noise_std_y=r_y, seed=RADAR_SEED)
    measurements = radar.measure_batch(true_states)

    # KF tracking
    kf = KalmanFilter(dt=DT, q=q, r_x=r_x, r_y=r_y)
    estimated = np.zeros((N_STEPS, 4))
    kf.init_state(measurements[0])
    estimated[0] = kf.get_state()
    for t in range(1, N_STEPS):
        estimated[t] = kf.step(measurements[t])

    return true_states, estimated


def main() -> None:
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Phase 5: Parameter Sweep Analysis")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Q sweep
    # ------------------------------------------------------------------
    q_values = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    print("\n--- Q Sweep (fixed R=25m) ---")
    q_results = sweep_q(cv_scenario, q_values, r_x=25.0, r_y=25.0)

    for q_val, rmse_val in q_results.items():
        print(f"  Q={q_val:<6.2f}  RMSE={rmse_val:.2f} m")

    best_q = min(q_results, key=q_results.get)
    print(f"\n  Best Q: {best_q} (RMSE={q_results[best_q]:.2f} m)")

    # Plot Q sweep
    fig_q, ax_q = plt.subplots(figsize=(10, 6))
    qs = list(q_results.keys())
    rmses_q = list(q_results.values())

    ax_q.semilogx(qs, rmses_q, "o-", color="#2980b9", linewidth=2, markersize=8)
    ax_q.axvline(x=best_q, color="#27ae60", linestyle="--", linewidth=1.5,
                 label=f"Optimal Q={best_q}")

    ax_q.set_xlabel("Process noise intensity Q", fontsize=12)
    ax_q.set_ylabel("Position RMSE (m)", fontsize=12)
    ax_q.set_title("RMSE vs Process Noise Q (fixed R=25m)", fontsize=14)
    ax_q.legend(fontsize=11)
    ax_q.grid(True, alpha=0.3)
    fig_q.tight_layout()
    fig_q.savefig(output_dir / "q_sweep.png", dpi=150)
    plt.close(fig_q)
    print("  Saved: output/q_sweep.png")

    # ------------------------------------------------------------------
    # 2. R sweep
    # ------------------------------------------------------------------
    r_values = [5.0, 10.0, 15.0, 25.0, 35.0, 50.0, 75.0, 100.0]

    print("\n--- R Sweep (fixed Q=0.5) ---")
    r_results = sweep_r(cv_scenario, r_values, q=0.5)

    for r_val, rmse_val in r_results.items():
        print(f"  R={r_val:<6.1f}  RMSE={rmse_val:.2f} m")

    best_r = min(r_results, key=r_results.get)
    print(f"\n  Best R: {best_r} (RMSE={r_results[best_r]:.2f} m)")

    # Plot R sweep
    fig_r, ax_r = plt.subplots(figsize=(10, 6))
    rs = list(r_results.keys())
    rmses_r = list(r_results.values())

    ax_r.plot(rs, rmses_r, "s-", color="#e74c3c", linewidth=2, markersize=8)
    ax_r.axvline(x=best_r, color="#27ae60", linestyle="--", linewidth=1.5,
                 label=f"Optimal R={best_r}")

    ax_r.set_xlabel("Measurement noise std R (m)", fontsize=12)
    ax_r.set_ylabel("Position RMSE (m)", fontsize=12)
    ax_r.set_title("RMSE vs Measurement Noise R (fixed Q=0.5)", fontsize=14)
    ax_r.legend(fontsize=11)
    ax_r.grid(True, alpha=0.3)
    fig_r.tight_layout()
    fig_r.savefig(output_dir / "r_sweep.png", dpi=150)
    plt.close(fig_r)
    print("  Saved: output/r_sweep.png")

    # ------------------------------------------------------------------
    # 3. Q × R heatmap
    # ------------------------------------------------------------------
    q_grid = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    r_grid = [5.0, 10.0, 15.0, 25.0, 35.0, 50.0, 75.0, 100.0]

    print("\n--- Q × R Heatmap ---")
    print(f"  Grid: {len(q_grid)} Q values × {len(r_grid)} R values "
          f"= {len(q_grid) * len(r_grid)} runs")

    heatmap = sweep_qr_heatmap(cv_scenario, q_grid, r_grid)

    # Find overall minimum
    min_idx = np.unravel_index(np.argmin(heatmap), heatmap.shape)
    best_q_h = q_grid[min_idx[0]]
    best_r_h = r_grid[min_idx[1]]
    best_rmse_h = heatmap[min_idx]
    print(f"  Best combination: Q={best_q_h}, R={best_r_h} "
          f"(RMSE={best_rmse_h:.2f} m)")

    # Plot heatmap
    fig_hm, ax_hm = plt.subplots(figsize=(10, 8))

    im = ax_hm.imshow(
        heatmap,
        aspect="auto",
        cmap="RdYlGn_r",
        origin="lower",
    )

    # Axis labels
    ax_hm.set_xticks(range(len(r_grid)))
    ax_hm.set_xticklabels([f"{r:.0f}" for r in r_grid])
    ax_hm.set_yticks(range(len(q_grid)))
    ax_hm.set_yticklabels([f"{q}" for q in q_grid])

    ax_hm.set_xlabel("Measurement noise R (m)", fontsize=12)
    ax_hm.set_ylabel("Process noise Q", fontsize=12)
    ax_hm.set_title("Position RMSE Heatmap — Q vs R Parameter Space", fontsize=14)

    # Annotate each cell with its RMSE value
    for i in range(len(q_grid)):
        for j in range(len(r_grid)):
            val = heatmap[i, j]
            text_color = "white" if val > np.median(heatmap) else "black"
            ax_hm.text(j, i, f"{val:.1f}", ha="center", va="center",
                       color=text_color, fontsize=9, fontweight="bold")

    # Mark the best cell
    ax_hm.plot(min_idx[1], min_idx[0], "w*", markersize=20,
               markeredgecolor="black", markeredgewidth=1.5)

    cbar = fig_hm.colorbar(im, ax=ax_hm, shrink=0.85)
    cbar.set_label("RMSE (m)", fontsize=11)

    fig_hm.tight_layout()
    fig_hm.savefig(output_dir / "qr_heatmap.png", dpi=150)
    plt.close(fig_hm)
    print("  Saved: output/qr_heatmap.png")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
