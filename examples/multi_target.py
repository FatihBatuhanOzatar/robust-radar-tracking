"""Phase 4 Demo: Multi-target tracking with data association.

This script demonstrates tracking 3 targets with appearance/disappearance,
using nearest-neighbor data association with distance gating.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from radarsim.analysis.metrics import rmse
from radarsim.sim.radar import Radar
from radarsim.sim.target import Target
from radarsim.tracker.multi_target import MultiTargetTracker


def main() -> None:
    np.random.seed(42)  # For reproducible noise
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    dt = 1.0
    n_steps = 100
    noise_std = 15.0

    # Configuration for tuning
    q_var = 0.1
    gate_threshold = 100.0  # Loose gate to absorb noise and prevent fragmentation
    max_missed = 5

    print("--- Multi-Target Tracking Demo ---")
    print(f"Setting up simulation ({n_steps} steps, dt={dt}s)...")

    # 1. Setup Targets
    target_A = Target(0.0, 0.0, 10.0, 0.0, model="cv")     # Always present
    target_B = Target(0.0, 100.0, 8.0, -3.0, model="cv")   # Always present
    target_C = Target(50.0, 50.0, 5.0, 5.0, model="cv")    # Present 20 to 70

    traj_A = target_A.get_trajectory(dt, n_steps)
    traj_B = target_B.get_trajectory(dt, n_steps)
    traj_C = target_C.get_trajectory(dt, n_steps)

    radar = Radar(noise_std_x=noise_std, noise_std_y=noise_std)
    tracker = MultiTargetTracker(
        dt=dt,
        q=q_var,
        r_x=noise_std,
        r_y=noise_std,
        max_missed=max_missed,
        gate_threshold=gate_threshold,
    )

    # Logging structures
    true_counts = np.zeros(n_steps, dtype=int)
    tracker_counts = np.zeros(n_steps, dtype=int)
    all_measurements = []

    # Store track histories: dict[track_id] -> dict[step] -> state
    track_histories: dict[int, dict[int, np.ndarray]] = {}

    print("Running tracking loop (measurements shuffled each step)...")
    for t in range(n_steps):
        true_states_at_t = []
        # Target A active all the time
        true_states_at_t.append(traj_A[t])

        # Target B active all the time
        true_states_at_t.append(traj_B[t])

        # Target C active 20 <= t < 70
        if 20 <= t < 70:
            true_states_at_t.append(traj_C[t])

        true_states_at_t_arr = np.array(true_states_at_t)
        true_counts[t] = len(true_states_at_t_arr)

        # Measure and inject noise
        meas = radar.measure_batch(true_states_at_t_arr)

        # Shuffle measurements array
        np.random.shuffle(meas)
        for m in meas:
            all_measurements.append((t, m[0], m[1]))

        # Tracker step
        active_tracks = tracker.step(meas)
        tracker_counts[t] = len(active_tracks)

        for tr in active_tracks:
            if tr.id not in track_histories:
                track_histories[tr.id] = {}
            track_histories[tr.id][t] = tr.kf.get_state().copy()

    # 3. Analytics & Logging
    print("\n--- Track Statistics ---")
    print(f"{'ID':<4} | {'Birth':<6} | {'Death':<6} | {'Length':<6} | {'RMSE':<6}")
    print("-" * 38)

    true_trajs = [traj_A, traj_B, traj_C]

    for tr_id, history in track_histories.items():
        steps = sorted(list(history.keys()))
        t_start = steps[0]
        t_end = steps[-1]
        length = len(steps)

        # Skip very short Tracks (ghosts from clutter)
        if length < 5:
            continue

        # Reconstruct trajectory array for calculating RMSE
        est_traj_array = np.zeros((length, 4))
        for i, t in enumerate(steps):
            est_traj_array[i] = history[t]

        # Find best matching true trajectory based on RMSE
        best_rmse = float('inf')
        for true_t in true_trajs:
            true_slice = true_t[t_start:t_end+1]
            if len(true_slice) == length:
                err = rmse(true_slice, est_traj_array)
                if err < best_rmse:
                    best_rmse = float(err)

        print(f"{tr_id:<4} | {t_start:<6} | {t_end:<6} | {length:<6} | {best_rmse:<4.2f}m")

    # Overall tracks check
    valid_tracks = [t for t in track_histories.values() if len(t) >= 5]
    print(f"\nTotal valid tracks (>5 steps): {len(valid_tracks)} (Target: 3)")

    # 4. Custom Visualization
    print("\nGenerating plots...")

    plt.figure(figsize=(10, 8))

    # Plot truth
    plt.plot(traj_A[:, 0], traj_A[:, 1], 'k--', label="True Target A", alpha=0.5)
    plt.plot(traj_B[:, 0], traj_B[:, 1], 'k--', label="True Target B", alpha=0.5)
    plt.plot(traj_C[20:70, 0], traj_C[20:70, 1], 'k--', label="True Target C (t=20-70)", alpha=0.5)

    # Plot measurements
    meas_x = [m[1] for m in all_measurements]
    meas_y = [m[2] for m in all_measurements]
    plt.scatter(meas_x, meas_y, c='gray', marker='x', s=10, alpha=0.4, label="Measurements (noise=15m)")

    # Plot tracks
    cmap = plt.get_cmap("tab10")
    for idx, (tr_id, history) in enumerate(track_histories.items()):
        if len(history) < 5:
            continue
        color = cmap(idx % 10)
        steps = sorted(list(history.keys()))
        tr_x = [history[t][0] for t in steps]
        tr_y = [history[t][1] for t in steps]
        plt.plot(tr_x, tr_y, color=color, linewidth=2, label=f"Track ID {tr_id}")
        # Mark birth/death
        plt.plot(tr_x[0], tr_y[0], marker='o', color=color, markersize=8)
        plt.plot(tr_x[-1], tr_y[-1], marker='s', color=color, markersize=8)

    plt.title("Multi-Target Tracking Scenario (Shuffle + Gating)")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid(True)
    tracking_path = output_dir / "multi_target_tracking.png"
    plt.savefig(tracking_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {tracking_path.relative_to(project_root)}")
    plt.close()

    # Track count vs target count
    plt.figure(figsize=(10, 4))
    plt.plot(range(n_steps), true_counts, 'k--', linewidth=2, label="True Targets Count")
    plt.plot(range(n_steps), tracker_counts, 'b-', linewidth=2, label="Active Tracks Count", alpha=0.8)
    plt.title("Active Tracks vs True Targets over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Number of Entities")
    plt.yticks([0, 1, 2, 3, 4, 5])
    plt.legend()
    plt.grid(True)
    count_path = output_dir / "multi_target_track_count.png"
    plt.savefig(count_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {count_path.relative_to(project_root)}")
    plt.close()

    print("Done!")


if __name__ == "__main__":
    main()
