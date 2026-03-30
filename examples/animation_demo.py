"""Phase 5 Demo: Animated tracking visualization.

Generates a GIF animation of the single-target CV tracking scenario,
showing the true trajectory, radar measurements, and KF estimate
building up frame by frame.
"""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from radarsim.sim.target import Target
from radarsim.sim.radar import Radar
from radarsim.tracker.kf import KalmanFilter
from radarsim.viz.animation import animate_tracking


def main() -> None:
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    images_dir = project_root / "docs" / "images"
    images_dir.mkdir(exist_ok=True)

    dt = 1.0
    n_steps = 60
    noise_std = 25.0
    q_var = 0.5

    print("--- Animation Demo ---")
    print("Generating tracking data...")

    # Ground truth
    target = Target(x0=0.0, y0=0.0, vx0=15.0, vy0=10.0, model="cv")
    true_states = target.get_trajectory(dt, n_steps)

    # Measurements
    radar = Radar(noise_std_x=noise_std, noise_std_y=noise_std, seed=42)
    measurements = radar.measure_batch(true_states)

    # KF tracking
    kf = KalmanFilter(dt=dt, q=q_var, r_x=noise_std, r_y=noise_std)
    estimated = np.zeros((n_steps, 4))
    kf.init_state(measurements[0])
    estimated[0] = kf.get_state()
    for t in range(1, n_steps):
        estimated[t] = kf.step(measurements[t])

    # Generate animation
    save_path = images_dir / "tracking_animation.gif"
    print(f"Creating animation ({n_steps} frames)...")
    animate_tracking(
        true=true_states,
        measured=measurements,
        estimated=estimated,
        dt=dt,
        save_path=str(save_path),
        fps=15,
        trail_length=8,
    )
    print(f"Saved: {save_path.relative_to(project_root)}")
    print("Done!")


if __name__ == "__main__":
    main()
