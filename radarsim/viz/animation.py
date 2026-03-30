"""Animated matplotlib visualization for tracking display."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def animate_tracking(
    true: np.ndarray,
    measured: np.ndarray,
    estimated: np.ndarray,
    dt: float,
    save_path: str,
    fps: int = 20,
    trail_length: int = 10,
) -> None:
    """Create an animated GIF of the tracking scenario.

    Shows the true trajectory, radar measurements, and KF estimate
    building up frame by frame. A trailing window of recent
    measurements fades out to keep the plot readable.

    Args:
        true: True trajectory, shape (n_steps, 4).
        measured: Radar measurements, shape (n_steps, 2).
        estimated: KF estimated trajectory, shape (n_steps, 4).
        dt: Time step duration (seconds), used for time display.
        save_path: File path to save the GIF (e.g., "tracking.gif").
        fps: Frames per second for the output GIF.
        trail_length: Number of recent measurements to display.
    """
    n_steps = len(true)

    # Compute axis limits with padding
    all_x = np.concatenate([true[:, 0], measured[:, 0], estimated[:, 0]])
    all_y = np.concatenate([true[:, 1], measured[:, 1], estimated[:, 1]])
    pad_x = (all_x.max() - all_x.min()) * 0.1
    pad_y = (all_y.max() - all_y.min()) * 0.1

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
    ax.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)

    # Plot elements
    true_line, = ax.plot([], [], "b-", linewidth=2, label="True trajectory")
    est_line, = ax.plot([], [], "g--", linewidth=1.5, label="KF estimate")
    meas_scatter = ax.scatter([], [], c="red", s=30, alpha=0.6,
                              label="Radar measurement", zorder=5)
    current_true, = ax.plot([], [], "bo", markersize=10, zorder=6)
    current_est, = ax.plot([], [], "gs", markersize=8, zorder=6)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                        fontsize=12, verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="wheat",
                                  alpha=0.8))
    ax.legend(loc="upper right")
    title = ax.set_title("Kalman Filter Tracking")

    def init():
        """Initialize animation elements."""
        true_line.set_data([], [])
        est_line.set_data([], [])
        meas_scatter.set_offsets(np.empty((0, 2)))
        current_true.set_data([], [])
        current_est.set_data([], [])
        time_text.set_text("")
        return true_line, est_line, meas_scatter, current_true, current_est, time_text

    def update(frame):
        """Update animation for a single frame."""
        i = frame + 1  # Show at least 1 point

        # True trajectory up to current frame
        true_line.set_data(true[:i, 0], true[:i, 1])

        # KF estimate up to current frame
        est_line.set_data(estimated[:i, 0], estimated[:i, 1])

        # Recent measurements (trailing window)
        start = max(0, i - trail_length)
        trail = measured[start:i]
        meas_scatter.set_offsets(trail)
        # Fade alpha for older measurements
        n_visible = len(trail)
        alphas = np.linspace(0.2, 0.8, n_visible)
        meas_scatter.set_alpha(None)
        colors = np.zeros((n_visible, 4))
        colors[:, 0] = 1.0  # Red
        colors[:, 3] = alphas
        meas_scatter.set_facecolors(colors)

        # Current position markers
        current_true.set_data([true[i - 1, 0]], [true[i - 1, 1]])
        current_est.set_data([estimated[i - 1, 0]], [estimated[i - 1, 1]])

        # Time display
        time_text.set_text(f"t = {(i - 1) * dt:.0f}s  (step {i - 1})")

        return true_line, est_line, meas_scatter, current_true, current_est, time_text

    anim = FuncAnimation(
        fig, update, init_func=init,
        frames=n_steps, interval=1000 // fps, blit=True,
    )

    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer)
    plt.close(fig)
