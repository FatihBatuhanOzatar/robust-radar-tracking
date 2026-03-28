"""Static matplotlib plots for tracking visualization."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_tracking_result(
    true: np.ndarray,
    measured: np.ndarray,
    estimated: np.ndarray,
    title: str = "Tracking Result",
) -> plt.Figure:
    """Plot true trajectory, noisy measurements, and KF estimate.

    Creates a 2D scatter/line plot comparing:
    - True target trajectory (solid blue line)
    - Raw radar measurements (red scattered dots)
    - Kalman filter estimates (dashed green line)

    Args:
        true: True trajectory, shape (n_steps, 4).
            Each row is [x, y, vx, vy].
        measured: Radar measurements, shape (n_steps, 2).
            Each row is [x, y].
        estimated: KF estimated trajectory, shape (n_steps, 4).
            Each row is [x, y, vx, vy].
        title: Plot title string.

    Returns:
        Matplotlib Figure object. Caller is responsible for saving
        via fig.savefig().
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(
        true[:, 0], true[:, 1],
        "b-", linewidth=2, label="True trajectory",
    )
    ax.scatter(
        measured[:, 0], measured[:, 1],
        c="red", s=15, alpha=0.5, label="Radar measurements", zorder=2,
    )
    ax.plot(
        estimated[:, 0], estimated[:, 1],
        "g--", linewidth=1.5, label="KF estimate",
    )

    # Mark start and end points
    ax.plot(true[0, 0], true[0, 1], "ko", markersize=8, label="Start")
    ax.plot(true[-1, 0], true[-1, 1], "k^", markersize=8, label="End")

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    return fig


def plot_error_over_time(
    errors: np.ndarray,
    title: str = "Position Error Over Time",
    vlines: Optional[list[dict]] = None,
) -> plt.Figure:
    """Plot per-step error over time.

    Args:
        errors: Error at each time step, shape (n_steps,).
        title: Plot title string.
        vlines: Optional list of vertical line annotations. Each entry
            is a dict with keys:
            - "x": time step position (required)
            - "label": annotation text (optional)
            - "color": line color (optional, default "orange")
            - "linestyle": line style (optional, default "--")

    Returns:
        Matplotlib Figure object. Caller is responsible for saving
        via fig.savefig().
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    steps = np.arange(len(errors))
    ax.plot(steps, errors, "r-", linewidth=1.2, label="Position error")
    ax.axhline(
        y=np.mean(errors), color="gray", linestyle="--",
        linewidth=1, label=f"Mean = {np.mean(errors):.2f} m",
    )

    if vlines:
        for vline in vlines:
            ax.axvline(
                x=vline["x"],
                color=vline.get("color", "orange"),
                linestyle=vline.get("linestyle", "--"),
                linewidth=1.5,
                label=vline.get("label"),
            )

    ax.set_xlabel("Time step")
    ax.set_ylabel("Error (m)")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig

