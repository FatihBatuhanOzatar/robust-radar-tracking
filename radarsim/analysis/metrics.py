"""Performance metrics for tracking evaluation."""

import numpy as np


def rmse(true_states: np.ndarray, estimated_states: np.ndarray) -> float:
    """Compute root mean square error on position.

    Calculates the RMSE between true and estimated positions across
    all time steps. Only the position components [x, y] are used.

    Args:
        true_states: True trajectory, shape (n_steps, 4).
            Each row is [x, y, vx, vy].
        estimated_states: Estimated trajectory, shape (n_steps, 4).
            Each row is [x, y, vx, vy].

    Returns:
        Scalar RMSE value (meters).
    """
    pos_true = true_states[:, :2]
    pos_est = estimated_states[:, :2]
    errors = np.linalg.norm(pos_true - pos_est, axis=1)
    return float(np.sqrt(np.mean(errors ** 2)))


def position_error_over_time(
    true_states: np.ndarray,
    estimated_states: np.ndarray,
) -> np.ndarray:
    """Compute per-step Euclidean position error.

    Args:
        true_states: True trajectory, shape (n_steps, 4).
        estimated_states: Estimated trajectory, shape (n_steps, 4).

    Returns:
        Position error at each time step, shape (n_steps,).
    """
    pos_true = true_states[:, :2]
    pos_est = estimated_states[:, :2]
    return np.linalg.norm(pos_true - pos_est, axis=1)


def velocity_error_over_time(
    true_states: np.ndarray,
    estimated_states: np.ndarray,
) -> np.ndarray:
    """Compute per-step Euclidean velocity error.

    Args:
        true_states: True trajectory, shape (n_steps, 4).
        estimated_states: Estimated trajectory, shape (n_steps, 4).

    Returns:
        Velocity error at each time step, shape (n_steps,).
    """
    vel_true = true_states[:, 2:]
    vel_est = estimated_states[:, 2:]
    return np.linalg.norm(vel_true - vel_est, axis=1)
