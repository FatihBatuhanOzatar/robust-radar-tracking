"""Multi-target tracker with data association."""

import numpy as np

from radarsim.tracker.kf import KalmanFilter


class Track:
    """Single target track wrapping a KalmanFilter instance.

    Each Track owns its own KalmanFilter and maintains bookkeeping
    counters for track lifecycle management (age, missed detections).
    The MultiTargetTracker orchestrates calling the KF methods and
    updating counters — Track itself is a data container.

    Args:
        track_id: Unique identifier for this track.
        kf: KalmanFilter instance dedicated to this track.
        initial_measurement: First measurement [x, y], shape (2,).
            Used to initialize the KF state via init_state().

    Attributes:
        id: Unique track identifier.
        kf: This track's KalmanFilter instance.
        age: Number of steps since track creation.
        missed: Consecutive steps without an assigned measurement.
    """

    def __init__(
        self,
        track_id: int,
        kf: KalmanFilter,
        initial_measurement: np.ndarray,
    ) -> None:
        self.id: int = track_id
        self.kf: KalmanFilter = kf
        self.age: int = 0
        self.missed: int = 0

        self.kf.init_state(initial_measurement)


def nearest_neighbor_associate(
    predictions: list[np.ndarray],
    measurements: np.ndarray | list[np.ndarray],
    gate_threshold: float | None = None,
) -> dict[int, int]:
    """Greedy nearest-neighbor data association.

    Computes a distance matrix between predicted positions and
    measurements, then greedily assigns the closest pairs. An
    optional gate rejects pairs whose distance exceeds a threshold.

    Args:
        predictions: List of predicted positions, each shape (2,).
            One entry per active track.
        measurements: Measurement array, shape (n_measurements, 2),
            or list of (2,) arrays. Each row is [x, y].
        gate_threshold: Maximum allowable distance for assignment.
            Pairs exceeding this distance are rejected even if they
            are the nearest available. None means no gating.

    Returns:
        Dictionary mapping track index to measurement index for
        matched pairs. Unassigned tracks and measurements can be
        inferred from missing keys/values.
    """
    n_tracks = len(predictions)

    # Normalize measurements to a 2D array
    if isinstance(measurements, list):
        if len(measurements) == 0:
            meas_arr = np.empty((0, 2))
        else:
            meas_arr = np.array(measurements)
    else:
        meas_arr = measurements

    n_meas = len(meas_arr)

    if n_tracks == 0 or n_meas == 0:
        return {}

    # Build distance matrix (Euclidean on position)
    dist_matrix = np.full((n_tracks, n_meas), np.inf)
    for i, pred in enumerate(predictions):
        for j in range(n_meas):
            dist_matrix[i, j] = np.linalg.norm(pred - meas_arr[j])

    # Apply gate — reject pairs beyond threshold
    if gate_threshold is not None:
        dist_matrix[dist_matrix > gate_threshold] = np.inf

    # Greedy assignment — closest pair first
    assignments: dict[int, int] = {}

    while True:
        # Find the global minimum
        if not np.isfinite(dist_matrix).any():
            break

        min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        track_idx, meas_idx = int(min_idx[0]), int(min_idx[1])

        assignments[track_idx] = meas_idx

        # Remove this track and measurement from the pool
        dist_matrix[track_idx, :] = np.inf
        dist_matrix[:, meas_idx] = np.inf

    return assignments
