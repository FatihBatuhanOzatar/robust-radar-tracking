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
