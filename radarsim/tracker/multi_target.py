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


class MultiTargetTracker:
    """Manages multiple tracks with nearest-neighbor data association.

    Each step: predict all tracks, associate predictions with
    measurements, update matched tracks, coast unmatched tracks.
    Track initialization (birth) and termination (death) are handled
    in step() — unassigned measurements create new tracks, and tracks
    exceeding max_missed consecutive misses are removed.

    Args:
        dt: Time step duration (seconds).
        q: Process noise intensity for KalmanFilter instances.
        r_x: Measurement noise std in x (meters).
        r_y: Measurement noise std in y (meters).
        max_missed: Maximum consecutive missed measurements before
            a track is terminated.
        gate_threshold: Maximum distance for measurement-track
            assignment. Pairs beyond this distance are rejected.
            None means no gating (accept all assignments).

    Attributes:
        dt: Time step duration.
        q: Process noise intensity.
        r_x: Measurement noise std in x.
        r_y: Measurement noise std in y.
        max_missed: Track termination threshold.
    """

    def __init__(
        self,
        dt: float,
        q: float,
        r_x: float,
        r_y: float,
        max_missed: int,
        gate_threshold: float | None = None,
    ) -> None:
        self.dt: float = dt
        self.q: float = q
        self.r_x: float = r_x
        self.r_y: float = r_y
        self.max_missed: int = max_missed
        self._gate_threshold: float | None = gate_threshold

        self._tracks: list[Track] = []
        self._next_id: int = 0

    def step(self, measurements: np.ndarray | list[np.ndarray]) -> list[Track]:
        """Process one time step of measurements.

        Core loop: predict all → associate → update matched →
        coast unmatched → increment age.

        Note: Track initialization (birth from unassigned
        measurements) and termination (death from excessive misses)
        are not yet implemented — they will be added in Tasks 4 & 5.

        Args:
            measurements: Array of measurements, shape
                (n_measurements, 2), or list of (2,) arrays.

        Returns:
            List of currently active tracks.
        """
        # Normalize measurements
        if isinstance(measurements, list):
            if len(measurements) == 0:
                meas_arr = np.empty((0, 2))
            else:
                meas_arr = np.array(measurements)
        else:
            meas_arr = measurements

        n_meas = len(meas_arr)

        # 1. Predict all active tracks, collect predicted positions
        predictions: list[np.ndarray] = []
        for track in self._tracks:
            predicted_state = track.kf.predict()
            predictions.append(predicted_state[:2])  # position only

        # 2. Associate predictions with measurements
        assignments = self.associate(predictions, meas_arr)

        # 3. Update matched tracks (predict already ran, only update)
        assigned_track_indices = set(assignments.keys())
        for track_idx, meas_idx in assignments.items():
            self._tracks[track_idx].kf.update(meas_arr[meas_idx])
            self._tracks[track_idx].missed = 0

        # 4. Coast unmatched tracks (predict already ran, just mark missed)
        for i, track in enumerate(self._tracks):
            if i not in assigned_track_indices:
                track.missed += 1

        # 5. Increment age for all tracks
        for track in self._tracks:
            track.age += 1

        return self.get_active_tracks()

    def associate(
        self,
        predictions: list[np.ndarray],
        measurements: np.ndarray | list[np.ndarray],
    ) -> dict[int, int]:
        """Run nearest-neighbor data association.

        Thin wrapper around the standalone nearest_neighbor_associate
        function, using this tracker's gate threshold.

        Args:
            predictions: List of predicted positions, each (2,).
            measurements: Measurement array, shape (n, 2).

        Returns:
            Dictionary mapping track index to measurement index.
        """
        return nearest_neighbor_associate(
            predictions, measurements, self._gate_threshold,
        )

    def get_active_tracks(self) -> list[Track]:
        """Return the list of currently active tracks.

        Returns:
            Shallow copy of the active track list.
        """
        return list(self._tracks)

    def _create_track(self, measurement: np.ndarray) -> Track:
        """Create a new track from a measurement.

        Instantiates a fresh KalmanFilter with this tracker's
        parameters and wraps it in a Track. The track is added
        to the active list and assigned the next available ID.

        Args:
            measurement: Initial measurement [x, y], shape (2,).

        Returns:
            The newly created Track.
        """
        kf = KalmanFilter(
            dt=self.dt, q=self.q, r_x=self.r_x, r_y=self.r_y,
        )
        track = Track(
            track_id=self._next_id,
            kf=kf,
            initial_measurement=measurement,
        )
        self._tracks.append(track)
        self._next_id += 1
        return track
