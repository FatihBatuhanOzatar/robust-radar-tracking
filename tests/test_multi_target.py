"""Unit tests for multi-target tracking (Track class and MultiTargetTracker)."""

import numpy as np
import pytest

from radarsim.tracker.kf import KalmanFilter
from radarsim.tracker.multi_target import Track, nearest_neighbor_associate


# ---------------------------------------------------------------------------
# Track class tests
# ---------------------------------------------------------------------------


class TestTrackCreation:
    """Tests for Track construction and initial state."""

    def test_track_creation_attributes(self) -> None:
        """Track initializes with correct id, age=0, missed=0."""
        kf = KalmanFilter(dt=1.0, q=0.5, r_x=25.0, r_y=25.0)
        measurement = np.array([100.0, 200.0])

        track = Track(track_id=7, kf=kf, initial_measurement=measurement)

        assert track.id == 7
        assert track.age == 0
        assert track.missed == 0

    def test_track_kf_initialized(self) -> None:
        """KF state matches initial measurement position, velocity is zero."""
        kf = KalmanFilter(dt=1.0, q=0.5, r_x=25.0, r_y=25.0)
        measurement = np.array([50.0, -30.0])

        track = Track(track_id=1, kf=kf, initial_measurement=measurement)

        state = track.kf.get_state()
        assert state.shape == (4,)
        np.testing.assert_allclose(state[:2], measurement)
        np.testing.assert_allclose(state[2:], [0.0, 0.0])

    def test_track_kf_is_same_instance(self) -> None:
        """Track stores the same KF instance (not a copy)."""
        kf = KalmanFilter(dt=1.0, q=0.5, r_x=25.0, r_y=25.0)
        measurement = np.array([10.0, 20.0])

        track = Track(track_id=0, kf=kf, initial_measurement=measurement)

        assert track.kf is kf

    def test_track_age_and_missed_are_mutable(self) -> None:
        """Age and missed counters can be incremented externally."""
        kf = KalmanFilter(dt=1.0, q=0.5, r_x=25.0, r_y=25.0)
        measurement = np.array([0.0, 0.0])

        track = Track(track_id=3, kf=kf, initial_measurement=measurement)

        track.age += 1
        track.missed += 1
        assert track.age == 1
        assert track.missed == 1

        track.missed = 0
        assert track.missed == 0

    def test_track_unique_ids(self) -> None:
        """Multiple tracks can have different IDs."""
        tracks = []
        for i in range(5):
            kf = KalmanFilter(dt=1.0, q=0.5, r_x=25.0, r_y=25.0)
            measurement = np.array([float(i * 10), float(i * 20)])
            tracks.append(Track(track_id=i, kf=kf, initial_measurement=measurement))

        ids = [t.id for t in tracks]
        assert ids == [0, 1, 2, 3, 4]
        assert len(set(ids)) == 5

    def test_track_independent_kf_instances(self) -> None:
        """Each track's KF is independent — updating one does not affect another."""
        kf_a = KalmanFilter(dt=1.0, q=0.5, r_x=25.0, r_y=25.0)
        kf_b = KalmanFilter(dt=1.0, q=0.5, r_x=25.0, r_y=25.0)

        track_a = Track(track_id=0, kf=kf_a, initial_measurement=np.array([0.0, 0.0]))
        track_b = Track(track_id=1, kf=kf_b, initial_measurement=np.array([100.0, 100.0]))

        # Step track A forward, track B should be unaffected
        track_a.kf.step(np.array([10.0, 10.0]))

        state_a = track_a.kf.get_state()
        state_b = track_b.kf.get_state()

        assert not np.allclose(state_a[:2], state_b[:2])
        np.testing.assert_allclose(state_b[:2], [100.0, 100.0])


# ---------------------------------------------------------------------------
# Nearest-neighbor data association tests
# ---------------------------------------------------------------------------


class TestNearestNeighborAssociate:
    """Tests for the nearest_neighbor_associate function."""

    def test_associate_perfect_match(self) -> None:
        """Predictions identical to measurements — all matched in order."""
        predictions = [
            np.array([0.0, 0.0]),
            np.array([100.0, 0.0]),
            np.array([0.0, 100.0]),
        ]
        measurements = np.array([
            [0.0, 0.0],
            [100.0, 0.0],
            [0.0, 100.0],
        ])

        result = nearest_neighbor_associate(predictions, measurements)

        assert result == {0: 0, 1: 1, 2: 2}

    def test_associate_shuffled_measurements(self) -> None:
        """Measurements in different order — correct pairing by proximity."""
        predictions = [
            np.array([0.0, 0.0]),
            np.array([100.0, 0.0]),
            np.array([0.0, 100.0]),
        ]
        # Measurements shuffled: [0,100] is meas 0, [100,0] is meas 1, [0,0] is meas 2
        measurements = np.array([
            [0.0, 100.0],
            [100.0, 0.0],
            [0.0, 0.0],
        ])

        result = nearest_neighbor_associate(predictions, measurements)

        # Track 0 ([0,0]) → meas 2 ([0,0])
        # Track 1 ([100,0]) → meas 1 ([100,0])
        # Track 2 ([0,100]) → meas 0 ([0,100])
        assert result == {0: 2, 1: 1, 2: 0}

    def test_associate_more_measurements_than_tracks(self) -> None:
        """Extra measurements remain unassigned."""
        predictions = [np.array([0.0, 0.0])]
        measurements = np.array([
            [0.0, 0.0],
            [500.0, 500.0],
        ])

        result = nearest_neighbor_associate(predictions, measurements)

        assert result == {0: 0}
        # Measurement index 1 is not in values → unassigned
        assigned_meas = set(result.values())
        assert 1 not in assigned_meas

    def test_associate_more_tracks_than_measurements(self) -> None:
        """Some tracks have no measurement — they are not in the result."""
        predictions = [
            np.array([0.0, 0.0]),
            np.array([100.0, 100.0]),
            np.array([200.0, 200.0]),
        ]
        measurements = np.array([[0.0, 0.0]])

        result = nearest_neighbor_associate(predictions, measurements)

        assert len(result) == 1
        assert result[0] == 0
        # Tracks 1 and 2 are unassigned
        assert 1 not in result
        assert 2 not in result

    def test_associate_gating_rejects_distant(self) -> None:
        """Gate rejects a distant measurement even if it is the nearest."""
        predictions = [
            np.array([0.0, 0.0]),
            np.array([100.0, 0.0]),
        ]
        measurements = np.array([
            [1.0, 1.0],      # close to track 0 (distance ~1.4)
            [500.0, 500.0],   # far from track 1 (distance ~500)
        ])

        result = nearest_neighbor_associate(
            predictions, measurements, gate_threshold=50.0,
        )

        # Track 0 matches meas 0 (distance ~1.4 < 50)
        # Track 1 has no match (closest meas is ~500 > 50)
        assert result == {0: 0}

    def test_associate_empty_measurements(self) -> None:
        """No measurements — empty result."""
        predictions = [np.array([0.0, 0.0])]
        measurements = np.array([]).reshape(0, 2)

        result = nearest_neighbor_associate(predictions, measurements)

        assert result == {}

    def test_associate_empty_predictions(self) -> None:
        """No predictions — empty result."""
        predictions: list[np.ndarray] = []
        measurements = np.array([[10.0, 20.0]])

        result = nearest_neighbor_associate(predictions, measurements)

        assert result == {}

