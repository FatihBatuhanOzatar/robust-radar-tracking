"""Unit tests for ECMModel class."""

import numpy as np
import pytest

from radarsim.sim.ecm import ECMModel


class TestNoiseSpike:
    """Tests for noise_spike ECM mode."""

    def test_returns_ndarray_and_true_during_ecm_window(self):
        """Noise spike returns (ndarray, True) with correct shape
        during the ECM window."""
        ecm = ECMModel(
            mode="noise_spike",
            ecm_start=10,
            ecm_end=20,
            noise_multiplier=5.0,
            noise_std=25.0,
            seed=42,
        )
        measurement = np.array([100.0, 200.0])

        result, is_valid = ecm.apply(measurement, t=15)

        assert is_valid is True
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)

    def test_does_not_modify_measurement_outside_ecm_window(self):
        """Noise spike passes measurement through unchanged outside
        the ECM window."""
        ecm = ECMModel(
            mode="noise_spike",
            ecm_start=10,
            ecm_end=20,
            noise_multiplier=5.0,
            noise_std=25.0,
            seed=42,
        )
        measurement = np.array([100.0, 200.0])

        # Before ECM window
        result_before, valid_before = ecm.apply(measurement, t=5)
        assert valid_before is True
        np.testing.assert_array_equal(result_before, measurement)

        # After ECM window
        result_after, valid_after = ecm.apply(measurement, t=25)
        assert valid_after is True
        np.testing.assert_array_equal(result_after, measurement)

    def test_adds_extra_noise_during_ecm_window(self):
        """Noise spike modifies the measurement during the ECM window
        (the result should differ from the original measurement)."""
        ecm = ECMModel(
            mode="noise_spike",
            ecm_start=10,
            ecm_end=20,
            noise_multiplier=5.0,
            noise_std=25.0,
            seed=42,
        )
        measurement = np.array([100.0, 200.0])

        result, _ = ecm.apply(measurement, t=15)

        # With multiplier=5 and noise_std=25, extra noise has std=100.
        # Extremely unlikely that extra noise is exactly zero.
        assert not np.array_equal(result, measurement)

    def test_requires_positive_noise_std(self):
        """Noise spike mode raises ValueError if noise_std is not positive."""
        with pytest.raises(ValueError, match="noise_std must be positive"):
            ECMModel(
                mode="noise_spike",
                ecm_start=10,
                ecm_end=20,
                noise_multiplier=5.0,
                noise_std=0.0,
            )


class TestDropout:
    """Tests for dropout ECM mode."""

    def test_returns_none_false_during_ecm_window(self):
        """Dropout with prob=1.0 always returns (None, False) during
        the ECM window."""
        ecm = ECMModel(
            mode="dropout",
            ecm_start=10,
            ecm_end=20,
            dropout_prob=1.0,
        )
        measurement = np.array([100.0, 200.0])

        result, is_valid = ecm.apply(measurement, t=15)

        assert result is None
        assert is_valid is False

    def test_returns_measurement_true_outside_ecm_window(self):
        """Dropout passes measurement through unchanged outside
        the ECM window."""
        ecm = ECMModel(
            mode="dropout",
            ecm_start=10,
            ecm_end=20,
            dropout_prob=1.0,
        )
        measurement = np.array([100.0, 200.0])

        # Before ECM window
        result_before, valid_before = ecm.apply(measurement, t=5)
        assert valid_before is True
        np.testing.assert_array_equal(result_before, measurement)

        # After ECM window
        result_after, valid_after = ecm.apply(measurement, t=25)
        assert valid_after is True
        np.testing.assert_array_equal(result_after, measurement)

    def test_partial_dropout_probability(self):
        """Dropout with prob < 1.0 sometimes keeps measurements."""
        ecm = ECMModel(
            mode="dropout",
            ecm_start=0,
            ecm_end=100,
            dropout_prob=0.5,
            seed=42,
        )
        measurement = np.array([100.0, 200.0])

        kept_count = 0
        dropped_count = 0
        for t in range(100):
            result, is_valid = ecm.apply(measurement, t=t)
            if is_valid:
                kept_count += 1
            else:
                dropped_count += 1

        # With prob=0.5 over 100 trials, expect roughly 50/50.
        # Allow wide margin but both must be > 0.
        assert kept_count > 0, "Expected some measurements to be kept"
        assert dropped_count > 0, "Expected some measurements to be dropped"


class TestBias:
    """Tests for bias ECM mode."""

    def test_adds_correct_offset_during_ecm_window(self):
        """Bias adds the exact configured offset during the ECM window."""
        bias_vector = np.array([50.0, 30.0])
        ecm = ECMModel(
            mode="bias",
            ecm_start=10,
            ecm_end=20,
            bias=bias_vector,
        )
        measurement = np.array([100.0, 200.0])

        result, is_valid = ecm.apply(measurement, t=15)

        assert is_valid is True
        expected = measurement + bias_vector
        np.testing.assert_array_almost_equal(result, expected)

    def test_does_not_modify_measurement_outside_ecm_window(self):
        """Bias passes measurement through unchanged outside
        the ECM window."""
        bias_vector = np.array([50.0, 30.0])
        ecm = ECMModel(
            mode="bias",
            ecm_start=10,
            ecm_end=20,
            bias=bias_vector,
        )
        measurement = np.array([100.0, 200.0])

        # Before ECM window
        result_before, valid_before = ecm.apply(measurement, t=5)
        assert valid_before is True
        np.testing.assert_array_equal(result_before, measurement)

        # After ECM window
        result_after, valid_after = ecm.apply(measurement, t=25)
        assert valid_after is True
        np.testing.assert_array_equal(result_after, measurement)

    def test_requires_bias_parameter(self):
        """Bias mode raises ValueError if bias is not provided."""
        with pytest.raises(ValueError, match="bias array is required"):
            ECMModel(
                mode="bias",
                ecm_start=10,
                ecm_end=20,
            )


class TestECMModelValidation:
    """Tests for ECMModel input validation."""

    def test_invalid_mode_raises_value_error(self):
        """Unknown mode string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown ECM mode"):
            ECMModel(
                mode="invalid",
                ecm_start=10,
                ecm_end=20,
            )

    def test_ecm_window_boundary_start_is_inclusive(self):
        """ECM effect is active at exactly ecm_start."""
        ecm = ECMModel(
            mode="dropout",
            ecm_start=10,
            ecm_end=20,
            dropout_prob=1.0,
        )
        measurement = np.array([100.0, 200.0])

        _, is_valid = ecm.apply(measurement, t=10)
        assert is_valid is False  # ecm_start is inclusive

    def test_ecm_window_boundary_end_is_exclusive(self):
        """ECM effect is NOT active at exactly ecm_end."""
        ecm = ECMModel(
            mode="dropout",
            ecm_start=10,
            ecm_end=20,
            dropout_prob=1.0,
        )
        measurement = np.array([100.0, 200.0])

        result, is_valid = ecm.apply(measurement, t=20)
        assert is_valid is True  # ecm_end is exclusive
        np.testing.assert_array_equal(result, measurement)
