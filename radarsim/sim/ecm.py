"""Electronic countermeasure models for radar tracking simulation."""

from typing import Optional

import numpy as np


class ECMModel:
    """Configurable electronic countermeasure effects.

    Wraps around radar measurements to simulate adversarial sensor
    degradation during a specified time window. Outside the ECM window,
    measurements pass through unchanged.

    Three ECM modes are supported:

    - ``noise_spike``: Amplifies measurement noise by adding extra
      Gaussian noise scaled by ``(noise_multiplier - 1) * noise_std``.
      The radar already added 1x noise, so the total effective noise
      becomes ``noise_multiplier * noise_std``.
    - ``dropout``: Drops measurements with probability ``dropout_prob``.
      When dropped, returns ``(None, False)`` so the tracker can
      fall back to predict-only mode.
    - ``bias``: Adds a fixed systematic offset to measurements. This
      is the hardest case for the KF because it cannot distinguish
      bias from true target motion.

    Args:
        mode: ECM mode identifier. Supported: ``"noise_spike"``,
            ``"dropout"``, ``"bias"``.
        ecm_start: Time step when ECM begins (inclusive).
        ecm_end: Time step when ECM ends (exclusive).
        noise_multiplier: Noise amplification factor for noise_spike
            mode. Must be >= 1.0. Default: 1.0.
        noise_std: Radar noise standard deviation (meters). Required
            for noise_spike mode so that extra noise can be scaled
            correctly.
        dropout_prob: Probability of dropping a measurement in dropout
            mode. 1.0 means always drop. Default: 1.0.
        bias: Systematic offset [bx, by] added during bias mode,
            shape (2,). Required when mode="bias".
        seed: Optional RNG seed for reproducible dropout decisions
            and noise_spike noise generation.

    Attributes:
        mode: ECM mode identifier string.
        ecm_start: Start of ECM window (inclusive).
        ecm_end: End of ECM window (exclusive).
    """

    SUPPORTED_MODES = ("noise_spike", "dropout", "bias")

    def __init__(
        self,
        mode: str,
        ecm_start: int,
        ecm_end: int,
        noise_multiplier: float = 1.0,
        noise_std: float = 0.0,
        dropout_prob: float = 1.0,
        bias: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> None:
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unknown ECM mode '{mode}'. "
                f"Supported: {self.SUPPORTED_MODES}"
            )

        if mode == "bias" and bias is None:
            raise ValueError("bias array is required when mode='bias'.")

        if mode == "noise_spike" and noise_std <= 0.0:
            raise ValueError(
                "noise_std must be positive when mode='noise_spike'."
            )

        self.mode: str = mode
        self.ecm_start: int = ecm_start
        self.ecm_end: int = ecm_end
        self._noise_multiplier: float = noise_multiplier
        self._noise_std: float = noise_std
        self._dropout_prob: float = dropout_prob
        self._bias: Optional[np.ndarray] = (
            np.asarray(bias, dtype=float) if bias is not None else None
        )
        self._rng: np.random.Generator = np.random.default_rng(seed)

    def apply(
        self,
        measurement: np.ndarray,
        t: int,
    ) -> tuple[np.ndarray | None, bool]:
        """Apply ECM effect to a radar measurement.

        Outside the ECM window, measurements pass through unchanged.
        During the ECM window, the effect depends on the configured mode.

        Args:
            measurement: Radar measurement [x, y], shape (2,).
            t: Current time step index.

        Returns:
            Tuple of (degraded_measurement_or_None, is_valid).
            When is_valid is False, the measurement was dropped and
            the tracker should use predict-only mode.
        """
        # Outside ECM window — pass through unchanged
        if t < self.ecm_start or t >= self.ecm_end:
            return measurement.copy(), True

        # Inside ECM window — apply mode-specific degradation
        if self.mode == "noise_spike":
            return self._apply_noise_spike(measurement)
        elif self.mode == "dropout":
            return self._apply_dropout(measurement)
        elif self.mode == "bias":
            return self._apply_bias(measurement)

        # Should never reach here due to __init__ validation
        raise RuntimeError(f"Unhandled ECM mode: {self.mode}")  # pragma: no cover

    def _apply_noise_spike(
        self,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Add extra noise to simulate amplified radar noise.

        The radar already added 1x noise_std of noise. This adds
        (noise_multiplier - 1) * noise_std more, so the total
        effective noise is noise_multiplier * noise_std.

        Args:
            measurement: Radar measurement [x, y], shape (2,).

        Returns:
            Tuple of (noisier_measurement, True).
        """
        extra_std = self._noise_std * (self._noise_multiplier - 1.0)
        extra_noise = self._rng.normal(0.0, extra_std, size=2)
        return measurement + extra_noise, True

    def _apply_dropout(
        self,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray | None, bool]:
        """Drop measurement with configured probability.

        Args:
            measurement: Radar measurement [x, y], shape (2,).

        Returns:
            Tuple of (None, False) if dropped, or
            (measurement, True) if kept.
        """
        if self._rng.random() < self._dropout_prob:
            return None, False
        return measurement.copy(), True

    def _apply_bias(
        self,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Add systematic offset to measurement.

        Args:
            measurement: Radar measurement [x, y], shape (2,).

        Returns:
            Tuple of (biased_measurement, True).
        """
        assert self._bias is not None  # guaranteed by __init__ validation
        return measurement + self._bias, True
