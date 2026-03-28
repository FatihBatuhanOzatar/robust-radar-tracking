"""Radar measurement simulator for radar tracking."""

import numpy as np


class Radar:
    """Simulates noisy radar measurements from true target positions.

    Adds independent Gaussian noise to the x and y position components
    of the true target state. Does not modify velocity components —
    the radar only observes position.

    Args:
        noise_std_x: Standard deviation of measurement noise in x (meters).
        noise_std_y: Standard deviation of measurement noise in y (meters).
        seed: Optional RNG seed for reproducible noise generation.

    Attributes:
        noise_std_x: Noise standard deviation in x.
        noise_std_y: Noise standard deviation in y.
    """

    def __init__(
        self,
        noise_std_x: float,
        noise_std_y: float,
        *,
        seed: int | None = None,
    ) -> None:
        self.noise_std_x: float = noise_std_x
        self.noise_std_y: float = noise_std_y
        self._rng: np.random.Generator = np.random.default_rng(seed)

    def measure(self, true_state: np.ndarray) -> np.ndarray:
        """Generate a single noisy measurement from a true state.

        Extracts position [x, y] from the state vector and adds
        independent Gaussian noise to each component.

        Args:
            true_state: True target state [x, y, vx, vy], shape (4,).

        Returns:
            Noisy measurement [x, y], shape (2,).
        """
        position = true_state[:2]
        noise = np.array([
            self._rng.normal(0.0, self.noise_std_x),
            self._rng.normal(0.0, self.noise_std_y),
        ])
        return position + noise

    def measure_batch(self, true_states: np.ndarray) -> np.ndarray:
        """Generate noisy measurements for a full trajectory.

        Vectorized batch measurement — processes all time steps at once.

        Args:
            true_states: True trajectory, shape (n_steps, 4).

        Returns:
            Noisy measurements, shape (n_steps, 2).
        """
        n_steps = true_states.shape[0]
        positions = true_states[:, :2]
        noise = np.column_stack([
            self._rng.normal(0.0, self.noise_std_x, size=n_steps),
            self._rng.normal(0.0, self.noise_std_y, size=n_steps),
        ])
        return positions + noise
