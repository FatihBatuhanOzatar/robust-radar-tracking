"""Target motion models for radar tracking simulation."""

from typing import Optional

import numpy as np


class Target:
    """Represents a single moving target with configurable motion model.

    The target maintains its true state [x, y, vx, vy] and advances it
    according to the selected motion model. No noise is added — this
    produces pure ground truth trajectories.

    Args:
        x0: Initial x position (meters).
        y0: Initial y position (meters).
        vx0: Initial x velocity (meters/second).
        vy0: Initial y velocity (meters/second).
        model: Motion model identifier. Supported: "cv" (constant
            velocity), "ct" (coordinated turn), "random" (random
            acceleration perturbations).
        turn_rate: Turn rate in rad/s (omega). Required when model="ct".
            If abs(turn_rate) < 1e-10, the CT model delegates to CV
            (zero turn rate = straight line).
        accel_std: Standard deviation of random acceleration
            perturbations in m/s². Required when model="random".
        seed: Optional RNG seed for reproducible random trajectories.

    Attributes:
        state: Current state vector [x, y, vx, vy], shape (4,).
        model: Motion model identifier string.
    """

    SUPPORTED_MODELS = ("cv", "ct", "random")

    def __init__(
        self,
        x0: float,
        y0: float,
        vx0: float,
        vy0: float,
        model: str = "cv",
        turn_rate: Optional[float] = None,
        accel_std: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown motion model '{model}'. "
                f"Supported: {self.SUPPORTED_MODELS}"
            )

        if model == "ct" and turn_rate is None:
            raise ValueError("turn_rate is required when model='ct'.")

        if model == "random" and accel_std is None:
            raise ValueError("accel_std is required when model='random'.")

        self.state: np.ndarray = np.array([x0, y0, vx0, vy0], dtype=float)
        self.model: str = model
        self._initial_state: np.ndarray = self.state.copy()

        # CT-specific internal state
        self._turn_rate: Optional[float] = turn_rate
        self._heading: float = float(np.arctan2(vy0, vx0))
        self._initial_heading: float = self._heading

        # Random-specific internal state
        self._accel_std: Optional[float] = accel_std
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._initial_rng_state = self._rng.bit_generator.state

    def step(self, dt: float) -> np.ndarray:
        """Advance the target state by one time step.

        Args:
            dt: Time step duration (seconds).

        Returns:
            Updated state vector [x, y, vx, vy], shape (4,).

        Raises:
            NotImplementedError: If the motion model is not yet implemented.
        """
        if self.model == "cv":
            self._step_cv(dt)
        elif self.model == "ct":
            self._step_ct(dt)
        elif self.model == "random":
            self._step_random(dt)

        return self.state.copy()

    def get_trajectory(self, dt: float, n_steps: int) -> np.ndarray:
        """Generate a full trajectory from the initial state.

        Resets to the initial state (including internal heading for CT
        model and RNG state for random model), runs n_steps forward,
        then restores all state. This makes the method non-destructive.

        Args:
            dt: Time step duration (seconds).
            n_steps: Number of time steps to simulate.

        Returns:
            Trajectory array of shape (n_steps, 4), where each row is
            [x, y, vx, vy] at that time step.
        """
        saved_state = self.state.copy()
        saved_heading = self._heading
        saved_rng_state = self._rng.bit_generator.state
        self.state = self._initial_state.copy()
        self._heading = self._initial_heading
        self._rng.bit_generator.state = self._initial_rng_state

        trajectory = np.zeros((n_steps, 4))
        for i in range(n_steps):
            self.step(dt)
            trajectory[i] = self.state

        self.state = saved_state
        self._heading = saved_heading
        self._rng.bit_generator.state = saved_rng_state
        return trajectory

    def _step_cv(self, dt: float) -> None:
        """Constant velocity motion update.

        Updates position using current velocity. Velocity remains unchanged.

        Args:
            dt: Time step duration (seconds).
        """
        self.state[0] += self.state[2] * dt  # x += vx * dt
        self.state[1] += self.state[3] * dt  # y += vy * dt

    def _step_ct(self, dt: float) -> None:
        """Coordinated turn motion update.

        Moves the target along a circular arc at constant speed. The
        turn rate (omega) determines the curvature. Heading is tracked
        internally; the exposed state stays [x, y, vx, vy].

        If the turn rate is near zero (abs < 1e-10), delegates to
        constant velocity to avoid division by zero.

        Args:
            dt: Time step duration (seconds).
        """
        omega = self._turn_rate
        assert omega is not None  # guaranteed by __init__ validation

        # Near-zero turn rate: straight-line motion
        if abs(omega) < 1e-10:
            self._step_cv(dt)
            return

        speed = np.sqrt(self.state[2] ** 2 + self.state[3] ** 2)
        theta = self._heading
        theta_new = theta + omega * dt

        # Position update (circular arc integral)
        self.state[0] += (speed / omega) * (np.sin(theta_new) - np.sin(theta))
        self.state[1] += (speed / omega) * (np.cos(theta) - np.cos(theta_new))

        # Velocity update (tangent to circle)
        self.state[2] = speed * np.cos(theta_new)
        self.state[3] = speed * np.sin(theta_new)

        self._heading = theta_new

    def _step_random(self, dt: float) -> None:
        """Random maneuver motion update.

        Applies random acceleration perturbations drawn from a normal
        distribution N(0, accel_std^2) independently in x and y.
        Position update uses the full kinematic equation.

        Args:
            dt: Time step duration (seconds).
        """
        accel_std = self._accel_std
        assert accel_std is not None  # guaranteed by __init__ validation

        ax = self._rng.normal(0.0, accel_std)
        ay = self._rng.normal(0.0, accel_std)

        # Position update (kinematic: x += v*dt + 0.5*a*dt²)
        self.state[0] += self.state[2] * dt + 0.5 * ax * dt ** 2
        self.state[1] += self.state[3] * dt + 0.5 * ay * dt ** 2

        # Velocity update
        self.state[2] += ax * dt
        self.state[3] += ay * dt
