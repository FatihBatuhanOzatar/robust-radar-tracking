"""Target motion models for radar tracking simulation."""

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
        model: Motion model identifier. Currently supported: "cv"
            (constant velocity). Future: "ct", "random".

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
    ) -> None:
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown motion model '{model}'. "
                f"Supported: {self.SUPPORTED_MODELS}"
            )

        self.state: np.ndarray = np.array([x0, y0, vx0, vy0], dtype=float)
        self.model: str = model
        self._initial_state: np.ndarray = self.state.copy()

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
            raise NotImplementedError(
                "Coordinated turn model not yet implemented (Phase 2)."
            )
        elif self.model == "random":
            raise NotImplementedError(
                "Random maneuver model not yet implemented (Phase 2)."
            )

        return self.state.copy()

    def get_trajectory(self, dt: float, n_steps: int) -> np.ndarray:
        """Generate a full trajectory from the initial state.

        Resets to the initial state, runs n_steps forward, then restores
        the current state. This makes the method non-destructive.

        Args:
            dt: Time step duration (seconds).
            n_steps: Number of time steps to simulate.

        Returns:
            Trajectory array of shape (n_steps, 4), where each row is
            [x, y, vx, vy] at that time step.
        """
        saved_state = self.state.copy()
        self.state = self._initial_state.copy()

        trajectory = np.zeros((n_steps, 4))
        for i in range(n_steps):
            self.step(dt)
            trajectory[i] = self.state

        self.state = saved_state
        return trajectory

    def _step_cv(self, dt: float) -> None:
        """Constant velocity motion update.

        Updates position using current velocity. Velocity remains unchanged.

        Args:
            dt: Time step duration (seconds).
        """
        self.state[0] += self.state[2] * dt  # x += vx * dt
        self.state[1] += self.state[3] * dt  # y += vy * dt
