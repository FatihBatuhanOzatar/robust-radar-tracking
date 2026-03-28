"""Standard Kalman Filter for constant-velocity target tracking."""

import numpy as np


class KalmanFilter:
    """Constant-velocity Kalman Filter for 2D target tracking.

    State vector: [x, y, vx, vy], shape (4,).
    Measurement vector: [x, y], shape (2,).

    Uses a physically-derived process noise covariance (Q) based on
    acceleration uncertainty, not arbitrary diagonal values.

    Args:
        dt: Time step duration (seconds).
        q: Process noise intensity — acceleration variance (m²/s⁴).
        r_x: Measurement noise standard deviation in x (meters).
            Same units as ``Radar.noise_std_x``. Squared internally
            to build the R covariance matrix: ``R = diag(r_x², r_y²)``.
        r_y: Measurement noise standard deviation in y (meters).
            Same units as ``Radar.noise_std_y``.

    Attributes:
        x: Current state estimate [x, y, vx, vy], shape (4,).
        P: Current state covariance matrix, shape (4, 4).
        F: State transition matrix, shape (4, 4).
        H: Measurement matrix, shape (2, 4).
        Q: Process noise covariance, shape (4, 4).
        R: Measurement noise covariance, shape (2, 2).
    """

    def __init__(
        self,
        dt: float,
        q: float,
        r_x: float,
        r_y: float,
    ) -> None:
        self.dt: float = dt

        # State transition matrix (constant velocity)
        self.F: np.ndarray = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        # Measurement matrix (observe position only)
        self.H: np.ndarray = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])

        # Process noise covariance — physically derived from acceleration
        # uncertainty. Models unknown acceleration as white noise with
        # spectral density q. See Bar-Shalom et al., "Estimation with
        # Applications to Tracking and Navigation", Chapter 6.
        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4
        self.Q: np.ndarray = q * np.array([
            [dt4 / 4, 0.0,     dt3 / 2, 0.0    ],
            [0.0,     dt4 / 4, 0.0,     dt3 / 2],
            [dt3 / 2, 0.0,     dt2,     0.0    ],
            [0.0,     dt3 / 2, 0.0,     dt2    ],
        ])

        # Measurement noise covariance
        self.R: np.ndarray = np.array([
            [r_x ** 2, 0.0     ],
            [0.0,      r_y ** 2],
        ])

        # State and covariance — initialized by init_state()
        self.x: np.ndarray = np.zeros(4)
        self.P: np.ndarray = np.eye(4) * 500.0

    def init_state(self, z: np.ndarray) -> None:
        """Initialize filter state from the first measurement.

        Sets position to the measurement and velocity to zero.
        Covariance is set large to reflect high uncertainty in
        the initial velocity estimate.

        Args:
            z: First measurement [x, y], shape (2,).
        """
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.diag([
            self.R[0, 0],   # position uncertainty = measurement noise
            self.R[1, 1],
            500.0,           # velocity uncertainty — large, unknown
            500.0,
        ])

    def predict(self) -> np.ndarray:
        """Run the prediction step.

        Projects state and covariance forward one time step using
        the constant-velocity motion model.

        Returns:
            Predicted state estimate [x, y, vx, vy], shape (4,).
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        """Run the measurement update step.

        Incorporates a new measurement to correct the predicted state.

        Args:
            z: Measurement vector [x, y], shape (2,).

        Returns:
            Updated state estimate [x, y, vx, vy], shape (4,).
        """
        # Innovation (measurement residual)
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return self.x.copy()

    def step(self, z: np.ndarray) -> np.ndarray:
        """Run one full predict-update cycle.

        Args:
            z: Measurement vector [x, y], shape (2,).

        Returns:
            Updated state estimate [x, y, vx, vy], shape (4,).
        """
        self.predict()
        return self.update(z)

    def step_no_measurement(self) -> np.ndarray:
        """Run prediction only — no measurement available.

        Used during measurement dropout (e.g., ECM jamming) when
        the radar cannot provide a valid measurement.

        Returns:
            Predicted state estimate [x, y, vx, vy], shape (4,).
        """
        return self.predict()

    def get_state(self) -> np.ndarray:
        """Return the current state estimate.

        Returns:
            State vector [x, y, vx, vy], shape (4,).
        """
        return self.x.copy()

    def get_covariance(self) -> np.ndarray:
        """Return the current state covariance matrix.

        Returns:
            Covariance matrix P, shape (4, 4).
        """
        return self.P.copy()
