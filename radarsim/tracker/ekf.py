"""Extended Kalman Filter for coordinated-turn target tracking.

The standard Kalman Filter assumes constant velocity (linear model).
When a target maneuvers, the CV model is wrong and tracking error
spikes. This EKF uses a Coordinated Turn (CT) nonlinear motion model,
linearised at each step via a Jacobian matrix.

State vector: [x, y, v, theta, omega], shape (5,).
  - x, y   : position (meters)
  - v      : scalar speed (m/s)
  - theta  : heading angle (radians)
  - omega  : turn rate (rad/s) — estimated from measurements

Measurement vector: [x, y], shape (2,). Same as standard KF.
"""

from typing import Optional, Union

import numpy as np


class ExtendedKalmanFilter:
    """Extended Kalman Filter with coordinated-turn motion model.

    Unlike the standard KF which fixes a linear state-transition matrix F,
    the EKF re-computes a Jacobian linearisation F(x) at every predict step
    based on the current state estimate. This lets the filter correctly
    handle curved trajectories without needing to know the turn rate in
    advance — omega is part of the state and is estimated from data.

    State vector: [x, y, v, theta, omega], shape (5,).
    Measurement vector: [x, y], shape (2,).

    Args:
        dt: Time step duration (seconds).
        q_params: Process noise parameters for the 5 state components.
            Accepts a dict with keys ``q_pos``, ``q_vel``, ``q_theta``,
            ``q_omega``; or a sequence of 5 values
            ``[q_x, q_y, q_v, q_theta, q_omega]``.
        r_x: Measurement noise standard deviation in x (meters).
            Squared internally: ``R[0,0] = r_x²``.
        r_y: Measurement noise standard deviation in y (meters).
            Squared internally: ``R[1,1] = r_y²``.

    Attributes:
        x: Current state estimate [x, y, v, theta, omega], shape (5,).
        P: Current state covariance matrix, shape (5, 5).
        H: Measurement matrix, shape (2, 5). Fixed (linear observation).
        Q: Process noise covariance, shape (5, 5). Diagonal.
        R: Measurement noise covariance, shape (2, 2).
        dt: Time step duration used for all predictions.
    """

    # Near-zero omega threshold: below this magnitude the CT equations
    # degenerate (division by omega). We fall back to a CV approximation.
    _OMEGA_THRESHOLD: float = 1e-6

    def __init__(
        self,
        dt: float,
        q_params: Union[dict, tuple, list, np.ndarray],
        r_x: float,
        r_y: float,
    ) -> None:
        self.dt: float = dt

        # --- Parse Q parameters ---
        q_diag = self._parse_q_params(q_params)

        # Process noise covariance — diagonal (empirically tuned for CT model)
        self.Q: np.ndarray = np.diag(q_diag)

        # Measurement noise covariance R (2×2)
        self.R: np.ndarray = np.array([
            [r_x ** 2, 0.0],
            [0.0,      r_y ** 2],
        ])

        # Measurement matrix H (2×5) — linear: we observe [x, y] directly
        self.H: np.ndarray = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
        ])

        # State and covariance — properly initialised by init_state()
        self.x: np.ndarray = np.zeros(5)
        self.P: np.ndarray = np.eye(5) * 500.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init_state(self, z: np.ndarray) -> None:
        """Initialise filter state from the first measurement.

        Sets position from measurement; speed, heading, and turn rate
        default to zero (unknown). Covariance is set large for the
        unknowns and equal to measurement noise for position.

        Args:
            z: First measurement [x, y], shape (2,).
        """
        self.x = np.array([z[0], z[1], 0.0, 0.0, 0.0], dtype=float)
        self.P = np.diag([
            self.R[0, 0],   # x  — position uncertainty from measurement noise
            self.R[1, 1],   # y
            500.0,           # v  — completely unknown initial speed
            np.pi ** 2,      # theta — could be anything in [-π, π]
            0.1,             # omega — likely small, but unknown sign/magnitude
        ])

    def predict(self) -> np.ndarray:
        """Run the EKF prediction step.

        Propagates state through the nonlinear CT motion model, then
        advances covariance using the Jacobian linearisation:
            x_pred = f(x)
            P_pred = F_jac @ P @ F_jac.T + Q

        Returns:
            Predicted state estimate [x, y, v, theta, omega], shape (5,).
        """
        F_jac = self._compute_jacobian(self.x)
        self.x = self._compute_f(self.x)
        self.P = F_jac @ self.P @ F_jac.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        """Run the measurement update step.

        The measurement model H is linear (we observe position directly),
        so this is identical to the standard KF update. Uses the Joseph
        form for numerical stability.

        Args:
            z: Measurement vector [x, y], shape (2,).

        Returns:
            Updated state estimate [x, y, v, theta, omega], shape (5,).
        """
        # Innovation (measurement residual)
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update — Joseph form for numerical stability
        I_KH = np.eye(5) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return self.x.copy()

    def step(self, z: np.ndarray) -> np.ndarray:
        """Run one full predict-update cycle.

        Args:
            z: Measurement vector [x, y], shape (2,).

        Returns:
            Updated state estimate [x, y, v, theta, omega], shape (5,).
        """
        self.predict()
        return self.update(z)

    def get_state(self) -> np.ndarray:
        """Return the current state estimate.

        Returns:
            State vector [x, y, v, theta, omega], shape (5,).
        """
        return self.x.copy()

    def get_position(self) -> np.ndarray:
        """Return the current position estimate.

        Convenience method for compatibility with metrics that expect
        a (2,) position vector (matching the standard KF interface).

        Returns:
            Position vector [x, y], shape (2,).
        """
        return self.x[:2].copy()

    def get_covariance(self) -> np.ndarray:
        """Return the current state covariance matrix.

        Returns:
            Covariance matrix P, shape (5, 5).
        """
        return self.P.copy()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_q_params(
        self,
        q_params: Union[dict, tuple, list, np.ndarray],
    ) -> np.ndarray:
        """Parse Q parameters into a 5-element diagonal array.

        Accepts either a dict with named keys or a sequence of 5 values.

        Dict keys (when passing a dict):
            ``q_pos``  : position noise (applied to x and y, default 1.0)
            ``q_vel``  : speed noise (default 0.1)
            ``q_theta``: heading noise (default 0.01)
            ``q_omega``: turn-rate noise (default 0.001)

        Sequence format: ``[q_x, q_y, q_v, q_theta, q_omega]`` (5 values).

        Args:
            q_params: Process noise specification.

        Returns:
            Q diagonal array, shape (5,).

        Raises:
            TypeError: If q_params is not a dict or sequence.
            ValueError: If a sequence does not have exactly 5 elements.
        """
        if isinstance(q_params, dict):
            q_pos = float(q_params.get("q_pos", 1.0))
            q_vel = float(q_params.get("q_vel", 0.1))
            q_theta = float(q_params.get("q_theta", 0.01))
            q_omega = float(q_params.get("q_omega", 0.001))
            return np.array([q_pos, q_pos, q_vel, q_theta, q_omega])
        else:
            arr = np.asarray(q_params, dtype=float)
            if arr.shape != (5,):
                raise ValueError(
                    f"q_params sequence must have exactly 5 elements, "
                    f"got {arr.shape}."
                )
            return arr

    def _compute_f(self, x: np.ndarray) -> np.ndarray:
        """Nonlinear CT state transition function f(x).

        Propagates state one dt step using the coordinated-turn equations.
        When |omega| < _OMEGA_THRESHOLD, falls back to the constant-velocity
        approximation to avoid division by zero.

        Args:
            x: Current state [x_pos, y_pos, v, theta, omega], shape (5,).

        Returns:
            Predicted state after dt seconds, shape (5,).
        """
        x_pos, y_pos, v, theta, omega = x
        dt = self.dt

        if abs(omega) < self._OMEGA_THRESHOLD:
            # CV approximation: straight-line at current heading
            x_new = x_pos + v * np.cos(theta) * dt
            y_new = y_pos + v * np.sin(theta) * dt
        else:
            theta_new = theta + omega * dt
            # Circular arc integral (same formula as Target._step_ct)
            x_new = x_pos + (v / omega) * (np.sin(theta_new) - np.sin(theta))
            y_new = y_pos + (v / omega) * (np.cos(theta) - np.cos(theta_new))

        theta_new = theta + omega * dt  # always update heading
        return np.array([x_new, y_new, v, theta_new, omega])

    def _compute_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Compute the 5×5 Jacobian of f with respect to state x.

        The Jacobian F_jac[i, j] = ∂fᵢ/∂xⱼ evaluated at the current
        state estimate. This linearises the nonlinear CT dynamics around x.

        State indexing: 0=x_pos, 1=y_pos, 2=v, 3=theta, 4=omega.

        Args:
            x: Current state [x_pos, y_pos, v, theta, omega], shape (5,).

        Returns:
            Jacobian matrix, shape (5, 5).
        """
        _, _, v, theta, omega = x
        dt = self.dt

        F = np.eye(5)  # start from identity (fᵢ/fᵢ = 1 for diagonal)

        if abs(omega) < self._OMEGA_THRESHOLD:
            # --- CV approximation Jacobian ---
            # f1 = x + v*cos(theta)*dt
            F[0, 2] = np.cos(theta) * dt          # ∂f1/∂v
            F[0, 3] = -v * np.sin(theta) * dt     # ∂f1/∂theta
            F[0, 4] = 0.0                          # ∂f1/∂omega (approx 0)
            # f2 = y + v*sin(theta)*dt
            F[1, 2] = np.sin(theta) * dt           # ∂f2/∂v
            F[1, 3] = v * np.cos(theta) * dt      # ∂f2/∂theta
            F[1, 4] = 0.0                          # ∂f2/∂omega (approx 0)
            # f4 = theta + omega*dt  →  ∂f4/∂omega = dt
            F[3, 4] = dt
        else:
            # --- Full CT Jacobian ---
            theta_new = theta + omega * dt
            sin_t = np.sin(theta)
            cos_t = np.cos(theta)
            sin_tn = np.sin(theta_new)
            cos_tn = np.cos(theta_new)

            # Row 0: ∂f1/∂(x,y,v,theta,omega)
            # f1 = x + (v/omega)*(sin(theta_new) - sin(theta))
            F[0, 2] = (sin_tn - sin_t) / omega        # ∂f1/∂v
            F[0, 3] = (v / omega) * (cos_tn - cos_t)  # ∂f1/∂theta
            # ∂f1/∂omega: d/dω [(v/ω)(sin(θ+ωdt)-sinθ)]
            #   = v * [ω * dt * cos(θ_new) - sin(θ_new) + sin(θ)] / ω²
            F[0, 4] = v * (omega * dt * cos_tn - sin_tn + sin_t) / (omega ** 2)

            # Row 1: ∂f2/∂(x,y,v,theta,omega)
            # f2 = y + (v/omega)*(cos(theta) - cos(theta_new))
            F[1, 2] = (cos_t - cos_tn) / omega         # ∂f2/∂v
            F[1, 3] = (v / omega) * (sin_tn - sin_t)   # ∂f2/∂theta
            # ∂f2/∂omega: d/dω [(v/ω)(cosθ - cos(θ+ωdt))]
            #   = v * [ω * dt * sin(θ_new) + cos(θ_new) - cos(θ)] / ω²
            F[1, 4] = v * (omega * dt * sin_tn + cos_tn - cos_t) / (omega ** 2)

            # Row 3: ∂f4/∂omega = dt  (f4 = theta + omega*dt)
            F[3, 4] = dt

        return F
