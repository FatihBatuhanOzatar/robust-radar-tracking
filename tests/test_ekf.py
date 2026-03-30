"""Unit tests for Extended Kalman Filter (ekf.py).

Tests cover:
  - Correct position propagation during straight flight
  - Near-zero omega does not crash
  - Update step reduces covariance trace
  - step() improves position estimate over raw measurements
  - get_position() returns (2,) shape
  - init_state() initialises velocity/heading/omega to zero
  - Jacobian is finite (no NaN/Inf) for both branches
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Allow running the file directly: python tests/test_ekf.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from radarsim.tracker.ekf import ExtendedKalmanFilter


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DEFAULT_Q = {"q_pos": 1.0, "q_vel": 0.1, "q_theta": 0.01, "q_omega": 0.001}
NOISE_STD = 25.0
DT = 1.0


def _make_ekf(**kwargs) -> ExtendedKalmanFilter:
    """Create a default EKF with given overrides."""
    params = dict(dt=DT, q_params=DEFAULT_Q, r_x=NOISE_STD, r_y=NOISE_STD)
    params.update(kwargs)
    return ExtendedKalmanFilter(**params)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInitState:
    def test_position_set_from_measurement(self):
        """init_state() places position at the measurement."""
        ekf = _make_ekf()
        z = np.array([100.0, 200.0])
        ekf.init_state(z)
        state = ekf.get_state()
        assert state[0] == pytest.approx(100.0)
        assert state[1] == pytest.approx(200.0)

    def test_velocity_heading_omega_zero(self):
        """init_state() sets v, theta, omega to zero."""
        ekf = _make_ekf()
        ekf.init_state(np.array([0.0, 0.0]))
        state = ekf.get_state()
        assert state[2] == pytest.approx(0.0)   # v
        assert state[3] == pytest.approx(0.0)   # theta
        assert state[4] == pytest.approx(0.0)   # omega

    def test_get_position_returns_2d(self):
        """get_position() returns a (2,) array matching [x, y]."""
        ekf = _make_ekf()
        ekf.init_state(np.array([50.0, 80.0]))
        pos = ekf.get_position()
        assert pos.shape == (2,)
        assert pos[0] == pytest.approx(50.0)
        assert pos[1] == pytest.approx(80.0)

    def test_get_state_returns_5d(self):
        """get_state() returns a (5,) array."""
        ekf = _make_ekf()
        ekf.init_state(np.array([0.0, 0.0]))
        assert ekf.get_state().shape == (5,)

    def test_get_covariance_returns_5x5(self):
        """get_covariance() returns a (5, 5) matrix."""
        ekf = _make_ekf()
        ekf.init_state(np.array([0.0, 0.0]))
        P = ekf.get_covariance()
        assert P.shape == (5, 5)


# ---------------------------------------------------------------------------
# Predict step
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_straight_flight(self):
        """With omega=0 and heading=0, predict advances x by v*dt."""
        ekf = _make_ekf(dt=1.0)
        ekf.init_state(np.array([0.0, 0.0]))
        # Manually set state: heading=0, speed=50, omega=0
        ekf.x = np.array([0.0, 0.0, 50.0, 0.0, 0.0])
        ekf.predict()
        state = ekf.get_state()
        # With theta=0, omega=0: x_new = x + v*cos(0)*dt = 0 + 50*1*1 = 50
        assert state[0] == pytest.approx(50.0, abs=1e-8)
        assert state[1] == pytest.approx(0.0, abs=1e-8)

    def test_predict_north_flight(self):
        """With heading=pi/2, speed 50, omega=0: y advances, x stays."""
        ekf = _make_ekf(dt=1.0)
        ekf.init_state(np.array([0.0, 0.0]))
        ekf.x = np.array([0.0, 0.0, 50.0, np.pi / 2, 0.0])
        ekf.predict()
        state = ekf.get_state()
        assert state[0] == pytest.approx(0.0, abs=1e-8)
        assert state[1] == pytest.approx(50.0, abs=1e-8)

    def test_predict_increases_covariance(self):
        """Predict step must grow covariance (no measurement to reduce it)."""
        ekf = _make_ekf()
        ekf.init_state(np.array([0.0, 0.0]))
        p_before = np.trace(ekf.get_covariance())
        ekf.predict()
        p_after = np.trace(ekf.get_covariance())
        assert p_after > p_before

    def test_predict_heading_advances_by_omega_dt(self):
        """theta_new = theta + omega * dt after predict."""
        dt = 1.0
        omega = 0.1
        ekf = _make_ekf(dt=dt)
        ekf.init_state(np.array([0.0, 0.0]))
        theta0 = 0.3
        ekf.x = np.array([0.0, 0.0, 30.0, theta0, omega])
        ekf.predict()
        state = ekf.get_state()
        assert state[3] == pytest.approx(theta0 + omega * dt, abs=1e-8)

    def test_predict_near_zero_omega_no_crash(self):
        """Very small omega (< threshold) must not raise ZeroDivisionError."""
        ekf = _make_ekf()
        ekf.init_state(np.array([0.0, 0.0]))
        ekf.x = np.array([0.0, 0.0, 50.0, 0.0, 1e-12])
        ekf.predict()  # should not raise

    def test_predict_jacobian_finite(self):
        """Jacobian must be all finite — no NaN or Inf — in both branches."""
        ekf = _make_ekf()
        ekf.init_state(np.array([0.0, 0.0]))

        # Non-zero omega branch
        ekf.x = np.array([100.0, 200.0, 50.0, 0.5, 0.05])
        F1 = ekf._compute_jacobian(ekf.x)
        assert np.all(np.isfinite(F1)), "Jacobian has non-finite values (CT branch)"

        # Near-zero omega branch
        ekf.x = np.array([100.0, 200.0, 50.0, 0.5, 1e-10])
        F2 = ekf._compute_jacobian(ekf.x)
        assert np.all(np.isfinite(F2)), "Jacobian has non-finite values (CV branch)"


# ---------------------------------------------------------------------------
# Update step
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_update_reduces_covariance(self):
        """Incorporating a measurement must strictly reduce covariance trace."""
        ekf = _make_ekf()
        ekf.init_state(np.array([0.0, 0.0]))
        ekf.predict()
        p_before = np.trace(ekf.get_covariance())
        ekf.update(np.array([0.5, 0.5]))
        p_after = np.trace(ekf.get_covariance())
        assert p_after < p_before

    def test_update_pulls_state_toward_measurement(self):
        """After update, state should be pulled toward the measurement."""
        ekf = _make_ekf()
        ekf.init_state(np.array([0.0, 0.0]))
        # Manually place predicted state at origin, then measure at (100, 100)
        ekf.x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        ekf.P = np.eye(5) * 500.0
        z = np.array([100.0, 100.0])
        ekf.update(z)
        state = ekf.get_state()
        # Update must pull x, y toward 100 (not exactly, but directionally)
        assert state[0] > 0.0
        assert state[1] > 0.0


# ---------------------------------------------------------------------------
# Full step cycle
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_improves_estimate_over_raw_measurement(self):
        """EKF RMSE over a maneuver sequence must beat raw measurement RMSE."""
        from radarsim.sim.target import Target
        from radarsim.sim.radar import Radar
        from radarsim.analysis.metrics import rmse

        rng = np.random.default_rng(42)
        dt = 1.0
        n_steps = 50
        noise_std = 25.0

        # CT target: speed~54 m/s, omega=0.05 rad/s
        target = Target(x0=0.0, y0=0.0, vx0=50.0, vy0=20.0,
                        model="ct", turn_rate=0.05)
        radar = Radar(noise_std_x=noise_std, noise_std_y=noise_std, seed=42)

        true_traj = np.zeros((n_steps, 4))
        estimated_traj = np.zeros((n_steps, 2))

        ekf = _make_ekf(q_params={"q_pos": 1.0, "q_vel": 0.5,
                                   "q_theta": 0.05, "q_omega": 0.005})
        measurements = np.zeros((n_steps, 2))

        for i in range(n_steps):
            true_traj[i] = target.step(dt)
            z = radar.measure(true_traj[i])
            measurements[i] = z
            if i == 0:
                ekf.init_state(z)
                estimated_traj[i] = ekf.get_position()
            else:
                ekf.step(z)
                estimated_traj[i] = ekf.get_position()

        # Build (n, 4) arrays for rmse() — velocity columns padded with zeros
        true_pos_4 = np.column_stack([true_traj[:, :2],
                                       np.zeros((n_steps, 2))])
        est_4 = np.column_stack([estimated_traj, np.zeros((n_steps, 2))])
        raw_4 = np.column_stack([measurements, np.zeros((n_steps, 2))])

        ekf_rmse = rmse(true_pos_4, est_4)
        raw_rmse = rmse(true_pos_4, raw_4)

        assert ekf_rmse < raw_rmse, (
            f"EKF RMSE ({ekf_rmse:.2f}) should be < raw RMSE ({raw_rmse:.2f})"
        )

    def test_step_returns_5d_state(self):
        """step() must return a (5,) array."""
        ekf = _make_ekf()
        ekf.init_state(np.array([0.0, 0.0]))
        result = ekf.step(np.array([1.0, 1.0]))
        assert result.shape == (5,)


# ---------------------------------------------------------------------------
# Q parameter parsing
# ---------------------------------------------------------------------------

class TestQParsing:
    def test_q_dict_creates_correct_diagonal(self):
        """Dict q_params populates Q diagonal correctly."""
        ekf = ExtendedKalmanFilter(
            dt=1.0,
            q_params={"q_pos": 2.0, "q_vel": 0.5,
                      "q_theta": 0.02, "q_omega": 0.003},
            r_x=25.0,
            r_y=25.0,
        )
        diag = np.diag(ekf.Q)
        assert diag[0] == pytest.approx(2.0)
        assert diag[1] == pytest.approx(2.0)
        assert diag[2] == pytest.approx(0.5)
        assert diag[3] == pytest.approx(0.02)
        assert diag[4] == pytest.approx(0.003)

    def test_q_sequence_creates_correct_diagonal(self):
        """5-element sequence q_params populates Q diagonal directly."""
        q_vals = [1.0, 2.0, 0.3, 0.04, 0.005]
        ekf = ExtendedKalmanFilter(
            dt=1.0, q_params=q_vals, r_x=25.0, r_y=25.0
        )
        diag = np.diag(ekf.Q)
        for i, expected in enumerate(q_vals):
            assert diag[i] == pytest.approx(expected)

    def test_q_sequence_wrong_length_raises(self):
        """Sequence with != 5 elements must raise ValueError."""
        with pytest.raises(ValueError, match="5 elements"):
            ExtendedKalmanFilter(
                dt=1.0, q_params=[1.0, 2.0, 3.0], r_x=25.0, r_y=25.0
            )

    def test_q_dict_defaults_used_on_missing_keys(self):
        """Dict with missing keys falls back to documented defaults."""
        ekf = ExtendedKalmanFilter(
            dt=1.0, q_params={}, r_x=25.0, r_y=25.0
        )
        # Defaults: q_pos=1.0, q_vel=0.1, q_theta=0.01, q_omega=0.001
        diag = np.diag(ekf.Q)
        assert diag[0] == pytest.approx(1.0)
        assert diag[2] == pytest.approx(0.1)
        assert diag[3] == pytest.approx(0.01)
        assert diag[4] == pytest.approx(0.001)


if __name__ == "__main__":
    # Allow direct execution: python tests/test_ekf.py
    pytest.main([__file__, "-v"])
