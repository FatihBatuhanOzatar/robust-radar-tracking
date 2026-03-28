"""Unit tests for the constant-velocity Kalman Filter."""

import numpy as np
import pytest

from radarsim.tracker.kf import KalmanFilter
from radarsim.sim.target import Target
from radarsim.sim.radar import Radar
from radarsim.analysis.metrics import rmse


def test_predict_constant_velocity():
    """Check predict-step correctness for constant velocity model."""
    dt = 1.0
    kf = KalmanFilter(dt=dt, q=0.5, r_x=10.0, r_y=10.0)
    
    # Force initialize the state to a known value
    # x, y, vx, vy
    kf.x = np.array([10.0, 20.0, 5.0, -2.0])
    
    predicted_state = kf.predict()
    
    # Expected position: x + vx*dt, y + vy*dt
    expected_x = 10.0 + 5.0 * 1.0
    expected_y = 20.0 + (-2.0) * 1.0
    
    # Expected velocity: unchanged
    expected_vx = 5.0
    expected_vy = -2.0
    
    expected_state = np.array([expected_x, expected_y, expected_vx, expected_vy])
    
    np.testing.assert_allclose(predicted_state, expected_state, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(kf.get_state(), expected_state, rtol=1e-5, atol=1e-5)


def test_update_reduces_uncertainty():
    """Check update-step reduces the covariance trace."""
    dt = 1.0
    kf = KalmanFilter(dt=dt, q=0.5, r_x=10.0, r_y=10.0)
    
    # Initialize state to set up realistic covariance
    kf.init_state(np.array([0.0, 0.0]))
    
    # Do one prediction step so P matrix grows
    kf.predict()
    
    p_pre_update = kf.get_covariance()
    trace_pre_update = np.trace(p_pre_update)
    
    # Do update step with a dummy measurement
    measurement = np.array([5.0, 5.0])
    kf.update(measurement)
    
    p_post_update = kf.get_covariance()
    trace_post_update = np.trace(p_post_update)
    
    # Trace represents total variance across all state dimensions
    # Update should strictly reduce this uncertainty
    assert trace_post_update < trace_pre_update


def test_straight_line_convergence():
    """Check convergence over a mock straight-line scenario."""
    # 1. Simulate target trajectory
    dt = 1.0
    n_steps = 50
    target = Target(x0=0.0, y0=0.0, vx0=20.0, vy0=0.0, model="cv")
    true_states = target.get_trajectory(dt, n_steps)
    
    # 2. Simulate radar measurements with fixed seed
    noise_std = 10.0
    radar = Radar(noise_std_x=noise_std, noise_std_y=noise_std, seed=42)
    measurements = radar.measure_batch(true_states)
    
    # 3. Track with Kalman Filter
    kf = KalmanFilter(dt=dt, q=0.5, r_x=noise_std, r_y=noise_std)
    
    estimated_states = np.zeros((n_steps, 4))
    kf.init_state(measurements[0])
    estimated_states[0] = kf.get_state()
    
    for t in range(1, n_steps):
        estimated_states[t] = kf.step(measurements[t])
        
    # 4. Check Assertions
    # Calculate raw radar RMSE
    padded_measurements = np.zeros((n_steps, 4))
    padded_measurements[:, :2] = measurements
    radar_rms_error = rmse(true_states, padded_measurements)
    
    # Calculate KF RMSE
    kf_rms_error = rmse(true_states, estimated_states)
    
    # Filter should perform better than raw measurements
    assert kf_rms_error < radar_rms_error
    
    # Positional error at final step should have converged to a reasonable value
    final_true_pos = true_states[-1, :2]
    final_est_pos = estimated_states[-1, :2]
    final_pos_error = np.linalg.norm(final_true_pos - final_est_pos)
    
    # Bound the error knowing the seed=42 generates specific values
    assert final_pos_error < 15.0
