"""Unit tests for the Target class motion models."""

import numpy as np
import pytest

from radarsim.sim.target import Target


# --- Constant Velocity Model Tests ---

def test_cv_step_returns_correct_shape():
    """CV step returns shape (4,)."""
    target = Target(x0=0.0, y0=0.0, vx0=10.0, vy0=5.0, model="cv")
    state = target.step(1.0)
    assert state.shape == (4,)


def test_cv_step_updates_position():
    """CV step advances position by v * dt."""
    target = Target(x0=0.0, y0=0.0, vx0=10.0, vy0=5.0, model="cv")
    state = target.step(1.0)
    np.testing.assert_allclose(state, [10.0, 5.0, 10.0, 5.0])


# --- Coordinated Turn Model Tests ---

def test_ct_step_returns_correct_shape():
    """CT step returns np.ndarray of shape (4,)."""
    target = Target(
        x0=0.0, y0=0.0, vx0=100.0, vy0=0.0,
        model="ct", turn_rate=0.1,
    )
    state = target.step(1.0)
    assert isinstance(state, np.ndarray)
    assert state.shape == (4,)


def test_ct_preserves_speed():
    """CT model preserves speed magnitude across multiple steps."""
    vx0, vy0 = 100.0, 50.0
    initial_speed = np.sqrt(vx0 ** 2 + vy0 ** 2)

    target = Target(
        x0=0.0, y0=0.0, vx0=vx0, vy0=vy0,
        model="ct", turn_rate=0.05,
    )

    for _ in range(100):
        state = target.step(1.0)
        speed = np.sqrt(state[2] ** 2 + state[3] ** 2)
        np.testing.assert_allclose(speed, initial_speed, rtol=1e-10)


def test_ct_near_zero_turn_rate_falls_back_to_cv():
    """Near-zero turn rate produces same trajectory as CV model."""
    dt = 1.0
    n_steps = 50

    # CV target
    target_cv = Target(x0=0.0, y0=0.0, vx0=20.0, vy0=10.0, model="cv")
    traj_cv = target_cv.get_trajectory(dt, n_steps)

    # CT target with near-zero turn rate
    target_ct = Target(
        x0=0.0, y0=0.0, vx0=20.0, vy0=10.0,
        model="ct", turn_rate=1e-15,
    )
    traj_ct = target_ct.get_trajectory(dt, n_steps)

    np.testing.assert_allclose(traj_ct, traj_cv, atol=1e-6)


def test_ct_get_trajectory_non_destructive():
    """get_trajectory() does not alter state or heading for CT model."""
    target = Target(
        x0=0.0, y0=0.0, vx0=100.0, vy0=0.0,
        model="ct", turn_rate=0.1,
    )

    # Advance a few steps to get a non-initial state
    for _ in range(5):
        target.step(1.0)

    state_before = target.state.copy()
    heading_before = target._heading

    # Generate trajectory — should not change current state
    trajectory = target.get_trajectory(1.0, 50)

    assert trajectory.shape == (50, 4)
    np.testing.assert_array_equal(target.state, state_before)
    assert target._heading == heading_before


def test_ct_actually_turns():
    """CT model changes velocity direction over time."""
    target = Target(
        x0=0.0, y0=0.0, vx0=100.0, vy0=0.0,
        model="ct", turn_rate=0.1,
    )
    initial_state = target.step(0.0)  # no movement at dt=0

    # After several steps, velocity direction should have changed
    for _ in range(20):
        target.step(1.0)

    final_state = target.state
    # vy should no longer be zero — the target has turned
    assert abs(final_state[3]) > 1.0


def test_ct_requires_turn_rate():
    """CT model raises ValueError when turn_rate is not provided."""
    with pytest.raises(ValueError, match="turn_rate is required"):
        Target(x0=0.0, y0=0.0, vx0=10.0, vy0=0.0, model="ct")


# --- Random Maneuver Model Tests ---

def test_random_step_returns_correct_shape():
    """Random step returns np.ndarray of shape (4,)."""
    target = Target(
        x0=0.0, y0=0.0, vx0=50.0, vy0=0.0,
        model="random", accel_std=2.0, seed=42,
    )
    state = target.step(1.0)
    assert isinstance(state, np.ndarray)
    assert state.shape == (4,)


def test_random_changes_velocity():
    """Random model changes velocity between steps (unlike CV)."""
    target = Target(
        x0=0.0, y0=0.0, vx0=50.0, vy0=0.0,
        model="random", accel_std=5.0, seed=42,
    )
    state1 = target.step(1.0)
    state2 = target.step(1.0)

    # Velocity should differ between steps due to random acceleration
    assert not np.allclose(state1[2:], state2[2:])


def test_random_reproducible_with_seed():
    """Same seed produces identical trajectories."""
    kwargs = dict(x0=0.0, y0=0.0, vx0=30.0, vy0=10.0,
                  model="random", accel_std=3.0, seed=123)

    target_a = Target(**kwargs)
    traj_a = target_a.get_trajectory(1.0, 50)

    target_b = Target(**kwargs)
    traj_b = target_b.get_trajectory(1.0, 50)

    np.testing.assert_array_equal(traj_a, traj_b)


def test_random_get_trajectory_non_destructive():
    """get_trajectory() restores state and RNG state for random model."""
    target = Target(
        x0=0.0, y0=0.0, vx0=50.0, vy0=0.0,
        model="random", accel_std=2.0, seed=42,
    )

    # Advance a few steps
    for _ in range(5):
        target.step(1.0)

    state_before = target.state.copy()
    rng_state_before = target._rng.bit_generator.state

    trajectory = target.get_trajectory(1.0, 50)

    assert trajectory.shape == (50, 4)
    np.testing.assert_array_equal(target.state, state_before)
    # RNG state must also be restored
    assert target._rng.bit_generator.state == rng_state_before


def test_random_requires_accel_std():
    """Random model raises ValueError when accel_std is not provided."""
    with pytest.raises(ValueError, match="accel_std is required"):
        Target(x0=0.0, y0=0.0, vx0=10.0, vy0=0.0, model="random")
