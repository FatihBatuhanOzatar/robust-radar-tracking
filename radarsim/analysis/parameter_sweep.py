"""Parameter sweep analysis for Q and R tuning."""

from typing import Callable

import numpy as np

from radarsim.analysis.metrics import rmse


# Type alias for scenario functions.
# A scenario function takes (q, r_x, r_y) and returns
# (true_states, estimated_states), both shape (n_steps, 4).
ScenarioFn = Callable[[float, float, float], tuple[np.ndarray, np.ndarray]]


def sweep_q(
    scenario_fn: ScenarioFn,
    q_values: list[float],
    r_x: float = 25.0,
    r_y: float = 25.0,
) -> dict[float, float]:
    """Run a scenario with different process noise (Q) values.

    Keeps measurement noise fixed and varies Q to measure its effect
    on tracking RMSE.

    Args:
        scenario_fn: Callable taking (q, r_x, r_y) and returning
            (true_states, estimated_states), both shape (n_steps, 4).
        q_values: List of Q values to test.
        r_x: Fixed measurement noise std in x (meters).
        r_y: Fixed measurement noise std in y (meters).

    Returns:
        Dict mapping each Q value to its position RMSE.
    """
    results: dict[float, float] = {}
    for q in q_values:
        true_states, estimated_states = scenario_fn(q, r_x, r_y)
        results[q] = rmse(true_states, estimated_states)
    return results


def sweep_r(
    scenario_fn: ScenarioFn,
    r_values: list[float],
    q: float = 0.5,
) -> dict[float, float]:
    """Run a scenario with different measurement noise (R) values.

    Keeps process noise fixed and varies R (both axes equally) to
    measure its effect on tracking RMSE.

    Args:
        scenario_fn: Callable taking (q, r_x, r_y) and returning
            (true_states, estimated_states), both shape (n_steps, 4).
        r_values: List of R noise std values to test (applied to both axes).
        q: Fixed process noise intensity.

    Returns:
        Dict mapping each R value to its position RMSE.
    """
    results: dict[float, float] = {}
    for r in r_values:
        true_states, estimated_states = scenario_fn(q, r, r)
        results[r] = rmse(true_states, estimated_states)
    return results


def sweep_qr_heatmap(
    scenario_fn: ScenarioFn,
    q_values: list[float],
    r_values: list[float],
) -> np.ndarray:
    """Run a scenario across all Q × R combinations.

    Produces a 2D RMSE grid for heatmap visualization. Each cell
    (i, j) contains the RMSE for q_values[i] and r_values[j].

    Args:
        scenario_fn: Callable taking (q, r_x, r_y) and returning
            (true_states, estimated_states), both shape (n_steps, 4).
        q_values: List of Q values (rows).
        r_values: List of R noise std values (columns, applied to both axes).

    Returns:
        RMSE grid, shape (len(q_values), len(r_values)).
    """
    grid = np.zeros((len(q_values), len(r_values)))
    for i, q in enumerate(q_values):
        for j, r in enumerate(r_values):
            true_states, estimated_states = scenario_fn(q, r, r)
            grid[i, j] = rmse(true_states, estimated_states)
    return grid
