from __future__ import annotations

from typing import Any, Dict

import pytest

from aqi_control_env.models import AQIAction
from aqi_control_env.server.aqi_environment import AQIControlEnvironment


def _physical_observation(observation: Any) -> Dict[str, Any]:
    """Exclude terminal grading fields when comparing real trajectories."""
    data = observation.model_dump()
    for key in ("reward", "message", "metadata", "grader_scores"):
        data.pop(key, None)
    return data


def _physical_trajectory(environment: AQIControlEnvironment) -> Dict[str, Any]:
    """Exclude the diagnostic baseline while retaining dense reward signals."""
    trajectories = {}
    for city, city_trajectory in environment._trajectories.items():
        trajectories[city] = [
            {key: value for key, value in step.items() if key != "pm25_base"}
            for step in city_trajectory
        ]
    return trajectories


def test_no_action_baseline_includes_same_day_dynamics() -> None:
    environment = AQIControlEnvironment()
    environment.reset(seed=3, task_id="crisis")

    observation = environment.step(AQIAction(action_type="no_action"))

    steps = []
    for trajectory in environment._trajectories.values():
        step = trajectory[0]
        steps.append(step)
        assert step["pm25_base"] == step["pm25_post"]
        assert step["pm25_input"] > 0
    assert any(step["pm25_input"] != step["pm25_base"] for step in steps)
    assert observation.pm25_post_intervention > 0


def test_counterfactual_computation_does_not_change_real_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with_counterfactual = AQIControlEnvironment()
    without_counterfactual = AQIControlEnvironment()
    with_counterfactual.reset(seed=2, task_id="crisis")
    without_counterfactual.reset(seed=2, task_id="crisis")

    monkeypatch.setattr(
        without_counterfactual,
        "_counterfactual_pm25",
        lambda pm25_input, *args: pm25_input,
    )

    for day in range(15):
        action = AQIAction(grap_stage=2 if day % 3 else 4)
        observed = with_counterfactual.step(action)
        control = without_counterfactual.step(action)

        assert _physical_observation(observed) == _physical_observation(control)

    assert (
        _physical_trajectory(with_counterfactual)
        == _physical_trajectory(without_counterfactual)
    )
