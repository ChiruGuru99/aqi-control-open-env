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
    reference_observation = with_counterfactual.reset(seed=2, task_id="crisis")
    reference_observations = [reference_observation]
    baselines = []

    for day in range(15):
        action = AQIAction(grap_stage=2 if day % 3 else 4)
        reference_observation = with_counterfactual.step(action)
        reference_observations.append(reference_observation)
        baselines.extend(
            trajectory[-1]["pm25_base"]
            for trajectory in with_counterfactual._trajectories.values()
        )

    baseline_iterator = iter(baselines)
    without_counterfactual = AQIControlEnvironment()
    control_observation = without_counterfactual.reset(seed=2, task_id="crisis")
    control_observations = [control_observation]

    monkeypatch.setattr(
        without_counterfactual,
        "_counterfactual_pm25",
        lambda pm25_input, *args: next(baseline_iterator),
    )

    for day in range(15):
        action = AQIAction(grap_stage=2 if day % 3 else 4)
        control_observation = without_counterfactual.step(action)
        control_observations.append(control_observation)

    assert list(map(_physical_observation, reference_observations)) == list(
        map(_physical_observation, control_observations)
    )

    assert (
        _physical_trajectory(with_counterfactual)
        == _physical_trajectory(without_counterfactual)
    )
