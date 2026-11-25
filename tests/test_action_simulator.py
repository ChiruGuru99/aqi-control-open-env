from __future__ import annotations

from typing import Callable, Dict, List

from aqi_control_env.env.grap import GRAP_STAGES
from aqi_control_env.models import AQIAction, AQIObservation, SingleAction
from aqi_control_env.server.aqi_environment import (
    AQIControlEnvironment,
    TASK_CONFIGS,
)


Policy = Callable[[AQIObservation], AQIAction]


def _threshold_stage(pm25: float) -> int:
    for stage in range(1, 5):
        if pm25 <= GRAP_STAGES[stage]["trigger_pm25"]:
            return stage - 1
    return 4


def _harmful_action(observation: AQIObservation) -> AQIAction:
    del observation
    return AQIAction(
        actions=[
            SingleAction(action_type="restrict_traffic", level=3),
            SingleAction(action_type="curtail_construction", level=3),
            SingleAction(action_type="limit_industry", level=3),
            SingleAction(action_type="close_schools", level=3),
            SingleAction(action_type="mandate_wfh", level=3),
            SingleAction(action_type="deploy_smog_response", level=3),
            SingleAction(action_type="emergency_transport_ban", level=3),
            SingleAction(action_type="issue_public_advisory", level=0),
        ]
    )


POLICIES: Dict[str, Policy] = {
    "noop": lambda observation: AQIAction(action_type="no_action"),
    "grap2": lambda observation: AQIAction(grap_stage=2),
    "grap4": lambda observation: AQIAction(grap_stage=4),
    "threshold": lambda observation: AQIAction(
        grap_stage=_threshold_stage(observation.pm25_today)
    ),
    "harmful": _harmful_action,
}


def test_grap_stage_resolves_on_every_task() -> None:
    for task_id in TASK_CONFIGS:
        environment = AQIControlEnvironment()
        environment.reset(seed=0, task_id=task_id)

        environment.step(AQIAction(grap_stage=2))

        primary_step = environment._trajectories[environment._primary_city][0]
        assert primary_step["economic_cost"] > 0, task_id
        assert primary_step["action_type"] != "no_action", task_id


def test_interventions_never_raise_pm25_above_counterfactual() -> None:
    violations: List[str] = []

    for task_id in TASK_CONFIGS:
        for seed in range(5):
            for policy_name, policy in POLICIES.items():
                environment = AQIControlEnvironment()
                observation = environment.reset(seed=seed, task_id=task_id)

                while not observation.done:
                    observation = environment.step(policy(observation))
                    day = environment._current_day
                    for city, trajectory in environment._trajectories.items():
                        if len(trajectory) < day:
                            continue
                        step = trajectory[-1]
                        if step["pm25_post"] > step["pm25_base"] + 1e-6:
                            violations.append(
                                f"{task_id}/seed={seed}/{policy_name}/"
                                f"day={day}/{city}: {step['action_type']} "
                                f"level={step['level']} post={step['pm25_post']} "
                                f"base={step['pm25_base']}"
                            )

    sample = "\n".join(violations[:30])
    assert not violations, (
        f"{len(violations)} PM2.5 invariant violations; first 30:\n{sample}"
    )


def test_seed_zero_is_not_replaced_by_default_seed() -> None:
    observations = {}
    for seed in (0, 42):
        environment = AQIControlEnvironment()
        observation = environment.reset(seed=seed, task_id="crisis")
        trajectory = []

        for _ in range(5):
            observation = environment.step(AQIAction(action_type="no_action"))
            trajectory.append(
                (
                    observation.wind_regime,
                    observation.pm25_post_intervention,
                )
            )
        observations[seed] = trajectory

    assert observations[0] != observations[42]
