from __future__ import annotations

import math
import statistics
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict

import pytest
from pydantic import ValidationError

from aqi_control_env.models import AQIAction, AQIObservation
from aqi_control_env.env.graders import grade_easy, grade_hard, grade_medium
from aqi_control_env.server.aqi_environment import (
    AQIControlEnvironment,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_policies import (
    POLICIES,
    POLICY_FACTORIES,
    SEEDS,
    TASKS,
    run_episode_with_details,
)


@lru_cache(maxsize=None)
def _score_and_details(
    task_id: str,
    seed: int,
    policy_name: str,
) -> tuple[float, Dict[str, object]]:
    return run_episode_with_details(task_id, seed, policy_name)


def _means(task_id: str) -> Dict[str, float]:
    return {
        policy_name: statistics.fmean(
            _score_and_details(task_id, seed, policy_name)[0]
            for seed in SEEDS
        )
        for policy_name in POLICIES
    }


def _fixed_action_episode(task_id: str, seed: int) -> tuple[
    tuple[Dict[str, object], ...],
    tuple[float, ...],
    float,
]:
    environment = AQIControlEnvironment()
    observation = environment.reset(
        seed=seed,
        episode_id=f"determinism-{task_id}-{seed}",
        task_id=task_id,
    )
    observations = [observation.model_dump()]
    rewards = [observation.reward]
    day = 0

    while not observation.done:
        stage = (day % 4) + 1
        observation = environment.step(AQIAction(grap_stage=stage))
        observations.append(observation.model_dump())
        rewards.append(observation.reward)
        day += 1

    return (
        tuple(observations),
        tuple(rewards),
        float(observation.grader_scores[task_id]),
    )


@pytest.mark.parametrize("task_id", TASKS)
@pytest.mark.parametrize("seed", SEEDS)
def test_determinism(task_id: str, seed: int) -> None:
    assert _fixed_action_episode(task_id, seed) == _fixed_action_episode(
        task_id, seed
    )


@pytest.mark.parametrize("policy_name", POLICIES)
@pytest.mark.parametrize("task_id", TASKS)
@pytest.mark.parametrize("seed", SEEDS)
def test_grader_score_range(
    task_id: str,
    seed: int,
    policy_name: str,
) -> None:
    score, _ = _score_and_details(task_id, seed, policy_name)
    assert math.isfinite(score)
    assert 0 < score < 1


@pytest.mark.parametrize("task_id", TASKS)
def test_threshold_beats_noop(task_id: str) -> None:
    means = _means(task_id)
    assert means["threshold"] > means["noop"]


@pytest.mark.parametrize("task_id", TASKS)
def test_noop_beats_anti_threshold(task_id: str) -> None:
    means = _means(task_id)
    assert means["noop"] > means["anti_threshold"]


@pytest.mark.parametrize("task_id", TASKS)
@pytest.mark.parametrize("seed", SEEDS)
def test_anti_threshold_is_worse_than_targeted_policies(
    task_id: str,
    seed: int,
) -> None:
    anti_score, _ = _score_and_details(task_id, seed, "anti_threshold")
    threshold_score, _ = _score_and_details(task_id, seed, "threshold")
    budget_score, _ = _score_and_details(task_id, seed, "budget_aware")
    assert anti_score < threshold_score
    assert anti_score < budget_score


@pytest.mark.parametrize("task_id", ("easy", "medium"))
def test_budget_aware_matches_threshold_without_budget(task_id: str) -> None:
    for seed in SEEDS:
        assert _score_and_details(task_id, seed, "budget_aware")[0] == (
            _score_and_details(task_id, seed, "threshold")[0]
        )


@pytest.mark.parametrize(
    "task_id",
    TASKS,
)
def test_max_everything_is_not_top_policy(task_id: str) -> None:
    means = _means(task_id)
    other_scores = [
        score
        for policy_name, score in means.items()
        if policy_name != "max_everything"
    ]
    assert means["max_everything"] < max(other_scores)


def test_hard_budget_aware_beats_noop() -> None:
    means = _means("hard")
    assert means["budget_aware"] > means["noop"]


def test_hard_budget_aware_beats_threshold() -> None:
    means = _means("hard")
    assert means["budget_aware"] > means["threshold"]


@pytest.mark.parametrize(
    "task_id",
    TASKS,
)
def test_grader_signal_spread(task_id: str) -> None:
    means = _means(task_id)
    assert max(means.values()) - min(means.values()) >= 0.10


@pytest.mark.parametrize("task_id", TASKS)
def test_noop_reduction_is_neutral(task_id: str) -> None:
    reductions = []
    for seed in SEEDS:
        environment = AQIControlEnvironment()
        observation = environment.reset(seed=seed, task_id=task_id)
        while not observation.done:
            observation = environment.step(AQIAction(action_type="no_action"))
        reductions.append(
            sum(
                (
                    step["pm25_base"] - step["pm25_post"]
                )
                / step["pm25_base"]
                for trajectory in environment._trajectories.values()
                for step in trajectory
            )
            / sum(
                len(trajectory)
                for trajectory in environment._trajectories.values()
            )
        )
    assert abs(statistics.fmean(reductions)) < 0.02


def test_no_severe_days_excludes_and_renormalizes_component() -> None:
    trajectory = [
        {
            "pm25_base": 100.0,
            "pm25_post": 100.0,
            "economic_cost": 0.0,
        }
    ]

    easy_score, easy_details = grade_easy(trajectory, 1)
    medium_score, medium_details = grade_medium(
        {"delhi": trajectory, "lucknow": trajectory}, 1
    )
    hard_score, hard_details = grade_hard(
        {
            "delhi": trajectory,
            "lucknow": trajectory,
            "patna": trajectory,
        },
        1,
        budget_total=150.0,
        budget_spent=0.0,
    )

    assert easy_details["component_severe"] is None
    assert medium_details["severe_days_present"] is False
    assert hard_details["severe_days_present"] is False
    assert easy_score == pytest.approx((0.5 * 0.30) / 0.70, abs=1e-4)
    assert medium_score > 0
    assert hard_score > 0


@pytest.mark.parametrize("task_id", ("hard", "expert", "crisis"))
@pytest.mark.parametrize("seed", SEEDS)
def test_budget_aware_does_not_overspend(task_id: str, seed: int) -> None:
    environment = AQIControlEnvironment()
    observation = environment.reset(seed=seed, task_id=task_id)

    while not observation.done:
        observation = environment.step(
            POLICY_FACTORIES["budget_aware"](observation)
        )
        assert environment._budget_spent <= environment._budget_total


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2 + 1
        for index in range(start, end):
            result[order[index]] = rank
        start = end
    return result


@pytest.mark.parametrize("task_id", TASKS)
def test_dense_reward_agrees_with_grader_order(task_id: str) -> None:
    grader_scores = []
    dense_scores = []

    for policy_name in POLICIES:
        episode_grades = []
        episode_rewards = []
        for seed in SEEDS:
            environment = AQIControlEnvironment()
            observation = environment.reset(seed=seed, task_id=task_id)
            while not observation.done:
                observation = environment.step(
                    POLICY_FACTORIES[policy_name](observation)
                )
            episode_grades.append(observation.grader_scores[task_id])
            episode_rewards.append(environment._cumulative_reward)
        grader_scores.append(statistics.fmean(episode_grades))
        dense_scores.append(statistics.fmean(episode_rewards))

    grader_ranks = _ranks(grader_scores)
    dense_ranks = _ranks(dense_scores)
    count = len(POLICIES)
    rank_error = sum(
        (grader - dense) ** 2
        for grader, dense in zip(grader_ranks, dense_ranks)
    )
    rho = 1 - (6 * rank_error) / (count * (count * count - 1))
    assert rho >= 0.8


def _run_action_sequence(
    action: AQIAction,
) -> tuple[tuple[float, ...], float, AQIObservation]:
    environment = AQIControlEnvironment()
    observation = environment.reset(seed=0, task_id="easy")
    rewards = []
    while not observation.done:
        observation = environment.step(action)
        rewards.append(observation.reward)
    return tuple(rewards), observation.grader_scores["easy"], observation


def test_out_of_range_level_is_rejected_and_runtime_clamped() -> None:
    with pytest.raises(ValidationError):
        AQIAction(action_type="restrict_traffic", level=99)

    invalid = AQIAction.model_construct(
        action_type="restrict_traffic",
        level=99,
        grap_stage=None,
        actions=[],
        city="",
    )
    assert _run_action_sequence(invalid)[:2] == _run_action_sequence(
        AQIAction(action_type="restrict_traffic", level=3)
    )[:2]


def test_out_of_range_grap_stage_is_rejected_and_runtime_clamped() -> None:
    with pytest.raises(ValidationError):
        AQIAction(grap_stage=7)

    invalid = AQIAction.model_construct(
        action_type="no_action",
        level=0,
        grap_stage=7,
        actions=[],
        city="",
    )
    assert _run_action_sequence(invalid)[:2] == _run_action_sequence(
        AQIAction(grap_stage=4)
    )[:2]


def test_unknown_city_is_rejected_without_reward_benefit() -> None:
    valid = AQIAction(action_type="no_action")
    invalid = AQIAction(
        action_type="restrict_traffic",
        level=2,
        city="unknown-city",
    )
    invalid_result = _run_action_sequence(invalid)
    valid_result = _run_action_sequence(valid)
    assert invalid_result[:2] == valid_result[:2]
    assert "Invalid city 'unknown-city'; action rejected" in (
        invalid_result[2].message
    )


def test_unaffordable_budget_action_is_rejected_whole() -> None:
    rejected = AQIControlEnvironment()
    control = AQIControlEnvironment()
    rejected.reset(seed=0, task_id="hard")
    control.reset(seed=0, task_id="hard")
    rejected._budget_spent = rejected._budget_total - 1.0
    control._budget_spent = control._budget_total - 1.0

    observation = rejected.step(AQIAction(grap_stage=4))
    expected = control.step(AQIAction(action_type="no_action"))

    step = rejected._trajectories[rejected._primary_city][0]
    assert step["action_type"] == "no_action"
    assert step["level"] == 0
    assert step["economic_cost"] == 0.0
    assert step["pm25_post"] == step["pm25_base"]
    assert rejected._budget_spent == rejected._budget_total - 1.0
    assert observation.reward == expected.reward
    assert observation.reward_breakdown["budget_rejected_action"] == 1.0
    assert observation.grap_stage == 0
    assert "action rejected; insufficient budget" in observation.message
