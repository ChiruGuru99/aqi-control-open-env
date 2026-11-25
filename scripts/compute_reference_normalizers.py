"""Compute grader reduction references from max-everything trajectories."""

from __future__ import annotations

import statistics
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.graders import _compute_core_metrics  # noqa: E402
from scripts.audit_policies import POLICY_FACTORIES, SEEDS, TASKS  # noqa: E402
from server.aqi_environment import AQIControlEnvironment  # noqa: E402


def _episode_metrics(task_id: str, seed: int) -> tuple[float, float]:
    environment = AQIControlEnvironment()
    observation = environment.reset(seed=seed, task_id=task_id)
    policy = POLICY_FACTORIES["max_everything"]

    while not observation.done:
        observation = environment.step(policy(observation))

    metrics = [
        _compute_core_metrics(trajectory)
        for trajectory in environment._trajectories.values()
    ]
    avg_reduction = statistics.fmean(metric["avg_reduction"] for metric in metrics)
    severe_reduction = statistics.fmean(
        metric["severe_reduction"] for metric in metrics
    )
    return avg_reduction, severe_reduction


def compute_references() -> Dict[str, Dict[str, float]]:
    """Return seed-mean raw reductions produced by max_everything."""
    references: Dict[str, Dict[str, float]] = {}
    for task_id in TASKS:
        episode_metrics = [
            _episode_metrics(task_id, seed)
            for seed in SEEDS
        ]
        references[task_id] = {
            "avg": statistics.fmean(metric[0] for metric in episode_metrics),
            "severe": statistics.fmean(metric[1] for metric in episode_metrics),
        }
    return references


def main() -> int:
    references = compute_references()
    print("# Generated from max_everything over seeds 0-4.")
    for task_id, values in references.items():
        prefix = task_id.upper()
        print(f"{prefix}_AVG_REDUCTION_REFERENCE = {values['avg']:.12f}")
        print(f"{prefix}_SEVERE_REDUCTION_REFERENCE = {values['severe']:.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
