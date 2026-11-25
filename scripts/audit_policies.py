"""Audit terminal grader scores for fixed AQI policy strategies."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Callable, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.grap import GRAP_STAGES  # noqa: E402
from models import AQIAction, AQIObservation, SingleAction  # noqa: E402
from server.aqi_environment import AQIControlEnvironment, TASK_CONFIGS  # noqa: E402


TASKS = tuple(TASK_CONFIGS)
SEEDS = tuple(range(5))
POLICIES = ("noop", "grap2", "grap4", "threshold", "harmful")

Policy = Callable[[AQIObservation], AQIAction]


def _threshold_stage(pm25: float) -> int:
    """Return the GRAP stage whose strict trigger is exceeded by PM2.5."""
    for stage in range(1, 5):
        if pm25 <= GRAP_STAGES[stage]["trigger_pm25"]:
            return stage - 1
    return 4


def _harmful_action(observation: AQIObservation) -> AQIAction:
    del observation
    action_types = (
        "restrict_traffic",
        "curtail_construction",
        "limit_industry",
        "issue_public_advisory",
        "close_schools",
        "mandate_wfh",
        "deploy_smog_response",
        "emergency_transport_ban",
    )
    return AQIAction(
        actions=[
            SingleAction(action_type=action_type, level=3)
            for action_type in action_types
            if action_type != "issue_public_advisory"
        ]
        + [SingleAction(action_type="issue_public_advisory", level=0)]
    )


POLICY_FACTORIES: Dict[str, Policy] = {
    "noop": lambda observation: AQIAction(action_type="no_action"),
    "grap2": lambda observation: AQIAction(grap_stage=2),
    "grap4": lambda observation: AQIAction(grap_stage=4),
    "threshold": lambda observation: AQIAction(
        grap_stage=_threshold_stage(observation.pm25_today)
    ),
    "harmful": _harmful_action,
}


def run_episode(task_id: str, seed: int, policy_name: str) -> float:
    """Run one complete episode and return its terminal task grader score."""
    environment = AQIControlEnvironment()
    observation = environment.reset(seed=seed, task_id=task_id)
    policy = POLICY_FACTORIES[policy_name]

    while not observation.done:
        observation = environment.step(policy(observation))

    return float(observation.grader_scores[task_id])


def audit() -> Dict[str, Dict[str, List[float]]]:
    """Collect terminal scores for every task, seed, and policy."""
    return {
        task_id: {
            policy_name: [
                run_episode(task_id, seed, policy_name)
                for seed in SEEDS
            ]
            for policy_name in POLICIES
        }
        for task_id in TASKS
    }


def render_table(results: Dict[str, Dict[str, List[float]]]) -> str:
    """Render the audit as a mean +/- population-standard-deviation table."""
    header = "| Task | " + " | ".join(POLICIES) + " |"
    separator = "|---|" + "---|" * len(POLICIES)
    lines = [header, separator]

    for task_id in TASKS:
        cells = []
        for policy_name in POLICIES:
            scores = results[task_id][policy_name]
            cells.append(
                f"{statistics.fmean(scores):.4f} \u00b1 "
                f"{statistics.pstdev(scores):.4f}"
            )
        lines.append(f"| {task_id} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit policy ordering in the AQI control environment."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional Markdown file to update with the audit table.",
    )
    args = parser.parse_args()

    results = audit()
    table = render_table(results)
    print(table)

    if args.output:
        document = (
            "# Policy audit (before reward fixes)\n\n"
            "Generated with `python scripts\\audit_policies.py --output "
            f"{args.output.as_posix()}` over seeds 0-4. Values are terminal "
            "grader score mean \u00b1 population standard deviation.\n\n"
            f"{table}\n"
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(document, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
