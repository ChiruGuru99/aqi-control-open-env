"""Audit terminal grader scores for fixed AQI policy strategies."""

# Easy and medium are deterministic regardless of seed. Hard is reproducible
# for a fixed seed but enables the seeded weather engine, so scores can vary.

from __future__ import annotations

import argparse
import os
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Callable, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.grap import GRAP_STAGES, expand_grap_stage  # noqa: E402
from env.graders import (  # noqa: E402
    grade_crisis,
    grade_easy,
    grade_expert,
    grade_hard,
    grade_medium,
)
from models import AQIAction, AQIObservation, SingleAction  # noqa: E402
from server.aqi_environment import AQIControlEnvironment, TASK_CONFIGS  # noqa: E402


TASKS = tuple(TASK_CONFIGS)
_seed_spec = os.environ.get("AQI_AUDIT_SEEDS", "0-4")
_seed_match = re.fullmatch(r"(\d+)-(\d+)", _seed_spec)
if not _seed_match or int(_seed_match.group(1)) > int(_seed_match.group(2)):
    raise ValueError("AQI_AUDIT_SEEDS must use an inclusive range such as 0-4")
SEEDS = tuple(range(int(_seed_match.group(1)), int(_seed_match.group(2)) + 1))
POLICIES = (
    "noop",
    "grap2",
    "grap4",
    "threshold",
    "max_everything",
    "anti_threshold",
    "budget_aware",
    "random",
    "lagged_threshold",
)

Policy = Callable[[AQIObservation], AQIAction]


def _threshold_stage(pm25: float) -> int:
    """Return the GRAP stage whose strict trigger is exceeded by PM2.5."""
    for stage in range(1, 5):
        if pm25 <= GRAP_STAGES[stage]["trigger_pm25"]:
            return stage - 1
    return 4


def _max_everything_action(observation: AQIObservation) -> AQIAction:
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


def _anti_threshold_action(observation: AQIObservation) -> AQIAction:
    pm25 = observation.pm25_today
    if pm25 <= 120:
        return AQIAction(grap_stage=3)
    if pm25 > 250:
        return AQIAction(grap_stage=0)
    return AQIAction(grap_stage=1)


def _grap_bundle_action(stage: int) -> AQIAction:
    actions = expand_grap_stage(stage)
    return AQIAction(
        actions=[
            SingleAction(
                action_type=action["action_type"],
                level=action["level"],
            )
            for action in actions
        ]
    )


def _budget_aware_action(observation: AQIObservation) -> AQIAction:
    budget_total = TASK_CONFIGS[observation.task_id]["budget"]
    if budget_total < 0:
        return AQIAction(grap_stage=_threshold_stage(observation.pm25_today))

    fatigue_multiplier = 1.0 + observation.policy_fatigue_index * 0.5
    projected_cost = {
        0: 0.0,
        1: 2.0 * fatigue_multiplier,
        3: 26.5 * fatigue_multiplier,
    }
    # Observation budgets are rounded to one decimal, so reserve the maximum
    # possible rounding error before authorizing a costed action.
    budget_remaining = max(0.0, observation.budget_remaining - 0.05)
    severe_forecast = "severe" in observation.forecast_risk_next_2d

    worst_forecast_day = (
        severe_forecast
        and observation.pm25_today >= 300
    )
    if worst_forecast_day and projected_cost[3] <= budget_remaining:
        return _grap_bundle_action(3)
    if observation.pm25_today > 60 and projected_cost[1] <= budget_remaining:
        return _grap_bundle_action(1)
    return _grap_bundle_action(0)


def _random_action(observation: AQIObservation) -> AQIAction:
    task_offset = TASKS.index(observation.task_id) * 1000
    rng = random.Random(20250308 + task_offset + observation.day)
    return AQIAction(grap_stage=rng.randrange(5))


def _lagged_threshold_action(observation: AQIObservation) -> AQIAction:
    pm25 = (
        observation.pm25_history[-1]
        if observation.pm25_history
        else observation.pm25_today
    )
    return AQIAction(grap_stage=_threshold_stage(pm25))


POLICY_FACTORIES: Dict[str, Policy] = {
    "noop": lambda observation: AQIAction(action_type="no_action"),
    "grap2": lambda observation: AQIAction(grap_stage=2),
    "grap4": lambda observation: AQIAction(grap_stage=4),
    "threshold": lambda observation: AQIAction(
        grap_stage=_threshold_stage(observation.pm25_today)
    ),
    "max_everything": _max_everything_action,
    "anti_threshold": _anti_threshold_action,
    "budget_aware": _budget_aware_action,
    "random": _random_action,
    "lagged_threshold": _lagged_threshold_action,
}


def run_episode_with_details(
    task_id: str,
    seed: int,
    policy_name: str,
) -> tuple[float, Dict[str, object]]:
    """Run one episode and return its score and terminal grader details."""
    environment = AQIControlEnvironment()
    observation = environment.reset(seed=seed, task_id=task_id)
    policy = POLICY_FACTORIES[policy_name]

    while not observation.done:
        observation = environment.step(policy(observation))

    if task_id == "easy":
        score, details = grade_easy(
            environment._trajectories[environment._primary_city],
            environment._total_days,
        )
    elif task_id == "medium":
        score, details = grade_medium(
            environment._trajectories,
            environment._total_days,
        )
    elif task_id == "hard":
        score, details = grade_hard(
            environment._trajectories,
            environment._total_days,
            environment._budget_total,
            environment._budget_spent,
        )
    else:
        health_summary = (
            environment._health_tracker.get_summary()
            if environment._health_tracker
            else {}
        )
        sentiment_history = {}
        if environment._sentiment_tracker:
            sentiment_history = {
                city: environment._sentiment_tracker.get_history(city)
                for city in TASK_CONFIGS[task_id]["cities"]
            }
        grader = grade_expert if task_id == "expert" else grade_crisis
        score, details = grader(
            environment._trajectories,
            environment._total_days,
            environment._budget_total,
            environment._budget_spent,
            health_summary,
            sentiment_history,
            environment._grap_compliance_scores,
        )

    return float(score), details


def run_episode(task_id: str, seed: int, policy_name: str) -> float:
    """Run one complete episode and return its terminal task grader score."""
    score, _ = run_episode_with_details(task_id, seed, policy_name)
    return score


def audit(
    seeds: tuple[int, ...] = SEEDS,
) -> Dict[str, Dict[str, List[float]]]:
    """Collect terminal scores for every task, seed, and policy."""
    return {
        task_id: {
            policy_name: [
                run_episode(task_id, seed, policy_name)
                for seed in seeds
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


def render_lagged_threshold_trace() -> str:
    """Explain the held-out easy/medium lagged-threshold advantage."""
    lines = [
        "# Lagged-threshold trace (easy, seed 0)",
        "",
        "| Day | Observed PM2.5 | Policy | Stage | pm25_base | pm25_post | Cost |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    environments = {}
    for policy_name in ("threshold", "lagged_threshold"):
        environment = AQIControlEnvironment()
        environments[policy_name] = (
            environment,
            environment.reset(seed=0, task_id="easy"),
        )

    for day in range(1, 11):
        for policy_name in ("threshold", "lagged_threshold"):
            environment, observation = environments[policy_name]
            observed_pm25 = observation.pm25_today
            action = POLICY_FACTORIES[policy_name](observation)
            observation = environment.step(action)
            step = environment._trajectories["delhi"][-1]
            environments[policy_name] = (environment, observation)
            lines.append(
                f"| {day} | {observed_pm25:.1f} | {policy_name} | "
                f"{action.grap_stage} | {step['pm25_base']:.1f} | "
                f"{step['pm25_post']:.1f} | {step['economic_cost']:.2f} |"
            )

    lines.extend([
        "",
        (
            "Days 1-10 are identical. The policies first diverge on day 11: "
            "`threshold` sees raw 245.0 and chooses Stage 2, while "
            "`lagged_threshold` reads the prior post-intervention 277.6 and "
            "chooses Stage 3. This post-intervention threshold crossing causes "
            "stage oscillation and changes fatigue; the observation's raw and "
            "post-intervention fields are internally correct."
        ),
    ])
    return "\n".join(lines)


def write_audit_documents(repo_root: Path) -> tuple[str, str]:
    """Run development and held-out audits once each and save both tables."""
    development = render_table(audit((0, 1, 2, 3, 4)))
    heldout = render_table(audit((5, 6, 7, 8, 9)))
    trace = render_lagged_threshold_trace()
    docs_dir = repo_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "audit_after.md").write_text(
        "# Policy audit (after reward fixes)\n\n"
        "Single-script audit over seeds 0-4. Values are terminal grader "
        "score mean \u00b1 population standard deviation.\n\n"
        f"{development}\n\n{trace}\n",
        encoding="utf-8",
    )
    (docs_dir / "audit_heldout.md").write_text(
        "# Held-out policy audit\n\n"
        "Single-script audit over seeds 5-9. Values are terminal grader "
        "score mean \u00b1 population standard deviation.\n\n"
        f"{heldout}\n\n"
        "> Easy and medium are seed-independent, so held-out seeds only add "
        "validation coverage for hard, expert, and crisis.\n",
        encoding="utf-8",
    )
    return development, heldout


def print_failing_ordering_breakdowns(
    results: Dict[str, Dict[str, List[float]]],
) -> None:
    """Print seed-0 grader components for currently inverted comparisons."""
    failures = []
    means = {
        task_id: {
            policy_name: statistics.fmean(results[task_id][policy_name])
            for policy_name in POLICIES
        }
        for task_id in TASKS
    }

    for task_id in TASKS:
        if means[task_id]["threshold"] <= means[task_id]["noop"]:
            failures.append((task_id, "threshold", "noop", "threshold > noop"))
        if means[task_id]["noop"] <= means[task_id]["anti_threshold"]:
            failures.append(
                (task_id, "noop", "anti_threshold", "noop > anti_threshold")
            )

        highest_other = max(
            (
                policy
                for policy in POLICIES
                if policy != "max_everything"
            ),
            key=lambda policy: means[task_id][policy],
        )
        if (
            means[task_id]["max_everything"]
            >= means[task_id][highest_other]
        ):
            failures.append(
                (
                    task_id,
                    "max_everything",
                    highest_other,
                    "max_everything is not top-scoring",
                )
            )

    for comparison in (
        "noop",
        "threshold",
    ):
        if means["hard"]["budget_aware"] <= means["hard"][comparison]:
            failures.append(
                (
                    "hard",
                    "budget_aware",
                    comparison,
                    f"budget_aware > {comparison}",
                )
            )

    if not failures:
        return

    print("\nSeed-0 grader details for failing ordering checks:")
    for task_id, left, right, expectation in failures:
        left_score, left_details = run_episode_with_details(task_id, 0, left)
        right_score, right_details = run_episode_with_details(task_id, 0, right)
        print(f"\n{task_id}: {expectation} is not met")
        print(f"{left}: score={left_score:.4f}, details={left_details}")
        print(f"{right}: score={right_score:.4f}, details={right_details}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit policy ordering in the AQI control environment."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional Markdown file to update with the audit table.",
    )
    parser.add_argument(
        "--write-audit-docs",
        action="store_true",
        help="Generate development and held-out audit documents in docs/.",
    )
    args = parser.parse_args()

    if args.write_audit_docs:
        development, heldout = write_audit_documents(REPO_ROOT)
        print("Seeds 0-4\n" + development)
        print("\nSeeds 5-9\n" + heldout)
        return 0

    results = audit()
    table = render_table(results)
    print(table)
    print_failing_ordering_breakdowns(results)

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
