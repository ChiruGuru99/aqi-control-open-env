# Copyright (c) 2025. AQI Control Environment for OpenEnv.
# All rights reserved.

"""
India's Graded Response Action Plan (GRAP) implementation.

GRAP is the real emergency response framework used by the Commission for
Air Quality Management (CAQM) in the NCR region. It defines four escalating
stages of action based on AQI/PM2.5 levels.

This module:
    - Maps GRAP stages to bundles of individual sector interventions
    - Enforces mandatory minimum durations per stage
    - Scores agents on protocol compliance (rewarding appropriate GRAP usage)
    - Handles stage transitions with hysteresis (prevents rapid flip-flopping)

Reference: CAQM GRAP Revised Framework, 2023.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# ── GRAP Stage Definitions ───────────────────────────────────────────────

GRAP_STAGES = {
    0: {
        "name": "Normal",
        "trigger_pm25": 0,
        "actions": [],
        "description": "No emergency measures active.",
    },
    1: {
        "name": "Stage I — Poor",
        "trigger_pm25": 60,
        "actions": [
            {"action_type": "issue_public_advisory", "level": 1},
            {"action_type": "deploy_smog_response", "level": 1},
        ],
        "description": (
            "Routine measures: health advisories, mechanized road sweeping, "
            "dust suppression at construction sites, enforce PUC norms."
        ),
    },
    2: {
        "name": "Stage II — Very Poor",
        "trigger_pm25": 120,
        "actions": [
            {"action_type": "issue_public_advisory", "level": 2},
            {"action_type": "curtail_construction", "level": 2},
            {"action_type": "deploy_smog_response", "level": 2},
            {"action_type": "restrict_traffic", "level": 1},
        ],
        "description": (
            "Escalated response: ban on coal/firewood, intensified road sweeping, "
            "construction dust control, augment public transport."
        ),
    },
    3: {
        "name": "Stage III — Severe",
        "trigger_pm25": 250,
        "actions": [
            {"action_type": "restrict_traffic", "level": 2},
            {"action_type": "curtail_construction", "level": 3},
            {"action_type": "limit_industry", "level": 2},
            {"action_type": "deploy_smog_response", "level": 3},
            {"action_type": "issue_public_advisory", "level": 3},
        ],
        "description": (
            "Emergency measures: odd-even vehicle rationing, halt non-essential "
            "construction, industrial emission limits, water sprinkling."
        ),
    },
    4: {
        "name": "Stage IV — Severe+",
        "trigger_pm25": 300,
        "actions": [
            {"action_type": "restrict_traffic", "level": 3},
            {"action_type": "curtail_construction", "level": 3},
            {"action_type": "limit_industry", "level": 3},
            {"action_type": "close_schools", "level": 3},
            {"action_type": "mandate_wfh", "level": 2},
            {"action_type": "emergency_transport_ban", "level": 2},
            {"action_type": "deploy_smog_response", "level": 3},
            {"action_type": "issue_public_advisory", "level": 3},
        ],
        "description": (
            "Total emergency lockdown: truck entry ban, school closures, "
            "50% WFH for offices, stop all construction, industrial shutdown, "
            "ban entry of non-essential vehicles."
        ),
    },
}

# Minimum days a GRAP stage must remain active before de-escalation
GRAP_MIN_DURATION = {0: 0, 1: 1, 2: 2, 3: 2, 4: 3}

# Hysteresis: PM2.5 must drop below this fraction of the trigger
# threshold before the stage can be de-escalated
GRAP_DEESCALATION_FACTOR = 0.85


def get_recommended_grap_stage(pm25: float) -> int:
    """Return the GRAP stage recommended for a given PM2.5 level.

    Uses standard CAQM thresholds.
    """
    if pm25 >= 300:
        return 4
    elif pm25 >= 250:
        return 3
    elif pm25 >= 120:
        return 2
    elif pm25 >= 60:
        return 1
    return 0


def expand_grap_stage(stage: int) -> List[Dict[str, Any]]:
    """Expand a GRAP stage number into a list of individual actions.

    Returns:
        List of action dicts, each with 'action_type' and 'level'.
    """
    stage = max(0, min(4, stage))
    return list(GRAP_STAGES[stage]["actions"])


def can_deescalate(
    current_stage: int,
    days_at_current_stage: int,
    current_pm25: float,
) -> bool:
    """Check whether the agent is allowed to de-escalate the GRAP stage.

    Returns False if minimum duration has not been met or if PM2.5 is still
    above the hysteresis threshold.
    """
    if current_stage <= 0:
        return True

    min_days = GRAP_MIN_DURATION.get(current_stage, 1)
    if days_at_current_stage < min_days:
        return False

    # Hysteresis: PM2.5 must be sufficiently below the trigger threshold
    trigger = GRAP_STAGES[current_stage]["trigger_pm25"]
    deesc_threshold = trigger * GRAP_DEESCALATION_FACTOR
    return current_pm25 < deesc_threshold


def compute_grap_compliance_score(
    recommended_stage: int,
    actual_stage: int,
    pm25: float,
) -> Tuple[float, str]:
    """Score how well the agent followed GRAP protocol.

    Returns:
        (score, explanation) where score is in [-3.0, +2.0].
        Positive = good compliance.  Negative = over/under-reaction.
    """
    diff = actual_stage - recommended_stage

    if diff == 0:
        return 2.0, "Perfect GRAP compliance"
    elif diff == 1:
        # Slightly over-reacting — mildly penalized
        return 0.5, "Slightly over-escalated vs GRAP recommendation"
    elif diff == -1:
        # Slightly under-reacting
        return -0.5, "Slightly under-escalated vs GRAP recommendation"
    elif diff >= 2:
        # Severe over-reaction
        return -2.0, f"Severe over-escalation: Stage {actual_stage} vs recommended {recommended_stage}"
    else:
        # Severe under-reaction (diff <= -2)
        penalty = -1.5 * abs(diff)
        return max(-3.0, penalty), (
            f"Dangerous under-escalation: Stage {actual_stage} vs "
            f"recommended {recommended_stage} (PM2.5={pm25:.0f})"
        )


class GRAPTracker:
    """Track GRAP stage transitions per city across an episode."""

    def __init__(self, cities: List[str]):
        self._current_stage: Dict[str, int] = {c: 0 for c in cities}
        self._days_at_stage: Dict[str, int] = {c: 0 for c in cities}
        self._transition_log: Dict[str, List[Dict[str, Any]]] = {c: [] for c in cities}

    def get_stage(self, city: str) -> int:
        return self._current_stage.get(city, 0)

    def get_days_at_stage(self, city: str) -> int:
        return self._days_at_stage.get(city, 0)

    def update(
        self, city: str, requested_stage: Optional[int], pm25: float, day: int,
    ) -> Tuple[int, str]:
        """Process a GRAP stage update request.

        Args:
            city: City identifier.
            requested_stage: The stage the agent wants (None = keep current).
            pm25: Today's PM2.5.
            day: Current episode day.

        Returns:
            (effective_stage, message) — the stage actually applied.
        """
        current = self._current_stage.get(city, 0)
        days = self._days_at_stage.get(city, 0)

        if requested_stage is None:
            # No change requested, increment days
            self._days_at_stage[city] = days + 1
            return current, f"{city}: GRAP Stage {current} maintained (day {days + 1})"

        requested_stage = max(0, min(4, requested_stage))

        if requested_stage > current:
            # Escalation is always allowed
            self._current_stage[city] = requested_stage
            self._days_at_stage[city] = 1
            self._transition_log[city].append({
                "day": day, "from": current, "to": requested_stage,
                "pm25": pm25, "type": "escalation",
            })
            return requested_stage, (
                f"{city}: GRAP escalated {current}→{requested_stage} "
                f"({GRAP_STAGES[requested_stage]['name']})"
            )

        elif requested_stage < current:
            # De-escalation — check constraints
            if can_deescalate(current, days, pm25):
                self._current_stage[city] = requested_stage
                self._days_at_stage[city] = 1
                self._transition_log[city].append({
                    "day": day, "from": current, "to": requested_stage,
                    "pm25": pm25, "type": "deescalation",
                })
                return requested_stage, (
                    f"{city}: GRAP de-escalated {current}→{requested_stage}"
                )
            else:
                # Cannot de-escalate yet
                self._days_at_stage[city] = days + 1
                min_d = GRAP_MIN_DURATION.get(current, 1)
                return current, (
                    f"{city}: GRAP de-escalation blocked — "
                    f"minimum {min_d} days required (current: {days + 1})"
                )
        else:
            # Same stage
            self._days_at_stage[city] = days + 1
            return current, f"{city}: GRAP Stage {current} maintained"

    def get_transition_log(self, city: str) -> List[Dict[str, Any]]:
        return self._transition_log.get(city, [])
