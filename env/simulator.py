# Copyright (c) 2025. AQI Control Environment for OpenEnv.
# All rights reserved.

"""
Causal intervention simulator for PM2.5 policy planning.

Models daily PM2.5 as:
    PM2.5_post = baseline_cpcb
                 * (1 - sum_of_controllable_reductions)
                 * compliance_factor
                 * fatigue_factor

Key features:
    - Per-city configurable source shares (not hardcoded claims)
    - Compliance multiplier (city enforcement capacity)
    - Policy fatigue (consecutive strong actions become less effective)
    - Hard cap on maximum daily reduction (~30% of baseline)
    - Weekend traffic dip
    - Wind regime effects (calm inversion amplifies, strong wind ventilates)
    - Festival sensitivity (Diwali amplification)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Default source shares (used if city profile not available) ───────────

DEFAULT_SOURCE_SHARES = {
    "vehicular": 0.35,
    "construction": 0.20,
    "industrial": 0.15,
    "residential": 0.08,
    "background": 0.22,
}

# ── Intervention reduction factors per level ─────────────────────────────
# These define what fraction of the relevant source component is removed

TRAFFIC_REDUCTION = {0: 0.0, 1: 0.10, 2: 0.22, 3: 0.30}
CONSTRUCTION_REDUCTION = {0: 0.0, 1: 0.15, 2: 0.30, 3: 0.40}
INDUSTRY_REDUCTION = {0: 0.0, 1: 0.10, 2: 0.18, 3: 0.25}

# Hard cap: maximum total PM2.5 reduction in a single day
MAX_DAILY_REDUCTION_FRAC = 0.30

# ── Economic cost per day per intervention level ─────────────────────────

BASE_ECONOMIC_COST = {
    "no_action": {0: 0.0},
    "restrict_traffic": {0: 0.0, 1: 2.0, 2: 5.0, 3: 10.0},
    "curtail_construction": {0: 0.0, 1: 1.5, 2: 4.0, 3: 8.0},
    "limit_industry": {0: 0.0, 1: 3.0, 2: 6.0, 3: 12.0},
    "issue_public_advisory": {0: 0.5},
}

# ── Policy fatigue parameters ────────────────────────────────────────────

FATIGUE_BUILDUP_RATE = 0.15    # per level-3 day
FATIGUE_DECAY_RATE = 0.08      # natural decay per day without strong action
FATIGUE_MAX = 1.0
FATIGUE_EFFECTIVENESS_FLOOR = 0.5  # at max fatigue, effectiveness is halved
FATIGUE_COST_CEILING = 1.5        # at max fatigue, costs are 50% higher

# ── NAAQS / WHO standards ───────────────────────────────────────────────

NAAQS_PM25_24HR = 60.0    # India's 24-hour PM2.5 standard (ug/m3)
WHO_PM25_24HR = 15.0       # WHO guideline (for reference)

# ── Wind regime effects ─────────────────────────────────────────────────

WIND_REGIME_MULTIPLIER = {
    "calm_inversion": 1.20,    # trapped pollution amplified
    "light": 1.05,
    "normal": 1.00,
    "strong_ventilation": 0.85,  # wind disperses pollution
}

# ── AQI category thresholds ─────────────────────────────────────────────

AQI_CATEGORIES = [
    (0, 30, "Good"),
    (31, 60, "Satisfactory"),
    (61, 90, "Moderate"),
    (91, 120, "Poor"),
    (121, 250, "Very Poor"),
    (251, 999, "Severe"),
]

RISK_BUCKETS = [
    (0, 60, "low"),
    (61, 120, "moderate"),
    (121, 250, "high"),
    (251, 999, "severe"),
]


def get_aqi_category(pm25: float) -> str:
    """Return the Indian AQI category for a given PM2.5 value."""
    for low, high, cat in AQI_CATEGORIES:
        if low <= pm25 <= high:
            return cat
    return "Severe+"


def get_risk_bucket(pm25: float) -> str:
    """Return risk bucket for forecast panel."""
    for low, high, bucket in RISK_BUCKETS:
        if low <= pm25 <= high:
            return bucket
    return "severe"


def classify_wind_regime(wind_kmh: float) -> str:
    """Classify wind speed into regime categories."""
    if wind_kmh <= 2:
        return "calm_inversion"
    elif wind_kmh <= 5:
        return "light"
    elif wind_kmh <= 12:
        return "normal"
    else:
        return "strong_ventilation"


def compute_forecast(
    city_days: List[Dict[str, Any]], current_day_idx: int
) -> List[str]:
    """Generate 2-day risk forecast from data (imperfect — buckets only)."""
    forecasts = []
    for offset in [1, 2]:
        idx = current_day_idx + offset
        if idx < len(city_days):
            pm25 = city_days[idx]["pm25"]
            forecasts.append(get_risk_bucket(pm25))
        else:
            forecasts.append("unknown")
    return forecasts


def update_fatigue(
    current_fatigue: float,
    action_type: str,
    level: int,
) -> float:
    """Update the policy fatigue index after an action.

    Fatigue builds when strong actions (level >= 2) are used consecutively.
    It decays naturally when mild or no action is taken.
    """
    if action_type in ("no_action", "issue_public_advisory") or level == 0:
        # Fatigue decays
        new_fatigue = max(0.0, current_fatigue - FATIGUE_DECAY_RATE)
    elif level == 1:
        # Mild action: slight decay
        new_fatigue = max(0.0, current_fatigue - FATIGUE_DECAY_RATE * 0.5)
    elif level == 2:
        # Moderate buildup
        new_fatigue = min(FATIGUE_MAX, current_fatigue + FATIGUE_BUILDUP_RATE * 0.5)
    else:  # level == 3
        # Strong buildup
        new_fatigue = min(FATIGUE_MAX, current_fatigue + FATIGUE_BUILDUP_RATE)

    return round(new_fatigue, 3)


def apply_intervention(
    base_pm25: float,
    action_type: str,
    level: int,
    city_profile: Dict[str, Any],
    fatigue: float = 0.0,
    is_weekend: bool = False,
    wind_regime: str = "normal",
    festival_flag: bool = False,
) -> Tuple[float, float, Dict[str, float], Dict[str, float]]:
    """Apply a policy intervention to the base PM2.5 value.

    Args:
        base_pm25: Today's real PM2.5 from CPCB data (ug/m3).
        action_type: One of the 5 action types.
        level: Intervention intensity 0-3.
        city_profile: City-specific scenario parameters.
        fatigue: Current policy fatigue index (0.0-1.0).
        is_weekend: Whether today is Saturday/Sunday.
        wind_regime: Wind regime classification.
        festival_flag: Whether today is a festival day.

    Returns:
        (post_pm25, economic_cost, activity_levels, reduction_detail)
    """
    shares = city_profile.get("source_shares", DEFAULT_SOURCE_SHARES)
    compliance = city_profile.get("compliance_multiplier", 0.80)
    enforcement = city_profile.get("enforcement_capacity", 0.70)
    festival_sens = city_profile.get("festival_sensitivity", 1.0)
    weekend_traffic = city_profile.get("weekend_traffic_factor", 0.85)

    # ── Decompose PM2.5 into source components ──────────────────────────
    vehicular_pm = base_pm25 * shares.get("vehicular", 0.35)
    construction_pm = base_pm25 * shares.get("construction", 0.20)
    industrial_pm = base_pm25 * shares.get("industrial", 0.15)
    residential_pm = base_pm25 * shares.get("residential", 0.08)
    background_pm = base_pm25 * shares.get("background", 0.22)

    # ── Apply wind regime to base ────────────────────────────────────────
    wind_mult = WIND_REGIME_MULTIPLIER.get(wind_regime, 1.0)

    # ── Weekend traffic dip (natural reduction) ──────────────────────────
    if is_weekend:
        vehicular_pm *= weekend_traffic

    # ── Festival amplification ───────────────────────────────────────────
    if festival_flag:
        # Residential + background spike during festivals
        residential_pm *= festival_sens
        background_pm *= (1.0 + (festival_sens - 1.0) * 0.5)

    # ── Policy fatigue reduces effectiveness ─────────────────────────────
    fatigue_effectiveness = 1.0 - fatigue * (1.0 - FATIGUE_EFFECTIVENESS_FLOOR)

    # ── Effective compliance = base compliance * enforcement * fatigue ───
    effective_compliance = compliance * enforcement * fatigue_effectiveness

    # ── Apply interventions to controllable components ───────────────────
    activity_traffic = 1.0
    activity_construction = 1.0
    activity_industry = 1.0
    reduction_detail = {}

    if action_type == "restrict_traffic":
        raw_reduction = TRAFFIC_REDUCTION.get(level, 0.0)
        effective_reduction = raw_reduction * effective_compliance
        vehicular_pm *= (1.0 - effective_reduction)
        activity_traffic = 1.0 - raw_reduction
        reduction_detail["traffic_reduction_pct"] = round(effective_reduction * 100, 1)

    elif action_type == "curtail_construction":
        raw_reduction = CONSTRUCTION_REDUCTION.get(level, 0.0)
        effective_reduction = raw_reduction * effective_compliance
        construction_pm *= (1.0 - effective_reduction)
        activity_construction = 1.0 - raw_reduction
        reduction_detail["construction_reduction_pct"] = round(effective_reduction * 100, 1)

    elif action_type == "limit_industry":
        raw_reduction = INDUSTRY_REDUCTION.get(level, 0.0)
        effective_reduction = raw_reduction * effective_compliance
        industrial_pm *= (1.0 - effective_reduction)
        activity_industry = 1.0 - raw_reduction
        reduction_detail["industry_reduction_pct"] = round(effective_reduction * 100, 1)

    elif action_type == "issue_public_advisory":
        reduction_detail["advisory_issued"] = 1.0

    # ── Reconstruct total PM2.5 ──────────────────────────────────────────
    post_pm25_raw = (
        vehicular_pm + construction_pm + industrial_pm
        + residential_pm + background_pm
    ) * wind_mult

    # ── Hard cap: max daily reduction ────────────────────────────────────
    min_allowed = base_pm25 * (1.0 - MAX_DAILY_REDUCTION_FRAC)
    post_pm25 = max(post_pm25_raw, min_allowed)
    # Floor: can't go below background
    post_pm25 = max(post_pm25, background_pm)

    # ── Economic cost (with fatigue surcharge) ───────────────────────────
    cost_table = BASE_ECONOMIC_COST.get(action_type, {0: 0.0})
    base_cost = cost_table.get(level, cost_table.get(0, 0.0))
    fatigue_surcharge = 1.0 + fatigue * (FATIGUE_COST_CEILING - 1.0)
    econ_cost = base_cost * fatigue_surcharge

    reduction_detail["fatigue_effect"] = round(fatigue_effectiveness, 3)
    reduction_detail["compliance_effect"] = round(effective_compliance, 3)
    reduction_detail["wind_effect"] = round(wind_mult, 2)

    return (
        round(post_pm25, 1),
        round(econ_cost, 2),
        {
            "traffic": round(activity_traffic, 2),
            "construction": round(activity_construction, 2),
            "industry": round(activity_industry, 2),
        },
        reduction_detail,
    )


def compute_daily_reward(
    pm25_post: float,
    economic_cost: float,
    base_pm25: float,
    action_type: str,
    level: int,
    event: str | None,
    fatigue: float = 0.0,
    city_equity_penalty: float = 0.0,
    acted_on_forecast: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """Compute the composite daily reward with transparent breakdown.

    Components:
        1. Health cost: penalty tied to NAAQS 24hr standard (60 ug/m3)
        2. Economic cost: penalty for economic disruption
        3. Policy quality: bonus/penalty for appropriate/inappropriate action
        4. Fatigue penalty: additional cost for fatigued interventions
        5. Equity penalty: penalty for ignoring cities (multi-city tasks)
        6. Forecast bonus: reward for proactive action ahead of spikes

    Returns:
        (total_reward, breakdown_dict)
    """
    breakdown: Dict[str, float] = {}

    # ── 1. Health cost (tied to NAAQS 60 ug/m3) ─────────────────────────
    if pm25_post > NAAQS_PM25_24HR:
        health_cost = -(pm25_post - NAAQS_PM25_24HR) * 0.1
    else:
        health_cost = 1.0  # bonus for staying below standard
    # Advisory gives health benefit even without AQI change
    if action_type == "issue_public_advisory" and pm25_post > NAAQS_PM25_24HR:
        health_cost += 0.5
    breakdown["health_cost"] = round(health_cost, 2)

    # ── 2. Economic cost ─────────────────────────────────────────────────
    econ_penalty = -economic_cost
    breakdown["economic_cost"] = round(econ_penalty, 2)

    # ── 3. Policy quality ────────────────────────────────────────────────
    policy_bonus = 0.0

    # Penalize over-reaction on clean days
    if base_pm25 < NAAQS_PM25_24HR and action_type not in ("no_action", "issue_public_advisory"):
        policy_bonus -= 2.0

    # Bonus for pre-empting known spikes
    if event in ("pre_diwali", "diwali_week", "diwali") and level >= 2:
        policy_bonus += 1.5

    # Bonus for acting on severe days
    if base_pm25 > 300 and level >= 2:
        policy_bonus += 1.0

    # Penalty for inaction on very severe days
    if base_pm25 > 400 and action_type == "no_action":
        policy_bonus -= 3.0

    breakdown["policy_bonus"] = round(policy_bonus, 2)

    # ── 4. Fatigue penalty ───────────────────────────────────────────────
    fatigue_penalty = 0.0
    if fatigue > 0.3 and level >= 2:
        fatigue_penalty = -fatigue * 1.5
    breakdown["fatigue_penalty"] = round(fatigue_penalty, 2)

    # ── 5. Equity penalty (passed from environment) ──────────────────────
    breakdown["equity_penalty"] = round(-city_equity_penalty, 2)

    # ── 6. Forecast bonus ────────────────────────────────────────────────
    forecast_bonus = 0.0
    if acted_on_forecast and level >= 1:
        forecast_bonus = 0.5
    breakdown["forecast_bonus"] = round(forecast_bonus, 2)

    total = (
        health_cost + econ_penalty + policy_bonus
        + fatigue_penalty - city_equity_penalty + forecast_bonus
    )

    return round(total, 2), breakdown


# ══════════════════════════════════════════════════════════════════════════
#  Counterfactual baselines
# ══════════════════════════════════════════════════════════════════════════


def baseline_no_action(city_days: List[Dict]) -> Dict[str, float]:
    """Baseline: no intervention at all for entire episode."""
    total_health = 0.0
    severe_days = 0
    for d in city_days:
        pm25 = d["pm25"]
        if pm25 > NAAQS_PM25_24HR:
            total_health += -(pm25 - NAAQS_PM25_24HR) * 0.1
        else:
            total_health += 1.0
        if pm25 > 250:
            severe_days += 1
    return {
        "total_reward": round(total_health, 2),
        "severe_days": severe_days,
        "avg_pm25": round(sum(d["pm25"] for d in city_days) / len(city_days), 1),
        "policy": "no_action",
    }


def baseline_always_max(
    city_days: List[Dict], city_profile: Dict
) -> Dict[str, float]:
    """Baseline: always apply strongest traffic restriction (level 3)."""
    total_reward = 0.0
    total_econ = 0.0
    fatigue = 0.0
    pm25_values = []

    for d in city_days:
        pm25, econ, _, _ = apply_intervention(
            d["pm25"], "restrict_traffic", 3, city_profile,
            fatigue=fatigue, wind_regime=classify_wind_regime(d.get("wind_kmh", 5)),
        )
        reward, _ = compute_daily_reward(
            pm25, econ, d["pm25"], "restrict_traffic", 3,
            d.get("event"), fatigue=fatigue,
        )
        total_reward += reward
        total_econ += econ
        fatigue = update_fatigue(fatigue, "restrict_traffic", 3)
        pm25_values.append(pm25)

    return {
        "total_reward": round(total_reward, 2),
        "total_economic_cost": round(total_econ, 2),
        "avg_pm25": round(sum(pm25_values) / len(pm25_values), 1),
        "final_fatigue": round(fatigue, 3),
        "policy": "always_max_traffic_L3",
    }


def baseline_threshold(
    city_days: List[Dict], city_profile: Dict,
    threshold: float = 120.0,
) -> Dict[str, float]:
    """Baseline: apply traffic L2 when PM2.5 > threshold, else no action."""
    total_reward = 0.0
    total_econ = 0.0
    fatigue = 0.0
    pm25_values = []
    intervention_days = 0

    for d in city_days:
        base = d["pm25"]
        if base > threshold:
            act, lvl = "restrict_traffic", 2
            intervention_days += 1
        else:
            act, lvl = "no_action", 0

        pm25, econ, _, _ = apply_intervention(
            base, act, lvl, city_profile,
            fatigue=fatigue, wind_regime=classify_wind_regime(d.get("wind_kmh", 5)),
        )
        reward, _ = compute_daily_reward(
            pm25, econ, base, act, lvl, d.get("event"), fatigue=fatigue,
        )
        total_reward += reward
        total_econ += econ
        fatigue = update_fatigue(fatigue, act, lvl)
        pm25_values.append(pm25)

    return {
        "total_reward": round(total_reward, 2),
        "total_economic_cost": round(total_econ, 2),
        "avg_pm25": round(sum(pm25_values) / len(pm25_values), 1),
        "intervention_days": intervention_days,
        "policy": f"threshold_{int(threshold)}",
    }
