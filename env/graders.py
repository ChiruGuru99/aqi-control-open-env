from __future__ import annotations

# Copyright (c) 2025. AQI Control Environment for OpenEnv.
# All rights reserved.

def grade_episode(
    task_name: str,
    arrived: bool,
    elapsed: int,
    deadline: int,
    safety_penalties: float,
    baseline_time: int,
) -> float:
    """
    Compute graded score for a completed episode.
    Returns a float in [0.0, 1.0].
    """
    if not arrived:
        return 0.01

    timeliness = max(0.0, (deadline - elapsed) / deadline)
    safety = max(0.0, 1.0 - safety_penalties)
    efficiency = min(1.0, baseline_time / max(1, elapsed)) if baseline_time < 999999 else 0.5

    # Weight components based on what each task tests
    if task_name == "easy":
        score = 0.50 * timeliness + 0.20 * safety + 0.30 * efficiency
    elif task_name == "medium":
        score = 0.40 * timeliness + 0.20 * safety + 0.40 * efficiency
    elif task_name == "hard":
        score = 0.30 * timeliness + 0.30 * safety + 0.40 * efficiency
    else:
        # Default balanced weights
        score = 0.50 * timeliness + 0.30 * safety + 0.20 * efficiency

    return round(max(0.01, min(0.99, score)), 4)
"""
Grader functions for the PM2.5 policy planning tasks.

Each grader scores 0.0–1.0 based on the agent's policy trajectory.
Terminal score is a transparent weighted sum of:
    - Average PM2.5 reduction vs baseline
    - Severe-day reduction (how well spikes were mitigated)
    - Protocol/budget discipline
    - Intervention efficiency
    - For multi-city tasks: City equity (penalizes uneven outcomes)
"""

from typing import Any, Dict, List, Tuple

from .simulator import NAAQS_PM25_24HR


def _compute_core_metrics(trajectory: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute base metrics for a single city's trajectory."""
    if not trajectory:
        return {"avg_reduction": 0.0, "severe_reduction": 0.0, "efficiency": 0.0}

    base_sum = 0.0
    post_sum = 0.0
    severe_base_sum = 0.0
    severe_post_sum = 0.0
    total_cost = 0.0
    total_health_saving = 0.0

    for t in trajectory:
        b = t.get("pm25_base", 0.0)
        p = t.get("pm25_post", 0.0)
        cost = t.get("economic_cost", 0.0)
        
        base_sum += b
        post_sum += p
        total_cost += cost
        
        # Calculate health cost saving. Health cost before vs after.
        # Health cost is -(pm25 - 60) * 0.1 if > 60, else 1.0
        def hc(pm25):
            return -(pm25 - NAAQS_PM25_24HR) * 0.1 if pm25 > NAAQS_PM25_24HR else 1.0
        
        total_health_saving += (hc(p) - hc(b))

        # Severe days (baseline > 250)
        if b > 250:
            severe_base_sum += b
            severe_post_sum += p

    avg_reduction = (base_sum - post_sum) / base_sum if base_sum > 0 else 0.0
    
    if severe_base_sum > 0:
        severe_reduction = (severe_base_sum - severe_post_sum) / severe_base_sum
    else:
        severe_reduction = 1.0  # No severe days to handle

    # Efficiency: health savings relative to economic cost
    # Bound between 0 and 1. If we saved health with little cost, good.
    if total_cost > 0:
        eff = total_health_saving / (total_cost + 1.0)
        efficiency = max(0.0, min(1.0, 0.5 + eff * 0.1)) # Center around 0.5 for neutral
    else:
        efficiency = 1.0 if total_health_saving >= 0 else 0.0

    return {
        "avg_reduction": avg_reduction,
        "severe_reduction": severe_reduction,
        "efficiency": efficiency,
    }


def grade_easy(trajectory: List[Dict[str, Any]], total_days: int) -> Tuple[float, Dict[str, Any]]:
    """Grade Task 1 (Easy): Single city, 30 days.

    Scoring:
        - 40% Average PM2.5 reduction
        - 30% Severe-day reduction
        - 30% Intervention efficiency
    """
    metrics = _compute_core_metrics(trajectory)
    
    # Scale reductions to sensible 0-1 bounds (max typical reduction is ~25-30%)
    avg_score = min(1.0, metrics["avg_reduction"] / 0.25)
    severe_score = min(1.0, metrics["severe_reduction"] / 0.30)
    eff_score = metrics["efficiency"]

    score = avg_score * 0.40 + severe_score * 0.30 + eff_score * 0.30

    details = {
        "avg_reduction_pct": round(metrics["avg_reduction"] * 100, 1),
        "severe_reduction_pct": round(metrics["severe_reduction"] * 100, 1),
        "efficiency_score": round(eff_score, 4),
        "component_avg": round(avg_score, 4),
        "component_severe": round(severe_score, 4),
    }

    # Ensure returned score is strictly between 0 and 1 and stays
    # safely above common rounding thresholds used by validators.
    return max(0.001, min(round(score, 4), 0.999)), details


def grade_medium(
    city_trajectories: Dict[str, List[Dict[str, Any]]], total_days: int,
) -> Tuple[float, Dict[str, Any]]:
    """Grade Task 2 (Medium): Multiple cities, includes Diwali.

    Scoring:
        - 30% Average PM2.5 reduction
        - 30% Severe-day reduction
        - 20% Intervention efficiency
        - 20% City equity (penalizes uneven health outcomes)
    """
    if not city_trajectories:
        return 0.0001, {"error": "empty trajectories"}

    city_metrics = {}
    health_scores = []

    for city, traj in city_trajectories.items():
        m = _compute_core_metrics(traj)
        city_metrics[city] = m
        
        # Calculate days below standard as health proxy for equity
        days_below = sum(1 for t in traj if t.get("pm25_post", 999) <= NAAQS_PM25_24HR)
        health_scores.append(days_below / len(traj) if len(traj) > 0 else 0.0)

    # Averages across cities
    avg_reduction = sum(m["avg_reduction"] for m in city_metrics.values()) / len(city_metrics)
    severe_reduction = sum(m["severe_reduction"] for m in city_metrics.values()) / len(city_metrics)
    efficiency = sum(m["efficiency"] for m in city_metrics.values()) / len(city_metrics)

    avg_score = min(1.0, avg_reduction / 0.25)
    severe_score = min(1.0, severe_reduction / 0.30)

    # Equity: standard deviation of health scores across cities
    mean_h = sum(health_scores) / len(health_scores) if health_scores else 0.0
    variance = sum((h - mean_h) ** 2 for h in health_scores) / len(health_scores) if health_scores else 0.0
    std_dev = variance ** 0.5
    equity_score = max(0.0, 1.0 - (std_dev * 3.0))  # Penetrate score if std dev is high

    score = avg_score * 0.30 + severe_score * 0.30 + efficiency * 0.20 + equity_score * 0.20

    # Keep scores within (0, 1) and avoid values that round to 0.0.
    return max(0.001, min(round(score, 4), 0.999)), {
        "avg_reduction_pct": round(avg_reduction * 100, 1),
        "severe_reduction_pct": round(severe_reduction * 100, 1),
        "efficiency_score": round(efficiency, 4),
        "equity_score": round(equity_score, 4),
        "city_health_rates": {c: round(h, 3) for c, h in zip(city_trajectories.keys(), health_scores)}
    }


def grade_hard(
    city_trajectories: Dict[str, List[Dict[str, Any]]], total_days: int,
    budget_total: float, budget_spent: float,
) -> Tuple[float, Dict[str, Any]]:
    """Grade Task 3 (Hard): Multi-city with budget constraints.

    Scoring:
        - 25% Average PM2.5 reduction
        - 20% Severe-day reduction
        - 15% Intervention efficiency
        - 15% City equity
        - 25% Budget discipline
    """
    med_score, med_details = grade_medium(city_trajectories, total_days)

    # Budget discipline
    if budget_total > 0:
        budget_ratio = budget_spent / budget_total
        if budget_ratio <= 1.0:
            budget_score = 1.0 - (budget_ratio * 0.2)  # reward efficiency
        else:
            budget_score = max(0.0, 1.0 - (budget_ratio - 1.0) * 3.0)  # harsh penalty for over-spending
    else:
        budget_score = 1.0

    # Extract medium components (they were scaled by 0.3, 0.3, 0.2, 0.2)
    # We can reconstruct them approximately from the details
    avg_red = med_details["avg_reduction_pct"] / 100.0
    sev_red = med_details["severe_reduction_pct"] / 100.0
    avg_score = min(1.0, avg_red / 0.25)
    severe_score = min(1.0, sev_red / 0.30)
    eff_score = med_details["efficiency_score"]
    equity_score = med_details["equity_score"]

    score = (
        avg_score * 0.25
        + severe_score * 0.20
        + eff_score * 0.15
        + equity_score * 0.15
        + budget_score * 0.25
    )

    details = dict(med_details)
    details["budget_score"] = round(budget_score, 4)
    details["budget_spent"] = round(budget_spent, 2)
    details["budget_total"] = round(budget_total, 2)

    return max(0.001, min(round(score, 4), 0.999)), details

def compute_city_equity_penalty(city_trajectories: Dict[str, List[Dict[str, Any]]], current_day: int) -> float:
    """Computes a daily penalty if cities are highly unequal in health consequences.
    Only calculated if multiple cities have data.
    """
    if len(city_trajectories) < 2:
        return 0.0
    
    # Calculate health cost running average for each city
    health_costs = []
    for city, traj in city_trajectories.items():
        if not traj:
            continue
        total_hc = sum(t.get("health_cost", 0) for t in traj)
        health_costs.append(total_hc / len(traj))
    
    if len(health_costs) < 2:
        return 0.0

    mean_hc = sum(health_costs) / len(health_costs)
    variance = sum((hc - mean_hc) ** 2 for hc in health_costs) / len(health_costs)
    std_dev = variance ** 0.5
    
    # If the standard deviation is high, penalty applies
    # Note: health costs are negative (penalties), so we just look at magnitude of variation.
    if std_dev > 5.0:
        return min((std_dev - 5.0) * 0.5, 10.0) # Cap penalty
    return 0.0
