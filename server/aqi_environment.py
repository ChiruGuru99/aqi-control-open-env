# Copyright (c) 2025. AQI Control Environment for OpenEnv.
# All rights reserved.

"""
AQI Control Environment Implementation.

A multi-task benchmark where an AI agent acts as a city AQI policy controller
during Indian winter seasons under NCAP constraints. Three difficulty tiers:
  - Easy:   single city (Delhi), 30 days
  - Medium: two cities (Delhi + Lucknow), 30 days, includes Diwali
  - Hard:   three cities, 30 days, budget-constrained, festival events
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import AQIAction, AQIObservation, RewardBreakdown
except ImportError:
    from models import AQIAction, AQIObservation, RewardBreakdown

try:
    from ..env.simulator import (
        apply_intervention, compute_daily_reward, get_aqi_category,
        classify_wind_regime, compute_forecast, update_fatigue
    )
    from ..env.graders import grade_easy, grade_medium, grade_hard, compute_city_equity_penalty
except ImportError:
    from env.simulator import (
        apply_intervention, compute_daily_reward, get_aqi_category,
        classify_wind_regime, compute_forecast, update_fatigue
    )
    from env.graders import grade_easy, grade_medium, grade_hard, compute_city_equity_penalty


_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ── Task configurations ─────────────────────────────────────────────────

TASK_CONFIGS = {
    "easy": {
        "cities": ["delhi"],
        "total_days": 30,
        "budget": -1.0,       # unlimited
        "description": "Single city (Delhi), 30-day winter. Basic AQI management.",
    },
    "medium": {
        "cities": ["delhi", "lucknow"],
        "total_days": 30,
        "budget": -1.0,
        "description": "Two cities including Diwali spike. Multi-city coordination and equity.",
    },
    "hard": {
        "cities": ["delhi", "lucknow", "patna"],
        "total_days": 30,
        "budget": 150.0,      # constrained
        "description": "Three cities, budget-constrained, festival events, fatigue management.",
    },
}

CITY_FILES = {
    "delhi": "delhi_winter.json",
    "lucknow": "lucknow_winter.json",
    "patna": "patna_winter.json",
}


class AQIControlEnvironment(Environment):
    """
    OpenEnv environment simulating Indian city PM2.5 policy planning.

    Episode flow:
        reset(task_id) → day 1 observation
        step(action)   → day N+1 observation  (repeat until total_days reached)
        done=True at end → grader scores in metadata
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # ── Episode state ────────────────────────────────────────────────
        self._task_id = "easy"
        self._total_days = 30
        self._current_day = 0
        self._done = False
        self._budget_total = -1.0
        self._budget_spent = 0.0

        # ── Data (pre-load ALL cities so stateless HTTP works) ──────
        self._all_city_data: Dict[str, List[Dict[str, Any]]] = {}
        for city, filename in CITY_FILES.items():
            path = _DATA_DIR / filename
            with open(path, "r", encoding="utf-8") as f:
                self._all_city_data[city] = json.load(f).get("days", [])

        # Load city profiles
        profile_path = _DATA_DIR / "city_profiles.json"
        self._city_profiles: Dict[str, Any] = {}
        if profile_path.exists():
            with open(profile_path, "r", encoding="utf-8") as f:
                self._city_profiles = json.load(f)

        # Active city data for current task
        self._city_data: Dict[str, List[Dict[str, Any]]] = {
            "delhi": self._all_city_data["delhi"],
        }
        self._primary_city = "delhi"

        # ── Trajectory tracking (for graders) ────────────────────────────
        self._trajectories: Dict[str, List[Dict[str, Any]]] = {"delhi": []}
        # Per-city post-intervention PM2.5 history
        self._pm25_histories: Dict[str, List[float]] = {"delhi": []}
        
        # ── Policy tracking ──────────────────────────────────────────────
        self._fatigue_levels: Dict[str, float] = {"delhi": 0.0}

        # ── Cumulative costs ─────────────────────────────────────────────
        self._cumulative_health_cost = 0.0
        self._cumulative_economic_cost = 0.0
        self._cumulative_reward = 0.0
        
        # Last breakdown
        self._last_reward_breakdown = {}

        # ── Activity levels per city ─────────────────────────────────────
        self._activity_levels: Dict[str, Dict[str, float]] = {
            "delhi": {"traffic": 1.0, "construction": 1.0, "industry": 1.0},
        }

        # ── Grader scores ────────────────────────────────────────────────
        self._grader_scores: Dict[str, float] = {}

    def _load_city(self, city: str) -> List[Dict[str, Any]]:
        """Load CPCB data for a city from the pre-loaded cache."""
        return self._all_city_data.get(city, [])

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AQIObservation:
        """Reset the environment for a new episode."""
        eid = episode_id or str(uuid4())
        self._state = State(episode_id=eid, step_count=0)

        self._task_id = kwargs.get("task_id", "easy")
        if self._task_id not in TASK_CONFIGS:
            self._task_id = "easy"

        config = TASK_CONFIGS[self._task_id]
        self._total_days = config["total_days"]
        self._budget_total = config["budget"]
        self._budget_spent = 0.0
        self._current_day = 0
        self._done = False

        self._city_data = {}
        for city in config["cities"]:
            self._city_data[city] = self._load_city(city)
        self._primary_city = config["cities"][0]

        self._trajectories = {city: [] for city in config["cities"]}
        self._pm25_histories = {city: [] for city in config["cities"]}
        self._fatigue_levels = {city: 0.0 for city in config["cities"]}
        self._activity_levels = {
            city: {"traffic": 1.0, "construction": 1.0, "industry": 1.0}
            for city in config["cities"]
        }
        self._cumulative_health_cost = 0.0
        self._cumulative_economic_cost = 0.0
        self._cumulative_reward = 0.0
        self._last_reward_breakdown = {}
        self._grader_scores = {}

        return self._make_observation(
            reward=0.0,
            message=(
                f"Episode started: {config['description']} "
                f"Cities: {', '.join(c.title() for c in config['cities'])}. "
                f"Choose daily interventions."
            ),
        )

    def _is_weekend(self, day_idx: int) -> bool:
        """Simple weekend heuristic based on start day (Oct 15, 2024 is Tuesday).
        Day 0 = Tuesday. Thus Days 4 and 5 are weekend."""
        return (day_idx % 7) in [4, 5]

    def _acted_on_forecast(self, action_type: str, level: int, base_pm25: float, forecast: List[str]) -> bool:
        """Heuristic to check if agent acted proactively based on severe forecast on a clean-ish day."""
        if base_pm25 < 120 and level >= 1:
            if "severe" in forecast or "high" in forecast:
                return True
        return False

    def step(
        self,
        action: AQIAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> AQIObservation:
        """Execute one day of policy action."""
        self._state.step_count += 1
        self._current_day += 1

        if self._done:
            return self._make_observation(
                reward=0.0,
                message="Episode already ended.",
            )

        config = TASK_CONFIGS[self._task_id]
        cities = config["cities"]

        target_city = action.city.lower() if action.city else self._primary_city
        if target_city not in cities:
            target_city = self._primary_city

        day_idx = self._current_day - 1

        total_reward = 0.0
        total_health = 0.0
        total_econ = 0.0
        day_message_parts = []
        breakdown_agg = {}

        # Pre-compute equity penalty based on states up to yesterday
        equity_penalty = compute_city_equity_penalty(self._trajectories, self._current_day)

        for city in cities:
            city_days = self._city_data[city]
            if day_idx >= len(city_days):
                continue

            day_data = city_days[day_idx]
            base_pm25 = day_data["pm25"]
            event = day_data.get("event") or ""
            wind_kmh = day_data.get("wind_kmh", 5)
            wind_regime = classify_wind_regime(wind_kmh)
            festival_flag = "diwali" in event
            is_weekend = self._is_weekend(day_idx)
            forecast = compute_forecast(city_days, day_idx)

            if city == target_city:
                act_type = action.action_type
                act_level = action.level
            else:
                act_type = "no_action"
                act_level = 0
                
            fatigue_current = self._fatigue_levels[city]
            city_profile = self._city_profiles.get(city, {})

            acted_on_forc = False
            if city == target_city:
                acted_on_forc = self._acted_on_forecast(act_type, act_level, base_pm25, forecast)

            # Apply action
            post_pm25, econ_cost, activities, red_details = apply_intervention(
                base_pm25, act_type, act_level, city_profile,
                fatigue=fatigue_current, is_weekend=is_weekend, 
                wind_regime=wind_regime, festival_flag=festival_flag
            )

            # Update fatigue for tomorrow
            self._fatigue_levels[city] = update_fatigue(fatigue_current, act_type, act_level)

            # Compute reward
            reward, breakdown = compute_daily_reward(
                post_pm25, econ_cost, base_pm25, act_type, act_level, event,
                fatigue=fatigue_current, city_equity_penalty=equity_penalty if city == self._primary_city else 0.0,
                acted_on_forecast=acted_on_forc
            )

            if self._budget_total > 0:
                self._budget_spent += econ_cost

            # Track
            self._activity_levels[city] = activities
            self._pm25_histories[city].append(post_pm25)
            self._trajectories[city].append({
                "day": self._current_day,
                "pm25_base": base_pm25,
                "pm25_post": post_pm25,
                "action_type": act_type,
                "level": act_level,
                "reward": reward,
                "economic_cost": econ_cost,
                "health_cost": breakdown["health_cost"],
                "event": event,
                **breakdown,
            })

            total_reward += reward
            total_health += breakdown["health_cost"]
            total_econ += econ_cost
            
            # Aggregate breakdown for primary observation
            if city == self._primary_city:
                breakdown_agg = breakdown

            aqi_cat = get_aqi_category(post_pm25)
            day_message_parts.append(
                f"{city.title()}: {base_pm25}→{post_pm25} ({aqi_cat})"
            )

        self._cumulative_health_cost += total_health
        self._cumulative_economic_cost += total_econ
        self._cumulative_reward += total_reward
        self._last_reward_breakdown = breakdown_agg

        if self._current_day >= self._total_days:
            self._done = True
            self._run_graders()

        msg_parts = [f"Day {self._current_day}/{self._total_days}"]
        msg_parts.extend(day_message_parts)
        if self._budget_total > 0:
            remaining = self._budget_total - self._budget_spent
            msg_parts.append(f"Budget: {remaining:.1f}/{self._budget_total:.1f}")
        if self._done:
            scores = " | ".join(f"{k}: {v:.4f}" for k, v in self._grader_scores.items())
            msg_parts.append(f"EPISODE COMPLETE. Scores: {scores}")

        return self._make_observation(
            reward=round(total_reward, 2),
            message=" | ".join(msg_parts),
        )

    @property
    def state(self) -> State:
        return State(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
            task_id=self._task_id,
            current_day=self._current_day,
            total_days=self._total_days,
            done=self._done,
            grader_scores=self._grader_scores,
            cumulative_reward=round(self._cumulative_reward, 2),
            budget_spent=round(self._budget_spent, 2),
            budget_total=self._budget_total,
        )

    def _run_graders(self) -> None:
        if self._task_id == "easy":
            traj = self._trajectories.get(self._primary_city, [])
            score, details = grade_easy(traj, self._total_days)
            self._grader_scores = {"easy": score}
        elif self._task_id == "medium":
            score, details = grade_medium(self._trajectories, self._total_days)
            self._grader_scores = {"medium": score}
        elif self._task_id == "hard":
            score, details = grade_hard(
                self._trajectories,
                self._total_days,
                self._budget_total,
                self._budget_spent,
            )
            self._grader_scores = {"hard": score}

    def _make_observation(self, reward: float, message: str) -> AQIObservation:
        config = TASK_CONFIGS[self._task_id]
        cities = config["cities"]
        day_idx = max(0, self._current_day - 1)

        primary_days = self._city_data.get(self._primary_city, [])
        if day_idx < len(primary_days):
            today = primary_days[day_idx]
        else:
            today = primary_days[-1] if primary_days else {}

        base_pm25 = today.get("pm25", 0)
        history = self._pm25_histories.get(self._primary_city, [])[-7:]
        post_pm25 = history[-1] if history else base_pm25

        forecast_2d = compute_forecast(primary_days, day_idx)
        wind_kmh = float(today.get("wind_kmh", 5.0))
        wind_regime = classify_wind_regime(wind_kmh)
        is_weekend = self._is_weekend(day_idx)
        festival_flag = "diwali" in str(today.get("event", ""))

        cities_obs: Dict[str, Any] = {}
        if len(cities) > 1:
            for city in cities:
                c_days = self._city_data.get(city, [])
                c_today = c_days[day_idx] if day_idx < len(c_days) else {}
                c_hist = self._pm25_histories.get(city, [])[-7:]
                c_acts = self._activity_levels.get(city, {})
                c_forecast = compute_forecast(c_days, day_idx)
                c_wind = float(c_today.get("wind_kmh", 5.0))
                cities_obs[city] = {
                    "pm25_today": c_today.get("pm25", 0),
                    "pm25_post": c_hist[-1] if c_hist else c_today.get("pm25", 0),
                    "pm25_history": c_hist,
                    "temperature_c": c_today.get("temp_c", 0),
                    "wind_speed_kmh": c_wind,
                    "humidity_pct": c_today.get("humidity_pct", 0),
                    "stubble_flag": c_today.get("stubble_flag", 0),
                    "wind_regime": classify_wind_regime(c_wind),
                    "event": c_today.get("event") or "",
                    "forecast_risk_next_2d": c_forecast,
                    "policy_fatigue_index": self._fatigue_levels.get(city, 0.0),
                    "traffic_level": c_acts.get("traffic", 1.0),
                    "construction_level": c_acts.get("construction", 1.0),
                    "industry_level": c_acts.get("industry", 1.0),
                }

        acts = self._activity_levels.get(self._primary_city, {})
        budget_remaining = self._budget_total - self._budget_spent if self._budget_total > 0 else -1.0

        return AQIObservation(
            day=self._current_day,
            date=today.get("date", ""),
            city=self._primary_city.title(),
            pm25_today=float(base_pm25),
            pm25_post_intervention=float(post_pm25),
            pm25_history=history,
            temperature_c=float(today.get("temp_c", 0)),
            wind_speed_kmh=wind_kmh,
            humidity_pct=float(today.get("humidity_pct", 0)),
            stubble_burning_flag=int(today.get("stubble_flag", 0)),
            wind_regime=wind_regime,
            event=today.get("event") or "",
            festival_flag=festival_flag,
            is_weekend=is_weekend,
            forecast_risk_next_2d=forecast_2d,
            traffic_level=acts.get("traffic", 1.0),
            construction_level=acts.get("construction", 1.0),
            industry_level=acts.get("industry", 1.0),
            policy_fatigue_index=round(self._fatigue_levels.get(self._primary_city, 0.0), 3),
            reward_breakdown=self._last_reward_breakdown,
            total_days=self._total_days,
            budget_remaining=round(budget_remaining, 1),
            cumulative_health_cost=round(self._cumulative_health_cost, 2),
            cumulative_economic_cost=round(self._cumulative_economic_cost, 2),
            task_id=self._task_id,
            cities_data=cities_obs,
            message=message,
            done=self._done,
            reward=reward,
            metadata={"grader_scores": dict(self._grader_scores)},
            grader_scores=dict(self._grader_scores),
        )
