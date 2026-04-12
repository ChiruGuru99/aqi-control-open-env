# Copyright (c) 2025. AQI Control Environment for OpenEnv.
# All rights reserved.

"""
AQI Control Environment Implementation.

A multi-task benchmark where an AI agent acts as a city AQI policy controller
during Indian winter seasons under NCAP constraints. Five difficulty tiers:
  - Easy:   single city (Delhi), 30 days
  - Medium: two cities (Delhi + Lucknow), 30 days, includes Diwali
  - Hard:   three cities, 30 days, budget-constrained, festival events
  - Expert: five cities, 45 days, GRAP protocol, inter-city transport,
            health impacts, public sentiment
  - Crisis: eight cities, 15-day emergency, hospital constraints,
            unprecedented pollution event

Integrates:
  - GRAP (Graded Response Action Plan) 4-stage protocol
  - Inter-city pollution transport model
  - Stochastic weather with forecast uncertainty
  - Public health impact tracking (DALYs, hospital admissions, school closures)
  - Public sentiment / political economy dynamics
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
        apply_intervention, apply_multi_action, compute_daily_reward,
        get_aqi_category, classify_wind_regime, compute_forecast, update_fatigue,
    )
    from ..env.graders import (
        grade_easy, grade_medium, grade_hard, grade_expert, grade_crisis,
        compute_city_equity_penalty,
    )
    from ..env.grap import (
        GRAPTracker, get_recommended_grap_stage, expand_grap_stage,
        compute_grap_compliance_score,
    )
    from ..env.transport import TransportModel
    from ..env.weather import WeatherEngine, compute_noisy_forecast
    from ..env.health import HealthTracker, get_school_status
    from ..env.politics import SentimentTracker
except ImportError:
    from env.simulator import (
        apply_intervention, apply_multi_action, compute_daily_reward,
        get_aqi_category, classify_wind_regime, compute_forecast, update_fatigue,
    )
    from env.graders import (
        grade_easy, grade_medium, grade_hard, grade_expert, grade_crisis,
        compute_city_equity_penalty,
    )
    from env.grap import (
        GRAPTracker, get_recommended_grap_stage, expand_grap_stage,
        compute_grap_compliance_score,
    )
    from env.transport import TransportModel
    from env.weather import WeatherEngine, compute_noisy_forecast
    from env.health import HealthTracker, get_school_status
    from env.politics import SentimentTracker


_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ── Task configurations ─────────────────────────────────────────────────

TASK_CONFIGS = {
    "easy": {
        "cities": ["delhi"],
        "total_days": 30,
        "budget": -1.0,       # unlimited
        "description": "Single city (Delhi), 30-day winter. Basic AQI management.",
        "enable_transport": False,
        "enable_health": False,
        "enable_sentiment": False,
        "enable_grap": False,
        "enable_weather_engine": False,
    },
    "medium": {
        "cities": ["delhi", "lucknow"],
        "total_days": 30,
        "budget": -1.0,
        "description": "Two cities including Diwali spike. Multi-city coordination and equity.",
        "enable_transport": False,
        "enable_health": False,
        "enable_sentiment": False,
        "enable_grap": False,
        "enable_weather_engine": False,
    },
    "hard": {
        "cities": ["delhi", "lucknow", "patna"],
        "total_days": 30,
        "budget": 150.0,      # constrained
        "description": "Three cities, budget-constrained, festival events, fatigue management.",
        "enable_transport": True,
        "enable_health": True,
        "enable_sentiment": False,
        "enable_grap": True,
        "enable_weather_engine": True,
    },
    "expert": {
        "cities": ["delhi", "lucknow", "patna", "kanpur", "chandigarh"],
        "total_days": 45,
        "budget": 250.0,
        "description": (
            "5 cities with GRAP protocol, inter-city pollution transport, "
            "health impact optimization (DALYs), public sentiment management. "
            "45-day episode."
        ),
        "enable_transport": True,
        "enable_health": True,
        "enable_sentiment": True,
        "enable_grap": True,
        "enable_weather_engine": True,
    },
    "crisis": {
        "cities": [
            "delhi", "lucknow", "patna", "kanpur",
            "chandigarh", "mumbai", "kolkata", "bengaluru",
        ],
        "total_days": 15,
        "budget": 300.0,
        "description": (
            "8 cities during unprecedented pollution crisis. Hospital capacity "
            "constraints, rolling GRAP Stage IV, public panic. "
            "15-day emergency where every decision counts."
        ),
        "enable_transport": True,
        "enable_health": True,
        "enable_sentiment": True,
        "enable_grap": True,
        "enable_weather_engine": True,
    },
}

CITY_FILES = {
    "delhi": "delhi_winter.json",
    "lucknow": "lucknow_winter.json",
    "patna": "patna_winter.json",
    "kanpur": "kanpur_winter.json",
    "chandigarh": "chandigarh_winter.json",
    "mumbai": "mumbai_winter.json",
    "kolkata": "kolkata_winter.json",
    "bengaluru": "bengaluru_winter.json",
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
        self._seed = 42

        # ── Data (pre-load ALL cities so stateless HTTP works) ──────
        self._all_city_data: Dict[str, List[Dict[str, Any]]] = {}
        for city, filename in CITY_FILES.items():
            path = _DATA_DIR / filename
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    self._all_city_data[city] = json.load(f).get("days", [])
            else:
                self._all_city_data[city] = []

        # Load city profiles
        profile_path = _DATA_DIR / "city_profiles.json"
        self._city_profiles: Dict[str, Any] = {}
        if profile_path.exists():
            with open(profile_path, "r", encoding="utf-8") as f:
                self._city_profiles = json.load(f)

        # Active city data for current task
        self._city_data: Dict[str, List[Dict[str, Any]]] = {
            "delhi": self._all_city_data.get("delhi", []),
        }
        self._primary_city = "delhi"

        # ── Trajectory tracking (for graders) ────────────────────────────
        self._trajectories: Dict[str, List[Dict[str, Any]]] = {"delhi": []}
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

        # ── Advanced subsystems (initialized on reset for relevant tasks) ─
        self._grap_tracker: Optional[GRAPTracker] = None
        self._transport_model: Optional[TransportModel] = None
        self._weather_engine: Optional[WeatherEngine] = None
        self._health_tracker: Optional[HealthTracker] = None
        self._sentiment_tracker: Optional[SentimentTracker] = None
        self._grap_compliance_scores: List[float] = []

        # Per-city last day PM2.5 for sentiment change tracking
        self._last_pm25: Dict[str, float] = {}

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
        self._seed = seed or 42

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
            data = self._load_city(city)
            # For tasks with more days than data, cycle the data
            if data and self._total_days > len(data):
                repeats = (self._total_days // len(data)) + 1
                extended = []
                for i in range(repeats):
                    for j, d in enumerate(data):
                        dd = dict(d)
                        dd["day"] = len(extended) + 1
                        extended.append(dd)
                data = extended[:self._total_days]
            self._city_data[city] = data
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
        self._grap_compliance_scores = []
        self._last_pm25 = {c: 100 for c in config["cities"]}

        # ── Initialize advanced subsystems ───────────────────────────────
        cities = config["cities"]

        if config.get("enable_grap"):
            self._grap_tracker = GRAPTracker(cities)
        else:
            self._grap_tracker = None

        if config.get("enable_transport"):
            self._transport_model = TransportModel(cities)
        else:
            self._transport_model = None

        if config.get("enable_weather_engine"):
            self._weather_engine = WeatherEngine(cities, seed=self._seed)
        else:
            self._weather_engine = None

        if config.get("enable_health"):
            self._health_tracker = HealthTracker(cities)
        else:
            self._health_tracker = None

        if config.get("enable_sentiment"):
            self._sentiment_tracker = SentimentTracker(cities, seed=self._seed)
        else:
            self._sentiment_tracker = None

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

    def _resolve_actions(self, action: AQIAction, city: str) -> List[Dict[str, Any]]:
        """Resolve an AQIAction into a list of concrete action dicts.

        Handles:
            1. GRAP stage → expanded action bundle
            2. Multi-action list
            3. Legacy single action (backward compatible)
        """
        # 1. GRAP stage takes precedence
        if action.grap_stage is not None and self._grap_tracker is not None:
            day_data = self._city_data.get(city, [{}])
            day_idx = max(0, self._current_day - 1)
            pm25 = day_data[day_idx]["pm25"] if day_idx < len(day_data) else 100

            effective_stage, msg = self._grap_tracker.update(
                city, action.grap_stage, pm25, self._current_day
            )
            actions_list = expand_grap_stage(effective_stage)

            # Score GRAP compliance
            recommended = get_recommended_grap_stage(pm25)
            compliance_score, _ = compute_grap_compliance_score(
                recommended, effective_stage, pm25
            )
            self._grap_compliance_scores.append(compliance_score)

            return actions_list if actions_list else [{"action_type": "no_action", "level": 0}]

        # 2. Multi-action bundle
        if action.actions:
            return [
                {"action_type": a.action_type, "level": a.level}
                for a in action.actions
            ]

        # 3. Legacy single action
        return [{"action_type": action.action_type, "level": action.level}]

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

        # ── Advance weather engine ───────────────────────────────────────
        weather_info: Dict[str, Dict[str, Any]] = {}
        if self._weather_engine:
            base_winds = {}
            for c in cities:
                c_days = self._city_data.get(c, [])
                if day_idx < len(c_days):
                    base_winds[c] = float(c_days[day_idx].get("wind_kmh", 5))
            weather_info = self._weather_engine.advance_day(self._current_day, base_winds)

        # ── Resolve actions for target city ───────────────────────────────
        resolved_actions = self._resolve_actions(action, target_city)

        total_reward = 0.0
        total_health = 0.0
        total_econ = 0.0
        day_message_parts = []
        breakdown_agg = {}

        # Pre-compute equity penalty based on states up to yesterday
        equity_penalty = compute_city_equity_penalty(self._trajectories, self._current_day)

        # ── Compute transport contributions ──────────────────────────────
        transport_contributions: Dict[str, float] = {c: 0.0 for c in cities}
        if self._transport_model and self._last_pm25:
            wind_dir = self._weather_engine.get_prevailing_wind_dir() if self._weather_engine else 315.0
            wind_speed = 5.0
            if weather_info:
                speeds = [w.get("wind_speed", 5) for w in weather_info.values()]
                wind_speed = sum(speeds) / len(speeds) if speeds else 5.0

            stubble_flags = {}
            for c in cities:
                c_days = self._city_data.get(c, [])
                if day_idx < len(c_days):
                    stubble_flags[c] = c_days[day_idx].get("stubble_flag", 0)

            backgrounds = {c: self._city_profiles.get(c, {}).get("ncap_baseline_pm25", 60) * 0.2 for c in cities}
            transport_contributions = self._transport_model.step(
                self._last_pm25, backgrounds, wind_dir, wind_speed, stubble_flags
            )

        for city in cities:
            city_days = self._city_data[city]
            if day_idx >= len(city_days):
                continue

            day_data = city_days[day_idx]
            base_pm25 = day_data["pm25"]

            # Add transport contribution
            base_pm25 += transport_contributions.get(city, 0)

            # Apply weather multiplier
            if city in weather_info:
                base_pm25 *= weather_info[city].get("pm25_multiplier", 1.0)

            event = day_data.get("event") or ""
            wind_kmh = day_data.get("wind_kmh", 5)
            if city in weather_info:
                wind_kmh = weather_info[city].get("wind_speed", wind_kmh)
            wind_regime = classify_wind_regime(wind_kmh)
            if city in weather_info:
                wind_regime = weather_info[city].get("regime", wind_regime)
                # Map weather engine regimes to simulator regimes
                regime_map = {
                    "calm_inversion": "calm_inversion",
                    "light_haze": "light",
                    "normal": "normal",
                    "western_disturbance": "strong_ventilation",
                    "strong_ventilation": "strong_ventilation",
                }
                wind_regime = regime_map.get(wind_regime, wind_regime)

            festival_flag = "diwali" in event
            is_weekend = self._is_weekend(day_idx)
            forecast = compute_forecast(city_days, day_idx)

            if city == target_city:
                city_actions = resolved_actions
            else:
                city_actions = [{"action_type": "no_action", "level": 0}]
                # If GRAP tracker exists, maintain the stage for non-target cities
                if self._grap_tracker:
                    self._grap_tracker.update(city, None, base_pm25, self._current_day)

            fatigue_current = self._fatigue_levels[city]
            city_profile = self._city_profiles.get(city, {})

            # Check for forced reversal from sentiment
            reversal_forced = False
            if self._sentiment_tracker and self._sentiment_tracker.is_reversal_forced(city):
                reversal_forced = True
                city_actions = [{"action_type": "no_action", "level": 0}]

            # Apply actions (multi-action if >1)
            if len(city_actions) > 1:
                post_pm25, econ_cost, activities, red_details = apply_multi_action(
                    base_pm25, city_actions, city_profile,
                    fatigue=fatigue_current, is_weekend=is_weekend,
                    wind_regime=wind_regime, festival_flag=festival_flag,
                )
            else:
                act = city_actions[0] if city_actions else {"action_type": "no_action", "level": 0}
                post_pm25, econ_cost, activities, red_details = apply_intervention(
                    base_pm25, act["action_type"], act.get("level", 0), city_profile,
                    fatigue=fatigue_current, is_weekend=is_weekend,
                    wind_regime=wind_regime, festival_flag=festival_flag,
                )

            # Determine highest-level action for fatigue tracking
            max_act = max(city_actions, key=lambda a: a.get("level", 0)) if city_actions else {"action_type": "no_action", "level": 0}
            self._fatigue_levels[city] = update_fatigue(
                fatigue_current, max_act["action_type"], max_act.get("level", 0)
            )

            # Check forecast-proactive action
            acted_on_forc = False
            if city == target_city:
                acted_on_forc = self._acted_on_forecast(
                    max_act["action_type"], max_act.get("level", 0), base_pm25, forecast
                )

            # Compute reward
            reward, breakdown = compute_daily_reward(
                post_pm25, econ_cost, base_pm25,
                max_act["action_type"], max_act.get("level", 0), event,
                fatigue=fatigue_current,
                city_equity_penalty=equity_penalty if city == self._primary_city else 0.0,
                acted_on_forecast=acted_on_forc,
            )

            # ── Health tracking ──────────────────────────────────────────
            health_data = {}
            if self._health_tracker:
                health_data = self._health_tracker.record_day(city, post_pm25, is_weekend)

            # ── Sentiment tracking ───────────────────────────────────────
            sentiment_data = {}
            if self._sentiment_tracker:
                school_status = get_school_status(post_pm25) if self._health_tracker else "open"
                sentiment_data = self._sentiment_tracker.update(
                    city, self._current_day, post_pm25,
                    max_act["action_type"], max_act.get("level", 0),
                    econ_cost, self._last_pm25.get(city, base_pm25),
                    school_status,
                )
                # Add sentiment bonus/penalty to reward
                sent = sentiment_data.get("sentiment", 0.5)
                if sent < 0.3:
                    breakdown["sentiment_penalty"] = round(-(0.3 - sent) * 2.0, 2)
                    reward += breakdown["sentiment_penalty"]
                elif sent > 0.7:
                    breakdown["sentiment_bonus"] = round((sent - 0.7) * 0.5, 2)
                    reward += breakdown["sentiment_bonus"]

            if reversal_forced:
                breakdown["forced_reversal_penalty"] = -3.0
                reward -= 3.0

            if self._budget_total > 0:
                self._budget_spent += econ_cost

            # Track
            self._activity_levels[city] = activities
            self._pm25_histories[city].append(post_pm25)
            self._last_pm25[city] = post_pm25
            self._trajectories[city].append({
                "day": self._current_day,
                "pm25_base": base_pm25,
                "pm25_post": post_pm25,
                "action_type": max_act["action_type"],
                "level": max_act.get("level", 0),
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
                f"{city.title()}: {day_data['pm25']}→{post_pm25} ({aqi_cat})"
            )

            # Add health info to message on advanced tasks
            if health_data and health_data.get("admissions"):
                adm = health_data["admissions"]["total"]
                if adm > 50:
                    day_message_parts.append(
                        f"  🏥 {city.title()}: ~{adm:.0f} hospital admissions"
                    )

            # Add sentiment warnings
            if sentiment_data.get("forced_reversal"):
                day_message_parts.append(sentiment_data["reversal_message"])
            for headline in sentiment_data.get("new_media_events", []):
                day_message_parts.append(f"  {headline}")

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

        # When episode is done, use the grader score as the reward
        if self._done and self._grader_scores:
            final_reward = list(self._grader_scores.values())[0]
        else:
            final_reward = round(total_reward, 2)

        return self._make_observation(
            reward=final_reward,
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
                self._trajectories, self._total_days,
                self._budget_total, self._budget_spent,
            )
            self._grader_scores = {"hard": score}
        elif self._task_id == "expert":
            health_summary = self._health_tracker.get_summary() if self._health_tracker else {}
            sentiment_history = {}
            if self._sentiment_tracker:
                for city in TASK_CONFIGS[self._task_id]["cities"]:
                    sentiment_history[city] = self._sentiment_tracker.get_history(city)
            score, details = grade_expert(
                self._trajectories, self._total_days,
                self._budget_total, self._budget_spent,
                health_summary, sentiment_history,
                self._grap_compliance_scores,
            )
            self._grader_scores = {"expert": score}
        elif self._task_id == "crisis":
            health_summary = self._health_tracker.get_summary() if self._health_tracker else {}
            sentiment_history = {}
            if self._sentiment_tracker:
                for city in TASK_CONFIGS[self._task_id]["cities"]:
                    sentiment_history[city] = self._sentiment_tracker.get_history(city)
            score, details = grade_crisis(
                self._trajectories, self._total_days,
                self._budget_total, self._budget_spent,
                health_summary, sentiment_history,
                self._grap_compliance_scores,
            )
            self._grader_scores = {"crisis": score}

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

        # Use weather engine data if available
        wind_dir = float(today.get("wind_dir", 315.0))
        weather_desc = ""
        regime = classify_wind_regime(wind_kmh)
        if self._weather_engine:
            regime = self._weather_engine.get_regime(self._primary_city)
            wind_dir = self._weather_engine.get_wind_direction(self._primary_city)
            from env.weather import WEATHER_REGIMES
            weather_desc = WEATHER_REGIMES.get(regime, {}).get("description", "")
            # Map to simulator regime for backward compat
            regime_map = {
                "calm_inversion": "calm_inversion",
                "light_haze": "light",
                "normal": "normal",
                "western_disturbance": "strong_ventilation",
                "strong_ventilation": "strong_ventilation",
            }
            regime = regime_map.get(regime, regime)

        is_weekend = self._is_weekend(day_idx)
        festival_flag = "diwali" in str(today.get("event", ""))

        # Forecast details (noisy if weather engine enabled)
        forecast_details = []
        if self._weather_engine:
            forecast_details = compute_noisy_forecast(
                primary_days, day_idx, self._seed, self._primary_city
            )
            forecast_2d = [f.get("risk_bucket", "unknown") for f in forecast_details]

        cities_obs: Dict[str, Any] = {}
        if len(cities) > 1:
            for city in cities:
                c_days = self._city_data.get(city, [])
                c_today = c_days[day_idx] if day_idx < len(c_days) else {}
                c_hist = self._pm25_histories.get(city, [])[-7:]
                c_acts = self._activity_levels.get(city, {})
                c_forecast = compute_forecast(c_days, day_idx)
                c_wind = float(c_today.get("wind_kmh", 5.0))
                c_wind_dir = float(c_today.get("wind_dir", 315.0))

                c_regime = classify_wind_regime(c_wind)
                if self._weather_engine:
                    c_regime = self._weather_engine.get_regime(city)
                    c_wind_dir = self._weather_engine.get_wind_direction(city)

                city_obs = {
                    "pm25_today": c_today.get("pm25", 0),
                    "pm25_post": c_hist[-1] if c_hist else c_today.get("pm25", 0),
                    "pm25_history": c_hist,
                    "temperature_c": c_today.get("temp_c", 0),
                    "wind_speed_kmh": c_wind,
                    "wind_direction_deg": c_wind_dir,
                    "humidity_pct": c_today.get("humidity_pct", 0),
                    "stubble_flag": c_today.get("stubble_flag", 0),
                    "wind_regime": c_regime,
                    "event": c_today.get("event") or "",
                    "forecast_risk_next_2d": c_forecast,
                    "policy_fatigue_index": self._fatigue_levels.get(city, 0.0),
                    "traffic_level": c_acts.get("traffic", 1.0),
                    "construction_level": c_acts.get("construction", 1.0),
                    "industry_level": c_acts.get("industry", 1.0),
                }

                # Add GRAP info
                if self._grap_tracker:
                    city_obs["grap_stage"] = self._grap_tracker.get_stage(city)
                    city_obs["grap_recommended"] = get_recommended_grap_stage(
                        c_today.get("pm25", 0)
                    )

                # Add health info
                if self._health_tracker:
                    city_obs["health_cumulative"] = self._health_tracker.get_cumulative(city)

                # Add sentiment info
                if self._sentiment_tracker:
                    city_obs["public_sentiment"] = self._sentiment_tracker.get_sentiment(city)
                    city_obs["media_headlines"] = [
                        e["headline"] for e in self._sentiment_tracker.get_active_events(city)
                    ]

                cities_obs[city] = city_obs

        acts = self._activity_levels.get(self._primary_city, {})
        budget_remaining = self._budget_total - self._budget_spent if self._budget_total > 0 else -1.0

        # When episode is done, ensure the reward field carries the grader score
        final_reward = reward
        if self._done and self._grader_scores:
            grader_score = list(self._grader_scores.values())[0]
            final_reward = max(0.001, min(float(grader_score), 0.999))

        # ── Build health observation fields ──────────────────────────────
        est_admissions = 0.0
        mortality_risk = 0.0
        school_status = "open"
        daly_today = 0.0
        hospital_stress = "normal"
        healthcare_cost = 0.0
        if self._health_tracker:
            cum = self._health_tracker.get_cumulative(self._primary_city)
            est_admissions = cum.get("total_admissions", 0) / max(1, self._current_day)
            from env.health import (
                compute_excess_mortality_risk, compute_daly_burden,
                get_school_status as _get_school, compute_hospital_capacity_stress,
                compute_hospital_admissions,
            )
            mortality_risk = compute_excess_mortality_risk(post_pm25, self._primary_city)
            daly_data = compute_daly_burden(post_pm25, self._primary_city)
            daly_today = daly_data["total_dalys"]
            school_status = _get_school(post_pm25)
            adm = compute_hospital_admissions(post_pm25, self._primary_city)
            est_admissions = adm["total"]
            cap = compute_hospital_capacity_stress(adm["total"], self._primary_city)
            hospital_stress = cap["stress_level"]
            healthcare_cost = cum.get("total_healthcare_cost", 0) / max(1, self._current_day)

        # ── Build sentiment fields ───────────────────────────────────────
        pub_sentiment = 0.55
        media_headlines = []
        forced_reversal = False
        if self._sentiment_tracker:
            pub_sentiment = self._sentiment_tracker.get_sentiment(self._primary_city)
            media_headlines = [
                e["headline"] for e in self._sentiment_tracker.get_active_events(self._primary_city)
            ]
            forced_reversal = self._sentiment_tracker.is_reversal_forced(self._primary_city)

        # ── GRAP fields ──────────────────────────────────────────────────
        grap_stage = 0
        grap_recommended = 0
        grap_days = 0
        if self._grap_tracker:
            grap_stage = self._grap_tracker.get_stage(self._primary_city)
            grap_recommended = get_recommended_grap_stage(base_pm25)
            grap_days = self._grap_tracker.get_days_at_stage(self._primary_city)

        # ── Transport fields ─────────────────────────────────────────────
        transport_received = 0.0
        if self._transport_model and self._primary_city in (self._last_pm25 or {}):
            # Estimate from last step's contributions
            pass  # Already accounted for in base_pm25

        return AQIObservation(
            day=self._current_day,
            date=today.get("date", ""),
            city=self._primary_city.title(),
            pm25_today=float(base_pm25),
            pm25_post_intervention=float(post_pm25),
            pm25_history=history,
            temperature_c=float(today.get("temp_c", 0)),
            wind_speed_kmh=wind_kmh,
            wind_direction_deg=round(wind_dir, 1),
            humidity_pct=float(today.get("humidity_pct", 0)),
            stubble_burning_flag=int(today.get("stubble_flag", 0)),
            wind_regime=regime,
            weather_description=weather_desc,
            event=today.get("event") or "",
            festival_flag=festival_flag,
            is_weekend=is_weekend,
            forecast_risk_next_2d=forecast_2d,
            forecast_details=forecast_details,
            traffic_level=acts.get("traffic", 1.0),
            construction_level=acts.get("construction", 1.0),
            industry_level=acts.get("industry", 1.0),
            policy_fatigue_index=round(self._fatigue_levels.get(self._primary_city, 0.0), 3),
            grap_stage=grap_stage,
            grap_recommended=grap_recommended,
            grap_days_at_stage=grap_days,
            estimated_hospital_admissions=round(est_admissions, 1),
            excess_mortality_risk_per_100k=mortality_risk,
            school_closure_status=school_status,
            daly_burden_today=round(daly_today, 4),
            hospital_capacity_stress=hospital_stress,
            healthcare_cost_crores=round(healthcare_cost, 2),
            public_sentiment=pub_sentiment,
            media_headlines=media_headlines,
            forced_reversal=forced_reversal,
            reward_breakdown=self._last_reward_breakdown,
            total_days=self._total_days,
            budget_remaining=round(budget_remaining, 1),
            cumulative_health_cost=round(self._cumulative_health_cost, 2),
            cumulative_economic_cost=round(self._cumulative_economic_cost, 2),
            task_id=self._task_id,
            cities_data=cities_obs,
            message=message,
            done=self._done,
            reward=final_reward,
            metadata={"grader_scores": dict(self._grader_scores)},
            grader_scores=dict(self._grader_scores),
        )
