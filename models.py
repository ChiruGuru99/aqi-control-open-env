# Copyright (c) 2025. AQI Control Environment for OpenEnv.
# All rights reserved.

"""
Data models for the AQI Control Environment.

Defines Action and Observation types for the city-level PM2.5 policy
planning benchmark under NCAP constraints.  Uses Pydantic BaseModel
via OpenEnv base classes.
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ── Action ──────────────────────────────────────────────────────────────────


class AQIAction(Action):
    """Daily policy action for the AQI control environment.

    The agent selects an action_type and (for interventions) a level 1-3.

    action_type values:
        no_action          – do nothing
        restrict_traffic   – odd-even, truck bans (level 1-3)
        curtail_construction – partial to full construction ban (level 1-3)
        limit_industry     – stack emission restrictions (level 1-3)
        issue_public_advisory – masks/stay-indoors advisory (no level)
    """

    action_type: str = Field(
        ...,
        description=(
            "One of: no_action, restrict_traffic, curtail_construction, "
            "limit_industry, issue_public_advisory"
        ),
    )
    level: int = Field(
        default=0,
        ge=0,
        le=3,
        description="Intervention intensity 0-3 (0 = none, 3 = strongest)",
    )
    city: str = Field(
        default="",
        description="Target city (for multi-city tasks). Empty = current/only city.",
    )


# ── Reward Breakdown ────────────────────────────────────────────────────────


class RewardBreakdown(Dict[str, float]):
    """Transparent reward components returned with each step.

    Keys: health_cost, economic_cost, policy_bonus, equity_penalty,
          fatigue_penalty, forecast_bonus.
    """
    pass


# ── Observation ─────────────────────────────────────────────────────────────


class AQIObservation(Observation):
    """Daily observation from the AQI control environment.

    Provides the agent with historical PM2.5, weather context, economic
    activity levels, forecasts, policy fatigue state, and episode progress.
    Inherits `done`, `reward`, `metadata` from the Observation base class.
    """

    # ── Current day context ──────────────────────────────────────────────
    day: int = Field(default=0, description="Current day number in the episode")
    date: str = Field(default="", description="Calendar date (YYYY-MM-DD)")
    city: str = Field(default="", description="Current primary city")

    # ── AQI readings ─────────────────────────────────────────────────────
    pm25_today: float = Field(
        default=0.0,
        description="Today's base PM2.5 before interventions (ug/m3, from CPCB data)",
    )
    pm25_post_intervention: float = Field(
        default=0.0,
        description="PM2.5 after applying today's intervention effects (ug/m3)",
    )
    pm25_history: List[float] = Field(
        default_factory=list,
        description="PM2.5 readings for the last 7 days (post-intervention)",
    )

    # ── Weather & external factors ───────────────────────────────────────
    temperature_c: float = Field(default=0.0, description="Temperature in Celsius")
    wind_speed_kmh: float = Field(default=0.0, description="Wind speed in km/h")
    humidity_pct: float = Field(default=0.0, description="Relative humidity %")
    stubble_burning_flag: int = Field(
        default=0,
        description="1 if stubble burning detected in region, 0 otherwise",
    )
    wind_regime: str = Field(
        default="normal",
        description="Wind regime: calm_inversion, light, normal, strong_ventilation",
    )
    event: str = Field(
        default="",
        description="Special event: pre_diwali, diwali_week, diwali, post_diwali, or empty",
    )
    festival_flag: bool = Field(
        default=False,
        description="True if today is a festival or high-emission event day",
    )
    is_weekend: bool = Field(
        default=False,
        description="True if today is Saturday or Sunday (natural traffic dip)",
    )

    # ── Forecast panel (next 2 days) ─────────────────────────────────────
    forecast_risk_next_2d: List[str] = Field(
        default_factory=list,
        description=(
            "Risk buckets for next 2 days: 'low' (<60), 'moderate' (60-120), "
            "'high' (120-250), 'severe' (>250). Empty if unavailable."
        ),
    )

    # ── Economic activity (0.0-1.0 scales) ───────────────────────────────
    traffic_level: float = Field(
        default=1.0, description="Traffic activity level (1.0 = normal, 0.0 = fully restricted)"
    )
    construction_level: float = Field(
        default=1.0, description="Construction activity level"
    )
    industry_level: float = Field(
        default=1.0, description="Industrial activity level"
    )

    # ── Policy fatigue state ─────────────────────────────────────────────
    policy_fatigue_index: float = Field(
        default=0.0,
        description=(
            "Fatigue index 0.0-1.0. Increases with consecutive strong interventions. "
            "High fatigue reduces intervention effectiveness and increases economic cost."
        ),
    )

    # ── Reward interpretability ──────────────────────────────────────────
    reward_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Transparent breakdown: {health_cost, economic_cost, policy_bonus, "
            "equity_penalty, fatigue_penalty, forecast_bonus}"
        ),
    )

    # ── Episode progress ─────────────────────────────────────────────────
    total_days: int = Field(default=30, description="Total days in this episode")
    budget_remaining: float = Field(
        default=-1.0,
        description="Remaining economic budget (-1 = unlimited, used in hard task)",
    )
    cumulative_health_cost: float = Field(
        default=0.0, description="Accumulated health cost so far"
    )
    cumulative_economic_cost: float = Field(
        default=0.0, description="Accumulated economic cost so far"
    )
    task_id: str = Field(
        default="easy",
        description="Current task: easy, medium, or hard",
    )
    cities_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Multi-city observations for medium/hard tasks {city: {...}}",
    )
    message: str = Field(default="", description="Human-readable status message")
