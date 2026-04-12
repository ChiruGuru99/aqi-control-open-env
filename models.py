# Copyright (c) 2025. AQI Control Environment for OpenEnv.
# All rights reserved.

"""
Data models for the AQI Control Environment.

Defines Action and Observation types for the city-level PM2.5 policy
planning benchmark under NCAP constraints.  Uses Pydantic BaseModel
via OpenEnv base classes.

Enhanced with:
    - Multi-action support (multiple simultaneous interventions)
    - GRAP stage selection
    - Public health impact fields
    - Public sentiment tracking
    - Inter-city pollution transport visibility
    - Weather regime details
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ── Single Action (used in multi-action bundles) ────────────────────────


class SingleAction(Action):
    """A single intervention action within a multi-action bundle."""
    action_type: str = Field(
        ...,
        description=(
            "One of: no_action, restrict_traffic, curtail_construction, "
            "limit_industry, issue_public_advisory, close_schools, "
            "mandate_wfh, deploy_smog_response, emergency_transport_ban"
        ),
    )
    level: int = Field(default=0, ge=0, le=3)
    city: str = Field(default="", description="Target city (empty = primary city)")


# ── Action ──────────────────────────────────────────────────────────────────


class AQIAction(Action):
    """Daily policy action for the AQI control environment.

    The agent can either:
        1. Set a GRAP stage (which auto-activates multiple interventions)
        2. Specify individual actions in a multi-action bundle
        3. Use legacy single-action format (backward compatible)

    action_type values (original + new):
        no_action              – do nothing
        restrict_traffic       – odd-even, truck bans (level 1-3)
        curtail_construction   – partial to full construction ban (level 1-3)
        limit_industry         – stack emission restrictions (level 1-3)
        issue_public_advisory  – masks/stay-indoors advisory (no level)
        close_schools          – school closures (level 1=advisory, 2=primary, 3=all) [NEW]
        mandate_wfh            – work from home mandates (level 1-3) [NEW]
        deploy_smog_response   – water sprinklers + anti-smog guns (level 1-3) [NEW]
        emergency_transport_ban – ban specific vehicle categories (level 1-3) [NEW]
    """

    # ── GRAP stage (takes precedence if set) ─────────────────────────────
    grap_stage: Optional[int] = Field(
        default=None,
        ge=0,
        le=4,
        description=(
            "GRAP (Graded Response Action Plan) stage 0-4. "
            "If set, this overrides individual actions with GRAP protocol bundle."
        ),
    )

    # ── Multi-action bundle ──────────────────────────────────────────────
    actions: List[SingleAction] = Field(
        default_factory=list,
        description="Multiple simultaneous actions for one timestep",
    )

    # ── Legacy single-action (backward compatible) ───────────────────────
    action_type: str = Field(
        default="no_action",
        description=(
            "One of: no_action, restrict_traffic, curtail_construction, "
            "limit_industry, issue_public_advisory, close_schools, "
            "mandate_wfh, deploy_smog_response, emergency_transport_ban"
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
          fatigue_penalty, forecast_bonus, grap_compliance, sentiment_bonus.
    """
    pass


# ── Observation ─────────────────────────────────────────────────────────────


class AQIObservation(Observation):
    """Daily observation from the AQI control environment.

    Provides the agent with historical PM2.5, weather context, economic
    activity levels, forecasts, policy fatigue state, health impacts,
    public sentiment, and episode progress.
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
    wind_direction_deg: float = Field(
        default=315.0,
        description="Wind direction in degrees from north (0=N, 90=E, 180=S, 270=W)",
    )
    humidity_pct: float = Field(default=0.0, description="Relative humidity %")
    stubble_burning_flag: int = Field(
        default=0,
        description="1 if stubble burning detected in region, 0 otherwise",
    )
    wind_regime: str = Field(
        default="normal",
        description="Wind regime: calm_inversion, light_haze, normal, western_disturbance, strong_ventilation",
    )
    weather_description: str = Field(
        default="",
        description="Human-readable weather regime description",
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
    forecast_details: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detailed forecast with PM2.5 estimates and confidence levels",
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

    # ── GRAP status ──────────────────────────────────────────────────────
    grap_stage: int = Field(
        default=0,
        description="Current GRAP stage (0-4). 0=normal, 4=severe+ emergency.",
    )
    grap_recommended: int = Field(
        default=0,
        description="GRAP stage recommended by protocol for current PM2.5 level.",
    )
    grap_days_at_stage: int = Field(
        default=0,
        description="Number of days at the current GRAP stage.",
    )

    # ── Public Health Dashboard ──────────────────────────────────────────
    estimated_hospital_admissions: float = Field(
        default=0.0,
        description="Estimated daily hospital admissions from PM2.5 exposure",
    )
    excess_mortality_risk_per_100k: float = Field(
        default=0.0,
        description="Excess mortality risk per 100k population per day",
    )
    school_closure_status: str = Field(
        default="open",
        description="School status: open, outdoor_restricted, primary_closed, all_closed",
    )
    daly_burden_today: float = Field(
        default=0.0,
        description="Disability-Adjusted Life Years lost today from PM2.5 exposure",
    )
    cumulative_dalys_averted: float = Field(
        default=0.0,
        description="Cumulative DALYs averted vs no-action baseline",
    )
    hospital_capacity_stress: str = Field(
        default="normal",
        description="Hospital capacity: normal, elevated, strained, crisis",
    )
    healthcare_cost_crores: float = Field(
        default=0.0,
        description="Estimated daily healthcare cost in ₹ crores",
    )

    # ── Public Sentiment ─────────────────────────────────────────────────
    public_sentiment: float = Field(
        default=0.55,
        description="Public sentiment score (0=outraged, 0.5=neutral, 1=supportive)",
    )
    media_headlines: List[str] = Field(
        default_factory=list,
        description="Active media headlines affecting public sentiment",
    )
    forced_reversal: bool = Field(
        default=False,
        description="True if public backlash forced a policy reversal this turn",
    )

    # ── Inter-city Transport ─────────────────────────────────────────────
    transport_pm25_received: float = Field(
        default=0.0,
        description="PM2.5 received from other cities via wind transport (ug/m3)",
    )
    regional_haze_contribution: float = Field(
        default=0.0,
        description="PM2.5 from regional stubble burning haze (ug/m3)",
    )

    # ── Reward interpretability ──────────────────────────────────────────
    reward_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Transparent breakdown: {health_cost, economic_cost, policy_bonus, "
            "equity_penalty, fatigue_penalty, forecast_bonus, grap_compliance, "
            "sentiment_bonus}"
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
        description="Current task: easy, medium, hard, expert, or crisis",
    )
    cities_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Multi-city observations for medium/hard/expert/crisis tasks {city: {...}}",
    )
    message: str = Field(default="", description="Human-readable status message")
    grader_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Terminal grader scores mapped by task_id.",
    )
