# Copyright (c) 2025. AQI Control Environment for OpenEnv.
# All rights reserved.

"""
Political economy and public sentiment model.

Real-world air quality policy doesn't exist in a vacuum — it faces
political constraints, public backlash, media pressure, and industry
lobbying. This module models:

    - Public sentiment (0=outraged, 1=supportive)
    - Media pressure events (viral stories about pollution deaths)
    - Industry lobbying resistance
    - Forced policy reversals (if sentiment drops too low)
    - Election proximity effects

The agent must balance health outcomes against political viability.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

# ── Sentiment parameters ────────────────────────────────────────────────

INITIAL_SENTIMENT = 0.55  # Slightly positive (people expect action in winter)
SENTIMENT_FLOOR = 0.0
SENTIMENT_CEILING = 1.0

# ── Sentiment change drivers ────────────────────────────────────────────

SENTIMENT_DRIVERS = {
    # Positive drivers (things that increase public support)
    "advisory_issued": 0.02,          # Transparency bonus
    "proactive_action": 0.03,         # Acting before crisis
    "visible_improvement": 0.04,      # PM2.5 dropped significantly
    "school_reopened": 0.03,          # Schools back open
    "media_positive": 0.05,           # Positive media coverage

    # Negative drivers (things that decrease support)
    "excessive_restriction_clean_day": -0.06,  # Lockdown on a clean day
    "prolonged_lockdown": -0.04,      # Per day of sustained L3 action
    "economic_pain": -0.03,           # Per unit of economic cost
    "inaction_during_crisis": -0.08,  # No action when PM2.5 > 300
    "child_hospitalization_news": -0.10,  # Media event
    "death_reported": -0.12,          # Pollution death in news
    "school_closed_extended": -0.03,  # Schools closed > 3 consecutive days
    "industry_layoffs": -0.05,        # Industrial closures causing job loss
}

# ── Forced policy reversal threshold ─────────────────────────────────────
REVERSAL_THRESHOLD = 0.18  # Below this, government overrides agent
REVERSAL_COOLDOWN_DAYS = 3  # Can't be forced again within this window

# ── Media event probabilities ────────────────────────────────────────────
# Media events are stochastic but deterministic given seed


def _det_hash(seed: int, day: int, salt: str) -> float:
    data = f"{seed}:{day}:{salt}".encode()
    h = hashlib.sha256(data).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


MEDIA_EVENTS = {
    "child_asthma_viral": {
        "base_probability": 0.08,  # per day when PM2.5 > 200
        "pm25_threshold": 200,
        "sentiment_impact": -0.10,
        "headline": "📰 Child rushed to hospital — 'Air is poison' goes viral",
        "duration_days": 2,
    },
    "pollution_death": {
        "base_probability": 0.04,
        "pm25_threshold": 300,
        "sentiment_impact": -0.15,
        "headline": "📰 Elderly man dies from pollution-related cardiac arrest",
        "duration_days": 3,
    },
    "clean_day_celebration": {
        "base_probability": 0.10,
        "pm25_threshold": -1,  # Triggers when PM2.5 < 60
        "pm25_below": 60,
        "sentiment_impact": 0.08,
        "headline": "📰 Citizens celebrate 'blue sky day' — policies praised",
        "duration_days": 1,
    },
    "industry_protest": {
        "base_probability": 0.06,
        "pm25_threshold": -1,  # Triggers on strong industrial restrictions
        "requires_action": "limit_industry",
        "min_level": 2,
        "sentiment_impact": -0.07,
        "headline": "📰 Factory workers protest shutdown, demand compensation",
        "duration_days": 2,
    },
    "doctor_warning": {
        "base_probability": 0.12,
        "pm25_threshold": 150,
        "sentiment_impact": -0.05,
        "headline": "📰 Doctors' association warns of 'public health emergency'",
        "duration_days": 1,
    },
}


def compute_sentiment_change(
    pm25: float,
    action_type: str,
    level: int,
    economic_cost: float,
    pm25_change: float,
    school_status: str,
    consecutive_lockdown_days: int,
    media_events_active: List[str],
) -> Tuple[float, List[str]]:
    """Compute the change in public sentiment for one day.

    Args:
        pm25: Today's PM2.5 (post-intervention).
        action_type: The action taken.
        level: Action level.
        economic_cost: Today's economic cost.
        pm25_change: PM2.5 change from yesterday (negative = improvement).
        school_status: Current school closure status.
        consecutive_lockdown_days: Days of sustained strong restrictions.
        media_events_active: Currently active media events.

    Returns:
        (sentiment_delta, [explanation_strings])
    """
    delta = 0.0
    explanations = []

    # ── Positive drivers ─────────────────────────────────────────────────

    if action_type == "issue_public_advisory":
        delta += SENTIMENT_DRIVERS["advisory_issued"]
        explanations.append("Advisory transparency (+)")

    if pm25_change < -30:
        delta += SENTIMENT_DRIVERS["visible_improvement"]
        explanations.append("Visible air quality improvement (+)")

    if pm25 < 60 and level == 0:
        # Clean day, no restrictions — good
        delta += 0.02
        explanations.append("Clean air with no restrictions (+)")

    # ── Negative drivers ─────────────────────────────────────────────────

    if pm25 < 60 and level >= 2:
        delta += SENTIMENT_DRIVERS["excessive_restriction_clean_day"]
        explanations.append("Excessive restrictions on clean day (-)")

    if pm25 > 300 and action_type == "no_action":
        delta += SENTIMENT_DRIVERS["inaction_during_crisis"]
        explanations.append("INACTION during air quality crisis (-)")

    if consecutive_lockdown_days > 3 and level >= 2:
        delta += SENTIMENT_DRIVERS["prolonged_lockdown"]
        explanations.append(f"Prolonged lockdown (day {consecutive_lockdown_days}) (-)")

    if economic_cost > 8:
        delta += SENTIMENT_DRIVERS["economic_pain"] * (economic_cost / 10)
        explanations.append("Significant economic disruption (-)")

    if school_status in ("primary_closed", "all_closed"):
        parents_delta = SENTIMENT_DRIVERS["school_closed_extended"]
        # Parents get more frustrated after day 3
        if consecutive_lockdown_days > 3:
            parents_delta *= 1.5
        delta += parents_delta
        explanations.append("School closures affecting families (-)")

    # ── Media events ─────────────────────────────────────────────────────

    for event_name in media_events_active:
        event = MEDIA_EVENTS.get(event_name)
        if event:
            delta += event["sentiment_impact"]
            explanations.append(event["headline"])

    return round(delta, 4), explanations


def check_media_events(
    pm25: float,
    action_type: str,
    level: int,
    seed: int,
    day: int,
    city: str,
) -> List[Dict[str, Any]]:
    """Check whether any media events trigger today.

    Returns list of triggered event dicts.
    """
    triggered = []

    for event_name, event in MEDIA_EVENTS.items():
        r = _det_hash(seed, day, f"{city}:{event_name}")

        # Check trigger conditions
        if event.get("pm25_below"):
            # Positive event: triggers when PM2.5 is LOW
            if pm25 >= event["pm25_below"]:
                continue
        elif event.get("pm25_threshold", 0) > 0:
            if pm25 < event["pm25_threshold"]:
                continue

        if event.get("requires_action"):
            if action_type != event["requires_action"]:
                continue
            if level < event.get("min_level", 0):
                continue

        # Probabilistic trigger
        if r < event["base_probability"]:
            triggered.append({
                "name": event_name,
                "headline": event["headline"],
                "sentiment_impact": event["sentiment_impact"],
                "duration_days": event["duration_days"],
                "day_triggered": day,
            })

    return triggered


class SentimentTracker:
    """Track public sentiment across an episode."""

    def __init__(self, cities: List[str], seed: int = 42):
        self._seed = seed
        self._sentiment: Dict[str, float] = {c: INITIAL_SENTIMENT for c in cities}
        self._consecutive_lockdown: Dict[str, int] = {c: 0 for c in cities}
        self._active_media_events: Dict[str, List[Dict[str, Any]]] = {c: [] for c in cities}
        self._reversal_cooldown: Dict[str, int] = {c: 0 for c in cities}
        self._history: Dict[str, List[Dict[str, Any]]] = {c: [] for c in cities}

    def get_sentiment(self, city: str) -> float:
        return round(self._sentiment.get(city, INITIAL_SENTIMENT), 3)

    def get_active_events(self, city: str) -> List[Dict[str, Any]]:
        return list(self._active_media_events.get(city, []))

    def is_reversal_forced(self, city: str) -> bool:
        """Check if public sentiment forces a policy reversal."""
        if self._reversal_cooldown.get(city, 0) > 0:
            return False
        return self._sentiment.get(city, INITIAL_SENTIMENT) < REVERSAL_THRESHOLD

    def update(
        self,
        city: str,
        day: int,
        pm25: float,
        action_type: str,
        level: int,
        economic_cost: float,
        pm25_yesterday: float,
        school_status: str,
    ) -> Dict[str, Any]:
        """Update sentiment for one city-day.

        Returns a summary dict with sentiment, events, and any forced reversals.
        """
        pm25_change = pm25 - pm25_yesterday

        # Track consecutive lockdown days
        if level >= 2 and action_type not in ("no_action", "issue_public_advisory"):
            self._consecutive_lockdown[city] = self._consecutive_lockdown.get(city, 0) + 1
        else:
            self._consecutive_lockdown[city] = 0

        # Expire old media events
        active = self._active_media_events.get(city, [])
        active = [
            e for e in active
            if day - e["day_triggered"] < e["duration_days"]
        ]

        # Check for new media events
        new_events = check_media_events(
            pm25, action_type, level, self._seed, day, city
        )
        active.extend(new_events)
        self._active_media_events[city] = active

        active_names = [e["name"] for e in active]

        # Compute sentiment change
        delta, explanations = compute_sentiment_change(
            pm25, action_type, level, economic_cost, pm25_change,
            school_status, self._consecutive_lockdown.get(city, 0),
            active_names,
        )

        # Apply change
        old_sentiment = self._sentiment.get(city, INITIAL_SENTIMENT)
        new_sentiment = max(SENTIMENT_FLOOR, min(SENTIMENT_CEILING, old_sentiment + delta))
        self._sentiment[city] = new_sentiment

        # Check for forced reversal
        forced_reversal = False
        reversal_message = ""
        if self._reversal_cooldown.get(city, 0) > 0:
            self._reversal_cooldown[city] -= 1
        elif new_sentiment < REVERSAL_THRESHOLD:
            forced_reversal = True
            reversal_message = (
                f"⚠️ PUBLIC BACKLASH in {city.title()}: Sentiment at "
                f"{new_sentiment:.2f} — Government forces policy reversal!"
            )
            self._reversal_cooldown[city] = REVERSAL_COOLDOWN_DAYS

        result = {
            "sentiment": round(new_sentiment, 3),
            "sentiment_change": round(delta, 4),
            "explanations": explanations,
            "media_events": [e["headline"] for e in active],
            "new_media_events": [e["headline"] for e in new_events],
            "consecutive_lockdown_days": self._consecutive_lockdown.get(city, 0),
            "forced_reversal": forced_reversal,
            "reversal_message": reversal_message,
        }

        self._history.setdefault(city, []).append({
            "day": day, "sentiment": new_sentiment, "delta": delta,
        })

        return result

    def get_history(self, city: str) -> List[Dict[str, Any]]:
        return self._history.get(city, [])
