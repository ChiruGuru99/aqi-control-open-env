# Copyright (c) 2025. AQI Control Environment for OpenEnv.
# All rights reserved.

"""
Stochastic weather model with forecast uncertainty.

Real meteorology is uncertain — forecasts 2 days out can be significantly
wrong. This module adds:
    - Wind direction (not just speed) — critical for pollution transport
    - Forecast noise: imperfect 2-day lookahead
    - Weather regime transitions (western disturbances, inversions)
    - Seasonal temperature/humidity patterns

Winter over the Indo-Gangetic Plain is characterized by:
    - Prevailing NW winds carrying pollution SE
    - Frequent calm inversions trapping pollution
    - Occasional western disturbances bringing temporary relief
    - Fog/smog episodes with high humidity and near-zero wind
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Tuple

# ── Weather regime definitions ───────────────────────────────────────────

WEATHER_REGIMES = {
    "calm_inversion": {
        "wind_speed_range": (0, 3),
        "wind_dir_range": (280, 350),  # NW ± variability
        "pm25_multiplier": 1.25,
        "description": "Atmospheric inversion traps pollution near surface",
        "transition_probs": {
            "calm_inversion": 0.65,
            "light_haze": 0.20,
            "normal": 0.10,
            "western_disturbance": 0.05,
        },
    },
    "light_haze": {
        "wind_speed_range": (2, 6),
        "wind_dir_range": (290, 340),
        "pm25_multiplier": 1.10,
        "description": "Light winds with hazy conditions",
        "transition_probs": {
            "calm_inversion": 0.30,
            "light_haze": 0.40,
            "normal": 0.25,
            "western_disturbance": 0.05,
        },
    },
    "normal": {
        "wind_speed_range": (4, 10),
        "wind_dir_range": (270, 360),
        "pm25_multiplier": 1.00,
        "description": "Normal winter conditions with moderate winds",
        "transition_probs": {
            "calm_inversion": 0.15,
            "light_haze": 0.25,
            "normal": 0.40,
            "western_disturbance": 0.15,
            "strong_ventilation": 0.05,
        },
    },
    "western_disturbance": {
        "wind_speed_range": (8, 18),
        "wind_dir_range": (200, 280),  # SW — brings moisture and clears air
        "pm25_multiplier": 0.70,
        "description": "Western disturbance brings rain/wind, clears pollution",
        "transition_probs": {
            "normal": 0.50,
            "strong_ventilation": 0.30,
            "light_haze": 0.15,
            "western_disturbance": 0.05,
        },
    },
    "strong_ventilation": {
        "wind_speed_range": (12, 22),
        "wind_dir_range": (250, 320),
        "pm25_multiplier": 0.75,
        "description": "Strong winds disperse pollution effectively",
        "transition_probs": {
            "normal": 0.45,
            "light_haze": 0.25,
            "calm_inversion": 0.05,
            "western_disturbance": 0.10,
            "strong_ventilation": 0.15,
        },
    },
}

# ── Forecast noise parameters ───────────────────────────────────────────

FORECAST_PM25_NOISE_PCT = 0.15   # ±15% error on 1-day forecasts
FORECAST_PM25_NOISE_2D_PCT = 0.25  # ±25% error on 2-day forecasts
FORECAST_WIND_DIR_NOISE = 30     # ±30 degrees on wind direction forecast


def _deterministic_hash(seed: int, day: int, city: str, salt: str = "") -> float:
    """Generate a deterministic pseudo-random float in [0, 1) from inputs.

    This ensures reproducibility without requiring numpy/random state.
    """
    data = f"{seed}:{day}:{city}:{salt}".encode()
    h = hashlib.sha256(data).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def generate_wind_direction(
    regime: str, seed: int, day: int, city: str,
) -> float:
    """Generate a wind direction (degrees) for the given conditions.

    Returns a value in [0, 360) based on the weather regime.
    """
    info = WEATHER_REGIMES.get(regime, WEATHER_REGIMES["normal"])
    dir_low, dir_high = info["wind_dir_range"]

    # Deterministic "random" within range
    r = _deterministic_hash(seed, day, city, "wind_dir")
    direction = dir_low + r * (dir_high - dir_low)
    return direction % 360


def generate_wind_speed(
    regime: str, base_speed: float, seed: int, day: int, city: str,
) -> float:
    """Generate a wind speed (km/h) blending regime expectations with base data.

    The base_speed from CPCB data is used as an anchor, then perturbed
    by the weather regime.
    """
    info = WEATHER_REGIMES.get(regime, WEATHER_REGIMES["normal"])
    lo, hi = info["wind_speed_range"]

    # Blend: 60% base data, 40% regime expectation
    regime_mid = (lo + hi) / 2
    r = _deterministic_hash(seed, day, city, "wind_speed")
    noise = (r - 0.5) * (hi - lo) * 0.4
    blended = 0.6 * base_speed + 0.4 * regime_mid + noise

    return round(max(0, blended), 1)


def transition_weather_regime(
    current_regime: str, seed: int, day: int, city: str,
) -> str:
    """Transition to the next day's weather regime using Markov chain.

    Transitions are deterministic given the seed/day/city.
    """
    info = WEATHER_REGIMES.get(current_regime, WEATHER_REGIMES["normal"])
    probs = info["transition_probs"]

    r = _deterministic_hash(seed, day, city, "regime_transition")

    cumulative = 0.0
    for regime, prob in probs.items():
        cumulative += prob
        if r < cumulative:
            return regime

    # Fallback (shouldn't reach here)
    return list(probs.keys())[-1]


def add_forecast_noise(
    true_pm25: float, seed: int, day: int, city: str, horizon: int = 1,
) -> float:
    """Add noise to a PM2.5 forecast value.

    Args:
        true_pm25: The actual (true) PM2.5 for the forecast day.
        seed: Episode seed for reproducibility.
        day: Current day number.
        city: City identifier.
        horizon: Forecast horizon (1 or 2 days ahead).

    Returns:
        Noisy forecast PM2.5 value.
    """
    noise_pct = FORECAST_PM25_NOISE_PCT if horizon == 1 else FORECAST_PM25_NOISE_2D_PCT
    r = _deterministic_hash(seed, day, city, f"forecast_noise_{horizon}")
    noise = (r - 0.5) * 2 * noise_pct  # [-noise_pct, +noise_pct]

    noisy = true_pm25 * (1.0 + noise)
    return round(max(5.0, noisy), 1)


def compute_noisy_forecast(
    city_days: List[Dict[str, Any]],
    current_day_idx: int,
    seed: int,
    city: str,
) -> List[Dict[str, Any]]:
    """Generate a 2-day forecast with realistic uncertainty.

    Returns list of forecast dicts with noisy PM2.5 and risk buckets.
    """
    from .simulator import get_risk_bucket

    forecasts = []
    for horizon in [1, 2]:
        idx = current_day_idx + horizon
        if idx < len(city_days):
            true_pm25 = city_days[idx]["pm25"]
            noisy_pm25 = add_forecast_noise(true_pm25, seed, current_day_idx, city, horizon)
            risk = get_risk_bucket(noisy_pm25)
            confidence = 0.85 if horizon == 1 else 0.65
            forecasts.append({
                "horizon_days": horizon,
                "pm25_forecast": noisy_pm25,
                "risk_bucket": risk,
                "confidence": confidence,
            })
        else:
            forecasts.append({
                "horizon_days": horizon,
                "pm25_forecast": None,
                "risk_bucket": "unknown",
                "confidence": 0.0,
            })

    return forecasts


class WeatherEngine:
    """Manage weather state across an episode with regime transitions."""

    def __init__(self, cities: List[str], seed: int = 42):
        self._cities = cities
        self._seed = seed
        self._regimes: Dict[str, str] = {}
        self._wind_dirs: Dict[str, float] = {}

        # Initialize all cities to a weather regime based on seed
        for city in cities:
            r = _deterministic_hash(seed, 0, city, "init_regime")
            if r < 0.4:
                self._regimes[city] = "calm_inversion"
            elif r < 0.65:
                self._regimes[city] = "light_haze"
            else:
                self._regimes[city] = "normal"

            self._wind_dirs[city] = generate_wind_direction(
                self._regimes[city], seed, 0, city
            )

    def get_regime(self, city: str) -> str:
        return self._regimes.get(city, "normal")

    def get_wind_direction(self, city: str) -> float:
        return self._wind_dirs.get(city, 315.0)

    def get_prevailing_wind_dir(self) -> float:
        """Get the average wind direction across all cities (for transport model)."""
        if not self._wind_dirs:
            return 315.0
        # Simple circular mean approximation
        sin_sum = sum(math.sin(math.radians(d)) for d in self._wind_dirs.values())
        cos_sum = sum(math.cos(math.radians(d)) for d in self._wind_dirs.values())
        avg = math.degrees(math.atan2(sin_sum, cos_sum)) % 360
        return round(avg, 1)

    def advance_day(self, day: int, base_wind_speeds: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Advance weather by one day for all cities.

        Args:
            day: New day number.
            base_wind_speeds: Base wind speeds from CPCB data.

        Returns:
            {city: {regime, wind_dir, wind_speed, pm25_multiplier}}
        """
        result = {}
        for city in self._cities:
            # Transition regime
            new_regime = transition_weather_regime(
                self._regimes[city], self._seed, day, city
            )
            self._regimes[city] = new_regime

            # Generate wind
            wind_dir = generate_wind_direction(new_regime, self._seed, day, city)
            self._wind_dirs[city] = wind_dir

            base_speed = base_wind_speeds.get(city, 5.0)
            wind_speed = generate_wind_speed(
                new_regime, base_speed, self._seed, day, city
            )

            pm25_mult = WEATHER_REGIMES[new_regime]["pm25_multiplier"]

            result[city] = {
                "regime": new_regime,
                "wind_dir": round(wind_dir, 1),
                "wind_speed": round(wind_speed, 1),
                "pm25_multiplier": pm25_mult,
                "description": WEATHER_REGIMES[new_regime]["description"],
            }

        return result
