# Copyright (c) 2025. AQI Control Environment for OpenEnv.
# All rights reserved.

"""
Inter-city pollution transport model.

Real air pollution doesn't respect city boundaries. Wind carries PM2.5
from one city to its neighbours, stubble burning creates regional haze
events, and weather patterns create pollution corridors.

This module models:
    - Directional wind-driven PM2.5 transport between city pairs
    - Transport lag (pollution takes 1-2 days to travel between cities)
    - Regional haze events from stubble burning affecting all Indo-Gangetic
      Plain cities simultaneously
    - Topographic and distance-based attenuation

Transport physics is simplified but captures the key insight that agents
must think *regionally*, not just per-city.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

# ── Wind direction encoding ─────────────────────────────────────────────
# Degrees from north (0=N, 90=E, 180=S, 270=W)
# The prevailing winter wind over the Indo-Gangetic Plain is NW → SE

WIND_DIR_NAMES = {
    "N": 0, "NE": 45, "E": 90, "SE": 135,
    "S": 180, "SW": 225, "W": 270, "NW": 315,
}

# ── City geographic positions (approximate lat/lon) ─────────────────────

CITY_POSITIONS = {
    "delhi": (28.61, 77.21),
    "lucknow": (26.85, 80.95),
    "patna": (25.60, 85.10),
    "kanpur": (26.45, 80.35),
    "chandigarh": (30.73, 76.77),
    "mumbai": (19.08, 72.88),
    "kolkata": (22.57, 88.36),
    "bengaluru": (12.97, 77.59),
}

# ── Transport corridor definitions ──────────────────────────────────────
# Only nearby cities on the same geographic corridor have significant
# transport. Each entry: (source, dest): {base_rate, favourable_wind_dir,
# wind_cone_deg, lag_days}

TRANSPORT_CORRIDORS = {
    # Indo-Gangetic Plain corridor (NW → SE)
    ("delhi", "lucknow"): {
        "base_rate": 0.06,
        "favourable_wind_dir": 315,  # NW wind pushes Delhi→Lucknow
        "wind_cone_deg": 60,
        "lag_days": 1,
        "distance_km": 500,
    },
    ("delhi", "kanpur"): {
        "base_rate": 0.05,
        "favourable_wind_dir": 315,
        "wind_cone_deg": 60,
        "lag_days": 1,
        "distance_km": 440,
    },
    ("lucknow", "patna"): {
        "base_rate": 0.05,
        "favourable_wind_dir": 270,  # W wind pushes Lucknow→Patna
        "wind_cone_deg": 60,
        "lag_days": 1,
        "distance_km": 530,
    },
    ("kanpur", "lucknow"): {
        "base_rate": 0.07,
        "favourable_wind_dir": 270,
        "wind_cone_deg": 90,
        "lag_days": 0,  # Very close cities
        "distance_km": 80,
    },
    ("chandigarh", "delhi"): {
        "base_rate": 0.08,
        "favourable_wind_dir": 315,  # NW: stubble smoke flows south
        "wind_cone_deg": 60,
        "lag_days": 1,
        "distance_km": 250,
    },
    ("delhi", "patna"): {
        "base_rate": 0.03,
        "favourable_wind_dir": 315,
        "wind_cone_deg": 45,
        "lag_days": 2,
        "distance_km": 1000,
    },
    # Reverse corridors (weaker — against prevailing wind)
    ("lucknow", "delhi"): {
        "base_rate": 0.02,
        "favourable_wind_dir": 135,  # SE wind pushes back
        "wind_cone_deg": 45,
        "lag_days": 1,
        "distance_km": 500,
    },
    ("patna", "lucknow"): {
        "base_rate": 0.02,
        "favourable_wind_dir": 90,
        "wind_cone_deg": 45,
        "lag_days": 1,
        "distance_km": 530,
    },
    # Kolkata receives from Patna
    ("patna", "kolkata"): {
        "base_rate": 0.04,
        "favourable_wind_dir": 270,
        "wind_cone_deg": 60,
        "lag_days": 1,
        "distance_km": 580,
    },
}

# ── Stubble burning regional haze ────────────────────────────────────────
# Cities affected by Punjab/Haryana stubble burning
STUBBLE_AFFECTED_CITIES = {
    "delhi": 1.0,       # Ground zero
    "chandigarh": 0.9,  # Very close to burning fields
    "lucknow": 0.5,     # Downwind
    "kanpur": 0.4,
    "patna": 0.2,       # Far but still affected
    "kolkata": 0.1,
}

# PM2.5 contribution from a regional stubble event (ug/m3 at sensitivity=1.0)
STUBBLE_REGIONAL_CONTRIBUTION = 40.0


def _angular_distance(dir1: float, dir2: float) -> float:
    """Compute the absolute angular distance between two directions (degrees)."""
    diff = abs(dir1 - dir2) % 360
    return min(diff, 360 - diff)


def compute_transport_contribution(
    source_city: str,
    dest_city: str,
    source_excess_pm25: float,
    wind_dir_deg: float,
    wind_speed_kmh: float,
) -> float:
    """Compute the PM2.5 transported from source to destination city.

    Args:
        source_city: City emitting excess pollution.
        dest_city: City receiving transported pollution.
        source_excess_pm25: Excess PM2.5 above background at source (ug/m3).
        wind_dir_deg: Wind direction in degrees from north (0-360).
        wind_speed_kmh: Wind speed in km/h.

    Returns:
        Additional PM2.5 at destination from transport (ug/m3).
    """
    corridor = TRANSPORT_CORRIDORS.get((source_city, dest_city))
    if corridor is None:
        return 0.0

    base_rate = corridor["base_rate"]
    favourable_dir = corridor["favourable_wind_dir"]
    cone = corridor["wind_cone_deg"]

    # Wind alignment: how well does the wind push source → dest?
    angle_off = _angular_distance(wind_dir_deg, favourable_dir)
    if angle_off > cone:
        alignment = 0.0
    else:
        # Cosine taper within the cone
        alignment = math.cos(math.radians(angle_off * 90 / cone))

    # Wind speed factor: stronger wind = more transport (up to a point)
    if wind_speed_kmh <= 2:
        speed_factor = 0.3  # Calm: some diffusion but slow
    elif wind_speed_kmh <= 8:
        speed_factor = 0.5 + 0.5 * (wind_speed_kmh - 2) / 6
    else:
        speed_factor = 1.0  # Saturates

    transport = source_excess_pm25 * base_rate * alignment * speed_factor
    return round(max(0.0, transport), 1)


def compute_stubble_haze(
    city: str,
    stubble_flags: Dict[str, int],
    wind_dir_deg: float,
) -> float:
    """Compute regional PM2.5 contribution from stubble burning.

    Args:
        city: Target city.
        stubble_flags: Dict of {city: stubble_flag} for all active cities.
        wind_dir_deg: Prevailing wind direction.

    Returns:
        Additional PM2.5 from regional stubble haze (ug/m3).
    """
    sensitivity = STUBBLE_AFFECTED_CITIES.get(city, 0.0)
    if sensitivity <= 0:
        return 0.0

    # Count how many cities have active stubble burning
    active_stubble = sum(1 for v in stubble_flags.values() if v)
    if active_stubble == 0:
        return 0.0

    # NW wind amplifies stubble transport from Punjab
    nw_alignment = 1.0 - _angular_distance(wind_dir_deg, 315) / 180.0
    nw_alignment = max(0.2, nw_alignment)  # Always some contribution

    contribution = STUBBLE_REGIONAL_CONTRIBUTION * sensitivity * nw_alignment
    # Scale slightly with number of burning regions
    contribution *= min(1.5, 0.5 + active_stubble * 0.25)

    return round(contribution, 1)


class TransportModel:
    """Track inter-city pollution transport with lag buffers."""

    def __init__(self, cities: List[str]):
        self._cities = cities
        # Buffer for lagged transport: {(src, dst): [pm25_day0, pm25_day1, ...]}
        self._lag_buffer: Dict[Tuple[str, str], List[float]] = {}
        for (src, dst), corridor in TRANSPORT_CORRIDORS.items():
            if src in cities and dst in cities:
                lag = corridor["lag_days"]
                self._lag_buffer[(src, dst)] = [0.0] * max(1, lag)

    def step(
        self,
        city_pm25: Dict[str, float],
        city_background: Dict[str, float],
        wind_dir_deg: float,
        wind_speed_kmh: float,
        stubble_flags: Dict[str, int],
    ) -> Dict[str, float]:
        """Compute transport contributions for all cities for one timestep.

        Args:
            city_pm25: Current PM2.5 at each city (post-intervention).
            city_background: Background PM2.5 levels per city.
            wind_dir_deg: Wind direction in degrees.
            wind_speed_kmh: Wind speed in km/h.
            stubble_flags: Stubble burning flags per city.

        Returns:
            Dict of {city: total_transport_pm25_added} for this day.
        """
        transport_contributions: Dict[str, float] = {c: 0.0 for c in self._cities}

        for (src, dst), corridor in TRANSPORT_CORRIDORS.items():
            if src not in self._cities or dst not in self._cities:
                continue

            lag = corridor["lag_days"]
            excess = max(0, city_pm25.get(src, 0) - city_background.get(src, 30))

            # Compute today's transport at source
            today_transport = compute_transport_contribution(
                src, dst, excess, wind_dir_deg, wind_speed_kmh
            )

            # Push into lag buffer and pop from the other end
            buf = self._lag_buffer.get((src, dst), [0.0])
            if lag > 0:
                arrived = buf.pop(0)
                buf.append(today_transport)
                self._lag_buffer[(src, dst)] = buf
            else:
                arrived = today_transport

            transport_contributions[dst] = round(
                transport_contributions[dst] + arrived, 1
            )

        # Add stubble haze contribution
        for city in self._cities:
            haze = compute_stubble_haze(city, stubble_flags, wind_dir_deg)
            transport_contributions[city] = round(
                transport_contributions[city] + haze, 1
            )

        return transport_contributions
