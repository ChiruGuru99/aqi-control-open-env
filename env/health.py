# Copyright (c) 2025. AQI Control Environment for OpenEnv.
# All rights reserved.

"""
Public health impact model for PM2.5 exposure.

This module translates abstract PM2.5 numbers into tangible human costs:
    - Hospital admissions (respiratory/cardiovascular)
    - Excess mortality risk (per 100k population)
    - School closure status and children affected
    - DALY (Disability-Adjusted Life Years) burden
    - Outdoor worker exposure
    - Healthcare system capacity stress

Exposure-response functions are based on:
    - WHO Air Quality Guidelines 2021
    - Global Burden of Disease (GBD) 2019 methodology
    - India-specific epidemiological data from IHME

All numbers are *per-city, per-day* estimates.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

# ── City population profiles ────────────────────────────────────────────
# Approximate metro population figures (millions) with demographic breakdown.

CITY_POPULATIONS = {
    "delhi": {
        "total_million": 20.0,
        "children_pct": 0.22,      # Under 14
        "elderly_pct": 0.08,       # Over 65
        "outdoor_workers_pct": 0.15,
        "hospital_beds_per_1k": 1.5,
        "baseline_respiratory_rate": 12.0,  # per 100k/day
    },
    "lucknow": {
        "total_million": 3.5,
        "children_pct": 0.25,
        "elderly_pct": 0.07,
        "outdoor_workers_pct": 0.18,
        "hospital_beds_per_1k": 1.0,
        "baseline_respiratory_rate": 10.0,
    },
    "patna": {
        "total_million": 2.5,
        "children_pct": 0.27,
        "elderly_pct": 0.06,
        "outdoor_workers_pct": 0.20,
        "hospital_beds_per_1k": 0.7,
        "baseline_respiratory_rate": 11.0,
    },
    "kanpur": {
        "total_million": 3.0,
        "children_pct": 0.24,
        "elderly_pct": 0.07,
        "outdoor_workers_pct": 0.22,
        "hospital_beds_per_1k": 0.8,
        "baseline_respiratory_rate": 13.0,
    },
    "chandigarh": {
        "total_million": 1.2,
        "children_pct": 0.20,
        "elderly_pct": 0.09,
        "outdoor_workers_pct": 0.12,
        "hospital_beds_per_1k": 2.0,
        "baseline_respiratory_rate": 8.0,
    },
    "mumbai": {
        "total_million": 21.0,
        "children_pct": 0.21,
        "elderly_pct": 0.08,
        "outdoor_workers_pct": 0.16,
        "hospital_beds_per_1k": 1.8,
        "baseline_respiratory_rate": 9.0,
    },
    "kolkata": {
        "total_million": 15.0,
        "children_pct": 0.22,
        "elderly_pct": 0.09,
        "outdoor_workers_pct": 0.17,
        "hospital_beds_per_1k": 1.2,
        "baseline_respiratory_rate": 11.0,
    },
    "bengaluru": {
        "total_million": 12.5,
        "children_pct": 0.20,
        "elderly_pct": 0.07,
        "outdoor_workers_pct": 0.13,
        "hospital_beds_per_1k": 2.5,
        "baseline_respiratory_rate": 7.0,
    },
}

# ── WHO exposure-response coefficients ──────────────────────────────────
# β coefficients for log-linear dose-response (per 10 µg/m³ PM2.5)

BETA_RESPIRATORY_ADMISSION = 0.0080  # RR per 10 µg/m³
BETA_CARDIOVASCULAR_ADMISSION = 0.0065
BETA_ALL_CAUSE_MORTALITY = 0.0040
BETA_CHILD_ASTHMA = 0.0120  # Children are more sensitive

# ── DALY computation parameters ─────────────────────────────────────────
# Simplified DALY model: DALYs = YLL + YLD
# YLL from premature deaths, YLD from morbidity (hospital days)

DALY_WEIGHT_RESPIRATORY = 0.133   # Disability weight for acute respiratory illness
DALY_WEIGHT_CARDIOVASCULAR = 0.186
AVG_HOSPITAL_STAY_DAYS = 3.5
YEARS_OF_LIFE_LOST_PER_DEATH = 25.0  # Average YLL

# ── School closure thresholds (GRAP-aligned) ─────────────────────────────

SCHOOL_THRESHOLDS = {
    "open": (0, 120),
    "outdoor_restricted": (121, 250),
    "primary_closed": (251, 350),
    "all_closed": (351, 9999),
}

# ── Healthcare cost (₹ crores per 1000 admissions) ──────────────────────
HEALTHCARE_COST_PER_1000_ADMISSIONS = 5.0  # ₹ crores


def compute_exposure_response(
    pm25: float, beta: float, baseline_rate: float, population: float,
) -> float:
    """Compute excess cases using log-linear exposure-response.

    Uses the standard WHO model:
        RR = exp(β × ΔPM2.5 / 10)
        excess_cases = baseline_rate × (RR - 1) × population

    Args:
        pm25: Ambient PM2.5 concentration (µg/m³).
        beta: Dose-response coefficient per 10 µg/m³.
        baseline_rate: Baseline event rate per 100k/day.
        population: Population in millions.

    Returns:
        Estimated excess cases per day.
    """
    # Reference concentration: WHO guideline (15 µg/m³)
    delta = max(0, pm25 - 15.0)
    rr = math.exp(beta * delta / 10.0)
    excess_rate = baseline_rate * (rr - 1)  # per 100k/day
    excess_cases = excess_rate * (population * 10)  # convert million to 100k units
    return max(0, excess_cases)


def compute_hospital_admissions(
    pm25: float, city: str,
) -> Dict[str, float]:
    """Estimate daily hospital admissions from PM2.5 exposure.

    Returns:
        Dict with respiratory, cardiovascular, and total admissions.
    """
    pop = CITY_POPULATIONS.get(city, CITY_POPULATIONS["delhi"])
    total_pop = pop["total_million"]
    baseline = pop["baseline_respiratory_rate"]

    respiratory = compute_exposure_response(
        pm25, BETA_RESPIRATORY_ADMISSION, baseline, total_pop
    )
    cardiovascular = compute_exposure_response(
        pm25, BETA_CARDIOVASCULAR_ADMISSION, baseline * 0.7, total_pop
    )

    # Children extra burden
    child_pop = total_pop * pop["children_pct"]
    child_asthma = compute_exposure_response(
        pm25, BETA_CHILD_ASTHMA, baseline * 1.5, child_pop
    )

    total = respiratory + cardiovascular + child_asthma

    return {
        "respiratory": round(respiratory, 1),
        "cardiovascular": round(cardiovascular, 1),
        "child_asthma": round(child_asthma, 1),
        "total": round(total, 1),
    }


def compute_excess_mortality_risk(
    pm25: float, city: str,
) -> float:
    """Estimate excess mortality risk per 100k population per day.

    Based on GBD 2019 all-cause mortality dose-response.
    """
    pop = CITY_POPULATIONS.get(city, CITY_POPULATIONS["delhi"])
    # Higher risk for elderly
    elderly_factor = 1.0 + pop["elderly_pct"] * 5.0
    delta = max(0, pm25 - 15.0)
    rr = math.exp(BETA_ALL_CAUSE_MORTALITY * delta / 10.0)
    risk = (rr - 1) * elderly_factor * 0.1  # per 100k/day, scaled
    return round(risk, 3)


def compute_daly_burden(
    pm25: float, city: str,
) -> Dict[str, float]:
    """Estimate daily DALY burden from PM2.5 exposure.

    Returns:
        Dict with YLL, YLD, and total DALYs.
    """
    pop = CITY_POPULATIONS.get(city, CITY_POPULATIONS["delhi"])
    total_pop = pop["total_million"]

    admissions = compute_hospital_admissions(pm25, city)
    mortality_risk = compute_excess_mortality_risk(pm25, city)

    # YLD from morbidity: admissions × avg stay × disability weight
    yld_respiratory = (
        admissions["respiratory"] * AVG_HOSPITAL_STAY_DAYS / 365
        * DALY_WEIGHT_RESPIRATORY
    )
    yld_cardiovascular = (
        admissions["cardiovascular"] * AVG_HOSPITAL_STAY_DAYS / 365
        * DALY_WEIGHT_CARDIOVASCULAR
    )

    # YLL from excess mortality
    excess_deaths = mortality_risk * total_pop * 10 / 100000  # deaths/day
    yll = excess_deaths * YEARS_OF_LIFE_LOST_PER_DEATH / 365  # DALYs/day

    total = yld_respiratory + yld_cardiovascular + yll

    return {
        "yll": round(yll, 4),
        "yld": round(yld_respiratory + yld_cardiovascular, 4),
        "total_dalys": round(total, 4),
        "excess_deaths_estimate": round(excess_deaths, 2),
    }


def get_school_status(pm25: float) -> str:
    """Determine school closure status based on PM2.5 level."""
    for status, (low, high) in SCHOOL_THRESHOLDS.items():
        if low <= pm25 <= high:
            return status
    return "all_closed"


def compute_outdoor_worker_exposure(
    pm25: float, city: str, is_weekend: bool = False,
) -> Dict[str, float]:
    """Estimate outdoor worker PM2.5 exposure hours.

    Args:
        pm25: Ambient PM2.5 (µg/m³).
        city: City identifier.
        is_weekend: Whether today is a weekend (fewer outdoor workers).

    Returns:
        Dict with exposure metrics.
    """
    pop = CITY_POPULATIONS.get(city, CITY_POPULATIONS["delhi"])
    total_pop = pop["total_million"]
    outdoor_pct = pop["outdoor_workers_pct"]

    # Average outdoor exposure: 8 hours on workday, 4 on weekend
    hours = 4.0 if is_weekend else 8.0
    outdoor_workers = total_pop * 1e6 * outdoor_pct
    if is_weekend:
        outdoor_workers *= 0.5

    # Person-µg/m³-hours (cumulative exposure dose)
    exposure_dose = outdoor_workers * pm25 * hours

    # Risk categorization
    if pm25 > 300:
        risk = "extreme"
    elif pm25 > 200:
        risk = "very_high"
    elif pm25 > 120:
        risk = "high"
    elif pm25 > 60:
        risk = "moderate"
    else:
        risk = "low"

    return {
        "workers_exposed": round(outdoor_workers),
        "hours": hours,
        "exposure_dose": round(exposure_dose / 1e9, 2),  # In billions
        "risk_category": risk,
    }


def compute_healthcare_cost(admissions_total: float) -> float:
    """Estimate daily healthcare cost in ₹ crores."""
    return round(admissions_total / 1000 * HEALTHCARE_COST_PER_1000_ADMISSIONS, 2)


def compute_hospital_capacity_stress(
    admissions_total: float, city: str,
) -> Dict[str, Any]:
    """Assess hospital capacity stress from pollution-related admissions.

    Returns stress level and occupancy metrics.
    """
    pop = CITY_POPULATIONS.get(city, CITY_POPULATIONS["delhi"])
    total_beds = pop["total_million"] * 1e6 * pop["hospital_beds_per_1k"] / 1000

    # Assume 70% baseline occupancy
    available_beds = total_beds * 0.30
    pollution_bed_need = admissions_total * AVG_HOSPITAL_STAY_DAYS

    if available_beds > 0:
        stress_ratio = pollution_bed_need / available_beds
    else:
        stress_ratio = 999

    if stress_ratio > 0.8:
        level = "crisis"
    elif stress_ratio > 0.5:
        level = "strained"
    elif stress_ratio > 0.2:
        level = "elevated"
    else:
        level = "normal"

    return {
        "stress_level": level,
        "stress_ratio": round(stress_ratio, 3),
        "available_beds": round(available_beds),
        "pollution_bed_demand": round(pollution_bed_need),
    }


class HealthTracker:
    """Track cumulative health impacts across an episode."""

    def __init__(self, cities: List[str]):
        self._cities = cities
        self._cumulative: Dict[str, Dict[str, float]] = {
            c: {
                "total_admissions": 0.0,
                "total_dalys": 0.0,
                "total_excess_deaths": 0.0,
                "total_healthcare_cost": 0.0,
                "school_days_lost": 0.0,
            }
            for c in cities
        }

    def record_day(
        self, city: str, pm25: float, is_weekend: bool = False,
    ) -> Dict[str, Any]:
        """Compute and record health impacts for one city-day.

        Returns a comprehensive health impact summary.
        """
        admissions = compute_hospital_admissions(pm25, city)
        mortality = compute_excess_mortality_risk(pm25, city)
        dalys = compute_daly_burden(pm25, city)
        school = get_school_status(pm25)
        workers = compute_outdoor_worker_exposure(pm25, city, is_weekend)
        capacity = compute_hospital_capacity_stress(admissions["total"], city)
        healthcare_cost = compute_healthcare_cost(admissions["total"])

        pop = CITY_POPULATIONS.get(city, CITY_POPULATIONS["delhi"])

        # School days lost
        children = pop["total_million"] * 1e6 * pop["children_pct"]
        if school == "all_closed":
            school_days = children
        elif school == "primary_closed":
            school_days = children * 0.5  # Only primary
        else:
            school_days = 0

        # Accumulate
        cum = self._cumulative[city]
        cum["total_admissions"] += admissions["total"]
        cum["total_dalys"] += dalys["total_dalys"]
        cum["total_excess_deaths"] += dalys["excess_deaths_estimate"]
        cum["total_healthcare_cost"] += healthcare_cost
        cum["school_days_lost"] += school_days

        return {
            "admissions": admissions,
            "excess_mortality_risk_per_100k": mortality,
            "dalys": dalys,
            "school_status": school,
            "outdoor_workers": workers,
            "hospital_capacity": capacity,
            "healthcare_cost_crores": healthcare_cost,
            "children_affected": round(school_days),
        }

    def get_cumulative(self, city: str) -> Dict[str, float]:
        return dict(self._cumulative.get(city, {}))

    def get_summary(self) -> Dict[str, Any]:
        """Return episode-wide health summary across all cities."""
        total_admissions = sum(
            c["total_admissions"] for c in self._cumulative.values()
        )
        total_dalys = sum(
            c["total_dalys"] for c in self._cumulative.values()
        )
        total_deaths = sum(
            c["total_excess_deaths"] for c in self._cumulative.values()
        )
        total_cost = sum(
            c["total_healthcare_cost"] for c in self._cumulative.values()
        )
        total_school = sum(
            c["school_days_lost"] for c in self._cumulative.values()
        )

        return {
            "total_hospital_admissions": round(total_admissions, 0),
            "total_dalys": round(total_dalys, 2),
            "total_excess_deaths": round(total_deaths, 1),
            "total_healthcare_cost_crores": round(total_cost, 2),
            "total_school_days_lost": round(total_school, 0),
            "per_city": {
                c: {k: round(v, 2) for k, v in d.items()}
                for c, d in self._cumulative.items()
            },
        }
