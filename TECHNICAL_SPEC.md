# 📑 AQI Control Environment: Technical Specification

This document summarizes the internal logic, Reward functions, and Observation/Action spaces for the **AQI Control Environment**.

---

## 🏗️ Core Mechanics

### 1. Episode Simulation
- **Duration**: 30-day winter seasons.
- **Data Source**: Real CPCB (Central Pollution Control Board) time-series data for non-attainment cities.
- **Meteorology**: Dynamic wind regimes and inversions that amplify or ventilate pollution.

### 2. Intervention Physics
Interventions reduce PM2.5 based on:
- **City Profile**: Local source shares (Vehicular vs Construction vs Industrial).
- **Policy Fatigue**: Consecutive high-level interventions build "fatigue," reducing effectiveness and increasing economic costs.
- **Meteorology**: Effectiveness is lower during "Calm Inversion" periods.

---

## 📊 Reward Function Detail

The environment provides a dense reward signal on every step:

| Component | Formula | Description |
|-----------|---------|-------------|
| **Health Penalty** | `-(PM2.5 - 60) × 0.1` | Tied to Indian NAAQS standards (60 µg/m³). |
| **Economic Penalty** | `-cost(action) × fatigue_surcharge` | Intervention costs scaled by cumulative fatigue. |
| **Policy Bonus** | `+1.5` per preemptive action | Proactive actions on forecasted risk days. |
| **Fatigue Penalty** | `-fatigue × 1.5` | Penalty for sustained emergency lockdowns. |
| **Equity Penalty** | `-std_dev(city_health) × 0.5` | Penalty for unequal outcomes across Patna, Lucknow, and Delhi. |

---

## 🔍 Data Models

### Action Space (AQIAction)
```json
{
  "action_type": "restrict_traffic | curtail_construction | limit_industry | issue_public_advisory | no_action",
  "level": 0-3,
  "city": "delhi | lucknow | patna"
}
```

### Observation Space (AQIObservation)
```json
{
  "day": 15,
  "city": "Delhi",
  "pm25_today": 380,
  "pm25_post_intervention": 304,
  "forecast_risk_next_2d": ["severe", "high"],
  "wind_regime": "calm_inversion",
  "policy_fatigue_index": 0.45,
  "reward_breakdown": {
    "health_cost": -24.4,
    "economic_cost": -12.5,
    "equity_penalty": -0.0
  }
}
```

---

## 📋 Evaluation Tasks

- **Easy**: Delhi only. Focus on single-city threshold management.
- **Medium**: Delhi + Lucknow. Adds City Equity constraints.
- **Hard**: Delhi + Lucknow + Patna + Budget. Limited "Money" units for the whole season.

---

## 📜 Metadata
- **OpenEnv Support**: Fully compliant.
- **License**: Apache-2.0.
- **Simulation**: Python-based Discrete Event Simulator.
