# 📑 AQI Policy Planner: Technical Specification

This document details the internal physics, epidemiologic models, and reward functions underpinning the **AQI Control Environment**.

---

## 🏥 Public Health Modeling
The environment uses a log-linear exposure-response model to estimate health impacts, based on WHO Air Quality Guidelines (2021) and the Global Burden of Disease (GBD) methodology.

### 1. Hospital Admissions
$$RR = \exp(\beta \times \Delta PM2.5 / 10)$$
Excess cases are computed for Respiratory ($\beta=0.008$) and Cardiovascular ($\beta=0.0065$) admissions.

### 2. DALYs (Disability-Adjusted Life Years)
$$DALYs = YLL (Years of Life Lost) + YLD (Years Lived with Disability)$$
- **YLL**: Derived from excess mortality risk per 100k population.
- **YLD**: Derived from pollution-related hospital days.

---

## 🗺️ Regional Pollution Transport
Pollution in the Indo-Gangetic Plain (IGP) is highly transboundary. The environment models this using a directional wind-driven corridor system.

- **Corridors**: Chandigarh → Delhi → Lucknow → Patna → Kolkata.
- **Transport Physics**: Wind direction (NW → SE prevailing) and wind speed determine the fraction of excess source PM2.5 carried to downwind cities.
- **Lag Buffers**: Pollution takes 1-2 days to travel between major nodes.
- **Regional Haze**: Stochastic stubble-burning events create a "regional basement" of PM2.5 affecting all IGP cities simultaneously.

---

## ⚖️ GRAP Protocol (Graded Response Action Plan)

The simulator implements India's official 4-stage emergency response framework:

| Stage | PM2.5 Threshold | Primary Interventions |
| :--- | :--- | :--- |
| **Stage I** | > 60 (Poor) | Dust control, trash-burning bans, construction monitoring. |
| **Stage II** | > 120 (Very Poor) | Diesel generator bans, increased parking fees, smog guns. |
| **Stage III** | > 250 (Severe) | Major construction ban, stone crusher shutdown, mining ban. |
| **Stage IV** | > 300 (Severe+) | Truck entry ban (except essentials), 50% WFH, school closures. |

---

## 📊 Reward Function (High-Fidelity)

The reward is a multi-objective function designed to penalize human suffering and economic disruption.

$$Reward = R_{health} + R_{econ} + R_{policy} + R_{sentiment} + R_{equity}$$

- **$R_{health}$**: Penalizes DALYs lost and hospital capacity stress.
- **$R_{econ}$**: Direct cost of interventions, amplified by **Policy Fatigue**.
- **$R_{sentiment}$**: Scaled bonus for keeping public support above 0.7; severe penalty for dropping below 0.3.
- **$R_{policy}$**: Bonus for proactive action on 48h forecasts.
- **$R_{equity}$**: Penalty for high variance in health outcomes across cities.

---

## 🔍 Data Models (Expanded)

### Action Space (`AQIAction`)
Now supports multi-action bundles and direct GRAP stage selection:
- `grap_stage`: 0-4
- `actions`: List of `{action_type, level}`
- `close_schools`, `mandate_wfh`, `deploy_smog_response`, `emergency_transport_ban`.

### Observation Space (`AQIObservation`)
High-fidelity monitoring including:
- `estimated_hospital_admissions`
- `daly_burden_today`
- `public_sentiment` (0-1)
- `transport_pm25_received`
- `wind_regime`: `calm_inversion`, `normal`, `strong_ventilation`.

---

## 📜 Metadata
- **OpenEnv Spec**: v1 Compliant.
- **Data Engine**: Stochastic Markov Transitions + Deterministic Hash Seeds for reproducibility.
- **Runtime**: Python 3.10+ / FastAPI.
