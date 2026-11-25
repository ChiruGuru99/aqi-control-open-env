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

## 📊 Reward and verification model

### Counterfactual baseline

For every city-day, `pm25_input` stores the input after transport and the
weather-engine multiplier. The terminal comparison baseline is:

```text
pm25_base = apply_multi_action(
    pm25_input,
    [{"action_type": "no_action", "level": 0}],
    city_profile,
    same fatigue, weekend, wind regime, and festival flag
)
```

The real action is evaluated through the same `apply_multi_action` path.
Thus, weekend, wind, festival, source-apportionment, and compliance dynamics
are identical in the two trajectories. The counterfactual consumes no RNG.

### Per-step dense reward

Let `B = max(0, pm25_base - pm25_post)` and `C` be economic cost. The dense
scalar returned for the step is:

$$
R_d = \frac{B}{\max(1,pm25_{base})}
      \times (1 + \mathbb{1}[pm25_{base}>250])
      \times (1+\tanh(B/C))
      \times A
$$

for `C > 0`; when `C = 0`, the dense air-quality term is zero. `A` is the
normalized GRAP compliance alignment for an explicit GRAP action and is 1
for a bundle/legacy action. Policy quality, fatigue, equity, forecast, and
sentiment remain in `reward_breakdown` as diagnostics, but are not summed
with `R_d` because their original unit scales reversed terminal-policy
ordering.

### Terminal efficiency

For a trajectory, total benefit is
`max(0, sum(pm25_base) - sum(pm25_post))`. A no-action trajectory has zero
benefit and zero cost and receives efficiency `0.5`. Costed efficiency is:

$$
E = \min(1, benefit / cost) \times A_{mean}
$$

where `A_mean` is the mean alignment for costed explicit GRAP days. Multi-city
efficiency is cost-weighted across city trajectories.

### Reduction normalizers

`scripts/compute_reference_normalizers.py` measures the raw mean and severe
reduction of `max_everything` over seeds 0-4. Named constants in
`env/graders.py` hold those measured references:

| Task | Mean reference | Severe reference |
|---|---:|---:|
| Easy | 0.102758466933 | 0.095745880584 |
| Medium | 0.051379233466 | 0.047872940292 |
| Hard | 0.034451784532 | 0.033028703707 |
| Expert | 0.016204830819 | 0.015620275932 |
| Crisis | 0.011285082270 | 0.135290134265 |

Each reduction component is `clip(metric / reference, 0, 1)`. If the
counterfactual has no severe days, that component is `None` rather than 1,
and `_weighted_score` divides by the remaining weight sum.

### Terminal grader formulas

The named weights below are renormalized over non-omitted components:

- **Easy**: `0.40 avg + 0.30 severe + 0.30 efficiency`.
- **Medium**: `0.30 avg + 0.30 severe + 0.20 efficiency + 0.20 equity`.
- **Hard**: `0.25 avg + 0.20 severe + 0.15 efficiency + 0.15 equity`,
  multiplied by `M_budget`.
- **Expert**: `0.25 DALY + 0.20 PM2.5 + 0.15 GRAP + 0.15 equity + 0.15 sentiment`,
  multiplied by `M_budget`.
- **Crisis**: `0.35 health + 0.20 hospital + 0.15 GRAP + 0.15 equity + 0.10 sentiment`,
  multiplied by `M_budget`. Crisis health is
  `(death_score + DALY_score + PM2.5_reduction_score) / 3`.

For a positive budget:

$$
M_{budget} = \max(0, 1 - s \times \max(0, spent/budget - 1))
$$

where `s=3` for hard/expert and `s=2` for crisis. Spending at or below budget
has multiplier 1, so inaction receives no free additive budget reward.

### Runtime constraints

Budget checking occurs before a cost is recorded. If
`spent + bundle_cost > budget`, the entire bundle is replaced by same-day
no-action; it is not downgraded or partially applied. The observation
message contains `action rejected; insufficient budget`, the reward
breakdown contains `budget_rejected_action=1`, and GRAP state/compliance is
rolled back. Unknown city values are rejected as no-action with an
`Invalid city ...` observation message. Out-of-range level/GRAP schema
values fail Pydantic validation and are defensively clamped if constructed
without validation.

### Calibration and held-out checks

`pytest -q tests/` checks deterministic replay, range/NaN safety, policy
ordering, no-op neutrality, severe-component renormalization, RNG
isolation, dense/grader Spearman rank agreement (`>= 0.8`), and invalid
inputs. `scripts/audit_policies.py --write-audit-docs` writes development
and held-out tables together. Known limitation: on easy/medium, the
lagged-threshold policy can narrowly beat current threshold because it reacts
to the prior post-intervention PM2.5, causing threshold-boundary oscillation
and changed fatigue. No grader was adjusted to hide this result.

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
