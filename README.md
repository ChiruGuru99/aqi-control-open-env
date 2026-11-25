---
title: AQI Policy Planner — Indian Air Quality Simulator
emoji: 🏙️
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 8000
tags: [openenv, openenv-hackathon, aqi, epidemiology, public-policy]
---

# 🌬️ AQI Policy Planner: A High-Fidelity Indian Air Quality Simulator

The **AQI Control Environment** is a sophisticated OpenEnv benchmark designed to evaluate AI agents on managing complex, real-world public policy trade-offs during the hazardous Indian winter air quality crisis. It goes beyond simple thresholding to model the **interconnected systems** of atmospheric physics, public health, and political economy.

## 🌟 The Challenge
You are the **Regional Air Quality Command Center**. You must navigate the "Airpocalypse" across 8 major Indian cities. Every decision has a human cost:
- If you lock down too late, **hospital admissions spike and mortality risk rises**.
- If you lock down too hard, **economic pain and public backlash** force a policy reversal.
- If you ignore your neighbors, **pollution transport** from upwind cities will overwhelm your local efforts.

## 📈 High-Fidelity Features (Expert-Tier)
Unlike generic benchmarks, this environment models the **true complexity** of Indian policy planning:

1. **🏥 Public Health Economics**: Models hospital admissions, excess mortality risk, and **DALYs (Disability-Adjusted Life Years)** based on WHO and GBD 2019 methodologies.
2. **⚖️ Graded Response Action Plan (GRAP)**: Full implementation of India's 4-stage emergency response framework, including mandatory durations and sector-specific bundles.
3. **🗺️ Regional Transport**: Models directional PM2.5 transport between cities (e.g., Chandigarh → Delhi → Lucknow) with 1-2 day lag buffers and regional stubble burning haze.
4. **⛈️ Stochastic Weather**: Weather regime transitions (Calm Inversion, Western Disturbances) with **forecast uncertainty**.
5. **📰 Political Economy**: Public sentiment tracking (0-1). Media pressure, industry protests, and **forced policy reversals** if the agent loses public trust.
6. **🔄 Multi-Sector Interventions**: Simultaneously manage traffic, construction, industry, school closures, WFH mandates, and anti-smog deployments.

## 📋 Evaluation Tasks

| Task | Cities | Difficulty | Focus |
| :--- | :--- | :--- | :--- |
| **Easy** | Delhi | 🟢 Easy | Basic threshold management (NAAQS standards). |
| **Medium** | 2 Cities | 🟡 Medium | Multi-city coordination and geographic equity. |
| **Hard** | 3 Cities | 🔴 Hard | Strict budget discipline and policy fatigue. |
| **Expert** | 5 Cities | 💎 Expert | **GRAP Protocol** and regional pollution transport. |
| **Crisis** | 8 Cities | 🔥 Crisis | **Airpocalypse**: Hospital capacity stress and survival. |

## Reward design & verification

Each episode receives one transparent terminal grader score in `(0, 1)`.
`pm25_base` is the no-action counterfactual computed with
`apply_multi_action([{"action_type": "no_action", "level": 0}])` using the
same city profile, fatigue, weekend status, wind regime, and festival flag as
the real action. `pm25_input` preserves the pre-intervention input for
debugging. This prevents ordinary weekend, festival, and weather dynamics
from being counted as intervention effects.

The terminal components are:

| Task | Components |
|---|---|
| Easy | Mean PM2.5 reduction (40%), severe-day reduction (30%), intervention efficiency (30%). |
| Medium | Mean reduction (30%), severe-day reduction (30%), efficiency (20%), regional equity (20%). |
| Hard | Mean reduction (25%), severe-day reduction (20%), efficiency (15%), equity (15%), then an overspend multiplier. |
| Expert | DALY outcome (25%), mean PM2.5 reduction (20%), GRAP compliance (15%), equity (15%), sentiment (15%), then an overspend multiplier. |
| Crisis | Health (35%), hospital capacity (20%), GRAP compliance (15%), equity (15%), sentiment (10%), then an overspend multiplier. Crisis health is the mean of death, DALY, and counterfactual PM2.5 scores. |

Reduction scores use task-specific max-everything reference reductions
generated over seeds 0-4 by
`scripts/compute_reference_normalizers.py`. If no counterfactual severe day
exists, the severe component is omitted and the remaining weights are
renormalized. No-action efficiency is `0.5`; costed GRAP efficiency is
PM2.5 benefit per economic cost, bounded to 1 and multiplied by protocol
alignment. Spending within budget has no terminal penalty. If a daily bundle
would exceed the budget, it is rejected in full as same-day no-action—there
is no downgrade or partial application.

`tests/test_reward_calibration.py` verifies determinism, finite `(0, 1)`
scores, `threshold > noop`, `noop > anti_threshold`,
`anti_threshold < threshold/budget_aware`, `max_everything` not being top,
hard `budget_aware > noop/threshold`, no-op neutrality, action/range
hardening, budget enforcement, and dense-reward/grader Spearman
correlation of at least `0.8`. The held-out random and lagged-threshold
policies are checked on seeds 5-9 after calibration.

### Policy audit

| Task | Phase | noop | threshold | cost-blind max | anti-threshold | budget-aware |
|---|---|---:|---:|---:|---:|---:|
| Easy | Before / after | 0.300 / 0.150 | 0.393 / 0.840 | 0.411 / 0.821 | 0.154 / 0.077 | 0.393 / 0.840 |
| Medium | Before / after | 0.400 / 0.300 | 0.453 / 0.889 | 0.460 / 0.881 | 0.351 / 0.252 | 0.453 / 0.889 |
| Hard | Before / after | 0.550 / 0.300 | 0.326 / 0.497 | 0.332 / 0.322 | 0.509 / 0.272 | 0.485 / 0.547 |
| Expert | Before / after | 0.338 / 0.265 | 0.314 / 0.406 | 0.250 / 0.291 | 0.297 / 0.239 | 0.323 / 0.331 |
| Crisis | Before / after | 0.439 / 0.311 | 0.506 / 0.485 | 0.389 / 0.351 | 0.417 / 0.295 | 0.441 / 0.350 |

See `docs/audit_before.md`, `docs/audit_after.md`, and
`docs/audit_heldout.md` for the complete tables and the easy seed-0
lagged-threshold trace. Held-out results retain known limitations:
lagged-threshold narrowly beats threshold on easy/medium, while random stays
below threshold on every task. Easy and medium are seed-independent; seeds
5-9 add stochastic coverage only for hard, expert, and crisis.

## 🛠️ Developer Resources
- **📄 [Technical Specification](TECHNICAL_SPEC.md)**: Details on DALY math, transport physics, and the reward engine.
- **🏃 [Inference Script](inference.py)**: Baseline script with rich multi-city prompt engineering.
- **⚙️ [OpenEnv.yaml](openenv.yaml)**: Task definitions and programmatic graders.

### Quick Start
```python
from openenv.core.client import RemoteEnvClient
from models import AQIAction

# Connect to the Regional Command Center
env = RemoteEnvClient("NaveenVenu/aqi-control-open-env")

obs = await env.reset(task_id="expert")
# Activate GRAP Stage III for Delhi
action = AQIAction(grap_stage=3, city="delhi")
obs, reward, done, info = await env.step(action)
```

---
**License**: Apache-2.0
**Built for**: OpenEnv-Hackathon
**Impact**: Helping AI solve the world's most pressing public health challenges.
