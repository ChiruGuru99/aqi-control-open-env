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
