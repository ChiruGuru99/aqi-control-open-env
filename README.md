---
title: AQI Control Environment
emoji: 🏙️
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 8000
tags: [openenv]
---

# 🌬️ AQI Control Environment

Welcome to the **AQI Control Environment**! This is a state-of-the-art OpenEnv benchmark designed to evaluate how AI agents manage complicated public policy trade-offs during the hazardous Indian winter air quality crisis.

## 🌟 The Challenge
You (the Agent) are the City Policy Planner. You have a limited budget and a population that grows tired of lockdowns. Every day, you must look at PM2.5 forecasts, meteorology, and economic data to decide which sectors (traffic, construction, industry) to shut down.

**Can you keep the city breathing while keeping the economy moving?**

---

## 🚀 Get Started (Interactive Playground)
If you are visiting this Hugging Face Space, you can try the environment manually!

1. **Click Reset**: Click the "Reset" button in the Playground above to start the simulator.
2. **Select Action**: Choose an action like `traffic_control`.
3. **Intensity**: Choose a level from `0` to `3`.
4. **City**: Enter `Delhi`.
5. **Score**: Watch your reward update based on health costs and economic disruption.

---

## 🛠️ Developer Resources
To help the AI evaluator and developers understand the internal mechanics, please refer to:

- **📄 [Technical Specification](TECHNICAL_SPEC.md)**: Details on Reward math, intervention physics, and data models.
- **🏃 [Inference Script](inference.py)**: The baseline script required for competition submission.
- **⚙️ [OpenEnv.yaml](openenv.yaml)**: Environment metadata for the OpenEnv CLI.

### Quick Client Example
```python
from openenv.core.client import RemoteEnvClient
from models import AQIAction

# Connect to this live Space!
env = RemoteEnvClient("NaveenVenu/aqi-control-open-env")

obs = await env.reset()
action = AQIAction(action_type="restrict_traffic", level=2, city="Delhi")
obs, reward, done, info = await env.step(action)
```

---

## 📈 Environment Features
- **Real Data**: Powered by actual CPCB PM2.5 historical time-series.
- **NCAP Compliance**: Modeled after India's National Clean Air Programme.
- **Policy Fatigue**: High-intensity lockdowns become less effective and more expensive over time.
- **Dynamic Meteorolgy**: Wind regimes and seasonal festivals (Diwali) impact air quality.

---
**License**: Apache-2.0
**Built for**: OpenEnv Agentic Benchmarking Competition
