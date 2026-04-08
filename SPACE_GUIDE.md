# 🌬️ AQI Control Playground Guide

Welcome to the **AQI Control Environment**! You are now the Policy Planner for a major Indian city. Your goal is to manage PM2.5 levels during the hazardous winter months while balancing the economy and citizen happiness.

### 🎮 How to Test
1. **Initialize**: Click the **Reset** button to start a 30-day winter simulation.
2. **Choose an Action**: 
   - `traffic_control`: implement odd-even or vehicle bans.
   - `industrial_curtailment`: shut down factories.
   - `construction_ban`: stop dust-heavy building work.
3. **Set Intensity**: Enter a **Level** from `0` (none) to `3` (emergency).
4. **Submit**: Click **Step** to see how the PM2.5 levels and your Reward (Score) change.

### 📊 Vital Signs
- **PM2.5**: Lower is better. Above 150 is "Very Poor", above 250 is "Severe".
- **Reward**: Your performance score. High health costs or high economic loss will drop this.
- **Policy Fatigue**: Be careful! If you keep the city locked down too long, citizens will stop complying, and your interventions will lose power.

### 🤖 For Developers
If you are an AI agent, connect to this space using:
```python
from openenv.core.client import RemoteEnvClient
env = RemoteEnvClient("NaveenVenu/aqi-control-open-env")
```
