# 🌬️ AQI Policy Planner: Command Guide

Welcome to the **Regional AQI Command Center**. You are the Policy Planner for India's National Clean Air Programme. Your mission is to navigate the "Airpocalypse" while balancing public health, economic stability, and political sentiment.

### 🎮 How to Test
1. **Initialize**: Click **Reset** to start. Use `task_id` "easy", "medium", "hard", "expert", or "crisis".
2. **Choose Your Framework**: 
   - **GRAP Mode**: Select `grap_stage` (1-4) to auto-activate mandated multi-sector interventions.
   - **Manual Mode**: Choose specific actions like `close_schools`, `mandate_wfh`, or `limit_industry`.
3. **Submit**: Click **Step** to advance by one day.

### 📊 Regional Crisis Metrics
- **🏥 Health Impact**: Monitor hospital admissions and mortality risk. Track **DALYs** to see the long-term human cost.
- **📰 Public Sentiment**: Keep an eye on the news! If sentiment drops too low, the public will force a policy reversal.
- **🔄 Inter-City Transport**: Watch out for pollution "flowing" in from your neighbors (e.g., Chandigarh → Delhi).
- **📉 Policy Fatigue**: Repeated lockdowns lose effectiveness. Learn when to ease restrictions to maintain "policy capital."

### 🤖 For Developers
Use the OpenEnv client to evaluate your agent:
```python
from openenv.core.client import RemoteEnvClient
env = RemoteEnvClient("NaveenVenu/aqi-control-open-env")
# Try the 'expert' task for the full transport/GRAP experience
obs = await env.reset(task_id="expert")
```
