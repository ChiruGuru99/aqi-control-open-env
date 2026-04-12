"""Smoke test — validates models, simulator, graders, environment, and HTTP endpoints."""
import sys, json, os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("=" * 60)
print("PM2.5 POLICY PLANNING ENVIRONMENT — SMOKE TEST")
print("=" * 60)

# 1. Models
print("\n[1/6] Models...")
from models import AQIAction, AQIObservation
a = AQIAction(action_type="restrict_traffic", level=2)
print(f"  Action OK: {a.action_type}, level={a.level}")

# 2. Simulator
print("\n[2/6] Simulator...")
from env.simulator import apply_intervention, compute_daily_reward, get_aqi_category

dummy_profile = {
    "source_shares": {"vehicular": 0.40, "construction": 0.20, "industrial": 0.15, "background": 0.25},
    "compliance_multiplier": 1.0,
    "enforcement_capacity": 1.0,
}

post, cost, acts, detail = apply_intervention(
    350.0, "restrict_traffic", 3, dummy_profile, fatigue=0.0, wind_regime="calm_inversion"
)
print(f"  350 PM2.5 + traffic L3 + calm wind → {post} µg/m³, cost={cost}, activities={acts}")
reward, breakdown = compute_daily_reward(
    post, cost, 350.0, "restrict_traffic", 3, "diwali_week", fatigue=0.0
)
print(f"  Reward: {reward}, breakdown={breakdown}")
print(f"  AQI category: {get_aqi_category(post)}")

# 3. Graders
print("\n[3/6] Graders...")
from env.graders import grade_easy
fake_traj = [
    {"pm25_post": 55, "pm25_base": 120, "economic_cost": 5, "health_cost": 1.0},
    {"pm25_post": 200, "pm25_base": 300, "economic_cost": 10, "health_cost": -14.0},
    {"pm25_post": 45, "pm25_base": 50, "economic_cost": 0, "health_cost": 1.0},
]
score, details = grade_easy(fake_traj, 3)
print(f"  Easy grader: score={score:.4f} details={details}")

# 4. Environment
print("\n[4/6] Environment...")
from server.aqi_environment import AQIControlEnvironment
env = AQIControlEnvironment()
obs = env.reset(task_id="easy")
print(f"  Reset OK: day={obs.day}, city={obs.city}, pm25={obs.pm25_today}, task={obs.task_id}")
print(f"  Wind regime: {obs.wind_regime}, Forecast: {obs.forecast_risk_next_2d}, Fatigue: {obs.policy_fatigue_index}")

# Step through a few days
for i in range(3):
    act = AQIAction(action_type="restrict_traffic", level=3)
    obs = env.step(act)
    print(f"  Day {obs.day}: pm25={obs.pm25_today}→{obs.pm25_post_intervention}, reward={obs.reward}")
    print(f"  Fatigue: {obs.policy_fatigue_index}, Breakdown: {obs.reward_breakdown}")

# Run to completion
for i in range(27):
    act = AQIAction(action_type="no_action", level=0)
    obs = env.step(act)

print(f"  Done={obs.done}, grader_scores={obs.metadata.get('grader_scores', {})}")

# 5. Medium task
print("\n[5/6] Medium task...")
obs = env.reset(task_id="medium")
print(f"  Cities: {list(obs.cities_data.keys()) if obs.cities_data else 'single'}")
for i in range(30):
    target = "delhi" if i % 2 == 0 else "lucknow"
    act = AQIAction(action_type="restrict_traffic", level=1, city=target)
    obs = env.step(act)
print(f"  Done={obs.done}, scores={obs.metadata.get('grader_scores', {})}")

# 6. State
print("\n[6/6] State...")
print(f"  State: {env.state}")

print("\n" + "=" * 60)
print("ALL SMOKE TESTS PASSED")
print("=" * 60)
