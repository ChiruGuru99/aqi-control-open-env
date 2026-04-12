"""HTTP endpoint test against the running server."""
import requests, json

BASE = "http://localhost:8000"

print("=== RESET (easy) ===")
r = requests.post(f"{BASE}/reset", json={})
print(f"Status: {r.status_code}")
obs = r.json()
o = obs["observation"]
print(f"Day: {o['day']}, City: {o['city']}, PM2.5: {o['pm25_today']}, Task: {o['task_id']}")

print("\n=== STEP (restrict_traffic L2) ===")
r = requests.post(f"{BASE}/step", json={
    "action": {"action_type": "restrict_traffic", "level": 2}
})
print(f"Status: {r.status_code}")
s = r.json()
print(f"Reward: {s['reward']}, Done: {s['done']}")
print(f"PM2.5 post: {s['observation']['pm25_post_intervention']}")

print("\n=== STEP (issue_public_advisory) ===")
r = requests.post(f"{BASE}/step", json={
    "action": {"action_type": "issue_public_advisory", "level": 0}
})
s = r.json()
print(f"Reward: {s['reward']}, Day: {s['observation']['day']}")

print("\n=== HEALTH ===")
r = requests.get(f"{BASE}/health")
print(f"Status: {r.status_code}, Body: {r.json()}")

print("\n=== SCHEMA ===")
r = requests.get(f"{BASE}/schema")
print(f"Status: {r.status_code}")
schema = r.json()
print(f"Action fields: {list(schema.get('action', {}).get('properties', {}).keys())}")

print("\n=== ALL HTTP ENDPOINTS WORKING ===")
