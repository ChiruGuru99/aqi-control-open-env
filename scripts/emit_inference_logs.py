#!/usr/bin/env python3
"""Emit sample [START]/[STEP]/[END] stdout logs for verification.

Run this inside the container or Space to confirm that stdout is captured
by the platform's logging (e.g., `docker logs` or Spaces logs).
"""
import json
import os
import sys
import time

TASK = os.getenv("OPENENV_TASK", "easy")
ENV = os.getenv("AQI_BENCHMARK", "aqi_control_env")
MODEL = os.getenv("MODEL_NAME", "gpt-4o-mini")


def main():
    print(f"[START] task={TASK} env={ENV} model={MODEL}", flush=True)

    rewards = []
    for step in range(1, 6):
        action = {"action_type": "no_action", "level": 0, "city": "Delhi"}
        action_str = json.dumps(action)
        reward = round(0.1 * step, 2)
        done = "false"
        error = "null"
        print(
            f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done} error={error}",
            flush=True,
        )
        rewards.append(reward)
        time.sleep(0.05)

    score = sum(rewards) / 150.0
    score = max(0.001, min(score, 0.999))
    success = "true" if score >= 0.5 else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success} steps={len(rewards)} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


if __name__ == "__main__":
    main()
