import json
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

import logging
import requests
from openai import OpenAI

# ── Configuration ────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")

# Kaggle typically requests specific tasks via environment variables mapping
TASK_NAME = os.getenv("OPENENV_TASK", os.getenv("AQI_TASK", "easy"))
BENCHMARK = os.getenv("AQI_BENCHMARK", "aqi_control_env")

MAX_STEPS = 35  # Episode max steps, typically 30 + buffer
SUCCESS_SCORE_THRESHOLD = 0.5  # Example threshold for success booleans
MAX_POSSIBLE_SCORE = 150.0  # Approx expected optimal cumulative reward for [0, 1] normalization

client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
)

# Dedicated logger for structured stdout markers. Use a bare message
# formatter so the output matches the exact required [START]/[STEP]/[END]
# line formats (no timestamps or level names).
logger = logging.getLogger("inference_logger")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


# ── Stdout Formatting ────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    logger.info(f"[START] task={task} env={env} model={model}")

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Replace newlines in action string to comply with Kaggle's single-line constraint
    action_clean = action.replace("\n", " ").replace("\r", "")
    logger.info(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}"
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    logger.info(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}")


# ── Environment Helpers ─────────────────────────────────────────────────

def env_reset(task_id: str) -> Dict[str, Any]:
    resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()

def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{ENV_URL}/step", json={"action": action}, timeout=30)
    resp.raise_for_status()
    return resp.json()

def env_close() -> None:
    # Not strictly required for the requests backend, but added for completeness
    try:
        requests.post(f"{ENV_URL}/close", timeout=10)
    except Exception:
        pass


# ── Agent Logic ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a PM2.5 policy planner for Indian cities under NCAP guidelines.
Each day you receive PM2.5 readings, forecasts, weather, and activity levels.
Respond with JSON only format: {"action_type": "<TYPE>", "level": <INT>, "city": "<str>"}
Types: no_action, restrict_traffic, curtail_construction, limit_industry, issue_public_advisory.
"""

def build_prompt(obs: Dict[str, Any]) -> str:
    o = obs.get("observation", obs)
    parts = [
        f"Day {o.get('day', 0)}/{o.get('total_days', 30)} — {o.get('city', 'Delhi')}",
        f"PM2.5 baseline (today): {o.get('pm25_today', 0)} ug/m3",
        f"Forecast risk next 2d: {o.get('forecast_risk_next_2d', [])}",
        f"Wind regime: {o.get('wind_regime', 'normal')}, Stubble burning: {'YES' if o.get('stubble_burning_flag') else 'no'}",
        f"Is Weekend: {'YES' if o.get('is_weekend') else 'NO'}",
        f"Policy Fatigue Index: {o.get('policy_fatigue_index', 0.0)}",
        "\nRespond with a single JSON action object. No markdown."
    ]
    return "\n".join(parts)

def call_llm(prompt: str) -> Dict[str, Any]:
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            content = (response.choices[0].message.content or "").strip()
            # Strip standard markdown block wrappers if present
            if content.startswith("```"):
                content = content.split("\n", 1)[-1]
            if content.endswith("```"):
                content = content.rsplit("```", 1)[0]
                
            return json.loads(content.strip())
        except Exception:
            pass
    return {"action_type": "no_action", "level": 0}


# ── Main Loop ────────────────────────────────────────────────────────────

def main() -> None:
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env_reset(TASK_NAME)
        
        for step in range(1, MAX_STEPS + 1):
            obs_inner = obs.get("observation", obs)
            
            # 1. Provide Context to Agent
            prompt = build_prompt(obs)
            
            # 2. Get Action
            action = call_llm(prompt)
            action_str = json.dumps(action)
            
            # 3. Step Environment
            error = None
            try:
                obs = env_step(action)
            except Exception as e:
                error = str(e)
            
            reward = obs.get("reward", 0.0) or 0.0
            done = obs.get("done", False) or obs_inner.get("done", False)

            rewards.append(reward)
            steps_taken = step

            # 4. Standard STDOUT Emission
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done or error:
                # Securely parse grader percentage directly from the observation fields
                grader_scores = obs_inner.get("grader_scores", {})
                
                if TASK_NAME in grader_scores:
                    score = grader_scores[TASK_NAME]
                else:
                    # Fallback default generic normalization 
                    score = sum(rewards) / MAX_POSSIBLE_SCORE if MAX_POSSIBLE_SCORE > 0 else 0.0
                    
                # Absolutely guarantee strict exclusive bounds (0 < score < 1)
                # Use the same safe floor/ceiling as graders to avoid external
                # validators rounding tiny values to 0.0 or 1.0.
                score = max(0.001, min(score, 0.999))
                
                success = score >= SUCCESS_SCORE_THRESHOLD
                break

    except Exception as e:
        print(f"[DEBUG] Runtime Exception: {e}\n{traceback.format_exc()}", file=sys.stderr)
    finally:
        env_close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
