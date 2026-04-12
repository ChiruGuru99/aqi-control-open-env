import json
import os
import sys
import traceback
import requests
from typing import Any, Dict, List, Optional
from openai import OpenAI

# ── Configuration (Hackathon Mandates) ───────────────────────────────────

# API_BASE_URL: The API endpoint for the LLM
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
# MODEL_NAME: The model identifier to use
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
# HF_TOKEN: Your Hugging Face / API key
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# LOCAL_IMAGE_NAME: Name of local image if starting via Docker
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")

# Environment Endpoint (Defaults to localhost if running in-container)
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

# Task Selection
TASK_NAME = os.getenv("OPENENV_TASK", os.getenv("AQI_TASK", "easy"))
BENCHMARK = "aqi_control_env"

# Logic Constraints
MAX_STEPS = 50  # Max episode length
SUCCESS_SCORE_THRESHOLD = 0.5

# OpenAI Client
client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
)

# ── Structured Logging (Hackathon Mandates) ─────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Ensure action string is sanitized for single-line stdout
    action_clean = str(action).replace("\n", " ").replace("\r", "")
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ── Environment API ─────────────────────────────────────────────────────

def env_reset(task_id: str) -> Dict[str, Any]:
    resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()

def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{ENV_URL}/step", json={"action": action}, timeout=30)
    resp.raise_for_status()
    return resp.json()

# ── Agent Logic (Restored Advanced Functionality) ───────────────────

SYSTEM_PROMPT = """You are a PM2.5 policy planner for Indian cities under NCAP guidelines.
You operate as a Regional Air Quality Command Center during the hazardous Indian winter.

Receive PM2.5 readings, forecasts, weather, and health data (hospital admissions,
school closures, DALYs), economic activity levels, public sentiment, and GRAP stage.

Respond with JSON only format:
  Single action:  {"action_type": "<TYPE>", "level": <INT>, "city": "<str>"}
  GRAP mode:      {"grap_stage": <0-4>, "city": "<str>"}
  Multi-action:   {"actions": [{"action_type": "<TYPE>", "level": <INT>}, ...], "city": "<str>"}

Action types:
  - no_action, restrict_traffic, curtail_construction, limit_industry
  - issue_public_advisory, close_schools, mandate_wfh
  - deploy_smog_response, emergency_transport_ban

GRAP stages: 0=Normal, 1=Poor(>60), 2=Very Poor(>120), 3=Severe(>250), 4=Severe+(>300)
Using grap_stage auto-activates the appropriate multi-sector interventions.

Balance health outcomes against economic cost, public sentiment, and budget.
"""

def build_prompt(obs_resp: Dict[str, Any]) -> str:
    o = obs_resp.get("observation", obs_resp)
    parts = [
        f"Day {o.get('day', 0)}/{o.get('total_days', 30)} — {o.get('city', 'Delhi')} | Task: {o.get('task_id', 'easy')}",
        f"PM2.5 baseline (today): {o.get('pm25_today', 0)} ug/m3 | Post-intervention (last step): {o.get('pm25_post_intervention', 0)}",
        f"Forecast risk next 2d: {o.get('forecast_risk_next_2d', [])}",
        f"Wind: {o.get('wind_regime', 'normal')} ({o.get('wind_speed_kmh', 0)} km/h, dir={o.get('wind_direction_deg', 315)}°)",
        f"Stubble burning: {'YES' if o.get('stubble_burning_flag') else 'no'} | Weekend: {'YES' if o.get('is_weekend') else 'NO'}",
        f"Policy Fatigue Index: {o.get('policy_fatigue_index', 0.0)}",
    ]

    # GRAP info
    if o.get('grap_stage', 0) > 0 or o.get('grap_recommended', 0) > 0:
        parts.append(f"GRAP: Stage {o.get('grap_stage', 0)} (recommended: {o.get('grap_recommended', 0)}, days at stage: {o.get('grap_days_at_stage', 0)})")

    # Health & Sentiment Metrics
    if o.get('estimated_hospital_admissions', 0) > 0:
        parts.append(f"\n🏥 HEALTH: ~{o.get('estimated_hospital_admissions', 0):.0f} daily admissions | Schools: {o.get('school_closure_status', 'open')} | Hospital stress: {o.get('hospital_capacity_stress', 'normal')}")
        parts.append(f"   Excess mortality risk: {o.get('excess_mortality_risk_per_100k', 0):.3f}/100k | DALYs today: {o.get('daly_burden_today', 0):.3f}")

    # Sentiment
    if o.get('public_sentiment', 0.55) != 0.55:
        parts.append(f"\n📊 PUBLIC SENTIMENT: {o.get('public_sentiment', 0.55):.2f} (0=outraged, 1=supportive)")
        if o.get('media_headlines'):
            for h in o.get('media_headlines', [])[:2]:
                parts.append(f"   {h}")
        if o.get('forced_reversal'):
            parts.append("   ⚠️ FORCED POLICY REVERSAL — public backlash!")

    # Budget
    if o.get('budget_remaining', -1) >= 0:
        parts.append(f"\n💰 Budget remaining: {o.get('budget_remaining', 0):.1f}")

    # Multi-city summary (Crucial for Expert/Crisis coordination)
    cities_data = o.get('cities_data', {})
    if cities_data:
        parts.append("\n🏙️ REGIONAL COMMAND STATUS:")
        for city, cd in cities_data.items():
            pm = cd.get('pm25_today', 0)
            post = cd.get('pm25_post', pm)
            sent = cd.get('public_sentiment', '')
            sent_str = f" | Sentiment: {sent:.2f}" if isinstance(sent, float) else ""
            grap = cd.get('grap_stage', '')
            grap_str = f" | GRAP: {grap}" if grap != '' else ""
            parts.append(f"   {city.title()}: PM2.5 {pm}→{post}{grap_str}{sent_str}")

    parts.append("\nRespond with a single JSON action object. No markdown.")
    return "\n".join(parts)

def get_action_from_llm(prompt: str) -> Dict[str, Any]:
    for attempt in range(2):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=250,
            )
            content = (completion.choices[0].message.content or "").strip()
            
            # Robust JSON cleaner (handles markdown blocks)
            if content.startswith("```"):
                content = content.replace("```json", "").replace("```", "").strip()
            
            # Find the first { and last } to avoid trailing text
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            if start_idx != -1 and end_idx != -1:
                content = content[start_idx:end_idx+1]
                
            return json.loads(content)
        except Exception:
            continue
    return {"action_type": "no_action", "level": 0}

# ── Execution Loop ───────────────────────────────────────────────────────

def run_episode(task_id: str) -> None:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    
    rewards: List[float] = []
    steps_taken = 0
    final_score = 0.0
    success = False
    
    try:
        obs = env_reset(task_id)
        
        for step in range(1, MAX_STEPS + 1):
            prompt = build_prompt(obs)
            action = get_action_from_llm(prompt)
            
            error = None
            try:
                obs = env_step(action)
            except Exception as e:
                error = str(e)
            
            reward = obs.get("reward", 0.0)
            done = obs.get("done", False)
            
            rewards.append(float(reward))
            steps_taken = step
            
            log_step(step=step, action=json.dumps(action), reward=reward, done=done, error=error)
            
            if done or error:
                break
        
        # Scoring Mandate: Return score strictly in (0, 1) to satisfy validator
        EPS = 0.005
        MAX_POSSIBLE_SCORE = 100.0  # Heuristic for a high-performing run
        
        meta = obs.get("metadata", {})
        grader_scores = meta.get("grader_scores", {})
        
        if task_id in grader_scores:
            # Use grader-provided score (expected in [0,1]) but map it
            # linearly into (EPS, 1-EPS) to avoid exact 0.0/1.0 values.
            raw = float(grader_scores[task_id])
            final_score = raw * (1.0 - 2.0 * EPS) + EPS
        else:
            # Fallback: map cumulative episode reward into (0,1) using a
            # smooth tanh mapping. This preserves ordering and gives an RL-
            # meaningful monotonic score rather than an arbitrary clamp.
            import math
            total = sum(rewards)
            scale = MAX_POSSIBLE_SCORE / 2.0 if MAX_POSSIBLE_SCORE > 0 else 1.0
            z = total / scale
            norm = 0.5 * (1.0 + math.tanh(z))
            final_score = norm * (1.0 - 2.0 * EPS) + EPS
            
        success = final_score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode failed: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

def main() -> None:
    # If explicit task requested, run that. Otherwise run tiers for evaluation.
    is_explicit = ("OPENENV_TASK" in os.environ) or ("AQI_TASK" in os.environ)
    tasks = [TASK_NAME] if is_explicit else ["easy", "medium", "hard", "expert", "crisis"]
    
    for t in tasks:
        run_episode(t)

if __name__ == "__main__":
    main()
