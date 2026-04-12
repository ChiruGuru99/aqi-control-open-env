import asyncio
import os
import sys

# Ensure current directory is in path for imports
sys.path.append(os.getcwd())

from server.aqi_environment import AQIControlEnvironment
from models import AQIAction

async def test_graders():
    env = AQIControlEnvironment()
    tasks = ["easy", "medium", "hard", "expert", "crisis"]
    
    print(f"=== Testing Grader Scores (Target: strictly 0.0 < score < 1.0) ===")
    
    all_passed = True
    for task_id in tasks:
        try:
            # Step 1: Reset
            obs = await env.reset(task_id=task_id)
            
            # Step 2: Take one step (no action)
            action = AQIAction(action_type="no_action", level=0, city=obs.city)
            result = await env.step(action)
            
            # Step 3: Check metadata grader scores
            meta = result.observation.metadata
            grader_scores = meta.get("grader_scores", {})
            score = grader_scores.get(task_id)
            
            if score is None:
                print(f"[FAIL] Task '{task_id}': No score found in metadata")
                all_passed = False
            elif not (0.0 < score < 1.0):
                print(f"[FAIL] Task '{task_id}': Score {score} out of range (0, 1)")
                all_passed = False
            else:
                print(f"[PASS] Task '{task_id}': Score = {score:.4f}")
                
        except Exception as e:
            print(f"[ERROR] Task '{task_id}': {e}")
            all_passed = False
            
    if all_passed:
        print("\n=== ALL GRADERS PASSED SMOKE TEST ===")
        sys.exit(0)
    else:
        print("\n=== SOME GRADERS FAILED ===")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_graders())
