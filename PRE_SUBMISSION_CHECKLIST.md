Pre-Submission Checklist
========================

This checklist mirrors the automated validator used by the benchmark and helps avoid common failures such as:
- "Not enough tasks with graders"
- "One or more task scores are out of range"

Required environment variables (for inference runs)
--------------------------------------------------
- `API_BASE_URL` — LLM API endpoint (example: `https://api.openai.com/v1`)
- `MODEL_NAME` — model identifier used by the inference script (example: `gpt-4o-mini`)
- `HF_TOKEN` — your Hugging Face / OpenAI API key

Files & manifest
----------------
- Ensure `inference.py` is present in the project root and uses the above env vars for all LLM calls.
- Ensure `openenv.yaml` contains at least 3 tasks and that each task has an explicit `grader` field.
- Grader paths should be package-qualified (importable after `pip install -e .`), e.g.: `aqi_control_env.env.graders:grade_easy`.

Grader requirements
-------------------
- Graders must be deterministic and reproducible given the same episode trajectory.
- Each task's grader must return a float strictly within the open interval (0.0, 1.0). Avoid returning `0.0` or `1.0` exactly. A common safe-floor is `max(0.001, min(score, 0.999))`.

Structured stdout logging (strict format)
----------------------------------------
The `inference.py` script MUST emit structured log lines on stdout following this exact marker ordering and field names:
- `[START] task=<task_id> env=<env_name> model=<model_name>`
- Repeated lines: `[STEP] step=<n> action=<json_action_single_line> reward=<reward> done=<true|false> error=<message|null>`
- Final line: `[END] success=<true|false> steps=<n> score=<0.000-0.999> rewards=<comma_separated_rewards>`

Example lines (single-line JSON for `action`):

```
[START] task=easy env=aqi_control_env model=gpt-4o-mini
[STEP] step=1 action={"action_type":"no_action","level":0,"city":"Delhi"} reward=0.00 done=false error=null
[END] success=false steps=30 score=0.203 rewards=0.00,0.01,0.02
```

Automated local validation
--------------------------
Install the package and the inference extras:

```powershell
.venv\Scripts\python.exe -m pip install -e .[inference]
```

Run the grader diagnostic (ensures 3 tasks with graders and scores in (0,1)):

```powershell
.venv\Scripts\python.exe test_grader_scores.py
```

Run the bundled validator (optional HF Space ping, Docker build if present, `openenv validate`, graders):

```powershell
./validate-submission.sh <PING_URL>
```

Docker check
------------
The repository includes a `Dockerfile` and the validator will attempt `docker build .` if `docker` is available on the machine. Locally, you can run:

```powershell
docker build -t aqi_control_env .
```

Troubleshooting
---------------
- "Not enough tasks with graders": verify every task entry in `openenv.yaml` has a `grader` field and that the grader path is importable after packaging (package-qualified path).
- "One or more task scores are out of range": ensure each grader clamps scores into (0.0, 1.0) and is deterministic.

Continuous integration
----------------------
The repository includes a GitHub Action `.github/workflows/validate_openenv.yml` that runs these checks on every push.
