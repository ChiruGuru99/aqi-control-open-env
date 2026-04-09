#!/usr/bin/env python3
"""
Local variant of the OpenEnv submission validator.
Runs feasible checks locally:
 - optional HF Space ping to /reset (if ping_url provided)
 - optional Docker build (if `docker` present)
 - `python -m openenv.cli validate`
 - run `test_grader_scores.py`

Usage:
  python scripts/validate_submission_local.py [--ping-url <url>] [--run-inference]

This script is intentionally permissive: missing Docker or missing HF URL
are reported but do not abort the whole run by default.
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import subprocess
import sys
import textwrap
from typing import Optional

import requests


DOCKER_BUILD_TIMEOUT = 600


def now():
    return datetime.datetime.utcnow().strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}")


def run_ping(ping_url: str) -> bool:
    url = ping_url.rstrip("/") + "/reset"
    log(f"Pinging HF Space {url} ...")
    try:
        r = requests.post(url, json={}, timeout=30)
        log(f"  HTTP {r.status_code}")
        return r.status_code == 200
    except Exception as e:
        log(f"  Ping failed: {e}")
        return False


def run_docker_build(repo_dir: str) -> bool:
    docker_path = shutil.which("docker")
    if not docker_path:
        log("Docker not found; skipping Docker build step")
        return False

    # Find Dockerfile
    df_root = os.path.join(repo_dir, "Dockerfile")
    df_server = os.path.join(repo_dir, "server", "Dockerfile")
    if os.path.exists(df_root):
        context = repo_dir
    elif os.path.exists(df_server):
        context = os.path.join(repo_dir, "server")
    else:
        log("No Dockerfile found in repo root or server/; skipping Docker build")
        return False

    log(f"Running `docker build {context}` (timeout {DOCKER_BUILD_TIMEOUT}s) ...")
    try:
        p = subprocess.run(
            ["docker", "build", context],
            capture_output=True,
            text=True,
            cwd=repo_dir,
            timeout=DOCKER_BUILD_TIMEOUT,
        )
        if p.returncode == 0:
            log("Docker build succeeded")
            return True
        else:
            log("Docker build failed. Last output:\n" + p.stdout[-2000:])
            return False
    except subprocess.TimeoutExpired:
        log("Docker build timed out")
        return False
    except Exception as e:
        log(f"Docker build error: {e}")
        return False


def run_openenv_validate(repo_dir: str) -> bool:
    log("Running `openenv validate` via python -m openenv.cli validate ...")
    try:
        cmd = [sys.executable, "-m", "openenv.cli", "validate"]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_dir)
        log("openenv validate RC: %d" % p.returncode)
        if p.stdout:
            log("openenv stdout:\n" + p.stdout.strip())
        if p.stderr:
            log("openenv stderr:\n" + p.stderr.strip())
        return p.returncode == 0
    except Exception as e:
        log(f"openenv validate error: {e}")
        return False


def run_test_grader_scores(repo_dir: str) -> bool:
    log("Running test_grader_scores.py ...")
    try:
        cmd = [sys.executable, "test_grader_scores.py"]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_dir)
        log(f"test_grader_scores.py RC: {p.returncode}")
        if p.stdout:
            print(p.stdout)
        if p.stderr:
            print(p.stderr, file=sys.stderr)
        return p.returncode == 0
    except Exception as e:
        log(f"test_grader_scores.py error: {e}")
        return False


def run_inference(repo_dir: str) -> bool:
    log("Running inference.py (dry-run). This may attempt network calls to LLM APIs and ENV_URL")
    try:
        env = os.environ.copy()
        # Protect real credentials: default to empty token so inference falls back
        env.setdefault("HF_TOKEN", "")
        # If no ENV_URL specified, point to localhost
        env.setdefault("ENV_URL", "http://localhost:8000")
        p = subprocess.run([sys.executable, "inference.py"], capture_output=True, text=True, cwd=repo_dir, env=env, timeout=600)
        log(f"inference.py RC: {p.returncode}")
        if p.stdout:
            print(p.stdout)
        if p.stderr:
            print(p.stderr, file=sys.stderr)
        return p.returncode == 0
    except subprocess.TimeoutExpired:
        log("inference.py timed out")
        return False
    except Exception as e:
        log(f"inference.py error: {e}")
        return False


def main(argv: Optional[list[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="Local OpenEnv submission validator")
    parser.add_argument("--ping-url", help="HF Space URL to ping (optional)")
    parser.add_argument("--run-inference", action="store_true", help="Run inference.py (may call LLMs)")
    parser.add_argument("repo_dir", nargs="?", default=".", help="Path to repo (default: current directory)")
    args = parser.parse_args(argv)

    repo_dir = os.path.abspath(args.repo_dir)
    log(f"Repo: {repo_dir}")

    overall_ok = True

    # Step 1: optional HF ping
    if args.ping_url:
        ok = run_ping(args.ping_url)
        if ok:
            log("HF Space ping OK")
        else:
            log("HF Space ping FAILED")
            overall_ok = False
    else:
        log("No --ping-url provided; skipping HF Space ping")

    # Step 2: Docker build (optional)
    docker_ok = run_docker_build(repo_dir)
    if not docker_ok:
        log("Docker check skipped or failed (not fatal for local run)")

    # Step 3: openenv validate (required)
    ok = run_openenv_validate(repo_dir)
    if not ok:
        log("openenv validate FAILED")
        overall_ok = False
    else:
        log("openenv validate PASSED")

    # Step 4: run grader diagnostic
    ok = run_test_grader_scores(repo_dir)
    if not ok:
        log("test_grader_scores.py FAILED")
        overall_ok = False
    else:
        log("test_grader_scores.py PASSED")

    # Optional: run inference
    if args.run_inference:
        inf_ok = run_inference(repo_dir)
        if not inf_ok:
            log("inference run had issues (check output) ")
            overall_ok = False
        else:
            log("inference.py completed")

    log("Validation complete. Overall status: %s" % ("OK" if overall_ok else "FAIL"))
    return 0 if overall_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
