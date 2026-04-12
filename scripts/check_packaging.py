"""Simple packaging/import smoke test used by CI.

Installs the local package in editable mode earlier in CI; this script verifies
that the grader module can be imported and that `openenv.yaml` is present in
the installed package folder.

Usage (CI): run after `pip install -e .` step.
"""

from __future__ import annotations

import importlib
import pathlib
import sys


def main() -> int:
    try:
        mod = importlib.import_module("aqi_control_env.env.graders")
    except Exception as e:
        print("ERROR: could not import grader module 'aqi_control_env.env.graders'", file=sys.stderr)
        print(str(e), file=sys.stderr)
        return 2

    ok = "grade_easy" in dir(mod) and "grade_medium" in dir(mod) and "grade_hard" in dir(mod)
    print("Grader functions present:", ok)

    # Check openenv.yaml presence in the installed package
    try:
        pkg = importlib.import_module("aqi_control_env")
        pkg_path = pathlib.Path(pkg.__file__).parent
        y = pkg_path / "openenv.yaml"
        print("openenv.yaml exists:", y.exists())
        if not y.exists():
            print("ERROR: openenv.yaml not found inside installed package at:", y, file=sys.stderr)
            return 3
    except Exception as e:
        print("ERROR: failed to locate installed package 'aqi_control_env'", file=sys.stderr)
        print(str(e), file=sys.stderr)
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
