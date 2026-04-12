# Copyright (c) 2025. AQI Control Environment for OpenEnv.
"""
FastAPI application for the AQI Control Environment.
Usage: uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required: pip install openenv-core[core]") from e

try:
    from ..models import AQIAction, AQIObservation
    from .aqi_environment import AQIControlEnvironment
except (ImportError, ModuleNotFoundError):
    from models import AQIAction, AQIObservation
    from server.aqi_environment import AQIControlEnvironment

app = create_app(
    AQIControlEnvironment,
    AQIAction,
    AQIObservation,
    env_name="aqi_control_env",
    max_concurrent_envs=1,
)


def _startup_diagnostic() -> None:
    """Log package installation details to stdout so remote logs show what was installed.

    This prints the `aqi_control_env` package __file__, whether
    `env/graders.py` and `openenv.yaml` exist, and a short head of the
    grader file when available. The output helps hackathon validators
    diagnose missing grader callables in deployed images.
    """
    try:
        import importlib
        import os
        import traceback

        try:
            pkg = importlib.import_module("aqi_control_env")
        except Exception as e:
            print(f"DIAGNOSTIC: failed to import aqi_control_env: {e}")
            return

        pkg_file = getattr(pkg, "__file__", None)
        pkg_dir = os.path.dirname(pkg_file) if pkg_file else None
        print(f"DIAGNOSTIC: aqi_control_env __file__={pkg_file}")

        if pkg_dir:
            env_graders = os.path.join(pkg_dir, "env", "graders.py")
            openenv_yaml = os.path.join(pkg_dir, "openenv.yaml")
            print(f"DIAGNOSTIC: graders.py path={env_graders} exists={os.path.exists(env_graders)}")
            print(f"DIAGNOSTIC: openenv.yaml path={openenv_yaml} exists={os.path.exists(openenv_yaml)}")

            if os.path.exists(env_graders):
                try:
                    with open(env_graders, "r", encoding="utf-8") as fh:
                        head = "".join(fh.readlines()[:200])
                    print(f"DIAGNOSTIC: begin graders.py head:\n{head}\n--- end graders.py head ---")
                except Exception as e:
                    print(f"DIAGNOSTIC: error reading {env_graders}: {e}")
        else:
            print("DIAGNOSTIC: package directory not found for aqi_control_env")
    except Exception as e:
        print("DIAGNOSTIC: unexpected error in startup diagnostic:", e)
        traceback.print_exc()


app.add_event_handler("startup", _startup_diagnostic)


from fastapi.responses import JSONResponse


@app.get("/.well-known/aqi-control-diagnostic")
def aqi_control_diagnostic():
    """Return a compact JSON listing of installed package files and grader head.

    This endpoint is optional and meant for debugging; it intentionally
    returns only a short excerpt of the graders file.
    """
    try:
        import importlib
        import pathlib
        pkg = importlib.import_module("aqi_control_env")
        pkg_path = pathlib.Path(getattr(pkg, "__file__")).parent

        files = [str(p.relative_to(pkg_path)) for p in pkg_path.rglob("**/*") if p.is_file()]
        graders_head = None
        gpath = pkg_path / "env" / "graders.py"
        if gpath.exists():
            try:
                graders_head = "".join(gpath.read_text(encoding="utf-8").splitlines(True)[:200])
            except Exception:
                graders_head = "<error reading file>"

        return JSONResponse({
            "pkg_file": getattr(pkg, "__file__", None),
            "files": files,
            "graders_head": graders_head,
        })
    except Exception as e:
        import traceback

        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point: uv run --project . server"""
    import os, uvicorn
    p = int(os.environ.get("PORT", port))
    uvicorn.run(app, host=host, port=p)


if __name__ == '__main__':
    main()
