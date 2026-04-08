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


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point: uv run --project . server"""
    import os, uvicorn
    p = int(os.environ.get("PORT", port))
    uvicorn.run(app, host=host, port=p)


if __name__ == '__main__':
    main()
