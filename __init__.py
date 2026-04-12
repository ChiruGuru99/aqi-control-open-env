# Copyright (c) 2025. AQI Control Environment for OpenEnv.
"""AQI Control Environment — City-level PM2.5 policy controller for Indian winters.

This module exposes a small public surface but avoids importing heavy
dependencies at module import time. Attributes are loaded lazily to keep
packaging checks (which import submodules) lightweight.
"""

__all__ = ["AQIAction", "AQIObservation", "AQIControlEnv"]


def __getattr__(name: str):
	"""Lazily import attributes to avoid heavy top-level imports.

	Importing `aqi_control_env.env.graders` should not trigger
	`openenv` or other heavy dependencies via the `client` module.
	"""
	if name == "AQIControlEnv":
		from .client import AQIControlEnv

		return AQIControlEnv
	if name in ("AQIAction", "AQIObservation"):
		from .models import AQIAction, AQIObservation

		return AQIAction if name == "AQIAction" else AQIObservation
	raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
	return __all__
