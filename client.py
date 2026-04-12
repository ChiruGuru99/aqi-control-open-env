# Copyright (c) 2025. AQI Control Environment for OpenEnv.
"""AQI Control Environment Client."""

from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from .models import AQIAction, AQIObservation


class AQIControlEnv(EnvClient[AQIAction, AQIObservation, State]):
    """
    Client for the AQI Control Environment.

    Example:
        >>> with AQIControlEnv(base_url="http://localhost:8000").sync() as client:
        ...     result = client.reset()
        ...     result = client.step(AQIAction(action_type="restrict_traffic", level=2))
    """

    def _step_payload(self, action: AQIAction) -> Dict:
        return action.model_dump(exclude_none=True, exclude_defaults=False)

    def _parse_result(self, payload: Dict) -> StepResult[AQIObservation]:
        obs_data = payload.get("observation", {})
        observation = AQIObservation(**obs_data)
        observation.done = payload.get("done", False)
        observation.reward = payload.get("reward")
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
