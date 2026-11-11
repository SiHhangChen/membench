"""Compatibility observation accessor for the WR-11 task package."""

from __future__ import annotations

from typing import Any


def get_observation(runtime: Any) -> dict[str, Any]:
    observation = getattr(runtime, "observation", None)
    if observation is not None:
        return dict(observation)
    env = getattr(runtime, "env", None)
    if env is not None and hasattr(env, "_get_observations"):
        return dict(env._get_observations(force_update=True))
    return {}


__all__ = ["get_observation"]
