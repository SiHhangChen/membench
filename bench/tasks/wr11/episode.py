"""Simulator-free WR-11 episode resolution."""

from bench.tasks.contracts import EpisodeRequest
from bench.tasks.episode_support import resolve_canonical_episode

from .config import (
    DEFAULT_TASK_CONFIG_FILE,
    load_task_config,
    resolve_configured_episode_plan,
    resolve_task_options,
)
from .domain.episode import resolve_scene_config
from .prompts import resolve_prompt_spec


def resolve_episode(request=None, **kwargs):
    request = request or EpisodeRequest(task_id="WR-11", **kwargs)
    return resolve_canonical_episode(
        request,
        task_id="WR-11",
        task_version="3.2.0",
        default_task_config_file=DEFAULT_TASK_CONFIG_FILE,
        load_task_config=load_task_config,
        resolve_task_options=resolve_task_options,
        resolve_scene=lambda config, req: resolve_scene_config(
            seed=req.seed, env_split=req.env_split, **dict(config.scene)
        ),
        resolve_plan=lambda config, req: resolve_configured_episode_plan(config, req.seed),
        resolve_prompt=lambda plan, req: resolve_prompt_spec(variant=req.variant),
    )
