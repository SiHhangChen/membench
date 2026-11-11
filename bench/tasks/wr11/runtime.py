"""WR-11 composition root for the same recording and policy runtime."""

import yaml

from bench.runtime import TaskRuntimeAdapter, default_close
from bench.tasks.runtime_support import (
    keyword_config_actions,
    standard_scene,
    station_navigation,
    variant_prompt,
)

from .config import (
    load_task_config,
    resolve_action_config,
    resolve_configured_episode_plan,
    resolve_task_options,
)
from .domain.contracts import TASK_REVISION
from .domain.episode import resolve_scene_config
from .evaluation import (
    REPORT_STAGE_PROFILE_ID,
    diagnostics_for_runtime,
    report_stages_for_runtime,
    success_for_runtime,
)
from .execution.navigation import build_navigation_config
from .expert.operations import build_operation_handlers
from .initialization import initialize_episode_state
from .prompts import resolve_prompt_spec


def _environment(config, scene, plan, resources):
    return {
        "update_fxtr_cfg_dict": yaml.safe_load(resources["fixture_overrides"].read_text()),
        **{
            key: getattr(plan, key)
            for key in ("sample_packaging", "packaging_by_item", "offset_xy_by_item", "yaw_by_item")
        },
        **dict(config.semantics),
    }


def _initialize(descriptor, env, observation):
    observation, setup = initialize_episode_state(descriptor, env, observation)
    return observation, setup, {}, {"delivery_pose": env.delivery_pose().tolist()}


_ADAPTER = TaskRuntimeAdapter(
    task_id="WR-11",
    load_task_config=load_task_config,
    resolve_task_options=resolve_task_options,
    scene=standard_scene(resolve_scene_config),
    plan=lambda config, episode, options: resolve_configured_episode_plan(config, episode.seed),
    actions=keyword_config_actions(resolve_action_config),
    navigation=station_navigation(build_navigation_config),
    prompt=variant_prompt(resolve_prompt_spec),
    environment=_environment,
    initialize=_initialize,
    diagnostics=diagnostics_for_runtime,
    success=success_for_runtime,
    close=default_close,
    report_stage_profile_id=REPORT_STAGE_PROFILE_ID,
    report_stages=report_stages_for_runtime,
    release_facts=lambda *args: {"task_revision": TASK_REVISION},
    expert_runtime=build_operation_handlers,
)
TASK_RUNTIME_API = _ADAPTER.api()
build_descriptor = _ADAPTER.build_descriptor
initialize_runtime = _ADAPTER.initialize_runtime
check_success = _ADAPTER.check_success
diagnostics = _ADAPTER.diagnostics
close_runtime = _ADAPTER.close_runtime
