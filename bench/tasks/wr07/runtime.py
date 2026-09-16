"""WR-07 task-local runtime composition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bench.runtime import TaskRuntimeAdapter, default_close
from bench.tasks.runtime_support import keyword_config_actions, standard_plan, standard_scene, station_navigation, variant_prompt
from bench.tasks.wr07.config import load_task_config, resolve_action_config, resolve_task_options
from bench.tasks.wr07.domain.contracts import TASK_REVISION
from bench.tasks.wr07.domain.episode import resolve_episode_plan, resolve_scene_config
from bench.tasks.wr07.evaluation import (
    REPORT_STAGE_PROFILE_ID, TASK_EVALUATOR_METADATA_KEY, WR07Evaluator,
    diagnostics_for_runtime, report_stages_for_runtime, success_for_runtime,
)
from bench.tasks.wr07.execution.geometry import resolve_episode_targets
from bench.tasks.wr07.execution.navigation import build_navigation_config
from bench.tasks.wr07.expert.operations import build_operation_handlers
from bench.tasks.wr07.initialization import initialize_episode_state
from bench.tasks.wr07.prompts import resolve_prompt_spec

def _environment(config: Any, scene: Any, plan: Any, resources: Mapping[str, Any]) -> Mapping[str, Any]:
    del scene, resources
    return {
        "source_plate_color_by_slot": dict(plan.source_plate_color_by_slot),
        "target_plate_color_by_slot": dict(plan.target_plate_color_by_slot),
        "source_slot_by_item": dict(plan.source_slot_by_item),
        "source_color_by_item": dict(plan.source_color_by_item),
        "target_slot_by_color": dict(plan.target_slot_by_color),
        "source_item_offset_xy_by_item": dict(plan.source_item_offset_xy_by_item),
        "source_item_yaw_by_item": dict(plan.source_item_yaw_by_item),
        "target_item_offset_xy_by_item": dict(plan.target_item_offset_xy_by_item),
        "target_item_yaw_by_item": dict(plan.target_item_yaw_by_item),
        "carton_type": str(plan.carton_type),
        "carton_asset": str(plan.carton_asset),
        "carton_material": str(plan.carton_material),
        "item_sample_radius_m": float(plan.item_sample_radius_m),
        "plate_threshold": float(config.semantics.get("plate_threshold", 0.16)),
    }

def _initialize(descriptor: Any, env: Any, observation: Mapping[str, Any]):
    updated, setup = initialize_episode_state(
        env, dict(observation), episode=descriptor.episode_plan,
        robot_base=descriptor.config.initial_state.robot_base,
        objects=dict(descriptor.config.initial_state.objects),
    )
    targets = resolve_episode_targets(env, descriptor.episode_plan)
    evaluator = WR07Evaluator()
    evaluator.reset(env.evaluation_snapshot())
    env._wr07_evaluator = evaluator
    return updated, {"setup": setup}, {TASK_EVALUATOR_METADATA_KEY: evaluator}, targets

_ADAPTER = TaskRuntimeAdapter(
    task_id="WR-07",
    load_task_config=load_task_config,
    resolve_task_options=resolve_task_options,
    scene=standard_scene(resolve_scene_config),
    plan=standard_plan(resolve_episode_plan),
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
    release_facts=lambda config, episode, resources, options: {
        "task_revision": TASK_REVISION,
        "task_id": episode.task_id,
        "carton_type": str(episode.facts["carton_type"]),
        "carton_asset": str(episode.facts["carton_asset"]),
        "carton_material": str(episode.facts["carton_material"]),
        "carton_type_candidates": list(episode.facts["carton_type_candidates"]),
        "carton_material_candidates": list(episode.facts["carton_material_candidates"]),
        "carton_combination_count": int(
            len(episode.facts["carton_type_candidates"])
            * len(episode.facts["carton_material_candidates"])
        ),
    },
    expert_runtime=build_operation_handlers,
)
TASK_RUNTIME_API = _ADAPTER.api()
build_descriptor = _ADAPTER.build_descriptor
initialize_runtime = _ADAPTER.initialize_runtime
check_success = _ADAPTER.check_success
diagnostics = _ADAPTER.diagnostics
close_runtime = _ADAPTER.close_runtime

__all__ = ["TASK_RUNTIME_API", "build_descriptor", "check_success", "close_runtime", "diagnostics", "initialize_runtime"]
