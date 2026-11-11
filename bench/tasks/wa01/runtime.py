"""WA-01 task-local RoboCasa runtime composition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from bench.runtime import TaskRuntimeAdapter, default_close
from bench.tasks.runtime_support import manual_station_navigation, standard_plan, standard_scene
from bench.tasks.wa01.config import (
    load_pose_defaults,
    load_task_config,
    resolve_action_config,
    resolve_task_options,
)
from bench.tasks.wa01.domain.episode import resolve_episode_plan, resolve_scene_config
from bench.tasks.wa01.domain.geometry import dining_region_config_summary
from bench.tasks.wa01.execution.navigation import (
    build_navigation_config,
    resolve_station_randomization,
)
from bench.tasks.wa01.evaluation import (
    REPORT_STAGE_PROFILE_ID,
    TASK_EVALUATOR_METADATA_KEY,
    WA01Evaluator,
    diagnostics_for_runtime,
    report_stages_for_runtime,
    success_for_runtime,
)
from bench.tasks.wa01.evaluation_prefix import build_prefix, validate_prefix
from bench.tasks.wa01.subtask_evaluation import build_subtask_scheduler
from bench.tasks.wa01.expert.operations import build_operation_handlers
from bench.tasks.wa01.initialization import (
    apply_fixed_initial_object_regions,
    apply_robot_base_initial_state,
)
from bench.tasks.wa01.prompts import resolve_prompt_spec


def _actions(config: Any, plan: Any, resources: Mapping[str, Any]) -> Any:
    del resources
    return resolve_action_config(
        episode=plan,
        pose_defaults=load_pose_defaults(config),
        operations=config.operations,
    )


def _environment(
    config: Any,
    scene: Any,
    plan: Any,
    resources: Mapping[str, Any],
) -> Mapping[str, Any]:
    del config, scene, resources
    return {
        "coffee_material_by_id": dict(plan.coffee_material_by_id),
        "sugar_cube_material": str(plan.sugar_cube_material),
    }


def _initialize(descriptor: Any, env: Any, observation: Mapping[str, Any]):
    objects, base = {}, {}
    objects_obs, objects = apply_fixed_initial_object_regions(
        env,
        seed=descriptor.episode_plan.seed,
        **dict(descriptor.config.initial_state.objects),
    )
    env.configure_dining_placement_regions(
        region_config=dining_region_config_summary(descriptor.actions.dining_region_config),
        regions_by_slot=descriptor.actions.dining_regions_by_slot,
        sampled_targets_by_slot=descriptor.actions.sampled_dining_targets_by_slot,
        planned_slot_assignment=descriptor.episode_plan.final_slot_assignment,
    )
    updated = env._get_observations(force_update=True)
    updated, base = apply_robot_base_initial_state(
        env, updated,
        config=descriptor.config.initial_state.robot_base,
        nav_config=descriptor.navigation,
        seed=descriptor.episode_plan.seed,
        randomization=resolve_station_randomization(descriptor.config.navigation),
        settings=descriptor.config.navigation,
    )
    evaluator = WA01Evaluator()
    evaluator.reset(env.evaluation_snapshot())
    env._wa01_evaluator = evaluator
    return (
        updated,
        {"objects": objects, "base": base},
        {TASK_EVALUATOR_METADATA_KEY: evaluator},
        None,
    )


_ADAPTER = TaskRuntimeAdapter(
    task_id="WA-01",
    load_task_config=load_task_config,
    resolve_task_options=resolve_task_options,
    scene=standard_scene(resolve_scene_config),
    plan=standard_plan(resolve_episode_plan),
    actions=_actions,
    navigation=manual_station_navigation(build_navigation_config),
    prompt=lambda plan, variant: resolve_prompt_spec(variant=variant),
    environment=_environment,
    initialize=_initialize,
    diagnostics=diagnostics_for_runtime,
    success=success_for_runtime,
    close=default_close,
    report_stage_profile_id=REPORT_STAGE_PROFILE_ID,
    report_stages=report_stages_for_runtime,
    expert_runtime=build_operation_handlers,
    subtask_scheduler=build_subtask_scheduler,
)
TASK_RUNTIME_API = replace(
    _ADAPTER.api(), build_evaluation_prefix=build_prefix,
    validate_evaluation_prefix=validate_prefix,
)
build_descriptor = _ADAPTER.build_descriptor
initialize_runtime = _ADAPTER.initialize_runtime
check_success = _ADAPTER.check_success
diagnostics = _ADAPTER.diagnostics
close_runtime = _ADAPTER.close_runtime
__all__ = [
    "TASK_RUNTIME_API",
    "build_descriptor",
    "check_success",
    "close_runtime",
    "diagnostics",
    "initialize_runtime",
]
