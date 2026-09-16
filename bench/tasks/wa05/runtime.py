"""WA-05 task-local RoboCasa runtime composition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bench.runtime import TaskRuntimeAdapter, default_close
from bench.tasks.runtime_support import (
    manual_station_navigation,
    standard_plan,
    standard_scene,
)
from bench.tasks.wa05.config import (
    load_task_config,
    resolve_action_config,
    resolve_task_options,
)
from bench.tasks.wa05.domain.episode import resolve_episode_plan, resolve_scene_config
from bench.tasks.wa05.domain.geometry import (
    resolve_cup_region_constraints,
    resolve_cup_sampling_regions,
)
from bench.tasks.wa05.evaluation import (
    REPORT_STAGE_PROFILE_ID,
    TASK_EVALUATOR_METADATA_KEY,
    WA05Evaluator,
    diagnostics_for_runtime,
    report_stages_for_runtime,
    success_for_runtime,
)
from bench.tasks.wa05.execution.navigation import build_navigation_config
from bench.tasks.wa05.expert.operations import build_operation_handlers
from bench.tasks.wa05.initialization import (
    apply_robot_base_initial_state,
    apply_sampled_initial_object_regions,
)
from bench.tasks.wa05.prompts import resolve_prompt_spec


def _environment(
    config: Any,
    scene: Any,
    plan: Any,
    resources: Mapping[str, Any],
) -> Mapping[str, Any]:
    initial_objects = dict(config.initial_state.objects)
    initial_objects.pop("enabled", None)
    regions = resolve_cup_sampling_regions(dict(initial_objects.get("cups") or {}))
    constraints = resolve_cup_region_constraints(
        dict(initial_objects.get("cup_region_constraints") or {})
    )
    gripper = dict(config.manipulation.get("gripper", {}))
    return {
        "sensor_faucet_layout_file": str(resources["sensor_faucet_layout"]),
        "answer_button_positions_file": str(resources["answer_button_positions"]),
        "button_pressed_threshold_m": float(
            config.semantics.get("button_pressed_threshold_m", 0.004)
        ),
        "button_released_threshold_m": float(
            config.semantics.get("button_released_threshold_m", 0.001)
        ),
        "button_release_dwell_steps": int(
            config.semantics.get("button_release_dwell_steps", 3)
        ),
        "service_threshold": float(
            config.semantics.get("service_threshold", 0.12)
        ),
        "cup_upright_threshold_deg": float(
            config.semantics.get("cup_upright_threshold_deg", 15.0)
        ),
        "cup_placement_seed": int(plan.seed),
        "cup_sampling_regions": {
            key: {
                "region_center_xy": list(value.center_xy),
                "region_size_xy": list(value.size_xy),
                "yaw_rad": float(value.yaw_rad),
            }
            for key, value in regions.items()
        },
        "cup_region_constraints": {
            "sink_exterior_local_x_max": float(constraints.sink_exterior_local_x_max),
            "minimum_sink_clearance": float(constraints.minimum_sink_clearance),
            "minimum_inter_region_gap": float(constraints.minimum_inter_region_gap),
            "sampling_inset_xy": list(constraints.sampling_inset_xy),
        },
        "gripper_force_limit": float(gripper.get("force_limit", 20.0)),
        "gripper_pad_friction": tuple(
            float(value)
            for value in gripper.get("pad_friction", (2.0, 0.05, 0.0001))
        ),
    }


def _initialize(descriptor: Any, env: Any, observation: Mapping[str, Any]):
    initial_objects = dict(descriptor.config.initial_state.objects)
    enabled = bool(initial_objects.pop("enabled", True))
    updated, objects = apply_sampled_initial_object_regions(
        env,
        settings=initial_objects,
        enabled=enabled,
    )
    env.configure_episode(descriptor.episode_plan)
    env.configure_canonical_physics()
    updated, base = apply_robot_base_initial_state(
        env, updated, config=descriptor.config.initial_state.robot_base,
        nav_config=descriptor.navigation, episode=descriptor.episode_plan,
        initial_objects=initial_objects, seed=descriptor.episode_plan.seed,
    )
    evaluator = WA05Evaluator(
        cup_id=env.episode_plan.cup_id,
        service_slot=env.episode_plan.service_slot,
    )
    evaluator.reset(env.evaluation_snapshot())
    env._wa05_evaluator = evaluator
    return (
        updated,
        {"objects": objects, "base": base},
        {TASK_EVALUATOR_METADATA_KEY: evaluator},
        None,
    )


_ADAPTER = TaskRuntimeAdapter(
    task_id="WA-05",
    load_task_config=load_task_config,
    resolve_task_options=resolve_task_options,
    scene=standard_scene(resolve_scene_config),
    plan=standard_plan(resolve_episode_plan),
    actions=lambda config, plan, resources: resolve_action_config(config=config),
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
)
TASK_RUNTIME_API = _ADAPTER.api()
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
