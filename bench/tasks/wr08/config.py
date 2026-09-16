"""WR-08 YAML loading and action configuration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from bench.tasks.wr08.domain.contracts import (
    ActionConfig,
    InitialStateConfig,
    ITEM_IDS,
    RobotBaseConfig,
    RobotBaseMode,
    RunContext,
    SOURCE_SLOTS,
    TASK_ID,
    TaskConfig,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_TASK_CONFIG_FILE = PACKAGE_ROOT / "resources" / "task.yaml"
_RESOURCE_NAMES = {"planner_script", "stations"}


def load_task_config(path: str | Path = DEFAULT_TASK_CONFIG_FILE) -> TaskConfig:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"WR-08 config root must be a mapping: {config_path}")
    required = {
        "schema_version",
        "task",
        "scene",
        "episode",
        "initial_state",
        "sensors",
        "resources",
        "navigation",
        "manipulation",
        "operations",
        "semantics",
        "recording",
    }
    unknown = set(payload) - required
    if unknown:
        raise ValueError("unknown WR-08 config sections: " + ", ".join(sorted(unknown)))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("WR-08 config requires schema_version=1")
    task = dict(_mapping(payload, "task"))
    _reject_unknown(
        task, {"id", "display_name", "description", "success_conditions"}, "task fields"
    )
    scene = dict(_mapping(payload, "scene"))
    _reject_unknown(
        scene,
        {"env_name", "fixed_scene", "layout_id", "style_id", "scene_seed"},
        "scene fields",
    )
    if task.get("id") != TASK_ID:
        raise ValueError(f"WR-08 config must declare task.id={TASK_ID!r}")
    if scene.get("env_name") != "WR08MatchMilkCartonsToColoredSugarPlates":
        raise ValueError("WR-08 config has an invalid scene.env_name")
    initial = _mapping(payload, "initial_state")
    _reject_unknown(
        initial, {"robot_base", "source_regions", "objects"}, "initial_state fields"
    )
    base = _mapping(initial, "robot_base")
    _reject_unknown(base, {"mode", "xy", "yaw"}, "robot_base fields")
    source_regions = _mapping(initial, "source_regions")
    if set(source_regions) != set(SOURCE_SLOTS):
        raise ValueError(
            "WR-08 initial_state.source_regions must define three source slots"
        )
    regions = {slot: _region(source_regions[slot], slot) for slot in SOURCE_SLOTS}
    resources = dict(_mapping(payload, "resources"))
    _reject_unknown(resources, _RESOURCE_NAMES, "resource names")
    if set(resources) != _RESOURCE_NAMES:
        raise ValueError(
            "WR-08 resources must define exactly: " + ", ".join(sorted(_RESOURCE_NAMES))
        )
    for name, value in resources.items():
        resource = config_path.parent / str(value)
        if not resource.is_file():
            raise FileNotFoundError(f"missing WR-08 resource {name!r}: {resource}")
    _reject_unknown(
        _mapping(payload, "episode"),
        {
            "max_horizon",
            "color_count",
            "item_sample_radius_m",
            "color_pool",
            "carton_type_candidates",
            "carton_material_candidates",
        },
        "episode fields",
    )
    _reject_unknown(
        _mapping(payload, "sensors"),
        {
            "cameras",
            "wrist_rgb_camera",
            "width",
            "height",
            "depth",
            "has_renderer",
            "has_offscreen_renderer",
            "use_camera_observations",
        },
        "sensor fields",
    )
    _reject_unknown(
        _mapping(payload, "navigation"),
        {
            "kinematic_base_navigation",
            "base_action_frame",
            "base_translation_action_sign",
            "base_tolerance",
            "base_yaw_tolerance",
            "base_max_steps",
            "base_holding_drop_tolerance",
            "base_profile_v_max",
            "base_profile_a_max",
            "base_profile_a_dec",
            "base_profile_v_min",
            "base_profile_dock_v_max",
            "base_profile_dock_v_min",
            "base_profile_w_max",
            "base_profile_alpha_max",
            "base_profile_alpha_dec",
            "base_profile_w_min",
            "station_nav_use_docking",
            "use_station_base_poses",
            "manual_base_path_enabled",
            "use_station_astar_paths",
        },
        "navigation fields",
    )
    manipulation = _mapping(payload, "manipulation")
    _reject_unknown(
        manipulation,
        {"pick_place", "planner", "runtime", "static_scene"},
        "manipulation sections",
    )
    _reject_unknown(
        _mapping(manipulation, "pick_place"),
        {
            "orientation_mode",
            "pregrasp_height",
            "grasp_clearance",
            "lift_height",
            "place_height",
            "retreat_height",
        },
        "pick_place fields",
    )
    _reject_unknown(
        _mapping(manipulation, "planner"),
        {
            "conda_env",
            "subprocess_timeout_s",
            "planning_time",
            "time_step",
            "qpos_step",
            "rrt_range",
            "nearest_ik_samples",
            "nearest_ik_threshold",
            "nearest_ik_max_joint_delta",
            "preserve_ik_branch_stages",
        },
        "planner fields",
    )
    _reject_unknown(
        _mapping(manipulation, "runtime"),
        {
            "max_joint_step",
            "settle_steps",
            "position_threshold",
            "orientation_threshold_deg",
            "qpos_threshold",
            "open_steps",
            "close_steps",
            "release_steps",
            "release_settle_steps",
        },
        "runtime fields",
    )
    _reject_unknown(
        _mapping(manipulation, "static_scene"), {"enabled"}, "static_scene fields"
    )
    operations = _mapping(payload, "operations")
    _reject_unknown(
        operations,
        {
            "source_pick",
            "target_place",
            "return_arm_empty",
            "return_arm_holding",
            "handlers",
        },
        "operation sections",
    )
    _reject_unknown(
        _mapping(operations, "source_pick"),
        {
            "orientation_mode",
            "pregrasp_height",
            "pregrasp_settle_steps",
            "allow_pregrasp_near_miss",
            "pregrasp_near_miss_max_position_error",
            "lift_height",
            "post_close_hold_steps",
            "grasp_yaw_offsets_deg",
            "grasp_settle_steps",
            "lift_settle_steps",
        },
        "source_pick fields",
    )
    _reject_unknown(
        _mapping(operations, "target_place"),
        {
            "placement_reference",
            "placement_object_upright",
            "placement_yaw_offsets_deg",
            "place_height",
            "release_steps",
            "release_settle_steps",
            "allow_place_near_miss",
        },
        "target_place fields",
    )
    _reject_unknown(
        _mapping(operations, "return_arm_empty"),
        {"gripper_action"},
        "return_arm_empty fields",
    )
    _reject_unknown(
        _mapping(operations, "return_arm_holding"),
        {"gripper_action", "settle_steps", "qpos_threshold"},
        "return_arm_holding fields",
    )
    _reject_unknown(
        _mapping(operations, "handlers"),
        {"accessible_place_dry_run_only"},
        "handler fields",
    )
    _reject_unknown(
        _mapping(payload, "semantics"),
        {
            "plate_threshold",
            "physical_event_dwell_steps",
            "terminal_verification_steps",
        },
        "semantics fields",
    )
    _reject_unknown(
        _mapping(payload, "recording"),
        {"record_depth", "depth_dtype", "depth_scale", "action_schema"},
        "recording fields",
    )
    return TaskConfig(
        config_path,
        task,
        scene,
        dict(_mapping(payload, "episode")),
        InitialStateConfig(
            resolve_robot_base_config(**dict(base)),
            regions,
            dict(_mapping(initial, "objects")),
        ),
        dict(_mapping(payload, "sensors")),
        resources,
        dict(_mapping(payload, "navigation")),
        dict(_mapping(payload, "manipulation")),
        dict(_mapping(payload, "operations")),
        dict(_mapping(payload, "semantics")),
        dict(_mapping(payload, "recording")),
    )


def resolve_action_config(*, config: TaskConfig) -> ActionConfig:
    operations = config.operations
    source = dict(_section(operations, "source_pick"))
    return ActionConfig(
        source_pick_metadata={item: dict(source) for item in ITEM_IDS},
        target_place_metadata={
            slot: dict(_section(operations, "target_place"))
            for slot in ("target_left", "target_center", "target_right")
        },
        return_arm_empty=dict(_section(operations, "return_arm_empty")),
        return_arm_holding=dict(_section(operations, "return_arm_holding")),
        handler_settings=dict(_section(operations, "handlers")),
    )


def resolve_task_options(task_options: Mapping[str, Any] | None) -> dict[str, Any]:
    if task_options:
        raise ValueError(
            "WR-08 v3 has no task options; all episode randomness is seed-owned"
        )
    return {}


def resolve_run_context(
    *,
    split: str,
    variant: str,
    execute: bool,
    extra_metadata: Mapping[str, Any] | None = None,
) -> RunContext:
    if str(variant) not in {"memory", "oracle_reveal"}:
        raise ValueError("WR-08 variant must be 'memory' or 'oracle_reveal'")
    return RunContext(
        str(split), str(variant), bool(execute), dict(extra_metadata or {})
    )


def resolve_robot_base_config(
    *,
    mode: str | RobotBaseMode = RobotBaseMode.PRESERVE_RESET_POSE,
    xy: Sequence[float] | None = None,
    yaw: float | None = None,
) -> RobotBaseConfig:
    raw = mode.value if isinstance(mode, RobotBaseMode) else str(mode).strip().lower()
    try:
        resolved = RobotBaseMode(raw)
    except ValueError as exc:
        raise ValueError(f"unsupported WR-08 robot-base mode: {raw!r}") from exc
    if resolved is RobotBaseMode.PRESERVE_RESET_POSE:
        if xy is not None or yaw is not None:
            raise ValueError("preserve_reset_pose does not accept xy or yaw")
        return RobotBaseConfig(resolved)
    if xy is None or isinstance(xy, (str, bytes)) or len(xy) != 2:
        raise ValueError("explicit_pose requires robot_base.xy")
    values = (float(xy[0]), float(xy[1]), float(yaw if yaw is not None else 0.0))
    if not all(isfinite(value) for value in values):
        raise ValueError("WR-08 explicit robot base pose must be finite")
    return RobotBaseConfig(resolved, values[:2], values[2])


def _region(value: Any, slot: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"WR-08 source region must be a mapping: {slot}")
    _reject_unknown(
        value, {"size", "pos", "margin", "surface_z"}, f"source region {slot} fields"
    )
    size = tuple(float(item) for item in value.get("size", ()))
    pos = tuple(float(item) for item in value.get("pos", ()))
    if len(size) != 2 or len(pos) != 2 or min(size) <= 0.0:
        raise ValueError(f"invalid WR-08 source region: {slot}")
    return {
        "size": size,
        "pos": pos,
        "margin": float(value.get("margin", 0.02)),
        "surface_z": float(value.get("surface_z", 0.93)),
    }


def _mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"WR-08 {name} must be a mapping")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(payload) - set(allowed)
    if unknown:
        raise ValueError(f"unknown WR-08 {label}: " + ", ".join(sorted(unknown)))


def _section(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"WR-08 operations.{name} must be a mapping")
    return value


__all__ = [
    "DEFAULT_TASK_CONFIG_FILE",
    "load_task_config",
    "resolve_action_config",
    "resolve_robot_base_config",
    "resolve_run_context",
    "resolve_task_options",
]
