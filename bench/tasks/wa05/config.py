"""Strict YAML/configuration boundary for the WA-05 production task."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from bench.tasks.wa05.domain.contracts import (
    CUP_IDS,
    SERVICE_SLOTS,
    ActionConfig,
    InitialStateConfig,
    RobotBaseConfig,
    RobotBaseMode,
    RunContext,
    TASK_ID,
    TaskConfig,
)
from bench.tasks.wa05.domain.episode import validate_temperature
from bench.tasks.wa05.domain.geometry import (
    resolve_cup_region_constraints,
    resolve_cup_sampling_regions,
    resolve_service_slot_layout,
    validate_cup_region_layout,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_TASK_CONFIG_FILE = PACKAGE_ROOT / "resources" / "task.yaml"
DEFAULT_INITIAL_STATION_ID = "cup_pick_station"


def _normalize_degrees(value: float) -> float:
    """Normalize one angle while keeping positive 180 as the sector center."""

    angle = float(value)
    normalized = (angle + 180.0) % 360.0 - 180.0
    if abs(normalized + 180.0) <= 1.0e-9 and angle >= 0.0:
        return 180.0
    return normalized


def _expand_placement_approach_sector(values: Mapping[str, Any]) -> dict[str, Any]:
    """Expand a task-level allowed entry sector into center-first candidates."""

    resolved = dict(values)
    raw_sector = resolved.pop("placement_approach_sector", None)
    if raw_sector is None:
        return resolved
    if "placement_approach_angles_deg" in resolved:
        raise ValueError(
            "WA-05 service placement cannot combine placement_approach_sector "
            "with placement_approach_angles_deg"
        )
    if not isinstance(raw_sector, Mapping):
        raise ValueError("WA-05 placement_approach_sector must be a mapping")
    sector = dict(raw_sector)
    _reject_unknown(
        sector,
        {"center_deg", "half_width_deg", "step_deg"},
        "placement_approach_sector fields",
    )
    center = float(sector.get("center_deg", 180.0))
    half_width = float(sector.get("half_width_deg", 60.0))
    step = float(sector.get("step_deg", 15.0))
    if not all(isfinite(value) for value in (center, half_width, step)):
        raise ValueError("WA-05 placement approach sector values must be finite")
    if not 0.0 < half_width <= 180.0:
        raise ValueError(
            "WA-05 placement approach sector half_width_deg must be in (0, 180]"
        )
    if not 0.0 < step <= half_width:
        raise ValueError(
            "WA-05 placement approach sector step_deg must be in (0, half_width_deg]"
        )

    positive_offsets: list[float] = []
    offset = step
    while offset < half_width - 1.0e-9:
        positive_offsets.append(float(offset))
        offset += step
    positive_offsets.append(float(half_width))

    candidates = [_normalize_degrees(center)]
    for offset in positive_offsets:
        candidates.extend(
            (
                _normalize_degrees(center - offset),
                _normalize_degrees(center + offset),
            )
        )
    resolved["placement_approach_angles_deg"] = list(
        dict.fromkeys(round(value, 10) for value in candidates)
    )
    return resolved


def _resolve_grasp_approach_sector(values: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a world-frame sector for filtering cup pregrasp positions."""

    resolved = dict(values)
    raw_sector = resolved.get("grasp_approach_sector")
    if raw_sector is None:
        return resolved
    if not isinstance(raw_sector, Mapping):
        raise ValueError("WA-05 grasp_approach_sector must be a mapping")
    sector = dict(raw_sector)
    _reject_unknown(
        sector,
        {"center_deg", "half_width_deg"},
        "grasp_approach_sector fields",
    )
    center = float(sector.get("center_deg", 180.0))
    half_width = float(sector.get("half_width_deg", 60.0))
    if not all(isfinite(value) for value in (center, half_width)):
        raise ValueError("WA-05 grasp approach sector values must be finite")
    if not 0.0 < half_width <= 180.0:
        raise ValueError(
            "WA-05 grasp approach sector half_width_deg must be in (0, 180]"
        )
    resolved["grasp_approach_sector"] = {
        "center_deg": _normalize_degrees(center),
        "half_width_deg": half_width,
        "frame": "world",
        "angle_reference": "cup_center_to_pregrasp",
    }
    return resolved


def load_task_config(path: str | Path = DEFAULT_TASK_CONFIG_FILE) -> TaskConfig:
    """Load one WA-05 YAML file and reject unsupported configuration surfaces."""

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"WA-05 config root must be a mapping: {config_path}")
    allowed = {
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
    _reject_unknown(payload, allowed, "config sections")
    task = dict(_mapping(payload, "task", config_path))
    if str(task.get("id")) != TASK_ID:
        raise ValueError(f"WA-05 config must declare task.id={TASK_ID!r}")
    initial = _mapping(payload, "initial_state", config_path)
    _reject_unknown(initial, {"robot_base", "objects"}, "initial_state fields")
    robot_base = _mapping(initial, "robot_base", config_path)
    _reject_unknown(
        robot_base,
        {"mode", "xy", "yaw", "sample_index", "station_id"},
        "initial_state.robot_base fields",
    )
    recording = dict(_mapping(payload, "recording", config_path))
    _reject_unknown(
        recording,
        {
            "record_depth",
            "depth_dtype",
            "depth_scale",
            "action_schema",
        },
        "recording fields",
    )
    config = TaskConfig(
        source_path=config_path,
        task=task,
        scene=dict(_mapping(payload, "scene", config_path)),
        episode=dict(_mapping(payload, "episode", config_path)),
        initial_state=InitialStateConfig(
            robot_base=resolve_robot_base_config(**dict(robot_base)),
            objects=dict(_mapping(initial, "objects", config_path)),
        ),
        sensors=dict(_mapping(payload, "sensors", config_path)),
        resources=dict(_mapping(payload, "resources", config_path)),
        navigation=dict(_mapping(payload, "navigation", config_path)),
        manipulation=dict(_mapping(payload, "manipulation", config_path)),
        operations=dict(_mapping(payload, "operations", config_path)),
        semantics=dict(_mapping(payload, "semantics", config_path)),
        recording=recording,
    )
    _validate_initial_state(config.initial_state.objects)
    _validate_production_operations(config.operations)
    # Validate the task-owned contract before resolving relative resources.
    # This makes a forbidden semantic bypass fail deterministically even when a
    # test or caller loads the YAML from a temporary directory.
    _validate_resources(config)
    return config


def resolve_robot_base_config(
    *,
    mode: str | RobotBaseMode = RobotBaseMode.MANUAL_PATH_SAMPLE,
    xy: Sequence[float] | None = None,
    yaw: float | None = None,
    sample_index: int | None = None,
    station_id: str | None = None,
) -> RobotBaseConfig:
    """Resolve the one reset-time base initialization method."""

    raw = mode.value if isinstance(mode, RobotBaseMode) else str(mode).strip().lower()
    try:
        resolved = RobotBaseMode(raw)
    except ValueError as exc:
        raise ValueError(f"unsupported WA-05 robot-base mode: {raw!r}") from exc
    supplied = {
        "xy": xy,
        "yaw": yaw,
        "sample_index": sample_index,
        "station_id": station_id,
    }
    if resolved is RobotBaseMode.EXPLICIT_POSE:
        _reject_supplied(resolved, supplied, allowed={"xy", "yaw"})
        if xy is None or isinstance(xy, (str, bytes)) or len(xy) != 2:
            raise ValueError("explicit_pose requires two-value robot_base.xy")
        result_xy = (float(xy[0]), float(xy[1]))
        if not all(isfinite(value) for value in result_xy):
            raise ValueError("robot_base.xy must be finite")
        return RobotBaseConfig(resolved, xy=result_xy, yaw=_finite(yaw, "robot_base.yaw"))
    if resolved is RobotBaseMode.MANUAL_PATH_SAMPLE:
        _reject_supplied(resolved, supplied, allowed={"sample_index"})
        index = 0 if sample_index is None else int(sample_index)
        if index < 0:
            raise ValueError("robot_base.sample_index must be non-negative")
        return RobotBaseConfig(resolved, sample_index=index)
    if resolved is RobotBaseMode.STATION_SAMPLE:
        _reject_supplied(resolved, supplied, allowed={"station_id"})
        station = DEFAULT_INITIAL_STATION_ID if station_id is None else str(station_id).strip()
        if not station:
            raise ValueError("robot_base.station_id must be non-empty")
        return RobotBaseConfig(resolved, station_id=station)
    _reject_supplied(resolved, supplied, allowed=set())
    return RobotBaseConfig(resolved)


def resolve_task_options(task_options: Mapping[str, Any] | None) -> dict[str, Any]:
    """Accept only typed label overrides used by calibration and debugging."""

    values = dict(task_options or {})
    allowed = {"expert_fill_temperature"}
    _reject_unknown(values, allowed, "task options")
    resolved: dict[str, Any] = {}
    if "expert_fill_temperature" in values:
        resolved["expert_fill_temperature"] = validate_temperature(values["expert_fill_temperature"])
    return resolved


def resolve_action_config(*, config: TaskConfig) -> ActionConfig:
    """Merge local resource paths and YAML operation settings into workflow metadata."""

    operations = config.operations
    cup_pick = dict(_section(operations, "cup_pick"))
    cup_pick_navigation_xy_tolerance = float(
        cup_pick.pop("navigation_xy_tolerance", 0.03)
    )
    if cup_pick_navigation_xy_tolerance <= 0.0:
        raise ValueError("WA-05 cup-pick navigation tolerance must be positive")
    cup_pick_navigation_yaw_tolerance = float(
        cup_pick.pop("navigation_yaw_tolerance_rad", 0.017453292519943295)
    )
    if (
        not isfinite(cup_pick_navigation_yaw_tolerance)
        or cup_pick_navigation_yaw_tolerance <= 0.0
    ):
        raise ValueError("WA-05 cup-pick yaw tolerance must be positive and finite")
    cup_pick = _resolve_grasp_approach_sector(cup_pick)
    sensor_faucets = dict(_section(operations, "sensor_faucets"))
    held_fill = dict(_section(operations, "held_fill"))
    held_fill_navigation_xy_tolerance = float(
        held_fill.pop("navigation_xy_tolerance", 0.03)
    )
    if held_fill_navigation_xy_tolerance <= 0.0:
        raise ValueError("WA-05 held-fill navigation tolerance must be positive")
    service_place = dict(_section(operations, "service_place"))
    answer_button = dict(_section(operations, "answer_button"))
    return_arm = dict(_section(operations, "return_arm"))
    return_arm_holding_cup = dict(_section(operations, "return_arm_holding_cup"))
    sink_exit_lift = dict(_section(operations, "sink_exit_lift"))
    by_temperature = _section(held_fill, "by_temperature")
    by_service_slot = _section(service_place, "by_slot")
    by_answer_temperature = _section(answer_button, "by_temperature")

    fill_base = {key: value for key, value in held_fill.items() if key != "by_temperature"}
    use_sink_fill_calibration = bool(
        fill_base.pop("use_sink_fill_calibration", False)
    )
    if use_sink_fill_calibration:
        fill_base["sink_fill_pose_file"] = str(_resource(config, "sink_fill_pose"))
    fill_by_temperature: dict[str, Mapping[str, Any]] = {}
    for temperature in ("hot", "cold"):
        override = _section(by_temperature, temperature)
        fill_by_temperature[temperature] = {
            **fill_base,
            **dict(override),
            "temperature": temperature,
        }

    service_base = {
        key: value for key, value in service_place.items() if key != "by_slot"
    }
    service_by_slot: dict[str, Mapping[str, Any]] = {}
    for slot in SERVICE_SLOTS:
        override = _section(by_service_slot, slot)
        service_by_slot[slot] = {
            **_expand_placement_approach_sector(
                {
                    **service_base,
                    **dict(override),
                }
            ),
            "service_slot": slot,
        }
    answer_base = {key: value for key, value in answer_button.items() if key != "by_temperature"}
    answer_by_temperature = {
        temperature: {**answer_base, **dict(_section(by_answer_temperature, temperature)), "temperature": temperature}
        for temperature in ("hot", "cold")
    }
    return ActionConfig(
        cup_pick_metadata=cup_pick,
        cup_pick_navigation_xy_tolerance=cup_pick_navigation_xy_tolerance,
        cup_pick_navigation_yaw_tolerance=cup_pick_navigation_yaw_tolerance,
        sensor_faucet_metadata=sensor_faucets,
        held_fill_navigation_xy_tolerance=held_fill_navigation_xy_tolerance,
        held_fill_metadata_by_temperature=fill_by_temperature,
        service_place_metadata_by_slot=service_by_slot,
        answer_button_metadata_by_temperature=answer_by_temperature,
        return_arm_metadata=return_arm,
        return_arm_holding_cup_metadata=return_arm_holding_cup,
        sink_exit_lift_metadata=sink_exit_lift,
        handler_settings=dict(_section(operations, "handlers")),
        semantic_thresholds=dict(config.semantics),
    )


def resolve_run_context(
    *,
    split: str,
    env_split: str,
    variant: str,
    execute: bool,
    force_manual_base_path_navigation: bool,
    extra_metadata: Mapping[str, Any] | None = None,
) -> RunContext:
    return RunContext(
        split=str(split),
        env_split=str(env_split),
        variant=str(variant),
        execute=bool(execute),
        force_manual_base_path_navigation=bool(force_manual_base_path_navigation),
        extra_metadata=dict(extra_metadata or {}),
    )


def load_json_resource(config: TaskConfig, name: str) -> dict[str, Any]:
    """Read one local JSON resource with an object root."""

    path = _resource(config, name)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"WA-05 resource {name!r} must contain a JSON object")
    return payload


def _validate_initial_state(objects: Mapping[str, Any]) -> None:
    cups = _section(objects, "cups")
    regions = resolve_cup_sampling_regions(cups)
    constraints = resolve_cup_region_constraints(
        _section(objects, "cup_region_constraints")
    )
    validate_cup_region_layout(regions, constraints)
    resolve_service_slot_layout(_section(objects, "service_slots"))
    fill_steps = int(objects.get("fill_under_water_steps", 8))
    if fill_steps < 1:
        raise ValueError("WA-05 fill_under_water_steps must be positive")
    starts = objects.get("initial_base_path_index_by_first_temperature")
    if starts is not None:
        if not isinstance(starts, Mapping) or set(starts) != {"hot", "cold"}:
            raise ValueError(
                "WA-05 initial_base_path_index_by_first_temperature must map hot and cold"
            )
        if any(int(starts[temperature]) < 0 for temperature in ("hot", "cold")):
            raise ValueError("WA-05 initial base-path indices must be non-negative")


def _validate_production_operations(operations: Mapping[str, Any]) -> None:
    """Forbid semantic bypasses in the production YAML surface."""

    forbidden = {
        "semantic_fill_proxy",
        "water_button_semantic_proxy",
        "service_place_semantic_proxy",
        "force_select_served_cup",
    }
    present = sorted(forbidden & _all_mapping_keys(operations))
    if present:
        raise ValueError("WA-05 production config forbids semantic proxies: " + ", ".join(present))


def _all_mapping_keys(value: Any) -> set[str]:
    if not isinstance(value, Mapping):
        return set()
    result = {str(key) for key in value}
    for item in value.values():
        result.update(_all_mapping_keys(item))
    return result


def _validate_resources(config: TaskConfig) -> None:
    for name in (
        "manual_base_path",
        "station_poses",
        "sensor_faucet_layout",
        "answer_button_positions",
        "sink_fill_pose",
        "planner_script",
    ):
        _resource(config, name)


def _resource(config: TaskConfig, name: str) -> Path:
    path = config.path(config.resources.get(name))
    if path is None or not path.is_file():
        raise FileNotFoundError(f"missing WA-05 resource {name!r}: {path}")
    return path


def _mapping(payload: Mapping[str, Any], name: str, path: Path) -> Mapping[str, Any]:
    value = payload.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"WA-05 {name} must be a mapping in {path}")
    return value


def _section(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"WA-05 {name} must be a mapping")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown WA-05 {label}: {', '.join(unknown)}")


def _reject_supplied(mode: RobotBaseMode, values: Mapping[str, Any], *, allowed: set[str]) -> None:
    rejected = sorted(name for name, value in values.items() if name not in allowed and value is not None)
    if rejected:
        raise ValueError(
            f"robot_base.mode={mode.value!r} does not accept: {', '.join(rejected)}"
        )


def _finite(value: Any, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is required")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


__all__ = [
    "DEFAULT_TASK_CONFIG_FILE",
    "load_json_resource",
    "load_task_config",
    "resolve_action_config",
    "resolve_robot_base_config",
    "resolve_run_context",
    "resolve_task_options",
]
