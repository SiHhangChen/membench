"""WR-07 YAML loading and action configuration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from bench.tasks.wr07.domain.contracts import (
    ActionConfig, InitialStateConfig, RobotBaseConfig, RobotBaseMode,
    RunContext, SOURCE_SLOTS, TASK_ID, TaskConfig,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_TASK_CONFIG_FILE = PACKAGE_ROOT / "resources" / "task.yaml"

def load_task_config(path: str | Path = DEFAULT_TASK_CONFIG_FILE) -> TaskConfig:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"WR-07 config root must be a mapping: {config_path}")
    required = {"schema_version", "task", "scene", "episode", "initial_state", "sensors", "resources", "navigation", "manipulation", "operations", "semantics", "recording"}
    unknown = set(payload) - required
    if unknown:
        raise ValueError("unknown WR-07 config sections: " + ", ".join(sorted(unknown)))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("WR-07 config requires schema_version=1")
    task = dict(_mapping(payload, "task"))
    scene = dict(_mapping(payload, "scene"))
    if task.get("id") != TASK_ID:
        raise ValueError(f"WR-07 config must declare task.id={TASK_ID!r}")
    if scene.get("env_name") != "WR07MatchMilkCartonToColoredSugarPlate":
        raise ValueError("WR-07 config has an invalid scene.env_name")
    initial = _mapping(payload, "initial_state")
    base = _mapping(initial, "robot_base")
    source_regions = _mapping(initial, "source_regions")
    if set(source_regions) != set(SOURCE_SLOTS):
        raise ValueError("WR-07 initial_state.source_regions must define the three source slots")
    regions = {slot: _region(source_regions[slot], slot) for slot in SOURCE_SLOTS}
    resources = dict(_mapping(payload, "resources"))
    for name, value in resources.items():
        resource = config_path.parent / str(value)
        if not resource.is_file():
            raise FileNotFoundError(f"missing WR-07 resource {name!r}: {resource}")
    return TaskConfig(
        config_path, task, scene, dict(_mapping(payload, "episode")),
        InitialStateConfig(resolve_robot_base_config(**dict(base)), regions, dict(_mapping(initial, "objects"))),
        dict(_mapping(payload, "sensors")), resources,
        dict(_mapping(payload, "navigation")), dict(_mapping(payload, "manipulation")),
        dict(_mapping(payload, "operations")), dict(_mapping(payload, "semantics")),
        dict(_mapping(payload, "recording")),
    )

def resolve_action_config(*, config: TaskConfig) -> ActionConfig:
    operations = config.operations
    source = dict(_section(operations, "source_pick"))
    target = dict(_section(operations, "target_place"))
    return ActionConfig(
        source_pick_metadata={"milk_carton": source},
        target_place_metadata={slot: dict(target) for slot in ("target_left", "target_center", "target_right")},
        return_arm_empty=dict(_section(operations, "return_arm_empty")),
        return_arm_holding=dict(_section(operations, "return_arm_holding")),
        handler_settings=dict(_section(operations, "handlers")),
    )

def resolve_task_options(task_options: Mapping[str, Any] | None) -> dict[str, Any]:
    if task_options:
        raise ValueError("WR-07 v9 has no task options; all episode randomness is seed-owned")
    return {}

def resolve_run_context(*, split: str, variant: str, execute: bool, extra_metadata: Mapping[str, Any] | None = None) -> RunContext:
    if str(variant) not in {"memory", "oracle_reveal"}:
        raise ValueError("WR-07 variant must be 'memory' or 'oracle_reveal'")
    return RunContext(str(split), str(variant), bool(execute), dict(extra_metadata or {}))

def resolve_robot_base_config(*, mode: str | RobotBaseMode = RobotBaseMode.PRESERVE_RESET_POSE, xy: Sequence[float] | None = None, yaw: float | None = None, radius_m: float | None = None) -> RobotBaseConfig:
    raw = mode.value if isinstance(mode, RobotBaseMode) else str(mode).strip().lower()
    try:
        resolved = RobotBaseMode(raw)
    except ValueError as exc:
        raise ValueError(f"unsupported WR-07 robot-base mode: {raw!r}") from exc
    if resolved is RobotBaseMode.RESET_CIRCLE:
        if xy is not None or yaw is not None:
            raise ValueError("reset_circle uses the reset pose; xy and yaw are not allowed")
        radius = float(0.05 if radius_m is None else radius_m)
        if not isfinite(radius) or radius < 0.0:
            raise ValueError("reset_circle radius_m must be finite and non-negative")
        return RobotBaseConfig(resolved, radius_m=radius)
    if radius_m is not None:
        raise ValueError("radius_m is only supported by reset_circle")
    if resolved is RobotBaseMode.PRESERVE_RESET_POSE:
        if xy is not None or yaw is not None:
            raise ValueError("preserve_reset_pose does not accept xy or yaw")
        return RobotBaseConfig(resolved)
    if xy is None or isinstance(xy, (str, bytes)) or len(xy) != 2:
        raise ValueError("explicit_pose requires robot_base.xy")
    values = (float(xy[0]), float(xy[1]), float(yaw if yaw is not None else 0.0))
    if not all(isfinite(value) for value in values):
        raise ValueError("WR-07 explicit robot base pose must be finite")
    return RobotBaseConfig(resolved, values[:2], values[2])

def _region(value: Any, slot: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"WR-07 source region must be a mapping: {slot}")
    raw = value
    size = tuple(float(item) for item in raw.get("size", ()))
    pos = tuple(float(item) for item in raw.get("pos", ()))
    if len(size) != 2 or len(pos) != 2 or min(size) <= 0.0:
        raise ValueError(f"invalid WR-07 source region: {slot}")
    return {"size": size, "pos": pos, "margin": float(raw.get("margin", 0.02)), "surface_z": float(raw.get("surface_z", 0.93))}

def _mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"WR-07 {name} must be a mapping")
    return value

def _section(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"WR-07 operations.{name} must be a mapping")
    return value

__all__ = [
    "DEFAULT_TASK_CONFIG_FILE", "load_task_config", "resolve_action_config",
    "resolve_robot_base_config", "resolve_run_context", "resolve_task_options",
]
