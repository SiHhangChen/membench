"""WR-11 configuration and validation."""

from collections.abc import Mapping
from math import isfinite
from pathlib import Path

import yaml

from .domain.contracts import ENV_NAME, TASK_ID, ActionConfig, TaskConfig
from .domain.sampling import load_sampling_rules, load_station_circles

DEFAULT_TASK_CONFIG_FILE = Path(__file__).parent / "resources/task.yaml"


def load_task_config(path=DEFAULT_TASK_CONFIG_FILE):
    path = Path(path).resolve()
    payload = yaml.safe_load(path.read_text())
    sections = (
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
    )
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("WR-11 requires a schema_version=1 mapping")
    if set(payload) != {"schema_version", *sections}:
        raise ValueError("WR-11 config has missing or unknown sections")
    if any(not isinstance(payload[key], Mapping) for key in sections):
        raise ValueError("WR-11 config sections must be mappings")
    if payload["task"].get("id") != TASK_ID or payload["scene"].get("env_name") != ENV_NAME:
        raise ValueError("WR-11 config identity mismatch")
    base = payload["initial_state"].get("robot_base", {})
    if set(payload["initial_state"]) != {"robot_base"} or base.get("mode") != "sample_station":
        raise ValueError("WR-11 must start at the sample station")
    if set(base) != {"mode"}:
        raise ValueError("WR-11 robot radii are owned by the station calibration")
    config = TaskConfig(path, **{key: dict(payload[key]) for key in sections})
    for name, value in config.resources.items():
        if not config.path(value).is_file():
            raise FileNotFoundError(f"missing WR-11 resource {name}: {config.path(value)}")
    load_sampling_rules(config.path(config.resources["sampling_rules"]))
    load_station_circles(config.path(config.resources["stations"]))
    for key in ("observation_dwell_steps", "placement_dwell_steps"):
        value = config.semantics[key]
        if int(value) != value or value < 1:
            raise ValueError(f"WR-11 {key} must be a positive integer")
    for key in ("minimum_lift_m", "sample_motion_tolerance_m"):
        value = float(config.semantics[key])
        if not isfinite(value) or value <= 0:
            raise ValueError(f"WR-11 {key} must be positive and finite")
    return config


def resolve_configured_episode_plan(config, seed):
    from .domain.episode import resolve_episode_plan

    return resolve_episode_plan(
        seed=seed,
        sampling_rules=load_sampling_rules(config.path(config.resources["sampling_rules"])),
        station_circles=load_station_circles(config.path(config.resources["stations"])),
        **dict(config.episode),
    )


def resolve_action_config(*, config):
    return ActionConfig(
        **{
            key: dict(config.operations[key])
            for key in ("source_pick", "target_place", "return_arm_empty", "return_arm_holding")
        }
    )


def resolve_task_options(options):
    if options:
        raise ValueError("WR-11 randomness is seed-owned; task options are unsupported")
    return {}
