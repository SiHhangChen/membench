"""Validated task-local configuration for WR-12."""

from collections.abc import Mapping
from math import isfinite
from pathlib import Path

import yaml

from .domain.contracts import ENV_NAME, TASK_ID, ActionConfig, TaskConfig
from .domain.episode import load_sampling_rules, load_station_circles, resolve_episode_plan
from .visibility import validate_box_asset, validate_shutter_asset

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
        raise ValueError("WR-12 requires schema_version=1")
    if set(payload) != {"schema_version", *sections}:
        raise ValueError("WR-12 missing or unknown configuration sections")
    if any(not isinstance(payload[k], Mapping) for k in sections):
        raise ValueError("WR-12 sections must be mappings")
    if payload["task"].get("id") != TASK_ID or payload["scene"].get("env_name") != ENV_NAME:
        raise ValueError("WR-12 identity mismatch")
    if payload["initial_state"] != {"robot_base": {"mode": "observe_station"}}:
        raise ValueError("WR-12 must start at the observation station")
    config = TaskConfig(path, **{key: dict(payload[key]) for key in sections})
    for name, value in config.resources.items():
        if not config.path(value).is_file():
            raise FileNotFoundError(f"WR-12 missing {name}: {config.path(value)}")
    load_sampling_rules(config.path(config.resources["sampling_rules"]))
    load_station_circles(config.path(config.resources["stations"]))
    validate_box_asset(config.path(config.resources["box_asset"]))
    validate_shutter_asset(config.path(config.resources["shutter_asset"]))
    if config.semantics["viewing_steps"] != 20 or config.semantics["shutter_closing_steps"] != 6:
        raise ValueError("WR-12 requires 1 second viewing and 0.3 second shutter closing at 20 Hz")
    for name in ("observation_dwell_steps", "hidden_dwell_steps", "placement_dwell_steps"):
        value = config.semantics[name]
        if int(value) != value or value < 1:
            raise ValueError(f"WR-12 {name} must be a positive integer")
    for name in ("minimum_lift_m", "can_motion_tolerance_m"):
        value = config.semantics[name]
        if not isfinite(value) or value <= 0:
            raise ValueError(f"WR-12 {name} must be positive and finite")
    return config


def resolve_configured_episode_plan(config, seed):
    return resolve_episode_plan(
        seed=seed,
        **dict(config.episode),
        sampling_rules=load_sampling_rules(config.path(config.resources["sampling_rules"])),
        station_circles=load_station_circles(config.path(config.resources["stations"])),
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
        raise ValueError("WR-12 uses seed-owned sampling; task options are unsupported")
    return {}
