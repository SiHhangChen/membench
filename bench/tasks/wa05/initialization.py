"""Reset-time auditing and non-random scene configuration for WA-05."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.navigation.base_motion import set_base_pose_kinematic
from bench.runtime.contracts import InitializationReport
from bench.execution.navigation.station_poses import quat_xyzw_to_yaw
from bench.execution.planning import NavigationConfig
from bench.tasks.wa05.domain.contracts import (
    CUP_IDS,
    EpisodePlan,
    RobotBaseConfig,
    RobotBaseMode,
)
from bench.tasks.wa05.domain.geometry import (
    resolve_cup_region_constraints,
    resolve_cup_sampling_regions,
    resolve_service_slot_layout,
    sample_service_slot_relative_xy,
    validate_cup_region_layout,
)
from bench.tasks.wa05.execution.navigation import (
    build_episode_station_sampler,
    move_station_torso_to_height,
)


def apply_sampled_initial_object_regions(
    env: Any,
    *,
    settings: Mapping[str, Any],
    enabled: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Audit independently sampled cups without overwriting their reset poses."""

    values = dict(settings)
    summary: dict[str, Any] = {
        "enabled": bool(enabled),
        "coordinate_frame": "counter_fixture_local",
        "placement_mode": "independent_sampled_regions",
        "objects": {},
    }
    if not enabled:
        return env._get_observations(force_update=True), summary

    regions = resolve_cup_sampling_regions(_mapping(values, "cups"))
    constraints = resolve_cup_region_constraints(
        _mapping(values, "cup_region_constraints")
    )
    validate_cup_region_layout(regions, constraints)
    slots = resolve_service_slot_layout(_mapping(values, "service_slots"))
    sampled_service_xy = sample_service_slot_relative_xy(
        slots,
        seed=int(env.cup_placement_seed),
    )

    summary["placement_seed"] = int(env.cup_placement_seed)
    summary["sampling_regions"] = env.cup_sampling_regions()
    summary["constraints"] = {
        "sink_exterior_local_x_max": float(
            constraints.sink_exterior_local_x_max
        ),
        "minimum_sink_clearance": float(constraints.minimum_sink_clearance),
        "minimum_inter_region_gap": float(
            constraints.minimum_inter_region_gap
        ),
        "sampling_inset_xy": list(constraints.sampling_inset_xy),
    }
    for cup_id in CUP_IDS:
        region = regions[cup_id]
        qpos = np.array(_free_joint_qpos(env, cup_id), dtype=float, copy=True)
        relative_xy = _counter_local_xy(env, qpos[:3])
        bbox_xy = _counter_local_bbox_xy(env, cup_id, qpos)
        lower = np.asarray(
            [region.x_bounds[0], region.y_bounds[0]],
            dtype=float,
        )
        upper = np.asarray(
            [region.x_bounds[1], region.y_bounds[1]],
            dtype=float,
        )
        bbox_min = np.min(bbox_xy, axis=0)
        bbox_max = np.max(bbox_xy, axis=0)
        if np.any(bbox_min < lower - 1e-6) or np.any(bbox_max > upper + 1e-6):
            raise RuntimeError(
                f"WA-05 sampled {cup_id} bbox escaped its region: "
                f"bbox_min={bbox_min.tolist()} bbox_max={bbox_max.tolist()} "
                f"region_min={lower.tolist()} region_max={upper.tolist()}"
            )
        summary["objects"][cup_id] = {
            "sampled_relative_xy": relative_xy.astype(float).tolist(),
            "sampled_world_position": qpos[:3].astype(float).tolist(),
            "sampled_world_quat_wxyz": qpos[3:7].astype(float).tolist(),
            "region_center_xy": list(region.center_xy),
            "region_size_xy": list(region.size_xy),
            "region_x_bounds": list(region.x_bounds),
            "region_y_bounds": list(region.y_bounds),
            "bbox_local_xy_min": bbox_min.astype(float).tolist(),
            "bbox_local_xy_max": bbox_max.astype(float).tolist(),
            "yaw_rad": float(region.yaw_rad),
            "pose_overwritten_after_reset": False,
        }

    env.configure_service_slots(
        regions_by_slot={
            slot: {
                "center_xy": list(region.center_xy),
                "size_xy": list(region.size_xy),
            }
            for slot, region in slots.regions_by_slot.items()
        },
        sampled_relative_xy_by_slot=sampled_service_xy,
        surface_z_offset=float(slots.surface_z_offset),
        service_threshold=float(values.get("service_threshold", env.service_threshold)),
        cup_upright_threshold_deg=float(
            values.get("cup_upright_threshold_deg", env.cup_upright_threshold_deg)
        ),
    )
    # This is a physical execution requirement rather than hidden task state.
    # Keeping it on the environment makes the configured duration inspectable
    # in recorder diagnostics without granting a semantic shortcut.
    env.fill_under_water_steps_required = int(values.get("fill_under_water_steps", 8))
    summary["service_slots"] = {
        "placement_mode": "independent_sampled_regions",
        "placement_seed": int(env.cup_placement_seed),
        "regions": {
            slot: {
                "center_xy": list(region.center_xy),
                "size_xy": list(region.size_xy),
                "x_bounds": list(region.x_bounds),
                "y_bounds": list(region.y_bounds),
            }
            for slot, region in slots.regions_by_slot.items()
        },
        "sampled_relative_xy_by_slot": {
            slot: list(xy) for slot, xy in sampled_service_xy.items()
        },
        "placement_center_margin_xy": list(slots.placement_center_margin_xy),
        "surface_z_offset": float(slots.surface_z_offset),
        "minimum_xy_separation": float(slots.minimum_xy_separation),
        "minimum_inter_region_gap": float(slots.minimum_inter_region_gap),
        "world_positions": {
            slot: np.asarray(env.service_slot_position(slot), dtype=float).tolist()
            for slot in slots.regions_by_slot
        },
    }
    summary["fill_under_water_steps_required"] = int(
        env.fill_under_water_steps_required
    )
    return env._get_observations(force_update=True), summary


def apply_robot_base_initial_state(
    env: Any,
    obs: dict[str, Any],
    *,
    config: RobotBaseConfig,
    nav_config: NavigationConfig,
    episode: EpisodePlan,
    initial_objects: Mapping[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the one allowed reset-time base initialization method.

    Station reset uses the same deterministic episode draw as navigation.
    Temperature is selected later by the faucet side, not by the base pose.
    """

    first_temperature = str(episode.fill_temperature_order[0])
    if config.mode is RobotBaseMode.PRESERVE_RESET_POSE:
        result = _base_summary(obs, enabled=False, mode=config.mode.value)
        result["first_fill_temperature"] = first_temperature
        return obs, result
    if config.mode is RobotBaseMode.EXPLICIT_POSE:
        if config.xy is None or config.yaw is None:
            raise ValueError("explicit_pose requires resolved xy and yaw")
        new_obs, result = _set_base_pose(
            env,
            obs,
            target_xy=np.asarray(config.xy, dtype=float),
            target_yaw=float(config.yaw),
            mode=config.mode.value,
        )
        result["first_fill_temperature"] = first_temperature
        return new_obs, result
    if config.mode is RobotBaseMode.MANUAL_PATH_SAMPLE:
        path = nav_config.manual_base_path_file
        if path is None:
            raise ValueError("manual_path_sample requires a manual base-path resource")
        sample_index, source = _manual_path_index_for_first_temperature(
            config,
            initial_objects,
            first_temperature=first_temperature,
        )
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        samples = {
            int(item.get("index", fallback)): dict(item)
            for fallback, item in enumerate(payload.get("samples", ()))
        }
        sample = samples.get(sample_index)
        if sample is None:
            raise KeyError(f"manual base sample {sample_index} not found in {path}")
        xy = np.asarray(sample.get("xy", ()), dtype=float).reshape(-1)
        yaw = float(sample.get("yaw"))
        if xy.size != 2 or not np.all(np.isfinite(xy)) or not np.isfinite(yaw):
            raise ValueError(f"invalid manual base sample {sample_index}")
        new_obs, result = _set_base_pose(
            env,
            obs,
            target_xy=xy,
            target_yaw=yaw,
            mode=config.mode.value,
        )
        result.update(
            {
                "manual_base_path_file": str(path),
                "sample_index": int(sample_index),
                "sample_name": sample.get("name"),
                "sample": sample,
                "first_fill_temperature": first_temperature,
                "sample_index_source": source,
            }
        )
        return new_obs, result
    if config.mode is RobotBaseMode.STATION_SAMPLE:
        path = nav_config.station_base_pose_file
        if path is None or config.station_id is None:
            raise ValueError("station_sample requires a station pose resource and station id")
        sampler = build_episode_station_sampler(nav_config, seed=int(seed))
        pose = sampler.sample(config.station_id)
        station_summary = dict(sampler.last_summary(config.station_id) or {})
        new_obs, result = _set_base_pose(
            env,
            obs,
            target_xy=np.asarray(pose.position[:2], dtype=float),
            target_yaw=quat_xyzw_to_yaw(pose.quaternion),
            mode=config.mode.value,
        )
        torso = move_station_torso_to_height(env, pose.torso_height)
        new_obs = env._get_observations(force_update=True)
        result.update(_base_summary(new_obs, enabled=True, mode=config.mode.value))
        result.update(
            {
                "station_id": station_summary["station_id"],
                "station_base_pose_file": str(path),
                "selection": station_summary["station_pose_selection"],
                "sample_index": int(station_summary["sample_index"]),
                "sample": dict(station_summary["sample"]),
                "station_torso": torso,
                "first_fill_temperature": first_temperature,
            }
        )
        return new_obs, result
    raise ValueError(f"unsupported WA-05 base mode: {config.mode!r}")


def _manual_path_index_for_first_temperature(
    config: RobotBaseConfig,
    initial_objects: Mapping[str, Any],
    *,
    first_temperature: str,
) -> tuple[int, str]:
    """Resolve optional per-temperature starts while preserving config fallback."""

    mapping = initial_objects.get("initial_base_path_index_by_first_temperature")
    if isinstance(mapping, Mapping) and first_temperature in mapping:
        index = int(mapping[first_temperature])
        if index < 0:
            raise ValueError("initial base-path indices must be non-negative")
        return index, "initial_state.objects.initial_base_path_index_by_first_temperature"
    if config.sample_index is None:
        raise ValueError("manual_path_sample requires a resolved sample_index")
    return int(config.sample_index), "initial_state.robot_base.sample_index"


def _counter_local_xy(env: Any, world_position: Any) -> np.ndarray:
    from robocasa.utils import object_utils as OU

    return np.asarray(
        OU.get_fixture_to_point_rel_offset(
            env.counter,
            np.asarray(world_position, dtype=float),
        ),
        dtype=float,
    )[:2]


def _counter_local_bbox_xy(
    env: Any,
    object_name: str,
    qpos: np.ndarray,
) -> np.ndarray:
    quat_xyzw = np.asarray(qpos[3:7], dtype=float)[[1, 2, 3, 0]]
    points = env.objects[object_name].get_bbox_points(
        trans=np.asarray(qpos[:3], dtype=float),
        rot=quat_xyzw,
    )
    return np.vstack([_counter_local_xy(env, point) for point in points])


def _free_joint_qpos(env: Any, object_name: str) -> np.ndarray:
    qpos = np.asarray(
        env.sim.data.get_joint_qpos(env.objects[object_name].joints[0]), dtype=float
    ).reshape(-1)
    if qpos.size < 7:
        raise ValueError(f"WA-05 object {object_name!r} must have a free joint")
    return qpos


def _set_base_pose(
    env: Any,
    obs: dict[str, Any],
    *,
    target_xy: np.ndarray,
    target_yaw: float,
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reset_xy = np.asarray(obs["robot0_base_pos"][:2], dtype=float)
    new_obs = set_base_pose_kinematic(
        env,
        target_xy=np.asarray(target_xy, dtype=float),
        target_yaw=float(target_yaw),
    )
    summary = _base_summary(new_obs, enabled=True, mode=mode)
    summary.update(
        {
            "reset_base_xy": reset_xy.astype(float).tolist(),
            "target_xy": np.asarray(target_xy, dtype=float).tolist(),
            "target_yaw": float(target_yaw),
        }
    )
    return new_obs, summary


def _base_summary(obs: Mapping[str, Any], *, enabled: bool, mode: str) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "mode": str(mode),
        "actual_base_pos": np.asarray(obs["robot0_base_pos"], dtype=float).tolist(),
        "actual_base_quat_xyzw": np.asarray(
            obs["robot0_base_quat"], dtype=float
        ).tolist(),
    }


def _mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"WA-05 initial_state.objects.{name} must be a mapping")
    return value


__all__ = [
    "apply_robot_base_initial_state",
    "apply_sampled_initial_object_regions",
]


def initialize_episode(runtime: Any, episode: Any) -> InitializationReport:
    """Expose reset facts already applied by the task runtime."""
    del episode
    return InitializationReport.applied(facts=dict(getattr(runtime, "initialization", {}) or {}))
