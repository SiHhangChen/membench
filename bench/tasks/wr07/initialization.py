"""Apply the resolved WR-07 v9 episode to the live environment."""

from __future__ import annotations

from typing import Any

import numpy as np

from bench.execution.navigation.base_motion import base_yaw, set_base_pose_kinematic
from bench.execution.sampling.circle_sampling import sample_uniform_disk
from bench.runtime.contracts import InitializationReport
from bench.tasks.wr07.domain.contracts import EpisodePlan, RobotBaseConfig, RobotBaseMode

def initialize_episode_state(env: Any, obs: dict[str, Any], *, episode: EpisodePlan, robot_base: RobotBaseConfig, objects: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if objects:
        raise ValueError("unknown WR-07 initial object settings: " + ", ".join(sorted(objects)))
    env.configure_episode(episode)
    base_sampling = None
    if robot_base.mode is RobotBaseMode.EXPLICIT_POSE:
        if robot_base.xy is None or robot_base.yaw is None:
            raise ValueError("explicit WR-07 robot base pose is incomplete")
        obs = set_base_pose_kinematic(env, target_xy=np.asarray(robot_base.xy, dtype=float), target_yaw=float(robot_base.yaw))
    elif robot_base.mode is RobotBaseMode.RESET_CIRCLE:
        if robot_base.radius_m is None:
            raise ValueError("WR-07 reset_circle radius is incomplete")
        center = np.asarray(obs["robot0_base_pos"], dtype=float)[:2]
        yaw = base_yaw(obs)
        sampled_xy = sample_uniform_disk(
            center, robot_base.radius_m, seed=episode.seed, salt=0x57523037,
        )
        obs = set_base_pose_kinematic(env, target_xy=sampled_xy, target_yaw=yaw)
        base_sampling = {
            "policy": "reset_circle_uniform_area",
            "center_xy": center.tolist(),
            "radius_m": robot_base.radius_m,
            "sampled_xy": sampled_xy.tolist(),
            "offset_xy": (sampled_xy - center).tolist(),
            "preserved_yaw": yaw,
        }
    elif robot_base.mode is not RobotBaseMode.PRESERVE_RESET_POSE:
        raise ValueError(f"unsupported WR-07 robot-base mode: {robot_base.mode!r}")
    return obs, {
        "episode_seed": episode.seed,
        "source_plate_color_by_slot": dict(episode.source_plate_color_by_slot),
        "target_plate_color_by_slot": dict(episode.target_plate_color_by_slot),
        "source_slot_by_item": dict(episode.source_slot_by_item),
        "source_color_by_item": dict(episode.source_color_by_item),
        "target_slot_by_color": dict(episode.target_slot_by_color),
        "source_item_offset_xy_by_item": {key: list(value) for key, value in episode.source_item_offset_xy_by_item.items()},
        "target_item_offset_xy_by_item": {key: list(value) for key, value in episode.target_item_offset_xy_by_item.items()},
        "carton_type": episode.carton_type,
        "carton_asset": episode.carton_asset,
        "carton_material": episode.carton_material,
        "carton_type_candidates": list(episode.carton_type_candidates),
        "carton_material_candidates": list(episode.carton_material_candidates),
        "carton_combination_count": len(episode.carton_type_candidates) * len(episode.carton_material_candidates),
        "initial_source_verified": True,
        "robot_base_mode": robot_base.mode.value,
        "initial_base_sampling": base_sampling,
        "actual_base_pos": np.asarray(obs.get("robot0_base_pos", (0.0, 0.0)), dtype=float).tolist(),
    }

def initialize_episode(runtime: Any, episode: Any) -> InitializationReport:
    del episode
    return InitializationReport.applied(facts=dict(getattr(runtime, "initialization", {}) or {}))

__all__ = ["initialize_episode", "initialize_episode_state"]
