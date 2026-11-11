"""WR-11 manipulation assembly for the milk-carton transport."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from bench.execution.manipulation import ManipulationController, ManipulationRuntimeConfig
from bench.execution.planning import MplibPlannerConfig, PickPlaceConfig
from bench.execution.planning.mplib_planner import run_mplib_planner_subprocess
from bench.tasks.wr11.execution.geometry import build_pick_payload_selector


@dataclass(frozen=True)
class ManipulationAssembly:
    controller: ManipulationController
    planner_settings: Mapping[str, Any]


def build_manipulation_assembly(
    env: Any,
    obs: dict[str, Any],
    *,
    settings: Mapping[str, Any],
    planner_script: Path,
    execute: bool,
) -> ManipulationAssembly:
    planner = _section(settings, "planner")
    controller = ManipulationController(
        env,
        obs,
        planner=_make_planner(
            script_path=planner_script,
            conda_env=str(planner.get("conda_env", "membench")),
            timeout_s=float(planner.get("subprocess_timeout_s", 60.0)),
        ),
        pick_place_config=_pick_place_config(_section(settings, "pick_place")),
        runtime_config=_runtime_config(
            execute=execute,
            runtime=_section(settings, "runtime"),
            planner=planner,
            static_scene=_section(settings, "static_scene"),
        ),
        pick_payload_selector=build_pick_payload_selector(),
    )
    return ManipulationAssembly(controller=controller, planner_settings=planner)


def _pick_place_config(values: Mapping[str, Any]) -> PickPlaceConfig:
    settings = dict(values)
    config = replace(
        PickPlaceConfig(),
        orientation_mode=str(settings.pop("orientation_mode", "object-y-topdown")),
        mug_orientation_mode=str(settings.pop("mug_orientation_mode", "mug-body-side")),
        pregrasp_height=float(settings.pop("pregrasp_height", 0.15)),
        grasp_clearance=float(settings.pop("grasp_clearance", -0.04)),
        lift_height=float(settings.pop("lift_height", 0.12)),
        place_height=float(settings.pop("place_height", 0.05)),
        retreat_height=float(settings.pop("retreat_height", 0.18)),
        retreat_x=float(settings.pop("retreat_x", 0.0)),
        retreat_y=float(settings.pop("retreat_y", -0.10)),
    )
    if settings:
        raise ValueError("unknown WR-11 pick_place settings: " + ", ".join(sorted(settings)))
    return config


def _runtime_config(
    *,
    execute: bool,
    runtime: Mapping[str, Any],
    planner: Mapping[str, Any],
    static_scene: Mapping[str, Any],
) -> ManipulationRuntimeConfig:
    values = dict(runtime)
    planner_known = {
        "planning_time",
        "time_step",
        "qpos_step",
        "rrt_range",
        "conda_env",
        "subprocess_timeout_s",
        "preserve_ik_branch_stages",
        "nearest_ik_samples",
        "nearest_ik_threshold",
        "nearest_ik_max_joint_delta",
    }
    unknown_planner = sorted(set(planner) - planner_known)
    if unknown_planner:
        raise ValueError("unknown WR-11 planner settings: " + ", ".join(unknown_planner))
    planner_config = MplibPlannerConfig(
        planning_time=float(planner.get("planning_time", 5.0)),
        time_step=float(planner.get("time_step", 0.1)),
        qpos_step=float(planner.get("qpos_step", 0.02)),
        rrt_range=float(planner.get("rrt_range", 0.1)),
    )
    planner_args = planner_config.planner_args()
    planner_args.update(
        {
            "preserve_ik_branch_stages": list(planner.get("preserve_ik_branch_stages", ["place"])),
            "nearest_ik_samples": int(planner.get("nearest_ik_samples", 64)),
            "nearest_ik_threshold": float(planner.get("nearest_ik_threshold", 0.01)),
            "nearest_ik_max_joint_delta": float(planner.get("nearest_ik_max_joint_delta", 2.0)),
        }
    )
    scene = dict(static_scene)
    scene_options = {
        "enabled": bool(scene.pop("enabled", False)),
        "validate_collisions": bool(scene.pop("validate_collisions", False)),
        "fixture_keywords": str(scene.pop("fixture_keywords", "counter,sink,cabinet")),
        "exclude_keywords": str(scene.pop("exclude_keywords", "floor,wall")),
        "spacing": float(scene.pop("spacing", 0.04)),
        "inflate": float(scene.pop("inflate", 0.05)),
        "point_radius": float(scene.pop("point_radius", 0.01)),
        "faces": str(scene.pop("faces", "bottom")),
        "point_cloud_stages": str(
            scene.pop("point_cloud_stages", "pregrasp,grasp,lift,place,clearance,retreat")
        ),
        "min_z": float(scene.pop("min_z", 1.0)),
        "max_points": int(scene.pop("max_points", 4000)),
    }
    if scene:
        raise ValueError("unknown WR-11 static_scene settings: " + ", ".join(sorted(scene)))
    config = ManipulationRuntimeConfig(
        execute=bool(execute),
        max_joint_step=float(values.pop("max_joint_step", 0.025)),
        settle_steps=int(values.pop("settle_steps", 8)),
        position_threshold=float(values.pop("position_threshold", 0.02)),
        orientation_threshold_deg=float(values.pop("orientation_threshold_deg", 15.0)),
        qpos_threshold=float(values.pop("qpos_threshold", 0.05)),
        open_steps=int(values.pop("open_steps", 4)),
        close_steps=int(values.pop("close_steps", 20)),
        release_steps=int(values.pop("release_steps", 20)),
        release_settle_steps=int(values.pop("release_settle_steps", 8)),
        planner_args=planner_args,
        static_scene_options=scene_options,
    )
    if values:
        raise ValueError("unknown WR-11 runtime settings: " + ", ".join(sorted(values)))
    return config


def _make_planner(*, script_path: Path, conda_env: str, timeout_s: float):
    def plan(payload: dict[str, Any]) -> dict[str, Any]:
        code, result = run_mplib_planner_subprocess(
            script_path=script_path,
            plan_request=payload,
            conda_env=conda_env,
            timeout_s=timeout_s,
        )
        if result is None:
            return {
                "subprocess_returncode": int(code),
                "plan": {"status": "planner_subprocess_no_json", "attempts": []},
            }
        result["subprocess_returncode"] = int(code)
        return result

    return plan


def _section(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"WR-11 manipulation.{name} must be a mapping")
    return value


__all__ = ["ManipulationAssembly", "build_manipulation_assembly"]
