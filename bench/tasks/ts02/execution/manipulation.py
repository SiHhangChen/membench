"""TS02 manipulation assembly and optional contact drawer primitives."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.manipulation import ManipulationController, ManipulationRuntimeConfig
from bench.execution.planning import (
    MplibPlannerConfig,
    PickPlaceConfig,
    matrix_from_base_obs,
    matrix_from_site_pose,
    pose_from_matrix,
)
from bench.execution.planning.mplib_planner import run_mplib_planner_subprocess
from bench.tasks.ts02.domain.contracts import ENV_NAME
from bench.tasks.ts02.execution.geometry import (
    DrawerMotionFrame,
    drawer_pull_target_positions,
    pose_matrix,
    resolve_drawer_motion_frame,
    vary_drawer_grasp_targets,
)


@dataclass(frozen=True)
class ManipulationAssembly:
    controller: ManipulationController
    planner_settings: Mapping[str, Any]


@dataclass(frozen=True)
class DrawerPullConfig:
    hook_orientation_blend: float = 0.90
    handle_side_offset: float = 0.0
    handle_vertical_offset: float = 0.0
    prehook_variant: str = "baseline"
    prehook_rotation_offset_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    prehook_distance: float = 0.125
    hook_offset: float = 0.005
    pull_distance: float = 0.125
    open_fraction_threshold: float = 0.70
    gripper_steps: int = 25
    grasp_pose_file: str | Path | None = None
    close_gripper_on_hook: bool = False
    close_gripper_action: float = 1.0
    close_gripper_steps: int = 25


@dataclass(frozen=True)
class DrawerPushCloseConfig:
    close_overshoot: float = 0.005
    closed_fraction_threshold: float = 0.08


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
            planner_script,
            conda_env=str(planner.get("conda_env", "membench")),
            timeout_s=_optional_float(planner.get("subprocess_timeout_s", 60.0)),
        ),
        pick_place_config=_pick_place_config(_section(settings, "pick_place")),
        runtime_config=_runtime_config(
            execute=execute,
            runtime=_section(settings, "runtime"),
            planner=planner,
            static_scene=_section(settings, "static_scene"),
        ),
    )
    return ManipulationAssembly(controller, planner)


def open_drawer_physically(
    controller: Any,
    *,
    fixture_name: str,
    config: DrawerPullConfig,
) -> dict[str, Any]:
    """Open a drawer through gripper-handle contact without setting its joint."""

    if not bool(getattr(controller.runtime_config, "execute", False)):
        return {
            "primitive": "physical_drawer_pull",
            "fixture": str(fixture_name),
            "pass": False,
            "reason": "physical_drawer_pull_requires_execute",
        }
    env = controller.env
    try:
        controller.refresh_obs()
        fixture = env.cabinet_for_slot(str(fixture_name))
        joint_name = str(fixture.door_joint_names[0])
        handle_id = int(env.sim.model.geom_name2id(fixture.handle_name))
        initial_qpos = _joint_qpos(env, joint_name)
        initial_fraction = float(fixture.open_fraction(env))
        locked_base = _locked_base_qpos(env)
        trace: list[dict[str, Any]] = []

        def post_step(_obs: dict[str, Any]) -> dict[str, Any]:
            trace.append(
                {
                    "drawer_qpos": _joint_qpos(env, joint_name),
                    "contacts": _robot_fixture_contacts(env, str(fixture.name)),
                }
            )
            return _restore_locked_base(env, locked_base)

        current_rotation = matrix_from_site_pose(env, env.robots[0])[:3, :3]
        frame = resolve_drawer_motion_frame(
            fixture_yaw=float(fixture.rot),
            current_eef_rotation_world=current_rotation,
            hook_orientation_blend=float(config.hook_orientation_blend),
        )
        initial_handle = np.asarray(env.sim.data.geom_xpos[handle_id], dtype=float)
        target_matrices, pose_metadata = _drawer_pull_targets(
            initial_handle, frame, config
        )
        gripper_result: dict[str, Any] | None = None
        if int(config.gripper_steps) > 0:
            gripper_result = controller.driver.run_gripper(
                arm_qpos=controller.driver.current_arm_qpos(),
                gripper_action=-1.0,
                steps=int(config.gripper_steps),
                post_step_callback=post_step,
            )
            controller._sync_obs(gripper_result["obs"])

        stages: dict[str, dict[str, Any]] = {}

        def run(
            name: str,
            target_world: np.ndarray,
            *,
            gripper_action: float = -1.0,
        ) -> dict[str, Any]:
            target_base = (
                np.linalg.inv(matrix_from_base_obs(controller.obs)) @ target_world
            )
            result = controller.stage_executor.run(
                pose_from_matrix(target_base),
                stage=f"ts02_drawer_{name}",
                gripper_action=float(gripper_action),
                post_step_callback=post_step,
            )
            controller._sync_obs(controller.stage_executor.obs)
            stages[name] = result
            return result

        prehook = run("prehook", target_matrices["prehook"])
        hook: dict[str, Any] | None = None
        pull: dict[str, Any] | None = None
        close_gripper: dict[str, Any] | None = None
        contacts_after_close: list[dict[str, Any]] = []
        if _stage_executed(prehook):
            hook = run("hook", target_matrices["hook"])
        if hook is not None and _stage_executed(hook):
            if bool(config.close_gripper_on_hook):
                close_gripper = controller.driver.run_gripper(
                    arm_qpos=controller.driver.current_arm_qpos(),
                    gripper_action=float(config.close_gripper_action),
                    steps=int(config.close_gripper_steps),
                    post_step_callback=post_step,
                )
                controller._sync_obs(close_gripper["obs"])
                contacts_after_close = _robot_fixture_contacts(
                    env, str(fixture.name)
                )
            live_handle = np.asarray(env.sim.data.geom_xpos[handle_id], dtype=float)
            live_targets, _ = _drawer_pull_targets(live_handle, frame, config)
            pull_target = live_targets["pull"]
            pull = run(
                "pull",
                pull_target,
                gripper_action=(
                    float(config.close_gripper_action)
                    if bool(config.close_gripper_on_hook)
                    else -1.0
                ),
            )

        final_qpos = _joint_qpos(env, joint_name)
        final_fraction = float(fixture.open_fraction(env))
        contact_samples = sum(bool(item["contacts"]) for item in trace)
        checks = {
            "prehook_executed": _stage_executed(prehook),
            "hook_executed": hook is not None and _stage_executed(hook),
            "pull_executed": pull is not None and _stage_executed(pull),
            "drawer_open": final_fraction >= float(config.open_fraction_threshold),
            "robot_fixture_contact_observed": contact_samples > 0,
        }
        if bool(config.close_gripper_on_hook):
            checks["gripper_closed_on_hook"] = close_gripper is not None
        reason = None
        if not all(checks.values()):
            if not checks["drawer_open"]:
                reason = "drawer_open_fraction_below_threshold"
            elif not checks["robot_fixture_contact_observed"]:
                reason = "drawer_handle_contact_not_observed"
            else:
                reason = "drawer_pull_stage_not_executed"
        return {
            "primitive": "physical_drawer_pull",
            "fixture": str(fixture_name),
            "drawer_joint": joint_name,
            "drawer_qpos_initial": initial_qpos,
            "drawer_qpos_final": final_qpos,
            "drawer_open_fraction_initial": initial_fraction,
            "drawer_open_fraction_final": final_fraction,
            "open_fraction_threshold": float(config.open_fraction_threshold),
            "open_direction_world": frame.open_direction_world.tolist(),
            "target_source": (
                "calibrated_pose_file"
                if config.grasp_pose_file is not None
                else "procedural_geometry"
            ),
            "grasp_pose": pose_metadata,
            "prehook_variant": config.prehook_variant,
            "prehook_rotation_offset_deg": list(config.prehook_rotation_offset_deg),
            "handle_side_offset": float(config.handle_side_offset),
            "handle_vertical_offset": float(config.handle_vertical_offset),
            "targets_world": {
                name: value[:3, 3].astype(float).tolist()
                for name, value in target_matrices.items()
            },
            "open_gripper": _compact_gripper(gripper_result),
            "close_gripper": _compact_gripper(close_gripper),
            "contacts_after_close_gripper": contacts_after_close,
            "stages": {name: _compact_stage(value) for name, value in stages.items()},
            "trace_summary": {
                "sample_count": len(trace),
                "contact_sample_count": contact_samples,
                "drawer_qpos_min": min((item["drawer_qpos"] for item in trace), default=initial_qpos),
                "drawer_qpos_max": max((item["drawer_qpos"] for item in trace), default=initial_qpos),
            },
            "checks": checks,
            "pass": bool(all(checks.values())),
            "reason": reason,
        }
    except Exception as exc:
        return {
            "primitive": "physical_drawer_pull",
            "fixture": str(fixture_name),
            "pass": False,
            "reason": "physical_drawer_pull_exception",
            "error": f"{type(exc).__name__}: {exc}",
        }


def close_drawer_physically(
    controller: Any,
    *,
    fixture_name: str,
    config: DrawerPushCloseConfig,
) -> dict[str, Any]:
    """Push the currently open drawer back through handle contact."""

    if not bool(getattr(controller.runtime_config, "execute", False)):
        return {
            "primitive": "physical_drawer_push_close",
            "fixture": str(fixture_name),
            "pass": False,
            "reason": "physical_drawer_push_close_requires_execute",
        }
    env = controller.env
    try:
        controller.refresh_obs()
        fixture = env.cabinet_for_slot(str(fixture_name))
        joint_name = str(fixture.door_joint_names[0])
        initial_qpos = _joint_qpos(env, joint_name)
        initial_fraction = float(fixture.open_fraction(env))
        locked_base = _locked_base_qpos(env)
        frame = resolve_drawer_motion_frame(
            fixture_yaw=float(fixture.rot),
            current_eef_rotation_world=matrix_from_site_pose(env, env.robots[0])[
                :3, :3
            ],
            hook_orientation_blend=0.0,
        )
        target_world = np.asarray(
            matrix_from_site_pose(env, env.robots[0]),
            dtype=float,
        )
        push_distance = initial_qpos + float(config.close_overshoot)
        target_world[:3, 3] -= frame.open_direction_world * push_distance
        trace: list[dict[str, Any]] = []

        def post_step(_obs: dict[str, Any]) -> dict[str, Any]:
            trace.append(
                {
                    "drawer_qpos": _joint_qpos(env, joint_name),
                    "contacts": _robot_fixture_contacts(env, str(fixture.name)),
                }
            )
            return _restore_locked_base(env, locked_base)

        target_base = np.linalg.inv(matrix_from_base_obs(controller.obs)) @ target_world
        stage = controller.stage_executor.run(
            pose_from_matrix(target_base),
            stage="ts02_drawer_push_close",
            gripper_action=-1.0,
            post_step_callback=post_step,
        )
        controller._sync_obs(controller.stage_executor.obs)
        final_qpos = _joint_qpos(env, joint_name)
        final_fraction = float(fixture.open_fraction(env))
        contacts = sum(bool(item["contacts"]) for item in trace)
        checks = {
            "push_executed": _stage_executed(stage),
            "drawer_closed": final_fraction <= float(config.closed_fraction_threshold),
            "robot_fixture_contact_observed": contacts > 0,
        }
        reason = None
        if not all(checks.values()):
            if not checks["drawer_closed"]:
                reason = "drawer_closed_fraction_above_threshold"
            elif not checks["robot_fixture_contact_observed"]:
                reason = "drawer_push_contact_not_observed"
            else:
                reason = "drawer_push_stage_not_executed"
        return {
            "primitive": "physical_drawer_push_close",
            "fixture": str(fixture_name),
            "drawer_joint": joint_name,
            "drawer_qpos_initial": initial_qpos,
            "drawer_qpos_final": final_qpos,
            "drawer_open_fraction_initial": initial_fraction,
            "drawer_open_fraction_final": final_fraction,
            "push_distance": push_distance,
            "stage": _compact_stage(stage),
            "trace_summary": {
                "sample_count": len(trace),
                "contact_sample_count": contacts,
            },
            "checks": checks,
            "pass": bool(all(checks.values())),
            "reason": reason,
        }
    except Exception as exc:
        return {
            "primitive": "physical_drawer_push_close",
            "fixture": str(fixture_name),
            "pass": False,
            "reason": "physical_drawer_push_close_exception",
            "error": f"{type(exc).__name__}: {exc}",
        }


def locked_base_post_step_callback(env: Any):
    locked = _locked_base_qpos(env)

    def callback(_obs: dict[str, Any]) -> dict[str, Any]:
        return _restore_locked_base(env, locked)

    return callback


def _pick_place_config(values: Mapping[str, Any]) -> PickPlaceConfig:
    settings = dict(values)
    defaults = PickPlaceConfig()
    allowed = {field.name for field in fields(PickPlaceConfig)}
    unknown = sorted(set(settings) - allowed)
    if unknown:
        raise ValueError("unknown TS02 pick_place settings: " + ", ".join(unknown))
    return replace(defaults, **settings)


def _runtime_config(
    *,
    execute: bool,
    runtime: Mapping[str, Any],
    planner: Mapping[str, Any],
    static_scene: Mapping[str, Any],
) -> ManipulationRuntimeConfig:
    values = dict(runtime)
    planner_config = MplibPlannerConfig(
        planning_time=float(planner.get("planning_time", 5.0)),
        time_step=float(planner.get("time_step", 0.1)),
        qpos_step=float(planner.get("qpos_step", 0.02)),
        rrt_range=float(planner.get("rrt_range", 0.1)),
    )
    return ManipulationRuntimeConfig(
        execute=bool(execute),
        max_joint_step=float(values.get("max_joint_step", 0.015)),
        settle_steps=int(values.get("settle_steps", 20)),
        position_threshold=float(values.get("position_threshold", 0.10)),
        orientation_threshold_deg=float(values.get("orientation_threshold_deg", 20.0)),
        qpos_threshold=float(values.get("qpos_threshold", 0.18)),
        open_steps=int(values.get("open_steps", 20)),
        close_steps=int(values.get("close_steps", 40)),
        release_steps=int(values.get("release_steps", 20)),
        release_settle_steps=int(values.get("release_settle_steps", 40)),
        planner_args=planner_config.planner_args(),
        static_scene_options=dict(static_scene),
    )


def _make_planner(script: Path, *, conda_env: str, timeout_s: float | None):
    def plan(payload: dict[str, Any]) -> dict[str, Any]:
        code, result = run_mplib_planner_subprocess(
            script_path=script,
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


def _drawer_pull_targets(
    handle_world: np.ndarray,
    frame: DrawerMotionFrame,
    config: DrawerPullConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any] | None]:
    metadata = None
    if config.grasp_pose_file is not None:
        targets, metadata = _load_drawer_grasp_targets(
            Path(config.grasp_pose_file), handle_world=handle_world
        )
        targets["pull"] = targets["hook"].copy()
        targets["pull"][:3, 3] += frame.open_direction_world * float(config.pull_distance)
    else:
        positions = drawer_pull_target_positions(
            handle_world,
            frame,
            prehook_distance=float(config.prehook_distance),
            hook_offset=float(config.hook_offset),
            pull_distance=float(config.pull_distance),
        )
        targets = {
            name: pose_matrix(position, frame.hook_rotation_world)
            for name, position in positions.items()
        }
    return vary_drawer_grasp_targets(
        targets,
        frame,
        handle_side_offset=float(config.handle_side_offset),
        handle_vertical_offset=float(config.handle_vertical_offset),
        prehook_rotation_offset_deg=config.prehook_rotation_offset_deg,
    ), metadata


def _load_drawer_grasp_targets(
    path: Path,
    *,
    handle_world: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Translate the center-drawer calibration to the selected handle."""

    pose_path = Path(path)
    payload = json.loads(pose_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("config_kind") != "anchor_pose":
        raise ValueError(f"TS02 drawer grasp pose must be an anchor_pose: {pose_path}")
    task = payload.get("task") or {}
    if not isinstance(task, Mapping) or str(task.get("env_name")) != ENV_NAME:
        raise ValueError(f"TS02 drawer grasp pose declares the wrong environment: {pose_path}")
    anchor = payload.get("anchor") or {}
    if not isinstance(anchor, Mapping):
        raise ValueError(f"TS02 drawer grasp pose has no anchor: {pose_path}")
    selector = str(anchor.get("selector", ""))
    if not selector.startswith("geom:ts02_") or not selector.endswith(
        "_fixture_drawer_handle"
    ):
        raise ValueError(f"TS02 drawer grasp pose has an invalid handle anchor: {selector!r}")
    anchor_world = np.asarray(anchor.get("world"), dtype=float).reshape(3)
    translation = np.asarray(handle_world, dtype=float).reshape(3) - anchor_world
    stages = payload.get("stages") or {}
    if not isinstance(stages, Mapping):
        raise ValueError(f"TS02 drawer grasp pose has no stages: {pose_path}")
    targets: dict[str, np.ndarray] = {}
    for stage_name in ("prehook", "hook"):
        stage = stages.get(stage_name) or {}
        world = stage.get("world") if isinstance(stage, Mapping) else None
        matrix = (
            None if not isinstance(world, Mapping) else world.get("matrix")
        )
        value = np.asarray(matrix, dtype=float)
        if value.shape != (4, 4) or not np.all(np.isfinite(value)):
            raise ValueError(
                f"TS02 drawer grasp stage {stage_name!r} has no finite 4x4 matrix"
            )
        if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise ValueError(
                f"TS02 drawer grasp stage {stage_name!r} has an invalid matrix"
            )
        rotation = value[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-7
        ):
            raise ValueError(
                f"TS02 drawer grasp stage {stage_name!r} rotation is not orthonormal"
            )
        target = value.copy()
        target[:3, 3] += translation
        targets[stage_name] = target
    return targets, {
        "path": str(pose_path),
        "pose_name": payload.get("pose_name"),
        "template_anchor_selector": selector,
        "template_anchor_world": anchor_world.astype(float).tolist(),
        "selected_handle_world": np.asarray(handle_world, dtype=float).tolist(),
        "template_translation_world": translation.astype(float).tolist(),
        "stage_order": ["prehook", "hook", "pull"],
    }


def _joint_qpos(env: Any, joint_name: str) -> float:
    return float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(joint_name)])


def _locked_base_qpos(env: Any) -> dict[str, float]:
    return {
        name: _joint_qpos(env, name)
        for name in (
            "mobilebase0_joint_mobile_forward",
            "mobilebase0_joint_mobile_side",
            "mobilebase0_joint_mobile_yaw",
        )
    }


def _restore_locked_base(env: Any, values: Mapping[str, float]) -> dict[str, Any]:
    for joint_name, value in values.items():
        env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(joint_name)] = float(value)
        env.sim.data.qvel[env.sim.model.get_joint_qvel_addr(joint_name)] = 0.0
    env.sim.forward()
    return env._get_observations(force_update=True)


def _robot_fixture_contacts(env: Any, fixture_name: str) -> list[dict[str, Any]]:
    contacts: list[dict[str, Any]] = []
    for contact in env.sim.data.contact[: env.sim.data.ncon]:
        first = env.sim.model.geom_id2name(contact.geom1) or ""
        second = env.sim.model.geom_id2name(contact.geom2) or ""
        if fixture_name not in first and fixture_name not in second:
            continue
        if "gripper0" not in first and "gripper0" not in second:
            continue
        contacts.append(
            {"geom1": first, "geom2": second, "distance": float(contact.dist)}
        )
    return contacts


def _compact_stage(summary: Mapping[str, Any]) -> dict[str, Any]:
    execution = summary.get("execution") or {}
    plan = summary.get("plan") or {}
    return {
        "pass": bool(summary.get("pass", False)),
        "plan_status": plan.get("status"),
        "failure_reason": summary.get("failure_reason"),
        "final_position_error_m": execution.get("final_position_error_m"),
        "final_orientation_error_deg": execution.get("final_orientation_error_deg"),
    }


def _compact_gripper(summary: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {
        "steps": summary.get("steps"),
        "gripper_action": summary.get("gripper_action"),
        "arm_qpos_error": summary.get("arm_qpos_error"),
    }


def _stage_executed(summary: Mapping[str, Any]) -> bool:
    return (
        bool(summary.get("pass", False))
        and (summary.get("plan") or {}).get("status") == "Success"
        and bool(summary.get("execution"))
    )


def _section(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"TS02 manipulation.{name} must be a mapping")
    return value


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


__all__ = [
    "DrawerPullConfig",
    "DrawerPushCloseConfig",
    "ManipulationAssembly",
    "build_manipulation_assembly",
    "close_drawer_physically",
    "locked_base_post_step_callback",
    "open_drawer_physically",
]
