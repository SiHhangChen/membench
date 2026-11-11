"""Physical TS02 completion-button press primitive."""

from __future__ import annotations

from typing import Any

import numpy as np

from bench.execution.planning import matrix_from_base_obs, pose_from_matrix
from bench.tasks.ts02.execution.manipulation import (
    locked_base_post_step_callback,
)


def press_completion_button_physically(
    controller: Any,
    *,
    button_name: str | None = None,
    prepress_z_offset: float,
    press_z_offset: float,
    hold_steps: int,
    gripper_action: float,
    near_miss_position_tolerance: float,
    press_depth_threshold: float,
    require_physical_button_motion: bool = True,
) -> dict[str, Any]:
    """Approach, depress, hold, retreat, and return the arm to its regular pose."""

    if not bool(getattr(controller.runtime_config, "execute", False)):
        return {
            "primitive": "physical_completion_button_press",
            "dry_run": True,
            "semantic_event_deferred": True,
            "checks": {"deferred_to_execute": True},
            "pass": True,
        }
    if int(hold_steps) < 1:
        return _failure("button_hold_steps_must_be_positive")

    env = controller.env
    try:
        controller.refresh_obs()
        initial_button = np.asarray(
            _button_query(env, "completion_button_position", button_name),
            dtype=float,
        ).reshape(3)
        initial_depth = float(
            _button_query(env, "completion_button_press_depth", button_name)
        )
        prepress_world = _button_pose_world(
            controller.obs,
            initial_button,
            z_offset=float(prepress_z_offset),
        )
        press_world = _button_pose_world(
            controller.obs,
            initial_button,
            z_offset=float(press_z_offset),
        )
        trace: list[dict[str, float]] = []
        lock_base = locked_base_post_step_callback(env)

        def post_step(obs: dict[str, Any]) -> dict[str, Any]:
            trace.append(
                {
                    "button_press_depth_m": float(
                        _button_query(
                            env,
                            "completion_button_press_depth",
                            button_name,
                        )
                    )
                }
            )
            return lock_base(obs)

        prepress = _run_world_stage(
            controller,
            prepress_world,
            stage="ts02_completion_button_prepress",
            gripper_action=float(gripper_action),
            post_step_callback=post_step,
        )
        press = (
            _run_world_stage(
                controller,
                press_world,
                stage="ts02_completion_button_press",
                gripper_action=float(gripper_action),
                post_step_callback=post_step,
                record_trajectory=True,
            )
            if _stage_executed(prepress)
            else _skipped("ts02_completion_button_press", "prepress_failed")
        )
        press_eef = np.asarray(
            controller.stage_executor.current_eef_site_world(), dtype=float
        ).reshape(4, 4)
        endpoint_error = float(
            np.linalg.norm(press_eef[:3, 3] - press_world[:3, 3])
        )
        endpoint_reached = bool(
            endpoint_error <= float(near_miss_position_tolerance)
        )
        depth_after_press = float(
            _button_query(env, "completion_button_press_depth", button_name)
        )
        trace.append({"button_press_depth_m": depth_after_press})
        hold = (
            controller.driver.run_gripper(
                arm_qpos=controller.driver.current_arm_qpos(),
                gripper_action=float(gripper_action),
                steps=int(hold_steps),
                post_step_callback=post_step,
            )
            if _stage_executed(press)
            else _skipped("ts02_completion_button_hold", "press_failed")
        )
        if isinstance(hold, dict) and "obs" in hold:
            controller._sync_obs(hold["obs"])
        depth_after_hold = float(
            _button_query(env, "completion_button_press_depth", button_name)
        )
        trace.append({"button_press_depth_m": depth_after_hold})
        retreat = (
            _run_world_stage(
                controller,
                prepress_world,
                stage="ts02_completion_button_retreat",
                gripper_action=float(gripper_action),
                post_step_callback=post_step,
            )
            if _gripper_command_pass(hold, required_steps=int(hold_steps))
            else _skipped("ts02_completion_button_retreat", "button_hold_failed")
        )
        arm_return = (
            controller.return_arm_to_regular_pose(
                stage="ts02_post_button_return_to_regular_pose",
                gripper_action=-1.0,
                post_step_callback=post_step,
            )
            if _stage_executed(retreat)
            else _skipped(
                "ts02_post_button_return_to_regular_pose", "button_retreat_failed"
            )
        )
        max_depth = max(
            [initial_depth, depth_after_press, depth_after_hold]
            + [float(item["button_press_depth_m"]) for item in trace]
        )
        physical_motion = bool(max_depth >= float(press_depth_threshold))
        checks = {
            "prepress_executed": _stage_executed(prepress),
            "press_executed": _stage_executed(press),
            "press_endpoint": endpoint_reached,
            "button_hold": _gripper_command_pass(
                hold,
                required_steps=int(hold_steps),
            ),
            "physical_button_motion": physical_motion,
            "retreat_executed": _stage_executed(retreat),
            "arm_returned": _stage_executed(arm_return),
        }
        if not bool(require_physical_button_motion):
            checks["physical_button_motion"] = True
        passed = bool(all(checks.values()))
        return {
            "primitive": "physical_completion_button_press",
            "button_name": button_name,
            "button_position_world": initial_button.astype(float).tolist(),
            "prepress_target_world": prepress_world.astype(float).tolist(),
            "press_target_world": press_world.astype(float).tolist(),
            "press_endpoint_world": press_eef[:3, 3].astype(float).tolist(),
            "press_endpoint_error_m": endpoint_error,
            "near_miss_position_tolerance": float(
                near_miss_position_tolerance
            ),
            "button_press_depth_initial_m": initial_depth,
            "button_press_depth_after_press_m": depth_after_press,
            "button_press_depth_after_hold_m": depth_after_hold,
            "button_press_depth_max_m": max_depth,
            "button_press_depth_threshold_m": float(press_depth_threshold),
            "physical_button_motion_observed": physical_motion,
            "trace_summary": {"sample_count": len(trace)},
            "stages": {
                "prepress": _compact(prepress),
                "press": _compact(press),
                "hold": _compact(hold),
                "retreat": _compact(retreat),
                "arm_return": _compact(arm_return),
            },
            "checks": checks,
            "pass": passed,
            "reason": None if passed else "completion_button_press_failed",
        }
    except Exception as exc:
        return _failure(
            "completion_button_press_exception",
            error=f"{type(exc).__name__}: {exc}",
        )


def _button_pose_world(
    obs: dict[str, Any],
    button_position: np.ndarray,
    *,
    z_offset: float,
) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = matrix_from_base_obs(dict(obs))[:3, :3] @ np.diag(
        [1.0, -1.0, -1.0]
    )
    pose[:3, 3] = np.asarray(button_position, dtype=float) + np.asarray(
        [0.0, 0.0, float(z_offset)], dtype=float
    )
    return pose


def _button_query(env: Any, method_name: str, button_name: str | None) -> Any:
    method = getattr(env, method_name)
    return method() if button_name is None else method(str(button_name))


def _run_world_stage(
    controller: Any,
    pose_world: np.ndarray,
    *,
    stage: str,
    gripper_action: float,
    post_step_callback: Any,
    record_trajectory: bool = False,
) -> dict[str, Any]:
    target_base = np.linalg.inv(matrix_from_base_obs(controller.obs)) @ pose_world
    result = controller.stage_executor.run(
        pose_from_matrix(target_base),
        stage=stage,
        gripper_action=float(gripper_action),
        post_step_callback=post_step_callback,
        record_trajectory=bool(record_trajectory),
    )
    controller._sync_obs(controller.stage_executor.obs)
    return result


def _stage_executed(result: dict[str, Any]) -> bool:
    return bool(result.get("pass", False)) and not bool(result.get("dry_run", False))


def _gripper_command_pass(result: dict[str, Any], *, required_steps: int) -> bool:
    return bool(
        not result.get("skipped", False)
        and int(result.get("steps", 0)) >= int(required_steps)
    )


def _skipped(stage: str, reason: str) -> dict[str, Any]:
    return {"stage": stage, "skipped": True, "pass": False, "reason": reason}


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"obs", "executed_trajectory", "planned_trajectory"}
    }


def _failure(reason: str, **details: Any) -> dict[str, Any]:
    return {
        "primitive": "physical_completion_button_press",
        "pass": False,
        "reason": reason,
        **details,
    }


__all__ = ["press_completion_button_physically"]
