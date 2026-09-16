"""纯旋转primitive - 原地旋转，不产生xy漂移"""

from typing import Any
import numpy as np
from bench.execution.navigation.velocity_profile import (
    AxialVelocityProfile,
    BaseVelocityLimits,
    PlanarVelocityProfile,
    resolve_control_dt,
)
from bench.execution.planning.nav_algo import normalize_angle


def rotate_in_place(
    env: Any,
    obs: dict[str, Any],
    *,
    target_yaw: float,
    arm_qpos: np.ndarray,
    gripper_action: float,
    yaw_tolerance: float = 0.06,
    yaw_gain: float = 2.0,
    max_steps: int = 350,
    holding_object_name: str | None = None,
    holding_drop_tolerance: float = 0.04,
    velocity_limits: BaseVelocityLimits | None = None,
    rotation_direction: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """原地旋转到目标yaw，完全不执行xy移动。

    这是一个专门的旋转primitive，避免了move_base_to_pose中xy/yaw耦合导致的漂移。
    """
    from bench.execution.navigation.base_motion import (
        base_yaw,
        step_base_and_hold_arm,
    )

    initial_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
    initial_yaw = float(base_yaw(obs))
    initial_held_pos = None
    if holding_object_name:
        obj = env.sim.model.body_name2id(holding_object_name)
        initial_held_pos = np.array(
            env.sim.data.body_xpos[obj],
            dtype=float,
        )

    steps = 0
    max_drift = 0.0
    yaw_profile = (
        AxialVelocityProfile(velocity_limits.yaw_params(), resolve_control_dt(env))
        if velocity_limits is not None and velocity_limits.enabled
        else None
    )

    for _ in range(max_steps):
        current_yaw = base_yaw(obs)
        yaw_error = normalize_angle(target_yaw - current_yaw)

        # 检查是否完成
        if abs(yaw_error) <= yaw_tolerance:
            break

        # Some task motions demonstrate a fixed turn direction. In that case
        # retain the requested signed branch even when it is slightly longer
        # than pi (for example, a clockwise -184 degree half-turn).
        if rotation_direction is not None:
            direction = int(rotation_direction)
            if direction not in (-1, 1):
                raise ValueError("rotation_direction must be -1, 1, or None")
            if direction < 0 and yaw_error > 0.0:
                yaw_error -= 2.0 * np.pi
            elif direction > 0 and yaw_error < 0.0:
                yaw_error += 2.0 * np.pi

        # 仅旋转，xy动作为0
        split = env.robots[0].composite_controller._action_split_indexes
        start, end = split["base"]
        base_action = np.zeros(end - start, dtype=float)

        # 只控制yaw
        if base_action.shape[0] >= 3:
            if yaw_profile is not None:
                base_action[2] = yaw_profile.step_action_signed(
                    yaw_error, tolerance=yaw_tolerance
                )
            else:
                base_action[2] = np.clip(yaw_error * yaw_gain, -1.0, 1.0)

        # 执行动作
        obs = step_base_and_hold_arm(
            env,
            arm_qpos=arm_qpos,
            base_action=base_action,
            base_mode=1.0,
            gripper_action=gripper_action,
        )
        steps += 1

        # 记录实际的xy漂移
        current_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
        drift = float(np.linalg.norm(current_xy - initial_xy))
        max_drift = max(max_drift, drift)

    # 最终状态
    final_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
    final_yaw = float(base_yaw(obs))
    final_yaw_error = normalize_angle(target_yaw - final_yaw)
    final_drift = float(np.linalg.norm(final_xy - initial_xy))

    # 检查holding object
    holding_motion = {"enabled": False}
    if holding_object_name and initial_held_pos is not None:
        obj = env.sim.model.body_name2id(holding_object_name)
        final_held_pos = np.array(env.sim.data.body_xpos[obj], dtype=float)
        held_delta = float(np.linalg.norm(final_held_pos - initial_held_pos))
        holding_motion = {
            "enabled": True,
            "object_name": holding_object_name,
            "initial_pos": initial_held_pos.tolist(),
            "final_pos": final_held_pos.tolist(),
            "delta": held_delta,
            "drop_ok": held_delta <= holding_drop_tolerance,
            "drop_tolerance": holding_drop_tolerance,
        }

    success = (
        abs(final_yaw_error) <= yaw_tolerance
        and (not holding_motion["enabled"] or holding_motion["drop_ok"])
    )

    return obs, {
        "primitive": "rotate_in_place",
        "pass": success,
        "initial_xy": initial_xy.tolist(),
        "initial_yaw": initial_yaw,
        "target_yaw": target_yaw,
        "final_xy": final_xy.tolist(),
        "final_yaw": final_yaw,
        "final_yaw_error": final_yaw_error,
        "drift": final_drift,
        "max_drift": max_drift,
        "steps": steps,
        "rotation_direction": rotation_direction,
        "holding_motion": holding_motion,
        "checks": {
            "yaw": abs(final_yaw_error) <= yaw_tolerance,
            "holding_motion": not holding_motion["enabled"] or holding_motion["drop_ok"],
        },
    }


def move_straight(
    env: Any,
    obs: dict[str, Any],
    *,
    target_xy: np.ndarray,
    arm_qpos: np.ndarray,
    gripper_action: float,
    tolerance: float = 0.06,
    xy_gain: float = 3.0,
    max_steps: int = 500,
    action_frame: str = "world",
    translation_action_sign: float = 1.0,
    holding_object_name: str | None = None,
    holding_drop_tolerance: float = 0.04,
    maintain_yaw: bool = True,
    velocity_limits: BaseVelocityLimits | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """直线移动到目标位置，可选保持当前yaw不变。

    这是一个专门的直线移动primitive，如果maintain_yaw=True，会主动保持初始yaw。
    """
    from bench.execution.navigation.base_motion import (
        base_yaw,
        step_base_and_hold_arm,
        world_xy_to_controller_action,
        world_vec_to_base,
    )

    target_xy = np.array(target_xy, dtype=float)
    initial_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
    initial_yaw = float(base_yaw(obs))
    target_yaw = initial_yaw if maintain_yaw else None

    initial_held_pos = None
    if holding_object_name:
        obj = env.sim.model.body_name2id(holding_object_name)
        initial_held_pos = np.array(env.sim.data.body_xpos[obj], dtype=float)

    steps = 0
    min_dist = float("inf")
    xy_profile = (
        PlanarVelocityProfile(velocity_limits.xy_params(), resolve_control_dt(env))
        if velocity_limits is not None and velocity_limits.enabled
        else None
    )

    for _ in range(max_steps):
        current_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
        current_yaw = base_yaw(obs)

        err_xy = target_xy - current_xy
        dist = float(np.linalg.norm(err_xy))
        min_dist = min(min_dist, dist)

        yaw_error = 0.0
        if maintain_yaw:
            yaw_error = normalize_angle(target_yaw - current_yaw)

        # 检查是否完成
        xy_done = dist <= tolerance
        yaw_done = not maintain_yaw or abs(yaw_error) <= 0.15  # 宽松的yaw容忍度

        if xy_done and yaw_done:
            break

        # 构造动作
        split = env.robots[0].composite_controller._action_split_indexes
        start, end = split["base"]
        base_action = np.zeros(end - start, dtype=float)

        # xy控制
        if not xy_done:
            if action_frame == "base":
                translate_error = world_vec_to_base(
                    obs,
                    np.array([err_xy[0], err_xy[1], 0.0], dtype=float),
                )[:2]
            elif action_frame == "world":
                translate_error = world_xy_to_controller_action(env, err_xy)
            else:
                raise ValueError(f"unsupported action_frame: {action_frame}")

            if xy_profile is not None:
                translate_action = xy_profile.step_action(
                    translate_error, tolerance=tolerance
                )
            else:
                translate_action = np.clip(translate_error * xy_gain, -1.0, 1.0)
            base_action[:2] = translate_action * translation_action_sign

        # yaw控制（如果需要保持yaw）
        # TODO: 姿态保持的小幅 yaw 修正暂保留 previous 行为（幅度小，影响有限）
        if maintain_yaw and not yaw_done and base_action.shape[0] >= 3:
            base_action[2] = np.clip(yaw_error * 1.5, -1.0, 1.0)

        # 执行
        obs = step_base_and_hold_arm(
            env,
            arm_qpos=arm_qpos,
            base_action=base_action,
            base_mode=1.0,
            gripper_action=gripper_action,
        )
        steps += 1

    # 最终状态
    final_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
    final_yaw = float(base_yaw(obs))
    final_dist = float(np.linalg.norm(target_xy - final_xy))

    # holding check
    holding_motion = {"enabled": False}
    if holding_object_name and initial_held_pos is not None:
        obj = env.sim.model.body_name2id(holding_object_name)
        final_held_pos = np.array(env.sim.data.body_xpos[obj], dtype=float)
        held_delta = float(np.linalg.norm(final_held_pos - initial_held_pos))
        holding_motion = {
            "enabled": True,
            "object_name": holding_object_name,
            "delta": held_delta,
            "drop_ok": held_delta <= holding_drop_tolerance,
        }

    success = (
        final_dist <= tolerance
        and (not holding_motion["enabled"] or holding_motion["drop_ok"])
    )

    return obs, {
        "primitive": "move_straight",
        "pass": success,
        "initial_xy": initial_xy.tolist(),
        "target_xy": target_xy.tolist(),
        "final_xy": final_xy.tolist(),
        "final_dist": final_dist,
        "min_dist": min_dist,
        "steps": steps,
        "maintain_yaw": maintain_yaw,
        "initial_yaw": initial_yaw,
        "final_yaw": final_yaw,
        "holding_motion": holding_motion,
        "checks": {
            "xy": final_dist <= tolerance,
            "holding_motion": not holding_motion["enabled"] or holding_motion["drop_ok"],
        },
    }
