"""Thin WA-05 adapters: physical evidence first, semantic event second."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from bench.execution.manipulation.object_interactions import get_object_hold_status, is_object_released
from bench.execution.navigation.base_motion import (
    base_yaw,
    qpos_for_controller,
    step_base_and_hold_arm,
    world_xy_to_controller_action,
)
from bench.execution.navigation.velocity_profile import (
    XY_ACTION_SPEED,
    AxialProfileParams,
    PlanarVelocityProfile,
    resolve_control_dt,
)
from bench.execution.planning import matrix_from_base_obs, pose_from_matrix, rotation_error_deg
from bench.execution.planning.nav_algo import normalize_angle
from bench.tasks.wa05.domain.contracts import CUP_IDS, SERVICE_SLOTS
from bench.tasks.wa05.execution.water_operations import (
    fill_held_cup_under_faucet,
    place_cup_on_service_slot,
)
from bench.tasks.wa05.execution.answer_button import press_temperature_answer_button


def build_stage_handlers(
    controller: Any,
    *,
    handler_settings: Mapping[str, Any] | None = None,
    sink_exit_target_provider: Callable[[], Any] | None = None,
) -> dict[str, Callable[..., dict[str, Any]]]:
    """Map workflow operation names to physical execution and verified events."""

    defaults = dict(handler_settings or {})

    def answer_button(target: str, **kwargs: Any) -> dict[str, Any]:
        del target
        return press_temperature_answer_button(controller, **{**defaults, **dict(kwargs)})

    def pick_upright_cup(target: str, **kwargs: Any) -> dict[str, Any]:
        cup_id = str(target)
        result = controller.pick(cup_id, **kwargs)
        if not bool(controller.runtime_config.execute):
            result["cup_orientation_validation"] = {"deferred_to_execute": True}
            return result

        env = _refresh(controller)
        cup_rotation = np.asarray(
            env.sim.data.body_xmat[env.obj_body_id[cup_id]], dtype=float
        ).reshape(3, 3)
        upright_z_axis = float(cup_rotation[2, 2])
        upright_threshold = float(
            np.cos(np.deg2rad(float(env.cup_upright_threshold_deg)))
        )
        upright = bool(upright_z_axis >= upright_threshold)
        checks = dict(result.get("checks") or {})
        checks["cup_upright_after_pick"] = upright
        result["checks"] = checks
        result["cup_orientation_validation"] = {
            "upright_z_axis": upright_z_axis,
            "upright_threshold": upright_threshold,
            "threshold_deg": float(env.cup_upright_threshold_deg),
            "pass": upright,
        }
        result["pass"] = bool(result.get("pass", False) and upright)
        if not result["pass"] and upright is False:
            result["reason"] = "cup_not_upright_after_pick"
        return result

    def fill(target: str, **kwargs: Any) -> dict[str, Any]:
        values = {**defaults, **dict(kwargs)}
        result = fill_held_cup_under_faucet(
            controller,
            cup_id=str(target),
            **values,
        )
        if bool(result.get("dry_run", False)):
            result["semantic_fill"] = {"deferred_to_execute": True}
            return result
        temperature = str(values.get("temperature", ""))
        applied = False
        if bool(result.get("pass", False)):
            applied = bool(controller.env.record_verified_fill(str(target), temperature))
        checks = dict(result.get("checks") or {})
        checks["verified_fill_event_committed"] = bool(applied)
        result["checks"] = checks
        result["semantic_fill"] = {
            "cup_id": str(target),
            "temperature": temperature,
            "applied": bool(applied),
            "event": "record_verified_fill",
            "state_after": {
                "filled_temperature": getattr(controller.env._semantic_state, "filled_temperature", None),
                "absorbing_failure": bool(getattr(controller.env._semantic_state, "absorbing_failure", False)),
                "failure_code": getattr(controller.env._semantic_state, "failure_code", None),
            },
        }
        result["pass"] = bool(result.get("pass", False) and applied)
        if not result["pass"]:
            result["reason"] = result.get("reason") or "verified_fill_event_rejected"
            restore_stream = getattr(
                controller.env, "set_water_stream_intercept", None
            )
            if callable(restore_stream):
                restore_stream(None)
        return result

    def place(target: str, **kwargs: Any) -> dict[str, Any]:
        values = dict(kwargs)
        slot = str(values.pop("service_slot", ""))
        return place_cup_on_service_slot(
            controller,
            cup_id=str(target),
            slot=slot,
            **values,
        )

    def return_holding_cup(target: str, **kwargs: Any) -> dict[str, Any]:
        return return_arm_holding_cup(
            controller,
            cup_id=str(target),
            **kwargs,
        )

    def lift_after_sink_exit(target: str, **kwargs: Any) -> dict[str, Any]:
        values = dict(kwargs)
        waypoint_xy = None
        if sink_exit_target_provider is not None and bool(
            controller.runtime_config.execute
        ):
            target_pose = sink_exit_target_provider()
            waypoint_xy = np.asarray(target_pose.position[:2], dtype=float)
            env = _refresh(controller)
            base_world = np.asarray(controller.obs["robot0_base_pos"], dtype=float)
            target_world = base_world.copy()
            target_world[:2] = waypoint_xy
            local_delta = (
                _counter_local_position(env, target_world)
                - _counter_local_position(env, base_world)
            )
            if not np.all(np.isfinite(local_delta)) or local_delta[0] >= -1e-6:
                return {
                    "primitive": "lift_held_cup_after_sink_exit",
                    "pass": False,
                    "reason": "sampled_service_target_not_beyond_sink_exit",
                }
            values["route_local_y_per_x"] = float(local_delta[1] / local_delta[0])
        result = lift_held_cup_after_sink_exit(
            controller,
            cup_id=str(target),
            **values,
        )
        if waypoint_xy is not None:
            result["sampled_service_target_xy"] = waypoint_xy.tolist()
        return result

    def record_staging(target: str, **kwargs: Any) -> dict[str, Any]:
        cup_id = str(target)
        slot = str(kwargs.get("service_slot", ""))
        if cup_id not in CUP_IDS or slot not in SERVICE_SLOTS:
            return {
                "primitive": "record_verified_staging",
                "cup_id": cup_id,
                "slot": slot,
                "pass": False,
                "reason": "invalid_cup_or_service_slot",
            }
        if not bool(controller.runtime_config.execute):
            return _dry_result("record_verified_staging", cup_id=cup_id, slot=slot)
        env = _refresh(controller)
        release_distance = float(kwargs.get("release_distance", 0.12))
        cup_position = np.asarray(env.object_position(cup_id), dtype=float).reshape(3)
        slot_position = np.asarray(env.service_slot_position(slot), dtype=float).reshape(3)
        slot_xy_error = float(np.linalg.norm(cup_position[:2] - slot_position[:2]))
        hold_status = get_object_hold_status(
            env,
            cup_id,
            max_end_effector_distance=release_distance,
        )
        try:
            from robocasa.utils import object_utils as OU

            counter_contact = bool(OU.check_obj_fixture_contact(env, cup_id, env.counter))
        except Exception:
            counter_contact = False
        # After the release/retreat sequence, RoboSuite can briefly miss the
        # contact pair even though the cup bottom is resting at the planned
        # counter height. Treat that independently measurable support pose as
        # valid staging evidence, while retaining the strict slot/upright/
        # release checks below.
        try:
            cup_body = int(env.obj_body_id[cup_id])
            cup_pos = np.asarray(env.sim.data.body_xpos[cup_body], dtype=float)
            cup_quat = np.asarray(env.sim.data.body_xquat[cup_body], dtype=float)
            cup_quat_xyzw = cup_quat[[1, 2, 3, 0]]
            cup_bbox = np.asarray(
                env.objects[cup_id].get_bbox_points(
                    trans=cup_pos, rot=cup_quat_xyzw
                ),
                dtype=float,
            )
            cup_bottom_z = float(np.min(cup_bbox[:, 2]))
            surface_support = bool(abs(cup_bottom_z - slot_position[2]) <= 0.012)
        except Exception:
            surface_support = False
        counter_contact = bool(counter_contact or surface_support)
        cup_rotation = np.asarray(
            env.sim.data.body_xmat[env.obj_body_id[cup_id]], dtype=float
        ).reshape(3, 3)
        upright_z_axis = float(cup_rotation[2, 2])
        gripper = env.robots[0].gripper["right"]
        gripper_joint_names = [str(name) for name in gripper.joints]
        gripper_joint_qpos = [
            float(env.sim.data.get_joint_qpos(name)) for name in gripper_joint_names
        ]
        gripper_joint_ranges = []
        for name in gripper_joint_names:
            joint_id = int(env.sim.model.joint_name2id(name))
            gripper_joint_ranges.append(
                np.asarray(env.sim.model.jnt_range[joint_id], dtype=float).tolist()
            )
        # Keep the staging evidence based on independently sampled state. The
        # composite helper performs another contact query and can disagree
        # after the gripper retreat has already released the cup.
        cup_at_slot = bool(
            slot_xy_error <= float(env.service_threshold)
            and env.cup_is_upright(cup_id)
        )
        cup_center_inside_region = bool(env.cup_is_in_service_region(cup_id, slot))
        cup_released = bool(
            is_object_released(env, cup_id, distance=release_distance)
        )
        checks = {
            "cup_filled_after_verified_fill": bool(env.cup_filled.get(cup_id, False)),
            "cup_at_planned_service_slot": cup_at_slot,
            "cup_center_inside_service_region": cup_center_inside_region,
            "counter_contact": counter_contact,
            "cup_released": cup_released,
        }
        applied = False
        if all(checks.values()):
            # The checks above are the authoritative physical evidence for
            # this operation. Re-querying contact after the arm has retreated
            # can be transient and would otherwise reject a valid release.
            applied = bool(env.record_verified_staging(cup_id, slot, currently_ready=True))
        checks["verified_staging_event_committed"] = applied
        return {
            "primitive": "record_verified_staging",
            "cup_id": cup_id,
            "slot": slot,
            "release_distance": release_distance,
            "physical_diagnostics": {
                "cup_position_world": cup_position.astype(float).tolist(),
                "slot_position_world": slot_position.astype(float).tolist(),
                "service_region": env.service_slot_region(slot),
                "slot_xy_error_m": slot_xy_error,
                "service_threshold_m": float(env.service_threshold),
                "counter_contact": counter_contact,
                "surface_support": surface_support,
                "upright_z_axis": upright_z_axis,
                "hold_status": hold_status,
                "gripper_joint_names": gripper_joint_names,
                "gripper_joint_qpos": gripper_joint_qpos,
                "gripper_joint_ranges": gripper_joint_ranges,
                "gripper_current_action": np.asarray(
                    gripper.current_action, dtype=float
                ).tolist(),
                "release_separation_assist": dict(
                    getattr(env, "_wa05_release_separation_trace", {}) or {}
                ),
            },
            "checks": checks,
            "semantic_staging": {"applied": applied, "event": "record_verified_staging"},
            "semantic_state": {
                "filled_temperature": getattr(env._semantic_state, "filled_temperature", None),
                "staged_before_answer": bool(getattr(env._semantic_state, "staged_before_answer", False)),
                "pressed_temperature": getattr(env._semantic_state, "pressed_temperature", None),
                "absorbing_failure": bool(getattr(env._semantic_state, "absorbing_failure", False)),
                "failure_code": getattr(env._semantic_state, "failure_code", None),
            },
            "pass": bool(all(checks.values())),
            "reason": None if all(checks.values()) else "physical_service_staging_not_verified",
        }

    return {
        "pick_upright_cup": pick_upright_cup,
        "return_arm_holding_cup": return_holding_cup,
        "lift_held_cup_after_sink_exit": lift_after_sink_exit,
        "fill_held_under_faucet": fill,
        "place_on_service_slot": place,
        "record_verified_staging": record_staging,
        "press_temperature_answer_button": answer_button,
    }


def return_arm_holding_cup(
    controller: Any,
    *,
    cup_id: str,
    retract_distance_base_x: float = 0.10,
    minimum_base_x: float = 0.35,
    lift_offset: float = 0.02,
    settle_steps: int = 2,
    gripper_action: float = 1.0,
    orientation_tolerance_deg: float = 15.0,
    max_end_effector_distance: float = 0.18,
    stage: str = "return_arm_holding_cup",
    **_: Any,
) -> dict[str, Any]:
    """Retract a held cup in Cartesian space without changing its orientation."""

    if cup_id not in CUP_IDS:
        raise ValueError(f"invalid WA-05 cup id: {cup_id!r}")
    retract_distance = float(retract_distance_base_x)
    minimum_x = float(minimum_base_x)
    lift = float(lift_offset)
    gripper = float(gripper_action)
    rotation_tolerance = float(orientation_tolerance_deg)
    hold_distance = float(max_end_effector_distance)
    numeric_values = (
        retract_distance,
        minimum_x,
        lift,
        gripper,
        rotation_tolerance,
        hold_distance,
    )
    if not all(np.isfinite(value) for value in numeric_values):
        raise ValueError("WA-05 holding-cup return parameters must be finite")
    if retract_distance < 0.0:
        raise ValueError("retract_distance_base_x must be non-negative")
    if gripper <= 0.0:
        raise ValueError("return_arm_holding_cup requires a closing gripper command")
    if int(settle_steps) < 0:
        raise ValueError("settle_steps must be non-negative")
    if rotation_tolerance <= 0.0 or hold_distance <= 0.0:
        raise ValueError("holding-cup validation tolerances must be positive")

    controller.refresh_obs()
    current_world = np.asarray(
        controller.stage_executor.current_eef_site_world(), dtype=float
    ).reshape(4, 4)
    base_world = matrix_from_base_obs(dict(controller.obs))
    current_base = np.linalg.inv(base_world) @ current_world
    target_base = np.asarray(current_base, dtype=float).copy()
    requested_x = float(current_base[0, 3]) - retract_distance
    target_base[0, 3] = max(requested_x, minimum_x)
    target_base[2, 3] += lift
    target_pose = pose_from_matrix(target_base)
    execute = bool(controller.runtime_config.execute)
    env = getattr(controller, "env", None)
    stream_trace = dict(
        getattr(env, "_water_stream_intercept_trace", {}) or {}
    )
    stream_handoff = bool(
        execute and stream_trace.get("intercepted", False)
    )
    callback_factory = getattr(env, "water_stream_intercept_callback", None)
    stream_callback = (
        callback_factory(cup_id)
        if stream_handoff and callable(callback_factory)
        else None
    )

    movement = controller.move_eef_to_pose(
        target_name=str(stage),
        target_pose=target_pose,
        frame="base",
        planner_stage=str(stage),
        gripper_action=gripper,
        holding_object=cup_id,
        record_trajectory=True,
        settle_steps=int(settle_steps),
        post_step_callback=stream_callback,
    )
    stream_restore: dict[str, Any] = {
        "required": stream_handoff,
        "restored": not stream_handoff,
    }
    deactivate_faucets = getattr(env, "deactivate_sensor_faucets", None)
    restore_stream = getattr(env, "set_water_stream_intercept", None)
    if (
        stream_handoff
        and bool(movement.get("pass", False))
        and (callable(deactivate_faucets) or callable(restore_stream))
    ):
        stream_restore = {
            "required": True,
            **dict(
                deactivate_faucets()
                if callable(deactivate_faucets)
                else restore_stream(None)
            ),
        }
    checks: dict[str, bool] = {
        "cartesian_return": bool(movement.get("pass", False)),
        "gripper_command_closed": bool(gripper > 0.0),
        "target_rotation_preserved": bool(
            np.allclose(target_base[:3, :3], current_base[:3, :3], atol=1.0e-8)
        ),
        "water_stream_restored_after_holding_return": bool(
            stream_restore.get("restored", False)
        ),
    }
    result: dict[str, Any] = {
        "primitive": "return_arm_holding_cup",
        "cup_id": cup_id,
        "stage": str(stage),
        "strategy": "cartesian_base_x_retract_preserve_eef_rotation",
        "initial_eef_site_in_base": pose_from_matrix(current_base),
        "target_eef_site_in_base": target_pose,
        "requested_retract_distance_base_x": retract_distance,
        "actual_retract_distance_base_x": float(
            current_base[0, 3] - target_base[0, 3]
        ),
        "minimum_base_x": minimum_x,
        "lift_offset": lift,
        "gripper_action": gripper,
        "movement": movement,
        "water_stream_handoff_from_fill": stream_handoff,
        "water_stream_restore": stream_restore,
        "checks": checks,
    }
    if not execute:
        result["dry_run"] = True
        result["physical_validation"] = {"deferred_to_execute": True}
        result["pass"] = bool(all(checks.values()))
        result["reason"] = None if result["pass"] else "holding_cup_return_plan_failed"
        return result

    env = _refresh(controller)
    final_world = np.asarray(
        controller.stage_executor.current_eef_site_world(), dtype=float
    ).reshape(4, 4)
    rotation_error = rotation_error_deg(
        current_world[:3, :3], final_world[:3, :3]
    )
    hold_status = get_object_hold_status(
        env,
        cup_id,
        max_end_effector_distance=hold_distance,
    )
    cup_rotation = np.asarray(
        env.sim.data.body_xmat[env.obj_body_id[cup_id]], dtype=float
    ).reshape(3, 3)
    upright_z_axis = float(cup_rotation[2, 2])
    upright_threshold = float(
        np.cos(np.deg2rad(float(env.cup_upright_threshold_deg)))
    )
    checks.update(
        {
            "eef_rotation_preserved": bool(rotation_error <= rotation_tolerance),
            "cup_still_held": bool(hold_status.get("held", False)),
            "cup_upright": bool(upright_z_axis >= upright_threshold),
        }
    )
    result["physical_validation"] = {
        "eef_rotation_error_deg": rotation_error,
        "orientation_tolerance_deg": rotation_tolerance,
        "hold_status": hold_status,
        "cup_upright_z_axis": upright_z_axis,
        "cup_upright_threshold": upright_threshold,
    }
    result["pass"] = bool(all(checks.values()))
    result["reason"] = None if result["pass"] else "holding_cup_return_not_verified"
    return result


def lift_held_cup_after_sink_exit(
    controller: Any,
    *,
    cup_id: str,
    sink_exterior_local_x_min: float,
    boundary_geometry_tolerance: float = 0.002,
    boundary_overshoot_tolerance: float = 0.020,
    route_local_y_per_x: float = 0.0,
    boundary_translation_gain: float = 3.0,
    boundary_min_action: float = 0.025,
    boundary_max_action: float = 0.35,
    boundary_velocity_profile_enabled: bool = True,
    boundary_profile_acceleration: float = 1.6,
    boundary_profile_deceleration: float = 1.2,
    boundary_profile_target_overshoot: float = 0.005,
    boundary_max_steps: int = 900,
    target_base_yaw: float | None = None,
    yaw_hold_gain: float = 0.30,
    lift_offset: float = 0.03,
    lift_position_tolerance: float = 0.003,
    lift_correction_attempts: int = 2,
    settle_steps: int = 40,
    gripper_action: float = 1.0,
    orientation_tolerance_deg: float = 15.0,
    max_end_effector_distance: float = 0.18,
    stage: str = "lift_held_cup_after_sink_exit",
    **_: Any,
) -> dict[str, Any]:
    """Cross the fixed sink edge with a held cup, then lift it vertically."""

    if cup_id not in CUP_IDS:
        raise ValueError(f"invalid WA-05 cup id: {cup_id!r}")
    boundary = float(sink_exterior_local_x_min)
    geometry_tolerance = float(boundary_geometry_tolerance)
    overshoot_tolerance = float(boundary_overshoot_tolerance)
    route_ratio = float(route_local_y_per_x)
    translation_gain = float(boundary_translation_gain)
    min_action = float(boundary_min_action)
    max_action = float(boundary_max_action)
    profile_enabled = bool(boundary_velocity_profile_enabled)
    profile_acceleration = float(boundary_profile_acceleration)
    profile_deceleration = float(boundary_profile_deceleration)
    profile_target_overshoot = float(boundary_profile_target_overshoot)
    yaw_gain = float(yaw_hold_gain)
    lift = float(lift_offset)
    lift_tolerance = float(lift_position_tolerance)
    gripper = float(gripper_action)
    rotation_tolerance = float(orientation_tolerance_deg)
    hold_distance = float(max_end_effector_distance)
    requested_yaw = None if target_base_yaw is None else float(target_base_yaw)
    numeric = (
        boundary,
        geometry_tolerance,
        overshoot_tolerance,
        route_ratio,
        translation_gain,
        min_action,
        max_action,
        profile_acceleration,
        profile_deceleration,
        profile_target_overshoot,
        yaw_gain,
        lift,
        lift_tolerance,
        gripper,
        rotation_tolerance,
        hold_distance,
    )
    if requested_yaw is not None:
        numeric += (requested_yaw,)
    if not all(np.isfinite(value) for value in numeric):
        raise ValueError("WA-05 sink-exit lift parameters must be finite")
    if geometry_tolerance <= 0.0 or overshoot_tolerance <= 0.0:
        raise ValueError("sink-exit boundary tolerances must be positive")
    if translation_gain <= 0.0 or not 0.0 < min_action <= max_action <= 1.0:
        raise ValueError("sink-exit Base action limits are invalid")
    if profile_acceleration <= 0.0 or profile_deceleration <= 0.0:
        raise ValueError("sink-exit velocity-profile rates must be positive")
    if profile_enabled and not 0.0 < profile_target_overshoot < overshoot_tolerance:
        raise ValueError(
            "sink-exit velocity-profile target overshoot must be positive and "
            "smaller than the boundary overshoot tolerance"
        )
    if int(boundary_max_steps) <= 0:
        raise ValueError("boundary_max_steps must be positive")
    if lift <= 0.0:
        raise ValueError("sink-exit lift_offset must be positive")
    if not 0.0 < lift_tolerance < lift:
        raise ValueError("lift_position_tolerance must be in (0, lift_offset)")
    if int(lift_correction_attempts) < 0:
        raise ValueError("lift_correction_attempts must be non-negative")
    if int(settle_steps) < 0:
        raise ValueError("settle_steps must be non-negative")
    if gripper <= 0.0:
        raise ValueError("sink-exit lift requires a closing gripper command")
    if rotation_tolerance <= 0.0 or hold_distance <= 0.0:
        raise ValueError("sink-exit physical tolerances must be positive")

    execute = bool(controller.runtime_config.execute)
    if not execute:
        return {
            "primitive": "lift_held_cup_after_sink_exit",
            "cup_id": cup_id,
            "stage": str(stage),
            "strategy": "cup_crosses_fixed_sink_local_x_then_vertical_lift",
            "configured_sink_exterior_local_x_min": boundary,
            "boundary_control_mode": (
                "velocity_profile_xy" if profile_enabled else "legacy_proportional_xy"
            ),
            "lift_offset": lift,
            "dry_run": True,
            "physical_validation": {"deferred_to_execute": True},
            "checks": {"configuration_valid": True},
            "pass": True,
            "reason": None,
        }

    env = _refresh(controller)
    live_bounds = _sink_exterior_counter_local_bounds(env)
    live_boundary = float(live_bounds["x_min"])
    geometry_error = abs(live_boundary - boundary)
    initial_cup_world = np.asarray(env.object_position(cup_id), dtype=float).reshape(3)
    initial_cup_local = _counter_local_position(env, initial_cup_world)
    initial_base_xy = np.asarray(controller.obs["robot0_base_pos"], dtype=float)[:2]
    initial_base_yaw = float(base_yaw(dict(controller.obs)))
    right = env.robots[0].composite_controller.part_controllers.get("right")
    if right is None:
        raise RuntimeError("robot does not expose the right-arm controller")
    arm_qpos = qpos_for_controller(env, right)
    base_start, base_end = env.robots[0].composite_controller._action_split_indexes["base"]
    base_action_size = int(base_end) - int(base_start)
    if base_action_size < 2:
        raise RuntimeError("robot Base action must expose planar X/Y commands")

    obs = dict(controller.obs)
    current_cup_local = initial_cup_local.copy()
    steps = 0
    crossed = bool(current_cup_local[0] <= boundary)
    translation_profile = _sink_exit_translation_profile(
        env,
        enabled=profile_enabled,
        min_action=min_action,
        max_action=max_action,
        acceleration=profile_acceleration,
        deceleration=profile_deceleration,
    )
    command_norm_min = float("inf")
    command_norm_max = 0.0
    command_norm_final = 0.0
    while not crossed and steps < int(boundary_max_steps):
        remaining_x = boundary - float(current_cup_local[0])
        control_remaining_x = (
            remaining_x - profile_target_overshoot
            if translation_profile is not None
            else remaining_x
        )
        local_error = np.asarray(
            [control_remaining_x, control_remaining_x * route_ratio], dtype=float
        )
        world_error = _counter_local_vector_to_world(env, local_error)
        controller_error = world_xy_to_controller_action(env, world_error)
        if translation_profile is not None:
            # The legacy proportional command spent hundreds of frames in its
            # low-speed tail.  Use the same acceleration/braking profile as the
            # calibrated manual Base path, while retaining the cup-boundary
            # crossing condition and overshoot gate below.
            command_xy = translation_profile.step_action(
                controller_error,
                tolerance=0.0,
            )
        else:
            command_xy = controller_error * translation_gain
            command_norm = float(np.linalg.norm(command_xy))
            if command_norm > max_action:
                command_xy *= max_action / command_norm
            elif 0.0 < command_norm < min_action:
                command_xy *= min_action / command_norm

        command_norm = float(np.linalg.norm(command_xy))
        if command_norm > 0.0:
            command_norm_min = min(command_norm_min, command_norm)
        command_norm_max = max(command_norm_max, command_norm)
        command_norm_final = command_norm

        action = np.zeros(base_action_size, dtype=float)
        action[:2] = command_xy[:2]
        if requested_yaw is not None and action.size >= 3:
            yaw_error = normalize_angle(requested_yaw - float(base_yaw(obs)))
            action[2] = float(np.clip(yaw_error * yaw_gain, -0.20, 0.20))
        obs = step_base_and_hold_arm(
            env,
            arm_qpos=arm_qpos,
            base_action=action,
            base_mode=1.0,
            gripper_action=gripper,
        )
        steps += 1
        current_cup_world = np.asarray(env.object_position(cup_id), dtype=float).reshape(3)
        current_cup_local = _counter_local_position(env, current_cup_world)
        crossed = bool(current_cup_local[0] <= boundary)

    controller.refresh_obs()
    boundary_cup_world = np.asarray(env.object_position(cup_id), dtype=float).reshape(3)
    boundary_cup_local = _counter_local_position(env, boundary_cup_world)
    boundary_base_xy = np.asarray(controller.obs["robot0_base_pos"], dtype=float)[:2]
    boundary_base_yaw = float(base_yaw(dict(controller.obs)))
    overshoot = max(0.0, boundary - float(boundary_cup_local[0]))
    boundary_control = {
        "mode": (
            "velocity_profile_xy"
            if translation_profile is not None
            else "legacy_proportional_xy"
        ),
        "translation_gain": translation_gain,
        "min_action": min_action,
        "max_action": max_action,
        "profile_acceleration": profile_acceleration,
        "profile_deceleration": profile_deceleration,
        "profile_target_overshoot": profile_target_overshoot,
        "control_dt": float(resolve_control_dt(env)),
        "command_norm_min": (
            0.0 if not np.isfinite(command_norm_min) else command_norm_min
        ),
        "command_norm_max": command_norm_max,
        "command_norm_final": command_norm_final,
    }
    boundary_checks = {
        "configured_edge_matches_live_sink": bool(
            geometry_error <= geometry_tolerance
        ),
        "cup_started_inside_sink_exit_edge": bool(initial_cup_local[0] > boundary),
        "cup_crossed_sink_exit_edge": bool(boundary_cup_local[0] <= boundary),
        "boundary_overshoot_bounded": bool(overshoot <= overshoot_tolerance),
        "boundary_reached_before_step_limit": bool(
            crossed and steps <= int(boundary_max_steps)
        ),
    }
    if not all(boundary_checks.values()):
        return {
            "primitive": "lift_held_cup_after_sink_exit",
            "cup_id": cup_id,
            "stage": str(stage),
            "strategy": "cup_crosses_fixed_sink_local_x_then_vertical_lift",
            "configured_sink_exterior_local_x_min": boundary,
            "live_sink_exterior_counter_local_bounds": live_bounds,
            "boundary_geometry_error_m": geometry_error,
            "initial_cup_position_world": initial_cup_world.tolist(),
            "initial_cup_position_counter_local": initial_cup_local.tolist(),
            "boundary_cup_position_world": boundary_cup_world.tolist(),
            "boundary_cup_position_counter_local": boundary_cup_local.tolist(),
            "initial_base_xy": initial_base_xy.tolist(),
            "initial_base_yaw": initial_base_yaw,
            "boundary_base_xy": boundary_base_xy.tolist(),
            "boundary_base_yaw": boundary_base_yaw,
            "base_translation_steps": int(steps),
            "boundary_control": boundary_control,
            "boundary_overshoot_m": overshoot,
            "checks": boundary_checks,
            "pass": False,
            "reason": "held_cup_sink_exit_boundary_not_verified",
        }

    current_world = np.asarray(
        controller.stage_executor.current_eef_site_world(), dtype=float
    ).reshape(4, 4)
    base_world = matrix_from_base_obs(dict(controller.obs))
    current_base = np.linalg.inv(base_world) @ current_world
    target_base = np.asarray(current_base, dtype=float).copy()
    target_base[2, 3] += lift
    movements: list[dict[str, Any]] = []
    for attempt in range(int(lift_correction_attempts) + 1):
        movement = controller.move_eef_to_pose(
            target_name=str(stage),
            target_pose=pose_from_matrix(target_base),
            frame="base",
            planner_stage=(
                str(stage) if attempt == 0 else f"{stage}_correction_{attempt}"
            ),
            gripper_action=gripper,
            holding_object=cup_id,
            record_trajectory=True,
            settle_steps=int(settle_steps),
        )
        controller.refresh_obs()
        attempt_world = np.asarray(
            controller.stage_executor.current_eef_site_world(), dtype=float
        ).reshape(4, 4)
        movement["attempt"] = int(attempt)
        movement["cumulative_eef_lift_m"] = float(
            attempt_world[2, 3] - current_world[2, 3]
        )
        movements.append(movement)
        if movement["cumulative_eef_lift_m"] >= lift - lift_tolerance:
            break

    env = _refresh(controller)
    final_world = np.asarray(
        controller.stage_executor.current_eef_site_world(), dtype=float
    ).reshape(4, 4)
    final_cup_world = np.asarray(env.object_position(cup_id), dtype=float).reshape(3)
    final_cup_local = _counter_local_position(env, final_cup_world)
    rotation_error = rotation_error_deg(current_world[:3, :3], final_world[:3, :3])
    actual_lift = float(final_world[2, 3] - current_world[2, 3])
    hold_status = get_object_hold_status(
        env,
        cup_id,
        max_end_effector_distance=hold_distance,
    )
    cup_rotation = np.asarray(
        env.sim.data.body_xmat[env.obj_body_id[cup_id]], dtype=float
    ).reshape(3, 3)
    upright_z_axis = float(cup_rotation[2, 2])
    upright_threshold = float(
        np.cos(np.deg2rad(float(env.cup_upright_threshold_deg)))
    )
    checks = {
        **boundary_checks,
        "vertical_lift_planned": bool(
            movements and all(item.get("pass", False) for item in movements)
        ),
        "eef_rotation_preserved": bool(rotation_error <= rotation_tolerance),
        "eef_lift_reached": bool(actual_lift >= lift - lift_tolerance),
        "cup_remains_outside_sink_edge": bool(final_cup_local[0] <= boundary),
        "cup_still_held": bool(hold_status.get("held", False)),
        "cup_upright": bool(upright_z_axis >= upright_threshold),
    }
    passed = bool(all(checks.values()))
    clear_affinity = getattr(env, "clear_sensor_faucet_affinity", None)
    affinity_clear = (
        dict(clear_affinity(cup_id))
        if callable(clear_affinity)
        else {"cup_id": cup_id, "cleared": False, "reason": "unsupported"}
    )
    return {
        "primitive": "lift_held_cup_after_sink_exit",
        "cup_id": cup_id,
        "stage": str(stage),
        "strategy": "cup_crosses_fixed_sink_local_x_then_vertical_lift",
        "configured_sink_exterior_local_x_min": boundary,
        "live_sink_exterior_counter_local_bounds": live_bounds,
        "boundary_geometry_error_m": geometry_error,
        "initial_cup_position_world": initial_cup_world.tolist(),
        "initial_cup_position_counter_local": initial_cup_local.tolist(),
        "boundary_cup_position_world": boundary_cup_world.tolist(),
        "boundary_cup_position_counter_local": boundary_cup_local.tolist(),
        "final_cup_position_world": final_cup_world.tolist(),
        "final_cup_position_counter_local": final_cup_local.tolist(),
        "initial_base_xy": initial_base_xy.tolist(),
        "initial_base_yaw": initial_base_yaw,
        "boundary_base_xy": boundary_base_xy.tolist(),
        "boundary_base_yaw": boundary_base_yaw,
        "base_translation_steps": int(steps),
        "boundary_control": boundary_control,
        "boundary_overshoot_m": overshoot,
        "lift_offset": lift,
        "lift_position_tolerance": lift_tolerance,
        "actual_eef_lift_m": actual_lift,
        "eef_rotation_error_deg": rotation_error,
        "movement": movements[-1],
        "movement_attempts": movements,
        "physical_validation": {
            "hold_status": hold_status,
            "cup_upright_z_axis": upright_z_axis,
            "cup_upright_threshold": upright_threshold,
            "orientation_tolerance_deg": rotation_tolerance,
        },
        "sensor_faucet_affinity_clear": affinity_clear,
        "checks": checks,
        "pass": passed,
        "reason": None if passed else "held_cup_sink_exit_lift_not_verified",
    }


def _sink_exit_translation_profile(
    env: Any,
    *,
    enabled: bool,
    min_action: float,
    max_action: float,
    acceleration: float,
    deceleration: float,
) -> PlanarVelocityProfile | None:
    """Build the bounded sink-exit profile in physical XY speed units."""

    if not bool(enabled):
        return None
    return PlanarVelocityProfile(
        AxialProfileParams(
            v_max=float(max_action) * XY_ACTION_SPEED,
            a_max=float(acceleration),
            a_dec=float(deceleration),
            v_min=float(min_action) * XY_ACTION_SPEED,
            action_speed=XY_ACTION_SPEED,
        ),
        dt=resolve_control_dt(env),
    )


def _sink_exterior_counter_local_bounds(env: Any) -> dict[str, float]:
    from robocasa.utils import object_utils as OU

    points_world = np.asarray(env.sink.get_bbox_points(), dtype=float)
    points_local = np.asarray(
        [
            OU.get_fixture_to_point_rel_offset(env.counter, point)
            for point in points_world
        ],
        dtype=float,
    )
    return {
        "x_min": float(np.min(points_local[:, 0])),
        "x_max": float(np.max(points_local[:, 0])),
        "y_min": float(np.min(points_local[:, 1])),
        "y_max": float(np.max(points_local[:, 1])),
    }


def _counter_local_position(env: Any, world_position: Any) -> np.ndarray:
    from robocasa.utils import object_utils as OU

    return np.asarray(
        OU.get_fixture_to_point_rel_offset(
            env.counter,
            np.asarray(world_position, dtype=float),
        ),
        dtype=float,
    )


def _counter_local_vector_to_world(env: Any, local_vector_xy: Any) -> np.ndarray:
    values = np.asarray(local_vector_xy, dtype=float).reshape(2)
    yaw = float(env.counter.rot)
    rotation = np.asarray(
        [
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)],
        ],
        dtype=float,
    )
    return rotation @ values


def _refresh(controller: Any) -> Any:
    controller.refresh_obs()
    controller.env.update_state()
    return controller.env


def _dry_result(primitive: str, **details: Any) -> dict[str, Any]:
    return {
        "primitive": primitive,
        "dry_run": True,
        "checks": {"deferred_to_execute": True},
        "pass": True,
        **details,
    }


__all__ = [
    "build_stage_handlers",
    "lift_held_cup_after_sink_exit",
    "return_arm_holding_cup",
]
