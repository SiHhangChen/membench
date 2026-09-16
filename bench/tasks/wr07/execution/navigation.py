"""WR-07 station mapping and mobile-base assembly."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import atan2
from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.contracts import StageKind, StageResult
from bench.execution.navigation.contracts import Pose, StationSpec
from bench.execution.navigation import (
    BaseDriverAdapter,
    NavigationController,
    RoboCasaNavigationAdapter,
    StationPathPlanner,
    StationPoseSampler,
    station_specs_from_candidates,
)
from bench.execution.navigation.base_motion import qpos_for_controller, step_base_and_hold_arm
from bench.execution.planning import NavigationConfig
from bench.execution.planning.nav_algo import normalize_angle
from bench.execution.sampling.rng import SeedStream
from bench.execution.sampling.station_sampler import SimpleStationSampler
from bench.tasks.wr07.domain.contracts import ITEM_ID, SOURCE_SLOTS
from bench.tasks.wr07.execution.stations import station_aliases, station_specs


@dataclass(frozen=True)
class NavigationAssembly:
    """Concrete navigation objects used by WR-07."""

    controller: NavigationController
    env_adapter: RoboCasaNavigationAdapter
    station_sampler: Any
    path_planner: StationPathPlanner | None
    base_driver: Any
    config: NavigationConfig


class SourcePlateStationSampler:
    """Move the base to the currently selected source plate."""

    def __init__(
        self,
        delegate: StationPoseSampler,
        env: Any,
    ):
        self._delegate = delegate
        self._env = env
        self._summary_by_station: dict[str, dict[str, Any]] = {}
        # Circle stations normally resample on every call.  WR-07 observes
        # the source area and then navigates to the selected source slot; keep
        # those two calls on the same sampled source pose, as the former
        # discrete calibrated candidates did.
        self._source_circle_pose: Pose | None = None
        self._source_circle_summary: dict[str, Any] | None = None

    def sample(self, station_name: str, *, purpose: str | None = None) -> Pose:
        station_id_resolver = getattr(self._delegate, "station_id", None)
        station_id = (
            station_id_resolver(station_name, purpose=purpose)
            if callable(station_id_resolver)
            else station_name
        )
        is_source_station = str(station_id) == "counter_pick_station"
        if is_source_station and self._source_circle_pose is not None:
            pose = self._source_circle_pose
            summary = dict(self._source_circle_summary or {})
            summary["purpose"] = purpose
        else:
            pose = self._delegate.sample(station_name, purpose=purpose)
            summary = dict(self._delegate.last_summary(station_name) or {})
        sample = dict(summary.get("sample") or {})
        if (
            is_source_station
            and self._source_circle_pose is None
            and isinstance(sample.get("circle"), dict)
        ):
            self._source_circle_pose = pose
            self._source_circle_summary = dict(summary)
        settings = sample.get("object_relative_xy")
        if not isinstance(settings, dict) or not bool(settings.get("enabled", False)):
            self._summary_by_station[station_name] = summary
            return pose
        required_purpose = str(settings.get("purpose", "source_pick"))
        actual_purpose = str(purpose or "")
        if actual_purpose != required_purpose and actual_purpose != "source_observe":
            self._summary_by_station[station_name] = summary
            return pose
        source_slot = str(self._env.source_slot_by_item[ITEM_ID])
        if source_slot not in SOURCE_SLOTS:
            raise ValueError(f"WR-07 unknown source slot: {source_slot!r}")
        reference_xy = _finite_xy(
            self._env.source_plate_pose(source_slot),
            label=f"source plate position for {source_slot}",
        )
        base_minus_plate_xy = _finite_xy(
            settings.get("base_minus_plate_xy", settings.get("base_minus_object_xy", (0.0, -0.52))),
            label=f"base-minus-plate offset for {station_name}",
        )
        target_xy = reference_xy + base_minus_plate_xy
        adjusted = Pose(
            position=(
                float(target_xy[0]),
                float(target_xy[1]),
                float(pose.position[2]),
            ),
            quaternion=pose.quaternion,
            torso_height=pose.torso_height,
        )
        summary["target_xy"] = target_xy.astype(float).tolist()
        summary["object_relative_xy"] = {
            "enabled": True,
            "purpose": actual_purpose,
            "source_slot": source_slot,
            "reference_kind": "source_plate",
            "object_xy": reference_xy.astype(float).tolist(),
            "base_minus_plate_xy": base_minus_plate_xy.astype(float).tolist(),
            "reference_base_xy": [
                float(pose.position[0]),
                float(pose.position[1]),
            ],
            "adjusted_base_xy": target_xy.astype(float).tolist(),
        }
        self._summary_by_station[station_name] = summary
        return adjusted

    def last_summary(self, station_name: str) -> dict[str, Any] | None:
        summary = self._summary_by_station.get(station_name)
        if summary is not None:
            return summary
        return self._delegate.last_summary(station_name)

    def include_points(self) -> list[np.ndarray]:
        return self._delegate.include_points()

    def reset_choices(self) -> None:
        self._delegate.reset_choices()
        self._summary_by_station.clear()
        self._source_circle_pose = None
        self._source_circle_summary = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)



def _pose_yaw(pose: Pose) -> float:
    x, y, z, w = (float(value) for value in pose.quaternion)
    return atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class StationHeightBaseDriver:
    """Apply the sampled station torso height after Base XY/yaw navigation."""

    def __init__(self, delegate: Any, env: Any, *, torso_mover: Any = None, seed: int = 0):
        self._delegate = delegate
        self._env = env
        self._torso_mover = torso_mover or move_station_torso_to_height
        self._seed = int(seed)

    def plan_and_move(
        self,
        target_xy: np.ndarray,
        target_yaw: float | None,
        *,
        holding: str | None = None,
        planning_summary: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        holding_navigation_mode = str(kwargs.pop("holding_navigation_mode", ""))
        station_name = str(kwargs.pop("station_name", ""))
        purpose = kwargs.pop("purpose", None)
        if holding is not None and holding_navigation_mode == "rotate_then_translate":
            result = self._plan_holding_round_trip(
                target_xy,
                target_yaw,
                station_name=station_name,
                purpose=purpose,
                holding=holding,
                planning_summary=planning_summary,
                **kwargs,
            )
        else:
            result = self._delegate.plan_and_move(
                target_xy,
                target_yaw,
                station_name=station_name,
                purpose=purpose,
                holding=holding,
                planning_summary=planning_summary,
                **kwargs,
            )
        if not bool(result.success):
            return result

        torso_height = _station_torso_height(planning_summary)
        try:
            torso = self._torso_mover(
                self._env,
                torso_height,
                holding=holding,
                post_step_callback=getattr(
                    self._delegate, "_post_step_callback", None
                ),
            )
        except Exception as exc:
            result.success = False
            result.failure_code = f"station torso adjustment failed: {exc}"
            result.details["station_torso"] = {
                "applied": False,
                "requested_height": torso_height,
                "reason": "station_torso_adjustment_failed",
                "error": str(exc),
            }
            return result

        result.details["station_torso"] = torso
        refresh = getattr(self._delegate, "refresh_obs", None)
        if callable(refresh) and bool(torso.get("applied", False)):
            refresh()
        return result

    def _plan_holding_round_trip(
        self,
        target_xy: np.ndarray,
        target_yaw: float | None,
        *,
        station_name: str,
        purpose: str | None,
        holding: str,
        planning_summary: dict[str, Any] | None,
        **kwargs: Any,
    ) -> StageResult:
        """Transport a held object with an explicit rotate-then-translate sequence.

        The ordinary base primitive couples XY and yaw control. That is useful
        for empty-base docking, but the coupled acceleration can make a
        marginal contact grasp slip. WR-07's held transport follows the
        demonstrated workflow: rotate in place, translate while keeping that
        yaw, then make only a small final yaw correction if needed.
        """

        delegate = self._delegate
        current_xy = np.asarray(delegate.current_base_xy(), dtype=float)[:2]
        desired_yaw = None if target_yaw is None else float(target_yaw)
        nav_config = getattr(delegate, "_nav_config", None)
        from bench.execution.navigation.base_motion import qpos_for_controller
        from bench.execution.navigation.rotation_primitive import move_straight, rotate_in_place
        from bench.execution.navigation.velocity_profile import BaseVelocityLimits

        # The source and dining stations face opposite directions.  During
        # held transport, sample the signed half-turn itself so the robot
        # always rotates in the demonstrated (clockwise) direction.  The
        # perturbation is independent from station XY/yaw draws.
        turn_randomization = None
        if desired_yaw is not None:
            source_yaw = 1.5697450892050213
            target_station_yaw = -1.551592653589793
            base_turn = target_station_yaw - source_yaw
            # Keep the signed difference near -pi rather than wrapping it.
            if base_turn > 0.0:
                base_turn -= 2.0 * np.pi
            turn_rng = np.random.default_rng(np.random.SeedSequence([self._seed, 0x57523037, 0x5455524E]))
            perturbation = float(turn_rng.uniform(-np.deg2rad(5.0), np.deg2rad(5.0)))
            initial_yaw_for_turn = float(delegate.current_base_yaw())
            sampled_turn = base_turn + perturbation
            desired_yaw = initial_yaw_for_turn + sampled_turn
            turn_randomization = {
                "base_turn_rad": float(base_turn),
                "base_turn_deg": float(np.rad2deg(base_turn)),
                "perturbation_rad": perturbation,
                "perturbation_deg": float(np.rad2deg(perturbation)),
                "sampled_turn_rad": float(sampled_turn),
                "sampled_turn_deg": float(np.rad2deg(sampled_turn)),
                "initial_yaw": initial_yaw_for_turn,
                "target_yaw": float(desired_yaw),
                "range_deg": [float(np.rad2deg(base_turn - np.deg2rad(5.0))), float(np.rad2deg(base_turn + np.deg2rad(5.0)))],
            }

        right = self._env.robots[0].composite_controller.part_controllers["right"]
        arm_qpos = qpos_for_controller(self._env, right)
        velocity_limits = (
            BaseVelocityLimits.from_navigation_config(nav_config)
            if nav_config is not None
            else None
        )
        yaw_tolerance = float(
            kwargs.get(
                "yaw_tolerance",
                getattr(nav_config, "base_yaw_tolerance", 0.08),
            )
        )
        xy_tolerance = float(
            kwargs.get(
                "tolerance",
                getattr(nav_config, "base_tolerance", 0.04),
            )
        )
        max_steps = int(
            kwargs.get(
                "max_steps",
                getattr(nav_config, "base_max_steps", 2000),
            )
        )

        # Use the dedicated primitive: move_base_to_pose couples yaw and XY
        # and can drift several decimeters during a 180-degree turn.
        obs, rotation_summary = rotate_in_place(
            self._env,
            delegate._obs,
            target_yaw=desired_yaw if desired_yaw is not None else float(delegate.current_base_yaw()),
            arm_qpos=arm_qpos,
            gripper_action=1.0,
            yaw_tolerance=yaw_tolerance,
            max_steps=min(max_steps, 500),
            # The task's logical object id is mapped through env.obj_body_id
            # by _holding_object_status; the generic primitive expects a raw
            # The task-level item id is resolved by the holding-status check;
            # the primitive itself only needs the live arm state.
            holding_object_name=None,
            holding_drop_tolerance=1.0e6,
            velocity_limits=velocity_limits,
            rotation_direction=-1,
        )
        delegate._obs = obs
        rotation = _navigation_stage_result(
            rotation_summary,
            station_name=station_name,
            purpose=f"{purpose}_rotate" if purpose else "rotate_then_translate_rotate",
            target_xy=current_xy,
            target_yaw=desired_yaw,
        )
        phases: list[dict[str, Any]] = [{"phase": "rotate", **dict(rotation.details)}]
        if not bool(rotation.success):
            rotation.details["holding_navigation"] = {
                "mode": "rotate_then_translate",
                "phases": phases,
                "failed_phase": "rotate",
                "turn_randomization": turn_randomization,
            }
            return rotation
        rotation_status = self._holding_object_status(holding)
        phases[-1]["holding_status"] = rotation_status
        if not rotation_status["safe"]:
            rotation.success = False
            rotation.failure_code = "holding_object_dropped_during_rotation"
            rotation.details["holding_navigation"] = {
                "mode": "rotate_then_translate",
                "phases": phases,
                "failed_phase": "rotate",
                "turn_randomization": turn_randomization,
            }
            return rotation

        # Move straight with no yaw target, so the base controller cannot
        # reintroduce simultaneous rotation while carrying the object.
        obs, translation_summary = move_straight(
            self._env,
            delegate._obs,
            target_xy=np.asarray(target_xy, dtype=float)[:2],
            arm_qpos=arm_qpos,
            gripper_action=1.0,
            tolerance=xy_tolerance,
            max_steps=max_steps,
            action_frame=str(getattr(nav_config, "base_action_frame", "world")),
            translation_action_sign=float(
                getattr(nav_config, "base_translation_action_sign", 1.0)
            ),
            holding_object_name=None,
            holding_drop_tolerance=1.0e6,
            maintain_yaw=False,
            velocity_limits=velocity_limits,
        )
        delegate._obs = obs
        translation = _navigation_stage_result(
            translation_summary,
            station_name=station_name,
            purpose=f"{purpose}_translate" if purpose else "rotate_then_translate_translate",
            target_xy=np.asarray(target_xy, dtype=float)[:2],
            target_yaw=None,
            planning_summary=planning_summary,
        )
        phases.append({"phase": "translate", **dict(translation.details)})
        translation_status = self._holding_object_status(holding)
        phases[-1]["holding_status"] = translation_status
        if not bool(translation.success) or not translation_status["safe"]:
            translation.success = False
            if not translation_status["safe"]:
                translation.failure_code = "holding_object_dropped_during_translation"
            translation.details["holding_navigation"] = {
                "mode": "rotate_then_translate",
                "phases": phases,
                "failed_phase": "translate" if not translation.success else None,
                "turn_randomization": turn_randomization,
            }
            return translation

        # Usually rotation already reached the station yaw. Correct only any
        # small drift introduced by translation, with XY held fixed.
        final_yaw_error = 0.0
        if desired_yaw is not None:
            current_yaw = float(delegate.current_base_yaw())
            final_yaw_error = _angle_error(desired_yaw, current_yaw)
        if desired_yaw is not None and abs(final_yaw_error) > yaw_tolerance:
            final_xy = np.asarray(delegate.current_base_xy(), dtype=float)[:2]
            obs, correction_summary = rotate_in_place(
                self._env,
                delegate._obs,
                target_yaw=desired_yaw,
                arm_qpos=arm_qpos,
                gripper_action=1.0,
                yaw_tolerance=yaw_tolerance,
                max_steps=min(max_steps, 500),
                holding_object_name=None,
                holding_drop_tolerance=1.0e6,
                velocity_limits=velocity_limits,
            )
            delegate._obs = obs
            correction = _navigation_stage_result(
                correction_summary,
                station_name=station_name,
                purpose=f"{purpose}_final_yaw" if purpose else "rotate_then_translate_final_yaw",
                target_xy=final_xy,
                target_yaw=desired_yaw,
                planning_summary=planning_summary,
            )
            phases.append({"phase": "final_yaw", **dict(correction.details)})
            correction_status = self._holding_object_status(holding)
            phases[-1]["holding_status"] = correction_status
            if not bool(correction.success) or not correction_status["safe"]:
                correction.success = False
                if not correction_status["safe"]:
                    correction.failure_code = "holding_object_dropped_during_final_yaw"
                correction.details["holding_navigation"] = {
                    "mode": "rotate_then_translate",
                    "phases": phases,
                    "failed_phase": "final_yaw",
                    "turn_randomization": turn_randomization,
                }
                return correction
            result = correction
        else:
            result = translation

        result.details["holding_navigation"] = {
            "mode": "rotate_then_translate",
            "phases": phases,
            "failed_phase": None,
            "final_yaw_error": float(final_yaw_error),
            "turn_randomization": turn_randomization,
        }
        result.details["holding_motion"] = self._holding_object_status(holding)
        return result

    def _holding_object_status(self, object_name: str) -> dict[str, Any]:
        try:
            body_id = self._env.obj_body_id[str(object_name)]
            position = np.asarray(self._env.sim.data.body_xpos[body_id], dtype=float)
        except Exception as exc:
            return {"safe": False, "object_name": str(object_name), "reason": str(exc)}
        # The milk carton is carried well above the counter. A floor-height
        # position is an unambiguous drop, while small contact/grasp jitter is
        # tolerated during the explicit motion phases.
        safe = bool(np.all(np.isfinite(position)) and float(position[2]) > 0.65)
        return {
            "safe": safe,
            "object_name": str(object_name),
            "position": position.astype(float).tolist(),
            "height_threshold": 0.65,
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


def _angle_error(target: float, current: float) -> float:
    return float(normalize_angle(float(target) - float(current)))


def _navigation_stage_result(
    summary: Mapping[str, Any],
    *,
    station_name: str,
    purpose: str,
    target_xy: np.ndarray,
    target_yaw: float | None,
    planning_summary: dict[str, Any] | None = None,
) -> StageResult:
    details = dict(summary)
    details["station_name"] = station_name
    details["purpose"] = purpose
    details["target_xy"] = np.asarray(target_xy, dtype=float)[:2].tolist()
    details["target_yaw"] = None if target_yaw is None else float(target_yaw)
    if planning_summary is not None:
        details["station_path"] = planning_summary
    success = bool(details.get("pass", False))
    return StageResult(
        index=-1,
        kind=StageKind.NAVIGATE,
        success=success,
        failure_code=None if success else str(details.get("reason", "navigation_failed")),
        details=details,
    )


def move_station_torso_to_height(
    env: Any,
    target_height: float | None,
    *,
    holding: str | None = None,
    tolerance: float = 0.002,
    max_steps: int = 240,
    post_step_callback: Any = None,
) -> dict[str, Any]:
    """Reach one calibrated station torso height through normal env steps."""

    robot = env.robots[0]
    composite = robot.composite_controller
    torso = composite.part_controllers.get("torso")
    current = (
        None
        if torso is None
        else float(qpos_for_controller(env, torso).reshape(-1)[0])
    )
    if target_height is None:
        return {
            "applied": False,
            "reason": "station_has_no_torso_height",
            "requested_height": None,
            "initial_height": current,
            "actual_height": current,
            "steps": 0,
        }
    target = float(target_height)
    if not np.isfinite(target):
        raise ValueError("station torso_height must be finite")
    if torso is None:
        raise RuntimeError("station requests a torso height but no torso exists")
    if tolerance <= 0.0 or max_steps <= 0:
        raise ValueError("station torso tolerance and max_steps must be positive")

    joint_indexes = np.asarray(torso.joint_index, dtype=int).reshape(-1)
    if joint_indexes.size < 1:
        raise RuntimeError("torso controller does not expose a joint index")
    joint_range = np.asarray(
        env.sim.model.jnt_range[int(joint_indexes[0])], dtype=float
    ).reshape(-1)
    if (
        joint_range.size < 2
        or not np.all(np.isfinite(joint_range[:2]))
        or target < float(joint_range[0]) - 1e-9
        or target > float(joint_range[1]) + 1e-9
    ):
        raise ValueError(
            f"station torso_height={target:.4f}m is outside joint range "
            f"{joint_range[:2].tolist()}"
        )

    right = composite.part_controllers.get("right")
    if right is None:
        raise RuntimeError("robot does not expose the right-arm controller")
    base_start, base_end = composite._action_split_indexes["base"]
    base_action = np.zeros(int(base_end) - int(base_start), dtype=float)
    arm_qpos = qpos_for_controller(env, right)
    initial = float(current)
    actual = initial
    executed_steps = 0
    for _ in range(int(max_steps)):
        error = target - actual
        if abs(error) <= float(tolerance):
            break
        torso_action = float(np.clip(error / 0.05, -1.0, 1.0))
        step_base_and_hold_arm(
            env,
            arm_qpos=arm_qpos,
            base_action=base_action,
            base_mode=1.0,
            gripper_action=1.0 if holding else -1.0,
            torso_action=torso_action,
            post_step_callback=post_step_callback,
        )
        executed_steps += 1
        actual = float(qpos_for_controller(env, torso).reshape(-1)[0])
    if abs(target - actual) > float(tolerance):
        raise RuntimeError(
            f"torso failed to reach {target:.4f}m; actual={actual:.4f}m"
        )
    return {
        "applied": True,
        "reason": "station_torso_height_reached",
        "requested_height": target,
        "initial_height": initial,
        "actual_height": actual,
        "error": actual - target,
        "tolerance": float(tolerance),
        "steps": int(executed_steps),
        "holding": holding,
        "joint_range": joint_range[:2].astype(float).tolist(),
    }


def _station_torso_height(
    planning_summary: dict[str, Any] | None,
) -> float | None:
    station_pose = dict((planning_summary or {}).get("station_pose") or {})
    value = station_pose.get("torso_height")
    return None if value is None else float(value)


def build_navigation_config(
    settings: dict[str, Any],
    *, manual_base_path_file: str | Path | None,
    station_base_pose_file: str | Path,
) -> NavigationConfig:
    values = dict(settings)
    config = replace(
        NavigationConfig(),
        kinematic_base_navigation=bool(values.pop("kinematic_base_navigation", False)),
        base_action_frame=str(values.pop("base_action_frame", "world")),
        base_translation_action_sign=float(values.pop("base_translation_action_sign", 1.0)),
        base_tolerance=float(values.pop("base_tolerance", 0.08)),
        base_yaw_tolerance=float(values.pop("base_yaw_tolerance", 0.08)),
        base_max_steps=int(values.pop("base_max_steps", 2000)),
        base_holding_drop_tolerance=float(values.pop("base_holding_drop_tolerance", 0.10)),
        base_profile_v_max=float(values.pop("base_profile_v_max", 0.35)),
        base_profile_a_max=float(values.pop("base_profile_a_max", 0.70)),
        base_profile_a_dec=float(values.pop("base_profile_a_dec", 0.65)),
        base_profile_v_min=float(values.pop("base_profile_v_min", 0.10)),
        base_profile_dock_v_max=float(values.pop("base_profile_dock_v_max", 0.22)),
        base_profile_dock_v_min=float(values.pop("base_profile_dock_v_min", 0.10)),
        base_profile_w_max=float(values.pop("base_profile_w_max", 0.50)),
        base_profile_alpha_max=float(values.pop("base_profile_alpha_max", 0.80)),
        base_profile_alpha_dec=float(values.pop("base_profile_alpha_dec", 0.70)),
        base_profile_w_min=float(values.pop("base_profile_w_min", 0.08)),
        manual_base_path_enabled=bool(values.pop("manual_base_path_enabled", False)),
        manual_base_path_file=(
            None if manual_base_path_file is None else Path(manual_base_path_file)
        ),
        manual_base_path_start_index=int(values.pop("manual_base_path_start_index", 0)),
        manual_base_path_xy_tolerance=float(values.pop("manual_base_path_xy_tolerance", 0.08)),
        manual_base_path_yaw_tolerance=float(values.pop("manual_base_path_yaw_tolerance", 0.08)),
        manual_base_path_segment_yaw_threshold=float(values.pop("manual_base_path_segment_yaw_threshold", 0.25)),
        manual_base_path_rotate_xy_tolerance=float(values.pop("manual_base_path_rotate_xy_tolerance", 0.18)),
        base_single_axis_tracking=bool(values.pop("base_single_axis_tracking", False)),
        use_station_astar_paths=bool(values.pop("use_station_astar_paths", False)),
        station_nav_use_docking=bool(values.pop("station_nav_use_docking", True)),
        station_base_pose_file=Path(station_base_pose_file),
        use_station_base_poses=bool(values.pop("use_station_base_poses", True)),
    )
    if values:
        raise ValueError("unknown WR-07 navigation settings: " + ", ".join(sorted(values)))
    return config


def build_navigation_assembly(
    env: Any,
    obs: dict[str, Any],
    *,
    nav_config: NavigationConfig | None = None,
    seed: int = 0,
    base_driver: Any | None = None,
) -> NavigationAssembly:
    if nav_config is None:
        raise ValueError("WR-07 navigation config is required")
    config = nav_config
    adapter = RoboCasaNavigationAdapter(env, obs, station_specs_provider=station_specs)
    raw_driver = (
        base_driver
        if base_driver is not None
        else BaseDriverAdapter(env, obs, nav_config=config)
    )
    driver = StationHeightBaseDriver(raw_driver, env, seed=seed)
    base_pose_provider = driver if hasattr(driver, "current_base_xy") else adapter
    sampler = _build_station_sampler(
        adapter, config, seed=seed, base_pose_provider=base_pose_provider
    )
    planner = (
        StationPathPlanner(obstacles=adapter.build_fixture_obstacles, config=config)
        if bool(config.use_station_astar_paths)
        else None
    )
    controller = NavigationController(
        station_sampler=sampler,
        base_driver=driver,
        path_planner=planner,
        base_pose_provider=base_pose_provider,
    )
    return NavigationAssembly(
        controller=controller,
        env_adapter=adapter,
        station_sampler=sampler,
        path_planner=planner,
        base_driver=driver,
        config=config,
    )


def build_navigation_controller(
    env: Any,
    obs: dict[str, Any],
    *,
    nav_config: NavigationConfig | None = None,
    seed: int = 0,
    base_driver: Any | None = None,
) -> NavigationController:
    return build_navigation_assembly(
        env,
        obs,
        nav_config=nav_config,
        seed=seed,
        base_driver=base_driver,
    ).controller


def _build_station_sampler(
    adapter: RoboCasaNavigationAdapter,
    config: NavigationConfig,
    *,
    seed: int,
    base_pose_provider: Any,
) -> Any:
    aliases = station_aliases()
    if bool(config.use_station_base_poses):
        source_sampler = SourcePlateStationSampler(
            StationPoseSampler.from_file(
                config.station_base_pose_file,
                config=config,
                seed=seed,
                base_xy_provider=base_pose_provider,
                station_aliases=aliases,
            ),
            adapter.env,
        )
        return source_sampler

    string_aliases = {
        alias: station_id
        for alias, station_id in aliases.items()
        if isinstance(alias, str)
    }
    generated_specs = station_specs_from_candidates(
        adapter.station_candidates(),
        aliases=string_aliases,
    )
    _apply_task_calibrated_station_specs(generated_specs, adapter.env, string_aliases)
    return SimpleStationSampler(
        stations=generated_specs,
        seeds=SeedStream(base_seed=int(seed), namespace="wr_07_navigation"),
    )


def _apply_task_calibrated_station_specs(
    specs: dict[str, StationSpec],
    env: Any,
    aliases: dict[str, str],
) -> None:
    """Add aliases for the two task-owned stations."""
    for alias, station_id in aliases.items():
        if station_id in specs:
            specs[str(alias)] = specs[station_id]


def _finite_xy(value: Any, *, label: str) -> np.ndarray:
    try:
        xy = np.asarray(value, dtype=float).reshape(-1)[:2]
    except Exception as exc:
        raise ValueError(f"invalid {label}: {value!r}") from exc
    if xy.size != 2 or not np.all(np.isfinite(xy)):
        raise ValueError(f"invalid {label}: {value!r}")
    return xy.astype(float)


__all__ = [
    "SourcePlateStationSampler",
    "NavigationAssembly",
    "StationHeightBaseDriver",
    "build_navigation_assembly",
    "build_navigation_config",
    "build_navigation_controller",
    "move_station_torso_to_height",
]
