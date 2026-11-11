"""Two identity-independent navigation stations for WR-11."""

from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np

from bench.execution.contracts import StageKind, StageResult
from bench.execution.manipulation.object_interactions import get_object_hold_status
from bench.execution.navigation import (
    BaseDriverAdapter,
    NavigationController,
)
from bench.execution.navigation.base_motion import base_yaw, qpos_for_controller
from bench.execution.navigation.contracts import Pose
from bench.execution.navigation.rotation_primitive import move_straight
from bench.execution.navigation.station_poses import yaw_to_quat_xyzw
from bench.execution.navigation.velocity_profile import BaseVelocityLimits
from bench.execution.planning import NavigationConfig
from bench.execution.planning.nav_algo import normalize_angle


class EpisodeStationSampler:
    """Reuse resolved station draws across observation, pickup and return."""

    def __init__(self, plan):
        self.plan = plan
        self._summaries = {}

    def sample(self, station_name, *, purpose=None):
        if station_name not in self.plan.base_xy_by_station:
            raise ValueError(f"unknown WR-11 station: {station_name}")
        xy = self.plan.base_xy_by_station[station_name]
        self._summaries[station_name] = {
            "station_id": station_name,
            "purpose": purpose,
            "target_xy": list(xy),
            "target_yaw": self.plan.base_yaw,
            "station_pose_selection": {
                "policy": "episode_uniform_area_disk",
                "episode_stable": True,
                "seed": self.plan.seed,
                "sampling_profile_id": self.plan.sampling_profile_id,
            },
        }
        return Pose(
            position=(*xy, 0.0), quaternion=yaw_to_quat_xyzw(self.plan.base_yaw), torso_height=0.0
        )

    def last_summary(self, station_name):
        return self._summaries.get(station_name)

    def include_points(self):
        return [np.asarray(xy) for xy in self.plan.base_xy_by_station.values()]


class TranslationDriver(BaseDriverAdapter):
    """Translate between island regions while retaining the starting heading."""

    def plan_and_move(
        self, target_xy, target_yaw, *, holding=None, planning_summary=None, **kwargs
    ):
        config, env = self._nav_config, self._env
        self.refresh_obs()
        initial_yaw = base_yaw(self._obs)
        # Both stations face the same island edge. A contrary calibration must
        # fail explicitly instead of silently adding a turn to the demonstration.
        if (
            target_yaw is not None
            and abs(normalize_angle(target_yaw - initial_yaw)) > config.base_yaw_tolerance
        ):
            return StageResult(
                index=-1,
                kind=StageKind.NAVIGATE,
                success=False,
                failure_code="translation_station_heading_mismatch",
                details={"initial_yaw": initial_yaw, "requested_yaw": target_yaw},
            )
        arm = env.robots[0].composite_controller.part_controllers["right"]
        arm_qpos = qpos_for_controller(env, arm)
        limits = BaseVelocityLimits.from_navigation_config(config)
        if holding is not None:
            limits = replace(
                limits,
                v_max=min(limits.v_max, 0.30),
                a_max=min(limits.a_max, 0.60),
                a_dec=min(limits.a_dec, 0.65),
            )
        self._obs, summary = move_straight(
            env,
            self._obs,
            target_xy=np.asarray(target_xy),
            arm_qpos=arm_qpos,
            gripper_action=1.0 if holding is not None else -1.0,
            tolerance=config.base_tolerance,
            max_steps=config.base_max_steps,
            action_frame=config.base_action_frame,
            translation_action_sign=config.base_translation_action_sign,
            velocity_limits=limits,
            maintain_yaw=True,
        )
        yaw_error = abs(normalize_angle(base_yaw(self._obs) - initial_yaw))
        heading_safe = yaw_error <= config.base_yaw_tolerance
        hold, hold_safe = None, True
        if holding is not None:
            hold = get_object_hold_status(env, holding)
            baseline = env._initial_positions[holding]
            hold_safe = bool(
                hold["held"] and env._obj_pos(holding)[2] - baseline[2] >= env.minimum_lift_m
            )
        passed = bool(summary["pass"] and heading_safe and hold_safe)
        return StageResult(
            index=-1,
            kind=StageKind.NAVIGATE,
            success=passed,
            failure_code=None if passed else "translation_failed",
            details={
                "pass": passed,
                "navigation_mode": "fixed_heading_translation",
                "translation": summary,
                "heading_error_rad": yaw_error,
                "heading_safe": heading_safe,
                "hold": hold,
                "hold_safe": hold_safe,
                "station_path": planning_summary or {},
                "steps": summary["steps"],
            },
        )


def station_specs(env):
    return {
        name: {
            "station_id": name,
            "fixture": fixture,
            "anchor_xy": fixture.pos[:2],
            "source_fixture": fixture.name,
        }
        for name, fixture in (
            ("sample_station", env.sample_counter),
            ("stock_station", env.stock_counter),
        )
    }


def build_navigation_config(settings, *, manual_base_path_file=None, station_base_pose_file):
    settings = dict(settings)
    if set(settings) - {field.name for field in fields(NavigationConfig)}:
        raise ValueError("unknown WR-11 navigation settings")
    if settings.get("kinematic_base_navigation", False):
        raise ValueError("WR-11 requires physical base navigation")
    if not settings.get("use_station_base_poses", True):
        raise ValueError("WR-11 requires calibrated station poses")
    return replace(NavigationConfig(), **settings, station_base_pose_file=station_base_pose_file)


def build_navigation_assembly(env, obs, *, nav_config, episode_plan):
    driver = TranslationDriver(env, obs, nav_config=nav_config)
    sampler = EpisodeStationSampler(episode_plan)
    controller = NavigationController(
        station_sampler=sampler, base_driver=driver, path_planner=None, base_pose_provider=driver
    )
    return SimpleNamespace(controller=controller, base_driver=driver, station_sampler=sampler)
