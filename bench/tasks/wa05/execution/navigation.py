"""WA-05 station mapping and mobile-base controller assembly."""

from __future__ import annotations

import json
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.navigation import (
    BaseDriverAdapter,
    NavigationController,
    RoboCasaNavigationAdapter,
    StationPathPlanner,
    StationPoseSampler,
    station_specs_from_candidates,
)
from bench.execution.navigation.base_motion import qpos_for_controller, step_base_and_hold_arm
from bench.execution.navigation.contracts import Pose
from bench.execution.navigation.env_adapter import fixture_xy, object_xy, safe_name
from bench.execution.navigation.manual_base_path import ManualBasePathSample
from bench.execution.navigation.station_poses import quat_xyzw_to_yaw, yaw_to_quat_xyzw
from bench.execution.planning import NavigationConfig
from bench.execution.planning.nav_algo import normalize_angle
from bench.execution.sampling.circle_sampling import sample_uniform_yaw
from bench.execution.sampling.rng import SeedStream
from bench.execution.sampling.station_sampler import SimpleStationSampler
from bench.tasks.wa05.domain.contracts import CUP_IDS, SERVICE_SLOTS


CUP_PICK_STATION = "cup_pick_station"
CUP_A_PICK_STATION = "cup_A_pick_station"
SINK_FILL_STATION = "sink_fill_station"
SERVICE_AREA_STATION = "service_area_station"
WATER_BUTTON_STATION = "water_button_station"


@dataclass
class EpisodeStationPoseSampler(StationPoseSampler):
    """Sample each work station and route waypoint once per episode."""

    waypoint_station_ids: dict[int, str] = field(default_factory=dict)
    route_poses: dict[int, Pose] = field(default_factory=dict)
    route_summaries: dict[int, dict[str, Any]] = field(default_factory=dict)
    _cached_poses: dict[str, tuple[Pose, dict[str, Any]]] = field(
        default_factory=dict,
        init=False,
    )

    def sample(self, station_name: str, *, purpose: str | None = None) -> Pose:
        station_id = self.station_id(station_name, purpose=purpose)
        if station_id not in self._cached_poses:
            pose = super().sample(station_name, purpose=purpose)
            summary = dict(self.last_summary(station_name) or {})
            summary["station_pose_selection"] = {
                **dict(summary.get("station_pose_selection") or {}),
                "episode_stable": True,
            }
            self._cached_poses[station_id] = pose, summary
        pose, summary = self._cached_poses[station_id]
        self._summary_by_station[station_name] = {
            **summary, "station_name": station_name, "purpose": purpose,
        }
        return pose

    def reset_choices(self) -> None:
        super().reset_choices()
        self._cached_poses.clear()
        self.route_poses.clear()
        self.route_summaries.clear()


def build_episode_station_sampler(
    config: NavigationConfig,
    *,
    seed: int,
    base_xy_provider: Any | None = None,
) -> EpisodeStationPoseSampler:
    sampler = EpisodeStationPoseSampler.from_file(
        config.station_base_pose_file,
        config=config,
        seed=int(seed),
        base_xy_provider=base_xy_provider,
        station_aliases=station_aliases(),
    )
    if config.manual_base_path_enabled and config.manual_base_path_file is not None:
        path = json.loads(config.manual_base_path_file.read_text(encoding="utf-8"))
        samples = sorted(path.get("samples", ()), key=lambda item: int(item["index"]))
        for item in samples:
            if item.get("active") is False:
                continue
            index = int(item["index"])
            if item.get("station_id") is not None:
                sampler.waypoint_station_ids[index] = str(item["station_id"])
            elif isinstance(item.get("circle"), dict):
                station_id = f"path_waypoint_{index}"
                sampler.payload.setdefault("stations", {})[station_id] = {
                    "circle": dict(item["circle"]), "samples": [],
                }
                sampler.waypoint_station_ids[index] = station_id
        for item in samples:
            if item.get("active") is False:
                continue
            index = int(item["index"])
            turn = item.get("relative_turn")
            if isinstance(turn, Mapping):
                _add_relative_turn_sample(sampler, index=index, turn=turn)
                continue
            station_id = sampler.waypoint_station_ids.get(index)
            if station_id is None:
                continue
            pose = sampler.sample(station_id)
            inherited_from = item.get("inherit_yaw_from")
            if inherited_from is not None:
                source = sampler.route_poses[int(inherited_from)]
                pose = _set_station_route_yaw(
                    sampler,
                    station_id=station_id,
                    pose=pose,
                    yaw=quat_xyzw_to_yaw(source.quaternion),
                    inherited_from=int(inherited_from),
                )
            sampler.route_poses[index] = pose
            sampler.route_summaries[index] = {
                "path_index": index,
                "route_role": "station",
                **dict(sampler.last_summary(station_id) or {}),
            }
    return sampler


def _add_relative_turn_sample(
    sampler: EpisodeStationPoseSampler,
    *,
    index: int,
    turn: Mapping[str, Any],
) -> None:
    source_index = int(turn["source_index"])
    source = sampler.route_poses[source_index]
    nominal_angle = float(turn["nominal_angle_rad"])
    half_range = float(turn["angle_half_range_rad"])
    sampled_angle = sample_uniform_yaw(
        nominal_angle,
        half_range,
        seed=int(sampler.selector.seed),
        salt=0x5455524E ^ int(zlib.crc32(f"wa05_turn_{index}".encode("utf-8"))),
    )
    source_yaw = quat_xyzw_to_yaw(source.quaternion)
    target_yaw = normalize_angle(source_yaw + sampled_angle)
    pose = Pose(
        position=source.position,
        quaternion=yaw_to_quat_xyzw(target_yaw),
    )
    sampler.route_poses[index] = pose
    sampler.route_summaries[index] = {
        "path_index": index,
        "route_role": "relative_turn",
        "source_index": source_index,
        "source_xy": list(source.position[:2]),
        "source_yaw": source_yaw,
        "nominal_turn_angle": nominal_angle,
        "turn_angle_half_range": half_range,
        "sampled_turn_angle": sampled_angle,
        "target_yaw": target_yaw,
        "record_source": turn.get("record_source"),
    }


def _set_station_route_yaw(
    sampler: EpisodeStationPoseSampler,
    *,
    station_id: str,
    pose: Pose,
    yaw: float,
    inherited_from: int,
) -> Pose:
    updated_pose = Pose(
        position=pose.position,
        quaternion=yaw_to_quat_xyzw(yaw),
        torso_height=pose.torso_height,
    )
    summary = dict(sampler.last_summary(station_id) or {})
    sample = dict(summary.get("sample") or {})
    sample["base_yaw"] = float(yaw)
    summary.update(
        {
            "target_yaw": float(yaw),
            "sample": sample,
            "yaw_inherited_from_path_index": int(inherited_from),
        }
    )
    sampler._cached_poses[station_id] = updated_pose, summary
    sampler._summary_by_station[station_id] = summary
    return updated_pose


class SampledPathBaseDriver(BaseDriverAdapter):
    """Execute the calibrated route with seed-owned waypoint poses."""

    def __init__(
        self,
        env: Any,
        obs: dict[str, Any],
        *,
        config: NavigationConfig,
        sampler: EpisodeStationPoseSampler,
    ) -> None:
        super().__init__(env, obs, nav_config=config)
        self._station_sampler = sampler
        self._sampled_executor: Any = None
        self._waypoint_summaries: dict[int, dict[str, Any]] = {}

    def _manual_path_executor(self) -> Any:
        executor = super()._manual_path_executor()
        if executor is None or executor is self._sampled_executor:
            return executor
        for index, pose in self._station_sampler.route_poses.items():
            if not 0 <= index < len(executor.samples):
                raise ValueError(f"WA-05 sampled waypoint {index} is outside the manual path")
            nominal = executor.samples[index]
            executor.samples[index] = ManualBasePathSample(
                index=index, xy=np.asarray(pose.position[:2], dtype=float),
                yaw=quat_xyzw_to_yaw(pose.quaternion), name=nominal.name,
            )
            self._waypoint_summaries[index] = {
                "path_index": index,
                "nominal_xy": nominal.xy.tolist(),
                "nominal_yaw": nominal.yaw,
                **dict(self._station_sampler.route_summaries[index]),
            }
        self._sampled_executor = executor
        return executor

    def plan_and_move(self, *args: Any, **kwargs: Any) -> Any:
        result = super().plan_and_move(*args, **kwargs)
        result.details["sampled_path_waypoints"] = [
            dict(self._waypoint_summaries[index])
            for index in result.details.get("route_indices", ())
            if index in self._waypoint_summaries
        ]
        return result


@dataclass(frozen=True)
class NavigationAssembly:
    """The task-local pieces around the shared navigation controller."""

    controller: NavigationController
    env_adapter: RoboCasaNavigationAdapter
    station_sampler: Any
    path_planner: StationPathPlanner | None
    base_driver: Any
    config: NavigationConfig


class StationHeightBaseDriver:
    """Apply each calibrated station's torso height after Base navigation."""

    def __init__(self, delegate: Any, env: Any, *, torso_mover: Any = None):
        self._delegate = delegate
        self._env = env
        self._torso_mover = torso_mover or move_station_torso_to_height

    def plan_and_move(
        self,
        target_xy: np.ndarray,
        target_yaw: float | None,
        *,
        holding: str | None = None,
        planning_summary: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        result = self._delegate.plan_and_move(
            target_xy,
            target_yaw,
            holding=holding,
            planning_summary=planning_summary,
            **kwargs,
        )
        if not bool(result.success):
            return result

        target_height = _station_torso_height(planning_summary)
        try:
            torso = self._torso_mover(
                self._env,
                target_height,
                holding=holding,
                post_step_callback=getattr(
                    self._delegate,
                    "_post_step_callback",
                    None,
                ),
            )
        except Exception as exc:
            result.success = False
            result.failure_code = f"station torso adjustment failed: {exc}"
            result.details["station_torso"] = {
                "applied": False,
                "requested_height": target_height,
                "reason": "station_torso_adjustment_failed",
                "error": str(exc),
            }
            return result

        result.details["station_torso"] = torso
        refresh = getattr(self._delegate, "refresh_obs", None)
        if callable(refresh) and bool(torso.get("applied", False)):
            refresh()
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


def move_station_torso_to_height(
    env: Any,
    target_height: float | None,
    *,
    holding: str | None = None,
    tolerance: float = 0.002,
    max_steps: int = 240,
    post_step_callback: Any = None,
) -> dict[str, Any]:
    """Move the lift column normally while holding arm and Base position."""

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


def station_aliases() -> dict[Any, str]:
    """Map workflow anchors to their task-owned station IDs."""

    return {
        CUP_PICK_STATION: CUP_PICK_STATION,
        CUP_A_PICK_STATION: CUP_PICK_STATION,
        SINK_FILL_STATION: SINK_FILL_STATION,
        SERVICE_AREA_STATION: SERVICE_AREA_STATION,
        WATER_BUTTON_STATION: WATER_BUTTON_STATION,
        (WATER_BUTTON_STATION, "hot_answer"): WATER_BUTTON_STATION,
        (WATER_BUTTON_STATION, "cold_answer"): WATER_BUTTON_STATION,
        f"{WATER_BUTTON_STATION}:hot_answer": WATER_BUTTON_STATION,
        f"{WATER_BUTTON_STATION}:cold_answer": WATER_BUTTON_STATION,
        "cup_A": CUP_PICK_STATION,
        "sink_basin": SINK_FILL_STATION,
        "sink": SINK_FILL_STATION,
        "service_area": SERVICE_AREA_STATION,
        "service_counter": SERVICE_AREA_STATION,
    }


def build_navigation_config(
    settings: dict[str, Any],
    *,
    manual_base_path_file: Path,
    station_base_pose_file: Path,
) -> NavigationConfig:
    """Convert the bounded YAML navigation surface into ``NavigationConfig``."""

    values = dict(settings)
    config = replace(
        NavigationConfig(),
        kinematic_base_navigation=bool(values.pop("kinematic_base_navigation", False)),
        base_action_frame=str(values.pop("base_action_frame", "world")),
        base_translation_action_sign=float(values.pop("base_translation_action_sign", 1.0)),
        base_tolerance=float(values.pop("base_tolerance", 0.08)),
        base_yaw_tolerance=float(values.pop("base_yaw_tolerance", 0.08)),
        base_max_steps=int(values.pop("base_max_steps", 1500)),
        manual_base_path_enabled=bool(values.pop("manual_base_path_enabled", True)),
        manual_base_path_file=Path(manual_base_path_file),
        manual_base_path_start_index=int(values.pop("manual_base_path_start_index", 1)),
        manual_base_path_xy_tolerance=float(
            values.pop("manual_base_path_xy_tolerance", 0.12)
        ),
        manual_base_path_yaw_tolerance=float(
            values.pop("manual_base_path_yaw_tolerance", 0.08)
        ),
        manual_base_path_segment_yaw_threshold=float(
            values.pop("manual_base_path_segment_yaw_threshold", 0.05)
        ),
        manual_base_path_intermediate_xy_tolerance=values.pop(
            "manual_base_path_intermediate_xy_tolerance", None,
        ),
        manual_base_path_intermediate_yaw_tolerance=values.pop(
            "manual_base_path_intermediate_yaw_tolerance", None,
        ),
        base_velocity_profile_enabled=bool(
            values.pop("base_velocity_profile_enabled", True)
        ),
        base_profile_v_max=float(values.pop("base_profile_v_max", 0.65)),
        base_profile_a_max=float(values.pop("base_profile_a_max", 1.6)),
        base_profile_a_dec=float(values.pop("base_profile_a_dec", 1.2)),
        base_profile_v_min=float(values.pop("base_profile_v_min", 0.22)),
        base_profile_w_max=float(values.pop("base_profile_w_max", 1.2)),
        base_profile_alpha_max=float(values.pop("base_profile_alpha_max", 3.0)),
        base_profile_alpha_dec=float(values.pop("base_profile_alpha_dec", 2.5)),
        base_profile_w_min=float(values.pop("base_profile_w_min", 0.20)),
        base_single_axis_tracking=bool(values.pop("base_single_axis_tracking", False)),
        base_snap_to_target_after_success=bool(
            values.pop("base_snap_to_target_after_success", False)
        ),
        use_station_astar_paths=bool(values.pop("use_station_astar_paths", False)),
        station_nav_use_docking=bool(values.pop("station_nav_use_docking", False)),
        station_nav_dock_standoff=float(values.pop("station_nav_dock_standoff", 0.0)),
        station_nav_require_footprint=bool(
            values.pop("station_nav_require_footprint", False)
        ),
        station_base_pose_file=Path(station_base_pose_file),
        use_station_base_poses=bool(values.pop("use_station_base_poses", False)),
    )
    if values:
        raise ValueError("unknown WA-05 navigation settings: " + ", ".join(sorted(values)))
    return config


def station_specs(env: Any) -> dict[str, dict[str, Any]]:
    """Read live scene anchors for non-manual navigation fallback only."""

    counter = getattr(env, "counter", None)
    sink = getattr(env, "sink", None)
    cup_points = {cup_id: object_xy(env, cup_id) for cup_id in CUP_IDS}
    cup_xy = _first_available_xy(
        _mean_xy(list(cup_points.values())),
        fixture_xy(counter),
    )
    water_xy = _first_available_xy(
        _xy_or_none(_water_site_position(env)),
        fixture_xy(sink),
    )
    service_points = [
        _xy_or_none(_service_slot_position(env, slot)) for slot in SERVICE_SLOTS
    ]
    service_xy = _first_available_xy(_mean_xy(service_points), fixture_xy(counter))
    return {
        CUP_PICK_STATION: {
            "station_id": CUP_PICK_STATION,
            "fixture": counter,
            "anchor_xy": cup_xy,
            "source_fixture": safe_name(counter),
            "source_object_or_phase": "cup_A",
            "stand_off": 0.16,
            "tangential_offsets": (-0.08, -0.04, 0.0, 0.04, 0.08),
        },
        CUP_A_PICK_STATION: {
            "station_id": CUP_A_PICK_STATION,
            "fixture": counter,
            "anchor_xy": _first_available_xy(cup_points.get("cup_A"), cup_xy),
            "source_fixture": safe_name(counter),
            "source_object_or_phase": "cup_A",
            "stand_off": 0.16,
            "tangential_offsets": (-0.08, -0.04, 0.0, 0.04, 0.08),
        },
        SINK_FILL_STATION: {
            "station_id": SINK_FILL_STATION,
            "fixture": sink or counter,
            "anchor_xy": water_xy,
            "source_fixture": safe_name(sink or counter),
            "source_object_or_phase": "sink_water_site",
            "stand_off": 0.24,
            "tangential_offsets": (-0.08, -0.04, 0.0, 0.04, 0.08),
        },
        SERVICE_AREA_STATION: {
            "station_id": SERVICE_AREA_STATION,
            "fixture": counter,
            "anchor_xy": service_xy,
            "source_fixture": safe_name(counter),
            "source_object_or_phase": "neutral_service_slots",
            "stand_off": 0.16,
            "tangential_offsets": (-0.08, -0.04, 0.0, 0.04, 0.08),
        },
    }


def build_navigation_assembly(
    env: Any,
    obs: dict[str, Any],
    *,
    config: NavigationConfig,
    seed: int,
    base_driver: Any | None = None,
) -> NavigationAssembly:
    """Assemble calibrated or generated station navigation without previous WA2 code."""

    adapter = RoboCasaNavigationAdapter(env, obs, station_specs_provider=station_specs)
    aliases = station_aliases()
    if config.use_station_base_poses:
        if config.station_base_pose_file is None:
            raise ValueError("WA-05 station poses require station_base_pose_file")
        sampler: Any = build_episode_station_sampler(
            config,
            seed=int(seed),
            base_xy_provider=adapter,
        )
        raw_driver = base_driver or SampledPathBaseDriver(
            env,
            obs,
            config=config,
            sampler=sampler,
        )
    else:
        sampler = SimpleStationSampler(
            stations=station_specs_from_candidates(
                adapter.station_candidates(),
                aliases={key: value for key, value in aliases.items() if isinstance(key, str)},
            ),
            seeds=SeedStream(base_seed=int(seed), namespace="wa_05_navigation"),
        )
        raw_driver = base_driver or BaseDriverAdapter(env, obs, nav_config=config)
    driver = StationHeightBaseDriver(raw_driver, env)
    pose_provider = driver if hasattr(driver, "current_base_xy") else adapter
    if isinstance(sampler, EpisodeStationPoseSampler):
        sampler.base_xy_provider = pose_provider
    planner = None
    if not config.manual_base_path_enabled and config.use_station_astar_paths:
        planner = StationPathPlanner(obstacles=adapter.build_fixture_obstacles, config=config)
    controller = NavigationController(
        station_sampler=sampler,
        base_driver=driver,
        path_planner=planner,
        base_pose_provider=pose_provider,
    )
    return NavigationAssembly(
        controller=controller,
        env_adapter=adapter,
        station_sampler=sampler,
        path_planner=planner,
        base_driver=driver,
        config=config,
    )


def _water_site_position(env: Any) -> np.ndarray | None:
    try:
        return np.asarray(env.water_site_position(), dtype=float)
    except Exception:
        return None


def _service_slot_position(env: Any, slot: str) -> np.ndarray | None:
    try:
        return np.asarray(env.service_slot_position(slot), dtype=float)
    except Exception:
        return None


def _xy_or_none(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size < 2 or not np.all(np.isfinite(array[:2])):
        return None
    return array[:2]


def _mean_xy(values: list[np.ndarray | None]) -> np.ndarray | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return np.mean(np.vstack(valid), axis=0)


def _first_available_xy(*values: np.ndarray | None) -> np.ndarray | None:
    """Return the first coordinate without evaluating arrays as booleans."""

    return next((value for value in values if value is not None), None)


__all__ = [
    "CUP_A_PICK_STATION",
    "CUP_PICK_STATION",
    "EpisodeStationPoseSampler",
    "NavigationAssembly",
    "SERVICE_AREA_STATION",
    "SINK_FILL_STATION",
    "SampledPathBaseDriver",
    "StationHeightBaseDriver",
    "WATER_BUTTON_STATION",
    "build_navigation_assembly",
    "build_navigation_config",
    "build_episode_station_sampler",
    "move_station_torso_to_height",
    "station_aliases",
    "station_specs",
]
