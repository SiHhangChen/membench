import json
from types import SimpleNamespace

import numpy as np
import pytest

from bench.execution.navigation.contracts import Pose
from bench.execution.navigation.station_poses import (
    quat_xyzw_to_yaw,
    yaw_to_quat_xyzw,
)
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest
from bench.tasks.wa05.execution import handlers
from bench.tasks.wa05.execution.navigation import (
    CUP_PICK_STATION,
    SERVICE_AREA_STATION,
    SINK_FILL_STATION,
    SampledPathBaseDriver,
    build_episode_station_sampler,
)
from bench.tasks.wa05.initialization import apply_robot_base_initial_state


STATION_INDICES = (4, 6, 8, 10)
TURN_INDICES = (5, 7)
ACTIVE_ROUTE_INDICES = (4, 5, 6, 7, 8, 10)


def _descriptor(seed: int = 0):
    episode = TaskCatalog.resolve(EpisodeRequest(task_id="WA-05", seed=seed))
    api = resolve_builtin_task_plugin("WA-05").load_runtime_api()
    return api.build_descriptor(episode, False)


def _sampler(seed: int = 0):
    descriptor = _descriptor(seed)
    return descriptor, build_episode_station_sampler(
        descriptor.navigation,
        seed=descriptor.episode_plan.seed,
    )


def test_wa05_route_has_four_sampled_stations_and_two_turns() -> None:
    descriptor, sampler = _sampler()
    station_payload = sampler.payload["stations"]
    for station_id in (CUP_PICK_STATION, SINK_FILL_STATION, SERVICE_AREA_STATION):
        assert station_payload[station_id]["circle"]["radius_m"] == 0.05

    path_payload = json.loads(
        descriptor.navigation.manual_base_path_file.read_text(encoding="utf-8")
    )
    by_index = {int(item["index"]): item for item in path_payload["samples"]}
    assert by_index[6]["circle"]["radius_m"] == 0.05
    assert by_index[6]["circle"]["yaw_half_range_rad"] == 0.0
    assert {index: by_index[index]["station_id"] for index in (4, 8, 10)} == {
        4: CUP_PICK_STATION,
        8: SINK_FILL_STATION,
        10: SERVICE_AREA_STATION,
    }
    for index in TURN_INDICES:
        turn = by_index[index]["relative_turn"]
        assert abs(turn["nominal_angle_rad"]) == pytest.approx(np.pi / 2.0)
        assert turn["angle_half_range_rad"] == pytest.approx(np.deg2rad(5.0))
        assert "circle" not in by_index[index]
    assert by_index[9]["active"] is False
    assert set(sampler.waypoint_station_ids) == set(STATION_INDICES)
    assert set(sampler.route_poses) == set(ACTIVE_ROUTE_INDICES)


def test_episode_path_draws_are_bounded_reproducible_and_seed_owned() -> None:
    first_descriptor, first = _sampler(11)
    _, repeat = _sampler(11)
    _, other = _sampler(12)
    assert first_descriptor.scene.scene_seed == 0

    changed = []
    for index in STATION_INDICES:
        station_id = first.waypoint_station_ids[index]
        one = first.sample(station_id)
        two = repeat.sample(station_id)
        three = other.sample(other.waypoint_station_ids[index])
        circle = first.payload["stations"][station_id]["circle"]
        assert one == two
        assert first.sample(station_id) == one
        assert np.linalg.norm(
            np.asarray(one.position[:2]) - np.asarray(circle["center_xy"])
        ) <= 0.05
        changed.append(one.position != three.position)
    assert all(changed)


def test_turn_angles_randomize_while_station_torso_stays_fixed() -> None:
    _, sampler = _sampler(3)
    _, another_seed = _sampler(4)
    expected_torso_heights = {
        CUP_PICK_STATION: 0.19820560131301382,
        SINK_FILL_STATION: 0.01,
        SERVICE_AREA_STATION: 0.01,
    }
    for station_id, expected_torso in expected_torso_heights.items():
        pose = sampler.sample(station_id)
        assert pose.torso_height == pytest.approx(expected_torso)
        assert another_seed.sample(station_id).torso_height == pytest.approx(
            expected_torso
        )
    left = sampler.route_summaries[5]["sampled_turn_angle"]
    right = sampler.route_summaries[7]["sampled_turn_angle"]
    assert abs(left - np.pi / 2.0) <= np.deg2rad(5.0)
    assert abs(right + np.pi / 2.0) <= np.deg2rad(5.0)
    assert sampler.route_poses[5].position == sampler.route_poses[4].position
    assert sampler.route_poses[7].position == sampler.route_poses[6].position
    assert quat_xyzw_to_yaw(sampler.route_poses[6].quaternion) == pytest.approx(
        quat_xyzw_to_yaw(sampler.route_poses[5].quaternion)
    )
    assert quat_xyzw_to_yaw(sampler.route_poses[8].quaternion) == pytest.approx(
        quat_xyzw_to_yaw(sampler.route_poses[7].quaternion)
    )
    assert quat_xyzw_to_yaw(sampler.route_poses[10].quaternion) == pytest.approx(
        quat_xyzw_to_yaw(sampler.route_poses[8].quaternion)
    )


def test_manual_path_executor_receives_all_sampled_waypoints() -> None:
    descriptor, sampler = _sampler(5)
    driver = SampledPathBaseDriver(
        SimpleNamespace(control_timestep=0.05),
        {
            "robot0_base_pos": np.zeros(3),
            "robot0_base_quat": np.asarray(yaw_to_quat_xyzw(0.0)),
        },
        config=descriptor.navigation,
        sampler=sampler,
    )
    executor = driver._manual_path_executor()
    for index in ACTIVE_ROUTE_INDICES:
        pose = sampler.route_poses[index]
        np.testing.assert_allclose(executor.samples[index].xy, pose.position[:2])
        assert executor.samples[index].yaw == pytest.approx(
            quat_xyzw_to_yaw(pose.quaternion)
        )
    target_index, resolution = executor.target_index_for(
        "service_area",
        purpose="slot_1",
        holding="cup_A",
    )
    assert target_index == 10
    assert executor._explicit_route(8, 10, target_resolution=resolution) == [8, 10]


def test_initial_base_uses_sampled_pick_pose_and_fixed_torso(monkeypatch) -> None:
    import bench.tasks.wa05.initialization as initialization

    descriptor, sampler = _sampler(9)
    expected = sampler.sample(CUP_PICK_STATION)
    env = SimpleNamespace(
        observation={
            "robot0_base_pos": np.asarray([0.0, 0.0, 0.9]),
            "robot0_base_quat": np.asarray(yaw_to_quat_xyzw(0.0)),
        }
    )
    env._get_observations = lambda **kwargs: env.observation

    def set_pose(env, *, target_xy, target_yaw):
        env.observation = {
            "robot0_base_pos": np.asarray([*target_xy, 0.9]),
            "robot0_base_quat": np.asarray(yaw_to_quat_xyzw(target_yaw)),
        }
        return env.observation

    monkeypatch.setattr(initialization, "set_base_pose_kinematic", set_pose)
    monkeypatch.setattr(
        initialization,
        "move_station_torso_to_height",
        lambda env, height: {"requested_height": height},
    )
    observation, summary = apply_robot_base_initial_state(
        env,
        env.observation,
        config=descriptor.config.initial_state.robot_base,
        nav_config=descriptor.navigation,
        episode=descriptor.episode_plan,
        initial_objects={},
        seed=descriptor.episode_plan.seed,
    )
    np.testing.assert_allclose(observation["robot0_base_pos"][:2], expected.position[:2])
    assert quat_xyzw_to_yaw(observation["robot0_base_quat"]) == pytest.approx(
        quat_xyzw_to_yaw(expected.quaternion)
    )
    assert summary["station_torso"]["requested_height"] == pytest.approx(
        0.19820560131301382
    )


def test_sink_exit_direction_uses_sampled_service_target(monkeypatch) -> None:
    captured = {}
    waypoint = Pose(
        position=(1.57, -2.24, 0.0),
        quaternion=yaw_to_quat_xyzw(-0.039),
    )
    controller = SimpleNamespace(
        env=SimpleNamespace(),
        runtime_config=SimpleNamespace(execute=True),
        obs={
            "robot0_base_pos": np.asarray([1.59, -3.05, 0.9]),
            "robot0_base_quat": np.asarray(yaw_to_quat_xyzw(-0.038)),
        },
    )
    monkeypatch.setattr(handlers, "_refresh", lambda value: value.env)
    monkeypatch.setattr(
        handlers,
        "_counter_local_position",
        lambda env, world: np.asarray([-world[1], world[0], world[2]]),
    )
    monkeypatch.setattr(
        handlers,
        "lift_held_cup_after_sink_exit",
        lambda controller, **kwargs: captured.update(kwargs) or {"pass": True},
    )
    operation = handlers.build_stage_handlers(
        controller,
        sink_exit_target_provider=lambda: waypoint,
    )["lift_held_cup_after_sink_exit"]

    result = operation("cup_A", route_local_y_per_x=999.0)

    assert result["pass"]
    assert result["sampled_service_target_xy"] == [1.57, -2.24]
    assert captured["route_local_y_per_x"] == pytest.approx((-0.02) / (-0.81))
