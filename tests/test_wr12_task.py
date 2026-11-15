from dataclasses import asdict

import numpy as np
import pytest

from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest, ExpertPlanInputs
from bench.tasks.layout import validate_placeholder_resources, validate_task_package
from bench.tasks.wr12.domain.episode import (
    load_sampling_rules,
    load_station_circles,
    resolve_episode_plan,
)
from bench.tasks.wr12.execution.navigation import EpisodeStationSampler
from bench.tasks.wr12.visibility import box_shell_occludes, segment_intersects_box


def test_target_pairs_are_balanced_with_independent_episode_poses():
    for seed in range(0, 200, 2):
        left, right = (asdict(resolve_episode_plan(seed=s)) for s in (seed, seed + 1))
        assert left["target_box"] != right["target_box"]
        assert left["pair_id"] == right["pair_id"]
        for name in (
            "block_offset_xy",
            "block_yaw",
            "placement_offset_xy",
            "base_xy_by_station",
            "base_yaw_by_station",
        ):
            assert left[name] != right[name]


def test_pose_supports_align_with_transport_tasks_and_use_uniform_area():
    circles = load_station_circles()
    plans = [resolve_episode_plan(seed=seed, station_circles=circles) for seed in range(2000)]
    offsets = {
        "block": (0.045, [p.block_offset_xy for p in plans]),
        "placement": (0.045, [p.placement_offset_xy for p in plans]),
        **{
            name: (
                circle["radius_m"],
                [np.array(p.base_xy_by_station[name]) - circle["center_xy"] for p in plans],
            )
            for name, circle in circles.items()
        },
    }
    for radius, values in offsets.values():
        normalized_squared_radius = np.sum(np.array(values) ** 2, axis=1) / radius**2
        assert np.max(normalized_squared_radius) <= 1 + 1e-12
        assert np.mean(normalized_squared_radius) == pytest.approx(0.5, abs=0.035)
    assert all(-np.pi <= p.block_yaw <= np.pi for p in plans)
    for name, circle in circles.items():
        delta = np.array([p.base_yaw_by_station[name] - circle["base_yaw"] for p in plans])
        assert np.max(np.abs(delta)) <= np.deg2rad(5) + 1e-12
        assert delta.min() < -np.deg2rad(4.5)
        assert delta.max() > np.deg2rad(4.5)


def test_random_streams_and_repeated_navigation_do_not_change_other_draws():
    seed = 25
    baseline = resolve_episode_plan(seed=seed)
    rules = load_sampling_rules()
    reduced = resolve_episode_plan(seed=seed, sampling_rules={**rules, "block_radius_m": 0.02})
    full, changed = asdict(baseline), asdict(reduced)
    full.pop("block_offset_xy")
    changed.pop("block_offset_xy")
    assert full == changed
    circles = load_station_circles()
    without_yaw = resolve_episode_plan(
        seed=seed,
        station_circles={name: {**c, "yaw_half_range_rad": 0} for name, c in circles.items()},
    )
    assert without_yaw.base_xy_by_station == baseline.base_xy_by_station
    sampler = EpisodeStationSampler(baseline)
    first = sampler.sample("observe_station", purpose="observe")
    sampler.sample("operate_station", purpose="pick")
    assert sampler.sample("observe_station", purpose="return") == first
    assert resolve_episode_plan(seed=seed) == baseline


def test_catalog_runtime_and_expert_contract():
    prompts = set()
    for seed in (0, 1):
        e = TaskCatalog.resolve(EpisodeRequest(task_id="wr12", seed=seed))
        api = resolve_builtin_task_plugin("wr_12").load_runtime_api()
        d = api.build_descriptor(e, False)
        plan = TaskCatalog.get("wr12").build_expert_plan(ExpertPlanInputs.from_descriptor(d))
        prompts.add(e.public["prompt"])
        assert len(plan.stages) == 7
        assert plan.stages[3].target == "placement_block"
        assert api.report_stage_profile_id == "WR-12/report-v2"
        assert "target_box" not in e.public
    assert len(prompts) == 1
    assert not validate_task_package("bench/tasks/wr12")
    validate_placeholder_resources("bench/tasks/wr12")


def test_occlusion_requires_panel_before_target_and_covers_corners():
    origin = np.array([0.0, 0.0, 0.0])
    end = np.array([0.0, 0.0, -1.0])
    half = np.array([0.2, 0.2, 0.02])
    assert segment_intersects_box(origin, end, np.array([0, 0, -0.5]), np.eye(3), half)
    assert not segment_intersects_box(origin, end, np.array([0, 0, -1.5]), np.eye(3), half)
    assert not segment_intersects_box(
        origin, np.array([0.5, 0.5, -1.0]), np.array([0, 0, -0.5]), np.eye(3), half
    )


def test_sealed_shell_blocks_seams_but_not_the_observation_opening():
    from itertools import product

    center = np.array([0.08, 0, 0.057])
    corners = np.array(
        [center + np.array(sign) * [0.031, 0.031, 0.041] for sign in product((-1, 1), repeat=3)]
    )
    assert box_shell_occludes(np.array([0, 0.5, 0.5]), corners, np.zeros(3))
    assert box_shell_occludes(np.array([-0.49, 0.46, 0.48]), corners, np.zeros(3))
    assert not box_shell_occludes(np.array([-1.0, 0, 0.15]), corners, np.zeros(3))
    assert not box_shell_occludes(np.array([0, 0, 0.15]), corners, np.zeros(3))


@pytest.mark.parametrize("change", ["missing_roof", "transparent", "opening", "collision"])
def test_occlusion_proof_rejects_incompatible_assets(tmp_path, change):
    import xml.etree.ElementTree as ET
    from pathlib import Path

    from bench.tasks.wr12.visibility import validate_box_asset

    root = ET.parse(Path("bench/tasks/wr12/resources/assets/box.xml")).getroot()
    body = root.find('.//body[@name="object"]')
    if change == "missing_roof":
        body.remove(body.find('geom[@name="roof_visual"]'))
    elif change == "transparent":
        body.find('geom[@name="front_visual"]').set("rgba", "1 1 1 .5")
    elif change == "opening":
        body.find('geom[@name="opening_front_lip_visual"]').set("size", ".008 .01 .134")
    else:
        body.find('geom[@name="front_collision"]').set("contype", "0")
    path = tmp_path / "invalid.xml"
    ET.ElementTree(root).write(path)
    with pytest.raises(ValueError):
        validate_box_asset(path)


@pytest.mark.parametrize("change", ["transparent", "gap", "rotated", "collision", "unlocked"])
def test_seal_proof_rejects_incompatible_shutters(tmp_path, change):
    import xml.etree.ElementTree as ET
    from pathlib import Path

    from bench.tasks.wr12.visibility import validate_shutter_asset

    root = ET.parse(Path("bench/tasks/wr12/resources/assets/shutter.xml")).getroot()
    body = root.find('.//body[@name="object"]')
    if change == "transparent":
        body.find('geom[@name="panel_visual"]').set("rgba", "1 1 1 .5")
    elif change == "gap":
        body.find('geom[@name="panel_visual"]').set("size", ".012 .18 .15")
    elif change == "rotated":
        body.find('geom[@name="panel_visual"]').set("euler", "0 0 .1")
    elif change == "collision":
        body.find('geom[@name="panel_collision"]').set("contype", "0")
    else:
        body.set("mocap", "false")
    path = tmp_path / "invalid.xml"
    ET.ElementTree(root).write(path)
    with pytest.raises(ValueError):
        validate_shutter_asset(path)
