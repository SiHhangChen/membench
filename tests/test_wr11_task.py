from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest, ExpertPlanInputs
from bench.tasks.wr11.domain.contracts import PACKAGING_RGB
from bench.tasks.wr11.domain.episode import resolve_episode_plan
from bench.tasks.wr11.domain.sampling import load_sampling_rules, load_station_circles
from bench.tasks.wr11.visibility import sphere_in_camera


def test_request_is_not_encoded_in_stock_layout_pose_or_prompt():
    requests_by_layout = defaultdict(set)
    prompts = set()
    for seed in range(240):
        episode = TaskCatalog.resolve(EpisodeRequest(task_id="wr11", seed=seed))
        plan = resolve_episode_plan(seed=seed)
        assert resolve_episode_plan(seed=seed) == plan
        colors = set(plan.packaging_by_item.values())
        assert len(colors) == 3 and colors <= set(PACKAGING_RGB)
        assert plan.packaging_by_item[plan.matching_item] == plan.sample_packaging
        requests_by_layout[tuple(plan.packaging_by_item.items())].add(plan.sample_packaging)
        prompts.add(episode.public["prompt"])
    assert len(prompts) == 1
    # A fixed candidate layout must permit multiple requests, so the layout
    # alone cannot identify the target. Audit enough draws for the larger pool.
    requests_by_layout.clear()
    requests, targets = Counter(), Counter()
    for seed in range(10000):
        plan = resolve_episode_plan(seed=seed)
        colors = tuple(plan.packaging_by_item.values())
        assert len(set(colors)) == 3
        assert colors.count(plan.sample_packaging) == 1
        requests_by_layout[colors].add(plan.sample_packaging)
        requests[plan.sample_packaging] += 1
        targets[plan.matching_item] += 1
    assert len(PACKAGING_RGB) == 8 and set(requests) == set(PACKAGING_RGB)
    assert len(requests_by_layout) > 300
    assert sum(len(values) >= 2 for values in requests_by_layout.values()) > 320
    assert all(1000 < count < 1500 for count in requests.values())
    assert all(3000 < count < 3700 for count in targets.values())


def test_catalog_aliases_and_expert_share_descriptor_actions():
    episode = TaskCatalog.resolve(EpisodeRequest(task_id="WR-11", seed=3))
    api = resolve_builtin_task_plugin("wr_11").load_runtime_api()
    descriptor = api.build_descriptor(episode, False)
    actions = replace(descriptor.actions, source_pick={"test_override": 1})
    plan = TaskCatalog.get("wr11").build_expert_plan(
        replace(ExpertPlanInputs.from_descriptor(descriptor), action_config=actions)
    )
    pick = next(stage for stage in plan.stages if getattr(stage, "operation", None) == "pick")
    assert pick.target == episode.facts["matching_item"]
    assert pick.metadata == {"test_override": 1}
    assert api.report_stage_profile_id == "WR-11/report-v1"


@pytest.mark.parametrize(
    "point,radius,expected",
    [
        ((0, 0, -1), 0.1, True),
        ((0, 0, 1), 0.1, False),
        ((3, 0, -1), 0.1, False),
        ((1.03, 0, -1), 0.1, True),
        ((0, 3, -1), 0.1, False),
        ((0, 0, 0.01), 0.1, True),
    ],
)
def test_sample_offscreen_requires_entire_geometry_outside_frustum(point, radius, expected):
    assert sphere_in_camera(point, radius, vertical_fov_deg=90) is expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": -1},
        {"seed": 1, "max_horizon": 0},
        {"seed": 1, "unknown_setting": 0},
    ],
)
def test_invalid_episode_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        resolve_episode_plan(**kwargs)


def test_each_object_yaw_and_station_draw_is_independent_of_offset_radius():
    rules = load_sampling_rules()
    zero_radius = deepcopy(rules)
    zero_radius["objects"]["radius_m"] = 0
    normal = resolve_episode_plan(seed=12, sampling_rules=rules)
    zero = resolve_episode_plan(seed=12, sampling_rules=zero_radius)
    assert normal.sample_packaging == zero.sample_packaging
    assert normal.packaging_by_item == zero.packaging_by_item
    assert normal.yaw_by_item == zero.yaw_by_item
    assert normal.base_xy_by_station == zero.base_xy_by_station
    assert all(offset == (0.0, 0.0) for offset in zero.offset_xy_by_item.values())


def test_reset_observation_and_return_reuse_the_resolved_sample_station():
    from bench.tasks.wr11.execution.navigation import EpisodeStationSampler

    plan = resolve_episode_plan(seed=13)
    sampler = EpisodeStationSampler(plan)
    observe = sampler.sample("sample_station", purpose="observe")
    stock = sampler.sample("stock_station", purpose="pick")
    returned = sampler.sample("sample_station", purpose="deliver")
    assert observe == returned
    assert observe.position[:2] == plan.base_xy_by_station["sample_station"]
    assert stock.position[:2] == plan.base_xy_by_station["stock_station"]
    assert sampler.last_summary("sample_station")["station_pose_selection"]["episode_stable"]


def test_configured_episode_records_all_sampling_and_stays_in_support():
    circles = load_station_circles()
    for seed in range(200):
        episode = TaskCatalog.resolve(EpisodeRequest(task_id="WR11", seed=seed))
        api = resolve_builtin_task_plugin("WR11").load_runtime_api()
        plan = api.build_descriptor(episode, False).episode_plan
        assert episode.facts["base_xy_by_station"] == {
            name: list(xy) for name, xy in plan.base_xy_by_station.items()
        }
        for name, xy in plan.base_xy_by_station.items():
            assert (
                np.linalg.norm(np.array(xy) - circles[name]["center_xy"])
                <= circles[name]["radius_m"]
            )
        assert all(np.linalg.norm(offset) <= 0.025 for offset in plan.offset_xy_by_item.values())
        assert all(-np.pi <= yaw <= np.pi for yaw in plan.yaw_by_item.values())


@pytest.mark.parametrize(
    "kind", ["large_object_radius", "large_station_radius", "station_heading", "region_mismatch"]
)
def test_sampling_rejects_uncalibrated_ranges_and_mismatched_regions(tmp_path, kind):
    import json

    from bench.tasks.wr11.domain.sampling import DEFAULT_RULES_FILE, DEFAULT_STATIONS_FILE

    source = DEFAULT_RULES_FILE if kind == "large_object_radius" else DEFAULT_STATIONS_FILE
    value = json.loads(source.read_text())
    if kind == "large_object_radius":
        value["entries"]["objects"]["radius_m"] = 0.045
    else:
        station = value["stations"]["stock_station"]
        if kind == "large_station_radius":
            station["circle"]["radius_m"] = 0.10
        elif kind == "station_heading":
            station["circle"]["base_yaw"] = np.pi / 2
        else:
            station["sampling_region"]["radius_m"] = 0.01
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        (load_sampling_rules if kind == "large_object_radius" else load_station_circles)(path)
