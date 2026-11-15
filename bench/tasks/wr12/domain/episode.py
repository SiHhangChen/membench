"""Balanced hidden locations and independent per-episode pose sampling."""

import json
import random
from math import cos, isfinite, pi, sin, sqrt
from pathlib import Path

from bench.execution.sampling.circle_sampling import sample_uniform_disk, sample_uniform_yaw

from .contracts import BOX_IDS, ENV_NAME, EpisodePlan, SceneConfig

RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources"
STATION_SALTS = {"observe_station": 0x57523120, "operate_station": 0x57523121}


def load_sampling_rules(path=None):
    rules = json.loads(Path(path or RESOURCE_ROOT / "sampling/regions.json").read_text())["entries"]
    if set(rules) != {"profile_id", "block_radius_m", "placement_radius_m"}:
        raise ValueError("WR-12 sampling keys mismatch")
    if rules["profile_id"] != "WR-12/sampling-v2":
        raise ValueError("WR-12 sampling profile mismatch")
    for name in ("block_radius_m", "placement_radius_m"):
        if not isfinite(rules[name]) or not 0 <= rules[name] <= 0.045:
            raise ValueError(f"WR-12 {name} must lie within [0, 0.045] m")
    return rules


def load_station_circles(path=None):
    payload = json.loads(Path(path or RESOURCE_ROOT / "calibration/stations.json").read_text())
    if (payload["env_name"], payload["layout_id"], payload["style_id"]) != (ENV_NAME, 33, 34):
        raise ValueError("WR-12 station identity mismatch")
    circles = {name: value["circle"] for name, value in payload["stations"].items()}
    if set(circles) != set(STATION_SALTS):
        raise ValueError("WR-12 requires observe and operate stations")
    for circle in circles.values():
        if len(circle["center_xy"]) != 2 or not all(isfinite(x) for x in circle["center_xy"]):
            raise ValueError("WR-12 station center must be finite XY")
        if not isfinite(circle["radius_m"]) or not 0 <= circle["radius_m"] <= 0.05:
            raise ValueError("WR-12 station radius outside [0, 0.05] m")
        if not isfinite(circle["base_yaw"]):
            raise ValueError("WR-12 station yaw must be finite")
        half_range = circle["yaw_half_range_rad"]
        if not isfinite(half_range) or not 0 <= half_range <= pi / 36:
            raise ValueError("WR-12 station yaw range outside [0, 5 degrees]")
    return circles


def resolve_scene_config(*, seed, env_split, **values):
    del seed
    if values != {
        "env_name": ENV_NAME,
        "fixed_scene": True,
        "layout_id": 33,
        "style_id": 34,
        "scene_seed": 0,
    }:
        raise ValueError("WR-12 requires layout/style/scene-seed 33/34/0")
    return SceneConfig(ENV_NAME, str(env_split))


def _disk(seed, label, radius):
    rng = random.Random(f"wr12:{seed}:{label}")
    r, angle = radius * sqrt(rng.random()), rng.uniform(-pi, pi)
    return (r * cos(angle), r * sin(angle))


def resolve_episode_plan(
    *, seed, max_horizon=2400, sampling_rules=None, station_circles=None, **unknown
):
    if unknown or isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("WR-12 requires a nonnegative integer seed and known settings")
    seed = int(seed)
    if int(max_horizon) != max_horizon or max_horizon <= 0:
        raise ValueError("WR-12 horizon must be a positive integer")
    pair = seed // 2
    rules = load_sampling_rules() if sampling_rules is None else sampling_rules
    circles = load_station_circles() if station_circles is None else station_circles
    # Pair only the answer order. Poses use the full seed, so each episode has
    # an independent layout while adjacent answers remain balanced.
    first = random.Random(f"wr12:{pair}:target_order").randrange(2)
    positions, headings = {}, {}
    for name, circle in circles.items():
        positions[name] = tuple(
            float(value)
            for value in sample_uniform_disk(
                circle["center_xy"], circle["radius_m"], seed=seed, salt=STATION_SALTS[name]
            )
        )
        headings[name] = sample_uniform_yaw(
            circle["base_yaw"],
            circle["yaw_half_range_rad"],
            seed=seed,
            salt=STATION_SALTS[name] + 0x100,
        )
    return EpisodePlan(
        seed=int(seed),
        target_box=BOX_IDS[(first + seed % 2) % 2],
        pair_id=pair,
        block_offset_xy=_disk(seed, "block", rules["block_radius_m"]),
        block_yaw=random.Random(f"wr12:{seed}:block_yaw").uniform(-pi, pi),
        placement_offset_xy=_disk(seed, "placement", rules["placement_radius_m"]),
        base_xy_by_station=positions,
        base_yaw_by_station=headings,
        base_radius_by_station={name: c["radius_m"] for name, c in circles.items()},
        sampling_profile_id=rules["profile_id"],
        max_horizon=int(max_horizon),
    )
