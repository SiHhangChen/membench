"""Independent seeded streams for the request, stock layout and object poses."""

import random
from math import cos, pi, sin, sqrt

from .contracts import ENV_NAME, ITEM_IDS, SAMPLE_ID, EpisodePlan, SceneConfig
from .sampling import load_sampling_rules, load_station_circles, sample_station_positions
from .scene import LAYOUT_ID, SCENE_SEED, STYLE_ID


def resolve_scene_config(*, seed, env_split, **values):
    del seed
    if values != {
        "env_name": ENV_NAME,
        "fixed_scene": True,
        "layout_id": LAYOUT_ID,
        "style_id": STYLE_ID,
        "scene_seed": SCENE_SEED,
    }:
        raise ValueError("WR-11 requires fixed layout/style/scene-seed 33/34/0")
    return SceneConfig(ENV_NAME, str(env_split))


def resolve_episode_plan(
    *, seed, max_horizon=3600, sampling_rules=None, station_circles=None, **unknown
):
    if unknown:
        raise ValueError(f"unknown WR-11 episode settings: {sorted(unknown)}")
    if int(seed) != seed or seed < 0:
        raise ValueError("WR-11 seed must be a non-negative integer")
    if int(max_horizon) != max_horizon or max_horizon <= 0:
        raise ValueError("WR-11 max_horizon must be a positive integer")
    rules = load_sampling_rules() if sampling_rules is None else sampling_rules
    circles = load_station_circles() if station_circles is None else station_circles
    radius = float(rules["objects"]["radius_m"])
    # Draw three distinct colors, then independently choose the request among
    # them. Given a stock layout, each candidate is equally likely to match;
    # across episodes each of the eight sample colors has probability 1/8.
    packaging = random.Random(f"wr11:{seed}:stock").sample(
        rules["request"]["packaging"], len(ITEM_IDS)
    )
    sample = random.Random(f"wr11:{seed}:request").choice(packaging)
    by_item = dict(zip(ITEM_IDS, packaging))
    offsets, yaws = {}, {}
    for item in (*ITEM_IDS, SAMPLE_ID):
        rng = random.Random(f"wr11:{seed}:offset:{item}")
        distance, angle = radius * sqrt(rng.random()), rng.uniform(-pi, pi)
        offsets[item] = (distance * cos(angle), distance * sin(angle))
        yaws[item] = random.Random(f"wr11:{seed}:yaw:{item}").uniform(-pi, pi)
    return EpisodePlan(
        int(seed),
        sample,
        by_item,
        next(item for item in ITEM_IDS if by_item[item] == sample),
        offsets,
        yaws,
        radius,
        int(max_horizon),
        sample_station_positions(int(seed), circles),
        float(circles["sample_station"]["base_yaw"]),
        {name: float(circle["radius_m"]) for name, circle in circles.items()},
        rules["profile_id"],
    )
