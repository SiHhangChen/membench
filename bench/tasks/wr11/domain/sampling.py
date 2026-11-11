"""Task-owned sampling rules shared by resolution, reset and navigation."""

import json
from math import isfinite, pi
from pathlib import Path

from bench.execution.sampling.circle_sampling import sample_uniform_disk

from .contracts import ITEM_IDS, PACKAGING_RGB, SAMPLE_ID
from .scene import LAYOUT_ID, SAMPLE_STATION_XY, STOCK_STATION_XY, STYLE_ID

DEFAULT_RULES_FILE = Path(__file__).parents[1] / "resources/sampling/regions.json"
DEFAULT_STATIONS_FILE = Path(__file__).parents[1] / "resources/calibration/stations.json"
STATION_IDS = ("sample_station", "stock_station")
BASE_SALTS = {"sample_station": 0x57523131, "stock_station": 0x57523132}


def load_sampling_rules(path=DEFAULT_RULES_FILE):
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != "membench.task_sampling/v1":
        raise ValueError("WR-11 sampling schema mismatch")
    rules = payload["entries"]
    if set(rules) != {"profile_id", "request", "stock", "objects", "robot", "fixed"}:
        raise ValueError("WR-11 sampling rules have missing or unknown entries")
    if rules["profile_id"] != "WR-11/sampling-v2":
        raise ValueError("WR-11 sampling profile mismatch")
    if rules["request"] != {
        "distribution": "uniform_among_stock",
        "packaging": list(PACKAGING_RGB),
    }:
        raise ValueError("WR-11 requires an equally likely request among three of eight colors")
    if rules["stock"] != {
        "distribution": "uniform_ordered_without_replacement",
        "count": len(ITEM_IDS),
    }:
        raise ValueError("WR-11 requires three distinct stock colors from the eight-color pool")
    objects = rules["objects"]
    if set(objects) != {
        "ids",
        "xy_distribution",
        "radius_m",
        "yaw_range_rad",
        "independent_streams",
    }:
        raise ValueError("WR-11 object sampling settings mismatch")
    if (
        objects["ids"] != [*ITEM_IDS, SAMPLE_ID]
        or objects["xy_distribution"] != "uniform_area_disk"
    ):
        raise ValueError("WR-11 object sampling must cover the sample and three stock items")
    radius = float(objects["radius_m"])
    if not isfinite(radius) or not 0 <= radius <= 0.025:
        raise ValueError("WR-11 object radius must stay in calibrated range [0, 0.025] m")
    if objects["yaw_range_rad"] != [-pi, pi] or objects["independent_streams"] is not True:
        raise ValueError("WR-11 requires independent full-angle object poses")
    if rules["robot"] != {
        "xy_distribution": "uniform_area_disk",
        "station_ids": list(STATION_IDS),
        "reuse_sample_station_for_return": True,
        "independent_station_streams": True,
        "heading": "fixed_shared",
        "radius_source": "station_calibration",
    }:
        raise ValueError("WR-11 requires episode-stable station circles with a shared heading")
    if rules["fixed"] != ["layout", "style", "trays", "carton_geometry", "torso_height"]:
        raise ValueError("WR-11 fixed sampling anchors mismatch")
    return rules


def load_station_circles(path=DEFAULT_STATIONS_FILE):
    payload = json.loads(Path(path).read_text())
    if (payload.get("layout_id"), payload.get("style_id")) != (LAYOUT_ID, STYLE_ID):
        raise ValueError("WR-11 station calibration scene mismatch")
    stations = payload["stations"]
    if set(stations) != set(STATION_IDS):
        raise ValueError("WR-11 needs exactly the sample and stock stations")
    circles = {}
    for name, center in zip(STATION_IDS, (SAMPLE_STATION_XY, STOCK_STATION_XY)):
        circle = stations[name]["circle"]
        radius = float(circle["radius_m"])
        if list(circle["center_xy"]) != list(center):
            raise ValueError(f"WR-11 {name} circle differs from calibrated scene geometry")
        if not isfinite(radius) or not 0 <= radius <= 0.05:
            raise ValueError("WR-11 station radius must stay in calibrated range [0, 0.05] m")
        if (
            circle["base_yaw"] != -pi / 2
            or circle["yaw_half_range_rad"] != 0
            or circle["torso_height"] != 0
        ):
            raise ValueError("WR-11 stations must preserve the shared heading and torso height")
        region = stations[name]["sampling_region"]
        if region != {"center_xy": list(center), "radius_m": radius, "shape": "disk"}:
            raise ValueError(f"WR-11 {name} region and station circle disagree")
        circles[name] = dict(circle)
    return circles


def sample_station_positions(seed, circles):
    return {
        name: tuple(
            float(value)
            for value in sample_uniform_disk(
                circles[name]["center_xy"],
                circles[name]["radius_m"],
                seed=seed,
                salt=BASE_SALTS[name],
            )
        )
        for name in STATION_IDS
    }
