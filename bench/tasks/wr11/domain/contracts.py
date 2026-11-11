"""Immutable WR-11 inputs; private assignments never enter policy state."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .scene import LAYOUT_ID, SCENE_SEED, STYLE_ID

TASK_ID = "WR-11"
TASK_REVISION = "WR11-v3.2-eight-colors"
ENV_NAME = "WR11FetchMatchingMilkCarton"
ITEM_IDS = ("stock_left", "stock_center", "stock_right")
SAMPLE_ID = "sample_carton"
CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
CARTON_ASSET = "objaverse/milk/milk_1/model.xml"
# Recolor only the original green packaging, preserving its white artwork.
# Fixed geometry makes the visual memory alternatives equally graspable.
PACKAGING_RGB = {
    "red": (220, 35, 40),
    "orange": (245, 120, 10),
    "yellow": (235, 210, 15),
    "green": (20, 155, 55),
    "cyan": (0, 190, 205),
    "blue": (35, 75, 220),
    "purple": (145, 45, 200),
    "black": (25, 25, 28),
}


@dataclass(frozen=True)
class SceneConfig:
    env_name: str
    env_split: str
    layout_id: int = LAYOUT_ID
    style_id: int = STYLE_ID
    scene_seed: int = SCENE_SEED
    fixed_scene: bool = True


@dataclass(frozen=True)
class EpisodePlan:
    seed: int
    sample_packaging: str
    packaging_by_item: Mapping[str, str]
    matching_item: str
    offset_xy_by_item: Mapping[str, tuple[float, float]]
    yaw_by_item: Mapping[str, float]
    item_sample_radius_m: float
    max_horizon: int
    base_xy_by_station: Mapping[str, tuple[float, float]]
    base_yaw: float
    base_radius_by_station: Mapping[str, float]
    sampling_profile_id: str


@dataclass(frozen=True)
class TaskConfig:
    source_path: Path
    task: Mapping[str, Any]
    scene: Mapping[str, Any]
    episode: Mapping[str, Any]
    initial_state: Mapping[str, Any]
    sensors: Mapping[str, Any]
    resources: Mapping[str, Any]
    navigation: Mapping[str, Any]
    manipulation: Mapping[str, Any]
    operations: Mapping[str, Any]
    semantics: Mapping[str, Any]
    recording: Mapping[str, Any]

    def path(self, value):
        if value is None:
            return None
        path = Path(value)
        return path if path.is_absolute() else self.source_path.parent / path


@dataclass(frozen=True)
class ActionConfig:
    source_pick: Mapping[str, Any]
    target_place: Mapping[str, Any]
    return_arm_empty: Mapping[str, Any]
    return_arm_holding: Mapping[str, Any]
