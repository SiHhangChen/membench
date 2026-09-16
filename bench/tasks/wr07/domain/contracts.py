"""Immutable WR-07 contracts for color-matched sugar-plate transport."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

TASK_ID = "WR-07"
TASK_REVISION = "WR07-v14-texture-randomization"
ENV_NAME = "WR07MatchMilkCartonToColoredSugarPlate"
DISPLAY_NAME = "Move the milk carton to the matching colored sugar plate"

ITEM_ID = "milk_carton"
ITEM_IDS = (ITEM_ID,)
ITEM_LABELS = {ITEM_ID: "milk carton"}

# Only assets that load from repository-relative dependencies are included in
# the first randomized carton pool.  The AIGen milk models currently refer to
# a machine-local absolute texture path and are therefore intentionally kept
# out of the benchmark pool until those dependencies are repaired.
MILK_CARTON_ASSET_BY_TYPE = {
    "milk_0": "objaverse/milk/milk_0/model.xml",
    "milk_1": "objaverse/milk/milk_1/model.xml",
    "milk_3": "objaverse/milk/milk_3/model.xml",
    "milk_4": "objaverse/milk/milk_4/model.xml",
    "milk_5": "objaverse/milk/milk_5/model.xml",
    "milk_7": "objaverse/milk/milk_7/model.xml",
    "milk_9": "objaverse/milk/milk_9/model.xml",
    "milk_10": "objaverse/milk/milk_10/model.xml",
}
# WR-07 uses one stable carton geometry. The broader asset map remains here
# for tooling and previews, but episode sampling does not vary object type.
MILK_CARTON_TYPES = ("milk_1",)

# Appearance randomization keeps the carton mesh and collision model fixed.
# Each profile recolors the green region of the existing milk_1 texture and
# applies a matching optical finish after MuJoCo compilation.
MILK_CARTON_MATERIAL_PROFILES = {
    "matte": {
        "texture_rgb": (156, 92, 38),
        "specular": 0.08,
        "shininess": 0.08,
        "reflectance": 0.0,
    },
    "satin": {
        "texture_rgb": (40, 160, 82),
        "specular": 0.28,
        "shininess": 0.25,
        "reflectance": 0.02,
    },
    "glossy": {
        "texture_rgb": (38, 105, 210),
        "specular": 0.62,
        "shininess": 0.65,
        "reflectance": 0.08,
    },
    "foil": {
        "texture_rgb": (145, 160, 175),
        "specular": 0.88,
        "shininess": 0.92,
        "reflectance": 0.18,
    },
}
MILK_CARTON_MATERIALS = tuple(MILK_CARTON_MATERIAL_PROFILES)
SOURCE_SLOTS = ("source_left", "source_center", "source_right")
TARGET_SLOTS = ("target_left", "target_center", "target_right")

COLOR_NAMES = (
    "red", "orange", "yellow", "lime", "green", "teal", "cyan", "sky",
    "blue", "indigo", "violet", "magenta", "pink", "rose", "coral",
    "gold", "mint", "navy", "plum", "gray",
)
COLOR_RGBA = {
    "red": (0.86, 0.10, 0.10, 1.0), "orange": (0.95, 0.35, 0.06, 1.0),
    "yellow": (0.95, 0.78, 0.08, 1.0), "lime": (0.58, 0.86, 0.08, 1.0),
    "green": (0.10, 0.64, 0.18, 1.0), "teal": (0.06, 0.66, 0.58, 1.0),
    "cyan": (0.05, 0.78, 0.88, 1.0), "sky": (0.18, 0.55, 0.90, 1.0),
    "blue": (0.08, 0.28, 0.82, 1.0), "indigo": (0.23, 0.12, 0.68, 1.0),
    "violet": (0.52, 0.12, 0.82, 1.0), "magenta": (0.82, 0.10, 0.65, 1.0),
    "pink": (0.95, 0.35, 0.65, 1.0), "rose": (0.90, 0.18, 0.38, 1.0),
    "coral": (0.95, 0.42, 0.34, 1.0), "gold": (0.88, 0.58, 0.06, 1.0),
    "mint": (0.25, 0.82, 0.52, 1.0), "navy": (0.06, 0.16, 0.42, 1.0),
    "plum": (0.40, 0.12, 0.32, 1.0), "gray": (0.45, 0.48, 0.52, 1.0),
}

class RobotBaseMode(str, Enum):
    PRESERVE_RESET_POSE = "preserve_reset_pose"
    EXPLICIT_POSE = "explicit_pose"
    RESET_CIRCLE = "reset_circle"

@dataclass(frozen=True)
class RobotBaseConfig:
    mode: RobotBaseMode
    xy: tuple[float, float] | None = None
    yaw: float | None = None
    radius_m: float | None = None

@dataclass(frozen=True)
class InitialStateConfig:
    robot_base: RobotBaseConfig
    source_regions: Mapping[str, Mapping[str, Any]]
    objects: Mapping[str, Any]

@dataclass(frozen=True)
class TaskConfig:
    source_path: Path
    task: Mapping[str, Any]
    scene: Mapping[str, Any]
    episode: Mapping[str, Any]
    initial_state: InitialStateConfig
    sensors: Mapping[str, Any]
    resources: Mapping[str, Any]
    navigation: Mapping[str, Any]
    manipulation: Mapping[str, Any]
    operations: Mapping[str, Any]
    semantics: Mapping[str, Any]
    recording: Mapping[str, Any]
    def path(self, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        path = Path(value)
        return path if path.is_absolute() else self.source_path.parent / path

@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    display_name: str
    description: str
    success_conditions: tuple[str, ...]

@dataclass(frozen=True)
class SceneConfig:
    env_name: str
    env_split: str
    layout_id: int
    style_id: int
    scene_seed: int
    fixed_scene: bool

@dataclass(frozen=True)
class EpisodePlan:
    seed: int
    source_plate_color_by_slot: Mapping[str, str]
    target_plate_color_by_slot: Mapping[str, str]
    source_slot_by_item: Mapping[str, str]
    source_color_by_item: Mapping[str, str]
    target_slot_by_color: Mapping[str, str]
    source_item_offset_xy_by_item: Mapping[str, tuple[float, float]]
    source_item_yaw_by_item: Mapping[str, float]
    target_item_offset_xy_by_item: Mapping[str, tuple[float, float]]
    target_item_yaw_by_item: Mapping[str, float]
    item_sample_radius_m: float
    carton_type: str
    carton_asset: str
    carton_material: str
    carton_type_candidates: tuple[str, ...]
    carton_material_candidates: tuple[str, ...]
    transport_order: tuple[str, ...]
    max_horizon: int

@dataclass(frozen=True)
class EpisodeTargets:
    target_plate_pose_by_item: Mapping[str, tuple[float, float, float]]

@dataclass(frozen=True)
class ActionConfig:
    source_pick_metadata: Mapping[str, Mapping[str, Any]]
    target_place_metadata: Mapping[str, Mapping[str, Any]]
    return_arm_empty: Mapping[str, Any]
    return_arm_holding: Mapping[str, Any]
    handler_settings: Mapping[str, Any]

@dataclass(frozen=True)
class RunContext:
    split: str
    variant: str
    execute: bool
    extra_metadata: Mapping[str, Any] = field(default_factory=dict)

__all__ = [
    "ActionConfig", "COLOR_NAMES", "COLOR_RGBA", "DISPLAY_NAME", "ENV_NAME",
    "EpisodePlan", "EpisodeTargets", "ITEM_ID", "ITEM_IDS",
    "ITEM_LABELS", "InitialStateConfig", "MILK_CARTON_ASSET_BY_TYPE",
    "MILK_CARTON_MATERIAL_PROFILES", "MILK_CARTON_MATERIALS", "MILK_CARTON_TYPES",
    "RobotBaseConfig", "RobotBaseMode",
    "RunContext", "SOURCE_SLOTS", "SceneConfig", "TARGET_SLOTS", "TASK_ID",
    "TASK_REVISION", "TaskConfig", "TaskDefinition",
]
