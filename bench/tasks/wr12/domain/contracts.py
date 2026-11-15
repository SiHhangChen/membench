"""WR-12 immutable inputs; target assignment is private episode metadata."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TASK_ID = "WR-12"
TASK_REVISION = "WR12-v1.4-one-second-view-then-seal"
ENV_NAME = "WR12PlaceAtRememberedBox"
BOX_IDS = ("left", "right")
BLOCK_ID = "placement_block"
CAN_ID = "hidden_can"
PICKABLE_IDS = (BLOCK_ID,)
CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")


@dataclass(frozen=True)
class SceneConfig:
    env_name: str
    env_split: str
    layout_id: int = 33
    style_id: int = 34
    scene_seed: int = 0
    fixed_scene: bool = True


@dataclass(frozen=True)
class EpisodePlan:
    seed: int
    target_box: str
    pair_id: int
    block_offset_xy: tuple[float, float]
    block_yaw: float
    placement_offset_xy: tuple[float, float]
    base_xy_by_station: Mapping[str, tuple[float, float]]
    base_yaw_by_station: Mapping[str, float]
    base_radius_by_station: Mapping[str, float]
    sampling_profile_id: str
    max_horizon: int


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
        path = Path(value)
        return path if path.is_absolute() else self.source_path.parent / path


@dataclass(frozen=True)
class ActionConfig:
    source_pick: Mapping[str, Any]
    target_place: Mapping[str, Any]
    return_arm_empty: Mapping[str, Any]
    return_arm_holding: Mapping[str, Any]
