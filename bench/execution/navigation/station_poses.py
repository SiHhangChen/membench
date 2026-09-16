"""Calibrated station-pose loading and selection.

This mirrors the station-base-pose path from ``src/planning/rule_context.py``
without depending on task-specific runtime objects.
"""

from __future__ import annotations

import json
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.navigation.contracts import Pose
from bench.execution.planning.config import NavigationConfig
from bench.execution.planning.nav_algo import normalize_angle
from bench.execution.sampling.circle_sampling import sample_uniform_disk, sample_uniform_yaw


def station_id_for_navigation(
    anchor: str,
    *,
    aliases: Mapping[Any, str] | None = None,
    purpose: str | None = None,
    holding: str | None = None,
    enabled: bool = True,
) -> str | None:
    """Resolve a stage station/anchor to a calibrated station id.

    This function is intentionally task-agnostic. Task-specific names such as
    object anchors must be supplied via ``aliases`` by the task layer.
    """
    del holding
    if not enabled:
        return None
    alias_map = aliases or {}
    alias_keys: list[Any] = []
    if purpose is not None:
        alias_keys.extend(
            [
                (anchor, purpose),
                f"{anchor}:{purpose}",
                f"{anchor}::{purpose}",
            ]
        )
    alias_keys.append(anchor)
    for key in alias_keys:
        if key in alias_map:
            return str(alias_map[key])
    return anchor if anchor.endswith("_station") else None


def load_station_pose_payload(path: str | Path | None) -> dict[str, Any]:
    """Load a station base-pose JSON payload.

    Missing paths intentionally return an empty payload, matching ``src``.
    """
    if path is None:
        return {}
    pose_path = Path(path)
    if not pose_path.exists():
        return {}
    return json.loads(pose_path.read_text())


def yaw_to_quat_xyzw(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(yaw)
    return (0.0, 0.0, float(np.sin(half)), float(np.cos(half)))


def quat_xyzw_to_yaw(quat: tuple[float, float, float, float] | list[float]) -> float:
    x, y, z, w = np.array(quat, dtype=float).reshape(-1)[:4]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def station_sample_pose(
    sample: dict[str, Any],
    *,
    default_yaw: float = 0.0,
) -> tuple[np.ndarray, float] | None:
    xy_value = sample.get("base_xy")
    if xy_value is None:
        base_pos = sample.get("base_pos")
        if base_pos is not None:
            xy_value = list(base_pos)[:2]
    try:
        target_xy = np.array(xy_value, dtype=float).reshape(-1)[:2]
    except Exception:
        return None
    if target_xy.size < 2 or not np.all(np.isfinite(target_xy)):
        return None

    yaw_value = sample.get("base_yaw")
    if yaw_value is None and sample.get("base_quat_xyzw") is not None:
        try:
            yaw_value = quat_xyzw_to_yaw(sample["base_quat_xyzw"])
        except Exception:
            yaw_value = default_yaw
    try:
        target_yaw = float(default_yaw if yaw_value is None else yaw_value)
    except Exception:
        target_yaw = float(default_yaw)
    if not np.isfinite(target_yaw):
        return None
    return target_xy, normalize_angle(target_yaw)


def station_sample_torso_height(sample: Mapping[str, Any]) -> float | None:
    """Read the measured torso height without confusing it with base world Z."""

    value = sample.get("torso_height")
    if value is None:
        torso_qpos = sample.get("torso_qpos")
        if torso_qpos is not None:
            try:
                value = np.array(torso_qpos, dtype=float).reshape(-1)[0]
            except Exception:
                return None
    if value is None:
        return None
    try:
        height = float(value)
    except (TypeError, ValueError):
        return None
    return height if np.isfinite(height) else None


def station_sample_records(
    samples: list[dict[str, Any]],
    *,
    default_yaw: float = 0.0,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        pose = station_sample_pose(sample, default_yaw=default_yaw)
        if pose is None:
            continue
        target_xy, target_yaw = pose
        records.append(
            {
                "index": int(index),
                "sample_index": int(sample.get("sample_index", index)),
                "sample": sample,
                "target_xy": target_xy,
                "target_yaw": float(target_yaw),
                "torso_height": station_sample_torso_height(sample),
            }
        )
    return records


def station_pose_include_points(payload: dict[str, Any]) -> list[np.ndarray]:
    points: list[np.ndarray] = []
    for station in (payload.get("stations", {}) or {}).values():
        for sample in list(station.get("samples", [])):
            pose = station_sample_pose(sample)
            if pose is not None:
                points.append(pose[0])
    return points


@dataclass
class StationPoseSelector:
    """Episode-stable calibrated station sample selector."""

    policy: str = "episode_random"
    seed: int = 0
    seed_offset: int = 314159
    _choice_by_station: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_config(cls, config: NavigationConfig, *, seed: int = 0) -> "StationPoseSelector":
        return cls(
            policy=str(config.station_pose_selection_policy),
            seed=int(seed),
            seed_offset=int(config.station_pose_selection_seed_offset),
        )

    def select(
        self,
        station_id: str,
        records: list[dict[str, Any]],
        *,
        base_xy: np.ndarray | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not records:
            raise ValueError("records must be non-empty")

        if station_id in self._choice_by_station:
            chosen_index = int(self._choice_by_station[station_id])
            for record in records:
                if int(record["index"]) == chosen_index:
                    return record, {
                        "policy": self.policy,
                        "episode_stable": True,
                        "chosen_index": chosen_index,
                        "chosen_sample_index": int(record["sample_index"]),
                        "available_indices": [int(item["index"]) for item in records],
                        "reason": "previous_episode_choice",
                    }

        policy = str(self.policy).strip().lower()
        seed = None
        if policy == "nearest":
            if base_xy is None:
                raise ValueError("base_xy is required for nearest station-pose selection")
            base_xy = np.array(base_xy, dtype=float)[:2]
            record = min(
                records,
                key=lambda item: float(
                    np.linalg.norm(np.array(item["target_xy"], dtype=float) - base_xy)
                ),
            )
        elif policy == "first":
            record = sorted(records, key=lambda item: int(item["index"]))[0]
        else:
            seed = (
                int(self.seed)
                + int(self.seed_offset)
                + int(zlib.crc32(station_id.encode("utf-8")))
            ) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            record = records[int(rng.integers(0, len(records)))]
            policy = "episode_random"

        chosen_index = int(record["index"])
        self._choice_by_station[station_id] = chosen_index
        return record, {
            "policy": policy,
            "episode_stable": True,
            "chosen_index": chosen_index,
            "chosen_sample_index": int(record["sample_index"]),
            "available_indices": [int(item["index"]) for item in records],
            "available_sample_indices": [int(item["sample_index"]) for item in records],
            "seed": seed,
            "reason": "fixed_station_pose_for_episode",
        }

    def reset_choices(self) -> None:
        self._choice_by_station.clear()


@dataclass
class StationPoseSampler:
    """StationSampler implementation backed by calibrated station JSON.

    Ordinary stations select one calibrated pose per episode. Stations with an
    explicit ``circle`` entry resample a uniform point on every call, which is
    useful when a task wants bounded positional variation without changing its
    semantic target.
    """

    payload: dict[str, Any]
    selector: StationPoseSelector = field(default_factory=StationPoseSelector)
    base_xy_provider: Any | None = None
    use_station_base_poses: bool = True
    station_aliases: dict[Any, str] = field(default_factory=dict)
    purpose_by_station: dict[str, str] = field(default_factory=dict)
    _summary_by_station: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)
    _circle_counts: dict[str, int] = field(default_factory=dict, init=False)

    @classmethod
    def from_file(
        cls,
        path: str | Path | None,
        *,
        config: NavigationConfig | None = None,
        seed: int = 0,
        base_xy_provider: Any | None = None,
        station_aliases: Mapping[Any, str] | None = None,
        purpose_by_station: Mapping[str, str] | None = None,
    ) -> "StationPoseSampler":
        nav_config = config or NavigationConfig()
        return cls(
            payload=load_station_pose_payload(path),
            selector=StationPoseSelector.from_config(nav_config, seed=seed),
            base_xy_provider=base_xy_provider,
            use_station_base_poses=bool(nav_config.use_station_base_poses),
            station_aliases=dict(station_aliases or {}),
            purpose_by_station=dict(purpose_by_station or {}),
        )

    def station_id(self, station_name: str, *, purpose: str | None = None) -> str:
        mapped = station_id_for_navigation(
            station_name,
            aliases=self.station_aliases,
            purpose=purpose if purpose is not None else self.purpose_by_station.get(station_name),
            enabled=self.use_station_base_poses,
        )
        return mapped or station_name

    def sample(self, station_name: str, *, purpose: str | None = None) -> Pose:
        station_id = self.station_id(station_name, purpose=purpose)
        station = (self.payload.get("stations", {}) or {}).get(station_id)
        if station is None:
            raise KeyError(f"station {station_id!r} not found in calibrated pose payload")

        circle = station.get("circle") if isinstance(station, Mapping) else None
        if isinstance(circle, Mapping):
            center = np.asarray(circle.get("center_xy", ()), dtype=float).reshape(-1)
            radius = float(circle.get("radius_m", -1.0))
            center_yaw = float(circle.get("base_yaw", 0.0))
            yaw_half_range = float(circle.get("yaw_half_range_rad", 0.0))
            torso_height = station_sample_torso_height(circle)
            if center.size < 2 or not np.all(np.isfinite(center[:2])):
                raise ValueError(f"station {station_id!r} has invalid circle center")
            if not np.isfinite(radius) or radius < 0.0:
                raise ValueError(f"station {station_id!r} has invalid circle radius")
            if not np.isfinite(center_yaw):
                raise ValueError(f"station {station_id!r} has invalid circle yaw")
            if (
                not np.isfinite(yaw_half_range)
                or not 0.0 <= yaw_half_range <= np.pi
            ):
                raise ValueError(
                    f"station {station_id!r} has invalid circle yaw half-range"
                )
            count = int(self._circle_counts.get(station_id, 0))
            self._circle_counts[station_id] = count + 1
            station_salt = int(zlib.crc32(station_id.encode("utf-8")))
            # Derive each episode/station/draw independently.  Using
            # ``episode_seed + count`` would make (seed=s, draw=1) collide
            # with (seed=s+1, draw=0).
            draw_seed = int(
                np.random.SeedSequence(
                    [int(self.selector.seed), station_salt & 0xFFFFFFFF, count]
                ).generate_state(1, dtype=np.uint32)[0]
            )
            sampled = sample_uniform_disk(
                center[:2],
                radius,
                seed=draw_seed,
                salt=0x53544154 ^ station_salt,
            )
            yaw = normalize_angle(
                sample_uniform_yaw(
                    center_yaw,
                    yaw_half_range,
                    seed=draw_seed,
                    salt=0x59415700 ^ station_salt,
                )
            )
            sample = {
                "sample_index": count,
                "base_xy": sampled.astype(float).tolist(),
                "base_yaw": yaw,
                "torso_height": torso_height,
                "record_source": str(circle.get("record_source", "circle_region")),
                "circle": {
                    "center_xy": center[:2].astype(float).tolist(),
                    "radius_m": radius,
                    "base_yaw": normalize_angle(center_yaw),
                    "yaw_half_range_rad": yaw_half_range,
                },
            }
            pose = Pose(
                position=(float(sampled[0]), float(sampled[1]), 0.0),
                quaternion=yaw_to_quat_xyzw(yaw),
                torso_height=torso_height,
            )
            self._summary_by_station[station_name] = {
                "station_name": station_name,
                "purpose": purpose,
                "station_id": station_id,
                "station_pose_selection": {
                    "policy": "circle_uniform",
                    "episode_stable": False,
                    "sample_index": count,
                    "seed": draw_seed,
                },
                "record_source": sample["record_source"],
                "sample_index": count,
                "target_xy": sampled.astype(float).tolist(),
                "target_yaw": yaw,
                "torso_height": torso_height,
                "sample": sample,
            }
            return pose

        records = station_sample_records(list(station.get("samples", [])))
        if not records:
            raise ValueError(f"station {station_id!r} has no valid calibrated samples")

        base_xy = self._current_base_xy()
        record, selection = self.selector.select(station_id, records, base_xy=base_xy)
        target_xy = np.array(record["target_xy"], dtype=float)[:2]
        target_yaw = float(record["target_yaw"])
        sample = dict(record["sample"])
        try:
            base_z = float((sample.get("base_pos") or [0.0, 0.0, 0.0])[2])
        except Exception:
            base_z = 0.0
        quat_value = sample.get("base_quat_xyzw") or yaw_to_quat_xyzw(target_yaw)
        quat = tuple(float(v) for v in quat_value)
        pose = Pose(
            position=(
                float(target_xy[0]),
                float(target_xy[1]),
                base_z,
            ),
            quaternion=quat,
            torso_height=record["torso_height"],
        )
        self._summary_by_station[station_name] = {
            "station_name": station_name,
            "purpose": purpose,
            "station_id": station_id,
            "station_pose_selection": selection,
            "record_source": sample.get("record_source", "manual_calibration"),
            "sample_index": int(record["sample_index"]),
            "target_xy": target_xy.astype(float).tolist(),
            "target_yaw": float(target_yaw),
            "torso_height": record["torso_height"],
            "sample": sample,
        }
        return pose

    def last_summary(self, station_name: str) -> dict[str, Any] | None:
        return self._summary_by_station.get(station_name)

    def include_points(self) -> list[np.ndarray]:
        return station_pose_include_points(self.payload)

    def reset_choices(self) -> None:
        self.selector.reset_choices()
        self._summary_by_station.clear()
        self._circle_counts.clear()

    def _current_base_xy(self) -> np.ndarray | None:
        provider = self.base_xy_provider
        if provider is None:
            return None
        value = provider() if callable(provider) else provider.current_base_xy()
        arr = np.array(value, dtype=float).reshape(-1)
        if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
            return None
        return arr[:2]


__all__ = [
    "StationPoseSampler",
    "StationPoseSelector",
    "load_station_pose_payload",
    "quat_xyzw_to_yaw",
    "station_id_for_navigation",
    "station_pose_include_points",
    "station_sample_pose",
    "station_sample_records",
    "station_sample_torso_height",
    "yaw_to_quat_xyzw",
]
