"""Deterministic WR-08 color and placement sampling."""

from __future__ import annotations

import random
from collections.abc import Sequence
from math import cos, isfinite, pi, sin, sqrt
from typing import Any

from bench.tasks.wr08.domain.contracts import (
    CARTON_ASSET_BY_ITEM,
    CARTON_TYPE_BY_ITEM,
    COLOR_NAMES,
    DISPLAY_NAME,
    ENV_NAME,
    EpisodePlan,
    ITEM_IDS,
    MILK_CARTON_MATERIALS,
    MILK_CARTON_TYPES,
    SOURCE_SLOTS,
    TARGET_SLOTS,
    TASK_ID,
    TaskDefinition,
    SceneConfig,
)


def validate_source_slot(value: Any) -> str:
    slot = str(value)
    if slot not in SOURCE_SLOTS:
        raise ValueError(f"invalid WR-08 source slot: {slot!r}")
    return slot


def validate_target_slot(value: Any) -> str:
    slot = str(value)
    if slot not in TARGET_SLOTS:
        raise ValueError(f"invalid WR-08 target slot: {slot!r}")
    return slot


def validate_color(value: Any) -> str:
    color = str(value)
    if color not in COLOR_NAMES:
        raise ValueError(f"invalid WR-08 color: {color!r}")
    return color


def validate_carton_type(value: Any) -> str:
    carton_type = str(value)
    if carton_type not in MILK_CARTON_TYPES:
        raise ValueError(f"invalid WR-08 milk-carton type: {carton_type!r}")
    return carton_type


def validate_carton_material(value: Any) -> str:
    material = str(value)
    if material not in MILK_CARTON_MATERIALS:
        raise ValueError(f"invalid WR-08 milk-carton material: {material!r}")
    return material


def _candidate_pool(
    value: Any,
    *,
    allowed: tuple[str, ...],
    label: str,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"WR-08 {label} must be a non-empty sequence")
    candidates = tuple(str(item) for item in value)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError(f"WR-08 {label} must be non-empty and unique")
    unknown = sorted(set(candidates) - set(allowed))
    if unknown:
        raise ValueError(f"unknown WR-08 {label}: " + ", ".join(unknown))
    return candidates


def resolve_task_definition(**values: Any) -> TaskDefinition:
    if str(values.get("id", TASK_ID)) != TASK_ID:
        raise ValueError(f"WR-08 config must use task.id={TASK_ID!r}")
    conditions = tuple(str(value) for value in values.get("success_conditions", ()))
    if not conditions:
        raise ValueError("WR-08 must declare success conditions")
    return TaskDefinition(
        TASK_ID,
        str(values.get("display_name", DISPLAY_NAME)),
        str(values.get("description", "Move two distinct milk carton models to matching colored sugar plates.")),
        conditions,
    )


def resolve_scene_config(*, seed: int, env_split: str, **values: Any) -> SceneConfig:
    del seed
    if str(values.get("env_name", ENV_NAME)) != ENV_NAME:
        raise ValueError(f"WR-08 config must use scene.env_name={ENV_NAME!r}")
    identity = (
        int(values.get("layout_id", 32)),
        int(values.get("style_id", 32)),
        int(values.get("scene_seed", 0)),
    )
    if not bool(values.get("fixed_scene", True)) or identity != (32, 32, 0):
        raise ValueError("WR-08 is fixed to layout/style/scene-seed 32/32/0")
    return SceneConfig(ENV_NAME, str(env_split), *identity, fixed_scene=True)


def resolve_episode_plan(*, seed: int, **values: Any) -> EpisodePlan:
    episode_seed = int(seed)
    if episode_seed < 0:
        raise ValueError("WR-08 seed must be non-negative")
    color_count = int(values.pop("color_count", 3))
    if color_count != 3:
        raise ValueError("WR-08 requires exactly three visible colors")
    item_radius = _nonnegative_scalar(
        values.pop("item_sample_radius_m", 0.045), label="item_sample_radius_m"
    )
    horizon = int(values.pop("max_horizon", 5200))
    values.pop("color_pool", None)
    if values:
        raise ValueError("unknown WR-08 episode settings: " + ", ".join(sorted(values)))

    rng = random.Random(episode_seed)
    selected = rng.sample(list(COLOR_NAMES), 3)
    source_colors = list(selected)
    rng.shuffle(source_colors)
    source_by_slot = dict(zip(SOURCE_SLOTS, source_colors, strict=True))

    # A cyclic shift guarantees that the target-side left/center/right layout
    # is a non-identity permutation while retaining the same color set.
    shift = rng.choice((1, 2))
    target_colors = [source_colors[(index + shift) % 3] for index in range(3)]
    target_by_slot = dict(zip(TARGET_SLOTS, target_colors, strict=True))
    target_by_color = {color: slot for slot, color in target_by_slot.items()}

    selected_slots = rng.sample(list(SOURCE_SLOTS), len(ITEM_IDS))
    source_slot_by_item = dict(zip(ITEM_IDS, selected_slots, strict=True))
    source_color_by_item = {
        item: source_by_slot[slot] for item, slot in source_slot_by_item.items()
    }
    order = list(ITEM_IDS)
    rng.shuffle(order)

    source_offsets = {
        item: _sample_source_offset(
            rng,
            item_radius,
            shelf_side=(source_slot_by_item[item] == "source_right"),
            item=item,
        )
        for item in ITEM_IDS
    }
    target_offsets = {item: _sample_disk(rng, item_radius) for item in ITEM_IDS}
    source_yaws = {item: rng.uniform(-pi, pi) for item in ITEM_IDS}
    target_yaws = {item: rng.uniform(-pi, pi) for item in ITEM_IDS}

    # Per-item appearance randomization (independent for each carton).
    # Geometry is still locked by CARTON_TYPE_BY_ITEM for each logical item id.
    carton_type_by_item = {
        item: CARTON_TYPE_BY_ITEM[item] for item in ITEM_IDS
    }
    carton_asset_by_item = {
        item: CARTON_ASSET_BY_ITEM[item] for item in ITEM_IDS
    }
    # Draw material choices from the candidate pool after all geometry decisions.
    carton_material_by_item = {
        item: rng.choice(carton_materials) for item in ITEM_IDS
    }

    return EpisodePlan(
        seed=episode_seed,
        source_plate_color_by_slot=source_by_slot,
        target_plate_color_by_slot=target_by_slot,
        source_slot_by_item=source_slot_by_item,
        source_color_by_item=source_color_by_item,
        target_slot_by_color=target_by_color,
        source_item_offset_xy_by_item=source_offsets,
        source_item_yaw_by_item=source_yaws,
        target_item_offset_xy_by_item=target_offsets,
        target_item_yaw_by_item=target_yaws,
        item_sample_radius_m=item_radius,
        carton_type_by_item=carton_type_by_item,
        carton_asset_by_item=carton_asset_by_item,
        carton_material_by_item=carton_material_by_item,
        carton_type_candidates=carton_types,
        carton_material_candidates=carton_materials,
        transport_order=tuple(order),
        max_horizon=horizon,
    )


def _sample_disk(rng: random.Random, radius: float) -> tuple[float, float]:
    if radius <= 0.0:
        return (0.0, 0.0)
    angle = rng.uniform(-pi, pi)
    distance = radius * sqrt(rng.random())
    return (float(distance * cos(angle)), float(distance * sin(angle)))


def _sample_source_offset(
    rng: random.Random,
    radius: float,
    *,
    shelf_side: bool,
    item: str,
) -> tuple[float, float]:
    """Keep the taller milk_0 carton away from the shelf-side edge."""

    shelf_margin = min(0.01, 0.5 * radius)
    offset = _sample_disk(rng, radius)
    if not (shelf_side and item == "boxed_food_a") or offset[1] <= -shelf_margin:
        return offset
    safe_y = -max(abs(offset[1]), shelf_margin)
    max_x = sqrt(max(radius * radius - safe_y * safe_y, 0.0))
    safe_x = max(-max_x, min(float(offset[0]), max_x))
    return (float(safe_x), float(safe_y))


def _nonnegative_scalar(value: Any, *, label: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"WR-08 {label} must be finite and non-negative") from exc
    if not isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"WR-08 {label} must be finite and non-negative")
    return resolved


def public_task_prompt() -> str:
    return "Move the two milk cartons from the coffee-machine-side sugar plates to the matching colored sugar plates on the opposite side."


def public_initial_prompt() -> str:
    return public_task_prompt()


def public_return_prompt() -> str:
    return public_task_prompt()


__all__ = [
    "public_initial_prompt", "public_return_prompt", "public_task_prompt",
    "resolve_episode_plan", "resolve_scene_config", "resolve_task_definition",
    "validate_color", "validate_source_slot", "validate_target_slot",
]
