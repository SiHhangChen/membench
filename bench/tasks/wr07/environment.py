"""RoboCasa scene and physical checks for WR-07 v9.

Three colored sugar plates are visible on each side of the kitchen.  The
milk carton starts on one source plate and must be placed on the matching
colored target plate.  The source/target color permutations are episode facts;
they are never inferred from robot state or changed after reset.
"""

from __future__ import annotations

from typing import Any, Mapping

import mujoco
import numpy as np

from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.models import assets_root
from robocasa.utils import object_utils as OU

from bench.tasks.wr07.domain.contracts import (
    COLOR_RGBA, ITEM_ID, ITEM_LABELS, MILK_CARTON_ASSET_BY_TYPE,
    MILK_CARTON_MATERIAL_PROFILES, SOURCE_SLOTS, TARGET_SLOTS,
)
from bench.tasks.wr07.domain.episode import (
    validate_carton_material,
    validate_carton_type,
    validate_color,
    validate_source_slot,
    validate_target_slot,
)
from bench.tasks.wr07.domain.semantics import (
    PhysicalState, SemanticState, evaluate_success, latch_source_color_memory,
    record_initial_source, record_item_picked, record_target_attempt,
    record_target_placement,
)
from bench.tasks.wr07.evaluation import WR07EvaluationSnapshot, WR07Evaluator

# Keep the rectangular tray geometry used by the original WR-07 scene.  The
# tray bodies are still recolored per episode to carry the matching-color cue.
TRAY_XML = f"{assets_root}/objects/lightwheel/oven_tray/OvenTray001/model.xml"
MILK_CARTON_XML_BY_TYPE = {
    carton_type: f"{assets_root}/objects/{relative_path}"
    for carton_type, relative_path in MILK_CARTON_ASSET_BY_TYPE.items()
}
COUNTER_FIXTURE_NAME = "counter_1_main_group_1"
MEAL_FIXTURE_NAME = "island_island_group_1"
MEAL_PAPER_TOWEL_FIXTURE_NAME = "papertowel_island_group_1"

SOURCE_LOCAL_OFFSETS = {
    "source_left": (-1.0, -1.0),
    "source_center": (0.0, -1.0),
    "source_right": (1.0, -1.0),
}
TARGET_PLATE_WORLD_CENTERS = {
    "target_left": (1.32, -2.701823059087081, 0.9291613871832717),
    "target_center": (1.69, -2.701823059087081, 0.9291613871832717),
    "target_right": (2.06, -2.701823059087081, 0.9291613871832717),
}


def _xy(value: Any, *, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.size != 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"WR-07 {label} must contain two finite values")
    return result


def _pose_quat_for_yaw(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=float)


class WR07MatchMilkCartonToColoredSugarPlate(Kitchen):
    """Move one milk carton from a source sugar plate to its color match."""

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(
        self,
        *,
        source_plate_color_by_slot: Mapping[str, str] | None = None,
        target_plate_color_by_slot: Mapping[str, str] | None = None,
        source_slot_by_item: Mapping[str, str] | None = None,
        source_color_by_item: Mapping[str, str] | None = None,
        target_slot_by_color: Mapping[str, str] | None = None,
        source_item_offset_xy_by_item: Mapping[str, Any] | None = None,
        source_item_yaw_by_item: Mapping[str, float] | None = None,
        target_item_offset_xy_by_item: Mapping[str, Any] | None = None,
        target_item_yaw_by_item: Mapping[str, float] | None = None,
        carton_type: str = "milk_1",
        carton_asset: str | None = None,
        carton_material: str = "satin",
        item_sample_radius_m: float = 0.045,
        plate_threshold: float = 0.16,
        **kwargs: Any,
    ) -> None:
        self.item_names = (ITEM_ID,)
        self.source_plate_color_by_slot = self._resolve_slot_map(
            source_plate_color_by_slot, SOURCE_SLOTS, "source_plate_color_by_slot"
        )
        self.target_plate_color_by_slot = self._resolve_slot_map(
            target_plate_color_by_slot, TARGET_SLOTS, "target_plate_color_by_slot"
        )
        self.source_slot_by_item = {ITEM_ID: validate_source_slot((source_slot_by_item or {}).get(ITEM_ID, "source_center"))}
        source_color = (source_color_by_item or {}).get(ITEM_ID, self.source_plate_color_by_slot[self.source_slot_by_item[ITEM_ID]])
        self.source_color_by_item = {ITEM_ID: validate_color(source_color)}
        self.target_slot_by_color = {str(color): validate_target_slot(slot) for color, slot in dict(target_slot_by_color or {}).items()}
        if self.target_slot_by_color.get(self.source_color_by_item[ITEM_ID]) is None:
            raise ValueError("WR-07 target_slot_by_color must include the milk-carton source color")
        self.source_item_offset_xy_by_item = {ITEM_ID: _xy((source_item_offset_xy_by_item or {}).get(ITEM_ID, (0.0, 0.0)), label="source item offset")}
        self.target_item_offset_xy_by_item = {ITEM_ID: _xy((target_item_offset_xy_by_item or {}).get(ITEM_ID, (0.0, 0.0)), label="target item offset")}
        self.source_item_yaw_by_item = {ITEM_ID: float((source_item_yaw_by_item or {}).get(ITEM_ID, 0.0))}
        self.target_item_yaw_by_item = {ITEM_ID: float((target_item_yaw_by_item or {}).get(ITEM_ID, 0.0))}
        self.carton_type = validate_carton_type(carton_type)
        expected_asset = MILK_CARTON_ASSET_BY_TYPE[self.carton_type]
        if carton_asset is not None and str(carton_asset) != expected_asset:
            raise ValueError("WR-07 carton_asset does not match carton_type")
        self.carton_asset = expected_asset
        self.carton_material = validate_carton_material(carton_material)
        self.item_sample_radius_m = float(item_sample_radius_m)
        self.plate_threshold = float(plate_threshold)
        if not np.isfinite(self.item_sample_radius_m) or self.item_sample_radius_m < 0.0:
            raise ValueError("WR-07 item_sample_radius_m must be finite and non-negative")
        if not np.isfinite(self.plate_threshold) or self.plate_threshold <= 0.0:
            raise ValueError("WR-07 plate_threshold must be positive")
        self._semantic_state = SemanticState()
        self._wr07_evaluator: WR07Evaluator | None = None
        self._source_plate_centers: dict[str, np.ndarray] = {}
        self._target_plate_centers: dict[str, np.ndarray] = {}
        self._source_surface_z = 0.93
        self._target_surface_z = 0.93
        self.counter_zones_fixture = None
        self.meal_counter = None
        self.meal_paper_towel = None
        kwargs.setdefault("obj_registries", ("objaverse", "aigen", "lightwheel"))
        super().__init__(**kwargs)

    @staticmethod
    def _resolve_slot_map(values: Mapping[str, str] | None, slots: tuple[str, ...], label: str) -> dict[str, str]:
        mapping = dict(values or {})
        if set(mapping) != set(slots):
            raise ValueError(f"WR-07 {label} must define exactly {slots}")
        return {slot: validate_color(mapping[slot]) for slot in slots}

    def _setup_kitchen_references(self) -> None:
        super()._setup_kitchen_references()
        self.counter_zones_fixture = self.register_fixture_ref("counter_zones_fixture", dict(id=COUNTER_FIXTURE_NAME, full_name_check=True, full_depth_region=True))
        self.meal_counter = self.register_fixture_ref("meal_counter", dict(id=MEAL_FIXTURE_NAME, full_name_check=True, full_depth_region=True))
        self.meal_paper_towel = self.register_fixture_ref("meal_paper_towel", dict(id=MEAL_PAPER_TOWEL_FIXTURE_NAME, full_name_check=True))
        if self.counter_zones_fixture is None or self.meal_counter is None:
            raise ValueError("WR-07 requires the shelf-side counter and dining island")
        self._plate_fixture = self.meal_counter
        self.init_robot_base_ref = self.counter_zones_fixture

    def _get_obj_cfgs(self) -> list[dict[str, Any]]:
        configs: list[dict[str, Any]] = []
        for slot in SOURCE_SLOTS:
            configs.append(dict(
                name=self.source_plate_name(slot),
                obj_groups=TRAY_XML,
                object_scale=[0.75, 0.75, 0.75],
                placement=dict(
                    fixture=self.counter_zones_fixture,
                    size=(0.34, 0.30),
                    pos=SOURCE_LOCAL_OFFSETS[slot],
                ),
            ))
        for slot in TARGET_SLOTS:
            configs.append(dict(
                name=self.target_plate_name(slot),
                obj_groups=TRAY_XML,
                object_scale=[0.75, 0.75, 0.75],
                placement=dict(
                    fixture=self.meal_counter,
                    sample_region_kwargs=dict(ref=self.meal_paper_towel, loc="right"),
                    size=(0.46, 0.42),
                    pos=(1.0, 0.0),
                    rotation=0.0,
                    # The three target trays are assigned their separated
                    # world positions in ``_reset_internal``.  They share
                    # this temporary sampler region only during model load.
                    ensure_valid_placement=False,
                ),
            ))
        source_name = self.source_plate_name(self.source_slot_by_item[ITEM_ID])
        configs.append(dict(
            name=ITEM_ID,
            obj_groups=MILK_CARTON_XML_BY_TYPE[self.carton_type],
            object_scale=[0.75, 0.75, 0.75],
            graspable=True,
            init_robot_here=True,
            placement=dict(size=(0.20, 0.20), rotation=0.0, sample_args={"reference": source_name}),
        ))
        return configs

    def _setup_scene(self) -> None:
        super()._setup_scene()
        self._semantic_state = SemanticState()
        self._wr07_evaluator = None
        missing = [self.source_plate_name(slot) for slot in SOURCE_SLOTS if self.source_plate_name(slot) not in getattr(self, "object_placements", {})]
        missing += [self.target_plate_name(slot) for slot in TARGET_SLOTS if self.target_plate_name(slot) not in getattr(self, "object_placements", {})]
        if ITEM_ID not in getattr(self, "object_placements", {}):
            missing.append(ITEM_ID)
        if missing:
            raise RuntimeError(f"WR-07 reset is missing sampled placements: {missing}")
        record_initial_source(self._semantic_state, self.source_slot_by_item[ITEM_ID], self.source_color_by_item[ITEM_ID])



    def _reset_internal(self) -> None:
        super()._reset_internal()
        self._source_plate_centers = {slot: self._obj_pos(self.source_plate_name(slot)).copy() for slot in SOURCE_SLOTS}
        self._source_surface_z = float(np.mean([pos[2] for pos in self._source_plate_centers.values()]))
        self._target_plate_centers = {}
        for slot in TARGET_SLOTS:
            name = self.target_plate_name(slot)
            center = np.asarray(TARGET_PLATE_WORLD_CENTERS[slot], dtype=float)
            self.set_object_body_position(name, center)
            self._target_plate_centers[slot] = self._obj_pos(name).copy()
        self._target_surface_z = float(np.mean([pos[2] for pos in self._target_plate_centers.values()]))
        self._apply_milk_carton_pose()
        self._apply_plate_colors()
        self._apply_carton_material()
        self._update_task_state()

    def configure_episode(self, episode: Any) -> None:
        expected_source = dict(episode.source_plate_color_by_slot)
        expected_target = dict(episode.target_plate_color_by_slot)
        if expected_source != self.source_plate_color_by_slot or expected_target != self.target_plate_color_by_slot:
            raise ValueError("WR-07 environment and EpisodePlan plate colors differ")
        if validate_carton_type(episode.carton_type) != self.carton_type:
            raise ValueError("WR-07 environment and EpisodePlan carton types differ")
        if str(episode.carton_asset) != self.carton_asset:
            raise ValueError("WR-07 EpisodePlan carton asset does not match carton type")
        if validate_carton_material(episode.carton_material) != self.carton_material:
            raise ValueError("WR-07 environment and EpisodePlan carton materials differ")

    def source_plate_name(self, slot: str) -> str:
        return f"source_plate_{validate_source_slot(slot).removeprefix('source_')}"

    def target_plate_name(self, slot: str) -> str:
        return f"target_plate_{validate_target_slot(slot).removeprefix('target_')}"

    def _obj_pos(self, name: str) -> np.ndarray:
        return np.asarray(self.sim.data.body_xpos[self.obj_body_id[str(name)]], dtype=float)

    def set_object_body_position(self, obj_name: str, pos: Any) -> None:
        obj = self.objects[str(obj_name)]
        target = np.asarray(pos, dtype=float).reshape(-1)[:3]
        qpos = np.asarray(self.sim.data.get_joint_qpos(obj.joints[0]), dtype=float).copy()
        qpos[:3] += target - self._obj_pos(str(obj_name))
        self.sim.data.set_joint_qpos(obj.joints[0], qpos)
        self.sim.forward()

    def set_object_pose(self, obj_name: str, pos: Any, yaw: float) -> None:
        obj = self.objects[str(obj_name)]
        qpos = np.asarray(self.sim.data.get_joint_qpos(obj.joints[0]), dtype=float).copy()
        qpos[:3] = np.asarray(pos, dtype=float).reshape(-1)[:3]
        qpos[3:7] = np.asarray(self.sim.data.get_joint_qpos(obj.joints[0]), dtype=float).reshape(-1)[3:7]
        qpos[3:7] = np.array([_pose_quat_for_yaw(yaw)[3], *_pose_quat_for_yaw(yaw)[:3]], dtype=float)
        self.sim.data.set_joint_qpos(obj.joints[0], qpos)
        self.sim.forward()

    def _apply_milk_carton_pose(self) -> None:
        slot = self.source_slot_by_item[ITEM_ID]
        center = self._source_plate_centers[slot].copy()
        plate_name = self.source_plate_name(slot)
        plate_body = self.obj_body_id[plate_name]
        rotation = np.asarray(self.sim.data.body_xmat[plate_body], dtype=float).reshape(3, 3)[:2, :2]
        offset = rotation @ self.source_item_offset_xy_by_item[ITEM_ID]
        current = self._obj_pos(ITEM_ID).copy()
        current[:2] = center[:2] + offset
        # The generic reference sampler can produce an invalid z for the
        # rectangular tray.  Recompute it from the actual object offsets so
        # the carton bottom rests on the selected source tray for every seed.
        plate_top_z = float(center[2] + self.objects[plate_name].top_offset[-1])
        carton_bottom_offset_z = float(self.objects[ITEM_ID].bottom_offset[-1])
        current[2] = plate_top_z - carton_bottom_offset_z
        self.set_object_pose(ITEM_ID, current, self.source_item_yaw_by_item[ITEM_ID])

    def _apply_plate_colors(self) -> None:
        for slot, color in self.source_plate_color_by_slot.items():
            self._set_body_color(self.source_plate_name(slot), COLOR_RGBA[color])
        for slot, color in self.target_plate_color_by_slot.items():
            self._set_body_color(self.target_plate_name(slot), COLOR_RGBA[color])

    def _apply_carton_material(self) -> None:
        """Recolor the milk texture and apply its matching optical finish."""
        profile = MILK_CARTON_MATERIAL_PROFILES[self.carton_material]
        model = self.sim.model
        root = int(self.obj_body_id[ITEM_ID])
        parents = np.asarray(model.body_parentid, dtype=int)
        material_ids: set[int] = set()
        for geom_id, body_id in enumerate(np.asarray(model.geom_bodyid, dtype=int)):
            current = int(body_id)
            belongs = False
            while current > 0:
                if current == root:
                    belongs = True
                    break
                current = int(parents[current])
            if not belongs or float(model.geom_rgba[geom_id, 3]) <= 0.0:
                continue
            material_id = int(model.geom_matid[geom_id])
            if material_id >= 0:
                material_ids.add(material_id)
            # A neutral multiplier displays the recolored texture directly.
            model.geom_rgba[geom_id, :] = (1.0, 1.0, 1.0, 1.0)
        for material_id in material_ids:
            model.mat_specular[material_id] = float(profile["specular"])
            model.mat_shininess[material_id] = float(profile["shininess"])
            model.mat_reflectance[material_id] = float(profile["reflectance"])

        raw_model = model._model
        texture_ids = {
            int(texture_id)
            for material_id in material_ids
            for texture_id in np.asarray(raw_model.mat_texid[material_id]).reshape(-1)
            if int(texture_id) >= 0
        }
        cache = getattr(self, "_carton_texture_original_data", None)
        if cache is None:
            cache = {}
            self._carton_texture_original_data = cache
        target = np.asarray(profile["texture_rgb"], dtype=np.float32)
        source_green = np.asarray((0, 107, 14), dtype=np.float32)
        white = np.full(3, 255.0, dtype=np.float32)
        green_axis = source_green - white
        green_axis_norm = float(np.dot(green_axis, green_axis))
        for texture_id in texture_ids:
            address = int(raw_model.tex_adr[texture_id])
            width = int(raw_model.tex_width[texture_id])
            height = int(raw_model.tex_height[texture_id])
            channels = int(raw_model.tex_nchannel[texture_id])
            size = width * height * channels
            original = cache.setdefault(
                texture_id,
                np.asarray(raw_model.tex_data[address:address + size], dtype=np.uint8).copy(),
            )
            pixels = original.reshape(height, width, channels).copy()
            rgb = pixels[..., :3].astype(np.float32)
            green_mask = (rgb[..., 1] > rgb[..., 0] + 2.0) & (rgb[..., 1] > rgb[..., 2] + 2.0)
            coverage = np.sum((rgb - white) * green_axis, axis=-1) / green_axis_norm
            coverage = np.clip(coverage, 0.0, 1.0)
            recolored = rgb + coverage[..., None] * (target - source_green)
            pixels[..., :3][green_mask] = np.clip(
                recolored[green_mask], 0.0, 255.0
            ).astype(np.uint8)
            raw_model.tex_data[address:address + size] = pixels.reshape(-1)
            self._upload_texture_if_rendering(texture_id)

    def _upload_texture_if_rendering(self, texture_id: int) -> None:
        """Refresh an existing GPU texture after changing model.tex_data."""
        context = getattr(self.sim, "_render_context_offscreen", None)
        if context is None:
            return
        context.gl_ctx.make_current()
        mujoco.mjr_uploadTexture(self.sim.model._model, context.con, int(texture_id))

    def _set_body_color(self, name: str, rgba: tuple[float, float, float, float]) -> None:
        model = self.sim.model
        root = int(self.obj_body_id[name])
        parents = np.asarray(model.body_parentid, dtype=int)
        for geom_id, body_id in enumerate(np.asarray(model.geom_bodyid, dtype=int)):
            current = int(body_id)
            belongs = False
            while current > 0:
                if current == root:
                    belongs = True
                    break
                current = int(parents[current])
            if belongs and float(model.geom_rgba[geom_id, 3]) > 0.0:
                model.geom_rgba[geom_id, :3] = np.asarray(rgba[:3], dtype=float)

    def source_plate_pose(self, slot: str) -> np.ndarray:
        return self._source_plate_centers[validate_source_slot(slot)].copy()

    def target_plate_pose(self, slot: str | None = None) -> np.ndarray:
        resolved = validate_target_slot(slot or self.target_slot_by_color[self.source_color_by_item[ITEM_ID]])
        target = self._target_plate_centers[resolved].copy()
        target[:2] += self.target_item_offset_xy_by_item[ITEM_ID]
        return target

    def target_slot_for_source_color(self) -> str:
        return self.target_slot_by_color[self.source_color_by_item[ITEM_ID]]

    def detect_source_slot(self) -> str | None:
        pos = self._obj_pos(ITEM_ID)
        candidates = {
            slot: float(np.linalg.norm(pos[:2] - center[:2]))
            for slot, center in self._source_plate_centers.items()
        }
        slot, distance = min(candidates.items(), key=lambda item: item[1])
        return slot if distance <= self.plate_threshold and abs(float(pos[2] - self._source_surface_z)) <= 0.35 else None

    def detect_target_slot(self) -> str | None:
        pos = self._obj_pos(ITEM_ID)
        candidates = {
            slot: float(np.linalg.norm(pos[:2] - self.target_plate_pose(slot)[:2]))
            for slot in TARGET_SLOTS
        }
        slot, distance = min(candidates.items(), key=lambda item: item[1])
        return slot if distance <= self.plate_threshold else None

    def item_at_target(self, slot: str | None = None) -> bool:
        resolved = validate_target_slot(slot or self.target_slot_for_source_color())
        if self.detect_target_slot() != resolved or not self._item_released():
            return False
        try:
            from bench.execution.planning import object_reference_pos
            bottom_z = float(object_reference_pos(self, ITEM_ID, placement_reference="bbox_bottom_center")[2])
        except Exception:
            return False
        return abs(bottom_z - float(self._target_plate_centers[resolved][2])) <= 0.06

    def physical_state(self) -> PhysicalState:
        source = {slot: self.detect_source_slot() == slot for slot in SOURCE_SLOTS}
        target = {slot: self.detect_target_slot() == slot for slot in TARGET_SLOTS}
        released = self._item_released()
        return PhysicalState(source, target, released, released)

    def latch_source_color_memory(self) -> bool:
        return latch_source_color_memory(self._semantic_state)

    def _update_task_state(self) -> None:
        physical = self.physical_state()
        # A held carton may still be geometrically close enough to the source
        # plate to satisfy the broad source-zone threshold.  Grasp state is
        # therefore also evidence that the pickup transition occurred.
        if not any(physical.item_in_source.values()) or not physical.item_released:
            record_item_picked(self._semantic_state, physically_picked=True)
        target = self.detect_target_slot()
        if target is not None and physical.item_released:
            matches = self.target_plate_color_by_slot[target] == self.source_color_by_item[ITEM_ID]
            record_target_attempt(self._semantic_state, target, matches=matches)
            record_target_placement(self._semantic_state, matches=matches)
    def evaluation_snapshot(self) -> WR07EvaluationSnapshot:
        self._update_task_state()
        target_slot = self.target_slot_for_source_color()
        report = evaluate_success(self._semantic_state, self.physical_state(), target_slot=target_slot)
        state = self._semantic_state
        return WR07EvaluationSnapshot(
            step_id=int(getattr(self, "timestep", 0)),
            source_slot=state.source_slot,
            source_color=state.source_color,
            initial_source_verified=state.initial_source_verified,
            item_picked=state.item_picked,
            source_color_memory_latched=state.source_color_memory_latched,
            target_slot_attempted=state.target_slot_attempted,
            item_placed=state.item_placed,
            target_color_matches=state.target_color_matches,
            protocol_violation=state.protocol_violation,
            domain_success=report.success,
            details={
                **report.checks(),
                "target_slot": target_slot,
                "target_color": self.target_plate_color_by_slot[target_slot],
                "carton_type": getattr(self, "carton_type", "milk_1"),
                "carton_material": getattr(self, "carton_material", "satin"),
                "current_item_position": self._obj_pos(ITEM_ID).tolist(),
            },
        )

    def update_state(self) -> None:
        super().update_state()
        self._update_task_state()

    def _check_success(self) -> bool:
        return self.evaluation_snapshot().domain_success

    def _success_report(self):
        return evaluate_success(self._semantic_state, self.physical_state(), target_slot=self.target_slot_for_source_color())

    def task_diagnostics(self) -> dict[str, Any]:
        snapshot = self.evaluation_snapshot()
        if self._wr07_evaluator is not None:
            return self._wr07_evaluator.observe(snapshot).to_diagnostics()
        return {**dict(snapshot.details), "success": snapshot.domain_success}

    def get_ep_meta(self) -> dict[str, Any]:
        from bench.tasks.wr07.prompts import resolve_prompt_spec
        meta = super().get_ep_meta()
        prompt = resolve_prompt_spec(variant="memory")
        meta["lang"], meta["initial_prompt"] = prompt.policy_prompt, prompt.initial_prompt
        meta.setdefault("refs", {}).update({
            "item_ids": list(self.item_names), "item_labels": ITEM_LABELS,
            "carton_type": self.carton_type,
            "carton_asset": self.carton_asset,
            "carton_material": self.carton_material,
            "source_plate_colors": dict(self.source_plate_color_by_slot),
            "target_plate_colors": dict(self.target_plate_color_by_slot),
        })
        return meta

    def _item_released(self) -> bool:
        try:
            return bool(OU.gripper_obj_far(self, ITEM_ID, th=0.15))
        except Exception:
            return True


__all__ = ["WR07MatchMilkCartonToColoredSugarPlate"]
