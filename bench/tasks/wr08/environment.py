"""RoboCasa scene and physical checks for WR-08 color-matched transport."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.models import assets_root
from robocasa.utils import object_utils as OU

from bench.tasks.wr08.domain.contracts import (
    CARTON_ASSET_BY_ITEM,
    COLOR_RGBA,
    ITEM_IDS,
    ITEM_LABELS,
    SOURCE_SLOTS,
    TARGET_SLOTS,
)
from bench.tasks.wr08.domain.episode import (
    validate_color,
    validate_source_slot,
    validate_target_slot,
)
from bench.tasks.wr08.domain.semantics import (
    PhysicalState,
    SemanticState,
    evaluate_success,
    latch_source_color_memory,
    record_initial_sources,
)
from bench.tasks.wr08.evaluation import WR08EvaluationSnapshot, WR08Evaluator
from bench.tasks.wr08.physical_events import PhysicalSample, WR08PhysicalObserver

TRAY_XML = f"{assets_root}/objects/lightwheel/oven_tray/OvenTray001/model.xml"
MILK_CARTON_XML_BY_ITEM = {
    item: f"{assets_root}/objects/{relative_path}"
    for item, relative_path in CARTON_ASSET_BY_ITEM.items()
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
        raise ValueError(f"WR-08 {label} must contain two finite values")
    return result


def _pose_quat_for_yaw(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=float)


class WR08MatchMilkCartonsToColoredSugarPlates(Kitchen):
    """Move two milk cartons from source sugar plates to color matches."""

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
        transport_order: Sequence[str] | None = None,
        item_sample_radius_m: float = 0.045,
        plate_threshold: float = 0.16,
        physical_event_dwell_steps: int = 3,
        terminal_verification_steps: int = 1,
        **kwargs: Any,
    ) -> None:
        self.item_names = ITEM_IDS
        self.source_plate_color_by_slot = self._resolve_slot_map(
            source_plate_color_by_slot, SOURCE_SLOTS, "source_plate_color_by_slot"
        )
        self.target_plate_color_by_slot = self._resolve_slot_map(
            target_plate_color_by_slot, TARGET_SLOTS, "target_plate_color_by_slot"
        )
        self.source_slot_by_item = {
            item: validate_source_slot(
                (source_slot_by_item or {}).get(item, SOURCE_SLOTS[index])
            )
            for index, item in enumerate(ITEM_IDS)
        }
        if len(set(self.source_slot_by_item.values())) != len(ITEM_IDS):
            raise ValueError("WR-08 source_slot_by_item must use distinct source slots")
        inferred_colors = {
            item: self.source_plate_color_by_slot[self.source_slot_by_item[item]]
            for item in ITEM_IDS
        }
        self.source_color_by_item = {
            item: validate_color(
                (source_color_by_item or {}).get(item, inferred_colors[item])
            )
            for item in ITEM_IDS
        }
        if any(
            self.source_color_by_item[item] != inferred_colors[item]
            for item in ITEM_IDS
        ):
            raise ValueError(
                "WR-08 source_color_by_item must match source plate colors"
            )
        self.target_slot_by_color = {
            str(color): validate_target_slot(slot)
            for color, slot in dict(target_slot_by_color or {}).items()
        }
        if set(self.target_slot_by_color) != set(
            self.source_plate_color_by_slot.values()
        ):
            raise ValueError(
                "WR-08 target_slot_by_color must cover all selected colors"
            )
        if set(self.target_slot_by_color.values()) != set(TARGET_SLOTS):
            raise ValueError(
                "WR-08 target_slot_by_color must use each target slot exactly once"
            )
        self.source_item_offset_xy_by_item = {
            item: _xy(
                (source_item_offset_xy_by_item or {}).get(item, (0.0, 0.0)),
                label=f"{item} source item offset",
            )
            for item in ITEM_IDS
        }
        self.target_item_offset_xy_by_item = {
            item: _xy(
                (target_item_offset_xy_by_item or {}).get(item, (0.0, 0.0)),
                label=f"{item} target item offset",
            )
            for item in ITEM_IDS
        }
        self.source_item_yaw_by_item = {
            item: float((source_item_yaw_by_item or {}).get(item, 0.0))
            for item in ITEM_IDS
        }
        self.target_item_yaw_by_item = {
            item: float((target_item_yaw_by_item or {}).get(item, 0.0))
            for item in ITEM_IDS
        }
        self.item_sample_radius_m = float(item_sample_radius_m)
        self._transport_order = tuple(
            str(item) for item in (transport_order or ITEM_IDS)
        )
        if len(self._transport_order) != len(ITEM_IDS) or set(
            self._transport_order
        ) != set(ITEM_IDS):
            raise ValueError(
                "WR-08 transport_order must contain each item exactly once"
            )
        self.plate_threshold = float(plate_threshold)
        self.physical_event_dwell_steps = int(physical_event_dwell_steps)
        self.terminal_verification_steps = int(terminal_verification_steps)
        if self.physical_event_dwell_steps < 1 or self.terminal_verification_steps < 1:
            raise ValueError("WR-08 physical observer dwell values must be positive")
        if (
            not np.isfinite(self.item_sample_radius_m)
            or self.item_sample_radius_m < 0.0
        ):
            raise ValueError(
                "WR-08 item_sample_radius_m must be finite and non-negative"
            )
        if not np.isfinite(self.plate_threshold) or self.plate_threshold <= 0.0:
            raise ValueError("WR-08 plate_threshold must be positive")
        self._semantic_state = SemanticState()
        self._physical_observer: WR08PhysicalObserver | None = None
        self._wr08_evaluator: WR08Evaluator | None = None
        # Raw environments retain recording checks. Canonical policy runtimes
        # select physical evaluation explicitly before executing any actions.
        self.require_expert_protocol = True
        self._physical_event_diagnostics: dict[str, Any] = {}
        self._evaluation_initial_item_positions: dict[str, np.ndarray] = {}
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
    def _resolve_slot_map(
        values: Mapping[str, str] | None, slots: tuple[str, ...], label: str
    ) -> dict[str, str]:
        mapping = dict(values or {})
        if set(mapping) != set(slots):
            raise ValueError(f"WR-08 {label} must define exactly {slots}")
        return {slot: validate_color(mapping[slot]) for slot in slots}

    def _setup_kitchen_references(self) -> None:
        super()._setup_kitchen_references()
        self.counter_zones_fixture = self.register_fixture_ref(
            "counter_zones_fixture",
            dict(id=COUNTER_FIXTURE_NAME, full_name_check=True, full_depth_region=True),
        )
        self.meal_counter = self.register_fixture_ref(
            "meal_counter",
            dict(id=MEAL_FIXTURE_NAME, full_name_check=True, full_depth_region=True),
        )
        self.meal_paper_towel = self.register_fixture_ref(
            "meal_paper_towel",
            dict(id=MEAL_PAPER_TOWEL_FIXTURE_NAME, full_name_check=True),
        )
        if self.counter_zones_fixture is None or self.meal_counter is None:
            raise ValueError("WR-08 requires the shelf-side counter and dining island")
        self._plate_fixture = self.meal_counter
        self.init_robot_base_ref = self.counter_zones_fixture

    def _get_obj_cfgs(self) -> list[dict[str, Any]]:
        configs: list[dict[str, Any]] = []
        for slot in SOURCE_SLOTS:
            configs.append(
                dict(
                    name=self.source_plate_name(slot),
                    obj_groups=TRAY_XML,
                    object_scale=[0.75, 0.75, 0.75],
                    placement=dict(
                        fixture=self.counter_zones_fixture,
                        size=(0.34, 0.30),
                        pos=SOURCE_LOCAL_OFFSETS[slot],
                    ),
                )
            )
        for slot in TARGET_SLOTS:
            configs.append(
                dict(
                    name=self.target_plate_name(slot),
                    obj_groups=TRAY_XML,
                    object_scale=[0.75, 0.75, 0.75],
                    placement=dict(
                        fixture=self.meal_counter,
                        sample_region_kwargs=dict(
                            ref=self.meal_paper_towel, loc="right"
                        ),
                        size=(0.46, 0.42),
                        pos=(1.0, 0.0),
                        rotation=0.0,
                        ensure_valid_placement=False,
                    ),
                )
            )
        for index, item in enumerate(ITEM_IDS):
            configs.append(
                dict(
                    name=item,
                    obj_groups=MILK_CARTON_XML_BY_ITEM[item],
                    object_scale=[0.75, 0.75, 0.75],
                    graspable=True,
                    init_robot_here=(index == 0),
                    placement=dict(
                        size=(0.20, 0.20),
                        rotation=0.0,
                        sample_args={
                            "reference": self.source_plate_name(
                                self.source_slot_by_item[item]
                            )
                        },
                    ),
                )
            )
        return configs

    def _setup_scene(self) -> None:
        super()._setup_scene()
        self._semantic_state = SemanticState()
        self._physical_observer = None
        self._wr08_evaluator = None
        self._physical_event_diagnostics = {}
        placements = getattr(self, "object_placements", {})
        names = [
            *(self.source_plate_name(slot) for slot in SOURCE_SLOTS),
            *(self.target_plate_name(slot) for slot in TARGET_SLOTS),
            *ITEM_IDS,
        ]
        missing = [name for name in names if name not in placements]
        if missing:
            raise RuntimeError(f"WR-08 reset is missing sampled placements: {missing}")
        if not record_initial_sources(
            self._semantic_state, self.source_slot_by_item, self.source_color_by_item
        ):
            raise RuntimeError("WR-08 initial source verification failed")

    def _reset_internal(self) -> None:
        super()._reset_internal()
        self._source_plate_centers = {
            slot: self._obj_pos(self.source_plate_name(slot)).copy()
            for slot in SOURCE_SLOTS
        }
        self._source_surface_z = float(
            np.mean([pos[2] for pos in self._source_plate_centers.values()])
        )
        self._target_plate_centers = {}
        for slot in TARGET_SLOTS:
            name = self.target_plate_name(slot)
            self.set_object_body_position(name, TARGET_PLATE_WORLD_CENTERS[slot])
            self._target_plate_centers[slot] = self._obj_pos(name).copy()
        self._target_surface_z = float(
            np.mean([pos[2] for pos in self._target_plate_centers.values()])
        )
        self._apply_milk_carton_poses()
        self._apply_plate_colors()
        self._update_task_state()

    def configure_episode(self, episode: Any) -> None:
        fields = (
            "source_plate_color_by_slot",
            "target_plate_color_by_slot",
            "source_slot_by_item",
            "source_color_by_item",
            "target_slot_by_color",
            "source_item_offset_xy_by_item",
            "target_item_offset_xy_by_item",
            "source_item_yaw_by_item",
            "target_item_yaw_by_item",
        )
        for field in fields:
            expected = getattr(episode, field)
            actual = getattr(self, field)
            if field.endswith("offset_xy_by_item"):
                if any(
                    not np.allclose(
                        _xy(expected[item], label=field),
                        actual[item],
                        atol=1e-12,
                        rtol=0.0,
                    )
                    for item in ITEM_IDS
                ):
                    raise ValueError(
                        f"WR-08 environment and EpisodePlan {field} differ"
                    )
            elif dict(expected) != dict(actual):
                raise ValueError(f"WR-08 environment and EpisodePlan {field} differ")
        order = tuple(str(item) for item in episode.transport_order)
        if len(order) != len(ITEM_IDS) or set(order) != set(ITEM_IDS):
            raise ValueError("WR-08 EpisodePlan transport_order is invalid")
        self._transport_order = order

    def source_plate_name(self, slot: str) -> str:
        return f"source_plate_{validate_source_slot(slot).removeprefix('source_')}"

    def target_plate_name(self, slot: str) -> str:
        return f"target_plate_{validate_target_slot(slot).removeprefix('target_')}"

    def _obj_pos(self, name: str) -> np.ndarray:
        return np.asarray(
            self.sim.data.body_xpos[self.obj_body_id[str(name)]], dtype=float
        )

    def set_object_body_position(self, obj_name: str, pos: Any) -> None:
        obj = self.objects[str(obj_name)]
        target = np.asarray(pos, dtype=float).reshape(-1)[:3]
        qpos = np.asarray(
            self.sim.data.get_joint_qpos(obj.joints[0]), dtype=float
        ).copy()
        qpos[:3] += target - self._obj_pos(str(obj_name))
        self.sim.data.set_joint_qpos(obj.joints[0], qpos)
        self.sim.data.set_joint_qvel(obj.joints[0], np.zeros(6))
        self.sim.forward()

    def set_object_pose(self, obj_name: str, pos: Any, yaw: float) -> None:
        obj = self.objects[str(obj_name)]
        qpos = np.asarray(
            self.sim.data.get_joint_qpos(obj.joints[0]), dtype=float
        ).copy()
        qpos[:3] = np.asarray(pos, dtype=float).reshape(-1)[:3]
        quat = _pose_quat_for_yaw(yaw)
        qpos[3:7] = np.array([quat[3], quat[0], quat[1], quat[2]], dtype=float)
        self.sim.data.set_joint_qpos(obj.joints[0], qpos)
        self.sim.data.set_joint_qvel(obj.joints[0], np.zeros(6))
        self.sim.forward()

    def _apply_milk_carton_poses(self) -> None:
        for item in ITEM_IDS:
            slot = self.source_slot_by_item[item]
            center = self._source_plate_centers[slot].copy()
            body_id = self.obj_body_id[self.source_plate_name(slot)]
            rotation = np.asarray(
                self.sim.data.body_xmat[body_id], dtype=float
            ).reshape(3, 3)[:2, :2]
            current = self._obj_pos(item).copy()
            current[:2] = (
                center[:2] + rotation @ self.source_item_offset_xy_by_item[item]
            )
            # As in WR-07, the reference sampler's Z is unreliable for this tray.
            plate = self.objects[self.source_plate_name(slot)]
            current[2] = (
                center[2] + plate.top_offset[-1]
                - self.objects[item].bottom_offset[-1]
            )
            self.set_object_pose(item, current, self.source_item_yaw_by_item[item])

    def _apply_plate_colors(self) -> None:
        for slot, color in self.source_plate_color_by_slot.items():
            self._set_body_color(self.source_plate_name(slot), COLOR_RGBA[color])
        for slot, color in self.target_plate_color_by_slot.items():
            self._set_body_color(self.target_plate_name(slot), COLOR_RGBA[color])

    def _set_body_color(
        self, name: str, rgba: tuple[float, float, float, float]
    ) -> None:
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

    def target_plate_pose(self, slot: str, item: str | None = None) -> np.ndarray:
        resolved = validate_target_slot(slot)
        target = self._target_plate_centers[resolved].copy()
        if item is not None:
            target[:2] += self.target_item_offset_xy_by_item[str(item)]
        return target

    def target_slot_for_item(self, item: str) -> str:
        return validate_target_slot(
            self.target_slot_by_color[self.source_color_by_item[str(item)]]
        )

    def detect_source_slot(self, item: str) -> str | None:
        pos = self._obj_pos(str(item))
        distances = {
            slot: float(np.linalg.norm(pos[:2] - center[:2]))
            for slot, center in self._source_plate_centers.items()
        }
        slot, distance = min(distances.items(), key=lambda pair: pair[1])
        return (
            slot
            if distance <= self.plate_threshold
            and abs(float(pos[2] - self._source_surface_z)) <= 0.35
            else None
        )

    def detect_target_slot(self, item: str) -> str | None:
        """Expert recording's geometric detector; policy evaluation uses contact."""
        pos = self._obj_pos(str(item))
        distances = {
            slot: float(
                np.linalg.norm(pos[:2] - self.target_plate_pose(slot, item)[:2])
            )
            for slot in TARGET_SLOTS
        }
        slot, distance = min(distances.items(), key=lambda pair: pair[1])
        return (
            slot
            if distance <= self.plate_threshold
            and abs(float(pos[2] - self._target_surface_z)) <= 0.35
            else None
        )

    def item_at_target(self, item: str, slot: str | None = None) -> bool:
        item = str(item)
        resolved = validate_target_slot(slot or self.target_slot_for_item(item))
        if self.detect_target_slot(item) != resolved or not self._item_released(item):
            return False
        try:
            from bench.execution.planning import object_reference_pos

            bottom_z = float(
                object_reference_pos(
                    self, item, placement_reference="bbox_bottom_center"
                )[2]
            )
        except Exception:
            return False
        return abs(bottom_z - float(self._target_plate_centers[resolved][2])) <= 0.06

    def physical_state(self) -> PhysicalState:
        """Legacy physical state consumed by the expert recording observer."""
        return PhysicalState(
            item_in_source={
                item: self.detect_source_slot(item) is not None for item in ITEM_IDS
            },
            item_in_target={
                item: self.detect_target_slot(item) is not None for item in ITEM_IDS
            },
            released={item: self._item_released(item) for item in ITEM_IDS},
        )

    def latch_source_color_memory(self, item: str) -> bool:
        return latch_source_color_memory(self._semantic_state, str(item))

    def reset_evaluation(self, *, expert_recording: bool = False) -> WR08Evaluator:
        """Select policy or recording evaluation independently of the data split."""
        self.require_expert_protocol = bool(expert_recording)
        self._evaluation_initial_item_positions = {
            item: self._obj_pos(item).copy() for item in ITEM_IDS
        }
        if self.require_expert_protocol:
            self.start_physical_observer()
        else:
            self._physical_observer = None
            self._physical_event_diagnostics = {}
        if self._wr08_evaluator is None:
            self._wr08_evaluator = WR08Evaluator(
                transport_order=self._transport_order,
                source_slot_by_item=self.source_slot_by_item,
                source_color_by_item=self.source_color_by_item,
                target_slot_by_item={item: self.target_slot_for_item(item) for item in ITEM_IDS},
            )
        self._wr08_evaluator.reset(self.evaluation_snapshot())
        return self._wr08_evaluator

    def start_physical_observer(self) -> None:
        """Start recording's pick/color-memory/placement event reconstruction."""
        self._evaluation_initial_item_positions = {
            item: self._obj_pos(item).copy() for item in ITEM_IDS
        }
        target_slots = {item: self.target_slot_for_item(item) for item in ITEM_IDS}
        self._physical_observer = WR08PhysicalObserver(
            self._semantic_state,
            transport_order=getattr(self, "_transport_order", ITEM_IDS),
            target_slot_by_item=target_slots,
            dwell_steps=self.physical_event_dwell_steps,
            terminal_verification_steps=self.terminal_verification_steps,
        )
        update = self._physical_observer.reset(self._physical_event_sample())
        self._physical_event_diagnostics = dict(update.diagnostics)

    def _update_task_state(self) -> None:
        if not self.require_expert_protocol:
            return
        observer = self._physical_observer
        if observer is None:
            return
        update = observer.observe(self._physical_event_sample())
        self._physical_event_diagnostics = {
            **dict(update.diagnostics),
            "last_step": int(update.step_id),
            "last_events": list(update.semantic_events),
            "duplicate_step": bool(update.duplicate_step),
        }

    def _physical_event_sample(self) -> PhysicalSample:
        return PhysicalSample(
            step_id=int(getattr(self, "timestep", 0)),
            item_in_source={
                item: self.detect_source_slot(item) is not None for item in ITEM_IDS
            },
            item_in_target={
                item: self.detect_target_slot(item) is not None for item in ITEM_IDS
            },
            released={item: self._item_released(item) for item in ITEM_IDS},
            observed_target_slot_by_item={
                item: self.detect_target_slot(item) for item in ITEM_IDS
            },
            item_grasped={
                item: bool(self._item_physical_grasp_evidence(item)["picked_up"])
                for item in ITEM_IDS
            },
            target_slot_by_item={
                item: self.target_slot_for_item(item) for item in ITEM_IDS
            },
        )

    def _policy_grasp_evidence(self, item: str) -> dict[str, Any]:
        """Use the same closed-gripper, contact, and 2 cm lift check as WR-07."""
        try:
            from bench.execution.manipulation.object_interactions import get_object_hold_status

            qpos = np.asarray(
                self.robots[0].get_gripper_joint_positions("right"), dtype=float
            ).reshape(-1)
            if qpos.size < 2 or not np.all(np.isfinite(qpos[:2])):
                raise ValueError("invalid gripper joint positions")
            hold = get_object_hold_status(self, item, max_end_effector_distance=0.18)
            baseline = self._evaluation_initial_item_positions.get(item)
            current = self._obj_pos(item)
            lift = None if baseline is None else float(current[2] - baseline[2])
            closed = bool(np.max(np.abs(qpos[:2])) <= 0.035)
            contact_known = "contact" in hold["contact_and_grasp"]
            released = bool(
                hold["checks"]["finite_pose"]
                and contact_known
                and not hold["checks"]["grasp_or_contact"]
            )
            return {
                "picked_up": bool(closed and hold["held"] and lift is not None and lift >= 0.02),
                "released": released,
                "gripper_closed": closed,
                "lift_m": lift,
                "hold": hold,
            }
        except Exception as exc:
            return {
                "picked_up": False,
                "released": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }

    def _policy_support_evidence(self, item: str) -> dict[str, Any]:
        """Read actual source/target tray and counter contacts, as in WR-07."""
        plate_names = [self.source_plate_name(slot) for slot in SOURCE_SLOTS]
        plate_names += [self.target_plate_name(slot) for slot in TARGET_SLOTS]
        try:
            support = {
                name: bool(self.check_contact(self.objects[item], self.objects[name]))
                for name in plate_names
            }
            on_counter = bool(OU.check_obj_any_counter_contact(self, item))
            return {
                "supported": bool(any(support.values()) or on_counter),
                "plate_support": support,
                "counter_contact": on_counter,
            }
        except Exception as exc:
            return {
                "supported": False,
                "plate_support": {name: False for name in plate_names},
                "counter_contact": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }

    def _policy_evaluation_snapshot(self) -> WR08EvaluationSnapshot:
        grasp = {item: self._policy_grasp_evidence(item) for item in ITEM_IDS}
        support = {item: self._policy_support_evidence(item) for item in ITEM_IDS}
        placed = {
            item: bool(grasp[item]["released"] and support[item]["supported"])
            for item in ITEM_IDS
        }
        observed_targets = {}
        for item in ITEM_IDS:
            contacts = [
                slot for slot in TARGET_SLOTS
                if support[item]["plate_support"][self.target_plate_name(slot)]
            ]
            # Multiple simultaneous target contacts do not identify a unique
            # destination. Neither XY position nor the expert's aim point is used.
            observed_targets[item] = contacts[0] if len(contacts) == 1 else None
        target_slots = {item: self.target_slot_for_item(item) for item in ITEM_IDS}
        collision_state = self._object_collision_diagnostics()
        return WR08EvaluationSnapshot(
            step_id=int(getattr(self, "timestep", 0)),
            transport_order=self._transport_order,
            source_slot_by_item=self.source_slot_by_item,
            source_color_by_item=self.source_color_by_item,
            target_slot_by_item=target_slots,
            item_picked_by_item={item: bool(grasp[item]["picked_up"]) for item in ITEM_IDS},
            item_placed_by_item=placed,
            item_matched_by_item={
                item: bool(placed[item] and observed_targets[item] == target_slots[item])
                for item in ITEM_IDS
            },
            released_by_item={item: bool(grasp[item]["released"]) for item in ITEM_IDS},
            target_slot_attempted_by_item={
                item: slot for item, slot in observed_targets.items()
                if placed[item] and slot is not None
            },
            details={
                "source_plate_color_by_slot": dict(self.source_plate_color_by_slot),
                "target_plate_color_by_slot": dict(self.target_plate_color_by_slot),
                "observed_target_slot_by_item": observed_targets,
                "current_item_positions": {item: self._obj_pos(item).tolist() for item in ITEM_IDS},
                "physical": {
                    "grasp_by_item": grasp,
                    "support_by_item": support,
                    "object_collisions_enabled": all(
                        entry["enabled"] for entries in collision_state.values() for entry in entries
                    ),
                    "object_collision_geoms": collision_state,
                },
                "collision_policy": "all_object_collisions_enabled",
                "execution_mode": "action_only_physics",
            },
        )

    def evaluation_snapshot(self) -> WR08EvaluationSnapshot:
        if not self.require_expert_protocol:
            return self._policy_evaluation_snapshot()
        self._update_task_state()
        state = self._semantic_state
        target_slots = {item: self.target_slot_for_item(item) for item in ITEM_IDS}
        physical = self.physical_state()
        report = evaluate_success(state, physical, target_slot_by_item=target_slots)
        collision_state = self._object_collision_diagnostics()
        grasp_evidence = {
            item: self._item_physical_grasp_evidence(item) for item in ITEM_IDS
        }
        return WR08EvaluationSnapshot(
            step_id=int(getattr(self, "timestep", 0)),
            transport_order=getattr(self, "_transport_order", ITEM_IDS),
            source_slot_by_item=self.source_slot_by_item,
            source_color_by_item=self.source_color_by_item,
            target_slot_by_item=target_slots,
            target_slot_attempted_by_item=state.target_slot_attempted_by_item,
            item_picked_by_item=dict(state.item_picked_by_item),
            memory_latched_by_item=dict(state.source_color_memory_latched_by_item),
            item_matched_by_item={
                item: bool(
                    state.item_placed_by_item.get(item, False)
                    and state.target_color_matches_by_item.get(item, False)
                )
                for item in ITEM_IDS
            },
            released_by_item=physical.released,
            protocol_violation=state.protocol_violation,
            domain_success=report.success,
            require_expert_protocol=True,
            details={
                **report.checks(),
                "step_id": int(getattr(self, "timestep", 0)),
                "source_plate_color_by_slot": dict(self.source_plate_color_by_slot),
                "target_plate_color_by_slot": dict(self.target_plate_color_by_slot),
                "source_slot_by_item": dict(self.source_slot_by_item),
                "source_color_by_item": dict(self.source_color_by_item),
                "target_slot_by_item": target_slots,
                "target_slot_attempted_by_item": dict(
                    state.target_slot_attempted_by_item
                ),
                "current_item_positions": {
                    item: self._obj_pos(item).tolist() for item in ITEM_IDS
                },
                "physical": {
                    "item_in_source": dict(physical.item_in_source),
                    "item_in_target": dict(physical.item_in_target),
                    "released": dict(physical.released),
                    "item_grasp_evidence": grasp_evidence,
                    "object_collisions_enabled": all(
                        entry["enabled"]
                        for entries in collision_state.values()
                        for entry in entries
                    ),
                    "object_collision_geoms": collision_state,
                },
                "physical_events": dict(self._physical_event_diagnostics),
                "collision_policy": "all_object_collisions_enabled",
                "execution_mode": "action_only_physics",
            },
        )

    def update_state(self) -> None:
        super().update_state()
        self._update_task_state()

    def _check_success(self) -> bool:
        evaluator = self._wr08_evaluator
        if evaluator is not None:
            return evaluator.observe(self.evaluation_snapshot()).success
        return self.evaluation_snapshot().domain_success

    def step(self, action: Any):
        # Recorded actions pass through unchanged; contacts determine transport.
        result = super().step(action)
        self._update_task_state()
        return result

    def task_diagnostics(self) -> dict[str, Any]:
        snapshot = self.evaluation_snapshot()
        if self._wr08_evaluator is not None:
            return self._wr08_evaluator.observe(snapshot).to_diagnostics()
        report = evaluate_success(
            self._semantic_state,
            self.physical_state(),
            target_slot_by_item={
                item: self.target_slot_for_item(item) for item in ITEM_IDS
            },
        )
        return {
            **dict(snapshot.details),
            "success": bool(report.success),
            "terminal": bool(report.success or self._semantic_state.protocol_violation),
            "failure_code": (
                "protocol_violation"
                if self._semantic_state.protocol_violation
                else None
            ),
            "success_checks": report.checks(),
        }

    def get_ep_meta(self) -> dict[str, Any]:
        from bench.tasks.wr08.prompts import resolve_prompt_spec

        meta = super().get_ep_meta()
        prompt = resolve_prompt_spec(variant="memory")
        meta["lang"], meta["initial_prompt"] = (
            prompt.policy_prompt,
            prompt.initial_prompt,
        )
        meta.setdefault("refs", {}).update(
            {
                "item_ids": list(ITEM_IDS),
                "item_labels": ITEM_LABELS,
                "source_plate_colors": dict(self.source_plate_color_by_slot),
                "target_plate_colors": dict(self.target_plate_color_by_slot),
                "source_slot_by_item": dict(self.source_slot_by_item),
                "target_slot_by_color": dict(self.target_slot_by_color),
                "target_slot_by_item": {
                    item: self.target_slot_for_item(item) for item in ITEM_IDS
                },
                "transport_order": list(getattr(self, "_transport_order", ITEM_IDS)),
            }
        )
        return meta

    def get_object_world_position(self, object_name: str) -> np.ndarray:
        return self._obj_pos(str(object_name)).copy()

    def get_object_world_quaternion_xyzw(self, object_name: str) -> np.ndarray:
        quat_wxyz = np.asarray(
            self.sim.data.body_xquat[self.obj_body_id[str(object_name)]],
            dtype=float,
        )
        return quat_wxyz[[1, 2, 3, 0]]

    def _item_physical_grasp_evidence(self, item_id: str) -> dict[str, Any]:
        """Recording's legacy grasp check; policies use ``_policy_grasp_evidence``."""

        try:
            from bench.execution.manipulation.object_interactions import (
                get_object_hold_status,
            )

            robot = self.robots[0]
            qpos = np.asarray(
                robot.get_gripper_joint_positions("right"), dtype=float
            ).reshape(-1)
            if qpos.size < 2 or not np.all(np.isfinite(qpos[:2])):
                return {"picked_up": False, "reason": "invalid_gripper_qpos"}
            hold = get_object_hold_status(
                self, str(item_id), max_end_effector_distance=0.18
            )
            current = self._obj_pos(str(item_id))
            baseline = self._evaluation_initial_item_positions.get(str(item_id))
            displacement = None if baseline is None else current - baseline
            lifted = bool(displacement is not None and float(displacement[2]) >= 0.02)
            lateral = bool(
                displacement is not None
                and float(np.linalg.norm(displacement[:2])) >= 0.04
            )
            checks = {
                "gripper_closed": bool(np.max(np.abs(qpos[:2])) <= 0.035),
                "held_near_end_effector": bool(hold.get("held", False)),
                "lifted_or_displaced": bool(lifted or lateral),
            }
            return {
                "picked_up": bool(all(checks.values())),
                "checks": checks,
                "gripper_qpos": qpos[:2].astype(float).tolist(),
                "initial_position": (
                    None if baseline is None else baseline.astype(float).tolist()
                ),
                "current_position": current.astype(float).tolist(),
                "displacement": (
                    None
                    if displacement is None
                    else displacement.astype(float).tolist()
                ),
                "hold": hold,
            }
        except Exception as exc:
            return {"picked_up": False, "reason": f"{type(exc).__name__}: {exc}"}

    def _object_collision_diagnostics(self) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {}
        for item_id in ITEM_IDS:
            entries: list[dict[str, Any]] = []
            for geom_name in self.objects[item_id].contact_geoms or ():
                geom_id = int(self.sim.model.geom_name2id(geom_name))
                contype = int(self.sim.model.geom_contype[geom_id])
                conaffinity = int(self.sim.model.geom_conaffinity[geom_id])
                entries.append(
                    {
                        "geom": str(geom_name),
                        "contype": contype,
                        "conaffinity": conaffinity,
                        "enabled": bool(contype != 0 and conaffinity != 0),
                    }
                )
            result[item_id] = entries
        return result

    def _item_released(self, item: str) -> bool:
        try:
            return bool(OU.gripper_obj_far(self, str(item), th=0.15))
        except Exception:
            return True


__all__ = ["WR08MatchMilkCartonsToColoredSugarPlates"]
