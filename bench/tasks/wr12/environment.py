"""Two boxes seal automatically after a shared one-second viewing period."""

import json
from pathlib import Path

import numpy as np
from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.models import assets_root
from robosuite.models.objects import MujocoXMLObject

from bench.execution.manipulation.object_interactions import get_object_hold_status

from .contact_evidence import ContactIndex
from .domain.contracts import BLOCK_ID, BOX_IDS, CAMERAS, CAN_ID
from .evaluation import WR12EvaluationSnapshot, WR12Evaluator
from .visibility import can_visibility


class WR12PlaceAtRememberedBox(Kitchen):
    def __init__(
        self,
        *,
        target_box,
        block_offset_xy,
        block_yaw,
        placement_offset_xy,
        base_xy_by_station,
        scene_geometry,
        block_asset,
        box_asset,
        tray_asset,
        shutter_asset,
        viewing_steps=20,
        shutter_closing_steps=6,
        observation_dwell_steps=3,
        hidden_dwell_steps=3,
        placement_dwell_steps=3,
        minimum_lift_m=0.02,
        can_motion_tolerance_m=0.015,
        **kwargs,
    ):
        if target_box not in BOX_IDS:
            raise ValueError("WR-12 unknown target")
        self.target_box = target_box
        self.block_offset_xy = np.array(block_offset_xy)
        self.block_yaw = float(block_yaw)
        self.placement_offset_xy = np.array(placement_offset_xy)
        self.base_xy_by_station = dict(base_xy_by_station)
        self.geometry = json.loads(Path(scene_geometry).read_text())["entries"]
        self.block_asset = str(block_asset)
        self.box_asset, self.tray_asset = str(box_asset), str(tray_asset)
        self.shutter_asset = str(shutter_asset)
        self.viewing_steps = int(viewing_steps)
        self.shutter_closing_steps = int(shutter_closing_steps)
        if self.viewing_steps != 20 or self.shutter_closing_steps != 6:
            raise ValueError("WR-12 timed shutters require 20 viewing and 6 closing steps")
        self._shutter_clock_start = None
        self._shutter_progress = 0.0
        self.minimum_lift_m = float(minimum_lift_m)
        self.can_motion_tolerance_m = float(can_motion_tolerance_m)
        self._evaluator_settings = {
            "observation_dwell_steps": observation_dwell_steps,
            "hidden_dwell_steps": hidden_dwell_steps,
            "placement_dwell_steps": placement_dwell_steps,
        }
        self.evaluator = None
        self._initial_positions = {}
        self._geom_ids = {}
        kwargs.setdefault("obj_registries", ("objaverse", "aigen", "lightwheel"))
        super().__init__(**kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref(
            "task_counter",
            {"id": self.geometry["fixture"], "full_name_check": True, "full_depth_region": True},
        )
        self.init_robot_base_ref = self.counter
        self.surface_z = float(self.counter.pos[2] + self.counter.size[2] / 2)
        self.boxes, self.trays, self.shutters = {}, {}, {}
        for kind, asset, positions, objects in (
            ("box", self.box_asset, self.geometry["box_xy"], self.boxes),
            ("tray", self.tray_asset, self.geometry["tray_xy"], self.trays),
            ("shutter", self.shutter_asset, self.geometry["box_xy"], self.shutters),
        ):
            for label in BOX_IDS:
                obj = MujocoXMLObject(
                    asset,
                    name=f"{kind}_{label}",
                    joints=None,
                    obj_type="all",
                    duplicate_collision_geoms=False,
                )
                obj.get_obj().set("pos", " ".join(map(str, [*positions[label], self.surface_z])))
                self.model.merge_objects([obj])
                objects[label] = obj

    def sample_object(self, groups, **kwargs):
        # Explicit task-owned primitive; all registry objects use the normal path.
        if str(groups) == self.block_asset:
            return (
                {
                    "mjcf_path": self.block_asset,
                    "scale": 1.0,
                    "density": 500,
                    "friction": (1.2, 0.01, 0.001),
                },
                {
                    "groups_containing_sampled_obj": [],
                    "groups": groups,
                    "cat": "block",
                    "split": None,
                    "mjcf_path": self.block_asset,
                },
            )
        return super().sample_object(groups, **kwargs)

    def _get_obj_cfgs(self):
        return [
            {
                "name": item,
                "obj_groups": asset,
                "graspable": True,
                "object_scale": self.geometry["can_scale"] if item == CAN_ID else 1.0,
                "init_robot_here": item == BLOCK_ID,
                "placement": {
                    "fixture": self.counter,
                    "size": (0.30, 0.30),
                    "pos": (0.0, -1.0),
                    "rotation": 0.0,
                    "ensure_valid_placement": False,
                },
            }
            for item, asset in (
                (BLOCK_ID, self.block_asset),
                (CAN_ID, f"{assets_root}/objects/objaverse/canned_food/canned_food_15/model.xml"),
            )
        ]

    def _obj_pos(self, item):
        return np.array(self.sim.data.body_xpos[self.obj_body_id[item]])

    def _set_reset_pose(self, item, position, yaw):
        joint = self.objects[item].joints[0]
        self.sim.data.set_joint_qpos(
            joint, np.array([*position, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        )
        self.sim.data.set_joint_qvel(joint, np.zeros(6))
        self.sim.forward()

    def _reset_internal(self):
        self.evaluator = None
        self._shutter_clock_start = None
        self._shutter_progress = 0.0
        super()._reset_internal()
        self._set_shutter_progress(0.0)
        block_xy = np.array(self.geometry["block_xy"]) + self.block_offset_xy
        self._set_reset_pose(
            BLOCK_ID,
            [*block_xy, self.surface_z - self.objects[BLOCK_ID].bottom_offset[-1] + 0.002],
            self.block_yaw,
        )
        can_xy = np.array(self.geometry["box_xy"][self.target_box])
        can_xy += self.geometry["can_offset_xy_by_box"][self.target_box]
        self._set_reset_pose(
            CAN_ID,
            [*can_xy, self.surface_z + 0.016 - self.objects[CAN_ID].bottom_offset[-1] + 0.002],
            0.0,
        )
        self._geom_ids = {}
        self._occluder_geom_ids = [
            self.sim.model.geom_name2id(name)
            for obj in (*self.boxes.values(), *self.shutters.values())
            for name in obj.visual_geoms
            if not name.endswith("reg_bbox")
        ]
        self._can_visual_geoms = {
            self.sim.model.geom_name2id(name)
            for name in self.objects[CAN_ID].visual_geoms
            if not name.endswith("reg_bbox")
        }
        for _ in range(int(0.4 / self.model_timestep)):
            self.sim.step()
        self.sim.forward()

    def configure_episode(self, plan):
        if self.target_box != plan.target_box or self.block_yaw != plan.block_yaw:
            raise ValueError("WR-12 episode/environment mismatch")

    def reset_evaluation(self):
        # Arm only after scene settling and robot initialization. This clock is
        # shared by expert recording, policy execution and action replay.
        self._set_shutter_progress(0.0)
        self._shutter_clock_start = float(self.sim.data.time)
        self._initial_positions = {item: self._obj_pos(item) for item in (BLOCK_ID, CAN_ID)}
        contacts = ContactIndex(self.sim, self._geom_ids)
        if not contacts.check(self.objects[BLOCK_ID], self.counter):
            raise RuntimeError("WR-12 block is not supported on the table")
        if not contacts.check(self.objects[CAN_ID], self.boxes[self.target_box]):
            raise RuntimeError("WR-12 can is not supported in its box")
        self.evaluator = WR12Evaluator(self.target_box, **self._evaluator_settings)
        self.evaluator.reset(self.evaluation_snapshot())
        return self.evaluator

    def _set_shutter_progress(self, progress):
        self._shutter_progress = float(np.clip(progress, 0.0, 1.0))
        for label, obj in self.shutters.items():
            position = [
                *self.geometry["box_xy"][label],
                self.surface_z + 0.32 * (1.0 - self._shutter_progress),
            ]
            self.sim.data.set_mocap_pos(obj.root_body, position)
            self.sim.data.set_mocap_quat(obj.root_body, [1, 0, 0, 0])
        self.sim.forward()

    @property
    def boxes_sealed(self):
        # Inspect the actual moving panel pose, not the timer's target value.
        for label, obj in self.shutters.items():
            body = self.sim.model.body_name2id(obj.root_body)
            expected = np.array([*self.geometry["box_xy"][label], self.surface_z])
            if not np.allclose(self.sim.data.body_xpos[body], expected, atol=1e-6):
                return False
            if not np.allclose(self.sim.data.body_xmat[body], np.eye(3).reshape(-1), atol=1e-6):
                return False
        return True

    @property
    def shutter_clock_step(self):
        if self._shutter_clock_start is None:
            return 0
        elapsed = float(self.sim.data.time) - self._shutter_clock_start
        return round(elapsed / self.control_timestep)

    def _pre_action(self, action, policy_step=False):
        if self._shutter_clock_start is not None:
            # Move both opaque collision panels continuously during simulation
            # substeps. Their prescribed motion never depends on the answer.
            elapsed = float(self.sim.data.time) - self._shutter_clock_start + self.model_timestep
            progress = (
                elapsed / self.control_timestep - self.viewing_steps
            ) / self.shutter_closing_steps
            progress = float(np.clip(progress, 0.0, 1.0))
            if progress < 1e-10:
                progress = 0.0
            elif progress > 1.0 - 1e-10:
                progress = 1.0
            if progress != self._shutter_progress:
                self._set_shutter_progress(progress)
        super()._pre_action(action, policy_step=policy_step)

    def delivery_pose(self):
        xy = np.array(self.geometry["tray_xy"][self.target_box]) + self.placement_offset_xy
        return np.array([*xy, self.surface_z + 0.012])

    def evaluation_snapshot(self):
        visibility_error = None
        try:
            visibility = can_visibility(self)
            hidden = len(visibility) == len(CAMERAS) and all(
                v["fully_hidden"] for v in visibility.values()
            )
            visible = any(v["visible"] for v in visibility.values())
            occluded = any(v["in_frustum"] and v["occluding_panel"] for v in visibility.values())
        except Exception as exc:  # noqa: BLE001 - unknown visibility cannot qualify
            visibility, hidden, visible, occluded = {}, False, False, False
            visibility_error = f"{type(exc).__name__}: {exc}"
        contacts = ContactIndex(self.sim, self._geom_ids)
        hold = get_object_hold_status(self, BLOCK_ID, contact_checker=contacts.check)
        qpos = np.asarray(self.robots[0].get_gripper_joint_positions("right"))
        closed = qpos.size >= 2 and np.all(np.isfinite(qpos)) and np.max(np.abs(qpos)) <= 0.035
        lift = float(self._obj_pos(BLOCK_ID)[2] - self._initial_positions[BLOCK_ID][2])
        picked = bool(closed and hold["held"] and lift >= self.minimum_lift_m)
        released = bool(hold["checks"]["finite_pose"] and not hold["checks"]["grasp_or_contact"])
        in_trays = []
        for label in BOX_IDS:
            relative = self._obj_pos(BLOCK_ID)[:2] - np.array(self.geometry["tray_xy"][label])
            # Support contact plus interior center, independent of expert target point.
            if (
                released
                and np.all(np.abs(relative) <= (0.12, 0.105))
                and contacts.check(self.objects[BLOCK_ID], self.trays[label])
            ):
                in_trays.append(label)
        base_site = self.sim.model.site_name2id(
            self.robots[0].robot_model.base.correct_naming("center")
        )
        base = np.asarray(self.sim.data.site_xpos[base_site])[:2]
        displacement = float(
            np.linalg.norm(self._obj_pos(CAN_ID) - self._initial_positions[CAN_ID])
        )
        return WR12EvaluationSnapshot(
            step_id=int(self.timestep),
            at_observe=bool(
                np.linalg.norm(base - self.base_xy_by_station["observe_station"]) < 0.20
            ),
            can_visible=visible,
            at_operate=bool(
                np.linalg.norm(base - self.base_xy_by_station["operate_station"]) < 0.20
            ),
            can_hidden=hidden,
            boxes_sealed=self.boxes_sealed,
            can_occluded_by_box=bool(occluded),
            can_undisturbed=displacement <= self.can_motion_tolerance_m,
            block_picked=picked,
            released_in=tuple(in_trays),
            details={
                "visibility": visibility,
                "visibility_error": visibility_error,
                "boxes_sealed": self.boxes_sealed,
                "shutter_progress": self._shutter_progress,
                "shutter_clock_step": self.shutter_clock_step,
                "can_displacement_m": displacement,
                "block_hold": hold,
                "block_lift_m": lift,
                "base_xy": base.tolist(),
            },
        )

    def update_state(self):
        super().update_state()
        if self.evaluator is not None:
            self.evaluator.observe(self.evaluation_snapshot())

    def _check_success(self):
        return bool(
            self.evaluator is not None
            and self.evaluator.observe(self.evaluation_snapshot()).success
        )

    def task_diagnostics(self):
        return (
            {"success": False}
            if self.evaluator is None
            else self.evaluator.observe(self.evaluation_snapshot()).to_diagnostics()
        )

    def get_ep_meta(self):
        from .prompts import PROMPT

        meta = super().get_ep_meta()
        meta["lang"] = meta["initial_prompt"] = PROMPT
        return meta
