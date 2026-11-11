"""Physical WR-11 scene. The sample remains visible at its original station."""

import numpy as np
from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.models import assets_root

from bench.execution.manipulation.object_interactions import get_object_hold_status

from .appearance import CartonAppearance
from .contact_evidence import ContactIndex
from .domain.contracts import CAMERAS, CARTON_ASSET, ITEM_IDS, PACKAGING_RGB, SAMPLE_ID
from .domain.scene import (
    DELIVERY_TRAY_XY,
    SAMPLE_FIXTURE,
    SAMPLE_STATION_XY,
    SAMPLE_TRAY_XY,
    STOCK_FIXTURE,
    STOCK_STATION_XY,
    STOCK_TRAY_XYS,
)
from .evaluation import WR11EvaluationSnapshot, WR11Evaluator
from .visibility import object_camera_visibility


class WR11FetchMatchingMilkCarton(CartonAppearance, Kitchen):
    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(
        self,
        *,
        sample_packaging,
        packaging_by_item,
        offset_xy_by_item,
        yaw_by_item,
        observation_dwell_steps=3,
        placement_dwell_steps=3,
        minimum_lift_m=0.02,
        sample_motion_tolerance_m=0.06,
        **kwargs,
    ):
        if sample_packaging not in PACKAGING_RGB or set(packaging_by_item) != set(ITEM_IDS):
            raise ValueError("invalid WR-11 sample or stock assignment")
        colors = set(packaging_by_item.values())
        if len(colors) != len(ITEM_IDS) or not colors <= set(PACKAGING_RGB):
            raise ValueError("WR-11 stock must contain three distinct valid colors")
        if sample_packaging not in colors:
            raise ValueError("WR-11 stock must contain exactly one match for the sample")
        self.sample_packaging = sample_packaging
        self.packaging_by_item = dict(packaging_by_item)
        self.matching_item = next(
            item for item in ITEM_IDS if packaging_by_item[item] == sample_packaging
        )
        self.offset_xy_by_item = {
            key: np.asarray(value, dtype=float) for key, value in offset_xy_by_item.items()
        }
        self.yaw_by_item = dict(yaw_by_item)
        if set(self.offset_xy_by_item) != {*ITEM_IDS, SAMPLE_ID} or set(self.yaw_by_item) != {
            *ITEM_IDS,
            SAMPLE_ID,
        }:
            raise ValueError("WR-11 poses must cover the sample and three stock items")
        for item, offset in self.offset_xy_by_item.items():
            if (
                offset.shape != (2,)
                or not np.all(np.isfinite(offset))
                or np.linalg.norm(offset) > 0.045 + 1e-9
            ):
                raise ValueError("invalid WR-11 object offset")
            if not np.isfinite(self.yaw_by_item[item]):
                raise ValueError("invalid WR-11 object yaw")
        self.minimum_lift_m = float(minimum_lift_m)
        self.sample_motion_tolerance_m = float(sample_motion_tolerance_m)
        self._evaluator_settings = {
            "observation_dwell_steps": observation_dwell_steps,
            "placement_dwell_steps": placement_dwell_steps,
        }
        self.evaluator = None
        self._initial_positions = {}
        self._object_geoms = {}
        self._geom_ids = {}
        kwargs.setdefault("obj_registries", ("objaverse", "aigen", "lightwheel"))
        super().__init__(**kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stock_counter = self.register_fixture_ref(
            "stock_counter",
            {"id": STOCK_FIXTURE, "full_name_check": True, "full_depth_region": True},
        )
        self.sample_counter = self.register_fixture_ref(
            "sample_counter",
            {"id": SAMPLE_FIXTURE, "full_name_check": True, "full_depth_region": True},
        )
        self.init_robot_base_ref = self.sample_counter

    @staticmethod
    def stock_tray_name(item):
        if item not in ITEM_IDS:
            raise ValueError("unknown WR-11 stock item")
        return f"{item}_tray"

    def _get_obj_cfgs(self):
        tray_xml = f"{assets_root}/objects/lightwheel/oven_tray/OvenTray001/model.xml"
        configs = []
        for index, item in enumerate(ITEM_IDS):
            configs.append(
                {
                    "name": self.stock_tray_name(item),
                    "obj_groups": tray_xml,
                    "object_scale": [0.75] * 3,
                    "placement": {
                        "fixture": self.stock_counter,
                        "size": (0.34, 0.30),
                        "pos": (float(index - 1), -1.0),
                        "rotation": 0.0,
                    },
                }
            )
        for name in ("sample_tray", "delivery_tray"):
            configs.append(
                {
                    "name": name,
                    "obj_groups": tray_xml,
                    "object_scale": [0.75] * 3,
                    "placement": {
                        "fixture": self.sample_counter,
                        "size": (0.46, 0.42),
                        "pos": (1.0, 0.0),
                        "rotation": 0.0,
                        "ensure_valid_placement": False,
                    },
                }
            )
        for item in (*ITEM_IDS, SAMPLE_ID):
            tray = "sample_tray" if item == SAMPLE_ID else self.stock_tray_name(item)
            configs.append(
                {
                    "name": item,
                    "obj_groups": f"{assets_root}/objects/{CARTON_ASSET}",
                    "object_scale": [0.75] * 3,
                    "graspable": True,
                    "init_robot_here": (item == SAMPLE_ID),
                    "placement": {
                        "size": (0.20, 0.20),
                        "rotation": 0.0,
                        "sample_args": {"reference": tray},
                    },
                }
            )
        return configs

    def _setup_scene(self):
        super()._setup_scene()
        self.evaluator = None

    def _obj_pos(self, item):
        return np.asarray(self.sim.data.body_xpos[self.obj_body_id[item]], dtype=float).copy()

    def _set_reset_pose(self, item, position, yaw):
        """Reset only: no object joint writes are performed during execution."""
        joint = self.objects[item].joints[0]
        qpos = np.array([*position, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        self.sim.data.set_joint_qpos(joint, qpos)
        self.sim.data.set_joint_qvel(joint, np.zeros(6))
        self.sim.forward()

    def _reset_internal(self):
        self.evaluator = None
        super()._reset_internal()
        for tray, xy, fixture in (
            ("sample_tray", SAMPLE_TRAY_XY, self.sample_counter),
            ("delivery_tray", DELIVERY_TRAY_XY, self.sample_counter),
            *(
                (self.stock_tray_name(item), xy, self.stock_counter)
                for item, xy in zip(ITEM_IDS, STOCK_TRAY_XYS)
            ),
        ):
            top_z = fixture.pos[2] + fixture.size[2] / 2
            position = (*xy, top_z - self.objects[tray].bottom_offset[-1])
            self._set_reset_pose(tray, position, 0.0)
        for item in (*ITEM_IDS, SAMPLE_ID):
            tray = "sample_tray" if item == SAMPLE_ID else self.stock_tray_name(item)
            position = self._obj_pos(tray)
            rotation = np.asarray(self.sim.data.body_xmat[self.obj_body_id[tray]]).reshape(3, 3)
            position[:2] += rotation[:2, :2] @ self.offset_xy_by_item[item]
            position[2] += self.objects[tray].top_offset[-1] - self.objects[item].bottom_offset[-1]
            self._set_reset_pose(item, position, self.yaw_by_item[item])
            packaging = self.sample_packaging if item == SAMPLE_ID else self.packaging_by_item[item]
            self._apply_carton_material(item, packaging)
        # All support trays are identical neutral surfaces, carrying no request cue.
        for tray in (
            "sample_tray",
            "delivery_tray",
            *(self.stock_tray_name(item) for item in ITEM_IDS),
        ):
            self._set_body_color(tray, (0.65, 0.65, 0.65, 1.0))
        self._index_object_geoms()
        # The tray top offset includes its rim. Let the cartons fall onto the
        # interior through physics before initialization verifies contact.
        # This runs only inside reset, before any observations are recorded.
        for _ in range(int(0.4 / self.model_timestep)):
            self.sim.step()
        self.sim.forward()

    def _index_object_geoms(self):
        parents = np.asarray(self.sim.model.body_parentid, dtype=int)
        self._object_geoms = {}
        for item in (*ITEM_IDS, SAMPLE_ID):
            root = self.obj_body_id[item]
            geoms = set()
            for gid, body in enumerate(self.sim.model.geom_bodyid):
                while body > 0 and body != root:
                    body = parents[body]
                if body == root:
                    geoms.add(gid)
            self._object_geoms[item] = geoms
        # Independent textures are required: recoloring the sample must not
        # recolor all stock items through a shared MuJoCo material/texture ID.
        texture_sets = []
        raw = self.sim.model._model
        for item in (*ITEM_IDS, SAMPLE_ID):
            textures = {
                int(texture)
                for gid in self._object_geoms[item]
                if self.sim.model.geom_matid[gid] >= 0
                for texture in np.asarray(raw.mat_texid[self.sim.model.geom_matid[gid]]).reshape(-1)
                if texture >= 0
            }
            if not textures or any(textures & previous for previous in texture_sets):
                raise RuntimeError("WR-11 packaging textures must be independent per carton")
            texture_sets.append(textures)
        self._geom_ids = {}

    def configure_episode(self, plan):
        for field in ("sample_packaging", "packaging_by_item", "matching_item", "yaw_by_item"):
            if getattr(self, field) != getattr(plan, field):
                raise ValueError(f"WR-11 environment and episode {field} differ")
        for item in (*ITEM_IDS, SAMPLE_ID):
            if not np.array_equal(self.offset_xy_by_item[item], plan.offset_xy_by_item[item]):
                raise ValueError("WR-11 environment and episode offsets differ")

    def reset_evaluation(self):
        self._initial_positions = {item: self._obj_pos(item) for item in (*ITEM_IDS, SAMPLE_ID)}
        contacts = ContactIndex(self.sim, self._geom_ids)
        for item in (*ITEM_IDS, SAMPLE_ID):
            tray = "sample_tray" if item == SAMPLE_ID else self.stock_tray_name(item)
            if not contacts.check(self.objects[item], self.objects[tray]):
                raise RuntimeError(f"WR-11 reset: {item} is not physically supported on {tray}")
        self.evaluator = WR11Evaluator(self.matching_item, **self._evaluator_settings)
        self.evaluator.reset(self.evaluation_snapshot())
        if not self.evaluation_snapshot().sample_visible:
            raise RuntimeError("WR-11 reset: sample is not visible at the sampled initial station")
        return self.evaluator

    def delivery_pose(self):
        position = self._obj_pos("delivery_tray")
        position[2] += self.objects["delivery_tray"].top_offset[-1]
        return position

    def sample_visibility(self):
        return object_camera_visibility(self, self._object_geoms[SAMPLE_ID], CAMERAS)

    def evaluation_snapshot(self):
        visibility_error = None
        try:
            visibility = self.sample_visibility()
            in_any = any(value["in_frustum"] for value in visibility.values())
            visible = any(value["visible"] for value in visibility.values())
        except Exception as exc:  # noqa: BLE001 - missing visibility must fail closed
            visibility, in_any, visible = {}, True, False
            visibility_error = f"{type(exc).__name__}: {exc}"
        contacts = ContactIndex(self.sim, self._geom_ids)
        picked, delivered, physical = [], [], {}
        qpos = np.asarray(self.robots[0].get_gripper_joint_positions("right"))
        closed = bool(
            qpos.size >= 2 and np.all(np.isfinite(qpos)) and np.max(np.abs(qpos)) <= 0.035
        )
        for item in ITEM_IDS:
            try:
                hold = get_object_hold_status(self, item, contact_checker=contacts.check)
                lift = float(self._obj_pos(item)[2] - self._initial_positions[item][2])
                if closed and hold["held"] and lift >= self.minimum_lift_m:
                    picked.append(item)
                released = (
                    hold["checks"]["finite_pose"]
                    and "contact" in hold["contact_and_grasp"]
                    and not hold["checks"]["grasp_or_contact"]
                )
                on_delivery = contacts.check(self.objects[item], self.objects["delivery_tray"])
                if released and on_delivery:
                    delivered.append(item)
                physical[item] = {
                    "hold": hold,
                    "lift_m": lift,
                    "released": released,
                    "delivery_contact": on_delivery,
                }
            except Exception as exc:  # noqa: BLE001 - unknown grasp/contact cannot qualify
                physical[item] = {"error": f"{type(exc).__name__}: {exc}"}
        base_site = self.sim.model.site_name2id(
            self.robots[0].robot_model.base.correct_naming("center")
        )
        base = np.asarray(self.sim.data.site_xpos[base_site])[:2]
        # Fixed station regions are independent of the sampled sample identity.
        at_sample = np.linalg.norm(base - np.array(SAMPLE_STATION_XY)) <= 0.35
        at_stock = np.linalg.norm(base - np.array(STOCK_STATION_XY)) <= 0.40
        sample_displacement = float(
            np.linalg.norm(self._obj_pos(SAMPLE_ID) - self._initial_positions[SAMPLE_ID])
        )
        return WR11EvaluationSnapshot(
            step_id=int(self.timestep),
            at_sample=bool(at_sample),
            sample_visible=visible,
            sample_in_any_camera=in_any,
            at_stock=bool(at_stock),
            sample_undisturbed=bool(sample_displacement <= self.sample_motion_tolerance_m),
            picked_items=tuple(picked),
            delivered_items=tuple(delivered),
            details={
                "sample_visibility": visibility,
                "visibility_error": visibility_error,
                "sample_displacement_m": sample_displacement,
                "physical": physical,
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
        if self.evaluator is None:
            return {"success": False}
        return self.evaluator.observe(self.evaluation_snapshot()).to_diagnostics()

    def get_ep_meta(self):
        from .prompts import PROMPT

        meta = super().get_ep_meta()
        meta["lang"] = meta["initial_prompt"] = PROMPT
        return meta
