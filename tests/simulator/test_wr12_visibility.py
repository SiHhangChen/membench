"""Exercise the calibrated shell proof with actual MuJoCo geometry and rays."""

import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bench.tasks.wr12.domain.contracts import CAMERAS, CAN_ID
from bench.tasks.wr12.visibility import can_visibility

pytestmark = pytest.mark.simulator


@pytest.mark.parametrize("hidden_camera", [(0, 0.5, 0.5), (-0.49, 0.46, 0.48)])
def test_side_observation_is_visible_and_front_or_seam_view_is_occluded(hidden_camera):
    pytest.importorskip("mujoco")
    from bench.env.robocasa_runtime import bootstrap_local_robocasa_sources

    bootstrap_local_robocasa_sources()
    from robosuite.utils.binding_utils import MjSim

    root = ET.parse(Path("bench/tasks/wr12/resources/assets/box.xml")).getroot()
    world = root.find("worldbody")
    body = ET.SubElement(world, "body", name="can", pos=".08 0 .057")
    ET.SubElement(body, "geom", name="can_visual", type="box", size=".031 .031 .041", group="1")
    for name in CAMERAS:
        ET.SubElement(
            world, "camera", name=name, pos="-1 0 .15", mode="targetbody", target="can", fovy="60"
        )
    sim = MjSim.from_xml_string(ET.tostring(root, encoding="unicode"))
    sim.forward()
    env = SimpleNamespace(
        sim=sim,
        obj_body_id={CAN_ID: sim.model.body_name2id("can")},
        objects={CAN_ID: SimpleNamespace(size=(0.062, 0.062, 0.082))},
        geometry={"box_xy": {"left": (0, 0)}},
        surface_z=0,
        _occluder_geom_ids=[
            sim.model.geom_name2id(name)
            for name in (
                "floor_visual",
                "roof_visual",
                "front_visual",
                "back_visual",
                "side_visual",
                "opening_front_lip_visual",
            )
        ],
        _can_visual_geoms={sim.model.geom_name2id("can_visual")},
    )
    initial = can_visibility(env)
    assert all(v["visible"] and not v["fully_hidden"] for v in initial.values())
    sim.model._model.cam_pos[:] = np.array(hidden_camera)
    sim.forward()
    hidden = can_visibility(env)
    assert all(v["in_frustum"] and v["fully_hidden"] and not v["visible"] for v in hidden.values())


@pytest.mark.parametrize(
    "camera_xyz",
    [(-1, 0, 0.15), (-1, 0.19, 0.29), (-1, -0.19, 0.01), (0, 0.5, 0.5), (0.8, 0, 0.15)],
)
def test_actual_sealed_shell_blocks_side_and_seam_rays(camera_xyz):
    mujoco = pytest.importorskip("mujoco")
    from bench.env.robocasa_runtime import bootstrap_local_robocasa_sources

    bootstrap_local_robocasa_sources()
    from robosuite.utils.binding_utils import MjSim

    from bench.tasks.wr12.visibility import box_shell_occludes

    root = ET.parse(Path("bench/tasks/wr12/resources/assets/box.xml")).getroot()
    shutter = ET.parse(Path("bench/tasks/wr12/resources/assets/shutter.xml")).getroot()
    world = root.find("worldbody")
    world.append(shutter.find('.//body[@name="object"]'))
    world[-1].set("name", "shutter")
    can = ET.SubElement(world, "body", name="can", pos=".06 -.09 .1375")
    ET.SubElement(can, "geom", name="can_visual", type="box", size=".091 .091 .1215", group="1")
    sim = MjSim.from_xml_string(ET.tostring(root, encoding="unicode"))
    sim.forward()
    from itertools import product

    corners = np.array(
        [
            np.array([0.06, -0.09, 0.1375]) + np.array(sign) * [0.091, 0.091, 0.1215]
            for sign in product((-1, 1), repeat=3)
        ]
    )
    origin = np.array(camera_xyz, dtype=float)
    assert box_shell_occludes(origin, corners, np.zeros(3), sealed=True)
    can_id = sim.model.geom_name2id("can_visual")
    for x in np.linspace(-0.031, 0.151, 5):
        for y in np.linspace(-0.181, 0.001, 5):
            for z in np.linspace(0.016, 0.259, 5):
                direction = np.array([x, y, z]) - origin
                hit = np.array([-1], dtype=np.int32)
                distance = mujoco.mj_ray(
                    sim.model._model,
                    sim.data._data,
                    origin,
                    direction / np.linalg.norm(direction),
                    np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8),
                    1,
                    -1,
                    hit,
                )
                assert distance >= 0 and hit[0] != can_id
