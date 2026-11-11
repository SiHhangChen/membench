"""Exercise the sample visibility gate with actual MuJoCo rays and cameras."""

from types import SimpleNamespace

import numpy as np
import pytest

from bench.tasks.wr11.domain.contracts import CAMERAS
from bench.tasks.wr11.visibility import object_camera_visibility

pytestmark = pytest.mark.simulator


def test_sample_observation_requires_line_of_sight_and_departure_hides_all_cameras():
    pytest.importorskip("mujoco")
    from bench.env.robocasa_runtime import bootstrap_local_robocasa_sources

    bootstrap_local_robocasa_sources()
    from robosuite.utils.binding_utils import MjSim

    cameras = "".join(f'<camera name="{name}" fovy="60"/>' for name in CAMERAS)
    sim = MjSim.from_xml_string(f"""
    <mujoco><worldbody>{cameras}
      <body name="sample" pos="0 0 -1">
        <geom name="sample_geom" type="box" size=".05 .05 .1" group="1"/>
      </body>
      <body name="occluder" pos="2 0 -.5"><freejoint name="occluder_joint"/>
        <geom type="box" size=".2 .2 .02" group="1"/>
      </body>
    </worldbody></mujoco>
    """)
    env = SimpleNamespace(sim=sim)
    geom_ids = {sim.model.geom_name2id("sample_geom")}
    sim.forward()
    assert all(
        value["visible"] for value in object_camera_visibility(env, geom_ids, CAMERAS).values()
    )
    sim.data.set_joint_qpos("occluder_joint", [0, 0, -0.5, 1, 0, 0, 0])
    sim.forward()
    occluded = object_camera_visibility(env, geom_ids, CAMERAS)
    assert all(value["in_frustum"] and not value["visible"] for value in occluded.values())
    # Rotate all cameras to face away, as the robot does when visiting stock.
    sim.model._model.cam_quat[:] = np.array([0, 0, 1, 0])
    sim.forward()
    assert all(
        not value["in_frustum"]
        for value in object_camera_visibility(env, geom_ids, CAMERAS).values()
    )
