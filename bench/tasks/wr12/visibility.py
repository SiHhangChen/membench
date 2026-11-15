"""Conservative whole-object occlusion by fixed opaque box panels.

For each camera, a single convex panel must block all eight corners of the
can's bounding box (or the whole bounding sphere must be out of view). The
shadow of that panel is convex, so this covers the entire bounding box; a
blocked center ray alone cannot establish invisibility. Observation uses
MuJoCo visual-geometry rays as a separate positive visibility check. The
multi-panel fallback projects the target cone through the sole side opening.
"""

import xml.etree.ElementTree as ET
from itertools import product

import numpy as np


def validate_box_asset(path):
    """The closed-shell proof is valid only for this opaque, sealed geometry."""
    expected = {
        "floor": ((0, 0, 0.008), (0.17, 0.20, 0.008)),
        "roof": ((0, 0, 0.292), (0.17, 0.20, 0.008)),
        "front": ((0, 0.192, 0.15), (0.17, 0.008, 0.134)),
        "back": ((0, -0.192, 0.15), (0.17, 0.008, 0.134)),
        "side": ((0.162, 0, 0.15), (0.008, 0.20, 0.134)),
        "opening_front_lip": ((-0.162, 0.14, 0.15), (0.008, 0.06, 0.134)),
    }
    root = ET.parse(path).getroot()
    for name, (position, halfsize) in expected.items():
        for kind, group in (("visual", 1), ("collision", 0)):
            geom = root.find(f'.//geom[@name="{name}_{kind}"]')
            if geom is None or geom.get("type") != "box" or geom.get("group") != str(group):
                raise ValueError("WR-12 occlusion requires the calibrated sealed box asset")
            if not np.allclose(
                np.fromstring(geom.get("pos", ""), sep=" "), position
            ) or not np.allclose(np.fromstring(geom.get("size", ""), sep=" "), halfsize):
                raise ValueError("WR-12 box geometry differs from the occlusion proof")
            if any(geom.get(key) for key in ("quat", "euler", "axisangle")):
                raise ValueError("WR-12 box panels must remain axis aligned")
            if kind == "visual" and np.fromstring(geom.get("rgba", ""), sep=" ")[-1] != 1:
                raise ValueError("WR-12 box must remain fully opaque")
            if kind == "collision" and (geom.get("contype"), geom.get("conaffinity")) != ("1", "1"):
                raise ValueError("WR-12 box physical collisions must remain enabled")


def validate_shutter_asset(path):
    """A sealed-shell proof requires this opaque panel to overlap every edge."""
    root = ET.parse(path).getroot()
    body = root.find('.//body[@name="object"]')
    if body is None or body.get("mocap") != "true" or body.find("joint") is not None:
        raise ValueError("WR-12 shutters must follow the environment clock")
    for kind, group in (("visual", 1), ("collision", 0)):
        geom = body.find(f'geom[@name="panel_{kind}"]')
        if geom is None or geom.get("type") != "box" or geom.get("group") != str(group):
            raise ValueError("WR-12 requires an opaque physical shutter")
        if not np.allclose(np.fromstring(geom.get("pos", ""), sep=" "), [-0.178, 0, 0.15]):
            raise ValueError("WR-12 shutter position differs from the sealed-shell proof")
        if not np.allclose(np.fromstring(geom.get("size", ""), sep=" "), [0.012, 0.20, 0.15]):
            raise ValueError("WR-12 shutter must overlap all opening edges")
        if any(geom.get(key) for key in ("quat", "euler", "axisangle")):
            raise ValueError("WR-12 shutter must remain axis aligned")
        if kind == "visual" and np.fromstring(geom.get("rgba", ""), sep=" ")[-1] != 1:
            raise ValueError("WR-12 shutter must remain opaque")
        if kind == "collision" and (geom.get("contype"), geom.get("conaffinity")) != ("1", "1"):
            raise ValueError("WR-12 shutter collisions must remain enabled")


def sphere_in_camera(center, radius, fov, aspect=1.0):
    x, y, z = center
    depth = -z
    ty = np.tan(np.deg2rad(fov) / 2)
    tx = ty * aspect
    return bool(
        depth + radius > 0
        and abs(x) <= depth * tx + radius * np.sqrt(1 + tx * tx)
        and abs(y) <= depth * ty + radius * np.sqrt(1 + ty * ty)
    )


def segment_intersects_box(origin, endpoint, center, rotation, halfsize, *, margin=0.001):
    """Require intersection strictly before the target, inside a shrunken panel."""
    start = rotation.T @ (origin - center)
    delta = rotation.T @ (endpoint - origin)
    half = np.asarray(halfsize) - margin
    if np.any(half <= 0):
        return False
    lo, hi = 0.0, 1.0 - 1e-6
    for i in range(3):
        if abs(delta[i]) < 1e-12:
            if abs(start[i]) > half[i]:
                return False
        else:
            a, b = (-half[i] - start[i]) / delta[i], (half[i] - start[i]) / delta[i]
            lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
            if lo > hi:
                return False
    return hi >= lo and hi > 1e-6


def box_shell_occludes(origin, corners, box_origin, *, sealed=False):
    """Prove the target cone cannot intersect the box's sole side opening."""
    camera = np.asarray(origin) - box_origin
    target = np.asarray(corners) - box_origin
    contained = np.all(
        (np.abs(target[:, 0]) < 0.17)
        & (np.abs(target[:, 1]) < 0.20)
        & (target[:, 2] > 0)
        & (target[:, 2] < 0.30)
    )
    outside = (
        camera[0] < (-0.191 if sealed else -0.171)
        or camera[0] > 0.171
        or abs(camera[1]) > 0.201
        or camera[2] < -0.001
        or camera[2] > 0.301
    )
    if not contained or not outside:
        return False
    if sealed:
        return True
    if camera[0] >= -0.169:
        return True
    # Project all bounding-box corners onto the side-opening plane. Perspective
    # projection preserves convexity here. If all projections lie beyond one
    # opening edge, the complete cone misses the opening and hits a closed face.
    t = (-0.17 - camera[0]) / (target[:, 0] - camera[0])
    projected = camera + t[:, None] * (target - camera)
    return bool(
        np.all(projected[:, 1] > 0.081)
        or np.all(projected[:, 1] < -0.185)
        or np.all(projected[:, 2] > 0.285)
        or np.all(projected[:, 2] < 0.015)
    )


def can_visibility(env):
    import mujoco

    from .domain.contracts import CAMERAS, CAN_ID

    model, data = env.sim.model, env.sim.data
    body = env.obj_body_id[CAN_ID]
    center = np.array(data.body_xpos[body])
    rotation = np.array(data.body_xmat[body]).reshape(3, 3)
    # Physical asset reg_bbox encloses every visible can mesh vertex.
    half = np.asarray(env.objects[CAN_ID].size) / 2
    corners = np.array(
        [center + rotation @ (half * np.array(sign)) for sign in product((-1, 1), repeat=3)]
    )
    radius = float(np.linalg.norm(half))
    samples = [
        center,
        *corners,
        *(center + rotation[:, i] * half[i] * s for i in range(3) for s in (-1, 1)),
    ]
    result = {}
    for name in CAMERAS:
        camera = model.camera_name2id(name)
        origin = np.array(data.cam_xpos[camera])
        camera_rotation = np.array(data.cam_xmat[camera]).reshape(3, 3)
        fov = float(model.cam_fovy[camera])
        in_frustum = sphere_in_camera(camera_rotation.T @ (center - origin), radius, fov)
        blocker = None
        if in_frustum:
            for gid in env._occluder_geom_ids:
                if all(
                    segment_intersects_box(
                        origin,
                        point,
                        np.array(data.geom_xpos[gid]),
                        np.array(data.geom_xmat[gid]).reshape(3, 3),
                        model.geom_size[gid],
                    )
                    for point in corners
                ):
                    blocker = model.geom_id2name(gid)
                    break
            if blocker is None:
                # Several joined panels can jointly block the can near a roof/
                # wall seam. If the camera is outside this sealed five-sided
                # shell on its closed side, every segment to the contained can
                # crosses a closed face. It cannot pass through the left opening.
                for label, xy in env.geometry["box_xy"].items():
                    box_origin = np.array([*xy, env.surface_z])
                    if box_shell_occludes(
                        origin, corners, box_origin, sealed=getattr(env, "boxes_sealed", False)
                    ):
                        blocker = f"box_{label}_closed_shell"
                        break
        visible = False
        if in_frustum and blocker is None:
            for point in samples:
                if not sphere_in_camera(camera_rotation.T @ (point - origin), 0, fov):
                    continue
                direction = point - origin
                hit = np.array([-1], dtype=np.int32)
                distance = mujoco.mj_ray(
                    model._model,
                    data._data,
                    origin,
                    direction / np.linalg.norm(direction),
                    np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8),
                    1,
                    -1,
                    hit,
                )
                if distance >= 0 and int(hit[0]) in env._can_visual_geoms:
                    visible = True
                    break
        result[name] = {
            "camera_xyz": origin.tolist(),
            "can_center_xyz": center.tolist(),
            "can_halfsize": half.tolist(),
            "corners_xyz": corners.tolist(),
            "in_frustum": in_frustum,
            "visible": visible,
            "fully_hidden": not in_frustum or blocker is not None,
            "occluding_panel": blocker,
        }
    return result
