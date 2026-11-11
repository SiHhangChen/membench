"""Read-only visibility checks; leaving the camera frustum hides the sample.

Offscreen is conservative: every visible geom's bounding sphere must be
outside every policy camera. Observation additionally requires an unobstructed
ray from a camera to a sample geom. No render buffers or world state are changed.
"""

import numpy as np


def sphere_in_camera(center_camera, radius, *, vertical_fov_deg, aspect=1.0):
    center = np.asarray(center_camera, dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("camera point must have finite XYZ")
    if not np.isfinite(radius) or radius < 0 or not 0 < vertical_fov_deg < 180 or aspect <= 0:
        raise ValueError("invalid camera sphere or frustum")
    x, y, z = center
    depth = -z
    ty = np.tan(np.deg2rad(vertical_fov_deg) / 2)
    tx = ty * aspect
    return bool(
        depth + radius > 0
        and abs(x) <= depth * tx + radius * np.sqrt(1 + tx * tx)
        and abs(y) <= depth * ty + radius * np.sqrt(1 + ty * ty)
    )


def object_camera_visibility(env, geom_ids, camera_names, *, aspect=1.0):
    import mujoco

    model, data = env.sim.model, env.sim.data
    raw_model, raw_data = model._model, data._data
    visible_geoms = [
        gid for gid in geom_ids if model.geom_rgba[gid, 3] > 0 and int(model.geom_group[gid]) == 1
    ]
    if not visible_geoms:
        raise ValueError("sample has no visible geometry")
    result = {}
    for name in camera_names:
        camera = model.camera_name2id(name)
        rotation = np.asarray(data.cam_xmat[camera]).reshape(3, 3)
        origin = np.asarray(data.cam_xpos[camera])
        in_frustum = False
        line_of_sight = False
        for gid in visible_geoms:
            center = np.asarray(data.geom_xpos[gid])
            local = rotation.T @ (center - origin)
            if not sphere_in_camera(
                local,
                float(model.geom_rbound[gid]),
                vertical_fov_deg=float(model.cam_fovy[camera]),
                aspect=aspect,
            ):
                continue
            in_frustum = True
            if not sphere_in_camera(
                local, 0, vertical_fov_deg=float(model.cam_fovy[camera]), aspect=aspect
            ):
                continue
            direction = center - origin
            length = np.linalg.norm(direction)
            if length <= 1e-9:
                continue
            hit = np.array([-1], dtype=np.int32)
            # Group 1 is the visual geometry used by the policy cameras.
            distance = mujoco.mj_ray(
                raw_model,
                raw_data,
                origin,
                direction / length,
                np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8),
                1,
                -1,
                hit,
            )
            if distance >= 0 and int(hit[0]) in geom_ids:
                line_of_sight = True
        result[name] = {"in_frustum": in_frustum, "visible": line_of_sight}
    return result
