#!/usr/bin/env python3
"""Render the current WR-07 / WR-08 counter-zone scene."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.runtime import RoboCasaRuntimeFactory  # noqa: E402
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin  # noqa: E402
from bench.tasks.contracts import EpisodeRequest  # noqa: E402

CAMERAS = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)


def _render_free_camera(
    env: Any,
    output_dir: Path,
    *,
    label: str,
    lookat: list[float],
    distance: float,
    azimuth: float,
    elevation: float,
    width: int,
    height: int,
) -> Path:
    import mujoco

    renderer = mujoco.Renderer(env.sim.model._model, width=width, height=height)
    try:
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = lookat
        camera.distance = distance
        camera.azimuth = azimuth
        camera.elevation = elevation
        options = mujoco.MjvOption()
        options.geomgroup[0] = int(getattr(env, "render_collision_mesh", False))
        options.geomgroup[1] = int(getattr(env, "render_visual_mesh", True))
        renderer.update_scene(env.sim.data._data, camera=camera, scene_option=options)
        rgb = renderer.render().copy()
    finally:
        renderer.close()
    path = output_dir / f"{label}.png"
    imageio.imwrite(path, rgb)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("WR-07", "WR-08"), default="WR-07")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    episode = TaskCatalog.resolve(EpisodeRequest(task_id=args.task, seed=args.seed))
    api = resolve_builtin_task_plugin(args.task).load_runtime_api()
    runtime = RoboCasaRuntimeFactory(api).create(episode, execute=False, expert=True)
    try:
        env = runtime.env
        env.sim.forward()
        args.output_dir.mkdir(parents=True, exist_ok=True)

        frames: list[Image.Image] = []
        camera_paths: dict[str, str] = {}
        for camera in CAMERAS:
            rgb = np.asarray(
                env.sim.render(
                    camera_name=camera,
                    width=args.width,
                    height=args.height,
                    depth=False,
                ),
                dtype=np.uint8,
            )[::-1]
            path = args.output_dir / f"{args.task.lower().replace('-', '')}_{camera}.png"
            imageio.imwrite(path, rgb)
            camera_paths[camera] = str(path.resolve())
            panel = Image.fromarray(rgb)
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 0, args.width, 36), fill=(20, 20, 20))
            draw.text((12, 10), camera, fill=(255, 255, 255))
            frames.append(panel)

        collage = Image.new("RGB", (args.width * len(frames), args.height))
        for index, frame in enumerate(frames):
            collage.paste(frame, (index * args.width, 0))
        collage_path = args.output_dir / f"{args.task.lower().replace('-', '')}_policy_views.png"
        collage.save(collage_path)

        object_names = ["counter_zone_left", "counter_zone_center", "counter_zone_right"]
        object_names.extend(getattr(env, "item_names", ()))
        object_names.append("plate")
        object_poses = {
            name: np.asarray(env._obj_pos(name), dtype=float).tolist()
            for name in object_names
            if name in getattr(env, "obj_body_id", {})
        }
        free_path = _render_free_camera(
            env,
            args.output_dir,
            label=f"{args.task.lower().replace('-', '')}_counter_overview",
            lookat=[2.15, -0.55, 1.0],
            distance=2.25,
            azimuth=90.0,
            elevation=-32.0,
            width=args.width,
            height=args.height,
        )
        print(
            json.dumps(
                {
                    "task": args.task,
                    "seed": args.seed,
                    "collage": str(collage_path.resolve()),
                    "policy_cameras": camera_paths,
                    "counter_overview": str(free_path.resolve()),
                    "object_poses_world": object_poses,
                    "episode_facts": dict(episode.facts),
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
