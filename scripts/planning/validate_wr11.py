#!/usr/bin/env python3
"""Record a WR-11 expert smoke episode and retain visibility evidence.

--scene-only is a diagnostic with robot reset-pose changes, never a dataset.
The default records the production expert plan through RecordingRunner.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for key, value in {
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
    "MUJOCO_EGL_DEVICE_ID": "0",
    "MEMBENCH_MPLIB_PYTHON": str(ROOT / ".venv-mplib/bin/python"),
    "MEMBENCH_MANISKILL_PANDA_DIR": str(
        ROOT / "outputs/test_env_setup/maniskill/mani_skill/assets/robots/panda"
    ),
    "MPLCONFIGDIR": str(ROOT / "outputs/test_env_setup/cache/matplotlib"),
    "NUMBA_CACHE_DIR": str(ROOT / "outputs/test_env_setup/cache/numba"),
}.items():
    os.environ.setdefault(key, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-only", action="store_true")
    parser.add_argument(
        "--palette-preview", action="store_true", help="Render all eight colors in scene-only mode"
    )
    parser.add_argument(
        "--sampling-boundaries",
        action="store_true",
        help="Check all station disk boundary headings in scene-only mode",
    )
    parser.add_argument(
        "--replay-episode",
        type=Path,
        help="Replay an existing native episode in a fresh policy runtime",
    )
    args = parser.parse_args()
    if args.scene_only and args.replay_episode:
        parser.error("--scene-only and --replay-episode cannot be combined")
    if args.sampling_boundaries and not args.scene_only:
        parser.error("--sampling-boundaries requires --scene-only")
    if args.palette_preview and not args.scene_only:
        parser.error("--palette-preview requires --scene-only")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    if args.replay_episode:
        from scripts.eval.replay_recorded_actions import main as replay_main

        episode_path = args.replay_episode.resolve()
        if not (episode_path / "trajectory.h5").is_file():
            raise FileNotFoundError(episode_path / "trajectory.h5")
        # Index the existing artifact using the native dataset layout; data
        # remain in their original directory and are never copied or modified.
        source = args.output / "source_dataset"
        task_dir = source / "train" / "WR-11"
        task_dir.mkdir(parents=True)
        (task_dir / f"seed_{args.seed:06d}_attempt_000").symlink_to(
            episode_path, target_is_directory=True
        )
        return replay_main(
            [
                "--task-id",
                "WR-11",
                "--dataset-root",
                str(source),
                "--seeds",
                str(args.seed),
                "--output-run",
                str(args.output / "replay"),
                "--record-video",
            ]
        )

    import numpy as np
    from PIL import Image

    from bench.recording.runner import RecordingRunner, build_recording_config
    from bench.runtime import RoboCasaRuntimeFactory
    from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
    from bench.tasks.contracts import EpisodeRequest, ExpertPlanInputs

    episode = TaskCatalog.resolve(
        EpisodeRequest(task_id="WR-11", seed=args.seed, allow_partial_calibration=False)
    )
    api = resolve_builtin_task_plugin("WR-11").load_runtime_api()
    runtime = RoboCasaRuntimeFactory(api).create(episode, expert=not args.scene_only)
    evidence = {}

    def snapshot(label):
        obs = runtime.observe()
        env = runtime.env
        value = asdict(env.evaluation_snapshot())
        evidence[label] = value
        for camera in env.camera_names:
            Image.fromarray(np.asarray(obs[camera + "_image"])[::-1]).save(
                args.output / f"{label}_{camera}.png"
            )
        return value

    try:
        snapshot("initial_sample")
        if args.scene_only:
            import mujoco

            from bench.execution.navigation.base_motion import set_base_pose_kinematic

            if args.palette_preview:
                from PIL import ImageDraw

                from bench.tasks.wr11.domain.contracts import PACKAGING_RGB, SAMPLE_ID

                tiles = []
                for color, rgb in PACKAGING_RGB.items():
                    runtime.env._apply_carton_material(SAMPLE_ID, color)
                    snapshot(f"palette_{color}")
                    close_camera = mujoco.MjvCamera()
                    close_camera.lookat[:] = runtime.env._obj_pos(SAMPLE_ID)
                    close_camera.distance = 0.45
                    close_camera.azimuth = 90.0
                    close_camera.elevation = -15.0
                    option = mujoco.MjvOption()
                    option.geomgroup[:] = (0, 1, 0, 0, 0, 0)
                    # A new renderer uploads the current recolored texture.
                    with mujoco.Renderer(
                        runtime.env.sim.model._model, height=320, width=320
                    ) as renderer:
                        renderer.update_scene(
                            runtime.env.sim.data._data, camera=close_camera, scene_option=option
                        )
                        tile = Image.fromarray(renderer.render())
                    tile.save(args.output / f"palette_closeup_{color}.png")
                    label = Image.new("RGB", (320, 355), "white")
                    label.paste(tile, (0, 0))
                    ImageDraw.Draw(label).text((12, 329), f"{color}  RGB {rgb}", fill="black")
                    tiles.append(label)
                montage = Image.new("RGB", (1280, 710), "white")
                for index, tile in enumerate(tiles):
                    montage.paste(tile, ((index % 4) * 320, (index // 4) * 355))
                montage.save(args.output / "palette_comparison.png")
                runtime.env._apply_carton_material(SAMPLE_ID, runtime.env.sample_packaging)

            # Read-only free camera for inspecting the complete room geometry.
            camera = mujoco.MjvCamera()
            camera.lookat[:] = (3.1, -1.4, 0.7)
            camera.distance = 8.0
            camera.azimuth = 90.0
            camera.elevation = -65.0
            option = mujoco.MjvOption()
            option.geomgroup[:] = (0, 1, 0, 0, 0, 0)
            with mujoco.Renderer(runtime.env.sim.model._model, height=480, width=640) as renderer:
                renderer.update_scene(
                    runtime.env.sim.data._data, camera=camera, scene_option=option
                )
                Image.fromarray(renderer.render()).save(args.output / "room_overview.png")
            stations = json.loads(runtime.initialized.descriptor.resources["stations"].read_text())
            stock = stations["stations"]["stock_station"]["circle"]
            sampled_stock = episode.facts["base_xy_by_station"]["stock_station"]
            set_base_pose_kinematic(
                runtime.env, target_xy=np.array(sampled_stock), target_yaw=stock["base_yaw"]
            )
            snapshot("diagnostic_stock_pose")
            passed = (
                evidence["initial_sample"]["sample_visible"]
                and not evidence["diagnostic_stock_pose"]["sample_in_any_camera"]
            )
            if args.sampling_boundaries:
                from bench.tasks.wr11.domain.contracts import ITEM_IDS, SAMPLE_ID

                checks = []
                for name, station in stations["stations"].items():
                    circle = station["circle"]
                    center = np.array(circle["center_xy"])
                    for index, angle in enumerate(np.linspace(0, 2 * np.pi, 8, endpoint=False)):
                        xy = center + circle["radius_m"] * np.array([np.cos(angle), np.sin(angle)])
                        set_base_pose_kinematic(
                            runtime.env, target_xy=xy, target_yaw=circle["base_yaw"]
                        )
                        value = snapshot(f"boundary_{name}_{index}")
                        visibility_pass = (
                            value["sample_visible"]
                            if name == "sample_station"
                            else not value["sample_in_any_camera"]
                        )
                        # Keep all reset cartons in real contact with their own trays.
                        from bench.tasks.wr11.contact_evidence import ContactIndex

                        contacts = ContactIndex(runtime.env.sim, runtime.env._geom_ids)
                        supports = {
                            item: contacts.check(
                                runtime.env.objects[item],
                                runtime.env.objects[
                                    "sample_tray"
                                    if item == SAMPLE_ID
                                    else runtime.env.stock_tray_name(item)
                                ],
                            )
                            for item in (*ITEM_IDS, SAMPLE_ID)
                        }
                        from bench.tasks.wr11.visibility import object_camera_visibility

                        stock_visible = (
                            {
                                item: any(
                                    camera["visible"]
                                    for camera in object_camera_visibility(
                                        runtime.env,
                                        runtime.env._object_geoms[item],
                                        runtime.env.camera_names,
                                    ).values()
                                )
                                for item in ITEM_IDS
                            }
                            if name == "stock_station"
                            else {}
                        )
                        checks.append(
                            {
                                "station": name,
                                "angle_rad": float(angle),
                                "base_xy": xy.tolist(),
                                "visibility_pass": visibility_pass,
                                "supports": supports,
                                "stock_visible": stock_visible,
                            }
                        )
                (args.output / "sampling_boundaries.json").write_text(
                    json.dumps(checks, indent=2) + "\n"
                )
                passed = passed and all(
                    check["visibility_pass"]
                    and all(check["supports"].values())
                    and all(check["stock_visible"].values())
                    for check in checks
                )
        else:
            descriptor = runtime.initialized.descriptor
            plan = TaskCatalog.get("WR-11").build_expert_plan(
                ExpertPlanInputs.from_descriptor(descriptor)
            )
            runner = runtime.initialized.runner

            class ProgressRunner:
                def run(self, plan, *, on_stage_end):
                    def report(index, stage, result):
                        on_stage_end(index, stage, result)
                        snapshot(f"stage_{index:02d}_{stage.name}")
                        print(
                            json.dumps(
                                {
                                    "stage": index,
                                    "name": stage.name,
                                    "success": result.success,
                                    "failure_code": result.failure_code,
                                    "stage_checks": runtime.env.evaluator.to_diagnostics()[
                                        "stage_checks"
                                    ],
                                }
                            ),
                            flush=True,
                        )

                    return runner.run(plan, on_stage_end=report)

            config = build_recording_config(
                runtime, output_dir=args.output / "episode", record_depth=False
            )
            result, _ = RecordingRunner(
                runtime=runtime, runner=ProgressRunner(), config=config
            ).run(plan)
            passed = result.success
        report = {
            "seed": args.seed,
            "passed": passed,
            "diagnostic_only": args.scene_only,
            "episode_identity": episode.identity,
            "facts": dict(episode.facts),
            "visibility_evidence": evidence,
            "diagnostics": runtime.env.task_diagnostics(),
            "report_stages": api.report_stages(runtime.initialized),
        }
        (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"passed": passed, "output": str(args.output)}), flush=True)
        return 0 if passed else 1
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
