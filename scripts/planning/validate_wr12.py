#!/usr/bin/env python3
"""Record a WR-12 expert smoke episode and retain visibility evidence.

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
    parser.add_argument("--sampling-boundaries", action="store_true")
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
        task_dir = source / "train" / "WR-12"
        task_dir.mkdir(parents=True)
        (task_dir / f"seed_{args.seed:06d}_attempt_000").symlink_to(
            episode_path, target_is_directory=True
        )
        return replay_main(
            [
                "--task-id",
                "WR-12",
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
        EpisodeRequest(task_id="WR-12", seed=args.seed, allow_partial_calibration=False)
    )
    api = resolve_builtin_task_plugin("WR-12").load_runtime_api()
    runtime = RoboCasaRuntimeFactory(api).create(episode, expert=not args.scene_only)
    evidence = {}

    def snapshot(label):
        obs = runtime.observe()
        env = runtime.env
        value = asdict(env.evaluation_snapshot())
        if args.scene_only:
            import mujoco

            pixels = {}
            for camera in env.camera_names:
                segmentation = env.sim.render(
                    camera_name=camera, width=224, height=224, segmentation=True
                )[::-1]
                mask = (segmentation[..., 0] == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(
                    segmentation[..., 1], list(env._can_visual_geoms)
                )
                raw_count = int(mask.sum())
                if env.boxes_sealed:
                    # Multisampled ID colors can decode an edge blend as an
                    # unrelated geom ID. Confirm every candidate pixel using
                    # its actual camera ray before calling it a visible can.
                    cid = env.sim.model.camera_name2id(camera)
                    rotation = env.sim.data.cam_xmat[cid].reshape(3, 3)
                    origin = np.array(env.sim.data.cam_xpos[cid])
                    focal = 112 / np.tan(np.deg2rad(env.sim.model.cam_fovy[cid]) / 2)
                    for yy, xx in zip(*np.nonzero(mask), strict=True):
                        direction = rotation @ np.array(
                            [(xx + 0.5 - 112) / focal, -(yy + 0.5 - 112) / focal, -1]
                        )
                        hit = np.array([-1], dtype=np.int32)
                        mujoco.mj_ray(
                            env.sim.model._model,
                            env.sim.data._data,
                            origin,
                            direction / np.linalg.norm(direction),
                            np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8),
                            1,
                            -1,
                            hit,
                        )
                        mask[yy, xx] = int(hit[0]) in env._can_visual_geoms
                yy, xx = np.nonzero(mask)
                # Diagnostic-only render of the same can with all occluders
                # excluded. Restore geom groups before any RGB or physics step.
                groups = env.sim.model.geom_group.copy()
                try:
                    env.sim.model.geom_group[:] = 0
                    env.sim.model.geom_group[list(env._can_visual_geoms)] = 1
                    isolated = env.sim.render(
                        camera_name=camera, width=224, height=224, segmentation=True
                    )[::-1]
                    full_count = int(
                        np.sum(
                            (isolated[..., 0] == mujoco.mjtObj.mjOBJ_GEOM)
                            & np.isin(isolated[..., 1], list(env._can_visual_geoms))
                        )
                    )
                finally:
                    env.sim.model.geom_group[:] = groups
                pixels[camera] = {
                    "count": len(xx),
                    "raw_segmentation_count": raw_count,
                    "unoccluded_count": full_count,
                    "visible_fraction": len(xx) / max(1, full_count),
                    "width": int(xx.max() - xx.min() + 1) if len(xx) else 0,
                    "height": int(yy.max() - yy.min() + 1) if len(xx) else 0,
                }
            value["visible_can_pixels"] = pixels
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

            from bench.execution.manipulation.drivers import run_gripper_command
            from bench.execution.navigation.base_motion import (
                qpos_for_controller,
                set_base_pose_kinematic,
            )

            env = runtime.env
            camera = mujoco.MjvCamera()
            camera.lookat[:] = (3.2, -2.2, 1.0)
            camera.distance = 2.5
            camera.azimuth = 90.0
            camera.elevation = -35.0
            option = mujoco.MjvOption()
            option.geomgroup[:] = (0, 1, 0, 0, 0, 0)
            with mujoco.Renderer(env.sim.model._model, height=480, width=640) as renderer:
                renderer.update_scene(env.sim.data._data, camera=camera, scene_option=option)
                Image.fromarray(renderer.render()).save(args.output / "scene_overview.png")
            plan = runtime.initialized.descriptor.episode_plan
            passed = evidence["initial_sample"]["can_visible"]
            checks = []

            def boundary_checks(name, circle, *, sealed):
                half_range = circle["yaw_half_range_rad"]
                for index, angle in enumerate(np.linspace(0, 2 * np.pi, 8, endpoint=False)):
                    xy = np.array(circle["center_xy"]) + circle["radius_m"] * np.array(
                        [np.cos(angle), np.sin(angle)]
                    )
                    for yaw_index, yaw_offset in enumerate((-half_range, 0, half_range)):
                        yaw = circle["base_yaw"] + yaw_offset
                        set_base_pose_kinematic(env, target_xy=xy, target_yaw=yaw)
                        prefix = "sealed" if sealed else "open"
                        value = snapshot(f"{prefix}_boundary_{name}_{index}_yaw{yaw_index}")
                        if sealed:
                            ok = (
                                value["boxes_sealed"]
                                and value["can_hidden"]
                                and value["can_occluded_by_box"]
                                and all(
                                    p["count"] == 0 for p in value["visible_can_pixels"].values()
                                )
                            )
                        else:
                            ok = value["can_visible"] and any(
                                pixels["count"] >= 200
                                and pixels["visible_fraction"] >= 0.65
                                and pixels["width"] >= 10
                                and pixels["height"] >= 18
                                for camera, pixels in value["visible_can_pixels"].items()
                                if "agentview" in camera
                            )
                        checks.append(
                            {
                                "station": name,
                                "boxes_sealed": sealed,
                                "index": index,
                                "yaw_rad": yaw,
                                "base_xy": xy.tolist(),
                                "visible_can_pixels": value["visible_can_pixels"],
                                "passed": ok,
                            }
                        )

            if args.sampling_boundaries:
                from bench.tasks.wr12.domain.episode import load_station_circles

                circles = load_station_circles()
                boundary_checks("observe_station", circles["observe_station"], sealed=False)
            set_base_pose_kinematic(
                env,
                target_xy=np.array(plan.base_xy_by_station["observe_station"]),
                target_yaw=plan.base_yaw_by_station["observe_station"],
            )
            arm = env.robots[0].composite_controller.part_controllers["right"]
            hold = qpos_for_controller(env, arm)
            timeline = []
            for step in range(1, 29):
                run_gripper_command(env, arm_qpos=hold, gripper_action=-1.0, steps=1)
                progress = env._shutter_progress
                ok = (
                    progress == 0
                    if step <= 20
                    else 0 < progress < 1
                    if step < 26
                    else env.boxes_sealed
                )
                passed = passed and ok
                timeline.append(
                    {"step": step, "progress": progress, "sealed": env.boxes_sealed, "passed": ok}
                )
                if step in (19, 20, 23, 26, 28):
                    snapshot(f"timed_step_{step:02d}")
            (args.output / "shutter_timeline.json").write_text(
                json.dumps(timeline, indent=2) + "\n"
            )
            sealed_side = snapshot("diagnostic_sealed_observe_pose")
            passed = passed and sealed_side["boxes_sealed"] and sealed_side["can_hidden"]
            set_base_pose_kinematic(
                env,
                target_xy=np.array(plan.base_xy_by_station["operate_station"]),
                target_yaw=plan.base_yaw_by_station["operate_station"],
            )
            operation = snapshot("diagnostic_operate_pose")
            passed = passed and operation["can_hidden"] and operation["boxes_sealed"]
            if args.sampling_boundaries:
                for name, circle in load_station_circles().items():
                    boundary_checks(name, circle, sealed=True)
                passed = passed and all(c["passed"] for c in checks)
                (args.output / "sampling_boundaries.json").write_text(
                    json.dumps(checks, indent=2) + "\n"
                )
        else:
            descriptor = runtime.initialized.descriptor
            plan = TaskCatalog.get("WR-12").build_expert_plan(
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
