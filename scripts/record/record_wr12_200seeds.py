#!/usr/bin/env python3
"""Record WR-12 v1.4 with isolated workers and incremental manifests."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def task_source_hash():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "bench/tasks/wr12").rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def worker(run_dir, worker_index):
    from bench.recording.plan import RecordingWorkItem
    from bench.tasks import TaskCatalog
    from bench.tasks.contracts import EpisodeRequest
    from scripts.record_entrypoint import _record_item, parse_args

    plan = json.loads((run_dir / "run_plan.json").read_text())
    spec = plan["workers"][worker_index]
    dataset_root = Path(plan["output_root"]) / spec["dataset_id"]
    dataset_root.mkdir(parents=True, exist_ok=False)
    args = parse_args(
        [
            "--task-id",
            "WR-12",
            "--dataset-id",
            spec["dataset_id"],
            "--output-root",
            plan["output_root"],
            "--variant",
            "memory",
            "--split",
            "train",
            "--env-split",
            "pretrain",
            "--fps",
            "20",
            "--camera",
            "robot0_agentview_left",
            "--camera",
            "robot0_agentview_right",
            "--camera",
            "robot0_eye_in_hand",
            "--no-record-depth",
        ]
    )
    items = []
    started = time.time()
    for seed in spec["seeds"]:
        item = RecordingWorkItem("WR-12", seed, "memory", 0)
        print(
            json.dumps({"event": "seed_started", "seed": seed, "worker": worker_index}), flush=True
        )
        # Stop this shard if the task is edited during recording. Other seeds
        # may fail physically or raise an error without changing the task.
        if task_source_hash() != plan["task_source_sha256"]:
            raise RuntimeError("WR-12 source changed during recording; refusing mixed versions")
        episode = TaskCatalog.resolve(
            EpisodeRequest(
                task_id="WR-12", seed=seed, split="train", allow_partial_calibration=False
            )
        )
        if episode.identity != plan["episode_identities"][str(seed)]:
            raise RuntimeError(f"WR-12 seed {seed} identity changed during recording")
        seed_started = time.time()
        try:
            summary = _record_item(item, args=args, dataset_root=dataset_root)
            episode_dir = Path(summary["episode_dir"])
            metadata = json.loads((episode_dir / "episode.json").read_text())
            planner = json.loads((episode_dir / "planner.json").read_text())
            if summary["episode_identity"] != episode.identity:
                raise RuntimeError("Recorded episode identity differs from the run plan")
            summary.update(
                {
                    "status": "success" if summary["success"] else "failure",
                    "failure_reason": None
                    if summary["success"]
                    else metadata["termination"]["reason"],
                    "step_count": metadata["step_count"],
                    "stage_steps": planner["final_diagnostics"].get("stage_steps", {}),
                }
            )
        except Exception as exc:  # noqa: BLE001 -- retain the failed seed and continue the shard
            traceback.print_exc()
            summary = {
                "work_id": item.work_id,
                "task_id": "WR-12",
                "seed": seed,
                "variant": "memory",
                "attempt": 0,
                "episode_identity": episode.identity,
                "success": False,
                "status": "error",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "episode_dir": str(dataset_root / "_staging/WR-12" / f"seed_{seed:06d}_attempt_00"),
            }
        summary["elapsed_s"] = round(time.time() - seed_started, 3)
        items.append(summary)
        write_json(
            dataset_root / "manifest.json",
            {
                "schema_version": "membench.dataset/v2",
                "dataset_id": spec["dataset_id"],
                "task_version": "1.4.0",
                "items": items,
            },
        )
        write_json(
            run_dir / f"part{worker_index:02d}_progress.json",
            {
                "worker": worker_index,
                "planned": len(spec["seeds"]),
                "finished": len(items),
                "counts": dict(Counter(value["status"] for value in items)),
                "latest_seed": seed,
                "elapsed_s": round(time.time() - started, 3),
            },
        )
        print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="wr12_200seeds_v14")
    parser.add_argument("--render-gpus", default="4,5,6,7,4,5,6,7,4,5,6,7,4,5,6,7")
    parser.add_argument(
        "--resume-from", type=Path, help="Preserve a stopped run and record its remaining seeds"
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/datasets")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--seeds", default="0-199")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.worker_index is not None:
        return worker(args.run_dir, args.worker_index)
    if not args.run_id or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in args.run_id
    ):
        parser.error("run-id must contain only letters, digits, underscores and hyphens")
    gpus = [int(value) for value in args.render_gpus.split(",")]
    if not 1 <= len(gpus) <= 32 or min(gpus) < 0:
        parser.error("render-gpus must list 1–32 nonnegative EGL device IDs")
    from bench.tasks import TaskCatalog
    from bench.tasks.contracts import EpisodeRequest
    from scripts.record_entrypoint import _seeds

    seeds = _seeds(args.seeds)
    if not seeds or min(seeds) < 0:
        parser.error("seeds must be nonnegative and nonempty")
    episodes = [
        TaskCatalog.resolve(
            EpisodeRequest(
                task_id="WR-12", seed=seed, split="train", allow_partial_calibration=False
            )
        )
        for seed in seeds
    ]
    assert all(
        ep.task_version == "1.4.0" and ep.facts["sampling_profile_id"] == "WR-12/sampling-v2"
        for ep in episodes
    )
    episodes_by_seed = {ep.seed: ep for ep in episodes}
    inherited = []
    if args.resume_from:
        old = json.loads((args.resume_from / "run_plan.json").read_text())
        if not set(old["seeds"]).issubset(seeds):
            raise ValueError("Previous seed set is not a subset of the requested seeds")
        if old["task_source_sha256"] != task_source_hash():
            raise RuntimeError("The previous run used a different WR-12 source")
        inherited.extend(old.get("inherited_items", []))
        for spec in old["workers"]:
            manifest = Path(old["output_root"]) / spec["dataset_id"] / "manifest.json"
            if manifest.exists():
                inherited.extend(json.loads(manifest.read_text())["items"])
        completed = {item["seed"] for item in inherited}
        if len(completed) != len(inherited):
            raise ValueError("Previous recording contains duplicate seeds")
        for item in inherited:
            if item["episode_identity"] != episodes_by_seed[item["seed"]].identity:
                raise ValueError(f"Previous seed {item['seed']} has a different identity")
    else:
        completed = set()
    remaining = [seed for seed in seeds if seed not in completed]
    workers = [
        {
            "index": index,
            "render_gpu": gpu,
            "dataset_id": f"{args.run_id}_part{index:02d}",
            "seeds": remaining[index :: len(gpus)],
        }
        for index, gpu in enumerate(gpus)
    ]
    plan = {
        "run_id": args.run_id,
        "task_id": "WR-12",
        "task_version": "1.4.0",
        "sampling_profile_id": "WR-12/sampling-v2",
        "target_counts": dict(Counter(ep.facts["target_box"] for ep in episodes)),
        "seeds": seeds,
        "attempts_per_seed": 1,
        "workers": workers,
        "inherited_items": inherited,
        "resume_from": str(args.resume_from.resolve()) if args.resume_from else None,
        "output_root": str(args.output_root.resolve()),
        "episode_identities": {str(ep.seed): ep.identity for ep in episodes},
        "task_source_sha256": task_source_hash(),
        "started_at_unix": time.time(),
        "controller_stages": 7,
        "recording": {
            "cameras": ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
            "width": 224,
            "height": 224,
            "fps": 20,
            "record_depth": False,
        },
    }
    if args.dry_run:
        print(
            json.dumps(
                {key: value for key, value in plan.items() if key != "episode_identities"}, indent=2
            )
        )
        return 0
    run_dir = args.run_dir or ROOT / "outputs/record" / args.run_id
    if run_dir.exists() or any(
        (args.output_root / spec["dataset_id"]).exists() for spec in workers
    ):
        raise FileExistsError("Run directory or dataset shard already exists; use a new run ID")
    run_dir.mkdir(parents=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "run_plan.json", plan)
    shutil.copytree(
        ROOT / "bench/tasks/wr12",
        run_dir / "source_snapshot/bench/tasks/wr12",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    snapshot_scripts = run_dir / "source_snapshot/scripts"
    snapshot_scripts.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/record_entrypoint.py", snapshot_scripts)
    shutil.copy2(Path(__file__), snapshot_scripts)
    shutil.copy2(ROOT / "scripts/record/audit_wr12_native.py", snapshot_scripts)
    write_json(run_dir / "supervisor.json", {"pid": os.getpid(), "run_id": args.run_id})
    processes = []
    for spec in workers:
        env = dict(
            os.environ,
            MUJOCO_GL="egl",
            PYOPENGL_PLATFORM="egl",
            MUJOCO_EGL_DEVICE_ID=str(spec["render_gpu"]),
            PYTHONUNBUFFERED="1",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            MEMBENCH_MPLIB_PYTHON=str(ROOT / ".venv-mplib/bin/python"),
            MEMBENCH_MANISKILL_PANDA_DIR=str(
                ROOT / "outputs/test_env_setup/maniskill/mani_skill/assets/robots/panda"
            ),
            MPLCONFIGDIR=str(ROOT / "outputs/test_env_setup/cache/matplotlib"),
            NUMBA_CACHE_DIR=str(ROOT / "outputs/test_env_setup/cache/numba"),
        )
        with (run_dir / f"part{spec['index']:02d}.log").open("x") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "--run-dir",
                    str(run_dir),
                    "--worker-index",
                    str(spec["index"]),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        (run_dir / f"part{spec['index']:02d}.pid").write_text(str(process.pid) + "\n")
        processes.append(process)
        print(
            f"Worker {spec['index']}: EGL {spec['render_gpu']}, {len(spec['seeds'])} seeds, PID {process.pid}",
            flush=True,
        )
    codes = [process.wait() for process in processes]
    items = list(inherited)
    for spec in workers:
        path = args.output_root / spec["dataset_id"] / "manifest.json"
        if path.exists():
            items.extend(json.loads(path.read_text())["items"])
    items.sort(key=lambda value: value["seed"])
    write_json(
        run_dir / "manifest.json",
        {
            "schema_version": "membench.dataset/v2",
            "dataset_id": args.run_id,
            "task_version": "1.4.0",
            "items": items,
        },
    )
    recorded = {value["seed"] for value in items}
    summary = {
        "run_id": args.run_id,
        "planned": len(seeds),
        "finished": len(items),
        "counts": dict(Counter(value["status"] for value in items)),
        "missing_seeds": sorted(set(seeds) - recorded),
        "failed_seeds": [value["seed"] for value in items if not value["success"]],
        "worker_exit_codes": codes,
        "elapsed_s": round(time.time() - plan["started_at_unix"], 3),
    }
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    passed = (
        len(items) == len(seeds)
        and recorded == set(seeds)
        and not any(codes)
        and all(item["success"] for item in items)
    )
    if passed:
        from scripts.record.audit_wr12_native import audit_run

        audit_run(run_dir)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
