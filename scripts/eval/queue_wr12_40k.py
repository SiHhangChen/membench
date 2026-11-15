#!/usr/bin/env python3
"""Validate both WR12 checkpoints, then evaluate sequentially on an idle GPU."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def prepare() -> list[dict]:
    from bench.evaluation.suite import load_eval_suite
    from bench.policy.backends.lerobot import (
        inspect_lerobot_checkpoint, inspect_lerobot_diffusion_checkpoint,
    )
    from bench.tasks.wr12.evaluation import REPORT_STAGE_ORDER, REPORT_STAGE_PROFILE_ID

    dataset = ROOT / "data/final/wr12-final"
    features = json.loads((dataset / "meta/info.json").read_text())["features"]
    cameras = {alias: f"observation.images.robot0_{alias}" for alias in
               ("agentview_left", "agentview_right", "eye_in_hand")}
    for name, shape, schema in (
        ("observation.state", [37], "membench.robot_state37.v2"),
        ("action", [13], "membench.pandaomron_hybrid13.v2"),
    ):
        if features[name]["shape"] != shape or features[name]["info"]["schema"] != schema:
            raise ValueError(f"Invalid dataset contract: {name}")
    jobs = []
    for policy in ("act", "diffusion"):
        checkpoint = ROOT / f"outputs/train/{policy}_wr12_40k_v1/wr12/checkpoints/040000/pretrained_model"
        config = json.loads((checkpoint / "config.json").read_text())
        if config["type"] != policy:
            raise ValueError(f"Wrong checkpoint type: {checkpoint}")
        if config["input_features"]["observation.state"]["shape"] != [37] or config["output_features"]["action"]["shape"] != [13]:
            raise ValueError(f"Invalid checkpoint dimensions: {checkpoint}")
        for key in cameras.values():
            if features[key]["shape"] != [224, 224, 3] or config["input_features"][key]["shape"] != [3, 224, 224]:
                raise ValueError(f"Invalid camera contract: {key}")
        metadata = {
            "protocol_version": "membench-policy/v2",
            "runtime_profile": "membench.robocasa_pandaomron.v2",
            "policy_type": policy, "state_schema": "membench.robot_state37.v2",
            "state_dim": 37, "action_schema": "membench.pandaomron_hybrid13.v2",
            "action_dim": 13, "training_step": 40000, "dataset_root": str(dataset),
            "camera_mapping": cameras, "image_size": [224, 224],
            "contract_validation": "dataset_and_checkpoint_metadata",
        }
        save(checkpoint / "membench_policy.json", metadata)
        inspect = inspect_lerobot_checkpoint if policy == "act" else inspect_lerobot_diffusion_checkpoint
        contract = inspect(checkpoint)
        suite_path = ROOT / ("configs/eval/suites/wr12_v1.yaml" if policy == "act"
                             else "configs/eval/diffusion/wr12_r1_v1.yaml")
        suite = load_eval_suite(suite_path)
        jobs.append({"policy": policy, "checkpoint": str(checkpoint), "contract": contract,
                     "suite": str(suite_path), "suite_hash": suite.content_hash,
                     "seeds": list(suite.seeds), "max_steps": suite.rollout.max_steps,
                     "replan_steps": suite.rollout.replan_steps,
                     "report_profile": REPORT_STAGE_PROFILE_ID,
                     "report_stages": list(REPORT_STAGE_ORDER), "state": "pending"})
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.queue_dir.mkdir(parents=True, exist_ok=True)
    # A single WR12 queue may run at a time, even with a different queue directory.
    lock = (ROOT / "outputs/eval/logs/wr12-40k-queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = args.queue_dir / "status.json"
    if status_path.exists():
        raise FileExistsError(f"Queue already exists: {status_path}; inspect it before restarting")
    status = {"pid": os.getpid(), "created_at": now(), "jobs": prepare()}
    for job in status["jobs"]:
        job["run_id"] = f"{job['policy']}-wr12-40000-stages-v2-{args.queue_dir.name}"
    save(args.queue_dir / "preflight.json", status)
    if args.prepare_only:
        print(json.dumps(status, indent=2))
        return
    child = None

    def stop(signum, _frame):
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            child.wait(timeout=60)
        status.update(state="stopped", updated_at=now())
        save(status_path, status)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for index, job in enumerate(status["jobs"]):
        while True:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=20,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                status.update(state="waiting_for_gpu", updated_at=now(), gpu_query_error=str(exc))
                save(status_path, status)
                time.sleep(30)
                continue
            available = []
            if result.returncode == 0:
                for row in result.stdout.strip().splitlines():
                    gpu, free, utilization = (int(value.strip()) for value in row.split(","))
                    if free >= 16000 and utilization <= 10:
                        available.append((free, gpu))
            status.update(state="waiting_for_gpu", updated_at=now(), gpu_query=result.stdout,
                          gpu_query_error=result.stderr if result.returncode else None)
            save(status_path, status)
            if available:
                gpu = max(available)[1]
                break
            time.sleep(30)
        job.update(state="running", gpu=gpu, port=57152 + index, started_at=now())
        status.update(state="running", updated_at=now())
        save(status_path, status)
        print(f"Starting {job['run_id']} GPU={gpu}", flush=True)
        with (args.queue_dir / f"{job['policy']}.log").open("a") as log:
            child = subprocess.Popen(
                ["bash", "scripts/eval/eval_wr12_40k.sh", job["policy"], str(gpu), str(job["port"]), job["run_id"]],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            )
            job["pid"] = child.pid
            save(status_path, status)
            returncode = child.wait()
        job.update(state="finished" if returncode == 0 else "failed", returncode=returncode, finished_at=now())
        save(status_path, status)
        child = None
    status.update(state="finished" if all(job["returncode"] == 0 for job in status["jobs"]) else "failed",
                  updated_at=now())
    save(status_path, status)


if __name__ == "__main__":
    main()
