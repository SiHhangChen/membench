"""Verify WR-12 native recordings and index successful episodes for conversion."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench.tasks.wr12.domain.contracts import CAMERAS
from bench.tasks.wr12.prompts import PROMPT


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_run(run_dir):
    from scripts.record.record_wr12_200seeds import task_source_hash, write_json

    run_dir = Path(run_dir).resolve()
    plan = json.loads((run_dir / "run_plan.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    items = manifest["items"]
    require(task_source_hash() == plan["task_source_sha256"], "Task source changed")
    require(len(items) == len(plan["seeds"]), "Incomplete recording")
    require(sorted(i["seed"] for i in items) == sorted(plan["seeds"]), "Seed set mismatch")
    records, targets = [], Counter()
    stages = ("can_observed", "memory_barrier", "block_picked", "block_placed", "success")
    for item in items:
        seed = item["seed"]
        native = Path(item["episode_dir"]).resolve()
        meta = json.loads((native / "episode.json").read_text())
        planner = json.loads((native / "planner.json").read_text())
        metadata = meta["metadata"]
        diagnostics = planner["final_diagnostics"]
        length = meta["step_count"]
        require(item["success"] and meta["accepted"] and meta["planner_pass"], f"{seed}: failed")
        require(metadata["task_id"] == "WR-12" and metadata["seed"] == seed, f"{seed}: identity")
        require(metadata["task_version"] == "1.4.0", f"{seed}: task version")
        identity = plan["episode_identities"][str(seed)]
        require(
            item["episode_identity"] == metadata["episode_identity"] == identity, f"{seed}: hash"
        )
        require(metadata["policy_prompt"] == PROMPT, f"{seed}: prompt")
        require(planner["pass"] and planner["episode_result"]["success"], f"{seed}: planner")
        require(diagnostics["failure_code"] is None, f"{seed}: evaluation violation")
        require(all(diagnostics["stage_checks"][s] for s in stages), f"{seed}: stages")
        steps = [diagnostics["stage_steps"][s] for s in stages]
        require(steps == sorted(steps) and steps[-1] <= length, f"{seed}: stage order")
        require(0 < steps[0] <= 20 and steps[1] >= 28, f"{seed}: viewing/sealing order")
        require(steps[2] > steps[1], f"{seed}: pick before completed memory barrier")
        require(
            diagnostics["boxes_sealed"]
            and diagnostics["shutter_progress"] == 1.0
            and diagnostics["shutter_clock_step"] >= 26,
            f"{seed}: boxes not sealed",
        )
        require(
            diagnostics["visibility_error"] is None
            and all(
                diagnostics["visibility"][camera]["fully_hidden"]
                and not diagnostics["visibility"][camera]["visible"]
                for camera in CAMERAS
            ),
            f"{seed}: can not fully hidden after sealing",
        )
        facts = metadata["resolved_episode"]["facts"]
        require(facts["sampling_profile_id"] == "WR-12/sampling-v2", f"{seed}: sampling")
        targets[facts["target_box"]] += 1
        with h5py.File(native / "trajectory.h5") as handle:
            demos = list(handle["data"].values())
            require(len(demos) == 1, f"{seed}: demo count")
            demo = demos[0]
            for name, dimension in (("actions", 13), ("obs/robot_state", 37)):
                value = demo[name][:]
                require(value.shape == (length, dimension), f"{seed}: {name} shape")
                require(np.all(np.isfinite(value)), f"{seed}: {name} finite")
            require(bool(np.any(demo["success"][:])), f"{seed}: no HDF5 success")
            h5meta = json.loads(demo.attrs["ep_meta"])
            require(h5meta["policy_prompt"] == PROMPT, f"{seed}: HDF5 prompt")
            require(h5meta["episode_identity"] == identity, f"{seed}: HDF5 identity")
        require(tuple(meta["video"]["cameras"]) == CAMERAS, f"{seed}: cameras")
        require(meta["video"]["fps"] == 20, f"{seed}: fps")
        frames = {}
        for camera in CAMERAS:
            capture = cv2.VideoCapture(str(native / "videos" / f"{camera}.mp4"))
            decoded = 0
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    require(frame.shape == (224, 224, 3), f"{seed}: video dimensions")
                    decoded += 1
            finally:
                capture.release()
            require(decoded == length == meta["video"]["frame_counts"][camera], f"{seed}: frames")
            frames[camera] = decoded
        records.append(
            {"seed": seed, "frames": length, "video_frames": frames, "identity": identity}
        )
    require(dict(targets) == plan["target_counts"], "Target distribution changed")
    report = {
        "passed": True,
        "task_version": "1.4.0",
        "episodes": len(records),
        "total_frames": sum(r["frames"] for r in records),
        "target_counts": dict(targets),
        "records": records,
    }
    write_json(run_dir / "native_audit.json", report)
    destination = Path(plan["output_root"]) / f"{plan['run_id']}_complete"
    require(not destination.exists(), f"Selection already exists: {destination}")
    destination.mkdir(parents=True)
    for item in items:
        native = Path(item["episode_dir"]).resolve()
        target = destination / "train/WR-12" / native.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(native, target_is_directory=True)
    write_json(destination / "manifest.json", manifest)
    write_json(destination / "native_audit.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "records"}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    audit_run(parser.parse_args().run_dir)
