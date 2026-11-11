#!/usr/bin/env python
"""Validate and annotate a single-task ACT checkpoint for MemBench serving."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    task = args.task.lower()
    dataset = ROOT / f"data/final/{task}-final"
    checkpoint = args.checkpoint or (
        ROOT / f"outputs/train/act/final_bs48_stride10_100k_torchcodec/{task}"
        / f"checkpoints/{args.step:06d}/pretrained_model"
    )
    info = json.loads((dataset / "meta/info.json").read_text())
    config = json.loads((checkpoint / "config.json").read_text())
    features = info["features"]
    expected_cameras = {
        alias: f"observation.images.robot0_{alias}"
        for alias in ("agentview_left", "agentview_right", "eye_in_hand")
    }
    checks = (
        (features["observation.state"]["shape"] == [37], "dataset state shape"),
        (features["action"]["shape"] == [13], "dataset action shape"),
        (features["observation.state"]["info"]["schema"] == "membench.robot_state37.v2", "state schema"),
        (features["action"]["info"]["schema"] == "membench.pandaomron_hybrid13.v2", "action schema"),
        (config["type"] == "act" and config["chunk_size"] > 0
         and 0 < config["n_action_steps"] <= config["chunk_size"], "ACT config"),
        (config["input_features"]["observation.state"]["shape"] == [37], "checkpoint state shape"),
        (config["output_features"]["action"]["shape"] == [13], "checkpoint action shape"),
    )
    for valid, name in checks:
        if not valid:
            raise ValueError(f"Invalid {name} for {task}")
    for key in expected_cameras.values():
        if features[key]["shape"] != [224, 224, 3] or config["input_features"][key]["shape"] != [3, 224, 224]:
            raise ValueError(f"Invalid camera contract: {key}")
    payload = {
        "protocol_version": "membench-policy/v2",
        "state_schema": "membench.robot_state37.v2",
        "state_dim": 37,
        "action_schema": "membench.pandaomron_hybrid13.v2",
        "action_dim": 13,
        "training_step": args.step,
        "dataset_root": str(dataset),
        "camera_mapping": expected_cameras,
        "image_size": [224, 224],
        "contract_validation": "dataset_and_checkpoint_metadata",
    }
    path = checkpoint / "membench_policy.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)
    print(path)


if __name__ == "__main__":
    main()
