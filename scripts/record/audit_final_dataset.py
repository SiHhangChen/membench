"""Compare a final LeRobot dataset against its selected native recordings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.record.package_final_sources import require, selected_native_episode


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def audit(dataset: Path, source: Path, expected_episodes: int) -> dict:
    info = json.loads((dataset / "meta/info.json").read_text())
    registry = [
        json.loads(line) for line in (dataset / "meta/seeds.jsonl").read_text().splitlines()
    ]
    require(len(registry) == expected_episodes == info["total_episodes"], "episode count mismatch")
    require([r["episode_index"] for r in registry] == list(range(expected_episodes)), "bad indices")
    require(len({r["seed"] for r in registry}) == expected_episodes, "duplicate seeds")
    tasks = (
        pq.read_table(dataset / "meta/tasks.parquet")
        .to_pandas()
        .reset_index()
        .set_index("task_index")
    )
    source_manifest = source / "manifest.json"
    prompt_revision_path = dataset / "meta/membench_prompt_revision.json"
    prompt_revision = (
        json.loads(prompt_revision_path.read_text()) if prompt_revision_path.exists() else None
    )
    selected = {
        int(item["seed"]): item for item in json.loads(source_manifest.read_text())["items"]
    }
    require(set(selected) == {r["seed"] for r in registry}, "source/final seed sets differ")
    native_columns = {
        "observation.state": "obs/robot_state",
        "action": "actions",
        "next.terminated": "terminated",
        "next.truncated": "truncated",
        "next.done": "dones",
        "next.success": "success",
    }
    for key in info["features"]:
        for prefix, leaf in (
            ("observation.camera_intrinsics.", "intrinsic"),
            ("observation.camera_extrinsics_c2w.", "extrinsic_c2w"),
        ):
            if key.startswith(prefix):
                native_columns[key] = f"calibration/{key.removeprefix(prefix)}/{leaf}"
    provenance, total_frames = [], 0
    for row in registry:
        index, seed, length = row["episode_index"], row["seed"], row["num_frames"]
        native, attempt = selected_native_episode(source, seed)
        selection = selected[seed]
        require(selection["attempt"] == attempt and selection["success"], f"seed {seed}: selection")
        episode_bytes = (native / "episode.json").read_bytes()
        episode = json.loads(episode_bytes)
        meta = episode["metadata"]
        require(episode["accepted"] and episode["planner_pass"], f"seed {seed}: unsuccessful")
        require(
            meta["seed"] == seed and meta["task_id"] == row["task_id"], f"seed {seed}: identity"
        )
        require(meta["episode_identity"] == selection["episode_identity"], f"seed {seed}: manifest")
        require(episode["step_count"] == length, f"seed {seed}: length")
        chunk, file = divmod(index, info["chunks_size"])
        table = pq.read_table(
            dataset / info["data_path"].format(chunk_index=chunk, file_index=file)
        )
        require(len(table) == length, f"seed {seed}: converted length")
        require(np.array_equal(table["frame_index"].to_numpy(), np.arange(length)), "frame indices")
        with h5py.File(native / "trajectory.h5", "r") as handle:
            demos = list(handle["data"].values())
            require(len(demos) == 1, f"seed {seed}: ambiguous trajectory")
            demo = demos[0]
            for column, h5path in native_columns.items():
                recorded = demo[h5path][:]
                converted = np.asarray(table[column].to_pylist()).reshape(recorded.shape)
                require(np.array_equal(converted, recorded), f"seed {seed}: {column} differs")
            ep_meta = json.loads(demo.attrs["ep_meta"])
            prompt = str(ep_meta.get("lang") or meta["policy_prompt"]).strip()
        published_prompt = prompt
        if prompt_revision is not None:
            require(
                meta["task_id"] == prompt_revision["task_id"]
                and prompt == prompt_revision["recording_prompt"],
                f"seed {seed}: unexpected recording prompt for published revision",
            )
            published_prompt = prompt_revision["published_prompt"]
        require(row["task"] == published_prompt, f"seed {seed}: registry prompt")
        task_indices = set(table["task_index"].to_pylist())
        require(len(task_indices) == 1, f"seed {seed}: multiple prompts")
        require(
            tasks.loc[next(iter(task_indices)), "task"] == published_prompt,
            f"seed {seed}: task prompt",
        )
        provenance.append(
            {
                "episode_index": index,
                "seed": seed,
                "attempt": attempt,
                "num_frames": length,
                "task_id": meta["task_id"],
                "task_version": meta["task_version"],
                "episode_identity": meta["episode_identity"],
                "release_facts": meta["release_facts"],
                "expert_grasp_profile": selection.get("expert_grasp_profile"),
                "native_episode": str(native.relative_to(ROOT)),
                "source_episode": str(native.resolve().relative_to(ROOT)),
                "episode_json_sha256": hashlib.sha256(episode_bytes).hexdigest(),
            }
        )
        total_frames += length
        if (index + 1) % 25 == 0:
            print(f"Verified {index + 1}/{expected_episodes} native episodes", flush=True)
    require(total_frames == info["total_frames"], "total frames differ")
    return {
        "schema_version": "membench.final_native_audit/v1",
        "dataset": str(dataset.relative_to(ROOT)),
        "source": str(source.relative_to(ROOT)),
        "source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
        "episodes": expected_episodes,
        "total_frames": total_frames,
        "attempt_counts": dict(Counter(str(item["attempt"]) for item in provenance)),
        "exactly_verified": list(native_columns) + ["task", "frame_index"],
        "prompt_revision": prompt_revision,
        "items": provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=200)
    parser.add_argument("--write-provenance", action="store_true")
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset.absolute()
    report = audit(dataset, args.source.absolute(), args.expected_episodes)
    write_json(args.output_report, report)
    if args.write_provenance:
        write_json(dataset / "meta/membench_provenance.json", report)
    print(f"PASS: {report['episodes']} episodes, {report['total_frames']} frames")


if __name__ == "__main__":
    main()
