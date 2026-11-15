"""Package successful native recordings corresponding to final LeRobot datasets.

Each ZIP contains a dataset directory with manifest.json and train/<task>/...
as in ts03-source.zip. Only episodes selected by the final seed registry are
included. Validate native actions against the final parquet data before packing,
then check ZIP CRCs and write a SHA256 report. Existing archives are never replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import h5py
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "tm01": "tm01_200success_20260917",
    "tm02": "tm02_200success_20260917",
    "ts02": "ts02-source",
    "wa01": "wa01_200success_20260916",
    "wa05": "wa05_200success_20260916",
    "wr01": "wr01_200seeds_20260916_174341",
    "wr07": "wr07_200seeds_v14_merged",
    "wr08": "wr08_200success_20260916",
    "wr11": "wr11_raw",
    "wx01": "wx01_200success_20260916",
}
CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_native_episode(source: Path, seed: int) -> tuple[Path, int]:
    candidates = list((source / "train").glob(f"*/seed_{seed:06d}_attempt_*"))
    require(len(candidates) == 1, f"missing/ambiguous native seed {seed} in {source}")
    native = candidates[0]
    return native, int(native.name.rsplit("_attempt_", 1)[1])


def package_task(task: str, output: Path) -> dict:
    source = ROOT / "outputs/datasets" / SOURCES[task]
    dataset = ROOT / "data/final" / f"{task}-final"
    destination = output / f"{task}-source.zip"
    require(not destination.exists(), f"archive already exists: {destination}")
    rows = [
        json.loads(line)
        for line in (dataset / "meta/seeds.jsonl").read_text().splitlines()
        if line.strip()
    ]
    info = json.loads((dataset / "meta/info.json").read_text())
    require(len(rows) == info["total_episodes"] == 200, f"{task}: expected 200 episodes")
    require(len({r["seed"] for r in rows}) == 200, f"{task}: duplicate seeds")
    require(
        sorted(r["episode_index"] for r in rows) == list(range(200)),
        f"{task}: invalid episode indices",
    )
    require(
        sum(r["num_frames"] for r in rows) == info["total_frames"],
        f"{task}: final frame total mismatch",
    )
    print(f"{task}: checking 200 successful recordings against final actions", flush=True)

    # Load only action and indexing columns; videos and camera metadata remain external.
    parts = {}
    for path in sorted((dataset / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "frame_index", "action"])
        episodes = table["episode_index"].to_numpy()
        frames = table["frame_index"].to_numpy()
        action_column = table["action"].combine_chunks()
        actions = action_column.values.to_numpy().reshape(len(table), action_column.type.list_size)
        for episode_index in np.unique(episodes):
            mask = episodes == episode_index
            parts.setdefault(int(episode_index), []).append((frames[mask], actions[mask]))
    require(set(parts) == set(range(200)), f"{task}: final parquet episode set mismatch")

    files, items = [], []
    for row in sorted(rows, key=lambda r: r["episode_index"]):
        seed, index, length = int(row["seed"]), int(row["episode_index"]), int(row["num_frames"])
        native, attempt = selected_native_episode(source, seed)
        episode = json.loads((native / "episode.json").read_text())
        metadata = episode["metadata"]
        require(
            episode["accepted"] is True and episode["planner_pass"] is True,
            f"{task}: seed {seed} is not successful",
        )
        require(
            metadata["seed"] == seed and metadata["task_id"] == row["task_id"],
            f"{task}: seed/task identity mismatch for {seed}",
        )
        require(episode["step_count"] == length, f"{task}: seed {seed} frame count mismatch")
        required = ["episode.json", "planner.json", "events.jsonl", "trajectory.h5"]
        required.extend(f"videos/{camera}.mp4" for camera in CAMERAS)
        require(
            all(
                (native / name).is_file() and (native / name).stat().st_size > 0
                for name in required
            ),
            f"{task}: seed {seed} incomplete recording",
        )
        planner = json.loads((native / "planner.json").read_text())
        require(
            planner["pass"] is True and planner["episode_result"]["success"] is True,
            f"{task}: seed {seed} planner failed",
        )
        chunks = parts.pop(index)
        frames = np.concatenate([p[0] for p in chunks])
        actions = np.concatenate([p[1] for p in chunks])
        order = np.argsort(frames)
        require(
            np.array_equal(frames[order], np.arange(length)),
            f"{task}: episode {index} final frame indices mismatch",
        )
        with h5py.File(native / "trajectory.h5", "r") as handle:
            demos = list(handle["data"].values())
            require(len(demos) == 1, f"{task}: seed {seed} ambiguous native trajectory")
            recorded = demos[0]["actions"][:]
        require(
            np.array_equal(actions[order], recorded),
            f"{task}: seed {seed} native/final action mismatch",
        )
        episode_files = sorted(p for p in native.rglob("*") if p.is_file())
        for path in episode_files:
            require(not path.is_symlink(), f"unexpected source symlink: {path}")
            stat = path.stat()
            files.append((path, stat.st_size, stat.st_mtime_ns))
        items.append(
            {
                "seed": seed,
                "attempt": attempt,
                "success": True,
                "task_id": metadata["task_id"],
                "variant": metadata["variant"],
                "episode_identity": metadata["episode_identity"],
                "stage_count": metadata["expert_plan"]["stage_count"],
                "episode_dir": str(native.relative_to(source)),
                "final_episode_index": index,
                "num_frames": length,
            }
        )
    require(not parts, f"{task}: unexpected final episodes")
    manifest = {"schema_version": "membench.dataset/v2", "dataset_id": source.name, "items": items}
    package_info = {
        "task": task,
        "episodes": 200,
        "failed_episodes": 0,
        "total_frames": info["total_frames"],
        "source_root": str(source),
        "final_dataset_root": str(dataset),
        "selection": "exact final meta/seeds.jsonl seed set",
        "validation": "every native action equals its final parquet action exactly",
        "manifest": "regenerated from selected successful episode metadata with portable paths",
    }
    generated = {
        f"{source.name}/manifest.json": json_bytes(manifest),
        f"{source.name}/package_info.json": json_bytes(package_info),
        f"{source.name}/final_seeds.jsonl": (dataset / "meta/seeds.jsonl").read_bytes(),
    }
    fd, temporary_name = tempfile.mkstemp(prefix=f".{task}-", suffix=".zip", dir=output)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        total_bytes = sum(size for _, size, _ in files)
        print(f"{task}: packing {len(files)} files ({total_bytes / 2**30:.2f} GiB)", flush=True)
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=3) as archive:
            for name, data in generated.items():
                archive.writestr(name, data)
            for path, size, mtime in files:
                stat = path.stat()
                require(
                    (stat.st_size, stat.st_mtime_ns) == (size, mtime),
                    f"source changed before packaging: {path}",
                )
                archive.write(
                    path,
                    path.relative_to(source.parent),
                    compress_type=ZIP_STORED if path.suffix == ".mp4" else ZIP_DEFLATED,
                )
                stat = path.stat()
                require(
                    (stat.st_size, stat.st_mtime_ns) == (size, mtime),
                    f"source changed during packaging: {path}",
                )
        with ZipFile(temporary) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            expected_names = set(generated) | {
                str(p.relative_to(source.parent)) for p, _, _ in files
            }
            require(
                len(names) == len(set(names)) and set(names) == expected_names,
                f"{task}: ZIP entries differ from selected files",
            )
            require(
                not any("/debug_failed/" in name for name in names),
                f"{task}: failed recordings in archive",
            )
            for path, size, _ in files:
                require(
                    archive.getinfo(str(path.relative_to(source.parent))).file_size == size,
                    f"{task}: ZIP file size mismatch",
                )
            require(archive.testzip() is None, f"{task}: ZIP CRC validation failed")
        digest = sha256(temporary)
        # All source files must still match their validated snapshots when publishing.
        require(
            all(
                (p.stat().st_size, p.stat().st_mtime_ns) == (size, mtime)
                for p, size, mtime in files
            ),
            f"{task}: source changed during packaging",
        )
        require(not destination.exists(), f"archive appeared during packaging: {destination}")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    report = {
        **package_info,
        "zip": str(destination),
        "zip_bytes": destination.stat().st_size,
        "zip_sha256": digest,
        "files": len(files) + len(generated),
        "crc_check": "passed",
        "source_files_unchanged": True,
    }
    (output / f"{task}-source.report.json").write_bytes(json_bytes(report))
    print(f"{task}: done, {destination.stat().st_size / 2**20:.1f} MiB, CRC passed", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/datasets/source-zips")
    parser.add_argument("--tasks", nargs="+", choices=sorted(SOURCES), default=list(SOURCES))
    parser.add_argument("--jobs", type=int, choices=range(1, 4), default=3)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    require(len(args.tasks) == len(set(args.tasks)), "duplicate tasks requested")
    for task in args.tasks:
        require(not (output / f"{task}-source.zip").exists(), f"existing archive for {task}")
    reports = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(package_task, task, output): task for task in args.tasks}
        for future in as_completed(futures):
            reports.append(future.result())
    reports.sort(key=lambda r: r["task"])
    (output / "packaging-report.json").write_bytes(json_bytes(reports))
    (output / "SHA256SUMS").write_text(
        "".join(f"{r['zip_sha256']}  {Path(r['zip']).name}\n" for r in reports), encoding="utf-8"
    )
    print(f"Completed {len(reports)} archives: {output}", flush=True)


if __name__ == "__main__":
    main()
