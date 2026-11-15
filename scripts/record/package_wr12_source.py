"""Export WR12 native recordings, recording-time sources and validation reports."""

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.record.audit_final_dataset import audit
from scripts.record.package_final_sources import require, sha256


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    source = ROOT / "outputs/datasets/wr12_200success_v14_complete"
    recording = ROOT / "outputs/record/wr12_200seeds_v14"
    conversion = ROOT / "outputs/conversion/wr12-final"
    final = ROOT / "data/final/wr12-final"
    export = ROOT / "outputs/datasets/wr12-source"
    archive_path = ROOT / "outputs/datasets/source-zips/wr12-source.zip"
    alias = ROOT / "outputs/datasets/wr12-source.zip"
    require(
        not any(p.exists() or p.is_symlink() for p in (export, archive_path, alias)),
        "WR12 source export/archive already exists",
    )
    verified = audit(final, source, 200)
    save(conversion / "source_package_native_audit.json", verified)
    plan = json.loads((recording / "run_plan.json").read_text())
    snapshot = recording / "source_snapshot/bench/tasks/wr12"
    digest = hashlib.sha256()
    for path in sorted(snapshot.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            digest.update(str(Path("bench/tasks/wr12") / path.relative_to(snapshot)).encode())
            digest.update(path.read_bytes())
    require(digest.hexdigest() == plan["task_source_sha256"], "Recording source snapshot differs")
    manifest = json.loads((source / "manifest.json").read_text())
    require(
        sorted(item["seed"] for item in manifest["items"])
        == [seed for seed in range(201) if seed != 70],
        "Seed set differs",
    )
    stage = Path(tempfile.mkdtemp(prefix=".wr12-source-", dir=export.parent))
    payload_hashes, originals = {}, {}
    try:
        registry = [
            json.loads(line) for line in (final / "meta/seeds.jsonl").read_text().splitlines()
        ]
        final_indices = {row["seed"]: row["episode_index"] for row in registry}
        require(len(final_indices) == 200, "Final seed registry differs")
        items = []
        for item in sorted(manifest["items"], key=lambda value: value["seed"]):
            native = Path(item["episode_dir"])
            relative = Path("train/WR-12") / native.name
            target = stage / relative
            shutil.copytree(native, target, copy_function=os.link)
            episode = json.loads((native / "episode.json").read_text())
            require(episode["accepted"] and item["success"], "Unsuccessful episode")
            original = (
                stage
                / "recording_provenance/original_episode_metadata"
                / native.name
                / "episode.json"
            )
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(native / "episode.json", original)
            # Break the metadata hard link before making file references portable.
            (target / "episode.json").unlink()
            episode["files"] = {
                key: Path(value).name for key, value in episode["files"].items() if key != "videos"
            }
            episode["files"]["videos"] = {
                camera: f"videos/{camera}.mp4" for camera in episode["video"]["cameras"]
            }
            episode["trajectory_h5"]["path"] = "trajectory.h5"
            save(target / "episode.json", episode)
            for key, value in episode["files"].items():
                paths = value.values() if key == "videos" else [value]
                require(all((target / p).is_file() for p in paths), "Broken relative episode path")
            for file in native.rglob("*"):
                if file.is_file():
                    originals[str(file)] = sha256(file)
                    if file.name != "episode.json":
                        name = str(relative / file.relative_to(native))
                        payload_hashes[name] = originals[str(file)]
            items.append(
                {
                    **item,
                    "episode_dir": relative.as_posix(),
                    "source_episode_dir": relative.as_posix(),
                    "final_episode_index": final_indices[item["seed"]],
                }
            )
        save(stage / "manifest.json", {**manifest, "dataset_id": "wr12-source", "items": items})
        shutil.copytree(
            recording / "source_snapshot",
            stage / "recording_provenance/source_snapshot",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        for name in ("run_plan.json", "manifest.json", "summary.json"):
            shutil.copy2(recording / name, stage / "recording_provenance" / name)
        shutil.copy2(
            ROOT / "outputs/record/wr12_200success_v14/summary.json",
            stage / "recording_summary.json",
        )
        shutil.copy2(final / "meta/seeds.jsonl", stage / "final_seeds.jsonl")
        for name in (
            "summary.json",
            "conversion_result.json",
            "plan.json",
            "native_audit.json",
            "source_package_native_audit.json",
            "video_repair.json",
            "validate_repaired.log",
        ):
            (stage / "validation").mkdir(exist_ok=True)
            shutil.copy2(conversion / name, stage / "validation" / name)
        # Preserve both recording batches and the final selection, including the replacement seed.
        for run_name in ("wr12_v14_supplement_seed200", "wr12_200success_v14"):
            run = ROOT / "outputs/record" / run_name
            batch = stage / "recording_provenance/batches" / run_name
            batch.mkdir(parents=True)
            for name in ("run_plan.json", "manifest.json", "summary.json", "native_audit.json"):
                if (run / name).is_file():
                    shutil.copy2(run / name, batch / name)
            if (run / "source_snapshot").is_dir():
                shutil.copytree(
                    run / "source_snapshot",
                    batch / "source_snapshot",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
        for name in ("run_conversion.py", "convert_wr12_parallel.py"):
            target = stage / "recording_provenance/conversion_scripts" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(conversion / name, target)
        scripts = stage / "recording_provenance/export_scripts"
        scripts.mkdir(parents=True)
        for name in (
            "package_wr12_source.py",
            "record_wr12_200seeds.py",
            "convert_bench_episodes_to_lerobot.py",
            "convert_bench_episodes_to_lerobot_v061.py",
            "lerobot_seed_dataset.py",
            "audit_final_dataset.py",
            "package_final_sources.py",
            "launch_wr12_200seeds.sh",
        ):
            shutil.copy2(ROOT / "scripts/record" / name, scripts / name)
        for path in (
            ROOT / "scripts/train/validate_lerobot_v3_dataset.py",
            ROOT / "scripts/tools/activate_sim_env.sh",
        ):
            shutil.copy2(path, scripts / path.name)
        save(
            stage / "package_info.json",
            {
                "task_id": "WR-12",
                "episodes": 200,
                "total_frames": 131328,
                "fps": 20,
                "cameras": 3,
                "state_dim": 37,
                "action_dim": 13,
                "repo_id": "wr12-final",
                "recording_task_source_sha256": plan["task_source_sha256"],
                "source": "200 successful episodes selected by final meta/seeds.jsonl",
                "paths": "manifest paths relative to source root; episode file paths relative to episode directory",
                "preservation": "HDF5, original videos, planner and events unchanged; only exported episode.json file references normalized; original episode.json preserved in recording_provenance",
                "annotations": "stage boundaries and semantic annotations have not been produced",
                "replacement": {"excluded_failed_seed": 70, "supplement_seed": 200},
                "task_version": "1.4.0",
                "target_counts": {"left": 100, "right": 100},
                "source_video_note": "Source videos are original recordings. Final LeRobot seed185 wrist video was reencoded from its complete 559-frame original; report included in validation/.",
                "excluded": "failed runs, caches, converted LeRobot videos and temporary conversion shards",
                "source_comparison": "state, action, terminal flags, camera calibration and task text exactly audited against final",
            },
        )
        (stage / "README.md").write_text(
            "# WR12 native source\n\n"
            "200 successful episodes, seeds 0–69 and 71–200, 131,328 frames at 20 Hz.\n"
            "train/WR-12 contains original HDF5 trajectories, three RGB videos, planner/events and episode metadata.\n"
            "manifest.json and episode.json use portable file references. Original episode metadata, recording-time task sources and recording scripts are preserved under recording_provenance/.\n"
            "validation/ contains exact native/final comparisons and video/training-loading checks.\n"
            "No stage-boundary or semantic annotation package has been produced. External simulator assets and Python environments are repository dependencies and are not bundled.\n"
            "file_inventory.json records SHA256 and size for every payload file.\n"
        )
        inventory = []
        for path in sorted(stage.rglob("*")):
            require(not path.is_symlink(), f"Export contains symlink: {path}")
            if path.is_file():
                relative = str(path.relative_to(stage))
                checksum = sha256(path)
                if relative in payload_hashes:
                    require(checksum == payload_hashes[relative], "Native payload changed")
                inventory.append(
                    {"path": relative, "bytes": path.stat().st_size, "sha256": checksum}
                )
        save(stage / "file_inventory.json", {"files": inventory})
        stage.replace(export)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_suffix(".zip.tmp")
    require(not temporary.exists(), "Temporary archive already exists")
    try:
        files = sorted(path for path in export.rglob("*") if path.is_file())
        print(f"Packing {len(files)} files", flush=True)
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=3) as archive:
            for path in files:
                archive.write(
                    path,
                    f"wr12-source/{path.relative_to(export).as_posix()}",
                    compress_type=ZIP_STORED if path.suffix == ".mp4" else ZIP_DEFLATED,
                )
        with ZipFile(temporary) as archive:
            require(archive.testzip() is None, "ZIP CRC failed")
            require(len(archive.namelist()) == len(files), "ZIP file count differs")
            for path in files:
                entry = f"wr12-source/{path.relative_to(export).as_posix()}"
                require(
                    hashlib.sha256(archive.read(entry)).hexdigest() == sha256(path),
                    f"ZIP content differs: {entry}",
                )
        require(
            all(sha256(Path(path)) == checksum for path, checksum in originals.items()),
            "Original recording changed",
        )
        checksum = sha256(temporary)
        temporary.replace(archive_path)
    finally:
        temporary.unlink(missing_ok=True)
    alias.symlink_to(archive_path.relative_to(alias.parent))
    report = {
        "zip": str(archive_path),
        "export": str(export),
        "episodes": 200,
        "frames": 131328,
        "files": len(files),
        "zip_bytes": archive_path.stat().st_size,
        "zip_sha256": checksum,
        "crc_check": "passed",
        "all_entry_sha256": "passed",
        "source_files_unchanged": True,
        "recording_snapshot_hash": "passed",
    }
    save(archive_path.with_suffix(".report.json"), report)
    save(conversion / "source_zip_report.json", report)
    archive_path.with_suffix(".zip.sha256").write_text(f"{checksum}  {archive_path.name}\n")
    alias.with_suffix(".zip.sha256").write_text(f"{checksum}  {alias.name}\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
