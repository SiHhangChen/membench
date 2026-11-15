"""Upload the verified WR-12 final and native source archives to ModelScope."""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

from modelscope.hub.api import HubApi

from scripts.record.upload_source_archives_modelscope import list_root, local_sha256, save

ROOT = Path(__file__).resolve().parents[2]
REPO = "ouhebai/wr07-final"
REPORT_ROOT = ROOT / "outputs/conversion/wr12-final/modelscope_upload"
ARCHIVES = (
    (
        ROOT / "data/final-zip/wr12-final.zip",
        ROOT / "outputs/conversion/wr12-final/final_zip_report.json",
    ),
    (
        ROOT / "outputs/datasets/wr12-source.zip",
        ROOT / "outputs/conversion/wr12-final/source_zip_report.json",
    ),
)


def matches(remote: dict, local: dict) -> bool:
    return int(remote.get("Size", -1)) == local.get("zip_bytes", -2) and str(
        remote.get("Sha256", "")
    ).lower() == local.get("zip_sha256", "missing")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument(
        "--replace-report",
        type=Path,
        help="Replace only remote files matching the exact hashes in this previous upload report.",
    )
    args = parser.parse_args()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    for path, metadata_path in ARCHIVES:
        metadata = json.loads(metadata_path.read_text())
        if (
            path.stat().st_size != metadata["zip_bytes"]
            or local_sha256(path) != metadata["zip_sha256"]
        ):
            raise RuntimeError(f"Archive differs from verified packaging result: {path.name}")
        records.append(
            {
                "name": path.name,
                "local_path": str(path),
                "zip_bytes": metadata["zip_bytes"],
                "zip_sha256": metadata["zip_sha256"],
                "status": "local_verified",
            }
        )
    api = HubApi()
    remote = list_root(api)
    save(REPORT_ROOT / "remote_before.json", remote)
    report = {"repo": REPO, "mode": "upload" if args.upload else "check", "files": records}
    report_path = REPORT_ROOT / "upload_report.json"
    previous = {}
    if args.replace_report:
        old_report = json.loads(args.replace_report.read_text())
        if (
            old_report.get("repo") != REPO
            or old_report.get("status") != "both_archives_remote_verified"
        ):
            raise RuntimeError(
                "Replacement requires a verified previous upload report for this repository"
            )
        previous = {item["name"]: item for item in old_report["files"]}
        report["replace_report"] = str(args.replace_report)
    for record in records:
        existing = remote.get(record["name"])
        if existing is not None:
            if not matches(existing, record):
                if matches(existing, previous.get(record["name"], {})):
                    record["status"] = "previous_release_verified_for_replacement"
                else:
                    record["status"] = "remote_content_conflict"
                    save(report_path, report)
                    raise RuntimeError(
                        f"Existing remote file has different content: {record['name']}"
                    )
            else:
                record["status"] = "existing_verified"
        else:
            record["status"] = "missing"
        print(f"CHECK {record['name']}: {record['status']}", flush=True)
    save(report_path, report)
    if not args.upload:
        return
    for record in records:
        name = record["name"]
        current = list_root(api).get(name)
        if current is not None:
            if matches(current, record):
                record["status"] = "existing_verified"
                save(report_path, report)
                continue
            if not matches(current, previous.get(name, {})):
                raise RuntimeError(f"Remote file appeared with different content: {name}")
            record["replaces_remote_sha256"] = current["Sha256"]
        print(f"UPLOAD {name}: {record['zip_bytes']} bytes", flush=True)
        stop = threading.Event()
        started = time.monotonic()

        def heartbeat(file_name=name, stop_event=stop, start_time=started) -> None:
            while not stop_event.wait(30):
                print(
                    f"UPLOAD {file_name}: running {int(time.monotonic() - start_time)}s", flush=True
                )

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            commit = api.upload_file(
                repo_id=REPO,
                path_or_fileobj=Path(record["local_path"]),
                path_in_repo=name,
                repo_type="dataset",
                disable_tqdm=True,
                commit_message=f"Add {name}: WR-12 v1.4, 200 successful episodes",
            )
        except Exception as error:  # noqa: BLE001 -- SDK errors may contain signed URLs
            record.update(status="upload_failed", error_type=type(error).__name__)
            save(report_path, report)
            raise RuntimeError(f"Upload failed: {name}, {type(error).__name__}") from None
        finally:
            stop.set()
            thread.join()
        uploaded = None
        for attempt in range(12):
            uploaded = list_root(api).get(name)
            if uploaded and matches(uploaded, record):
                break
            if attempt < 11:
                time.sleep(5)
        if not uploaded or not matches(uploaded, record):
            raise RuntimeError(f"Uploaded file failed remote size/SHA256 verification: {name}")
        record.update(
            status="uploaded_verified",
            remote_size=uploaded["Size"],
            remote_sha256=uploaded["Sha256"],
            commit_url=getattr(commit, "commit_url", None),
        )
        save(report_path, report)
        print(f"DONE {name}: remote size and SHA256 verified", flush=True)
    after = list_root(api)
    for record in records:
        if not matches(after.get(record["name"], {}), record):
            raise RuntimeError(f"Final remote verification failed: {record['name']}")
    save(REPORT_ROOT / "remote_after.json", after)
    report["status"] = "both_archives_remote_verified"
    save(report_path, report)
    print("PASS: both WR-12 archives uploaded and verified on ModelScope", flush=True)


if __name__ == "__main__":
    main()
