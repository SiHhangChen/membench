"""Check or upload the nine final-source ZIPs without replacing existing files.

Use the existing ModelScope login. Only --upload permits remote writes. Root
file names and SHA256 values are checked before each upload and afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from pathlib import Path

from modelscope.hub.api import HubApi

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "outputs/datasets/source-zips"
REPO = "ouhebai/wr07-final"
TASKS = ("tm01", "tm02", "ts02", "wa01", "wa05", "wr01", "wr07", "wr08", "wx01")


def save(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def list_root(api: HubApi) -> dict:
    result = {}
    for page in range(1, 1000):
        files = api.get_dataset_files(REPO, recursive=False, page_number=page, page_size=100)
        for file in files:
            if file.get("Type") == "blob":
                result[file["Path"].lstrip("/")] = file
        if len(files) < 100:
            return result
    raise RuntimeError("ModelScope listing did not finish pagination")


def local_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def existing_record(name: str, local: dict, remote: dict) -> dict:
    size = int(remote.get("Size", 0))
    digest = remote.get("Sha256", "").lower()
    return {
        "name": name,
        "status": "skipped_existing",
        "remote_size": size,
        "remote_sha256": digest,
        "local_size": local["zip_bytes"],
        "local_sha256": local["zip_sha256"],
        "same_content": size == local["zip_bytes"] and digest == local["zip_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true", help="Upload missing ZIPs after checking")
    args = parser.parse_args()
    packaged = json.loads((DIRECTORY / "packaging-report.json").read_text())
    by_name = {Path(r["zip"]).name: r for r in packaged}
    names = [f"{task}-source.zip" for task in TASKS]
    for name in names:
        path = DIRECTORY / name
        if not path.is_file() or path.stat().st_size != by_name[name]["zip_bytes"]:
            raise RuntimeError(f"Local archive changed or is missing: {name}")
    api = HubApi()
    remote = list_root(api)
    save(DIRECTORY / "modelscope-remote-before.json", remote)
    records = []
    for name in names:
        if name in remote:
            record = existing_record(name, by_name[name], remote[name])
            print(f"SKIP {name}: exists; same_content={record['same_content']}", flush=True)
        else:
            record = {
                "name": name,
                "status": "missing",
                "local_size": by_name[name]["zip_bytes"],
                "local_sha256": by_name[name]["zip_sha256"],
            }
            print(f"MISSING {name}: {record['local_size'] / 2**20:.1f} MiB", flush=True)
        records.append(record)
    report = {"repo": REPO, "mode": "upload" if args.upload else "check", "files": records}
    report_path = DIRECTORY / "modelscope-upload-report.json"
    save(report_path, report)
    if not args.upload:
        return

    for index, name in enumerate(names):
        if records[index]["status"] == "skipped_existing":
            continue
        # Refresh immediately before writing, so another uploader's additions are skipped.
        remote = list_root(api)
        if name in remote:
            records[index] = existing_record(name, by_name[name], remote[name])
            save(report_path, report)
            print(f"SKIP {name}: appeared before upload", flush=True)
            continue
        path = DIRECTORY / name
        digest = local_sha256(path)
        if digest != by_name[name]["zip_sha256"]:
            raise RuntimeError(f"Local SHA256 changed: {name}")
        duplicates = [r for r in remote.values() if r.get("Sha256", "").lower() == digest]
        if duplicates:
            records[index].update(
                status="skipped_duplicate_content", remote_path=duplicates[0]["Path"]
            )
            save(report_path, report)
            print(
                f"SKIP {name}: identical content already exists as {duplicates[0]['Path']}",
                flush=True,
            )
            continue
        print(f"UPLOAD {name}: started", flush=True)
        stop = threading.Event()
        started = time.monotonic()

        def heartbeat(
            file_name: str = name, stop_event: threading.Event = stop, start_time: float = started
        ) -> None:
            while not stop_event.wait(30):
                print(
                    f"UPLOAD {file_name}: running {int(time.monotonic() - start_time)}s", flush=True
                )

        worker = threading.Thread(target=heartbeat, daemon=True)
        worker.start()
        try:
            commit = api.upload_file(
                path_or_fileobj=path,
                path_in_repo=name,
                repo_id=REPO,
                repo_type="dataset",
                commit_message=f"Add {name}: 200 successful final dataset recordings",
                disable_tqdm=True,
            )
        # SDK exceptions can contain signed URLs; retain the error type without exposing them.
        except Exception as error:  # noqa: BLE001
            records[index].update(status="upload_failed", error_type=type(error).__name__)
            save(report_path, report)
            raise RuntimeError(f"Upload failed for {name}: {type(error).__name__}") from None
        finally:
            stop.set()
            worker.join()
        uploaded = None
        for attempt in range(6):
            uploaded = list_root(api).get(name)
            if uploaded:
                break
            if attempt < 5:
                time.sleep(5)
        if not uploaded:
            raise RuntimeError(f"Uploaded file is absent from remote listing: {name}")
        if (
            int(uploaded.get("Size", 0)) != path.stat().st_size
            or uploaded.get("Sha256", "").lower() != digest
        ):
            raise RuntimeError(f"Remote size/SHA256 validation failed after uploading {name}")
        records[index].update(
            status="uploaded_verified",
            remote_size=uploaded["Size"],
            remote_sha256=digest,
            commit_url=getattr(commit, "commit_url", None),
        )
        save(report_path, report)
        print(f"DONE {name}: remote size and SHA256 match", flush=True)
    final = list_root(api)
    save(DIRECTORY / "modelscope-remote-after.json", final)
    for record in records:
        if record["status"] == "uploaded_verified":
            current = final.get(record["name"], {})
            if current.get("Sha256", "").lower() != record["remote_sha256"]:
                raise RuntimeError(f"Remote file changed after upload: {record['name']}")
    save(report_path, report)
    print(
        "FINISHED "
        + json.dumps(
            {
                status: sum(r["status"] == status for r in records)
                for status in sorted({r["status"] for r in records})
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
