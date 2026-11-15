from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.record import audit_final_dataset as audit_module
from scripts.record.package_final_sources import selected_native_episode


@pytest.fixture
def final_pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    monkeypatch.setattr(audit_module, "ROOT", tmp_path)
    source = tmp_path / "source"
    native = tmp_path / "retry/train/WR-11/seed_000036_attempt_01"
    native.mkdir(parents=True)
    selected = source / "train/WR-11/seed_000036_attempt_01"
    selected.parent.mkdir(parents=True)
    selected.symlink_to(native, target_is_directory=True)
    dataset = tmp_path / "wr11-final"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "data/chunk-000").mkdir(parents=True)
    length = 5
    prompt = "Remember the sample and move sideways to choose the matching carton."
    meta = {
        "seed": 36,
        "task_id": "WR-11",
        "task_version": "3.2.0",
        "episode_identity": "identity-36",
        "policy_prompt": prompt,
        "release_facts": {"task_revision": "WR11-v3.2-eight-colors"},
    }
    (native / "episode.json").write_text(
        json.dumps(
            {
                "metadata": meta,
                "accepted": True,
                "planner_pass": True,
                "step_count": length,
            }
        )
    )
    state = np.arange(length * 37, dtype=np.float32).reshape(length, 37)
    action = np.arange(length * 13, dtype=np.float32).reshape(length, 13)
    flags = np.array([False, False, False, False, True])
    intrinsics = np.tile(np.eye(3, dtype=np.float32), (length, 1, 1))
    with h5py.File(native / "trajectory.h5", "w") as handle:
        demo = handle.create_group("data/demo_0")
        demo.attrs["ep_meta"] = json.dumps({"lang": prompt})
        demo["obs/robot_state"] = state
        demo["actions"] = action
        for key in ("terminated", "dones", "success"):
            demo[key] = flags
        demo["truncated"] = np.zeros(length, dtype=bool)
        demo["calibration/robot0_agentview_left/intrinsic"] = intrinsics
    columns = {
        "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), 37)),
        "action": pa.array(action.tolist(), type=pa.list_(pa.float32(), 13)),
        "observation.camera_intrinsics.robot0_agentview_left": intrinsics.tolist(),
        "next.terminated": flags.tolist(),
        "next.truncated": [False] * length,
        "next.done": flags.tolist(),
        "next.success": flags.tolist(),
        "frame_index": list(range(length)),
        "task_index": [0] * length,
    }
    pq.write_table(pa.table(columns), dataset / "data/chunk-000/file-000.parquet")
    info = {
        "total_episodes": 1,
        "total_frames": length,
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "features": {key: {} for key in columns},
    }
    (dataset / "meta/info.json").write_text(json.dumps(info))
    (dataset / "meta/seeds.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "seed": 36,
                "num_frames": length,
                "task_id": "WR-11",
                "task": prompt,
            }
        )
        + "\n"
    )
    pq.write_table(pa.table({"task_index": [0], "task": [prompt]}), dataset / "meta/tasks.parquet")
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "seed": 36,
                        "attempt": 1,
                        "success": True,
                        "episode_identity": "identity-36",
                        "expert_grasp_profile": "WR-11/grasp-v2",
                    }
                ]
            }
        )
    )
    return dataset, source


def test_audit_preserves_retry_provenance_through_selected_symlink(final_pair) -> None:
    dataset, source = final_pair
    report = audit_module.audit(dataset, source, 1)
    assert report["attempt_counts"] == {"1": 1}
    assert report["total_frames"] == 5
    assert report["items"][0]["source_episode"] == "retry/train/WR-11/seed_000036_attempt_01"
    assert report["items"][0]["expert_grasp_profile"] == "WR-11/grasp-v2"


@pytest.mark.parametrize("column", ["observation.state", "action", "next.success"])
def test_audit_rejects_changed_native_values(final_pair, column: str) -> None:
    dataset, source = final_pair
    path = dataset / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    values = table[column].to_pylist()
    if isinstance(values[0], list):
        values[0][0] += 1.0
    else:
        values[0] = not values[0]
    table = table.set_column(
        table.schema.get_field_index(column),
        column,
        pa.array(values, type=table.schema.field(column).type),
    )
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="differs"):
        audit_module.audit(dataset, source, 1)


def test_audit_rejects_changed_task_prompt(final_pair) -> None:
    dataset, source = final_pair
    pq.write_table(
        pa.table({"task_index": [0], "task": ["Pick the red carton."]}),
        dataset / "meta/tasks.parquet",
    )
    with pytest.raises(ValueError, match="task prompt"):
        audit_module.audit(dataset, source, 1)


def test_source_selection_rejects_ambiguous_attempts(final_pair) -> None:
    _, source = final_pair
    (source / "train/WR-11/seed_000036_attempt_00").mkdir()
    with pytest.raises(ValueError, match="ambiguous"):
        selected_native_episode(source, 36)


@pytest.mark.parametrize("wrong_recording", [False, True])
def test_audit_accepts_only_explicit_matching_prompt_revision(final_pair, wrong_recording) -> None:
    dataset, source = final_pair
    registry_path = dataset / "meta/seeds.jsonl"
    registry = json.loads(registry_path.read_text())
    original = registry["task"]
    revised = "Observe the reference color and place a matching carton in the empty tray."
    registry["task"] = revised
    registry_path.write_text(json.dumps(registry) + "\n")
    pq.write_table(pa.table({"task_index": [0], "task": [revised]}), dataset / "meta/tasks.parquet")
    revision = {
        "task_id": "WR-11",
        "recording_prompt": "incorrect" if wrong_recording else original,
        "published_prompt": revised,
    }
    (dataset / "meta/membench_prompt_revision.json").write_text(json.dumps(revision))
    if wrong_recording:
        with pytest.raises(ValueError, match="unexpected recording prompt"):
            audit_module.audit(dataset, source, 1)
    else:
        assert audit_module.audit(dataset, source, 1)["prompt_revision"] == revision
