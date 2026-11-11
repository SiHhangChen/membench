from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bench.evaluation.replay import (
    ReplayPolicyClient,
    discover_recorded_episodes,
    parse_seed_spec,
    replay_policy_contract,
    task_suite_path,
)


def test_default_replay_suite_uses_latest_numeric_version(tmp_path, monkeypatch) -> None:
    from bench.evaluation import replay

    monkeypatch.setattr(replay, "ROOT", tmp_path)
    suites = tmp_path / "configs/eval/suites"
    suites.mkdir(parents=True)
    for name in ("wr06_v1.yaml", "wr06_v2.yaml", "wr06_v10.yaml", "wr06_v99_debug.yaml"):
        (suites / name).touch()
    assert task_suite_path("WR06") == suites / "wr06_v10.yaml"
    assert task_suite_path("WR-05") == suites / "wr05_v1.yaml"


def _episode(root: Path, branch: str, seed: int, attempt: int = 0) -> Path:
    path = root / branch / "WR-06" / f"seed_{seed:06d}_attempt_{attempt:02d}"
    path.mkdir(parents=True)
    (path / "trajectory.h5").touch()
    return path


def test_parse_seed_spec_accepts_ranges_and_descending_ranges() -> None:
    assert parse_seed_spec("0-2,5;7-6") == {0, 1, 2, 5, 6, 7}


def test_recording_discovery_includes_failed_tree_with_partial_manifest(
    tmp_path: Path,
) -> None:
    accepted = _episode(tmp_path, "train", seed=0)
    failed = _episode(tmp_path, "debug_failed", seed=1)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "task_id": "WR-06",
                        "seed": 0,
                        "attempt": 0,
                        "success": True,
                        "episode_identity": "recorded-zero",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    episodes = discover_recorded_episodes(
        tmp_path,
        "WR-06",
        set(),
        source_status="all",
    )

    assert [(item["seed"], item["source_status"]) for item in episodes] == [
        (0, "success"),
        (1, "failed"),
    ]
    assert episodes[0]["episode_dir"] == accepted
    assert episodes[0]["episode_identity"] == "recorded-zero"
    assert episodes[1]["episode_dir"] == failed


def test_recording_discovery_status_filter_selects_failed_copy(
    tmp_path: Path,
) -> None:
    _episode(tmp_path, "train", seed=3)
    failed = _episode(tmp_path, "debug_failed", seed=3)

    episodes = discover_recorded_episodes(
        tmp_path,
        "WR-06",
        {3},
        source_status="failed",
    )

    assert len(episodes) == 1
    assert episodes[0]["episode_dir"] == failed
    assert episodes[0]["source_success"] is False


def test_recording_discovery_uses_manifest_episode_directory(
    tmp_path: Path,
) -> None:
    episode = tmp_path / "custom" / "recording"
    episode.mkdir(parents=True)
    (episode / "trajectory.h5").touch()
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "task_id": "WR-06",
                        "seed": 8,
                        "attempt": 2,
                        "success": True,
                        "episode_dir": str(episode),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    episodes = discover_recorded_episodes(
        tmp_path,
        "WR-06",
        set(),
        source_status="all",
    )

    assert len(episodes) == 1
    assert episodes[0]["seed"] == 8
    assert episodes[0]["episode_dir"] == episode


def test_replay_policy_returns_recorded_actions_without_skips() -> None:
    actions = np.arange(65, dtype=np.float32).reshape(5, 13)
    actions[:, 7] = 1.0
    client = ReplayPolicyClient(
        actions,
        contract=replay_policy_contract(horizon=2),
        chunk_size=2,
    )
    client.reset("episode")

    chunks = [
        client.infer(None, episode_id="episode", step=0).actions,
        client.infer(None, episode_id="episode", step=2).actions,
        client.infer(None, episode_id="episode", step=4).actions,
    ]

    np.testing.assert_array_equal(np.concatenate(chunks), actions)


def test_wr11_prompt_revision_preserves_strict_scene_and_seed_identity() -> None:
    from dataclasses import replace

    from bench.core.identity import canonical_hash
    from bench.evaluation.replay import _wr11_recording_prompt_identity_matches
    from bench.tasks import TaskCatalog
    from bench.tasks.contracts import EpisodeRequest

    episode = TaskCatalog.resolve(
        EpisodeRequest(task_id="WR-11", seed=36, allow_partial_calibration=False)
    )
    payload = episode.identity_payload()
    payload["public"]["prompt"] = (
        "Observe the sample milk carton on the dining island and remember its packaging. "
        "Move sideways to the stock area at the other end of the island, pick the carton with the same "
        "packaging, and bring it back to the empty tray beside the sample. "
        "Choose correctly on your first pickup. Leave the sample in place."
    )
    recorded = canonical_hash(payload)
    assert _wr11_recording_prompt_identity_matches(episode, recorded)
    assert not _wr11_recording_prompt_identity_matches(replace(episode, seed=37), recorded)
    assert not _wr11_recording_prompt_identity_matches(
        replace(episode, resource_hash="changed"), recorded
    )
    assert not _wr11_recording_prompt_identity_matches(replace(episode, facts={}), recorded)
    assert not _wr11_recording_prompt_identity_matches(replace(episode, task_id="WR-08"), recorded)
    assert not _wr11_recording_prompt_identity_matches(episode, "incorrect")
