from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.evaluation.suite import load_eval_suite
from bench.release import build_release_lock, resolve_release_entry, verify_release_lock
from bench.tasks import resolve_builtin_task_plugin


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.contract
@pytest.mark.parametrize(
    "task_id",
    [
        "TM01", "TM02", "TM03", "TM04", "TS01", "TS02", "TS03", "TS04",
        "WA-01", "WA-02", "WA-03", "WA-04", "WA-05", "WR-01", "WR-02", "WR-03", "WR-04",
        "WR-05", "WR-06", "WR-07", "WR-08", "WR-11", "WX-01", "WX-02",
    ],
)
def test_all_plugins_expose_canonical_runtime_and_expert_apis(task_id: str) -> None:
    plugin = resolve_builtin_task_plugin(task_id)

    api = plugin.load_runtime_api()
    assert callable(api.build_descriptor)
    assert callable(api.initialize_runtime)
    assert callable(api.check_success)
    assert callable(api.diagnostics)
    assert callable(api.close_runtime)
    assert callable(api.build_expert_runtime)
    assert callable(plugin.load_expert_plan_factory())


@pytest.mark.contract
@pytest.mark.parametrize(
    ("suite_name", "expected_horizon"),
    [
        ("tm01_v1.yaml", 3200),
        ("tm02_v1.yaml", 3600),
        ("ts01_v1.yaml", 2800),
        ("ts02_v1.yaml", 2800),
        ("ts03_v1.yaml", 1800),
        ("ts04_v1.yaml", 1800),
        ("wa01_v1.yaml", 5000),
        ("wa02_v1.yaml", 6000),
        ("wa03_v1.yaml", 6500),
        ("wa05_v1.yaml", 1600),
        ("wr01_v1.yaml", 2600),
        ("wr03_v1.yaml", 8000),
        ("wr05_v1.yaml", 5000),
        ("wr06_v1.yaml", 3200),
        ("wr07_v1.yaml", 1000),
        ("wr08_v1.yaml", 2000),
        ("wr11_v3.yaml", 3600),
        ("wx01_v1.yaml", 3200),
        ("wx02_v1.yaml", 5200),
    ],
)
def test_all_suites_use_expected_effective_horizon(
    suite_name: str,
    expected_horizon: int,
) -> None:
    suite = load_eval_suite(ROOT / "configs" / "eval" / "suites" / suite_name)

    assert suite.rollout.max_steps == expected_horizon
    assert suite.task_options["observation_profile_file"].endswith("rgbd_v3.yaml")


@pytest.mark.contract
@pytest.mark.release_gate
def test_release_lock_is_deterministic_and_tamper_evident() -> None:
    entries = [
        resolve_release_entry("WR-01", seed=0),
        resolve_release_entry("WX-01", seed=0),
    ]
    first = build_release_lock(entries, release_id="pilot_v1")
    second = build_release_lock(reversed(entries), release_id="pilot_v1")

    assert first == second
    verify_release_lock(first)
    tampered = json.loads(json.dumps(first))
    tampered["entries"][0]["max_horizon"] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_release_lock(tampered)
