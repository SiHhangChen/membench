from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TASK_NAMES = (
    "TM01",
    "TM02",
    "TM03",
    "TM04",
    "TS01",
    "TS02",
    "TS03",
    "TS04",
    "WA01",
    "WA02",
    "WA03",
    "WA04",
    "WA05",
    "WR01",
    "WR02",
    "WR03",
    "WR04",
    "WR05",
    "WR06",
    "WR07",
    "WR08",
    "WR11",
    "WX01",
    "WX02",
)
EXPECTED_TASK_IDS = {
    "TM01",
    "TM02",
    "TM03",
    "TM04",
    "TS01",
    "TS02",
    "TS03",
    "TS04",
    "WA-01",
    "WA-02",
    "WA-03",
    "WA-04",
    "WA-05",
    "WR-01",
    "WR-02",
    "WR-03",
    "WR-04",
    "WR-05",
    "WR-06",
    "WR-07",
    "WR-08",
    "WR-11",
    "WX-01",
    "WX-02",
}


def test_legacy_task_directories_are_absent() -> None:
    assert list((ROOT / "bench" / "tasks").glob("hs_*")) == []
    assert list((ROOT / "configs" / "tasks").glob("hs_*")) == []


def test_builtin_catalog_contains_only_canonical_tasks() -> None:
    code = """
import json
from bench.tasks import TaskCatalog
print(json.dumps(TaskCatalog.ids()))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert set(json.loads(result.stdout)) == EXPECTED_TASK_IDS


def test_all_canonical_task_resources_resolve_without_legacy_paths() -> None:
    task_root = (ROOT / "bench" / "tasks").resolve()
    expected_missing = {
        ("TM03", "toaster_oven_time_pinch_on_pose"),
        ("TM03", "toaster_oven_time_pinch_on_for_off_pose"),
    }
    actual_missing = set()
    for name in TASK_NAMES:
        package = name.lower()
        config_module = importlib.import_module(f"bench.tasks.{package}.config")
        config = config_module.load_task_config()
        for resource_name, value in config.resources.items():
            path = config.path(value)
            assert path is not None, (name, resource_name)
            resolved = path.resolve()
            if not resolved.is_file():
                actual_missing.add((name, resource_name))
                continue
            if resolved.is_relative_to(task_root):
                assert not resolved.relative_to(task_root).parts[0].startswith("new_")
    assert actual_missing == expected_missing


def test_composite_tasks_own_local_copied_resources() -> None:
    tm01 = importlib.import_module("bench.tasks.tm01.config").load_task_config()
    assert tm01.path(tm01.resources["drawer_grasp_pose"]).is_relative_to(ROOT / "bench/tasks/tm01")

    tm03 = importlib.import_module("bench.tasks.tm03.config").load_task_config()
    assert all(tm03.path(tm03.resources[name]).is_relative_to(ROOT / "bench/tasks/tm03") for name in ("stove_knob_interaction_poses", "drawer_grasp_pose", "manual_base_path"))

    ts04 = importlib.import_module("bench.tasks.ts04.config").load_task_config()
    assert ts04.path(ts04.resources["drawer_grasp_pose"]).is_relative_to(ROOT / "bench/tasks/ts04")

    tm02 = importlib.import_module("bench.tasks.tm02.config").load_task_config()
    assert all(tm02.path(tm02.resources[name]).is_relative_to(ROOT / "bench/tasks/tm02") for name in ("water_button_positions", "water_station_poses", "can_grasp_pose"))

    wr03 = importlib.import_module("bench.tasks.wr03.config").load_task_config()
    for name in (
        "lower_door_grasp_pose",
        "lower_retrieval_prepose",
        "upper_door_grasp_pose",
    ):
        assert wr03.path(wr03.resources[name]).is_relative_to(ROOT / "bench/tasks/wr03")
    assert wr03.path(wr03.resources["stations"]).is_relative_to(ROOT / "bench/tasks/wr03")
    assert wr03.path(wr03.resources["manual_base_path"]).is_relative_to(ROOT / "bench/tasks/wr03")

    wr04 = importlib.import_module("bench.tasks.wr04.config").load_task_config()
    assert all(wr04.path(wr04.resources[name]).is_relative_to(ROOT / "bench/tasks/wr04") for name in ("item_shelf_grasp", "sink_stage_place", "stations"))
