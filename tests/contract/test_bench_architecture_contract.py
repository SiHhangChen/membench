from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from bench.tasks import TaskCatalog
from bench.tasks.contracts import EpisodeRequest, ResolvedEpisode
from bench.tasks.layout import (
    REQUIRED_RESOURCE_FILES,
    TASK_PACKAGE_LAYOUT_VERSION,
    validate_placeholder_resources,
    validate_task_package,
)


TASK_IDS = TaskCatalog.ids()


@pytest.mark.contract
def test_catalog_has_exactly_twenty_four_stable_tasks() -> None:
    assert len(TASK_IDS) == 24
    assert len(set(TASK_IDS)) == 24
    assert TASK_IDS == (
        "TM01", "TM02", "TM03", "TM04", "TS01", "TS02", "TS03", "TS04",
        "WA-01", "WA-02", "WA-03", "WA-04", "WA-05", "WR-01", "WR-02", "WR-03", "WR-04",
        "WR-05", "WR-06", "WR-07", "WR-08", "WR-11", "WX-01", "WX-02",
    )


@pytest.mark.contract
def test_leaf_task_layout_and_public_exports_are_uniform() -> None:
    root = Path("bench/tasks")
    for task_id in TASK_IDS:
        package = task_id.lower().replace("-", "")
        path = root / package
        assert validate_task_package(path) == []
        validate_placeholder_resources(path)
        module = importlib.import_module(f"bench.tasks.{package}")
        assert module.TASK.task_id == task_id
        assert module.TASK.version == TaskCatalog.get(task_id).version
        for name in (
            "config", "episode", "environment", "initialization", "observation",
            "semantics", "prompts", "expert.spec", "expert.stations", "expert.targets", "expert.operations",
        ):
            assert importlib.import_module(f"bench.tasks.{package}.{name}")
    assert TASK_PACKAGE_LAYOUT_VERSION.endswith("/v1")
    assert len(REQUIRED_RESOURCE_FILES) == 7


@pytest.mark.contract
@pytest.mark.parametrize("task_id", TASK_IDS)
def test_new_definition_resolves_serializable_episode(task_id: str) -> None:
    definition = TaskCatalog.get(task_id)
    first = definition.resolve(EpisodeRequest(task_id=task_id, seed=17, variant="memory"))
    second = definition.resolve(EpisodeRequest(task_id=task_id, seed=17, variant="memory"))
    assert isinstance(first, ResolvedEpisode)
    assert first == second
    assert first.identity == second.identity
    assert first.to_dict()["identity"] == first.identity


@pytest.mark.contract
def test_leaf_tasks_do_not_import_each_other() -> None:
    root = Path("bench/tasks")
    packages = {task_id.lower().replace("-", "") for task_id in TASK_IDS}
    for package in packages:
        for source in (root / package).rglob("*.py"):
            text = source.read_text(encoding="utf-8")
            for other in packages - {package}:
                assert f"bench.tasks.{other}" not in text
