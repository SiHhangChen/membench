"""The one task discovery boundary for MobileMemBench.

Catalog import is intentionally cheap: it stores stable descriptors and lazy
module references only.  RoboCasa classes, recording builders, and policy
adapters are never registered here.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from bench.core.identity import canonical_hash
from bench.tasks.contracts import (
    EpisodeRequest,
    ExpertPlanInputs,
    ResolvedEpisode,
    TaskDefinition,
)


TASK_CATALOG_VERSION = "membench.task_catalog/v1"
_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    task_id: str
    package: str
    env_name: str
    version: str
    default_horizon: int


_ENTRIES: tuple[CatalogEntry, ...] = (
    CatalogEntry("TM01", "tm01", "TM01BoilWaterAndInspectTwoDrawers", "1.0.0", 3200),
    CatalogEntry("TM02", "tm02", "TM02FillPotAndRetrieveCan", "1.0.0", 3600),
    CatalogEntry("TM03", "tm03", "TM03BoilWaterInspectDrawersAndRunToasterOven", "1.0.0", 4800),
    CatalogEntry("TM04", "tm04", "TM04FillPotRetrieveCanAndAddTwoSugars", "1.0.0", 5200),
    CatalogEntry("TS01", "ts01", "TS01AddKSugarToHotChocolate", "1.0.0", 2800),
    CatalogEntry("TS02", "ts02", "TS02InspectThreeCabinets", "1.1.0", 2800),
    CatalogEntry("TS03", "ts03", "TS03HierarchicalAddKSugarToHotChocolate", "1.0.0", 2800),
    CatalogEntry("TS04", "ts04", "TS04InspectTwoDrawers", "1.1.0", 1800),
    CatalogEntry("WA-01", "wa01", "WA01SortSugaredAndUnsugaredCoffee", "2.0.0", 5000),
    CatalogEntry("WA-02", "wa02", "WA02ServeHotColdWaterCups", "1.0.0", 6000),
    CatalogEntry("WA-03", "wa03", "WA03SortSugaredAndUnsugaredCoffee", "1.0.0", 6500),
    CatalogEntry("WA-04", "wa04", "WA04ServeUniqueTemperatureCupAmongThree", "1.0.0", 8500),
    CatalogEntry("WA-05", "wa05", "WA05FillOneCupAndMatchTemperatureButton", "1.0.0", 4000),
    CatalogEntry("WR-01", "wr01", "WR01RetrieveRememberedCan", "1.0.0", 15000),
    CatalogEntry("WR-02", "wr02", "WR02ReturnCondimentPair", "1.0.0", 5000),
    CatalogEntry("WR-03", "wr03", "WR03StoreAndRetrieveRememberedCan", "1.0.0", 15000),
    CatalogEntry("WR-04", "wr04", "WR04ReturnSingleObjectToOriginalLevel", "1.0.0", 3200),
    CatalogEntry("WR-05", "wr05", "WR05ReturnCondimentPair", "1.0.0", 5000),
    CatalogEntry("WR-06", "wr06", "WR06ReturnSingleBottleToOriginalLevel", "1.0.0", 3200),
    CatalogEntry("WR-07", "wr07", "WR07MatchMilkCartonToColoredSugarPlate", "3.0.0", 3600),
    CatalogEntry("WR-08", "wr08", "WR08MatchMilkCartonsToColoredSugarPlates", "2.0.0", 2000),
    CatalogEntry("WR-11", "wr11", "WR11FetchMatchingMilkCarton", "3.2.0", 3600),
    CatalogEntry("WX-01", "wx01", "WX01ReturnObjectToOriginalPlace", "1.0.0", 3200),
    CatalogEntry("WX-02", "wx02", "WX02CleanUnderCupAndReturnIt", "1.0.0", 5200),
)

_ALIASES: dict[str, str] = {}
for _entry in _ENTRIES:
    for _value in (_entry.task_id, _entry.package, f"{_entry.task_id[:2]}-{_entry.task_id[2:]}"):
        _ALIASES["".join(ch for ch in _value.casefold() if ch.isalnum())] = _entry.task_id


def _entry(task_id: str) -> CatalogEntry:
    key = "".join(ch for ch in str(task_id).casefold() if ch.isalnum())
    try:
        canonical = _ALIASES[key]
    except KeyError as exc:
        raise KeyError(f"unknown task id {task_id!r}") from exc
    return next(item for item in _ENTRIES if item.task_id == canonical)


def _resolve_episode(entry: CatalogEntry, request: EpisodeRequest) -> ResolvedEpisode:
    # The catalog dispatches to the canonical leaf package.  Importing the
    # resolver is lazy so discovery remains simulator-free.
    import importlib

    module = importlib.import_module(f"bench.tasks.{entry.package}.episode")
    resolver = getattr(module, "resolve_episode")
    return resolver(request)


def _expert_plan(entry: CatalogEntry, inputs: ExpertPlanInputs) -> Any:
    if not isinstance(inputs, ExpertPlanInputs):
        raise TypeError(
            "build_expert_plan() requires ExpertPlanInputs from a RuntimeDescriptor"
        )
    if inputs.episode.task_id != entry.task_id:
        raise ValueError(
            f"{entry.task_id} expert plan cannot build {inputs.episode.task_id!r}"
        )
    import importlib

    module = importlib.import_module(f"bench.tasks.{entry.package}.expert.spec")
    return module.build_expert_plan(inputs)


def _jsonable(value: Any) -> Any:
    from dataclasses import asdict, is_dataclass

    if hasattr(value, "value"):
        return _jsonable(value.value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted(_jsonable(item) for item in value)
    return repr(value)


def _definition(entry: CatalogEntry) -> TaskDefinition:
    metadata = _task_metadata(entry)
    return TaskDefinition(
        task_id=entry.task_id,
        version=entry.version,
        display_name=metadata["display_name"],
        env_name=entry.env_name,
        package=f"bench.tasks.{entry.package}",
        resource_root=_ROOT / f"{entry.package}" / "resources",
        resolve_episode=lambda request, item=entry: _resolve_episode(item, request),
        build_expert_plan=lambda episode, item=entry: _expert_plan(item, episode),
        success_conditions=metadata["success_conditions"],
        default_horizon=entry.default_horizon,
        metadata={"catalog_version": TASK_CATALOG_VERSION, "task_config": metadata["config_path"]},
    )


def _task_metadata(entry: CatalogEntry) -> dict[str, Any]:
    path = _ROOT / entry.package / "resources" / "task.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    task = payload.get("task", {}) if isinstance(payload, dict) else {}
    if not isinstance(task, dict):
        task = {}
    return {
        "config_path": str(path),
        "display_name": str(task.get("display_name", entry.task_id)),
        "success_conditions": tuple(str(item) for item in task.get("success_conditions", ()) or ()),
    }


class TaskCatalog:
    """Stable, explicit task discovery and definition loading."""

    version = TASK_CATALOG_VERSION

    @classmethod
    def entries(cls) -> tuple[CatalogEntry, ...]:
        return _ENTRIES

    @classmethod
    def ids(cls) -> tuple[str, ...]:
        return tuple(item.task_id for item in _ENTRIES)

    @classmethod
    def canonical_id(cls, task_id: str) -> str:
        return _entry(task_id).task_id

    @classmethod
    def get(cls, task_id: str) -> TaskDefinition:
        return _definition(_entry(task_id))

    @classmethod
    def resolve(cls, request: EpisodeRequest) -> ResolvedEpisode:
        canonical = cls.canonical_id(request.task_id)
        if request.task_id == canonical:
            normalized_request = request
        else:
            normalized_request = replace(request, task_id=canonical)
        return cls.get(canonical).resolve(normalized_request)


def get_task_definition(task_id: str) -> TaskDefinition:
    return TaskCatalog.get(task_id)


def list_task_definitions() -> tuple[TaskDefinition, ...]:
    return tuple(_definition(item) for item in _ENTRIES)


__all__ = [
    "CatalogEntry",
    "TASK_CATALOG_VERSION",
    "TaskCatalog",
    "get_task_definition",
    "list_task_definitions",
]
