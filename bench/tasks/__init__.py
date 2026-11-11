"""Builtin canonical task descriptors and opt-in environment registration.

Importing this package stays lightweight: the descriptor table contains only
metadata, paths, and lazy canonical import references.
"""

from __future__ import annotations

from pathlib import Path

from bench.registry import (
    TaskPlugin,
    get_task_plugin,
    load_task_plugin,
    register_task_plugin,
    resolve_task_plugin,
)
from bench.tasks.catalog import (
    TASK_CATALOG_VERSION,
    TaskCatalog,
    get_task_definition,
    list_task_definitions,
)


_TASKS_ROOT = Path(__file__).resolve().parent


def _builtin_task_plugin(
    canonical_id: str,
    *,
    aliases: tuple[str, ...],
    env_name: str,
) -> TaskPlugin:
    leaf_package = canonical_id.lower().replace("-", "")
    package = f"bench.tasks.{leaf_package}"
    return TaskPlugin(
        canonical_id=canonical_id,
        aliases=aliases,
        env_name=env_name,
        default_task_config=(
            _TASKS_ROOT / leaf_package / "resources" / "task.yaml"
        ),
        environment_module=f"{package}.environment",
        runtime_api_symbol=f"{package}.runtime:TASK_RUNTIME_API",
        expert_plan_symbol=f"{package}.expert.spec:build_expert_plan",
    )


# This is the single source of truth for the task packages shipped by bench.
BUILTIN_TASK_PLUGINS: tuple[TaskPlugin, ...] = (
    _builtin_task_plugin(
        "TM01",
        aliases=("TM-01", "TM_01"),
        env_name="TM01BoilWaterAndInspectTwoDrawers",
    ),
    _builtin_task_plugin(
        "TM02",
        aliases=("TM-02", "TM_02"),
        env_name="TM02FillPotAndRetrieveCan",
    ),
    _builtin_task_plugin(
        "TM03",
        aliases=("TM-03", "TM_03"),
        env_name="TM03BoilWaterInspectDrawersAndRunToasterOven",
    ),
    _builtin_task_plugin(
        "TM04",
        aliases=("TM-04", "TM_04"),
        env_name="TM04FillPotRetrieveCanAndAddTwoSugars",
    ),
    _builtin_task_plugin(
        "TS01",
        aliases=("TS-01", "TS_01"),
        env_name="TS01AddKSugarToHotChocolate",
    ),
    _builtin_task_plugin(
        "TS02",
        aliases=("TS-02", "TS_02"),
        env_name="TS02InspectThreeCabinets",
    ),
    _builtin_task_plugin(
        "TS03",
        aliases=("TS-03", "TS_03"),
        env_name="TS03HierarchicalAddKSugarToHotChocolate",
    ),
    _builtin_task_plugin(
        "TS04",
        aliases=("TS-04", "TS_04"),
        env_name="TS04InspectTwoDrawers",
    ),
    _builtin_task_plugin(
        "WA-01",
        aliases=("WA01", "WA_01"),
        env_name="WA01SortSugaredAndUnsugaredCoffee",
    ),
    _builtin_task_plugin(
        "WA-02",
        aliases=("WA02", "WA_02"),
        env_name="WA02ServeHotColdWaterCups",
    ),
    _builtin_task_plugin(
        "WA-03",
        aliases=("WA03", "WA_03"),
        env_name="WA03SortSugaredAndUnsugaredCoffee",
    ),
    _builtin_task_plugin(
        "WA-04",
        aliases=("WA04", "WA_04"),
        env_name="WA04ServeUniqueTemperatureCupAmongThree",
    ),
    _builtin_task_plugin(
        "WA-05",
        aliases=("WA05", "WA_05"),
        env_name="WA05FillOneCupAndMatchTemperatureButton",
    ),
    _builtin_task_plugin(
        "WR-01",
        aliases=("WR01", "WR_01"),
        env_name="WR01RetrieveRememberedCan",
    ),
    _builtin_task_plugin(
        "WR-02",
        aliases=("WR02", "WR_02"),
        env_name="WR02ReturnCondimentPair",
    ),
    _builtin_task_plugin(
        "WR-03",
        aliases=("WR03", "WR_03"),
        env_name="WR03StoreAndRetrieveRememberedCan",
    ),
    _builtin_task_plugin(
        "WR-04",
        aliases=("WR04", "WR_04"),
        env_name="WR04ReturnSingleObjectToOriginalLevel",
    ),
    _builtin_task_plugin(
        "WR-05",
        aliases=("WR05", "WR_05"),
        env_name="WR05ReturnCondimentPair",
    ),
    _builtin_task_plugin(
        "WR-06",
        aliases=("WR06", "WR_06"),
        env_name="WR06ReturnSingleBottleToOriginalLevel",
    ),
    _builtin_task_plugin(
        "WR-07",
        aliases=("WR07", "WR_07"),
        env_name="WR07MatchMilkCartonToColoredSugarPlate",
    ),
    _builtin_task_plugin(
        "WR-08",
        aliases=("WR08", "WR_08"),
        env_name="WR08MatchMilkCartonsToColoredSugarPlates",
    ),
    _builtin_task_plugin(
        "WR-11",
        aliases=("WR11", "WR_11"),
        env_name="WR11FetchMatchingMilkCarton",
    ),
    _builtin_task_plugin(
        "WX-01",
        aliases=("WX01", "WX_01"),
        env_name="WX01ReturnObjectToOriginalPlace",
    ),
    _builtin_task_plugin(
        "WX-02",
        aliases=("WX02", "WX_02"),
        env_name="WX02CleanUnderCupAndReturnIt",
    ),
)


def register_builtin_task_plugins() -> tuple[TaskPlugin, ...]:
    """Register builtin descriptors without loading task implementation modules."""
    return tuple(register_task_plugin(plugin) for plugin in BUILTIN_TASK_PLUGINS)


def get_builtin_task_plugin(canonical_id: str) -> TaskPlugin:
    """Return a builtin descriptor by exact canonical id."""
    register_builtin_task_plugins()
    plugin = get_task_plugin(canonical_id)
    if plugin not in BUILTIN_TASK_PLUGINS:
        raise KeyError(f"task plugin {canonical_id!r} is not a builtin")
    return plugin


def resolve_builtin_task_plugin(name: str) -> TaskPlugin:
    """Resolve a builtin canonical id or spelling alias."""
    register_builtin_task_plugins()
    plugin = resolve_task_plugin(name)
    if plugin not in BUILTIN_TASK_PLUGINS:
        raise KeyError(f"task plugin {name!r} is not a builtin")
    return plugin


def list_builtin_task_plugins() -> list[TaskPlugin]:
    """Return the builtin descriptors in declaration order."""
    register_builtin_task_plugins()
    return list(BUILTIN_TASK_PLUGINS)


def load_builtin_task_plugin(
    name: str,
    *,
    runtime: bool = False,
    expert: bool = False,
    environment: bool = False,
    evaluation: bool = False,
) -> TaskPlugin:
    """Register builtins, resolve ``name``, and load requested lazy components."""
    register_builtin_task_plugins()
    plugin = resolve_builtin_task_plugin(name)
    return load_task_plugin(
        plugin.canonical_id,
        runtime=runtime,
        expert=expert,
        environment=environment,
        evaluation=evaluation,
    )


def load_builtin_task_plugins() -> tuple[TaskPlugin, ...]:
    """Compatibility spelling for loading all builtin descriptor metadata."""
    return register_builtin_task_plugins()


def register_builtin_robocasa_tasks() -> None:
    """Register bench-owned RoboCasa task classes shipped with bench."""
    from bench.env.robocasa_runtime import bootstrap_local_robocasa_sources

    bootstrap_local_robocasa_sources()
    for plugin in register_builtin_task_plugins():
        plugin.load_environment_module()


# Make a direct import of ``bench.tasks.<id>.environment`` work from a source
# checkout while keeping catalog discovery simulator-free.  This only adds the
# vendored source roots to ``sys.path``; it does not import RoboCasa itself.
import sys as _sys

for _project in ("robosuite", "robocasa"):
    _source_root = _TASKS_ROOT.parents[1] / _project
    if (_source_root / _project / "__init__.py").is_file():
        _source_text = str(_source_root)
        if _source_text not in _sys.path:
            _sys.path.insert(0, _source_text)
del _project, _source_root, _sys
if "_source_text" in globals():
    del _source_text


__all__ = [
    "BUILTIN_TASK_PLUGINS",
    "get_builtin_task_plugin",
    "list_builtin_task_plugins",
    "load_builtin_task_plugin",
    "load_builtin_task_plugins",
    "register_builtin_robocasa_tasks",
    "register_builtin_task_plugins",
    "resolve_builtin_task_plugin",
    "TASK_CATALOG_VERSION",
    "TaskCatalog",
    "get_task_definition",
    "list_task_definitions",
]
