from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from bench.registry import (
    TaskPlugin,
    get_task_plugin,
    list_task_plugin_ids,
    list_task_plugins,
    resolve_task_plugin,
)
from bench.tasks import (
    BUILTIN_TASK_PLUGINS,
    get_builtin_task_plugin,
    list_builtin_task_plugins,
    load_builtin_task_plugin,
    load_builtin_task_plugins,
    register_builtin_robocasa_tasks,
    register_builtin_task_plugins,
    resolve_builtin_task_plugin,
)


ROOT = Path(__file__).resolve().parents[1]
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


def test_builtin_descriptor_table_is_complete_and_self_contained() -> None:
    assert len(BUILTIN_TASK_PLUGINS) == len(EXPECTED_TASK_IDS) == 24
    assert {plugin.canonical_id for plugin in BUILTIN_TASK_PLUGINS} == EXPECTED_TASK_IDS

    for plugin in BUILTIN_TASK_PLUGINS:
        assert plugin.task_id == plugin.canonical_id
        assert plugin.aliases
        assert plugin.env_name
        assert plugin.default_task_config_file == plugin.default_task_config
        assert plugin.default_task_config.is_file()
        assert plugin.environment_module.endswith(".environment")
        assert plugin.runtime_api_symbol.endswith(".runtime:TASK_RUNTIME_API")
        assert plugin.expert_plan_symbol.endswith(".expert.spec:build_expert_plan")


def test_importing_and_registering_descriptors_remains_task_module_lazy() -> None:
    code = """
import json
import sys
from bench.tasks import load_builtin_task_plugins

before = sorted(name for name in sys.modules if name.startswith("bench.tasks.") and name.count(".") > 2)
plugins = load_builtin_task_plugins()
after = sorted(name for name in sys.modules if name.startswith("bench.tasks.") and name.count(".") > 2)
print(json.dumps({"before": before, "after": after, "ids": [p.canonical_id for p in plugins]}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["before"] == payload["after"] == []
    assert set(payload["ids"]) == EXPECTED_TASK_IDS


def test_register_get_resolve_and_list_plugin_apis() -> None:
    registered = register_builtin_task_plugins()
    assert registered == BUILTIN_TASK_PLUGINS
    assert load_builtin_task_plugins() == BUILTIN_TASK_PLUGINS
    assert list_builtin_task_plugins() == list(BUILTIN_TASK_PLUGINS)
    assert EXPECTED_TASK_IDS.issubset(list_task_plugin_ids())
    assert EXPECTED_TASK_IDS.issubset(
        {plugin.canonical_id for plugin in list_task_plugins()}
    )

    assert get_task_plugin("WR-04") is get_builtin_task_plugin("WR-04")
    assert resolve_task_plugin("wr_04").canonical_id == "WR-04"
    assert resolve_task_plugin("WR04").canonical_id == "WR-04"
    assert resolve_builtin_task_plugin("tm-01").canonical_id == "TM01"
    assert resolve_builtin_task_plugin("tm_03").canonical_id == "TM03"

    with pytest.raises(KeyError, match="not found"):
        get_task_plugin("wr_04")
    with pytest.raises(KeyError, match="not found"):
        resolve_task_plugin("not-a-builtin-task")


def test_selected_runtime_and_expert_components_load_through_descriptor() -> None:
    plugin = load_builtin_task_plugin("wa_01", runtime=True, expert=True)
    assert plugin.canonical_id == "WA-01"
    assert callable(plugin.load_runtime_api().build_descriptor)
    assert callable(plugin.load_expert_plan_factory())


def test_legacy_robocasa_registration_uses_plugin_environment_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []
    monkeypatch.setattr(
        "bench.env.robocasa_runtime.bootstrap_local_robocasa_sources",
        lambda: None,
    )
    monkeypatch.setattr(
        TaskPlugin,
        "load_environment_module",
        lambda self: imported.append(self.environment_module),
    )

    register_builtin_robocasa_tasks()

    assert imported == [plugin.environment_module for plugin in BUILTIN_TASK_PLUGINS]


def test_evaluation_load_request_uses_the_shared_adapter_module() -> None:
    plugin = load_builtin_task_plugin("wx01", evaluation=True)
    assert plugin.canonical_id == "WX-01"
