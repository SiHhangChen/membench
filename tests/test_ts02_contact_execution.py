"""Regression: physical drawer motion cannot hide a failed arm execution."""
from types import SimpleNamespace

import numpy as np
import pytest

from bench.tasks.ts02.execution import manipulation


@pytest.mark.parametrize("failed_stage", [None, "prehook", "hook", "pull"])
def test_open_requires_all_contact_executions_to_pass(monkeypatch, failed_stage):
    fixture = SimpleNamespace(name="drawer", rot=np.pi, door_joint_names=["slide"],
                              handle_name="handle", open_fraction=lambda env: 1.0)
    env = SimpleNamespace(
        robots=[object()], cabinet_for_slot=lambda name: fixture,
        sim=SimpleNamespace(model=SimpleNamespace(geom_name2id=lambda name: 0),
                            data=SimpleNamespace(geom_xpos=np.zeros((1, 3)))),
    )
    monkeypatch.setattr(manipulation, "_joint_qpos", lambda *args: 0.2)
    monkeypatch.setattr(manipulation, "_locked_base_qpos", lambda env: None)
    monkeypatch.setattr(manipulation, "_restore_locked_base", lambda env, base: {})
    monkeypatch.setattr(manipulation, "_robot_fixture_contacts", lambda *args: [{}])
    monkeypatch.setattr(manipulation, "matrix_from_site_pose", lambda *args: np.eye(4))
    monkeypatch.setattr(manipulation, "matrix_from_base_obs", lambda *args: np.eye(4))
    monkeypatch.setattr(manipulation, "_drawer_pull_targets", lambda *args: (
        {k: np.eye(4) for k in ("prehook", "hook", "pull")}, {}))
    calls = []

    def run(target, *, stage, post_step_callback, **kwargs):
        name = stage.removeprefix("ts02_drawer_")
        calls.append(name)
        post_step_callback({})
        return {"pass": name != failed_stage, "plan": {"status": "Success"},
                "execution": {"executed_steps": 1}}

    controller = SimpleNamespace(
        env=env, runtime_config=SimpleNamespace(execute=True), obs={},
        refresh_obs=lambda: None, _sync_obs=lambda obs: None,
        stage_executor=SimpleNamespace(run=run, obs={}),
    )
    result = manipulation.open_drawer_physically(
        controller, fixture_name="drawer",
        config=manipulation.DrawerPullConfig(gripper_steps=0),
    )
    assert result["drawer_open_fraction_final"] == 1.0
    assert result["pass"] is (failed_stage is None)
    assert calls == {None: ["prehook", "hook", "pull"],
                     "prehook": ["prehook"], "hook": ["prehook", "hook"],
                     "pull": ["prehook", "hook", "pull"]}[failed_stage]


@pytest.mark.parametrize("execution_pass", [False, True])
def test_close_requires_push_execution_to_pass(monkeypatch, execution_pass):
    fixture = SimpleNamespace(name="drawer", rot=np.pi, door_joint_names=["slide"],
                              open_fraction=lambda env: 0.0)
    env = SimpleNamespace(robots=[object()], cabinet_for_slot=lambda name: fixture)
    monkeypatch.setattr(manipulation, "_joint_qpos", lambda *args: 0.0)
    monkeypatch.setattr(manipulation, "_locked_base_qpos", lambda env: None)
    monkeypatch.setattr(manipulation, "_restore_locked_base", lambda env, base: {})
    monkeypatch.setattr(manipulation, "_robot_fixture_contacts", lambda *args: [{}])
    monkeypatch.setattr(manipulation, "matrix_from_site_pose", lambda *args: np.eye(4))
    monkeypatch.setattr(manipulation, "matrix_from_base_obs", lambda *args: np.eye(4))

    def run(target, *, post_step_callback, **kwargs):
        post_step_callback({})
        return {"pass": execution_pass, "plan": {"status": "Success"},
                "execution": {"executed_steps": 1}}

    controller = SimpleNamespace(
        env=env, runtime_config=SimpleNamespace(execute=True), obs={},
        refresh_obs=lambda: None, _sync_obs=lambda obs: None,
        stage_executor=SimpleNamespace(run=run, obs={}),
    )
    result = manipulation.close_drawer_physically(
        controller, fixture_name="drawer", config=manipulation.DrawerPushCloseConfig())
    assert result["drawer_open_fraction_final"] == 0.0
    assert result["pass"] is execution_pass
