"""Task-level evaluation tests for WA-05."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import bench.tasks.wa05.environment as wa05_environment
import bench.tasks.wa05.runtime as wa05_runtime
from bench.tasks.wa05.domain.contracts import CUP_IDS, SERVICE_SLOTS
from bench.tasks.wa05.domain.semantics import SemanticState
from bench.tasks.wa05.evaluation import (
    REPORT_STAGE_ORDER,
    WA05EvaluationSnapshot,
    WA05Evaluator,
    diagnostics_for_runtime,
    report_stages_for_runtime,
    success_for_runtime,
)


CUP_ID = CUP_IDS[0]
SERVICE_SLOT = SERVICE_SLOTS[0]


def _snapshot(
    step: int,
    *,
    filled: str | None = None,
    staged: bool = False,
    pressed: str | None = None,
    success: bool = False,
    failure_code: str | None = None,
) -> WA05EvaluationSnapshot:
    return WA05EvaluationSnapshot(
        step_id=step,
        cup_id=CUP_ID,
        service_slot=SERVICE_SLOT,
        filled_temperature=filled,
        staged_before_answer=staged,
        pressed_temperature=pressed,
        domain_success=success,
        absorbing_failure=failure_code is not None,
        failure_code=failure_code,
    )


def _evaluator() -> WA05Evaluator:
    evaluator = WA05Evaluator(cup_id=CUP_ID, service_slot=SERVICE_SLOT)
    evaluator.reset(_snapshot(0))
    return evaluator


@pytest.mark.parametrize("temperature", ("hot", "cold"))
def test_three_stage_funnel_uses_actual_filled_temperature(
    temperature: str,
) -> None:
    evaluator = _evaluator()

    filled = evaluator.observe(_snapshot(1, filled=temperature))
    assert filled.stages["cup_filled"].qualified
    assert filled.stages["cup_filled"].qualified_step == 1
    assert not filled.stages["cup_staged"].qualified
    assert not filled.success

    staged = evaluator.observe(
        _snapshot(2, filled=temperature, staged=True)
    )
    assert staged.stages["cup_filled"].qualified
    assert staged.stages["cup_staged"].qualified
    assert staged.stages["cup_staged"].qualified_step == 2
    assert not staged.success

    completed = evaluator.observe(
        _snapshot(
            3,
            filled=temperature,
            staged=True,
            pressed=temperature,
            success=True,
        )
    )
    assert completed.success
    assert completed.terminal
    assert completed.stages["success"].qualified_step == 3


def test_wrong_temperature_button_preserves_completed_work_stages() -> None:
    evaluator = _evaluator()
    evaluator.observe(_snapshot(1, filled="hot"))
    evaluator.observe(_snapshot(2, filled="hot", staged=True))

    result = evaluator.observe(
        _snapshot(
            3,
            filled="hot",
            staged=True,
            pressed="cold",
            failure_code="wrong_temperature_button",
        )
    )

    assert result.stages["cup_filled"].qualified
    assert result.stages["cup_staged"].qualified
    assert not result.stages["success"].qualified
    assert result.stages["success"].invalidated
    assert result.stages["success"].reason == "wrong_temperature_button"
    assert result.failure_code == "wrong_temperature_button"
    assert not result.success
    assert result.terminal


def test_early_answer_blocks_all_report_stages() -> None:
    evaluator = _evaluator()

    result = evaluator.observe(
        _snapshot(1, failure_code="button_pressed_before_fill")
    )

    assert not result.stages["cup_filled"].qualified
    assert not result.stages["cup_staged"].qualified
    assert not result.stages["success"].qualified
    assert result.stages["success"].invalidated
    assert result.failure_code == "button_pressed_before_fill"


def test_later_second_button_revokes_episode_success() -> None:
    evaluator = _evaluator()
    evaluator.observe(_snapshot(1, filled="cold"))
    evaluator.observe(_snapshot(2, filled="cold", staged=True))
    evaluator.observe(
        _snapshot(
            3,
            filled="cold",
            staged=True,
            pressed="cold",
            success=True,
        )
    )

    result = evaluator.observe(
        _snapshot(
            4,
            filled="cold",
            staged=True,
            pressed="cold",
            failure_code="multiple_answer_buttons",
        )
    )

    assert result.stages["cup_filled"].qualified
    assert result.stages["cup_staged"].qualified
    assert result.stages["success"].observed
    assert not result.stages["success"].qualified
    assert result.stages["success"].invalidated
    report = result.to_report_stages()
    assert report["stages"]["success"]["passed"] is False
    assert report["stages"]["success"]["first_pass_step"] is None


def test_public_report_contains_only_three_semantic_stages() -> None:
    evaluator = _evaluator()
    evaluator.observe(_snapshot(1, filled="hot"))
    evaluator.observe(_snapshot(2, filled="hot", staged=True))
    result = evaluator.observe(
        _snapshot(
            3,
            filled="hot",
            staged=True,
            pressed="hot",
            success=True,
        )
    )

    report = result.to_report_stages()

    assert report["profile_id"] == "WA-05/report-v1"
    assert report["order"] == list(REPORT_STAGE_ORDER)
    assert tuple(report["stages"]) == REPORT_STAGE_ORDER
    assert "button_pressed" not in report["stages"]
    assert "button_correct" not in report["stages"]
    assert report["stages"] == {
        "cup_filled": {
            "applicable": True,
            "passed": True,
            "first_pass_step": 1,
        },
        "cup_staged": {
            "applicable": True,
            "passed": True,
            "first_pass_step": 2,
        },
        "success": {
            "applicable": True,
            "passed": True,
            "first_pass_step": 3,
        },
    }


def test_snapshot_rejects_staging_without_fill() -> None:
    with pytest.raises(ValueError, match="requires a verified fill"):
        _snapshot(1, staged=True)


def test_same_step_is_idempotent_and_backward_step_is_rejected() -> None:
    evaluator = WA05Evaluator(cup_id=CUP_ID, service_slot=SERVICE_SLOT)
    evaluator.reset(_snapshot(1))
    initial = evaluator.result()

    assert evaluator.observe(_snapshot(1, filled="hot")) is initial
    with pytest.raises(ValueError, match="moved backwards"):
        evaluator.observe(_snapshot(0))


def test_environment_physical_event_updates_are_once_per_timestep(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        wa05_environment.Kitchen,
        "update_state",
        lambda self: None,
    )
    env = object.__new__(
        wa05_environment.WA05FillOneCupAndMatchTemperatureButton
    )
    env.timestep = 7
    env._last_task_event_update_step = None
    env._semantic_state = SemanticState()
    env._physical_event_diagnostics = {}
    calls = {"faucets": 0, "events": 0}
    env._refresh_sensor_faucets = lambda: calls.__setitem__(
        "faucets", calls["faucets"] + 1
    )
    env._update_task_events_from_physics = lambda: calls.__setitem__(
        "events", calls["events"] + 1
    )
    env._answer_button_press_edges = lambda: ()

    env.update_state()
    env.update_state()
    assert calls == {"faucets": 1, "events": 1}

    env.timestep = 8
    env.update_state()
    assert calls == {"faucets": 2, "events": 2}


def test_runtime_callbacks_share_one_evaluator_result() -> None:
    evaluator = _evaluator()
    snapshot = _snapshot(
        3,
        filled="hot",
        staged=True,
        pressed="hot",
        success=True,
    )

    class _Env:
        def evaluation_snapshot(self) -> WA05EvaluationSnapshot:
            return snapshot

    runtime = SimpleNamespace(
        env=_Env(),
        metadata={"task_evaluator": evaluator},
    )

    assert success_for_runtime(runtime)
    assert diagnostics_for_runtime(runtime)["success"]
    assert (
        report_stages_for_runtime(runtime)["stages"]["success"]["passed"]
        is True
    )


def test_runtime_initialize_mounts_fresh_evaluator(monkeypatch) -> None:
    calls: dict[str, object] = {"reset": 0}

    class _FakeEvaluator:
        def __init__(self, **kwargs: object) -> None:
            calls["kwargs"] = dict(kwargs)

        def reset(self, snapshot: object) -> None:
            calls["reset"] = int(calls["reset"]) + 1
            calls["snapshot"] = snapshot

    monkeypatch.setattr(wa05_runtime, "WA05Evaluator", _FakeEvaluator)
    monkeypatch.setattr(
        wa05_runtime,
        "apply_sampled_initial_object_regions",
        lambda env, settings, enabled: ({"obs": 1}, {"objects": True}),
    )
    monkeypatch.setattr(
        wa05_runtime,
        "apply_robot_base_initial_state",
        lambda env, updated, **kwargs: ({"obs": 2}, {"base": True}),
    )

    plan = SimpleNamespace(cup_id=CUP_ID, service_slot=SERVICE_SLOT, seed=7)

    class _FakeEnv:
        def configure_episode(self, episode: object) -> None:
            self.episode_plan = episode

        def configure_canonical_physics(self) -> None:
            calls["physics_configured"] = True

        def evaluation_snapshot(self) -> str:
            return "SNAPSHOT"

    descriptor = SimpleNamespace(
        config=SimpleNamespace(
            initial_state=SimpleNamespace(
                objects={"enabled": True},
                robot_base=object(),
            ),
        ),
        episode_plan=plan,
        navigation=object(),
        scene=SimpleNamespace(scene_seed=0),
    )
    env = _FakeEnv()

    updated, initialization, metadata, targets = wa05_runtime._initialize(
        descriptor,
        env,
        {},
    )

    assert updated == {"obs": 2}
    assert initialization == {
        "objects": {"objects": True},
        "base": {"base": True},
    }
    assert targets is None
    assert metadata["task_evaluator"] is env._wa05_evaluator
    assert calls == {
        "reset": 1,
        "snapshot": "SNAPSHOT",
        "kwargs": {"cup_id": CUP_ID, "service_slot": SERVICE_SLOT},
        "physics_configured": True,
    }


def test_runtime_registers_public_report_stage_projector() -> None:
    assert wa05_runtime.TASK_RUNTIME_API.report_stage_profile_id == "WA-05/report-v1"
    assert callable(wa05_runtime.TASK_RUNTIME_API.report_stages)
