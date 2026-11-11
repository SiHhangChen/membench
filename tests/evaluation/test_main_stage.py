from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.evaluation.main_stage import (
    import_main_rollout_stage,
    main_rollout_stage_result,
)
from bench.evaluation.stage_artifacts import StageArtifactWriter
from bench.evaluation.stage_rollout import FAILURE, SUCCESS


def _episode(*, seed: int = 1050, first_pass_step: int | None = 565) -> dict:
    return {
        "task_id": "WA-05",
        "seed": seed,
        "status": "failure",
        "failure_reason": "max_steps",
        "policy_actions_consumed": 1300,
        "policy_calls": 33,
        "report_stages": {
            "task_id": "WA-05",
            "stages": {
                "cup_filled": {
                    "applicable": True,
                    "passed": first_pass_step is not None,
                    "first_pass_step": first_pass_step,
                }
            },
        },
    }


def test_expert_controlled_stages_cannot_be_imported_as_policy_scores():
    episode = _episode()
    episode["episode_metadata"] = {"expert_prefix": {"valid": True, "handoff_step": 733}}
    with pytest.raises(ValueError, match="expert-controlled"):
        main_rollout_stage_result(
            episode, task_id="WA-05", target_stage="cup_filled", model_step_budget=1200,
        )


def test_main_rollout_stage_result_applies_stage_budget() -> None:
    within = main_rollout_stage_result(
        _episode(first_pass_step=565),
        task_id="WA-05",
        target_stage="cup_filled",
        model_step_budget=1200,
        replan_steps=40,
    )
    assert within.status == SUCCESS
    assert within.success_step == 565
    assert within.model_control_steps == 565
    assert within.environment_steps == 565
    assert within.total_model_inference_calls == 15

    late = main_rollout_stage_result(
        _episode(first_pass_step=1201),
        task_id="WA-05",
        target_stage="cup_filled",
        model_step_budget=1200,
        replan_steps=40,
    )
    assert late.status == FAILURE
    assert late.failure_reason == "model_step_budget"
    assert late.model_control_steps == 1200
    assert late.environment_steps == 1200
    assert late.total_model_inference_calls == 30

    missing = main_rollout_stage_result(
        _episode(first_pass_step=None),
        task_id="WA-05",
        target_stage="cup_filled",
        model_step_budget=1200,
        replan_steps=40,
    )
    assert missing.status == FAILURE
    assert missing.failure_reason == "model_step_budget"
    assert missing.environment_steps == 1200
    assert missing.total_model_inference_calls == 30


def test_main_rollout_stage_result_supports_strict_diagnostic_stage() -> None:
    episode = _episode(first_pass_step=None)
    episode["diagnostics"] = {
        "stages": {
            "first_cup_pick": {
                "qualified": True,
                "qualified_step": 317,
            }
        }
    }

    result = main_rollout_stage_result(
        episode,
        task_id="WA-05",
        target_stage="first_cup_pick",
        model_step_budget=1200,
        replan_steps=50,
    )

    assert result.status == SUCCESS
    assert result.success_step == 317
    assert result.total_model_inference_calls == 7


def test_import_main_rollout_stage_validates_identity_and_commits(tmp_path: Path) -> None:
    main_run = tmp_path / "main"
    episode_dir = main_run / "episodes/seed_001050"
    episode_dir.mkdir(parents=True)
    contract = {
        "policy": {"name": "groot", "family": "gr00t", "config": "groot"},
        "checkpoint": {"fingerprint": "sha256:checkpoint", "uri": "file:///old"},
        "action": {"dim": 13, "horizon": 40},
        "observation": {"state_dim": 37},
        "control_frequency_hz": 20.0,
    }
    (main_run / "run_manifest.json").write_text(
        json.dumps(
            {
                "task": {"id": "WA-05"},
                "suite": {"hash": "sha256:base"},
                "policy_contract": contract,
            }
        ),
        encoding="utf-8",
    )
    (episode_dir / "result.json").write_text(
        json.dumps(_episode()),
        encoding="utf-8",
    )

    writer = StageArtifactWriter(tmp_path / "stage")
    imported = import_main_rollout_stage(
        writer,
        main_run,
        task_id="WA-05",
        target_stage="cup_filled",
        model_step_budget=1200,
        replan_steps=40,
        seeds=(1050,),
        expected_suite_hash="sha256:base",
        expected_policy_contract={
            **contract,
            "checkpoint": {
                "fingerprint": "sha256:checkpoint",
                "uri": "file:///new-location",
            },
        },
    )
    assert [result.status for result in imported] == [SUCCESS]
    payload = json.loads(
        (
            writer.run_dir
            / "stages/cup_filled/episodes/seed_001050/result.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["source"] == "main_rollout"
    assert payload["evaluator_metadata"]["main_rollout_result_sha256"].startswith(
        "sha256:"
    )

    with pytest.raises(ValueError, match="policy identity"):
        import_main_rollout_stage(
            StageArtifactWriter(tmp_path / "bad-stage"),
            main_run,
            task_id="WA-05",
            target_stage="cup_filled",
            model_step_budget=1200,
            replan_steps=40,
            seeds=(1050,),
            expected_policy_contract={
                **contract,
                "checkpoint": {"fingerprint": "sha256:different"},
            },
        )
