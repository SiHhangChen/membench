"""Import first-stage verdicts from an existing standard evaluation run."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from bench.evaluation.stage_artifacts import StageArtifactWriter
from bench.evaluation.stage_rollout import FAILURE, SUCCESS, StageEvaluationResult


def import_main_rollout_stage(
    writer: StageArtifactWriter,
    main_run_dir: str | Path,
    *,
    task_id: str,
    target_stage: str,
    model_step_budget: int,
    replan_steps: int,
    seeds: Sequence[int],
    expected_suite_hash: str | None = None,
    expected_policy_contract: Mapping[str, Any] | None = None,
    resume: bool = True,
) -> list[StageEvaluationResult]:
    """Convert main-rollout report-stage verdicts into stage result artifacts."""

    run_dir = Path(main_run_dir).expanduser().resolve()
    manifest = _load_json_object(run_dir / "run_manifest.json")
    _validate_main_manifest(
        manifest,
        task_id=task_id,
        expected_suite_hash=expected_suite_hash,
        expected_policy_contract=expected_policy_contract,
    )
    results: list[StageEvaluationResult] = []
    for seed in tuple(int(value) for value in seeds):
        if resume and writer.completed(target_stage, seed):
            continue
        source_path = run_dir / "episodes" / f"seed_{seed:06d}" / "result.json"
        episode = _load_json_object(source_path)
        result = main_rollout_stage_result(
            episode,
            task_id=task_id,
            target_stage=target_stage,
            model_step_budget=model_step_budget,
            replan_steps=replan_steps,
        )
        if result.seed != seed:
            raise ValueError(
                f"main rollout result seed {result.seed} does not match path seed {seed}"
            )
        writer.write_result(
            result,
            source="main_rollout",
            evaluator_metadata={
                "main_rollout_run": str(run_dir),
                "main_rollout_result": str(source_path.relative_to(run_dir)),
                "main_rollout_result_sha256": _sha256(source_path),
                "main_rollout_status": episode.get("status"),
                "model_step_budget": int(model_step_budget),
                "replan_steps": int(replan_steps),
                "main_rollout_policy_actions_consumed": episode.get(
                    "policy_actions_consumed"
                ),
                "main_rollout_policy_calls": episode.get("policy_calls"),
            },
        )
        results.append(result)
    return results


def main_rollout_stage_result(
    episode: Mapping[str, Any],
    *,
    task_id: str,
    target_stage: str,
    model_step_budget: int,
    replan_steps: int = 1,
) -> StageEvaluationResult:
    """Apply a frozen stage budget to one standard-rollout stage verdict."""

    if str(episode.get("task_id")) != str(task_id):
        raise ValueError(
            f"main rollout task {episode.get('task_id')!r} does not match {task_id!r}"
        )
    metadata = episode.get("episode_metadata")
    if isinstance(metadata, Mapping) and metadata.get("expert_prefix") is not None:
        raise ValueError(
            "expert-controlled main rollout stages cannot be imported as policy stage scores; "
            "use a legacy v1 rollout for stage diagnostics"
        )
    status = str(episode.get("status"))
    if status not in {"success", "failure"}:
        raise ValueError(f"main rollout episode is not a completed test: status={status!r}")
    seed = episode.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"main rollout seed must be an integer: {seed!r}")
    if isinstance(model_step_budget, bool) or int(model_step_budget) <= 0:
        raise ValueError("model_step_budget must be a positive integer")
    budget = int(model_step_budget)
    if isinstance(replan_steps, bool) or int(replan_steps) <= 0:
        raise ValueError("replan_steps must be a positive integer")
    cadence = int(replan_steps)
    report = episode.get("report_stages")
    if not isinstance(report, Mapping):
        raise ValueError("main rollout result is missing report_stages")
    if str(report.get("task_id")) != str(task_id):
        raise ValueError("main rollout report_stages task_id mismatch")
    verdict, verdict_source = _stage_verdict(episode, report, target_stage)
    if verdict_source == "report_stages":
        if verdict.get("applicable") is not True:
            raise ValueError(f"main rollout stage {target_stage!r} is not applicable")
        passed = verdict.get("passed") is True
        first_pass_step = verdict.get("first_pass_step")
    else:
        passed = verdict.get("qualified") is True
        first_pass_step = verdict.get("qualified_step")
    if first_pass_step is not None:
        if isinstance(first_pass_step, bool) or not isinstance(first_pass_step, int):
            raise ValueError("main rollout first_pass_step must be an integer or null")
        if first_pass_step <= 0:
            raise ValueError("main rollout first_pass_step must be positive")
    if passed != (first_pass_step is not None):
        raise ValueError("main rollout stage passed and first_pass_step disagree")

    action_count = episode.get("policy_actions_consumed", 0)
    if isinstance(action_count, bool) or not isinstance(action_count, int):
        raise ValueError("main rollout policy_actions_consumed must be an integer")
    inference_calls = episode.get("policy_calls", 0)
    if isinstance(inference_calls, bool) or not isinstance(inference_calls, int):
        raise ValueError("main rollout policy_calls must be an integer")
    within_budget = passed and int(first_pass_step) <= budget
    if within_budget:
        failure_reason = None
        model_steps = int(first_pass_step)
    elif passed:
        failure_reason = "model_step_budget"
        model_steps = budget
    else:
        model_steps = min(max(int(action_count), 0), budget)
        failure_reason = (
            "model_step_budget"
            if int(action_count) >= budget
            else str(episode.get("failure_reason") or "stage_not_completed")
        )
    inference_calls = math.ceil(model_steps / cadence) if model_steps else 0
    return StageEvaluationResult(
        task_id=str(task_id),
        target_stage=str(target_stage),
        seed=int(seed),
        status=SUCCESS if within_budget else FAILURE,
        success=within_budget,
        success_step=int(first_pass_step) if within_budget else None,
        expert_prefix_steps=0,
        model_control_steps=model_steps,
        total_model_inference_calls=inference_calls,
        environment_steps=model_steps,
        handoff_observation_id=0,
        failure_reason=failure_reason,
    )


def _stage_verdict(
    episode: Mapping[str, Any],
    report: Mapping[str, Any],
    target_stage: str,
) -> tuple[Mapping[str, Any], str]:
    stages = report.get("stages")
    if isinstance(stages, Mapping) and target_stage in stages:
        verdict = stages[target_stage]
        if not isinstance(verdict, Mapping):
            raise ValueError(f"main rollout stage {target_stage!r} verdict must be a map")
        return verdict, "report_stages"

    diagnostics = episode.get("diagnostics")
    diagnostic_stages = (
        diagnostics.get("stages") if isinstance(diagnostics, Mapping) else None
    )
    if isinstance(diagnostic_stages, Mapping) and target_stage in diagnostic_stages:
        verdict = diagnostic_stages[target_stage]
        if not isinstance(verdict, Mapping):
            raise ValueError(
                f"main rollout diagnostic stage {target_stage!r} verdict must be a map"
            )
        return verdict, "diagnostics.stages"
    raise ValueError(f"main rollout report is missing stage {target_stage!r}")


def _validate_main_manifest(
    manifest: Mapping[str, Any],
    *,
    task_id: str,
    expected_suite_hash: str | None,
    expected_policy_contract: Mapping[str, Any] | None,
) -> None:
    task = manifest.get("task")
    if not isinstance(task, Mapping) or str(task.get("id")) != str(task_id):
        raise ValueError("main rollout manifest task does not match the stage suite")
    suite = manifest.get("suite")
    if expected_suite_hash is not None and (
        not isinstance(suite, Mapping) or suite.get("hash") != expected_suite_hash
    ):
        raise ValueError("main rollout suite hash does not match the stage base suite")
    if expected_policy_contract is not None:
        actual_contract = manifest.get("policy_contract")
        if not isinstance(actual_contract, Mapping):
            raise ValueError("main rollout manifest is missing policy_contract")
        if _policy_identity(actual_contract) != _policy_identity(expected_policy_contract):
            raise ValueError("main rollout policy identity does not match the stage policy")


def _policy_identity(contract: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = contract.get("checkpoint")
    checkpoint_map = checkpoint if isinstance(checkpoint, Mapping) else {}
    return {
        "policy": contract.get("policy"),
        "checkpoint_fingerprint": checkpoint_map.get("fingerprint"),
        "checkpoint_uri": (
            checkpoint_map.get("uri")
            if checkpoint_map.get("fingerprint") is None
            else None
        ),
        "action": contract.get("action"),
        "observation": contract.get("observation"),
        "control_frequency_hz": contract.get("control_frequency_hz"),
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"required main rollout artifact is missing: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


__all__ = ["import_main_rollout_stage", "main_rollout_stage_result"]
