"""Policy-neutral closed-loop rollout runner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import time
import traceback
from typing import Any

import numpy as np

from bench.evaluation.action import ActionChunkQueue
from bench.evaluation.adapter import EpisodeRuntime, EvaluationAdapter, preflight_adapter
from bench.evaluation.artifacts import ArtifactWriter, EpisodeArtifactSession
from bench.evaluation.client import PolicyClient
from bench.evaluation.report_stages import validate_report_stages
from bench.evaluation.suite import EvalSuiteConfig


@dataclass
class EpisodeResult:
    seed: int
    episode_id: str
    task_id: str
    suite_version: str
    status: str
    success: bool
    success_step: int | None
    env_steps: int
    policy_calls: int
    reward_sum: float
    failure_reason: str | None = None
    error: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    report_stages: Mapping[str, Any] | None = None
    inference_latency_ms: list[float] = field(default_factory=list)
    action_mean: list[float] = field(default_factory=list)
    action_std: list[float] = field(default_factory=list)
    action_min: list[float] = field(default_factory=list)
    action_max: list[float] = field(default_factory=list)
    policy_actions_returned: int = 0
    policy_actions_consumed: int = 0
    duration_s: float = 0.0
    episode_metadata: Mapping[str, Any] = field(default_factory=dict)


class RolloutRunner:
    def __init__(
        self,
        *,
        client: PolicyClient,
        suite: EvalSuiteConfig,
        artifacts: ArtifactWriter | None = None,
    ) -> None:
        self.client = client
        self.suite = suite
        self.artifacts = artifacts
        if suite.rollout.replan_steps > client.contract.action.horizon:
            raise ValueError(
                f"replan_steps={suite.rollout.replan_steps} exceeds policy horizon "
                f"{client.contract.action.horizon}"
            )

    def run_episode(
        self,
        adapter: EvaluationAdapter,
        seed: int,
    ) -> EpisodeResult:
        preflight_adapter(
            adapter,
            suite=self.suite,
            policy_contract=self.client.contract,
        )
        started_at = time.monotonic()
        runtime: EpisodeRuntime | None = None
        session = self._start_artifact_session(seed)
        actions: list[np.ndarray] = []
        latencies: list[float] = []
        env_steps = 0
        policy_calls = 0
        reward_sum = 0.0
        success = False
        task_stop_execution = False
        success_step: int | None = None
        failure_reason: str | None = None
        policy_actions_returned = 0
        policy_actions_consumed = 0
        episode_id = f"{self.suite.task_id.lower()}-seed-{seed:06d}"
        result: EpisodeResult | None = None
        interrupted: KeyboardInterrupt | None = None
        scheduler: Any | None = None
        prefix: Any | None = None
        prefix_finished = False
        policy_steps = 0
        try:
            runtime = adapter.create_episode(
                seed=seed,
                policy_contract=self.client.contract,
                suite=self.suite,
            )
            episode_id = runtime.episode_id
            prefix_factory = getattr(adapter, "create_evaluation_prefix", None)
            if self.suite.expert_prefix is not None:
                if not callable(prefix_factory):
                    raise ValueError("adapter does not support configured expert_prefix")
                prefix = prefix_factory(runtime, self.suite)
                if prefix is None:
                    raise ValueError("adapter did not provide the configured expert_prefix")
            scheduler_factory = getattr(adapter, "create_subtask_scheduler", None)
            if callable(scheduler_factory):
                scheduler = scheduler_factory(runtime, self.suite)
                if scheduler is not None:
                    prompt = str(getattr(scheduler, "current_prompt", "")).strip()
                    if not prompt:
                        raise ValueError("subtask scheduler returned an empty initial prompt")
                    runtime.prompt = prompt
            self.client.reset(episode_id)
            queue = ActionChunkQueue(
                expected_dim=self.client.contract.action.dim,
                action_schema=self.client.contract.action.schema,
            )
            while policy_steps < self.suite.rollout.max_steps:
                if prefix is not None and prefix.complete and not prefix_finished:
                    prefix.handoff(env_steps)
                    prefix_finished = True
                    queue.clear()
                expert_control = prefix is not None and not prefix_finished
                if len(queue) == 0:
                    observation = adapter.build_observation(runtime)
                    start = time.monotonic()
                    response = self.client.infer(
                        observation,
                        episode_id=episode_id,
                        step=env_steps,
                    )
                    client_latency_ms = (time.monotonic() - start) * 1000.0
                    latencies.append(client_latency_ms)
                    policy_calls += 1
                    policy_actions_returned += int(response.actions.shape[0])
                    queued_count = queue.push(
                        response.actions,
                        limit=self.suite.rollout.replan_steps,
                    )
                    if session is not None:
                        session.append_request_metric(
                            {
                                "request_id": response.request_id,
                                "step": env_steps,
                                "prompt": observation.prompt,
                                "subtask": (
                                    getattr(scheduler, "current_subtask", None)
                                    if scheduler is not None
                                    else None
                                ),
                                "client_latency_ms": client_latency_ms,
                                "returned_action_count": int(response.actions.shape[0]),
                                "queued_action_count": int(queued_count),
                                "action_source": "expert" if expert_control else "policy",
                                "server_timing": dict(response.timing),
                            }
                        )
                action = queue.pop()
                if expert_control:
                    # Policy predictions preserve the observation history but
                    # only the frozen expert action reaches the simulator.
                    action = prefix.pop()
                else:
                    policy_actions_consumed += 1
                prompt_changed = False
                for _ in range(self.suite.rollout.action_repeat):
                    if policy_steps >= self.suite.rollout.max_steps:
                        break
                    shared_runtime = runtime.metadata.get("shared_runtime")
                    task_terminal = False
                    task_stop_execution = False
                    task_failure_code = None
                    if shared_runtime is not None:
                        transition = shared_runtime.step(action)
                        observation = transition.observation
                        reward = transition.reward
                        done = transition.terminal
                        info = transition.info
                        if transition.diagnostics is not None:
                            task_terminal = bool(transition.diagnostics.terminal)
                            task_failure_code = transition.diagnostics.failure_code
                            task_stop_execution = bool(transition.diagnostics.stop_execution)
                    else:
                        step_result = runtime.env.step(action)
                        observation, reward, done, info = _normalize_env_step(step_result)
                    runtime.observation = observation
                    env_steps += 1
                    if expert_control:
                        prefix.after_step()
                    else:
                        policy_steps += 1
                    reward_sum += reward
                    actions.append(action.copy())
                    if session is not None:
                        session.append_action(action)
                        if self.suite.artifacts.record_video:
                            capture_frames = getattr(adapter, "artifact_frames", None)
                            if callable(capture_frames):
                                frames = capture_frames(
                                    runtime,
                                    self.suite.artifacts.video_cameras,
                                )
                            else:
                                frames = adapter.build_observation(runtime).images
                            session.append_frames(frames)
                    step_success = adapter.check_success(runtime, info)
                    if expert_control:
                        if step_success or task_terminal or task_stop_execution or task_failure_code:
                            raise ValueError(
                                "invalid expert prefix: target completed or task failed "
                                f"before handoff ({task_failure_code})"
                            )
                        if done and not self.suite.rollout.ignore_done:
                            raise ValueError("invalid expert prefix: environment ended")
                        continue
                    # A task-owned final decision stops execution regardless of
                    # stop_on_success / ignore_done. Do not run later subtask
                    # handoffs, repeated actions, or the rest of the action chunk.
                    if task_stop_execution:
                        success = bool(step_success and not task_failure_code)
                        if success:
                            if success_step is None:
                                success_step = env_steps
                        else:
                            failure_reason = task_failure_code or "task_terminal"
                        break
                    # A semantic protocol violation is terminal even if this
                    # episode reached a candidate success on an earlier step.
                    # This matters for full-horizon tasks such as WR-04, where
                    # removing the object again must revoke that candidate.
                    if task_terminal and task_failure_code:
                        success = False
                        failure_reason = task_failure_code
                        break
                    if scheduler is not None:
                        subtask_event = scheduler.after_step(runtime, info)
                        if subtask_event.changed:
                            next_prompt = str(subtask_event.prompt or "").strip()
                            if not next_prompt:
                                raise ValueError(
                                    "subtask scheduler returned an empty handoff prompt"
                                )
                            runtime.prompt = next_prompt
                            queue.clear()
                            prompt_changed = True
                        if subtask_event.failed:
                            failure_reason = (
                                subtask_event.failure_reason or "subtask_failed"
                            )
                            break
                        if subtask_event.complete and not step_success:
                            failure_reason = "subtask_complete_without_task_success"
                            break
                    if step_success:
                        success = True
                        if success_step is None:
                            success_step = env_steps
                        if self.suite.rollout.stop_on_success:
                            break
                    elif not self.suite.rollout.stop_on_success:
                        # With full-horizon evaluation, success is a final-state
                        # property rather than a historical latch.
                        success = False
                    if task_terminal and not step_success:
                        failure_reason = "task_terminal"
                        break
                    if done and not self.suite.rollout.ignore_done:
                        failure_reason = "env_done"
                        break
                    if prompt_changed:
                        break
                if task_stop_execution or (success and self.suite.rollout.stop_on_success):
                    break
                if failure_reason is not None:
                    break
            if not success and failure_reason is None:
                failure_reason = (
                    "terminal_state_not_preserved"
                    if success_step is not None
                    else "max_steps"
                )
            result = self._make_result(
                seed=seed,
                episode_id=episode_id,
                status="success" if success else "failure",
                success=success,
                success_step=success_step,
                env_steps=env_steps,
                policy_calls=policy_calls,
                reward_sum=reward_sum,
                failure_reason=failure_reason,
                error=None,
                diagnostics=adapter.diagnostics(runtime),
                report_stages=self._report_stages(
                    adapter,
                    runtime,
                    expected_success=success,
                ),
                latencies=latencies,
                actions=actions,
                policy_actions_returned=policy_actions_returned,
                policy_actions_consumed=policy_actions_consumed,
                episode_metadata=_episode_metadata(runtime, scheduler),
            )
        except KeyboardInterrupt as exc:
            interrupted = exc
            result = self._make_result(
                seed=seed,
                episode_id=episode_id,
                status="cancelled",
                success=False,
                success_step=success_step,
                env_steps=env_steps,
                policy_calls=policy_calls,
                reward_sum=reward_sum,
                failure_reason=None,
                error="KeyboardInterrupt",
                diagnostics=self._safe_diagnostics(adapter, runtime),
                report_stages=self._safe_report_stages(
                    adapter,
                    runtime,
                    expected_success=False,
                ),
                latencies=latencies,
                actions=actions,
                policy_actions_returned=policy_actions_returned,
                policy_actions_consumed=policy_actions_consumed,
                episode_metadata=_episode_metadata(runtime, scheduler),
            )
        except Exception as exc:
            result = self._make_result(
                seed=seed,
                episode_id=episode_id,
                status="error",
                success=False,
                success_step=success_step,
                env_steps=env_steps,
                policy_calls=policy_calls,
                reward_sum=reward_sum,
                failure_reason=None,
                error="".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip(),
                diagnostics=self._safe_diagnostics(adapter, runtime),
                report_stages=self._safe_report_stages(
                    adapter,
                    runtime,
                    expected_success=False,
                ),
                latencies=latencies,
                actions=actions,
                policy_actions_returned=policy_actions_returned,
                policy_actions_consumed=policy_actions_consumed,
                episode_metadata=_episode_metadata(runtime, scheduler),
            )
        finally:
            if runtime is not None:
                try:
                    adapter.close(runtime)
                except Exception as exc:
                    close_error = f"adapter.close:{type(exc).__name__}: {exc}"
                    if result is None:
                        result = self._make_result(
                            seed=seed,
                            episode_id=episode_id,
                            status="error",
                            success=False,
                            success_step=success_step,
                            env_steps=env_steps,
                            policy_calls=policy_calls,
                            reward_sum=reward_sum,
                            failure_reason=None,
                            error=close_error,
                            diagnostics={},
                            report_stages=None,
                            latencies=latencies,
                            actions=actions,
                            policy_actions_returned=policy_actions_returned,
                            policy_actions_consumed=policy_actions_consumed,
                            episode_metadata=_episode_metadata(runtime, scheduler),
                        )
                    else:
                        result.status = "error"
                        result.success = False
                        result.failure_reason = None
                        result.error = (
                            close_error
                            if result.error is None
                            else f"{result.error}; {close_error}"
                        )
                        result.report_stages = None
        assert result is not None
        result.duration_s = time.monotonic() - started_at
        if session is not None:
            try:
                session.finish(asdict(result))
            except BaseException:
                session.abort()
                raise
        if self.artifacts is not None:
            self.artifacts.write_summary()
        if interrupted is not None:
            raise interrupted
        return result

    def run_suite(
        self,
        adapter: EvaluationAdapter,
        *,
        resume: bool = True,
        retry_errors: bool = False,
    ) -> list[EpisodeResult]:
        completed = (
            self.artifacts.completed_seeds(retry_errors=retry_errors)
            if resume and self.artifacts is not None
            else set()
        )
        pending = [seed for seed in self.suite.seeds if seed not in completed]
        results = []
        for index, seed in enumerate(pending, start=1):
            print(
                f"[eval] [{index}/{len(pending)}] seed={seed:06d} running...",
                flush=True,
            )
            result = self.run_episode(adapter, seed)
            print(_format_episode_progress(result, index, len(pending)), flush=True)
            results.append(result)
        if self.artifacts is not None:
            self.artifacts.write_summary()
        return results

    def _start_artifact_session(self, seed: int) -> EpisodeArtifactSession | None:
        if self.artifacts is None:
            return None
        return self.artifacts.start_episode(
            seed=seed,
            record_video=self.suite.artifacts.record_video,
            video_cameras=self.suite.artifacts.video_cameras,
            video_fps=self.suite.artifacts.video_fps,
        )

    def _make_result(
        self,
        *,
        seed: int,
        episode_id: str,
        status: str,
        success: bool,
        success_step: int | None,
        env_steps: int,
        policy_calls: int,
        reward_sum: float,
        failure_reason: str | None,
        error: str | None,
        diagnostics: Mapping[str, Any],
        report_stages: Mapping[str, Any] | None,
        latencies: list[float],
        actions: list[np.ndarray],
        policy_actions_returned: int,
        policy_actions_consumed: int,
        episode_metadata: Mapping[str, Any],
    ) -> EpisodeResult:
        if actions:
            array = np.stack(actions)
            action_mean = array.mean(axis=0).tolist()
            action_std = array.std(axis=0).tolist()
            action_min = array.min(axis=0).tolist()
            action_max = array.max(axis=0).tolist()
        else:
            action_mean = action_std = action_min = action_max = []
        return EpisodeResult(
            seed=seed,
            episode_id=episode_id,
            task_id=self.suite.task_id,
            suite_version=self.suite.suite_version,
            status=status,
            success=success,
            success_step=success_step,
            env_steps=env_steps,
            policy_calls=policy_calls,
            reward_sum=reward_sum,
            failure_reason=failure_reason,
            error=error,
            diagnostics=dict(diagnostics),
            report_stages=(
                None if report_stages is None else dict(report_stages)
            ),
            inference_latency_ms=latencies,
            action_mean=action_mean,
            action_std=action_std,
            action_min=action_min,
            action_max=action_max,
            policy_actions_returned=int(policy_actions_returned),
            policy_actions_consumed=int(policy_actions_consumed),
            episode_metadata=dict(episode_metadata),
        )

    @staticmethod
    def _safe_diagnostics(
        adapter: EvaluationAdapter,
        runtime: EpisodeRuntime | None,
    ) -> dict[str, Any]:
        if runtime is None:
            return {}
        try:
            return dict(adapter.diagnostics(runtime))
        except Exception as exc:
            return {"diagnostics_error": f"{type(exc).__name__}: {exc}"}

    def _report_stages(
        self,
        adapter: EvaluationAdapter,
        runtime: EpisodeRuntime,
        *,
        expected_success: bool,
    ) -> dict[str, Any] | None:
        profile_id = getattr(adapter, "report_stage_profile_id", None)
        callback = getattr(adapter, "report_stages", None)
        if profile_id is None:
            if callable(callback) and callback(runtime) is not None:
                raise ValueError(
                    f"Task {self.suite.task_id} returned report_stages without "
                    "declaring report_stage_profile_id"
                )
            return None
        if not callable(callback):
            raise ValueError(
                f"Task {self.suite.task_id} declares report stage profile "
                f"{profile_id!r} but has no report_stages callback"
            )
        payload = callback(runtime)
        if payload is None:
            raise ValueError(
                f"Task {self.suite.task_id} declares report stage profile "
                f"{profile_id!r} but returned no report"
            )
        return validate_report_stages(
            payload,
            expected_task_id=self.suite.task_id,
            expected_profile_id=str(profile_id),
            expected_success=expected_success,
        )

    def _safe_report_stages(
        self,
        adapter: EvaluationAdapter,
        runtime: EpisodeRuntime | None,
        *,
        expected_success: bool,
    ) -> dict[str, Any] | None:
        if runtime is None:
            return None
        try:
            return self._report_stages(
                adapter,
                runtime,
                expected_success=expected_success,
            )
        except Exception:
            return None


def _format_episode_progress(
    result: EpisodeResult,
    index: int,
    total: int,
) -> str:
    """Format one completed episode as a single progress line for the console."""
    tag = f"[eval] [{index}/{total}] seed={result.seed:06d}"
    if result.status == "success":
        outcome = "SUCCESS"
        detail = f"step={result.success_step}"
    elif result.status == "failure":
        outcome = "FAILURE"
        detail = f"reason={result.failure_reason}"
    else:
        outcome = result.status.upper()
        detail = f"error={result.error}" if result.error is not None else ""
        if len(detail) > 160:
            detail = detail[:160]
    return (
        f"{tag} {outcome} {detail} "
        f"steps={result.env_steps} calls={result.policy_calls} "
        f"dur={result.duration_s:.1f}s"
    )


def _episode_metadata(
    runtime: EpisodeRuntime | None,
    scheduler: Any | None,
) -> Mapping[str, Any]:
    if runtime is None:
        return {}
    metadata = dict(runtime.result_metadata)
    if scheduler is not None:
        to_metadata = getattr(scheduler, "to_metadata", None)
        if not callable(to_metadata):
            raise TypeError("subtask scheduler must expose to_metadata()")
        metadata["subtask_scheduler"] = dict(to_metadata())
    return metadata


def _normalize_env_step(
    value: Any,
) -> tuple[Mapping[str, Any], float, bool, Mapping[str, Any]]:
    if not isinstance(value, tuple):
        raise TypeError("env.step() must return a tuple")
    if len(value) == 4:
        observation, reward, done, info = value
    elif len(value) == 5:
        observation, reward, terminated, truncated, info = value
        done = bool(terminated or truncated)
    else:
        raise ValueError(f"env.step() returned {len(value)} values")
    if not isinstance(observation, Mapping) or not isinstance(info, Mapping):
        raise TypeError("env.step() observation and info must be mappings")
    return observation, float(reward), bool(done), info
