"""Canonical task evaluation adapter backed by ``ResolvedEpisode``."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from bench.evaluation.adapter import EpisodeRuntime
from bench.evaluation.observation import build_canonical_observation
from bench.evaluation.prefix import (
    EvaluationPrefix,
    load_evaluation_prefix,
    validate_prefix_configuration,
)
from bench.evaluation.suite import EvalSuiteConfig
from bench.policy.protocol import CanonicalObservation, PolicyContract
from bench.policy.protocol.errors import ErrorCode, ProtocolError
from bench.runtime import RoboCasaRuntimeFactory, SensorRequirements
from bench.runtime.data_contract import ACTION_SCHEMA, OBSERVATION_PROFILE, STATE_SCHEMA
from bench.runtime.resolver import resolve_rollout_max_steps
from bench.runtime.sensors import validate_sensor_requirements
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest, ResolvedEpisode


_ROOT = Path(__file__).resolve().parents[2]
_ALLOWED_OPTIONS = frozenset(
    {
        "task_config_file",
        "observation_profile_file",
        "task_options",
        "variant",
        "prompt_scheduler",
    }
)


class CanonicalEvaluationAdapter:
    """Policy-neutral adapter backed only by ResolvedEpisode and EpisodeRuntime."""

    control_frequency_hz = 20.0

    def __init__(self, task_id: str) -> None:
        self.task_id = TaskCatalog.canonical_id(task_id)
        self._contract: PolicyContract | None = None

    def create_episode(
        self,
        *,
        seed: int,
        policy_contract: PolicyContract,
        suite: EvalSuiteConfig,
    ) -> EpisodeRuntime:
        options = self._parse_options(suite)
        episode = self._resolve_episode(seed=seed, suite=suite, options=options)
        self._validate_episode(episode, suite=suite, policy_contract=policy_contract)
        api = self._runtime_api()
        shared = RoboCasaRuntimeFactory(api).create(episode, execute=True)
        descriptor = shared.initialized.descriptor
        self._contract = policy_contract
        prompt = getattr(descriptor.prompt, "policy_prompt", str(descriptor.prompt))
        from bench.runtime.robot_state import RobotStateExtractor

        return EpisodeRuntime(
            episode_id=f"{_safe_name(self.task_id)}-seed-{int(seed):06d}",
            seed=int(seed),
            env=shared.env,
            observation=shared.observation,
            prompt=str(prompt),
            metadata={
                "resolved_episode": episode,
                "runtime": shared,
                "shared_runtime": shared,
                "initialized_task_runtime": shared.initialized,
                "descriptor": descriptor,
                "robot_state_extractor": RobotStateExtractor(),
            },
            result_metadata={
                "prompt": str(prompt),
                "resolved_episode": episode.to_dict(),
                "task_id": self.task_id,
                "task_version": episode.task_version,
                "observation_profile": descriptor.sensors.profile_id,
                "initialization": dict(shared.initialized.initialization),
            },
        )

    def build_observation(self, runtime: EpisodeRuntime) -> CanonicalObservation:
        if self._contract is None:
            raise RuntimeError(f"{self.task_id} episode has not been created")
        descriptor = runtime.metadata["descriptor"]
        aliases = {
            alias: descriptor.sensors.observation_key(alias)
            for alias in descriptor.sensors.camera_aliases
        }
        return build_canonical_observation(
            runtime.observation,
            task_id=self.task_id,
            prompt=runtime.prompt,
            contract=self._contract,
            camera_aliases=aliases,
            env=runtime.env,
            robot_state_extractor=runtime.metadata["robot_state_extractor"],
        )

    def create_evaluation_prefix(self, runtime: EpisodeRuntime, suite: EvalSuiteConfig):
        if suite.expert_prefix is None:
            return None
        api = self._runtime_api()
        descriptor = runtime.metadata["descriptor"]
        validate_prefix_configuration(api, descriptor, suite.expert_prefix)
        artifact = load_evaluation_prefix(suite.expert_prefix, descriptor.episode)
        return EvaluationPrefix(artifact, api.validate_evaluation_prefix, runtime)

    def artifact_frames(
        self,
        runtime: EpisodeRuntime,
        cameras: tuple[str, ...],
    ) -> Mapping[str, np.ndarray]:
        descriptor = runtime.metadata["descriptor"]
        frames: dict[str, np.ndarray] = {}
        for alias in cameras:
            source_key = descriptor.sensors.observation_key(alias)
            if source_key not in runtime.observation:
                raise KeyError(f"Task {self.task_id} is missing camera alias {alias!r}")
            frame = np.asarray(runtime.observation[source_key])
            frames[alias] = np.ascontiguousarray(frame)
        return frames

    def check_success(self, runtime: EpisodeRuntime, info: Mapping[str, Any]) -> bool:
        return bool(self._runtime_api().check_success(runtime.metadata["initialized_task_runtime"], info))

    def diagnostics(self, runtime: EpisodeRuntime) -> Mapping[str, Any]:
        return dict(self._runtime_api().diagnostics(runtime.metadata["initialized_task_runtime"]))

    @property
    def report_stage_profile_id(self) -> str | None:
        value = self._runtime_api().report_stage_profile_id
        return None if value is None else str(value)

    def report_stages(
        self,
        runtime: EpisodeRuntime,
    ) -> Mapping[str, Any] | None:
        callback = self._runtime_api().report_stages
        if callback is None:
            return None
        return dict(callback(runtime.metadata["initialized_task_runtime"]) or {})

    def close(self, runtime: EpisodeRuntime) -> None:
        try:
            runtime.metadata["runtime"].close()
        finally:
            self._contract = None

    def validate_suite(
        self,
        *,
        suite: EvalSuiteConfig,
        policy_contract: PolicyContract,
    ) -> Mapping[str, Any]:
        options = self._parse_options(suite)
        episode = self._resolve_episode(seed=suite.seeds[0], suite=suite, options=options)
        self._validate_episode(episode, suite=suite, policy_contract=policy_contract)
        descriptor = self._runtime_api().build_descriptor(episode, True)
        validate_prefix_configuration(self._runtime_api(), descriptor, suite.expert_prefix)
        scheduler_id = options.get("prompt_scheduler")
        if scheduler_id is not None:
            builder = getattr(self._runtime_api(), "build_subtask_scheduler", None)
            if not callable(builder):
                raise ValueError(
                    f"task {self.task_id} does not support prompt_scheduler={scheduler_id!r}"
                )
        if descriptor.sensors.profile_id != OBSERVATION_PROFILE:
            raise ProtocolError(
                ErrorCode.CONTRACT_MISMATCH,
                f"task uses {descriptor.sensors.profile_id!r}; expected {OBSERVATION_PROFILE}",
            )
        if episode.action_schema != ACTION_SCHEMA:
            raise ProtocolError(
                ErrorCode.CONTRACT_MISMATCH,
                f"task action schema {episode.action_schema!r} != {ACTION_SCHEMA}",
            )
        if policy_contract.observation.state_schema != STATE_SCHEMA:
            raise ProtocolError(ErrorCode.CONTRACT_MISMATCH, f"expected state schema {STATE_SCHEMA}")
        return {
            "adapter_mode": "resolved_episode_runtime",
            "report_stage_profile_id": self.report_stage_profile_id,
            "resolved_episode_identity": episode.identity,
            "task_config_path": episode.task_config_file,
            "observation_profile_path": episode.observation_profile_file,
            "observation_profile": descriptor.sensors.profile_id,
            "effective_rollout_max_steps": resolve_rollout_max_steps(
                suite.rollout.max_steps, episode
            ),
            "prompt_scheduler": scheduler_id,
            "expert_prefix": (
                None if suite.expert_prefix is None else {
                    "profile": suite.expert_prefix.profile,
                    "root": str(suite.expert_prefix.root),
                    "policy_observes_prefix": True,
                    "budget": "policy_steps_after_handoff",
                }
            ),
        }

    def _resolve_episode(
        self,
        *,
        seed: int,
        suite: EvalSuiteConfig,
        options: Mapping[str, Any],
    ) -> ResolvedEpisode:
        return TaskCatalog.resolve(
            EpisodeRequest(
                task_id=self.task_id,
                seed=int(seed),
                env_split=suite.env_split,
                split="evaluation",
                variant=str(options["variant"]),
                options=dict(options["task_options"]),
                task_config_file=options["task_config_file"],
                observation_profile_file=options["observation_profile_file"],
                allow_partial_calibration=False,
            )
        )

    def _validate_episode(
        self,
        episode: ResolvedEpisode,
        *,
        suite: EvalSuiteConfig,
        policy_contract: PolicyContract,
    ) -> None:
        requested_aliases = tuple(
            dict.fromkeys(
                (
                    *policy_contract.observation.required_cameras,
                    *(suite.artifacts.video_cameras if suite.artifacts.record_video else ()),
                )
            )
        )
        descriptor = self._runtime_api().build_descriptor(episode, True)
        if descriptor.sensors.profile_id != OBSERVATION_PROFILE:
            raise ProtocolError(
                ErrorCode.CONTRACT_MISMATCH,
                f"task uses {descriptor.sensors.profile_id!r}; expected {OBSERVATION_PROFILE}",
            )
        if episode.action_schema != ACTION_SCHEMA:
            raise ProtocolError(
                ErrorCode.CONTRACT_MISMATCH,
                f"task action schema {episode.action_schema!r} != {ACTION_SCHEMA}",
            )
        if policy_contract.observation.state_schema != STATE_SCHEMA:
            raise ProtocolError(
                ErrorCode.CONTRACT_MISMATCH,
                f"expected state schema {STATE_SCHEMA}",
            )
        try:
            validate_sensor_requirements(
                descriptor.sensors,
                SensorRequirements(
                    required_camera_aliases=requested_aliases,
                    required_image_size=policy_contract.observation.image_size,
                    require_depth=any(
                        modality == "depth" or modality.startswith("depth.")
                        for modality in policy_contract.observation.required_modalities
                    ),
                ),
            )
        except ValueError as exc:
            raise ProtocolError(ErrorCode.CONTRACT_MISMATCH, str(exc)) from exc

    def _parse_options(self, suite: EvalSuiteConfig) -> dict[str, Any]:
        if TaskCatalog.canonical_id(suite.task_id) != self.task_id:
            raise ValueError(f"{self.task_id} adapter cannot run task {suite.task_id}")
        raw = dict(suite.task_options)
        unknown = sorted(set(raw) - _ALLOWED_OPTIONS)
        if unknown:
            raise ValueError(f"Unknown {self.task_id} evaluation options: {unknown}")
        task_options = raw.get("task_options", {})
        if not isinstance(task_options, Mapping):
            raise ValueError("task.options.task_options must be a map")
        return {
            "task_config_file": _optional_path(raw.get("task_config_file")),
            "observation_profile_file": _optional_path(raw.get("observation_profile_file")),
            "task_options": dict(task_options),
            "variant": str(raw.get("variant", "memory")),
            "prompt_scheduler": _optional_scheduler_id(raw.get("prompt_scheduler")),
        }

    def create_subtask_scheduler(
        self,
        runtime: EpisodeRuntime,
        suite: EvalSuiteConfig,
    ) -> Any | None:
        """Create an optional task-owned scheduler selected by suite options."""

        options = self._parse_options(suite)
        schedule_id = options.get("prompt_scheduler")
        if schedule_id is None:
            return None
        if not isinstance(schedule_id, str) or not schedule_id.strip():
            raise ValueError("task.options.prompt_scheduler must be a non-empty string")
        builder = getattr(self._runtime_api(), "build_subtask_scheduler", None)
        if not callable(builder):
            raise ValueError(
                f"task {self.task_id} does not support prompt_scheduler={schedule_id!r}"
            )
        return builder(runtime, schedule_id.strip())

    def _runtime_api(self) -> Any:
        api = resolve_builtin_task_plugin(self.task_id).load_runtime_api()
        if api is None:
            raise LookupError(f"task {self.task_id!r} has no canonical runtime API")
        return api


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return (path if path.is_absolute() else _ROOT / path).resolve()


def _optional_scheduler_id(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("task.options.prompt_scheduler must be a non-empty string")
    return value.strip()


def _safe_name(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in value)


__all__ = ["CanonicalEvaluationAdapter"]
