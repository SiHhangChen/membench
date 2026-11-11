"""Evaluation suite configuration and stable hashing."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class RolloutConfig:
    max_steps: int | None
    replan_steps: int
    action_repeat: int = 1
    stop_on_success: bool = True
    ignore_done: bool = True


@dataclass(frozen=True)
class ArtifactConfig:
    record_video: bool = True
    video_cameras: tuple[str, ...] = (
        "agentview_left",
        "agentview_right",
        "eye_in_hand",
    )
    video_fps: int = 20


@dataclass(frozen=True)
class TransportConfig:
    timeout_s: float = 30.0
    connect_attempts: int = 10
    connect_retry_delay_s: float = 2.0
    inference_retries: int = 0


@dataclass(frozen=True)
class ExpertPrefixConfig:
    profile: str
    root: Path


@dataclass(frozen=True)
class EvalSuiteConfig:
    suite_version: str
    task_id: str
    env_split: str
    seeds: tuple[int, ...]
    rollout: RolloutConfig
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    task_options: Mapping[str, Any] = field(default_factory=dict)
    source_path: Path | None = None
    content_hash: str = ""
    expert_prefix: ExpertPrefixConfig | None = None


_SUITE_FIELDS = frozenset(
    {"suite_version", "task", "seeds", "rollout", "artifacts", "transport", "expert_prefix"}
)
_TASK_FIELDS = frozenset({"id", "env_split", "options"})
_ROLLOUT_FIELDS = frozenset(
    {"max_steps", "replan_steps", "action_repeat", "stop_on_success", "ignore_done"}
)
_ARTIFACT_FIELDS = frozenset({"record_video", "video_cameras", "video_fps"})
_TRANSPORT_FIELDS = frozenset(
    {"timeout_s", "connect_attempts", "connect_retry_delay_s", "inference_retries"}
)
_SEED_RANGE_FIELDS = frozenset({"start", "stop", "step"})
_MISSING = object()


def load_eval_suite(path: Path) -> EvalSuiteConfig:
    source_path = path.expanduser().resolve()
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Eval suite must be a YAML map: {source_path}")
    _reject_unknown_fields(payload, _SUITE_FIELDS, "eval suite")
    task = _mapping(payload, "task")
    rollout = _mapping(payload, "rollout")
    artifacts = _mapping(payload, "artifacts", default={})
    transport = _mapping(payload, "transport", default={})
    _reject_unknown_fields(task, _TASK_FIELDS, "task")
    _reject_unknown_fields(rollout, _ROLLOUT_FIELDS, "rollout")
    _reject_unknown_fields(artifacts, _ARTIFACT_FIELDS, "artifacts")
    _reject_unknown_fields(transport, _TRANSPORT_FIELDS, "transport")
    task_options = _mapping(task, "options", default={})
    prefix = None
    if "expert_prefix" in payload:
        prefix_payload = _mapping(payload, "expert_prefix")
        _reject_unknown_fields(prefix_payload, frozenset({"profile", "root"}), "expert_prefix")
        prefix_root = Path(_string(prefix_payload, "root", context="expert_prefix")).expanduser()
        if not prefix_root.is_absolute():
            prefix_root = Path(__file__).resolve().parents[2] / prefix_root
        prefix = ExpertPrefixConfig(
            profile=_string(prefix_payload, "profile", context="expert_prefix"),
            root=prefix_root.resolve(),
        )
    video_cameras = _string_list(
        artifacts,
        "video_cameras",
        default=("agentview_left", "agentview_right", "eye_in_hand"),
        context="artifacts",
    )
    suite = EvalSuiteConfig(
        suite_version=_string(payload, "suite_version", context="eval suite"),
        task_id=_string(task, "id", context="task"),
        env_split=_string(task, "env_split", default="pretrain", context="task"),
        seeds=expand_seeds(payload.get("seeds")),
        rollout=RolloutConfig(
            max_steps=_optional_integer(rollout, "max_steps", context="rollout"),
            replan_steps=_integer(rollout, "replan_steps", context="rollout"),
            action_repeat=_integer(
                rollout,
                "action_repeat",
                default=1,
                context="rollout",
            ),
            stop_on_success=_boolean(
                rollout,
                "stop_on_success",
                default=True,
                context="rollout",
            ),
            ignore_done=_boolean(
                rollout,
                "ignore_done",
                default=True,
                context="rollout",
            ),
        ),
        artifacts=ArtifactConfig(
            record_video=_boolean(
                artifacts,
                "record_video",
                default=True,
                context="artifacts",
            ),
            video_cameras=video_cameras,
            video_fps=_integer(
                artifacts,
                "video_fps",
                default=20,
                context="artifacts",
            ),
        ),
        transport=TransportConfig(
            timeout_s=_number(
                transport,
                "timeout_s",
                default=30.0,
                context="transport",
            ),
            connect_attempts=_integer(
                transport,
                "connect_attempts",
                default=10,
                context="transport",
            ),
            connect_retry_delay_s=_number(
                transport,
                "connect_retry_delay_s",
                default=2.0,
                context="transport",
            ),
            inference_retries=_integer(
                transport,
                "inference_retries",
                default=0,
                context="transport",
            ),
        ),
        task_options=dict(task_options),
        source_path=source_path,
        content_hash=hash_eval_suite(payload),
        expert_prefix=prefix,
    )
    suite = _resolve_task_horizon(suite)
    validate_eval_suite(suite)
    return suite


def validate_eval_suite(suite: EvalSuiteConfig) -> None:
    _validate_non_empty_string(suite.suite_version, "suite_version")
    _validate_non_empty_string(suite.task_id, "task.id")
    _validate_non_empty_string(suite.env_split, "task.env_split")
    if suite.expert_prefix is not None:
        _validate_non_empty_string(suite.expert_prefix.profile, "expert_prefix.profile")
        if suite.task_options.get("prompt_scheduler") is not None:
            raise ValueError("expert_prefix requires the unchanged full-task prompt")
        if suite.rollout.action_repeat != 1:
            raise ValueError("expert_prefix requires action_repeat=1")
    if not suite.seeds:
        raise ValueError("seeds must be non-empty and unique")
    for seed in suite.seeds:
        _validate_integer(seed, "seed")
    if len(set(suite.seeds)) != len(suite.seeds):
        raise ValueError("seeds must be non-empty and unique")
    _validate_integer(suite.rollout.max_steps, "rollout.max_steps")
    _validate_integer(suite.rollout.replan_steps, "rollout.replan_steps")
    _validate_integer(suite.rollout.action_repeat, "rollout.action_repeat")
    _validate_boolean(suite.rollout.stop_on_success, "rollout.stop_on_success")
    _validate_boolean(suite.rollout.ignore_done, "rollout.ignore_done")
    if suite.rollout.max_steps <= 0:
        raise ValueError("rollout.max_steps must be positive")
    if suite.rollout.replan_steps <= 0:
        raise ValueError("rollout.replan_steps must be positive")
    if suite.rollout.action_repeat <= 0:
        raise ValueError("rollout.action_repeat must be positive")
    _validate_boolean(suite.artifacts.record_video, "artifacts.record_video")
    if not isinstance(suite.artifacts.video_cameras, tuple):
        raise ValueError("artifacts.video_cameras must be a list of strings")
    for camera in suite.artifacts.video_cameras:
        _validate_non_empty_string(camera, "artifacts.video_cameras item")
    _validate_integer(suite.artifacts.video_fps, "artifacts.video_fps")
    if suite.artifacts.video_fps <= 0:
        raise ValueError("artifacts.video_fps must be positive")
    _validate_finite_number(suite.transport.timeout_s, "transport.timeout_s")
    _validate_integer(suite.transport.connect_attempts, "transport.connect_attempts")
    _validate_finite_number(
        suite.transport.connect_retry_delay_s,
        "transport.connect_retry_delay_s",
    )
    _validate_integer(suite.transport.inference_retries, "transport.inference_retries")
    if suite.transport.timeout_s <= 0:
        raise ValueError("transport.timeout_s must be positive")
    if suite.transport.connect_attempts <= 0:
        raise ValueError("transport.connect_attempts must be positive")
    if suite.transport.connect_retry_delay_s < 0:
        raise ValueError("transport.connect_retry_delay_s must be non-negative")
    if suite.transport.inference_retries != 0:
        raise ValueError(
            "membench-policy/v1 does not support safe inference retries; "
            "transport.inference_retries must be 0"
        )


def hash_eval_suite(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def expand_seeds(value: Any) -> tuple[int, ...]:
    if isinstance(value, list):
        seeds = tuple(_seed(item) for item in value)
    elif isinstance(value, Mapping):
        _reject_unknown_fields(value, _SEED_RANGE_FIELDS, "seeds")
        start = _integer(value, "start", context="seeds")
        stop = _integer(value, "stop", context="seeds")
        step = _integer(value, "step", default=1, context="seeds")
        if step == 0:
            raise ValueError("seed step cannot be zero")
        seeds = tuple(range(start, stop, step))
    else:
        raise ValueError("seeds must be a list or {start, stop, step} map")
    if not seeds:
        raise ValueError("seed expansion produced an empty list")
    return seeds


def _mapping(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    value = payload.get(key, default)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a map")
    return value


def _string(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: Any = _MISSING,
    context: str = "",
) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{_field_name(context, key)} must be a non-empty string")
    return value


def _integer(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: Any = _MISSING,
    context: str = "",
) -> int:
    value = payload.get(key, default)
    _validate_integer(value, _field_name(context, key))
    return value


def _optional_integer(
    payload: Mapping[str, Any],
    key: str,
    *,
    context: str = "",
) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    _validate_integer(value, _field_name(context, key))
    return value


def _resolve_task_horizon(suite: EvalSuiteConfig) -> EvalSuiteConfig:
    """Resolve and constrain max_steps through the canonical task catalog."""

    from bench.runtime.resolver import resolve_rollout_max_steps
    from bench.tasks import TaskCatalog
    from bench.tasks.contracts import EpisodeRequest
    raw = dict(suite.task_options)
    task_options = raw.get("task_options", {})
    if not isinstance(task_options, Mapping):
        raise ValueError("task.options.task_options must be a map")
    root = Path(__file__).resolve().parents[2]

    def option_path(name: str) -> Path | None:
        value = raw.get(name)
        if value is None:
            return None
        path = Path(value).expanduser()
        return (path if path.is_absolute() else root / path).resolve()

    episode = TaskCatalog.resolve(
        EpisodeRequest(
            task_id=TaskCatalog.canonical_id(suite.task_id),
            seed=suite.seeds[0],
            env_split=suite.env_split,
            split="evaluation",
            variant=str(raw.get("variant", "memory")),
            task_config_file=option_path("task_config_file"),
            observation_profile_file=option_path("observation_profile_file"),
            options=dict(task_options),
            allow_partial_calibration=True,
        )
    )
    max_steps = resolve_rollout_max_steps(suite.rollout.max_steps, episode)
    return replace(suite, rollout=replace(suite.rollout, max_steps=max_steps))


def _boolean(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: Any = _MISSING,
    context: str = "",
) -> bool:
    value = payload.get(key, default)
    _validate_boolean(value, _field_name(context, key))
    return value


def _number(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: Any = _MISSING,
    context: str = "",
) -> float:
    value = payload.get(key, default)
    _validate_finite_number(value, _field_name(context, key))
    return float(value)


def _string_list(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: tuple[str, ...],
    context: str = "",
) -> tuple[str, ...]:
    value = payload.get(key, default)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{_field_name(context, key)} must be a list of strings")
    for item in value:
        _validate_non_empty_string(item, f"{_field_name(context, key)} item")
    return tuple(value)


def _reject_unknown_fields(
    payload: Mapping[str, Any],
    allowed: frozenset[str],
    context: str,
) -> None:
    unknown = [key for key in payload if key not in allowed]
    if unknown:
        rendered = ", ".join(sorted((repr(key) for key in unknown)))
        raise ValueError(f"Unknown field(s) in {context}: {rendered}")


def _field_name(context: str, key: str) -> str:
    return f"{context}.{key}" if context else key


def _validate_non_empty_string(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_integer(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")


def _validate_boolean(value: Any, field_name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")


def _validate_finite_number(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be finite")


def _seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Invalid seed: {value!r}")
    return value
