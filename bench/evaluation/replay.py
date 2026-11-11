"""Action-file policy client and source discovery for evaluator replays.

Replay is an ordinary evaluation whose policy happens to read a fixed action
stream from disk. Environment execution, evaluator calls, report stages,
videos, and result artifacts remain owned by ``RolloutRunner``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from bench.core.identity import canonical_hash
from bench.evaluation.action import validate_action_chunk
from bench.evaluation.artifacts import ArtifactWriter
from bench.evaluation.canonical_adapter import CanonicalEvaluationAdapter
from bench.evaluation.rollout import EpisodeResult, RolloutRunner
from bench.evaluation.suite import EvalSuiteConfig, load_eval_suite
from bench.policy.protocol import (
    ActionContract,
    CanonicalObservation,
    CheckpointIdentity,
    InferenceResponse,
    ObservationContract,
    PolicyContract,
)
from bench.runtime.data_contract import (
    ACTION_SCHEMA,
    STATE_DIM,
    STATE_SCHEMA,
)
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest, ResolvedEpisode

ROOT = Path(__file__).resolve().parents[2]
RUN_REPLAY_MODE = "saved_actions_evaluation_replay"
DATASET_REPLAY_MODE = "recorded_actions_evaluation_replay"
REPLAY_PROFILE = "membench-action-replay-v1"


def _wr11_recording_prompt_identity_matches(episode: ResolvedEpisode, expected: str | None) -> bool:
    """Accept only the known WR11 wording revision with every other input identical."""
    if episode.task_id != "WR-11" or expected is None:
        return False
    from bench.tasks.wr11.prompts import PROMPT

    if episode.public.get("prompt") != PROMPT:
        return False
    payload = episode.identity_payload()
    payload["public"]["prompt"] = (
        "Observe the sample milk carton on the dining island and remember its packaging. "
        "Move sideways to the stock area at the other end of the island, pick the carton with the same "
        "packaging, and bring it back to the empty tray beside the sample. "
        "Choose correctly on your first pickup. Leave the sample in place."
    )
    return canonical_hash(payload) == expected


@dataclass(frozen=True, slots=True)
class ReplayEpisode:
    seed: int
    actions: np.ndarray
    request: EpisodeRequest
    source: Mapping[str, Any]
    expected_identity: str | None = None

    def __post_init__(self) -> None:
        actions = validate_action_chunk(self.actions)
        if int(self.request.seed) != int(self.seed):
            raise ValueError("ReplayEpisode seed and EpisodeRequest seed differ")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "source", dict(self.source))


class ReplayPolicyClient:
    """In-process policy client that returns one recorded action stream."""

    def __init__(
        self,
        actions: np.ndarray,
        *,
        contract: PolicyContract,
        chunk_size: int,
    ) -> None:
        self.contract = contract
        self.actions = validate_action_chunk(actions)
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_size > self.contract.action.horizon:
            raise ValueError("chunk_size exceeds replay policy action horizon")
        self._episode_id: str | None = None
        self._cursor = 0
        self._calls = 0

    def reset(self, episode_id: str) -> None:
        self._episode_id = str(episode_id)
        self._cursor = 0
        self._calls = 0

    def infer(
        self,
        observation: CanonicalObservation,
        *,
        episode_id: str,
        step: int,
    ) -> InferenceResponse:
        del observation
        if episode_id != self._episode_id:
            raise ValueError(f"reset({episode_id!r}) must be called before infer")
        if int(step) != self._cursor:
            raise ValueError(
                f"replay action cursor mismatch: rollout step={step}, cursor={self._cursor}"
            )
        if self._cursor >= len(self.actions):
            raise RuntimeError("recorded action stream exhausted")
        stop = min(len(self.actions), self._cursor + self.chunk_size)
        chunk = np.ascontiguousarray(self.actions[self._cursor : stop])
        self._cursor = stop
        self._calls += 1
        return InferenceResponse(
            request_id=f"replay-{self._calls:06d}",
            episode_id=episode_id,
            actions=chunk,
            timing={"replay_ms": 0.0},
        )


class ReplayEvaluationAdapter(CanonicalEvaluationAdapter):
    """Canonical adapter pinned to the episode request stored by a source."""

    def __init__(
        self,
        replay: ReplayEpisode,
        *,
        require_identity: bool,
    ) -> None:
        super().__init__(replay.request.task_id)
        self.replay = replay
        self.require_identity = bool(require_identity)
        self._identity_matches: bool | None = None
        self._prompt_revision_matches = False

    def _resolve_episode(
        self,
        *,
        seed: int,
        suite: EvalSuiteConfig,
        options: Mapping[str, Any],
    ) -> ResolvedEpisode:
        del suite, options
        if int(seed) != self.replay.seed:
            raise ValueError(f"unexpected replay seed {seed}")
        episode = TaskCatalog.resolve(self.replay.request)
        expected = self.replay.expected_identity
        self._identity_matches = None if expected is None else expected == episode.identity
        self._prompt_revision_matches = (
            self._identity_matches is False
            and _wr11_recording_prompt_identity_matches(episode, expected)
        )
        if (
            self.require_identity
            and self._identity_matches is False
            and not self._prompt_revision_matches
        ):
            raise ValueError(
                f"episode identity mismatch for seed={seed}: "
                f"recorded={expected} current={episode.identity}"
            )
        return episode

    def create_episode(
        self,
        *,
        seed: int,
        policy_contract: PolicyContract,
        suite: EvalSuiteConfig,
    ):
        runtime = super().create_episode(
            seed=seed,
            policy_contract=policy_contract,
            suite=suite,
        )
        runtime.result_metadata["replay"] = {
            **dict(self.replay.source),
            "profile": REPLAY_PROFILE,
            "expected_episode_identity": self.replay.expected_identity,
            "current_episode_identity": runtime.result_metadata["resolved_episode"].get("identity"),
            "episode_identity_matches": self._identity_matches,
            "wr11_prompt_revision_identity_matches": self._prompt_revision_matches,
            "source_action_steps": int(len(self.replay.actions)),
        }
        return runtime


def run_action_replay(
    episodes: list[ReplayEpisode],
    *,
    suite: EvalSuiteConfig,
    contract: PolicyContract,
    writer: ArtifactWriter,
    require_identity: bool,
) -> list[EpisodeResult]:
    """Evaluate fixed action streams through the production rollout path."""

    results: list[EpisodeResult] = []
    base_replan = min(suite.rollout.replan_steps, contract.action.horizon)
    if base_replan <= 0:
        raise ValueError("replay requires a positive replanning interval")
    for index, episode in enumerate(episodes, start=1):
        print(
            f"[replay] [{index}/{len(episodes)}] seed={episode.seed:06d} running...",
            flush=True,
        )
        replay_suite = replace(
            suite,
            expert_prefix=None,  # Saved actions already include the expert prefix.
            seeds=(episode.seed,),
            rollout=replace(
                suite.rollout,
                max_steps=len(episode.actions),
                replan_steps=base_replan,
                action_repeat=1,
                stop_on_success=False,
                ignore_done=True,
            ),
        )
        client = ReplayPolicyClient(
            episode.actions,
            contract=contract,
            chunk_size=base_replan,
        )
        adapter = ReplayEvaluationAdapter(
            episode,
            require_identity=require_identity,
        )
        result = RolloutRunner(
            client=client,
            suite=replay_suite,
            artifacts=writer,
        ).run_episode(adapter, episode.seed)
        results.append(result)
        detail = (
            f"step={result.success_step}"
            if result.success
            else f"reason={result.failure_reason or result.error}"
        )
        print(
            f"[replay] [{index}/{len(episodes)}] seed={episode.seed:06d} "
            f"{result.status.upper()} {detail} steps={result.env_steps}",
            flush=True,
        )
    writer.write_summary()
    return results


def replay_policy_contract(*, horizon: int = 50) -> PolicyContract:
    """Build the canonical local-policy contract used for dataset replay."""

    return PolicyContract(
        server_id="recorded-actions",
        policy={"family": "action-replay", "profile": REPLAY_PROFILE},
        checkpoint=CheckpointIdentity(
            uri="replay://recorded-actions",
            fingerprint="recorded-actions",
        ),
        observation=ObservationContract(
            state_schema=STATE_SCHEMA,
            state_dim=STATE_DIM,
            required_cameras=(
                "agentview_left",
                "agentview_right",
                "eye_in_hand",
            ),
            image_size=(224, 224),
        ),
        action=ActionContract(
            schema=ACTION_SCHEMA,
            dim=13,
            horizon=max(1, int(horizon)),
        ),
        control_frequency_hz=20.0,
    )


def parse_seed_spec(value: str | None) -> set[int]:
    if value is None:
        return set()
    seeds: set[int] = set()
    for token in str(value).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, stop = (int(part) for part in token.split("-", 1))
            step = 1 if stop >= start else -1
            seeds.update(range(start, stop + step, step))
        else:
            seeds.add(int(token))
    return seeds


def load_npz_actions(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if "actions" not in payload:
            raise ValueError(f"{path} does not contain an actions array")
        return validate_action_chunk(np.asarray(payload["actions"], dtype=np.float32))


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def canonical_task_id(value: str) -> str:
    return resolve_builtin_task_plugin(str(value)).canonical_id


def task_suite_path(task_id: str) -> Path:
    name = canonical_task_id(task_id).replace("-", "").lower()
    directory = ROOT / "configs" / "eval" / "suites"
    candidates = [
        (int(version), path)
        for path in directory.glob(f"{name}_v*.yaml")
        if (version := path.stem.removeprefix(f"{name}_v")).isdigit()
    ]
    return max(candidates)[1] if candidates else directory / f"{name}_v1.yaml"


def load_source_suite(manifest: Mapping[str, Any], task_id: str) -> EvalSuiteConfig:
    suite_value = manifest.get("suite", {})
    raw_path = suite_value.get("path") if isinstance(suite_value, Mapping) else None
    candidates: list[Path] = []
    if raw_path:
        path = Path(str(raw_path)).expanduser()
        candidates.append(path if path.is_absolute() else ROOT / path)
        candidates.append(ROOT / "configs" / "eval" / "legacy_suites" / path.name)
    candidates.append(task_suite_path(task_id))
    for candidate in candidates:
        if candidate.is_file():
            suite = load_eval_suite(candidate)
            if canonical_task_id(suite.task_id) != canonical_task_id(task_id):
                raise ValueError(f"suite task {suite.task_id} does not match replay task {task_id}")
            return suite
    raise FileNotFoundError("could not resolve source or canonical evaluation suite")


def request_from_metadata(
    *,
    task_id: str,
    seed: int,
    metadata: Mapping[str, Any],
    defaults: EpisodeRequest,
    task_config_file: Path | None = None,
    observation_profile_file: Path | None = None,
) -> tuple[EpisodeRequest, str | None]:
    resolved_value = metadata.get("resolved_episode", {})
    resolved = dict(resolved_value) if isinstance(resolved_value, Mapping) else {}
    resolved_task = resolved.get("task_id")
    if resolved_task and canonical_task_id(str(resolved_task)) != canonical_task_id(task_id):
        raise ValueError(
            f"recorded task {resolved_task!r} does not match requested task {task_id!r}"
        )
    resolved_seed = resolved.get("seed")
    if resolved_seed is not None and int(resolved_seed) != int(seed):
        raise ValueError(f"recorded seed {resolved_seed} does not match source seed {seed}")
    options = resolved.get("options", defaults.options)
    if not isinstance(options, Mapping):
        raise ValueError("resolved episode options must be a mapping")
    request = EpisodeRequest(
        task_id=canonical_task_id(task_id),
        seed=int(seed),
        variant=str(resolved.get("variant", metadata.get("variant", defaults.variant))),
        split=str(resolved.get("split", metadata.get("split", defaults.split))),
        env_split=str(resolved.get("env_split", metadata.get("env_split", defaults.env_split))),
        options=dict(options),
        task_config_file=(
            task_config_file
            or _optional_path(resolved.get("task_config_file"))
            or defaults.task_config_file
        ),
        observation_profile_file=(
            observation_profile_file
            or _optional_path(resolved.get("observation_profile_file"))
            or defaults.observation_profile_file
        ),
        allow_partial_calibration=False,
    )
    identity = metadata.get("episode_identity") or resolved.get("identity")
    return request, None if identity is None else str(identity)


def suite_request_defaults(suite: EvalSuiteConfig, seed: int) -> EpisodeRequest:
    raw = dict(suite.task_options)
    options = raw.get("task_options", {})
    if not isinstance(options, Mapping):
        raise ValueError("suite task.options.task_options must be a mapping")
    return EpisodeRequest(
        task_id=canonical_task_id(suite.task_id),
        seed=int(seed),
        variant=str(raw.get("variant", "memory")),
        split="evaluation",
        env_split=suite.env_split,
        options=dict(options),
        task_config_file=_optional_path(raw.get("task_config_file")),
        observation_profile_file=_optional_path(raw.get("observation_profile_file")),
        allow_partial_calibration=False,
    )


def discover_recorded_episodes(
    dataset_root: Path,
    task_id: str,
    seeds: set[int],
    *,
    source_status: str,
) -> list[dict[str, Any]]:
    """Discover one selected native recording per seed."""

    if source_status not in {"all", "success", "failed"}:
        raise ValueError("source_status must be all, success, or failed")
    task_id = canonical_task_id(task_id)
    manifest_items = _recording_manifest_items(dataset_root, task_id)
    branches = {
        "success": dataset_root / "train",
        "failed": dataset_root / "debug_failed",
    }
    statuses = ("success", "failed") if source_status == "all" else (source_status,)
    candidates: dict[tuple[str, Path], dict[str, Any]] = {}
    for status in statuses:
        branch = branches[status]
        if not branch.is_dir():
            continue
        for task_dir in branch.iterdir():
            if not task_dir.is_dir() or _task_key(task_dir.name) != _task_key(task_id):
                continue
            for episode_dir in sorted(task_dir.glob("seed_*_attempt_*")):
                trajectory = episode_dir / "trajectory.h5"
                if not trajectory.is_file():
                    continue
                seed, attempt = _episode_key(episode_dir)
                if seeds and seed not in seeds:
                    continue
                manifest_item = manifest_items.get((seed, attempt), {})
                candidates[(status, episode_dir.resolve())] = {
                    "seed": seed,
                    "attempt": attempt,
                    "episode_dir": episode_dir.resolve(),
                    "source_status": status,
                    "source_success": status == "success",
                    "manifest_selected": bool(manifest_item),
                    "episode_identity": manifest_item.get("episode_identity"),
                }
    for (seed, attempt), manifest_item in manifest_items.items():
        if seeds and seed not in seeds:
            continue
        status = "success" if manifest_item.get("success", True) else "failed"
        if status not in statuses:
            continue
        episode_dir = _manifest_episode_dir(
            dataset_root,
            task_id=task_id,
            seed=seed,
            attempt=attempt,
            status=status,
            value=manifest_item.get("episode_dir"),
        )
        if episode_dir is None:
            continue
        key = (status, episode_dir)
        existing = candidates.get(key, {})
        candidates[key] = {
            "seed": seed,
            "attempt": attempt,
            "episode_dir": episode_dir,
            "source_status": status,
            "source_success": status == "success",
            "manifest_selected": True,
            "episode_identity": (
                manifest_item.get("episode_identity") or existing.get("episode_identity")
            ),
        }
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidates.values():
        by_seed.setdefault(int(candidate["seed"]), []).append(candidate)
    selected = [
        max(
            values,
            key=lambda item: (
                bool(item["manifest_selected"]),
                bool(item["source_success"]),
                int(item["attempt"]),
            ),
        )
        for values in by_seed.values()
    ]
    return sorted(selected, key=lambda item: int(item["seed"]))


def load_recording_metadata(episode_dir: Path) -> dict[str, Any]:
    path = episode_dir / "episode.json"
    if not path.is_file():
        return {}
    payload = load_json(path)
    metadata_value = payload.get("metadata", {})
    metadata = dict(metadata_value) if isinstance(metadata_value, Mapping) else {}
    metadata["recorded_step_count"] = payload.get("step_count")
    metadata["recorded_termination"] = payload.get("termination")
    return metadata


def _recording_manifest_items(
    dataset_root: Path,
    task_id: str,
) -> dict[tuple[int, int], dict[str, Any]]:
    path = dataset_root / "manifest.json"
    if not path.is_file():
        return {}
    payload = load_json(path)
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for raw in payload.get("items", ()):
        if not isinstance(raw, Mapping):
            continue
        value = raw.get("task_id")
        if value is None or _task_key(str(value)) != _task_key(task_id):
            continue
        key = (int(raw["seed"]), int(raw.get("attempt", 0)))
        result[key] = dict(raw)
    return result


def _episode_key(path: Path) -> tuple[int, int]:
    parts = path.name.removeprefix("seed_").split("_attempt_", 1)
    if len(parts) != 2:
        raise ValueError(f"invalid recorded episode directory: {path.name}")
    return int(parts[0]), int(parts[1])


def _manifest_episode_dir(
    dataset_root: Path,
    *,
    task_id: str,
    seed: int,
    attempt: int,
    status: str,
    value: Any,
) -> Path | None:
    candidates: list[Path] = []
    if value:
        path = Path(str(value)).expanduser()
        candidates.append(path if path.is_absolute() else dataset_root / path)
    branch = "train" if status == "success" else "debug_failed"
    candidates.append(dataset_root / branch / task_id / f"seed_{seed:06d}_attempt_{attempt:02d}")
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "trajectory.h5").is_file():
            return resolved
    return None


def _optional_path(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _task_key(value: str) -> str:
    return str(value).upper().replace("-", "").replace("_", "")


__all__ = [
    "DATASET_REPLAY_MODE",
    "RUN_REPLAY_MODE",
    "ReplayEpisode",
    "ReplayEvaluationAdapter",
    "ReplayPolicyClient",
    "canonical_task_id",
    "discover_recorded_episodes",
    "load_json",
    "load_npz_actions",
    "load_recording_metadata",
    "load_source_suite",
    "parse_seed_spec",
    "replay_policy_contract",
    "request_from_metadata",
    "run_action_replay",
    "suite_request_defaults",
    "task_suite_path",
]
