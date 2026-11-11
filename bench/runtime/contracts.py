"""RoboCasa episode lifecycle contracts shared by recording and evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from bench.tasks.contracts import ResolvedEpisode


CONTROL_FREQUENCY_HZ = 20


@dataclass(frozen=True, slots=True)
class SensorRequirements:
    """Consumer requirements that validate, but never mutate, a profile."""

    required_camera_aliases: tuple[str, ...] = ()
    required_image_size: tuple[int, int] | None = None
    require_depth: bool = False


@dataclass(frozen=True, slots=True)
class ObservationProfile:
    """Fully resolved simulator observation configuration."""

    profile_id: str
    schema_version: int
    camera_aliases: tuple[str, ...]
    camera_names: tuple[str, ...]
    image_size: tuple[int, int]
    camera_depths: bool
    has_renderer: bool
    has_offscreen_renderer: bool
    use_camera_observations: bool
    source_path: Path
    content_hash: str

    @property
    def height(self) -> int:
        return self.image_size[0]

    @property
    def width(self) -> int:
        return self.image_size[1]

    def physical_camera(self, alias: str) -> str:
        try:
            index = self.camera_aliases.index(alias)
        except ValueError as exc:
            raise KeyError(f"observation profile has no camera alias {alias!r}") from exc
        return self.camera_names[index]

    def observation_key(self, alias: str, *, modality: str = "rgb") -> str:
        suffix = "image" if modality == "rgb" else "depth"
        return f"{self.physical_camera(alias)}_{suffix}"


@dataclass(frozen=True, slots=True)
class EnvironmentDescriptor:
    """Task-owned, simulator-independent description of a RoboCasa scene."""

    env_name: str
    env_split: str
    scene_seed: int
    layout_id: int | None
    style_id: int | None
    env_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RuntimeDescriptor:
    """All task-local facts required to construct and initialize one runtime."""

    episode: ResolvedEpisode
    config: Any
    scene: Any
    episode_plan: Any
    actions: Any
    navigation: Any
    sensors: ObservationProfile
    prompt: Any
    resources: Mapping[str, Path]
    environment: EnvironmentDescriptor
    release_facts: Mapping[str, Any] = field(default_factory=dict)
    execute: bool = True
    control_frequency_hz: int = CONTROL_FREQUENCY_HZ

    def __post_init__(self) -> None:
        if not isinstance(self.episode, ResolvedEpisode):
            raise TypeError("RuntimeDescriptor.episode must be ResolvedEpisode")
        if self.actions is None:
            raise ValueError(
                f"RuntimeDescriptor for {self.episode.task_id} has no resolved ActionConfig"
            )
        object.__setattr__(self, "execute", bool(self.execute))


@dataclass(slots=True)
class InitializedTaskRuntime:
    """Task-owned state created after the shared RoboCasa reset."""

    descriptor: RuntimeDescriptor
    env: Any
    observation: Mapping[str, Any]
    targets: Any = None
    task: Any = None
    runner: Any = None
    initialization: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def episode(self) -> ResolvedEpisode:
        return self.descriptor.episode

@dataclass(frozen=True, slots=True)
class TaskRuntimeAPI:
    """Explicit task runtime operations consumed by the shared factory."""

    build_descriptor: Callable[[ResolvedEpisode, bool], RuntimeDescriptor]
    initialize_runtime: Callable[
        [RuntimeDescriptor, Any, Mapping[str, Any], Mapping[str, Any]],
        InitializedTaskRuntime,
    ]
    check_success: Callable[[InitializedTaskRuntime, Mapping[str, Any]], bool]
    diagnostics: Callable[[InitializedTaskRuntime], Mapping[str, Any]]
    close_runtime: Callable[[InitializedTaskRuntime], None]
    report_stage_profile_id: str | None = None
    report_stages: Callable[[InitializedTaskRuntime], Mapping[str, Any]] | None = None
    build_expert_runtime: Callable[
        [RuntimeDescriptor, Any, Mapping[str, Any]], Mapping[str, Any]
    ] | None = None
    build_subtask_scheduler: Callable[[Any, str], Any] | None = None
    build_evaluation_prefix: Callable[[RuntimeDescriptor, str], Any] | None = None
    validate_evaluation_prefix: Callable[[InitializedTaskRuntime, str], Mapping[str, Any]] | None = None


@dataclass(frozen=True, slots=True)
class InitializationReport:
    """Structured reset/initialization result shared by both product lines."""

    status: str
    reason: str = ""
    facts: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def noop(cls, *, reason: str) -> "InitializationReport":
        return cls(status="noop", reason=str(reason))

    @classmethod
    def applied(cls, *, facts: Mapping[str, Any] | None = None) -> "InitializationReport":
        return cls(status="applied", facts=dict(facts or {}))


@dataclass(frozen=True, slots=True)
class TaskDiagnostics:
    """Task semantics after a reset or transition.

    ``stop_execution`` is an explicit, irreversible task stop, including a
    successful outcome. Consumers must not execute another action until reset,
    even when ordinary success or environment-done stopping is disabled.
    """

    success: bool
    terminal: bool = False
    failure_code: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
    stop_execution: bool = False


@dataclass(frozen=True, slots=True)
class RuntimeStep:
    """Normalized result of one environment transition."""

    observation: Mapping[str, Any]
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: TaskDiagnostics | None = None

    @property
    def terminal(self) -> bool:
        return bool(
            self.terminated
            or self.truncated
            or (
                self.diagnostics is not None
                and (self.diagnostics.terminal or self.diagnostics.stop_execution)
            )
        )


class EpisodeRuntime(Protocol):
    """Active environment port consumed by recording and evaluation."""

    episode: ResolvedEpisode
    observation: Mapping[str, Any]

    def reset(self) -> Mapping[str, Any]: ...

    def initialize(self) -> InitializationReport: ...

    def observe(self) -> Mapping[str, Any]: ...

    def step(self, action: Sequence[float] | Any) -> RuntimeStep: ...

    def diagnostics(self) -> TaskDiagnostics: ...

    def close(self) -> None: ...


class RuntimeFactory(Protocol):
    def create(
        self,
        episode: ResolvedEpisode,
        *,
        execute: bool = True,
        expert: bool = False,
    ) -> EpisodeRuntime: ...

    def create_from_descriptor(
        self,
        descriptor: RuntimeDescriptor,
        *,
        expert: bool = False,
    ) -> EpisodeRuntime: ...


__all__ = [
    "CONTROL_FREQUENCY_HZ",
    "EnvironmentDescriptor",
    "EpisodeRuntime",
    "InitializationReport",
    "InitializedTaskRuntime",
    "ObservationProfile",
    "RuntimeFactory",
    "RuntimeDescriptor",
    "RuntimeStep",
    "SensorRequirements",
    "TaskDiagnostics",
    "TaskRuntimeAPI",
]
