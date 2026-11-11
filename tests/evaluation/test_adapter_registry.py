from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

from bench.evaluation.adapter import EpisodeRuntime, get_adapter, preflight_adapter
from bench.evaluation.canonical_adapter import CanonicalEvaluationAdapter
from bench.evaluation.observation import build_canonical_observation
from bench.evaluation.report_profiles import REPORT_STAGE_PROFILES
from bench.evaluation.replay import task_suite_path
from bench.evaluation.suite import (
    ArtifactConfig,
    EvalSuiteConfig,
    RolloutConfig,
    load_eval_suite,
)
from bench.policy.protocol import (
    ActionContract,
    CheckpointIdentity,
    ObservationContract,
    PolicyContract,
)
from bench.policy.protocol.contracts import ACTION_SCHEMA
from bench.policy.protocol.errors import ProtocolError
from bench.tasks import BUILTIN_TASK_PLUGINS


_BENCHMARK_TASK_IDS = {profile.task_id for profile in REPORT_STAGE_PROFILES}


def _suite_path(task_id: str) -> Path:
    name = task_id.lower().replace("-", "")
    if task_id in _BENCHMARK_TASK_IDS:
        return task_suite_path(task_id)
    return Path("configs/eval/legacy_suites") / f"{name}_smoke_v1.yaml"


def _contract(
    *,
    image_size: tuple[int, int] = (2, 3),
    cameras: tuple[str, ...] = ("agentview_left", "agentview_right", "eye_in_hand"),
    modalities: tuple[str, ...] = ("rgb", "state", "prompt"),
    frequency: float = 20.0,
) -> PolicyContract:
    return PolicyContract(
        server_id="fake",
        policy={"family": "fake"},
        checkpoint=CheckpointIdentity("test://fake", "sha256:fake"),
        observation=ObservationContract(
            "membench.robot_state37.v2",
            37,
            cameras,
            image_size,
            modalities,
        ),
        action=ActionContract(ACTION_SCHEMA, 13, 2),
        control_frequency_hz=frequency,
    )


def _suite(config_file: Path, *, record_video: bool = True) -> EvalSuiteConfig:
    return EvalSuiteConfig(
        suite_version="fake_v1",
        task_id="FAKE-01",
        env_split="pretrain",
        seeds=(1,),
        rollout=RolloutConfig(max_steps=3, replan_steps=2),
        artifacts=ArtifactConfig(
            record_video=record_video,
            video_cameras=("eye_in_hand",),
        ),
        task_options={"task_config_file": str(config_file)},
    )


def test_all_builtin_tasks_resolve_to_an_evaluation_adapter() -> None:
    adapters = {plugin.canonical_id: get_adapter(plugin.canonical_id) for plugin in BUILTIN_TASK_PLUGINS}

    assert set(adapters) == {plugin.canonical_id for plugin in BUILTIN_TASK_PLUGINS}
    assert all(isinstance(adapter, CanonicalEvaluationAdapter) for adapter in adapters.values())


@pytest.mark.contract
@pytest.mark.parametrize(
    "task_id",
    [
        "TM01", "TM02", "TM04", "TS01", "TS02", "TS03", "TS04",
        "WA-01", "WA-02", "WA-03", "WA-04", "WA-05", "WR-01", "WR-02", "WR-03", "WR-04",
        "WR-05", "WR-06", "WR-07", "WR-08", "WR-11", "WX-01", "WX-02",
    ],
)
def test_all_calibrated_preflights_consume_the_versioned_observation_profile(
    task_id: str,
) -> None:
    suite = load_eval_suite(_suite_path(task_id))
    adapter = get_adapter(task_id)

    report = preflight_adapter(
        adapter,
        suite=suite,
        policy_contract=_contract(image_size=(224, 224)),
    )

    assert report["task"]["adapter_mode"] == "resolved_episode_runtime"
    assert report["task"]["observation_profile"] == "rgbd_v3"
    with pytest.raises(ProtocolError, match="image_size"):
        preflight_adapter(
            adapter,
            suite=suite,
            policy_contract=_contract(image_size=(240, 320)),
        )


def test_tm03_production_preflight_rejects_incomplete_calibration() -> None:
    from bench.tasks.tm03.config import TM03CalibrationError

    suite = load_eval_suite(
        Path("configs/eval/legacy_suites/tm03_smoke_v1.yaml")
    )
    with pytest.raises(TM03CalibrationError):
        preflight_adapter(
            get_adapter("TM03"),
            suite=suite,
            policy_contract=_contract(image_size=(224, 224)),
        )


def test_unknown_task_is_rejected_before_environment_creation() -> None:
    with pytest.raises(KeyError, match="available"):
        get_adapter("NOT-A-TASK")


def test_preflight_rejects_unimplemented_modality_and_frequency(tmp_path: Path) -> None:
    adapter = _FakeCanonicalAdapter()
    suite = _suite(tmp_path / "task.yaml", record_video=False)

    with pytest.raises(ValueError, match="not implemented"):
        preflight_adapter(
            adapter,
            suite=suite,
            policy_contract=_contract(modalities=("rgb", "state", "prompt", "depth")),
        )
    with pytest.raises(ValueError, match="control_frequency_hz"):
        preflight_adapter(
            adapter,
            suite=suite,
            policy_contract=_contract(frequency=10.0),
        )


class _FakeEnv:
    def __init__(self) -> None:
        self.closed = False
        self.observation = {
            "robot0_arm_joint_qpos": np.zeros(7),
            "robot0_arm_joint_qvel": np.zeros(7),
            "robot0_gripper_qpos": np.zeros(2),
            "robot0_gripper_qvel": np.zeros(2),
            "robot0_base_pos": np.zeros(3),
            "robot0_base_quat": np.asarray([0, 0, 0, 1]),
            "robot0_base_joint_qvel": np.zeros(3),
            "robot0_torso_qpos": np.zeros(1),
            "robot0_torso_qvel": np.zeros(1),
            "robot0_base_to_eef_pos": np.zeros(3),
            "robot0_base_to_eef_quat_site": np.asarray([0, 0, 0, 1]),
            "robot0_agentview_left_image": np.zeros((2, 3, 3), dtype=np.uint8),
            "robot0_agentview_right_image": np.zeros((2, 3, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.ones((2, 3, 3), dtype=np.uint8),
        }

    def _get_observations(self, force_update: bool = False):
        del force_update
        return self.observation

    def get_ep_meta(self):
        return {"lang": "finish the fake task"}

    def _check_success(self):
        return True

    def task_diagnostics(self):
        return {"success": True}

    def close(self):
        self.closed = True


class _FakeCanonicalAdapter:
    task_id = "FAKE-01"
    control_frequency_hz = 20.0

    def __init__(self, env: _FakeEnv | None = None) -> None:
        self.env = env or _FakeEnv()
        self.contract: PolicyContract | None = None

    def validate_suite(self, **_: Any) -> Mapping[str, Any]:
        return {"adapter_mode": "resolved_episode_runtime"}

    def create_episode(
        self,
        *,
        seed: int,
        policy_contract: PolicyContract,
        suite: EvalSuiteConfig,
    ) -> EpisodeRuntime:
        del suite
        self.contract = policy_contract
        return EpisodeRuntime(
            episode_id=f"fake-{seed}",
            seed=seed,
            env=self.env,
            observation=self.env.observation,
            prompt="finish the fake task",
            result_metadata={"prompt": "finish the fake task"},
        )

    def build_observation(self, runtime: EpisodeRuntime):
        assert self.contract is not None
        return build_canonical_observation(
            runtime.observation,
            task_id=self.task_id,
            prompt=runtime.prompt,
            contract=self.contract,
            camera_aliases={
                "agentview_left": "robot0_agentview_left_image",
                "agentview_right": "robot0_agentview_right_image",
                "eye_in_hand": "robot0_eye_in_hand_image",
            },
        )

    def artifact_frames(
        self,
        runtime: EpisodeRuntime,
        cameras: tuple[str, ...],
    ) -> Mapping[str, np.ndarray]:
        aliases = {
            "agentview_right": "robot0_agentview_right_image",
            "eye_in_hand": "robot0_eye_in_hand_image",
        }
        return {alias: np.asarray(runtime.observation[aliases[alias]]) for alias in cameras}

    def check_success(self, runtime: EpisodeRuntime, info: Mapping[str, Any]) -> bool:
        del info
        return bool(runtime.env._check_success())

    def diagnostics(self, runtime: EpisodeRuntime) -> Mapping[str, Any]:
        return runtime.env.task_diagnostics()

    def close(self, runtime: EpisodeRuntime) -> None:
        runtime.env.close()


def test_canonical_adapter_separates_policy_and_video_cameras(tmp_path: Path) -> None:
    config_file = tmp_path / "task.yaml"
    config_file.write_text("task: fake\n", encoding="utf-8")
    env = _FakeEnv()
    adapter = _FakeCanonicalAdapter(env)
    suite = _suite(config_file)
    contract = _contract()

    runtime = adapter.create_episode(seed=1, policy_contract=contract, suite=suite)
    observation = adapter.build_observation(runtime)
    frames = adapter.artifact_frames(runtime, suite.artifacts.video_cameras)

    assert set(observation.images) == {
        "agentview_left",
        "agentview_right",
        "eye_in_hand",
    }
    assert set(frames) == {"eye_in_hand"}
    assert runtime.result_metadata["prompt"] == "finish the fake task"
    adapter.close(runtime)
    assert env.closed
