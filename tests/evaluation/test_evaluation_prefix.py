from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from bench.evaluation.adapter import EpisodeRuntime
from bench.evaluation.artifacts import ArtifactWriter
from bench.evaluation.expert_prefix import ExpertPrefixArtifact, write_expert_prefix
from bench.evaluation.prefix import (
    EvaluationPrefix, evaluation_prefix_inventory, load_evaluation_prefix,
)
from bench.evaluation.rollout import RolloutRunner
from bench.evaluation.suite import (
    ArtifactConfig, EvalSuiteConfig, ExpertPrefixConfig, RolloutConfig, validate_eval_suite,
)
from bench.policy.protocol import CanonicalObservation, InferenceResponse
from bench.evaluation.replay import replay_policy_contract


class Client:
    contract = replay_policy_contract(horizon=2)

    def __init__(self):
        self.resets = []
        self.steps = []
        self.observations = []

    def reset(self, episode_id):
        self.resets.append(episode_id)

    def infer(self, observation, *, episode_id, step):
        self.steps.append(step)
        self.observations.append(observation)
        # This prefix proposal must never execute, including a buffered tail.
        actions = np.full((2, 13), 0.1 if step < 3 else 0.2, dtype=np.float32)
        return InferenceResponse(request_id=str(step), episode_id=episode_id, actions=actions)


class Adapter:
    task_id = "WA-01"
    control_frequency_hz = 20.0

    def __init__(self, invalid=False):
        self.executed = []
        self.handoffs = 0
        self.invalid = invalid
        self.closed = False

    def create_episode(self, *, seed, **kwargs):
        return EpisodeRuntime("wa01-seed-001050", seed, self, {}, "full instruction")

    def step(self, action):
        self.executed.append(action.copy())
        return {}, 0.0, False, {}

    def create_evaluation_prefix(self, runtime, suite):
        artifact = ExpertPrefixArtifact(
            task_id=self.task_id, seed=runtime.seed, episode_identity="scene",
            expert_plan_version="v1", stage_boundary_steps={"sugar": 3},
            actions=np.zeros((3, 13), dtype=np.float32), success=True,
            provenance={"evaluation_prefix_profile": "sugar"},
        )
        runtime.metadata["initialized_task_runtime"] = runtime

        def validate(initialized, profile):
            self.handoffs += 1
            if self.invalid:
                raise ValueError("invalid physical prefix")
            return {"sugared_coffee_id": "coffee_A"}

        return EvaluationPrefix(artifact, validate, runtime)

    def build_observation(self, runtime):
        return CanonicalObservation(
            self.task_id, np.zeros(37, dtype=np.float32),
            {name: np.zeros((224, 224, 3), dtype=np.uint8)
             for name in self_contract.observation.required_cameras}, runtime.prompt,
        )

    def check_success(self, runtime, info):
        return len(self.executed) >= 5

    def diagnostics(self, runtime):
        return {}

    def close(self, runtime):
        self.closed = True


self_contract = replay_policy_contract(horizon=2)


def suite(tmp_path, budget=2):
    return EvalSuiteConfig(
        "prefix-v2", "WA-01", "pretrain", (1050,), RolloutConfig(budget, 2),
        artifacts=ArtifactConfig(record_video=False),
        expert_prefix=ExpertPrefixConfig("sugar", tmp_path),
    )


def test_single_handoff_preserves_history_discards_predictions_and_counts_policy_budget(tmp_path):
    client, adapter = Client(), Adapter()
    writer = ArtifactWriter(tmp_path / "run", expected_seeds=(1050,))
    result = RolloutRunner(client=client, suite=suite(tmp_path), artifacts=writer).run_episode(adapter, 1050)
    assert result.status == "success", result.error
    assert result.env_steps == 5 and result.policy_actions_consumed == 2
    assert len(client.resets) == 1 and client.steps == [0, 2, 3]
    assert adapter.handoffs == 1 and adapter.closed
    assert all(np.all(a == 0) for a in adapter.executed[:3])
    assert all(np.all(a[:7] == np.float32(0.2)) for a in adapter.executed[3:])
    assert all(o.prompt == "full instruction" for o in client.observations)
    assert all(not hasattr(o, "answer") for o in client.observations)
    metadata = result.episode_metadata["expert_prefix"]
    assert metadata["valid"] and metadata["handoff_step"] == 3
    assert metadata["answer"] == {"sugared_coffee_id": "coffee_A"}
    saved = np.load(writer.episodes_dir / "seed_001050/actions.npz")
    np.testing.assert_array_equal(saved["actions"], adapter.executed)


def test_failed_handoff_is_error_without_executing_model_actions(tmp_path):
    adapter = Adapter(invalid=True)
    result = RolloutRunner(client=Client(), suite=suite(tmp_path)).run_episode(adapter, 1050)
    assert result.status == "error" and "invalid physical prefix" in result.error
    assert result.policy_actions_consumed == 0 and len(adapter.executed) == 3
    assert not result.episode_metadata["expert_prefix"]["valid"]


def test_missing_prefix_never_falls_back_to_model_control(tmp_path):
    adapter = Adapter()

    def missing(*args):
        raise FileNotFoundError("missing frozen prefix")

    adapter.create_evaluation_prefix = missing
    result = RolloutRunner(client=Client(), suite=suite(tmp_path)).run_episode(adapter, 1050)
    assert result.status == "error" and "missing frozen prefix" in result.error
    assert not adapter.executed and adapter.closed


def test_policy_budget_expires_after_handoff(tmp_path):
    adapter = Adapter()
    result = RolloutRunner(client=Client(), suite=suite(tmp_path, budget=1)).run_episode(adapter, 1050)
    assert result.status == "failure" and result.failure_reason == "max_steps"
    assert result.env_steps == 4 and result.policy_actions_consumed == 1
    assert result.episode_metadata["expert_prefix"]["valid"]


def test_saved_action_replay_does_not_insert_prefix_twice(tmp_path, monkeypatch):
    from bench.evaluation import replay

    captured = []

    class Runner:
        def __init__(self, *, client, suite, artifacts):
            captured.append(suite)

        def run_episode(self, adapter, seed):
            return SimpleNamespace(success=True, success_step=5, status="success", env_steps=5)

    monkeypatch.setattr(replay, "RolloutRunner", Runner)
    monkeypatch.setattr(replay, "ReplayEvaluationAdapter", lambda *args, **kwargs: None)
    replay.run_action_replay(
        [SimpleNamespace(seed=1050, actions=np.zeros((5, 13), dtype=np.float32))],
        suite=suite(tmp_path), contract=self_contract,
        writer=SimpleNamespace(write_summary=lambda: None), require_identity=True,
    )
    assert captured[0].expert_prefix is None
    assert captured[0].rollout.max_steps == 5


@pytest.mark.parametrize("override", [
    {"task_options": {"prompt_scheduler": "oracle_subtasks_v1"}},
    {"rollout": RolloutConfig(2, 1, action_repeat=2)},
])
def test_prefix_rejects_prompt_switching_and_repeated_actions(tmp_path, override):
    with pytest.raises(ValueError, match="expert_prefix"):
        validate_eval_suite(replace(suite(tmp_path), **override))


def test_inventory_validates_all_seeds_and_fingerprints_frozen_actions(tmp_path, monkeypatch):
    import bench.evaluation.canonical_stage as resolver

    monkeypatch.setattr(resolver, "resolve_suite_episode", lambda suite, seed:
                        SimpleNamespace(seed=seed, task_id="WA-01", identity="scene"))
    config = replace(suite(tmp_path), seeds=(1050, 1051))
    for seed in config.seeds:
        write_expert_prefix(tmp_path, ExpertPrefixArtifact(
            "WA-01", seed, "scene", "v1", {"sugar": 3},
            np.zeros((3, 13), dtype=np.float32), True,
            provenance={"evaluation_prefix_profile": "sugar"},
        ))
    first = evaluation_prefix_inventory(config)
    assert [entry["seed"] for entry in first["episodes"]] == [1050, 1051]
    (tmp_path / "seed_001051/actions.npz").unlink()
    with pytest.raises(FileNotFoundError):
        evaluation_prefix_inventory(config)
    write_expert_prefix(tmp_path, ExpertPrefixArtifact(
        "WA-01", 1051, "scene", "v1", {"sugar": 3},
        np.full((3, 13), 0.1, dtype=np.float32), True,
        provenance={"evaluation_prefix_profile": "sugar"},
    ))
    assert evaluation_prefix_inventory(config)["sha256"] != first["sha256"]


def test_changed_frozen_actions_prevent_resuming_same_run(tmp_path):
    from scripts.eval.run_remote import _run_manifest

    config = suite(tmp_path)
    writer = ArtifactWriter(tmp_path / "run", expected_seeds=config.seeds)

    def manifest(fingerprint, config=config):
        return _run_manifest(
            run_id="prefix", suite=config, endpoint="ws://localhost:8000", contract={},
            env_name="test", preflight={"expert_prefix_inventory": {"sha256": fingerprint}},
        )

    writer.write_run_manifest(manifest("first"))
    writer.write_run_manifest(manifest("first"))
    with pytest.raises(ValueError, match="identity mismatch"):
        writer.write_run_manifest(manifest("second"))
    legacy = manifest("unused", replace(config, expert_prefix=None))
    assert "expert_prefix" not in legacy["rollout"]


@pytest.mark.parametrize("change", ["identity", "profile", "boundary", "failure"])
def test_prefix_rejects_stale_or_incompatible_artifact(tmp_path, change):
    artifact = ExpertPrefixArtifact(
        "WA-01", 1050, "scene", "v1", {"sugar": 3},
        np.zeros((3, 13), dtype=np.float32), True,
        provenance={"evaluation_prefix_profile": "sugar"},
    )
    if change == "identity": artifact = replace(artifact, episode_identity="other")
    if change == "profile": artifact = replace(artifact, provenance={"evaluation_prefix_profile": "other"})
    if change == "boundary": artifact = replace(artifact, stage_boundary_steps={"sugar": 2})
    if change == "failure": artifact = replace(artifact, success=False)
    write_expert_prefix(tmp_path, artifact)
    with pytest.raises(ValueError):
        load_evaluation_prefix(ExpertPrefixConfig("sugar", tmp_path),
                               SimpleNamespace(seed=1050, task_id="WA-01", identity="scene"))


@pytest.mark.parametrize("task,profile,last,subtasks", [
    ("WA-01", "wa01_sugar_v1", "return_after_place_sugar_to_regular_pose", 1),
    ("WA-05", "wa05_pick_fill_v1", "lift_after_sink_exit_cup_A", 2),
])
def test_formal_prefix_stops_at_complete_subtask_boundary(task, profile, last, subtasks):
    from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
    from bench.tasks.contracts import EpisodeRequest

    api = resolve_builtin_task_plugin(task).load_runtime_api()
    descriptor = api.build_descriptor(TaskCatalog.resolve(EpisodeRequest(task_id=task, seed=1050)), True)
    plan = api.build_evaluation_prefix(descriptor, profile)
    assert plan.stages[-1].name == last
    operations = [getattr(s, "operation", None) for s in plan.stages]
    assert "press_temperature_answer_button" not in operations
    assert "place_on_service_slot" not in operations
    assert "place_dynamic" not in operations
    assert ("fill_held_under_faucet" in operations) is (subtasks == 2)
