"""Load a verified, fixed expert prefix for an ordinary evaluation episode."""

from __future__ import annotations

import hashlib
import json

from bench.evaluation.expert_prefix import load_expert_prefix


class EvaluationPrefix:
    def __init__(self, artifact, validate_handoff, runtime):
        self.actions = artifact.actions
        self._validate = validate_handoff
        self.runtime = runtime
        self.index = 0
        self.metadata = {
            "profile": artifact.provenance["evaluation_prefix_profile"],
            "valid": False,
            "planned_steps": len(self.actions),
            "executed_steps": 0,
            "action_sha256": hashlib.sha256(self.actions.tobytes()).hexdigest(),
            "expert_plan_version": artifact.expert_plan_version,
        }
        runtime.result_metadata["expert_prefix"] = self.metadata

    @property
    def complete(self):
        return self.index == len(self.actions)

    def pop(self):
        return self.actions[self.index].copy()

    def after_step(self):
        self.index += 1
        self.metadata["executed_steps"] = self.index

    def handoff(self, env_steps):
        if not self.complete:
            raise ValueError("expert prefix ended before its frozen boundary")
        answer = self._validate(
            self.runtime.metadata["initialized_task_runtime"], self.metadata["profile"]
        )
        self.metadata.update(valid=True, handoff_step=env_steps, answer=dict(answer))


def load_evaluation_prefix(config, episode):
    artifact = load_expert_prefix(config.root, episode.seed)
    if (artifact.task_id != episode.task_id or artifact.seed != episode.seed
            or artifact.episode_identity != episode.identity):
        raise ValueError("expert prefix identity does not match the evaluation episode")
    if not artifact.success:
        raise ValueError(f"invalid expert prefix: {artifact.failure_reason}")
    if len(artifact.actions) == 0:
        raise ValueError("expert prefix must contain actions")
    if artifact.provenance.get("evaluation_prefix_profile") != config.profile:
        raise ValueError("expert prefix was not built for this formal handoff profile")
    if artifact.stage_boundary_steps != {config.profile: len(artifact.actions)}:
        raise ValueError("expert prefix must end exactly at the formal handoff boundary")
    return artifact


def evaluation_prefix_inventory(suite):
    """Validate every frozen input before a run and fingerprint resume identity."""
    if suite.expert_prefix is None:
        return None
    from bench.evaluation.canonical_stage import resolve_suite_episode

    entries = []
    for seed in suite.seeds:
        artifact = load_evaluation_prefix(
            suite.expert_prefix, resolve_suite_episode(suite, seed)
        )
        entries.append({
            "seed": seed,
            "episode_identity": artifact.episode_identity,
            "expert_plan_version": artifact.expert_plan_version,
            "action_count": len(artifact.actions),
            "action_sha256": hashlib.sha256(artifact.actions.tobytes()).hexdigest(),
        })
    digest = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"sha256": digest, "episodes": entries}


def validate_prefix_configuration(api, descriptor, config):
    if config is None:
        return
    if api.build_evaluation_prefix is None or api.validate_evaluation_prefix is None:
        raise ValueError(f"{descriptor.episode.task_id} does not support expert_prefix")
    api.build_evaluation_prefix(descriptor, config.profile)
