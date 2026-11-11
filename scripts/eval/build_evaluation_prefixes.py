#!/usr/bin/env python3
"""Build only the fixed expert subtasks used by formal WA evaluations."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench.evaluation.canonical_stage import resolve_suite_episode
from bench.evaluation.expert_prefix import ExpertPrefixArtifact, write_expert_prefix
from bench.evaluation.prefix import load_evaluation_prefix, validate_prefix_configuration
from bench.evaluation.suite import load_eval_suite
from bench.runtime import RoboCasaRuntimeFactory
from bench.runtime.data_contract import validate_action
from bench.tasks import resolve_builtin_task_plugin


def capture_prefix(shared, api, plan, config, provenance):
    actions = []
    original_step = shared.env.step
    stage_results = []

    def tapped_step(action):
        value = validate_action(action)
        transition = original_step(value)
        actions.append(value.copy())
        # Expert handlers step the env directly. Still advance the authoritative
        # evaluator on every control step, as the policy replay will do.
        shared.diagnostics()
        return transition

    shared.env.step = tapped_step
    try:
        result = shared.initialized.runner.run(
            plan,
            on_stage_end=lambda index, stage, value: stage_results.append({
                "name": stage.name, "success": value.success,
                "failure_code": value.failure_code,
            }),
        )
        if not result.success:
            raise RuntimeError(f"expert prefix failed: {stage_results[-1]}")
        answer = api.validate_evaluation_prefix(shared.initialized, config.profile)
        diagnostics = asdict(shared.diagnostics())
    finally:
        shared.env.step = original_step
    if not actions:
        raise ValueError("expert prefix executed no actions")
    return ExpertPrefixArtifact(
        task_id=shared.episode.task_id, seed=shared.episode.seed,
        episode_identity=shared.episode.identity, expert_plan_version=plan.version,
        stage_boundary_steps={config.profile: len(actions)},
        actions=np.stack(actions), success=True,
        diagnostics={**diagnostics, "answer": dict(answer), "expert_stages": stage_results},
        provenance={**provenance, "evaluation_prefix_profile": config.profile},
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    args = parser.parse_args(argv)
    suite = load_eval_suite(args.suite)
    config = suite.expert_prefix
    if config is None:
        parser.error("suite must configure expert_prefix")
    api = resolve_builtin_task_plugin(suite.task_id).load_runtime_api()
    for seed in args.seeds if args.seeds is not None else suite.seeds:
        episode = resolve_suite_episode(suite, seed)
        descriptor = api.build_descriptor(episode, True)
        validate_prefix_configuration(api, descriptor, config)
        path = config.root / f"seed_{seed:06d}" / "manifest.json"
        if path.exists():
            load_evaluation_prefix(config, episode)
            print(f"seed={seed} existing prefix validated", flush=True)
            continue
        plan = api.build_evaluation_prefix(descriptor, config.profile)
        shared = RoboCasaRuntimeFactory(api).create_from_descriptor(descriptor, expert=True)
        try:
            artifact = capture_prefix(shared, api, plan, config, {
                "suite_version": suite.suite_version, "suite_hash": suite.content_hash,
            })
        finally:
            shared.close()
        write_expert_prefix(config.root, artifact)
        print(f"seed={seed} written actions={len(artifact.actions)} "
              f"sha256={hashlib.sha256(artifact.actions.tobytes()).hexdigest()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
