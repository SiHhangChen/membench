#!/usr/bin/env python
"""Run a policy-neutral MobileMemBench closed-loop evaluation suite."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.evaluation.adapter import get_adapter, preflight_adapter  # noqa: E402
from bench.evaluation.artifacts import ArtifactWriter  # noqa: E402
from bench.evaluation.client import RemotePolicyClient  # noqa: E402
from bench.evaluation.prefix import evaluation_prefix_inventory  # noqa: E402
from bench.evaluation.rollout import RolloutRunner  # noqa: E402
from bench.evaluation.suite import load_eval_suite  # noqa: E402
from bench.tasks import get_builtin_task_plugin, register_builtin_task_plugins  # noqa: E402


DEFAULT_SUITE = ROOT / "configs" / "eval" / "suites" / "wx01_v1.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True, help="ws://host:port")
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "eval")
    parser.add_argument("--run-id")
    parser.add_argument("--api-key-env", default="MEMBENCH_POLICY_API_KEY")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--retry-errors",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate suite, task, and remote policy contract without creating an env.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    suite = load_eval_suite(args.suite)
    prefix_inventory = evaluation_prefix_inventory(suite)
    register_builtin_task_plugins()
    plugin = get_builtin_task_plugin(suite.task_id)
    adapter = get_adapter(suite.task_id)
    api_key = os.environ.get(args.api_key_env)
    client = RemotePolicyClient(
        args.endpoint,
        api_key=api_key,
        timeout_s=suite.transport.timeout_s,
        connect_attempts=suite.transport.connect_attempts,
        connect_retry_delay_s=suite.transport.connect_retry_delay_s,
    )
    try:
        contract = client.connect()
        if suite.rollout.replan_steps > contract.action.horizon:
            raise ValueError(
                f"Suite replan_steps={suite.rollout.replan_steps} exceeds "
                f"server action horizon={contract.action.horizon}"
            )
        preflight = preflight_adapter(
            adapter,
            suite=suite,
            policy_contract=contract,
        )
        if prefix_inventory is not None:
            preflight = {**preflight, "expert_prefix_inventory": prefix_inventory}
        if args.preflight_only:
            print(
                _json_dump(
                    {
                        "status": "ok",
                        "suite": suite.suite_version,
                        "task": suite.task_id,
                        "policy_server": contract.server_id,
                        "preflight": preflight,
                    }
                )
            )
            return 0
        run_id = args.run_id or _default_run_id(suite.task_id, contract.server_id)
        writer = ArtifactWriter(
            args.output_root / run_id,
            expected_seeds=suite.seeds,
        )
        writer.write_run_manifest(
            _run_manifest(
                run_id=run_id,
                suite=suite,
                endpoint=args.endpoint,
                contract=contract.to_hello(),
                env_name=plugin.env_name,
                preflight=preflight,
            )
        )
        runner = RolloutRunner(client=client, suite=suite, artifacts=writer)
        interrupted = False
        try:
            runner.run_suite(
                adapter,
                resume=args.resume,
                retry_errors=args.retry_errors,
            )
        except KeyboardInterrupt:
            interrupted = True
        summary = writer.write_summary()
        print(f"run_dir={writer.run_dir}")
        print(f"summary={_json_dump(summary)}")
        if interrupted:
            return 130
        return 0 if summary.get("publishable", False) else 2
    finally:
        client.close()


def _run_manifest(
    *,
    run_id: str,
    suite: Any,
    endpoint: str,
    contract: dict[str, Any],
    env_name: str,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "membench.run_manifest/v1",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite": {
            "path": str(suite.source_path),
            "hash": suite.content_hash,
            "version": suite.suite_version,
        },
        "task": {
            "id": suite.task_id,
            "env_name": env_name,
            "env_split": suite.env_split,
            "options": dict(suite.task_options),
            "identity": dict(preflight.get("task", {})),
        },
        "seeds": list(suite.seeds),
        "policy_contract": contract,
        "server_endpoint": endpoint,
        "rollout": {
            "max_steps": suite.rollout.max_steps,
            "replan_steps": suite.rollout.replan_steps,
            "action_repeat": suite.rollout.action_repeat,
            "stop_on_success": suite.rollout.stop_on_success,
            "ignore_done": suite.rollout.ignore_done,
            **(
                {} if suite.expert_prefix is None else {
                    "expert_prefix": {
                        "profile": suite.expert_prefix.profile,
                        "root": str(suite.expert_prefix.root),
                        "inventory": preflight.get("expert_prefix_inventory"),
                    }
                }
            ),
        },
        "artifacts": {
            "record_video": suite.artifacts.record_video,
            "video_cameras": list(suite.artifacts.video_cameras),
            "video_fps": suite.artifacts.video_fps,
        },
        "client_code_revision": _git_revision(ROOT),
        "client_code_dirty": _git_dirty(ROOT),
        "runtime_versions": _runtime_versions(),
        "preflight": preflight,
    }


def _default_run_id(task_id: str, server_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_server = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in server_id
    )
    return f"{task_id.lower()}-{safe_server}-{timestamp}"


def _git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _git_dirty(path: Path) -> bool | None:
    result = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def _runtime_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for distribution in ("numpy", "pydantic", "PyYAML", "websockets"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "unavailable"
    return versions


def _json_dump(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
