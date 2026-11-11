#!/usr/bin/env python
"""Serve one read-only GR00T N1.7 LoRA checkpoint over membench-policy/v2.

The checkpoint is an NVIDIA Isaac-GR00T training output (PEFT adapter +
serialized pre/post processors + train_config.json); the base model and the
Cosmos-Reason2 tokenizer live in separate local directories.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
for source in (
    ROOT,
    ROOT / "policy" / "lerobot-0.6.1" / "src",
):
    if source.is_dir() and str(source) not in sys.path:
        sys.path.insert(0, str(source))

from bench.policy.backends.gr00t import Gr00TBackend  # noqa: E402
from bench.policy.server import WebsocketPolicyServer  # noqa: E402
from bench.runtime.data_contract import STATE_SCHEMA  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--vlm-model-dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--image-size", type=int, choices=(224, 256), required=True)
    parser.add_argument(
        "--state-schema",
        choices=(STATE_SCHEMA,),
        required=True,
    )
    parser.add_argument("--pytorch-device")
    parser.add_argument("--api-key-env", default="MEMBENCH_POLICY_API_KEY")
    parser.add_argument("--inference-timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--seed-by-episode",
        action="store_true",
        help="Reset Python/NumPy/Torch RNG from seed-<integer> in each episode ID.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args()
    logging.info(
        "Loading GR00T checkpoint %s (base: %s)",
        args.checkpoint,
        args.base_model_dir,
    )
    backend = Gr00TBackend.from_checkpoint(
        checkpoint_dir=args.checkpoint,
        base_model_dir=args.base_model_dir,
        vlm_model_dir=args.vlm_model_dir,
        image_size=(args.image_size, args.image_size),
        state_schema=args.state_schema,
        pytorch_device=args.pytorch_device,
        seed_by_episode=args.seed_by_episode,
        code_revision={"membench": _git_revision(ROOT), "policy_repo": _git_revision(ROOT / "policy" / "lerobot-0.6.1")},
    )
    logging.info("Model loaded")
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except OSError:
        local_ip = "unknown"
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
    WebsocketPolicyServer(
        backend,
        host=args.host,
        port=args.port,
        api_key=os.environ.get(args.api_key_env),
        inference_timeout_s=args.inference_timeout_s,
    ).serve_forever()


def _git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


if __name__ == "__main__":
    main()
