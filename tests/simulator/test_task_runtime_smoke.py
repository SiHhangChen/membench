from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
TASK_IDS = (
    "TM01", "TM02", "TM03", "TM04", "TS01", "TS02", "TS03", "TS04",
    "WA-01", "WA-02", "WA-03", "WA-04", "WA-05", "WR-01", "WR-02", "WR-03", "WR-04",
    "WR-05", "WR-06", "WR-07", "WR-08", "WR-11", "WX-01", "WX-02",
)


@pytest.mark.simulator
@pytest.mark.parametrize("task_id", TASK_IDS)
def test_episode_creates_and_closes_real_runtime(task_id: str) -> None:
    # MuJoCo selects its GL backend on first import, so each task gets a clean
    # process with the same headless backend used by production evaluation.
    code = """
import json
import sys
from bench.runtime import RoboCasaRuntimeFactory
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest

task_id = sys.argv[1]
episode = TaskCatalog.resolve(EpisodeRequest(
    task_id=task_id,
    seed=0,
    allow_partial_calibration=(task_id == "TM03"),
))
api = resolve_builtin_task_plugin(task_id).load_runtime_api()
runtime = RoboCasaRuntimeFactory(api).create(episode, execute=False, expert=True)
try:
    result = {
        "control_freq": runtime.env.control_freq,
        "has_observation": bool(runtime.observation),
        "missing_cameras": [
            alias for alias in runtime.initialized.descriptor.sensors.camera_aliases
            if runtime.initialized.descriptor.sensors.observation_key(alias) not in runtime.observation
        ],
        "has_targets": runtime.initialized.targets is not None,
        "episode_identity": episode.identity,
        "prompt": runtime.initialized.descriptor.prompt.policy_prompt,
    }
    print(json.dumps(result, sort_keys=True))
finally:
    runtime.close()
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, task_id],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"},
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])

    assert result["control_freq"] == 20
    assert result["has_observation"] is True
    assert result["missing_cameras"] == []
    assert result["episode_identity"].startswith("sha256:")
    assert result["prompt"].strip()
    assert result["has_targets"] is (task_id in {"WR-02", "WR-04", "WR-05", "WR-06", "WR-07", "WR-08", "WR-11", "WX-01", "WX-02"})
