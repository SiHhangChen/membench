from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.contract
def test_generic_execution_has_no_task_semantic_tokens() -> None:
    forbidden = (
        "press_button", "operate_drawer", "operate_slide_door", "rotate_knob",
        "TM01", "TM02", "TM03", "TM04", "TS01", "TS02", "TS03", "TS04",
        "WA-01", "WA-02", "WA-03", "WA-04", "WA-05",
        "WR-01", "WR-02", "WR-03", "WR-04", "WR-05", "WR-06", "WR-07", "WR-08", "WR-11", "WX-01", "WX-02",
    )
    for source in Path("bench/execution").rglob("*.py"):
        text = source.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), source


@pytest.mark.contract
def test_new_evaluation_core_has_no_expert_or_recording_dependency() -> None:
    for source in Path("bench/evaluation").glob("*.py"):
        text = source.read_text(encoding="utf-8")
        assert "bench.stages" not in text
        assert "bench.recording.task_bundle" not in text
        assert "ExpertPlan" not in text
