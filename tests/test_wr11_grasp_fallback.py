from types import SimpleNamespace

import pytest

from bench.tasks.wr11.execution import geometry


@pytest.mark.parametrize("reachable", ["standard", "low_clearance", "none"])
def test_grasp_fallback_preserves_standard_or_requires_complete_lower_sequence(
    monkeypatch, reachable
):
    # The planner stub rejects either the high pregrasp or high lift. A
    # partial approach is never sufficient to select a grasp for execution.
    monkeypatch.setattr(geometry, "make_object_pick_target_poses", lambda *args, **kwargs: kwargs)
    planned = []

    def dry_sequence(payload, *, object_name):
        planned.append(payload)
        low = payload["pregrasp_height"] <= 0.04 and payload["lift_height"] <= 0.08
        passed = reachable == "standard" or reachable == "low_clearance" and low
        stages = [
            {"stage": name, "pass": passed, "plan": {"trajectory_len": 10}}
            for name in ("pregrasp", "grasp", "lift")
        ]
        if not passed:
            stages[-1]["pass"] = False
        return {"pass": passed, "stages": stages}

    request = {
        "object_name": "stock_right",
        "env": object(),
        "obs": {},
        "kwargs": {"grasp_yaw_offsets_deg": [0, 90, 180, 270], "pregrasp_height": 0.06},
        "pick_place_config": SimpleNamespace(
            orientation_mode="object-y-topdown", pregrasp_height=0.06, grasp_clearance=-0.04
        ),
        "lift_height": 0.12,
        "dry_run_pick_sequence": dry_sequence,
    }
    selected = geometry.build_pick_payload_selector()(request)
    assert len(planned) == (4 if reachable == "standard" else 8)
    if reachable == "none":
        assert selected["payload"] is None
        assert not selected["selection"]["pass"]
    else:
        assert selected["selection"]["pass"]
        assert selected["selection"]["selected_profile"] == reachable
        assert selected["payload"]["grasp_yaw_offset_deg"] in (0, 180)
        assert selected["payload"]["object_name"] == "stock_right"
        assert selected["payload"]["pregrasp_height"] == (0.06 if reachable == "standard" else 0.04)
        assert selected["payload"]["lift_height"] == (0.12 if reachable == "standard" else 0.08)
