"""Live WR-11 target geometry and object-relative top-down grasps."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from bench.execution.planning import make_object_pick_target_poses
from bench.tasks.wr11.domain.contracts import ITEM_IDS


def build_pick_payload_selector():
    """Compare full pick plans without changing the live carton or robot.

    Like WA-01, targets follow the object's current orientation. Different
    top-down wrist directions can reach the same grasp center through very
    different arm paths. Planning checks reachability, not contact dynamics;
    the normal executed pick must still confirm a physical grasp or lift.
    Explicitly calibrated single-pose files retain the generic controller path.
    """

    def selector(request: dict[str, Any]) -> dict[str, Any] | None:
        name = str(request.get("object_name", ""))
        kwargs = dict(request.get("kwargs") or {})
        raw_yaws = kwargs.get("grasp_yaw_offsets_deg")
        if name not in ITEM_IDS or raw_yaws is None:
            return None
        if kwargs.get("manual_grasp_pose_file") or kwargs.get("grasp_pose_file"):
            return None
        yaws = np.asarray(raw_yaws, dtype=float)
        if yaws.ndim != 1 or yaws.size == 0 or not np.all(np.isfinite(yaws)):
            raise ValueError("WR-11 grasp_yaw_offsets_deg must be a nonempty finite list")
        config = request["pick_place_config"]
        orientation = str(kwargs.get("orientation_mode", config.orientation_mode))
        if orientation not in {"object-x-topdown", "object-y-topdown"}:
            raise ValueError("WR-11 multi-direction grasp requires a top-down orientation")
        dry_sequence = request["dry_run_pick_sequence"]
        records: list[dict[str, Any]] = []
        passing: list[tuple[tuple[int, int, int, int], dict[str, Any], dict[str, Any]]] = []
        pregrasp_height = float(kwargs.get("pregrasp_height", config.pregrasp_height))
        lift_height = float(request["lift_height"])
        profiles = [("standard", pregrasp_height, lift_height)]
        if pregrasp_height > 0.04 or lift_height > 0.08:
            profiles.append(("low_clearance", min(pregrasp_height, 0.04), min(lift_height, 0.08)))
        for profile, pregrasp, lift in profiles:
            for index, yaw in enumerate(yaws):
                payload, record, score = _validate_candidate(
                    request, kwargs, name, orientation, pregrasp, lift, index, yaw, dry_sequence
                )
                record["profile"] = profile
                records.append(record)
                if record["pass"]:
                    passing.append((score, payload, record))
            # Preserve the existing successful grasp choice. Shorter motions
            # are a fallback only when every full standard sequence fails.
            if passing:
                break
        selection = {
            "mode": "wr11_object_relative_topdown_candidates",
            "expert_grasp_profile": "WR-11/grasp-v2",
            "validation_scope": "plan_reachability_only",
            "score_order": [
                "perpendicular_axis",
                "max_stage_trajectory_len",
                "total_trajectory_len",
                "candidate_index",
            ],
            "candidates": records,
            "pass": bool(passing),
        }
        if not passing:
            selection["reason"] = "no_reachable_topdown_grasp_candidate"
            return {"payload": None, "selection": selection}
        score, payload, chosen = min(passing, key=lambda entry: entry[0])
        selection.update(
            {
                "selected_index": chosen["index"],
                "selected_yaw_offset_deg": chosen["yaw_offset_deg"],
                "selected_profile": chosen["profile"],
                "selected_pregrasp_height_m": chosen["pregrasp_height_m"],
                "selected_lift_height_m": chosen["lift_height_m"],
                "selected_score": list(score),
            }
        )
        return {"payload": payload, "selection": selection}

    return selector


def _validate_candidate(
    request, kwargs, name, orientation, pregrasp, lift, index, yaw, dry_sequence
):
    config = request["pick_place_config"]
    payload = make_object_pick_target_poses(
        request["env"],
        request["obs"],
        object_name=name,
        orientation_mode=orientation,
        pregrasp_height=pregrasp,
        grasp_clearance=float(kwargs.get("grasp_clearance", config.grasp_clearance)),
        lift_height=lift,
        grasp_yaw_offset_deg=float(yaw),
        object_pos_world=kwargs.get("object_pos_world"),
        grasp_offset_x=float(kwargs.get("grasp_offset_x", 0.0)),
        grasp_offset_y=float(kwargs.get("grasp_offset_y", 0.0)),
        grasp_offset_z=float(kwargs.get("grasp_offset_z", 0.0)),
    )
    validation = dry_sequence(payload, object_name=name)
    stages = [
        {
            "stage": stage.get("stage"),
            "pass": bool(stage.get("pass", False)),
            "status": (stage.get("plan") or {}).get("status"),
            "trajectory_len": (stage.get("plan") or {}).get("trajectory_len"),
            "failure_reason": stage.get("failure_reason"),
        }
        for stage in validation.get("stages", ())
        if isinstance(stage, Mapping)
    ]
    passed = (
        bool(validation.get("pass", False))
        and [stage["stage"] for stage in stages] == ["pregrasp", "grasp", "lift"]
        and all(stage["pass"] for stage in stages)
    )
    lengths = [int(stage["trajectory_len"] or 0) for stage in stages]
    # Opposing approaches along the native Y axis retained the carton
    # through transport in physical probes. The perpendicular axis
    # can lift successfully but slip during transport. Prefer
    # that supported grasp family before comparing path lengths.
    perpendicular = int(not np.isclose(float(yaw) % 180.0, 0.0))
    score = (perpendicular, max(lengths, default=0), sum(lengths), index)
    record = {
        "index": index,
        "yaw_offset_deg": float(yaw),
        "pregrasp_height_m": pregrasp,
        "lift_height_m": lift,
        "pass": passed,
        "score": list(score) if passed else None,
        "stages": stages,
    }
    return payload, record, score
