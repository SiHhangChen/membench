"""WR-07 single-item memory-barrier plan with color-matched sugar plates."""

from __future__ import annotations

from bench.execution.plan_builder import finish_plan, navigate, operate
from bench.tasks.contracts import ExpertPlanInputs
from bench.tasks.wr07.domain.contracts import ITEM_ID, TASK_REVISION

def build_expert_plan(inputs: ExpertPlanInputs):
    episode = inputs.episode
    if episode.task_id != "WR-07":
        raise ValueError(f"WR-07 cannot build a plan for {episode.task_id!r}")
    facts = dict(episode.facts)
    item = ITEM_ID
    source_slot = str(facts["source_slot_by_item"][item])
    source_color = str(facts["source_color_by_item"][item])
    target_slot = str(facts["target_slot_by_color"][source_color])
    actions = inputs.action_config
    stages = [
        navigate("counter_zones", purpose="source_observe", name="observe_source_sugar_plates", metadata={"observe": "three_colored_source_sugar_plates"}),
        navigate(source_slot, purpose="source_pick", name="navigate_to_source_plate", metadata={"xy_tolerance": 0.08, "yaw_tolerance": 0.12}),
        operate("pick", item, name="pick_milk_carton_from_source_plate", metadata=actions.source_pick_metadata[item]),
        operate("return_arm", item, name="stow_item_after_source_pick", metadata={**dict(actions.return_arm_holding), "stage": "stow_item_after_source_pick"}),
        operate("latch_source_color_memory", item, name="latch_source_color_memory", metadata={"source_color": source_color, "source_slot": source_slot}),
        navigate("plate_station", purpose="target_place", holding=item, name="navigate_to_matching_target_plate", metadata={"holding_navigation_mode": "rotate_then_translate", "target_slot": target_slot, "source_color": source_color, "xy_tolerance": 0.08, "yaw_tolerance": 0.12}),
        operate("place_at_accessible", item, name="place_milk_carton_on_matching_plate", metadata={**dict(actions.target_place_metadata[target_slot]), "target_name": f"{item}_{target_slot}", "target_resolver": "target_plate_pose", "target_slot": target_slot, "source_color": source_color}),
        operate("verify_target_plate", item, name="verify_target_plate", metadata={"target_slot": target_slot, "source_color": source_color}),
        operate("return_arm", item, name="return_arm_after_target_place", metadata={**dict(actions.return_arm_empty), "stage": "return_after_target_place"}),
    ]
    return finish_plan(episode, stages, metadata={
        "task_revision": TASK_REVISION,
        "carton_type": str(facts["carton_type"]),
        "carton_asset": str(facts["carton_asset"]),
        "carton_material": str(facts["carton_material"]),
        "carton_type_candidates": list(facts["carton_type_candidates"]),
        "carton_material_candidates": list(facts["carton_material_candidates"]),
        "carton_combination_count": int(
            len(facts["carton_type_candidates"])
            * len(facts["carton_material_candidates"])
        ),
        "source_slot": source_slot,
        "source_color": source_color,
        "target_slot": target_slot,
        "source_plate_color_by_slot": dict(facts["source_plate_color_by_slot"]),
        "target_plate_color_by_slot": dict(facts["target_plate_color_by_slot"]),
    })

__all__ = ["build_expert_plan"]
