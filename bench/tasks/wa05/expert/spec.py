"""Task-local WA-05 expert plan."""

from __future__ import annotations

from bench.execution.contracts import ExpertPlan
from bench.execution.plan_builder import finish_plan, navigate, operate
from bench.tasks.contracts import ExpertPlanInputs


def build_expert_plan(inputs: ExpertPlanInputs) -> ExpertPlan:
    episode = inputs.episode
    if episode.task_id != "WA-05":
        raise ValueError(f"WA-05 cannot build a plan for {episode.task_id!r}")
    facts = dict(episode.facts)
    actions = inputs.action_config
    cup = str(facts["cup_id"])
    temperature = str(facts["expert_fill_temperature"])
    slot = str(facts["service_slot"])

    def ret(target: str, name: str, gripper: float = -1.0) -> object:
        return operate(
            "return_arm",
            target,
            name=name,
            metadata={
                **dict(actions.return_arm_metadata),
                "stage": name,
                "gripper_action": gripper,
            },
        )

    navigation_metadata = {
        "yaw_tolerance": float(actions.cup_pick_navigation_yaw_tolerance),
        "terminal_tolerance_only": True,
        "use_sampled_terminal_pose": True,
    }
    stages = [
        navigate(
            cup,
            purpose="pick",
            name=f"navigate_pick_{cup}",
            metadata={
                **navigation_metadata,
                "xy_tolerance": float(actions.cup_pick_navigation_xy_tolerance),
            },
        ),
        operate(
            "pick_upright_cup",
            cup,
            name=f"pick_{cup}",
            metadata=actions.cup_pick_metadata,
        ),
        ret(cup, f"return_after_pick_{cup}", 1.0),
        navigate(
            "sink_basin",
            purpose="fill",
            holding=cup,
            name=f"navigate_fill_{temperature}_{cup}",
            metadata={
                **navigation_metadata,
                "xy_tolerance": float(actions.held_fill_navigation_xy_tolerance),
            },
        ),
        operate(
            "fill_held_under_faucet",
            cup,
            name=f"fill_{temperature}_{cup}",
            metadata=actions.held_fill_metadata_by_temperature[temperature],
        ),
        operate(
            "return_arm_holding_cup",
            cup,
            name=f"return_after_fill_{temperature}_{cup}",
            metadata={
                **dict(actions.return_arm_holding_cup_metadata),
                "stage": f"return_after_fill_{temperature}_{cup}",
                "gripper_action": 1.0,
            },
        ),
        operate(
            "lift_held_cup_after_sink_exit",
            cup,
            name=f"lift_after_sink_exit_{cup}",
            metadata={
                **dict(actions.sink_exit_lift_metadata),
                "stage": f"lift_after_sink_exit_{cup}",
            },
        ),
        navigate(
            "service_area",
            purpose=slot,
            holding=cup,
            name=f"navigate_service_{slot}",
            metadata={
                **navigation_metadata,
                "xy_tolerance": float(actions.held_fill_navigation_xy_tolerance),
            },
        ),
        operate(
            "place_on_service_slot",
            cup,
            name=f"place_{cup}_{slot}",
            metadata=actions.service_place_metadata_by_slot[slot],
        ),
        operate(
            "record_verified_staging",
            cup,
            name=f"verify_staging_{cup}_{slot}",
            metadata=actions.service_place_metadata_by_slot[slot],
        ),
        ret(cup, f"return_after_service_{cup}"),
        operate(
            "press_temperature_answer_button",
            "answer_buttons",
            name=f"press_{temperature}_answer_button",
            metadata=actions.answer_button_metadata_by_temperature[temperature],
        ),
        ret("answer_buttons", f"return_after_{temperature}_answer"),
    ]
    return finish_plan(
        episode,
        stages,
        metadata={
            "expert_fill_temperature": temperature,
            "cup_id": cup,
            "service_slot": slot,
        },
    )


__all__ = ["build_expert_plan"]
