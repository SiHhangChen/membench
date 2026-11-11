"""Expert pick and fill subtasks before policy placement and answer."""

from dataclasses import replace

from bench.execution.manipulation.object_interactions import get_object_hold_status
from bench.tasks.contracts import ExpertPlanInputs
from bench.tasks.wa05.evaluation import evaluate_initialized_runtime
from bench.tasks.wa05.expert.spec import build_expert_plan

PROFILE = "wa05_pick_fill_v1"


def build_prefix(descriptor, profile):
    if profile != PROFILE:
        raise ValueError(f"unsupported WA-05 evaluation prefix: {profile!r}")
    plan = build_expert_plan(ExpertPlanInputs.from_descriptor(descriptor))
    end = next(
        i for i, stage in enumerate(plan.stages)
        if stage.name.startswith("lift_after_sink_exit_")
    )
    return replace(plan, stages=plan.stages[:end + 1])


def validate_prefix(initialized, profile):
    if profile != PROFILE:
        raise ValueError(f"unsupported WA-05 evaluation prefix: {profile!r}")
    state = evaluate_initialized_runtime(initialized)
    plan = initialized.descriptor.episode_plan
    env = initialized.env
    if (
        not state.stages["cup_filled"].qualified or state.failure_code
        or env.semantic_state.filled_temperature != plan.expert_fill_temperature
        or state.stages["cup_staged"].observed or state.stages["success"].observed
        or env.semantic_state.pressed_temperature is not None
    ):
        raise ValueError("WA-05 expert prefix did not complete only verified pick and fill")
    if not get_object_hold_status(env, plan.cup_id)["held"]:
        raise ValueError("WA-05 expert prefix must hand over a physically held cup")
    # Expert setup uses strict service-slot placement. The policy remainder uses
    # the canonical evaluation rule, with the same evaluator and event history.
    env.require_service_region_for_staging = False
    return {
        "filled_temperature": env.semantic_state.filled_temperature,
        "completed_subtasks": ["pick_cup", "fill_cup"],
    }
