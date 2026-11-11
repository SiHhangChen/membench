"""One fixed expert sugar subtask before the policy-controlled remainder."""

from dataclasses import replace

from bench.tasks.contracts import ExpertPlanInputs
from bench.tasks.wa01.domain.episode import sugared_coffee_for_seed
from bench.tasks.wa01.evaluation import evaluate_initialized_runtime
from bench.tasks.wa01.expert.spec import build_expert_plan

PROFILE = "wa01_sugar_v1"


def build_prefix(descriptor, profile):
    if profile != PROFILE:
        raise ValueError(f"unsupported WA-01 evaluation prefix: {profile!r}")
    plan = build_expert_plan(ExpertPlanInputs.from_descriptor(descriptor))
    end = next(
        i for i, stage in enumerate(plan.stages)
        if stage.name == "return_after_place_sugar_to_regular_pose"
    )
    return replace(plan, stages=plan.stages[:end + 1])


def validate_prefix(initialized, profile):
    if profile != PROFILE:
        raise ValueError(f"unsupported WA-01 evaluation prefix: {profile!r}")
    state = evaluate_initialized_runtime(initialized)
    env = initialized.env
    actual = env.sugared_coffee_id
    if (
        not state.stages["sugar"].qualified or state.failure_code
        or actual != sugared_coffee_for_seed(initialized.episode.seed)
        or any(state.stages[name].observed for name in ("one_ready", "ready", "selection"))
    ):
        raise ValueError("WA-01 expert prefix did not complete only the verified sugar subtask")
    return {"sugared_coffee_id": actual, "completed_subtasks": ["add_sugar"]}
