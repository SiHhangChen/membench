"""Observe first, travel away, select once, then physically deliver."""

from bench.execution.plan_builder import finish_plan, navigate, operate
from bench.tasks.wr11.domain.contracts import SAMPLE_ID, TASK_REVISION


def build_expert_plan(inputs):
    episode, actions = inputs.episode, inputs.action_config
    if episode.task_id != "WR-11":
        raise ValueError("WR-11 expert requires a WR-11 episode")
    item = episode.facts["matching_item"]
    return finish_plan(
        episode,
        [
            navigate("sample_station", purpose="observe", name="observe_sample_station"),
            operate("observe_sample", SAMPLE_ID, name="observe_sample_packaging"),
            navigate("stock_station", purpose="pick", name="travel_to_stock_counter"),
            operate("pick", item, name="pick_matching_carton", metadata=actions.source_pick),
            operate(
                "return_arm", item, name="stow_matching_carton", metadata=actions.return_arm_holding
            ),
            navigate(
                "sample_station", purpose="deliver", holding=item, name="return_to_sample_station"
            ),
            operate(
                "place_delivery", item, name="deliver_beside_sample", metadata=actions.target_place
            ),
            operate("verify_delivery", item, name="verify_released_delivery"),
            operate("return_arm", item, name="return_empty_arm", metadata=actions.return_arm_empty),
        ],
        metadata={"task_revision": TASK_REVISION, "matching_item": item},
    )
