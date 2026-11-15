"""Common observation, transit and pickup; only placement depends on memory."""

from bench.execution.plan_builder import finish_plan, navigate, operate

from ..domain.contracts import BLOCK_ID, CAN_ID, TASK_REVISION


def build_expert_plan(inputs):
    episode, actions = inputs.episode, inputs.action_config
    if episode.task_id != "WR-12":
        raise ValueError("WR-12 expert requires a WR-12 episode")
    return finish_plan(
        episode,
        [
            operate("observe_can", CAN_ID, name="observe_box_contents"),
            navigate("operate_station", purpose="pick", name="move_to_box_fronts"),
            operate("establish_barrier", CAN_ID, name="confirm_occlusion"),
            operate("pick", BLOCK_ID, name="pick_common_block", metadata=actions.source_pick),
            operate(
                "place_delivery",
                BLOCK_ID,
                name="place_at_remembered_box",
                metadata=actions.target_place,
            ),
            operate("verify_delivery", BLOCK_ID, name="verify_supported_release"),
            operate(
                "return_arm", BLOCK_ID, name="return_empty_arm", metadata=actions.return_arm_empty
            ),
        ],
        metadata={"task_revision": TASK_REVISION, "target_box": episode.facts["target_box"]},
    )
