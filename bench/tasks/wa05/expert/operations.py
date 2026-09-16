"""Build WA-05's task-local navigation and manipulation runtime."""

from typing import Any

from bench.execution.runner import StageRunner
from bench.tasks.wa05.execution.handlers import build_stage_handlers
from bench.tasks.wa05.execution.manipulation import build_manipulation_assembly
from bench.tasks.wa05.execution.navigation import build_navigation_assembly


def build_operation_handlers(
    descriptor: Any,
    env: Any,
    observation: dict[str, Any],
) -> dict[str, Any]:
    navigation = build_navigation_assembly(
        env,
        observation,
        config=descriptor.navigation,
        seed=descriptor.episode_plan.seed,
    )
    manipulation = build_manipulation_assembly(
        env,
        observation,
        settings=descriptor.config.manipulation,
        planner_script=descriptor.resources["planner_script"],
        cup_grasp_candidates_file=descriptor.resources["cup_grasp_candidates"],
        execute=descriptor.execute,
    )
    sampler = navigation.station_sampler

    def sampled_sink_exit_target() -> Any:
        return sampler.route_poses[10]

    sink_exit_target_provider = (
        sampled_sink_exit_target
        if 10 in getattr(sampler, "route_poses", {})
        else None
    )
    manipulation.controller.operation_handlers = build_stage_handlers(
        manipulation.controller,
        handler_settings=descriptor.actions.handler_settings,
        sink_exit_target_provider=sink_exit_target_provider,
    )
    return {
        "runner": StageRunner(
            navigation=navigation.controller,
            manipulation=manipulation.controller,
        )
    }


__all__ = ["build_operation_handlers"]
