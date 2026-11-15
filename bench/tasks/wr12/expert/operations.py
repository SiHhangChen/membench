"""Attach the navigation and manipulation controllers for expert recording."""

from bench.execution.runner import StageRunner

from ..execution.handlers import build_operation_handlers as build_stage_handlers
from ..execution.manipulation import build_manipulation_assembly
from ..execution.navigation import build_navigation_assembly


def build_operation_handlers(descriptor, env, observation):
    navigation = build_navigation_assembly(
        env,
        dict(observation),
        nav_config=descriptor.navigation,
        episode_plan=descriptor.episode_plan,
    )
    manipulation = build_manipulation_assembly(
        env,
        dict(observation),
        settings=descriptor.config.manipulation,
        planner_script=descriptor.resources["planner_script"],
        execute=descriptor.execute,
    )
    manipulation.controller.operation_handlers = build_stage_handlers(manipulation.controller)
    return {
        "runner": StageRunner(
            navigation=navigation.controller, manipulation=manipulation.controller
        )
    }
