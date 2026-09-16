"""Assemble WR-08 navigation and manipulation handlers."""

from __future__ import annotations

from typing import Any

from bench.execution.runner import StageRunner
from bench.tasks.wr08.execution.handlers import build_operation_handlers as build_stage_handlers
from bench.tasks.wr08.execution.manipulation import build_manipulation_assembly
from bench.tasks.wr08.execution.navigation import build_navigation_assembly


def build_operation_handlers(descriptor: Any, env: Any, observation: dict[str, Any]) -> dict[str, Any]:
    env.reset_evaluation(expert_recording=True)
    navigation = build_navigation_assembly(
        env,
        observation,
        nav_config=descriptor.navigation,
        seed=descriptor.episode_plan.seed,
    )
    manipulation = build_manipulation_assembly(
        env,
        observation,
        settings=descriptor.config.manipulation,
        planner_script=descriptor.resources["planner_script"],
        execute=descriptor.execute,
    )
    manipulation.controller.operation_handlers = build_stage_handlers(manipulation.controller)
    return {"runner": StageRunner(navigation=navigation.controller, manipulation=manipulation.controller)}


__all__ = ["build_operation_handlers"]
