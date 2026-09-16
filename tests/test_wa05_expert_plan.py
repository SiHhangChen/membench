from types import SimpleNamespace

import numpy as np

from bench.execution.contracts import NavigateStage, OperateStage
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest, ExpertPlanInputs
from bench.tasks.wa05.execution.water_operations import place_cup_on_service_slot


def test_expert_plan_uses_one_fill_one_slot_and_matching_answer() -> None:
    for seed, temperature in ((0, "hot"), (1, "cold")):
        episode = TaskCatalog.resolve(EpisodeRequest(task_id="WA-05", seed=seed))
        api = resolve_builtin_task_plugin("WA-05").load_runtime_api()
        descriptor = api.build_descriptor(episode, False)
        plan = TaskCatalog.get("WA-05").build_expert_plan(
            ExpertPlanInputs.from_descriptor(descriptor)
        )

        operations = [
            stage for stage in plan.stages if isinstance(stage, OperateStage)
        ]
        fills = [
            stage for stage in operations
            if stage.operation == "fill_held_under_faucet"
        ]
        answers = [
            stage for stage in operations
            if stage.operation == "press_temperature_answer_button"
        ]
        placements = [
            stage for stage in operations
            if stage.operation == "place_on_service_slot"
        ]
        service_navigation = [
            stage for stage in plan.stages
            if isinstance(stage, NavigateStage)
            and stage.target.station_id == "service_area"
        ]
        button_navigation = [
            stage for stage in plan.stages
            if isinstance(stage, NavigateStage)
            and stage.target.station_id == "water_button_station"
        ]

        assert len(fills) == 1
        assert fills[0].target == "cup_A"
        assert fills[0].metadata["temperature"] == temperature
        assert len(service_navigation) == 1
        assert service_navigation[0].target.purpose == "slot_1"
        assert button_navigation == []
        assert len(placements) == 1
        assert placements[0].metadata["placement_approach_angles_deg"] == [
            180.0,
            165.0,
            150.0,
            135.0,
            124.0,
        ]
        assert descriptor.config.initial_state.objects["service_slots"]["regions"][
            "slot_1"
        ]["region_center_xy"] == [-1.145, -0.5]
        assert len(answers) == 1
        assert answers[0].metadata["temperature"] == temperature
        assert all(stage.operation != "record_final_take" for stage in operations)


def test_service_place_releases_before_the_independent_physical_check() -> None:
    class FakeEnv:
        def service_slot_position(self, slot: str) -> np.ndarray:
            assert slot == "slot_1"
            return np.asarray([2.0, -2.0, 0.9], dtype=float)

    class FakeController:
        def __init__(self) -> None:
            self.runtime_config = SimpleNamespace(execute=True)
            self.env = FakeEnv()
            self.call: dict[str, object] | None = None

        def place_at_world_pose(
            self,
            object_name: str,
            **kwargs: object,
        ) -> dict[str, object]:
            self.call = {"object_name": object_name, **kwargs}
            return {"pass": True}

    controller = FakeController()
    result = place_cup_on_service_slot(
        controller,
        cup_id="cup_A",
        slot="slot_1",
        require_pre_release_check=True,
        pre_release_min_height_from_slot=0.0,
        pre_release_max_height_from_slot=0.1,
    )

    assert result["pass"] is True
    assert result["semantic_event_deferred"] == "record_verified_staging"
    assert controller.call is not None
    assert "pre_release_check" not in controller.call
