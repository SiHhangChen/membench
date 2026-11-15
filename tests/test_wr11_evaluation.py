import pytest

from bench.evaluation.report_stages import validate_report_stages
from bench.evaluation.report_profiles import get_report_stage_profile
from bench.tasks.wr11.evaluation import WR11EvaluationSnapshot as Snapshot
from bench.tasks.wr11.evaluation import WR11Evaluator


def ready():
    evaluator = WR11Evaluator("stock_center")
    evaluator.reset(Snapshot(0))
    for step in range(1, 4):
        evaluator.observe(Snapshot(step, at_sample=True, sample_visible=True))
    evaluator.observe(Snapshot(4, at_stock=True, sample_in_any_camera=False))
    return evaluator


def test_success_requires_observation_departure_first_selection_and_supported_delivery():
    evaluator = ready()
    evaluator.observe(
        Snapshot(5, at_stock=True, sample_in_any_camera=False, picked_items=("stock_center",))
    )
    for step in (6, 7):
        evaluator.observe(Snapshot(step, delivered_items=("stock_center",)))
        assert not evaluator.success
    evaluator.observe(Snapshot(8, delivered_items=("stock_center",)))
    assert evaluator.success and evaluator.to_diagnostics()["stop_execution"]
    report = evaluator.to_report_stages()
    validate_report_stages(report, expected_task_id="WR-11", expected_profile_id="WR-11/report-v2")
    assert report["order"] == ["item_picked", "item_delivered", "success"]
    assert all(stage["passed"] for stage in report["stages"].values())
    evaluator.observe(Snapshot(9, sample_undisturbed=False))
    assert evaluator.success  # terminal verdict is frozen


def test_wrong_color_pick_continues_and_delivery_counts_before_color_failure():
    evaluator = ready()
    evaluator.observe(
        Snapshot(5, at_stock=True, sample_in_any_camera=False, picked_items=("stock_left",))
    )
    assert evaluator.stage_steps["item_picked"] == 5
    assert not evaluator.terminal
    for step in (6, 7):
        evaluator.observe(Snapshot(step, delivered_items=("stock_left",)))
        assert not evaluator.terminal
    evaluator.observe(Snapshot(8, delivered_items=("stock_left",)))
    assert evaluator.failure_code == "wrong_color_delivered"
    assert evaluator.first_picked_item == "stock_left"
    assert not evaluator.success
    report = validate_report_stages(evaluator.to_report_stages(), expected_success=False)
    assert report["stages"]["item_picked"]["passed"]
    assert report["stages"]["item_delivered"]["passed"]
    assert not report["stages"]["success"]["passed"]


def test_wrong_first_pick_cannot_be_repaired_by_switching_cartons():
    evaluator = ready().observe(
        Snapshot(5, at_stock=True, sample_in_any_camera=False, picked_items=("stock_left",))
    )
    evaluator.observe(Snapshot(6, picked_items=("stock_center",)))
    assert evaluator.failure_code == "extra_carton_picked"
    assert evaluator.first_picked_item == "stock_left"
    assert not evaluator.success


def test_delivery_must_be_of_the_first_picked_carton():
    evaluator = ready().observe(
        Snapshot(5, at_stock=True, sample_in_any_camera=False, picked_items=("stock_left",))
    )
    for step in range(6, 10):
        evaluator.observe(Snapshot(step, delivered_items=("stock_center",)))
    assert evaluator.stage_steps["item_delivered"] is None
    assert not evaluator.terminal


def test_sample_disturbance_after_wrong_color_pick_still_stops_episode():
    evaluator = ready().observe(
        Snapshot(5, at_stock=True, sample_in_any_camera=False, picked_items=("stock_left",))
    )
    evaluator.observe(Snapshot(6, sample_undisturbed=False))
    assert evaluator.failure_code == "sample_disturbed"
    assert evaluator.stage_steps["item_picked"] == 5
    assert not evaluator.success


def test_historical_five_stage_report_remains_readable():
    order = [
        "sample_observed", "stock_reached", "first_pick_correct", "item_delivered", "success"
    ]
    report = {
        "schema_version": "membench.report_stages/v1",
        "task_id": "WR-11",
        "profile_id": "WR-11/report-v1",
        "order": order,
        "stages": {
            stage: {"applicable": True, "passed": True, "first_pass_step": step}
            for stage, step in zip(order, (3, 4, 5, 8, 8), strict=True)
        },
    }
    assert validate_report_stages(report, expected_success=True)["order"] == order
    current = get_report_stage_profile("WR-11")
    assert current.profile_id == "WR-11/report-v2"
    assert current.stage_ids == ("item_picked", "item_delivered", "success")


def test_duplicate_and_skipped_steps_cannot_satisfy_observation_or_delivery_dwell():
    evaluator = WR11Evaluator("stock_center")
    evaluator.reset(Snapshot(0))
    for step in (1, 1, 3, 5):
        evaluator.observe(Snapshot(step, at_sample=True, sample_visible=True))
    assert evaluator.stage_steps["sample_observed"] is None
    evaluator = ready()
    evaluator.observe(
        Snapshot(5, at_stock=True, sample_in_any_camera=False, picked_items=("stock_center",))
    )
    for step in (6, 6, 8, 10):
        evaluator.observe(Snapshot(step, delivered_items=("stock_center",)))
    assert not evaluator.success
    evaluator.observe(Snapshot(11, delivered_items=("stock_center",)))
    evaluator.observe(Snapshot(12, delivered_items=("stock_center",)))
    assert evaluator.success


@pytest.mark.parametrize(
    "snapshot,code",
    [
        (
            Snapshot(5, at_stock=True, sample_in_any_camera=True, picked_items=("stock_center",)),
            "sample_visible_at_selection",
        ),
        (
            Snapshot(5, at_stock=False, sample_in_any_camera=False, picked_items=("stock_center",)),
            "pick_before_memory_barrier",
        ),
        (Snapshot(5, sample_undisturbed=False), "sample_disturbed"),
        (Snapshot(5, picked_items=("stock_center", "stock_left")), "ambiguous_first_pick"),
    ],
)
def test_invalid_selection_or_sample_manipulation_stops_episode(snapshot, code):
    evaluator = ready().observe(snapshot)
    assert evaluator.failure_code == code
    assert evaluator.terminal and not evaluator.success


def test_unobserved_sample_cannot_qualify_a_lucky_pick():
    evaluator = WR11Evaluator("stock_center")
    evaluator.reset(Snapshot(0))
    evaluator.observe(Snapshot(1, at_stock=True, sample_in_any_camera=False))
    evaluator.observe(
        Snapshot(2, at_stock=True, sample_in_any_camera=False, picked_items=("stock_center",))
    )
    assert evaluator.failure_code == "pick_before_memory_barrier"


def test_putdown_without_pick_is_not_delivery_and_interrupted_support_resets_dwell():
    evaluator = ready()
    for step in range(5, 10):
        evaluator.observe(Snapshot(step, delivered_items=("stock_center",)))
    assert not evaluator.success
    evaluator.observe(
        Snapshot(10, at_stock=True, sample_in_any_camera=False, picked_items=("stock_center",))
    )
    evaluator.observe(Snapshot(11, delivered_items=("stock_center",)))
    evaluator.observe(Snapshot(12))
    evaluator.observe(Snapshot(13, delivered_items=("stock_center",)))
    assert not evaluator.success


def test_extra_stock_pick_after_correct_selection_is_terminal():
    evaluator = ready().observe(
        Snapshot(5, at_stock=True, sample_in_any_camera=False, picked_items=("stock_center",))
    )
    evaluator.observe(Snapshot(6, picked_items=("stock_right",)))
    assert evaluator.failure_code == "extra_carton_picked"
