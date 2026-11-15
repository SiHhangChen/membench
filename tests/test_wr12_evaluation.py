from dataclasses import replace

import pytest

from bench.evaluation.report_profiles import get_report_stage_profile
from bench.evaluation.report_stages import validate_report_stages
from bench.tasks.wr12.evaluation import WR12EvaluationSnapshot as S
from bench.tasks.wr12.evaluation import WR12Evaluator


def ready(target="left"):
    evaluator = WR12Evaluator(target)
    for step in range(3):
        evaluator.observe(S(step, at_observe=True, can_visible=True))
    for step in range(3, 6):
        evaluator.observe(S(step, boxes_sealed=True, can_hidden=True, can_occluded_by_box=True))
    evaluator.observe(S(6, at_operate=True, boxes_sealed=True, can_hidden=True, block_picked=True))
    return evaluator


@pytest.mark.parametrize("target", ["left", "right"])
def test_first_correct_supported_placement_succeeds(target):
    evaluator = ready(target)
    for step in range(7, 10):
        evaluator.observe(
            S(step, at_operate=True, boxes_sealed=True, can_hidden=True, released_in=(target,))
        )
    assert evaluator.success
    assert evaluator.stage_steps["block_picked"] == 6
    assert evaluator.stage_steps["block_placed"] == 9
    assert evaluator.first_release_step == 7
    assert all(v["passed"] for v in evaluator.to_report_stages()["stages"].values())


def test_wrong_choice_counts_manipulation_but_cannot_be_repaired():
    evaluator = ready()
    for step in range(7, 10):
        evaluator.observe(
            S(step, at_operate=True, boxes_sealed=True, can_hidden=True, released_in=("right",))
        )
    assert evaluator.failure_code == "wrong_box_placement"
    assert evaluator.stage_steps["block_placed"] == 9
    for step in range(10, 14):
        evaluator.observe(
            S(step, at_operate=True, boxes_sealed=True, can_hidden=True, released_in=("left",))
        )
    assert not evaluator.success
    assert evaluator.first_placed_box == "right"


@pytest.mark.parametrize("correct", [True, False])
def test_public_report_has_only_pick_place_success_with_internal_memory_checks(correct):
    evaluator = ready()
    box = "left" if correct else "right"
    for step in range(7, 10):
        evaluator.observe(
            S(step, at_operate=True, boxes_sealed=True, can_hidden=True, released_in=(box,))
        )
    report = validate_report_stages(
        evaluator.to_report_stages(),
        expected_task_id="WR-12",
        expected_profile_id="WR-12/report-v2",
        expected_success=correct,
    )
    assert report["order"] == ["block_picked", "block_placed", "success"]
    assert set(report["stages"]) == set(report["order"])
    assert report["stages"]["block_picked"]["first_pass_step"] == 6
    assert report["stages"]["block_placed"]["first_pass_step"] == 9
    assert report["stages"]["success"]["passed"] is correct
    diagnostics = evaluator.to_diagnostics()
    assert diagnostics["stage_checks"]["can_observed"]
    assert diagnostics["stage_checks"]["memory_barrier"]


def test_historical_five_stage_report_remains_readable():
    evaluator = ready()
    for step in range(7, 10):
        evaluator.observe(
            S(step, at_operate=True, boxes_sealed=True, can_hidden=True, released_in=("left",))
        )
    stages = get_report_stage_profile("WR-12", profile_id="WR-12/report-v1").stage_ids
    report = evaluator.to_report_stages()
    report.update(
        profile_id="WR-12/report-v1",
        order=list(stages),
        stages={
            stage: {
                "applicable": True,
                "passed": True,
                "first_pass_step": evaluator.stage_steps[stage],
            }
            for stage in stages
        },
    )
    assert validate_report_stages(report, expected_success=True)["order"] == list(stages)


def test_first_contact_release_locks_choice_before_dwell():
    evaluator = ready()
    evaluator.observe(
        S(7, at_operate=True, boxes_sealed=True, can_hidden=True, released_in=("right",))
    )
    evaluator.observe(
        S(8, at_operate=True, boxes_sealed=True, can_hidden=True, released_in=("left",))
    )
    assert evaluator.failure_code == "placement_switched"


@pytest.mark.parametrize("visible", [True, False])
def test_reobservation_or_unknown_visibility_after_barrier_fails(visible):
    evaluator = ready()
    evaluator.observe(S(7, can_visible=visible, can_hidden=False))
    assert evaluator.failure_code == "can_reobserved_after_barrier"


def test_offscreen_alone_cannot_establish_occlusion():
    evaluator = WR12Evaluator("left")
    for step in range(3):
        evaluator.observe(S(step, at_observe=True, can_visible=True))
    for step in range(3, 10):
        evaluator.observe(S(step, at_operate=True, boxes_sealed=True, can_hidden=True))
    assert evaluator.stage_steps["memory_barrier"] is None


def test_dwell_requires_distinct_contiguous_control_steps():
    evaluator = ready()
    sample = S(7, at_operate=True, boxes_sealed=True, can_hidden=True, released_in=("left",))
    for _ in range(10):
        evaluator.observe(sample)
    evaluator.observe(replace(sample, step_id=9))
    evaluator.observe(replace(sample, step_id=10))
    assert not evaluator.success
    evaluator.observe(replace(sample, step_id=11))
    assert evaluator.success


@pytest.mark.parametrize(
    "sample,code",
    [
        (S(0, block_picked=True), "pick_before_memory_barrier"),
        (S(0, released_in=("left",)), "placement_without_pick"),
        (S(0, can_undisturbed=False), "can_disturbed"),
    ],
)
def test_physical_prerequisites(sample, code):
    evaluator = WR12Evaluator("left").observe(sample)
    assert evaluator.failure_code == code


def test_open_boxes_cannot_establish_barrier_even_when_currently_occluded():
    evaluator = WR12Evaluator("left")
    for step in range(3):
        evaluator.observe(S(step, at_observe=True, can_visible=True))
    for step in range(3, 8):
        evaluator.observe(S(step, at_operate=True, can_hidden=True, can_occluded_by_box=True))
    assert evaluator.stage_steps["memory_barrier"] is None


def test_reopening_after_seal_fails_even_if_can_is_out_of_view():
    evaluator = ready()
    evaluator.observe(S(7, can_hidden=True, boxes_sealed=False))
    assert evaluator.failure_code == "can_reobserved_after_barrier"
