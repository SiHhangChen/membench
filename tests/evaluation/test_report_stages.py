from __future__ import annotations

import copy

import pytest

from bench.evaluation.report_stages import (
    REPORT_STAGES_SCHEMA,
    validate_report_stages,
)
from bench.evaluation.report_profiles import (
    REPORT_STAGE_PROFILES,
    get_report_stage_profile,
)


def _k1_report() -> dict[str, object]:
    return {
        "schema_version": REPORT_STAGES_SCHEMA,
        "task_id": "TS01",
        "profile_id": "TS01/report-v1",
        "order": ["sugar_1", "sugar_2", "sugar_3", "success"],
        "stages": {
            "sugar_1": {
                "applicable": True,
                "passed": True,
                "first_pass_step": 4,
            },
            "sugar_2": {
                "applicable": False,
                "passed": None,
                "first_pass_step": None,
            },
            "sugar_3": {
                "applicable": False,
                "passed": None,
                "first_pass_step": None,
            },
            "success": {
                "applicable": True,
                "passed": True,
                "first_pass_step": 7,
            },
        },
    }


def test_valid_report_normalizes_conditional_stages() -> None:
    report = validate_report_stages(
        _k1_report(),
        expected_task_id="TS01",
        expected_profile_id="TS01/report-v1",
        expected_success=True,
    )

    assert report["order"] == ["sugar_1", "sugar_2", "sugar_3", "success"]
    assert report["stages"]["sugar_2"] == {
        "applicable": False,
        "passed": None,
        "first_pass_step": None,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (("passed", False), ("first_pass_step", 5)),
)
def test_non_applicable_stage_requires_null_verdicts(
    field: str,
    value: object,
) -> None:
    report = _k1_report()
    report["stages"]["sugar_2"][field] = value

    with pytest.raises(ValueError, match="non-applicable"):
        validate_report_stages(report)


def test_downstream_stage_cannot_pass_after_applicable_failure() -> None:
    report = _k1_report()
    report["stages"]["sugar_1"] = {
        "applicable": True,
        "passed": False,
        "first_pass_step": None,
    }

    with pytest.raises(ValueError, match="predecessor failed"):
        validate_report_stages(report)


def test_only_profile_declared_stages_may_be_non_applicable() -> None:
    report = _k1_report()
    report["stages"]["sugar_1"] = {
        "applicable": False,
        "passed": None,
        "first_pass_step": None,
    }

    with pytest.raises(ValueError, match="not conditional"):
        validate_report_stages(report)


def test_report_order_must_match_the_frozen_profile() -> None:
    report = _k1_report()
    report["order"] = ["sugar_1", "sugar_3", "sugar_2", "success"]

    with pytest.raises(ValueError, match="does not match"):
        validate_report_stages(report)


def test_unknown_report_profile_is_rejected() -> None:
    report = _k1_report()
    report["profile_id"] = "TS01/report-v2"

    with pytest.raises(ValueError, match="unsupported benchmark report profile"):
        validate_report_stages(report)


@pytest.mark.parametrize(
    ("expected", "value", "message"),
    (
        ("task_id", "TS02", "task mismatch"),
        ("profile_id", "TS01/report-v2", "profile mismatch"),
        ("success", False, "episode success"),
    ),
)
def test_report_identity_and_success_must_match_episode(
    expected: str,
    value: object,
    message: str,
) -> None:
    report = copy.deepcopy(_k1_report())
    kwargs = {
        "expected_task_id": "TS01",
        "expected_profile_id": "TS01/report-v1",
        "expected_success": True,
    }
    if expected == "success":
        report["stages"]["success"]["passed"] = value
        report["stages"]["success"]["first_pass_step"] = None
    else:
        kwargs[f"expected_{expected}"] = value

    with pytest.raises(ValueError, match=message):
        validate_report_stages(report, **kwargs)


def test_frozen_profile_catalog_covers_all_report_tasks() -> None:
    expected = {
        "TM01": (
            ("burner_on", "one_drawer_inspected", "one_thread_completed", "both_threads_completed", "success"),
            frozenset(),
        ),
        "TM02": (
            ("faucet_on", "pot_filled_and_can_retrieved", "one_thread_completed", "both_threads_completed", "success"),
            frozenset(),
        ),
        "TS01": (
            ("sugar_1", "sugar_2", "sugar_3", "success"),
            frozenset({"sugar_2", "sugar_3"}),
        ),
        "TS02": (
            (
                "one_drawer_checked",
                "target_found",
                "search_stopped_after_target",
                "button_pressed",
                "correct_press_count",
            ),
            frozenset(),
        ),
        "TS03": (
            ("sugar_1", "sugar_2", "success"),
            frozenset({"sugar_2"}),
        ),
        "TS04": (
            (
                "one_drawer_checked",
                "target_found",
                "search_stopped_after_target",
                "button_pressed",
                "correct_press_count",
            ),
            frozenset(),
        ),
        "WA-01": (
            ("sugar_added", "one_cup_ready", "two_cups_ready", "select", "success"),
            frozenset(),
        ),
        "WA-02": (
            (
                "first_cup_filled",
                "first_cup_ready",
                "second_cup_filled",
                "both_cups_ready",
                "success",
            ),
            frozenset(),
        ),
        "WA-03": (
            (
                "sugar_added",
                "one_cup_ready",
                "two_cups_ready",
                "three_cups_ready",
                "select",
                "success",
            ),
            frozenset(),
        ),
        "WA-05": (
            ("cup_filled", "cup_staged", "success"),
            frozenset(),
        ),
        "WR-01": (
            (
                "one_can_stored",
                "two_cans_stored",
                "correct_cabinet_opened",
                "success",
            ),
            frozenset(),
        ),
        "WR-03": (
            ("one_door_closed", "both_doors_closed", "correct_cabinet_opened", "success"),
            frozenset(),
        ),
        "WR-05": (
            (
                "one_item_staged",
                "two_items_staged",
                "one_item_correctly_returned",
                "success",
            ),
            frozenset(),
        ),
        "WR-06": (
            ("item_removed", "item_staged", "correct_return_level", "success"),
            frozenset(),
        ),
        "WR-07": (
            ("item_picked", "item_placed", "target_plate_matched", "success"),
            frozenset(),
        ),
        "WR-08": (
            (
                "one_item_picked",
                "one_item_placed",
                "one_item_matched",
                "two_items_picked",
                "two_items_placed",
                "two_items_matched",
                "success",
            ),
            frozenset(),
        ),
        "WR-11": (
            ("sample_observed", "stock_reached", "first_pick_correct", "item_delivered", "success"),
            frozenset(),
        ),
        "WX-01": (
            ("wipe_completed", "cleaning_tool_released", "success"),
            frozenset(),
        ),
        "WX-02": (
            (
                "tool_acquired",
                "primary_cup_moved",
                "primary_stain_cleaned",
                "primary_cup_returned",
                "secondary_cup_moved",
                "secondary_stain_cleaned",
                "secondary_cup_returned",
                "tool_returned",
                "success",
            ),
            frozenset(),
        ),
    }

    assert {profile.task_id for profile in REPORT_STAGE_PROFILES} == set(expected)
    for task_id, (stage_ids, conditional_stage_ids) in expected.items():
        profile = get_report_stage_profile(task_id)
        expected_version = {"TM01": 2, "TM02": 2, "WA-01": 2, "WA-02": 2, "WA-03": 2, "WR-01": 2, "WR-03": 2, "WR-07": 4, "WR-08": 3}.get(task_id, 1)
        assert profile.profile_id == f"{task_id}/report-v{expected_version}"
        assert profile.stage_ids == stage_ids
        assert profile.conditional_stage_ids == conditional_stage_ids
        assert profile.success_stage_id == (
            "correct_press_count" if task_id in {"TS02", "TS04"} else "success"
        )


def test_profile_final_stage_can_carry_episode_success_without_extra_stage() -> None:
    report = {
        "schema_version": REPORT_STAGES_SCHEMA,
        "task_id": "TS02",
        "profile_id": "TS02/report-v1",
        "order": [
            "one_drawer_checked",
            "target_found",
            "search_stopped_after_target",
            "button_pressed",
            "correct_press_count",
        ],
        "stages": {
            stage_id: {
                "applicable": True,
                "passed": True,
                "first_pass_step": index,
            }
            for index, stage_id in enumerate(
                (
                    "one_drawer_checked",
                    "target_found",
                    "search_stopped_after_target",
                    "button_pressed",
                    "correct_press_count",
                ),
                start=1,
            )
        },
    }

    normalized = validate_report_stages(report, expected_success=True)

    assert normalized["order"][-1] == "correct_press_count"
    assert "success" not in normalized["stages"]
