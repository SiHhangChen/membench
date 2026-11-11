"""Frozen public report-stage profiles for the benchmark task set."""

from __future__ import annotations

from dataclasses import dataclass
import re


_STAGE_ID = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class ReportStageProfile:
    """One versioned public funnel owned by the benchmark specification."""

    task_id: str
    profile_id: str
    stage_ids: tuple[str, ...]
    conditional_stage_ids: frozenset[str] = frozenset()
    success_stage_id: str = "success"

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("report profile task_id must be non-empty")
        if not self.profile_id:
            raise ValueError("report profile profile_id must be non-empty")
        if not self.stage_ids or self.stage_ids[-1] != self.success_stage_id:
            raise ValueError(
                "report profile must end with its declared success stage"
            )
        if len(set(self.stage_ids)) != len(self.stage_ids):
            raise ValueError("report profile stage_ids must be unique")
        invalid = [
            stage_id
            for stage_id in self.stage_ids
            if not _STAGE_ID.fullmatch(stage_id)
        ]
        if invalid:
            raise ValueError(f"invalid report profile stage ids: {invalid}")
        unknown_conditional = self.conditional_stage_ids - set(self.stage_ids)
        if unknown_conditional:
            raise ValueError(
                "conditional report stages must belong to the profile: "
                f"{sorted(unknown_conditional)}"
            )
        if self.success_stage_id in self.conditional_stage_ids:
            raise ValueError("the report profile success stage cannot be conditional")


def _profile(
    task_id: str,
    *stage_ids: str,
    conditional: tuple[str, ...] = (),
    success_stage_id: str = "success",
    version: int = 1,
) -> ReportStageProfile:
    return ReportStageProfile(
        task_id=task_id,
        profile_id=f"{task_id}/report-v{int(version)}",
        stage_ids=tuple(stage_ids),
        conditional_stage_ids=frozenset(conditional),
        success_stage_id=success_stage_id,
    )


TM01_REPORT_PROFILE_V1 = _profile(
    "TM01",
    "burner_on",
    "one_drawer_inspected",
    "both_threads_completed",
    "success",
)
TM02_REPORT_PROFILE_V1 = _profile(
    "TM02",
    "faucet_on",
    "pot_filled_and_can_retrieved",
    "both_threads_completed",
    "success",
)
TM01_REPORT_PROFILE = _profile(
    "TM01",
    "burner_on",
    "one_drawer_inspected",
    "one_thread_completed",
    "both_threads_completed",
    "success",
    version=2,
)
TM02_REPORT_PROFILE = _profile(
    "TM02",
    "faucet_on",
    "pot_filled_and_can_retrieved",
    "one_thread_completed",
    "both_threads_completed",
    "success",
    version=2,
)
TS01_REPORT_PROFILE = _profile(
    "TS01",
    "sugar_1",
    "sugar_2",
    "sugar_3",
    "success",
    conditional=("sugar_2", "sugar_3"),
)
TS02_REPORT_PROFILE = _profile(
    "TS02",
    "one_drawer_checked",
    "target_found",
    "search_stopped_after_target",
    "button_pressed",
    "correct_press_count",
    success_stage_id="correct_press_count",
)
TS03_REPORT_PROFILE = _profile(
    "TS03",
    "sugar_1",
    "sugar_2",
    "sugar_3",
    "success",
    conditional=("sugar_2", "sugar_3"),
    version=2,
)
TS04_REPORT_PROFILE = _profile(
    "TS04",
    "one_drawer_checked",
    "target_found",
    "search_stopped_after_target",
    "button_pressed",
    "correct_press_count",
    success_stage_id="correct_press_count",
)
WA01_REPORT_PROFILE = _profile(
    "WA-01",
    "sugar_added",
    "one_cup_ready",
    "two_cups_ready",
    "success",
)
WA02_REPORT_PROFILE = _profile(
    "WA-02",
    "first_cup_filled",
    "first_cup_ready",
    "second_cup_filled",
    "both_cups_ready",
    "success",
    version=2,
)
WA03_REPORT_PROFILE = _profile(
    "WA-03",
    "sugar_added",
    "one_cup_ready",
    "two_cups_ready",
    "three_cups_ready",
    "success",
)
WA05_REPORT_PROFILE = _profile(
    "WA-05",
    "cup_filled",
    "cup_staged",
    "success",
)
WA06_REPORT_PROFILE = _profile(
    "WA-06",
    "sugar_added",
    "sugared_cup_selected",
    "success",
)
WR01_REPORT_PROFILE = _profile(
    "WR-01",
    "one_can_stored",
    "two_cans_stored",
    "correct_cabinet_opened",
    "success",
    version=2,
)
WR03_REPORT_PROFILE = _profile(
    "WR-03",
    "can_stored",
    "storage_hidden",
    "correct_cabinet_opened",
    "success",
)
WR05_REPORT_PROFILE = _profile(
    "WR-05",
    "one_item_staged",
    "two_items_staged",
    "one_item_correctly_returned",
    "success",
)
WR06_REPORT_PROFILE = _profile(
    "WR-06",
    "item_removed",
    "item_staged",
    "correct_return_level",
    "success",
)
WR07_REPORT_PROFILE = _profile(
    "WR-07",
    "item_picked",
    "memory_color_latched",
    "target_plate_matched",
    "success",
    version=3,
)
WR08_REPORT_PROFILE = _profile(
    "WR-08",
    "one_item_picked",
    "one_item_placed",
    "one_item_matched",
    "two_items_picked",
    "two_items_placed",
    "two_items_matched",
    "success",
    version=3,
)
WR11_REPORT_PROFILE = _profile(
    "WR-11",
    "sample_observed",
    "stock_reached",
    "first_pick_correct",
    "item_delivered",
    "success",
)
WX01_REPORT_PROFILE = _profile(
    "WX-01",
    "wipe_completed",
    "cleaning_tool_released",
    "success",
    version=2,
)
WX02_REPORT_PROFILE = _profile(
    "WX-02",
    "tool_acquired",
    "primary_cup_moved",
    "primary_stain_cleaned",
    "primary_cup_returned",
    "secondary_cup_moved",
    "secondary_stain_cleaned",
    "secondary_cup_returned",
    "tool_returned",
    "success",
)


REPORT_STAGE_PROFILES = (
    TM01_REPORT_PROFILE,
    TM02_REPORT_PROFILE,
    TS01_REPORT_PROFILE,
    TS02_REPORT_PROFILE,
    TS03_REPORT_PROFILE,
    TS04_REPORT_PROFILE,
    WA01_REPORT_PROFILE,
    WA02_REPORT_PROFILE,
    WA03_REPORT_PROFILE,
    WA05_REPORT_PROFILE,
    WA06_REPORT_PROFILE,
    WR01_REPORT_PROFILE,
    WR03_REPORT_PROFILE,
    WR05_REPORT_PROFILE,
    WR06_REPORT_PROFILE,
    WR07_REPORT_PROFILE,
    WR08_REPORT_PROFILE,
    WR11_REPORT_PROFILE,
    WX01_REPORT_PROFILE,
    WX02_REPORT_PROFILE,
)

_PROFILES_BY_TASK = {profile.task_id: profile for profile in REPORT_STAGE_PROFILES}
# Historical reports remain readable only under their explicit profile id.
# Keep them out of the default, one-profile-per-task catalog.
_HISTORICAL_REPORT_STAGE_PROFILES = (TM01_REPORT_PROFILE_V1, TM02_REPORT_PROFILE_V1)
_PROFILES_BY_ID = {
    profile.profile_id: profile
    for profile in (*REPORT_STAGE_PROFILES, *_HISTORICAL_REPORT_STAGE_PROFILES)
}
if len(_PROFILES_BY_TASK) != len(REPORT_STAGE_PROFILES):
    raise RuntimeError("benchmark report profiles must have unique task ids")
if len(_PROFILES_BY_ID) != len(REPORT_STAGE_PROFILES) + len(_HISTORICAL_REPORT_STAGE_PROFILES):
    raise RuntimeError("benchmark report profiles must have unique profile ids")


def get_report_stage_profile(
    task_id: str,
    *,
    profile_id: str | None = None,
) -> ReportStageProfile:
    """Resolve the current default or an explicitly requested historical profile."""

    try:
        profile = _PROFILES_BY_TASK[str(task_id)]
    except KeyError as exc:
        raise KeyError(f"task {task_id!r} has no benchmark report profile") from exc
    if profile_id is not None:
        selected = _PROFILES_BY_ID.get(str(profile_id))
        if selected is None or selected.task_id != profile.task_id:
            raise KeyError(
                f"task {task_id!r} does not use report profile {profile_id!r}"
            )
        return selected
    return profile


__all__ = [
    "REPORT_STAGE_PROFILES",
    "ReportStageProfile",
    "TM01_REPORT_PROFILE",
    "TM01_REPORT_PROFILE_V1",
    "TM02_REPORT_PROFILE",
    "TM02_REPORT_PROFILE_V1",
    "TS01_REPORT_PROFILE",
    "TS02_REPORT_PROFILE",
    "TS03_REPORT_PROFILE",
    "TS04_REPORT_PROFILE",
    "WA01_REPORT_PROFILE",
    "WA02_REPORT_PROFILE",
    "WA03_REPORT_PROFILE",
    "WA05_REPORT_PROFILE",
    "WA06_REPORT_PROFILE",
    "WR01_REPORT_PROFILE",
    "WR03_REPORT_PROFILE",
    "WR05_REPORT_PROFILE",
    "WR06_REPORT_PROFILE",
    "WR07_REPORT_PROFILE",
    "WR08_REPORT_PROFILE",
    "WR11_REPORT_PROFILE",
    "WX01_REPORT_PROFILE",
    "WX02_REPORT_PROFILE",
    "get_report_stage_profile",
]
