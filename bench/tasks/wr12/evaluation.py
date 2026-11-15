"""First supported release fixes the choice; re-observation cannot repair it."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from bench.evaluation.report_profiles import WR12_REPORT_PROFILE

from .domain.contracts import BOX_IDS

REPORT_STAGE_PROFILE_ID = WR12_REPORT_PROFILE.profile_id
REPORT_STAGE_ORDER = WR12_REPORT_PROFILE.stage_ids
STAGE_ORDER = ("can_observed", "memory_barrier", *REPORT_STAGE_ORDER)


@dataclass(frozen=True)
class WR12EvaluationSnapshot:
    step_id: int
    at_observe: bool = False
    can_visible: bool = False
    at_operate: bool = False
    can_hidden: bool = False
    boxes_sealed: bool = False
    can_occluded_by_box: bool = False
    can_undisturbed: bool = True
    block_picked: bool = False
    released_in: tuple[str, ...] = ()
    details: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if int(self.step_id) != self.step_id or self.step_id < 0:
            raise ValueError("WR-12 step must be a nonnegative integer")
        if not set(self.released_in) <= set(BOX_IDS):
            raise ValueError("WR-12 unknown release box")


class WR12Evaluator:
    def __init__(
        self,
        target_box,
        *,
        observation_dwell_steps=3,
        hidden_dwell_steps=3,
        placement_dwell_steps=3,
    ):
        if target_box not in BOX_IDS:
            raise ValueError("WR-12 unknown target box")
        for value in (observation_dwell_steps, hidden_dwell_steps, placement_dwell_steps):
            if int(value) != value or value < 1:
                raise ValueError("WR-12 dwell must be a positive integer")
        self.target_box = target_box
        self.observation_dwell_steps = observation_dwell_steps
        self.hidden_dwell_steps = hidden_dwell_steps
        self.placement_dwell_steps = placement_dwell_steps
        self.reset()

    def reset(self, snapshot=None):
        self.stage_steps = dict.fromkeys(STAGE_ORDER)
        self.first_placed_box = self.first_release_step = None
        self.failure_code = None
        self.success = False
        self._last = None if snapshot is None else snapshot.step_id
        self._seen_run = self._hidden_run = self._place_run = 0
        self._candidate = None
        self._details = {}
        return self

    @property
    def terminal(self):
        return bool(self.success or self.failure_code)

    def observe(self, sample):
        step = sample.step_id
        if self.terminal:
            return self
        if self._last is not None and step < self._last:
            raise ValueError("WR-12 evaluation moved backwards")
        if step == self._last:
            return self
        if self._last is None or step != self._last + 1:
            self._seen_run = self._hidden_run = self._place_run = 0
            self._candidate = None
        self._last = step
        self._details = dict(sample.details)
        if not sample.can_undisturbed:
            self.failure_code = "can_disturbed"
            return self
        if self.stage_steps["can_observed"] is None:
            self._seen_run = self._seen_run + 1 if sample.at_observe and sample.can_visible else 0
            if self._seen_run >= self.observation_dwell_steps:
                self.stage_steps["can_observed"] = step
        if (
            self.stage_steps["can_observed"] is not None
            and self.stage_steps["memory_barrier"] is None
        ):
            hidden = sample.boxes_sealed and sample.can_hidden and sample.can_occluded_by_box
            self._hidden_run = self._hidden_run + 1 if hidden else 0
            if self._hidden_run >= self.hidden_dwell_steps:
                self.stage_steps["memory_barrier"] = step
        elif self.stage_steps["memory_barrier"] is not None and (
            not sample.boxes_sealed or not sample.can_hidden
        ):
            # Includes unknown visibility. Once the robot reaches the operation
            # area, returning to peek or exposing the can with the wrist fails.
            self.failure_code = "can_reobserved_after_barrier"
            return self
        if sample.block_picked and self.stage_steps["block_picked"] is None:
            self.stage_steps["block_picked"] = step
            if self.stage_steps["memory_barrier"] is None or not sample.at_operate:
                self.failure_code = "pick_before_memory_barrier"
                return self
        releases = set(sample.released_in)
        if releases:
            if self.stage_steps["block_picked"] is None:
                self.failure_code = "placement_without_pick"
                return self
            if not sample.at_operate or len(releases) != 1:
                self.failure_code = "invalid_placement"
                return self
            box = next(iter(releases))
            if self.first_placed_box is None:
                self.first_placed_box, self.first_release_step = box, step
            elif box != self.first_placed_box:
                self.failure_code = "placement_switched"
                return self
            self._place_run = self._place_run + 1 if box == self._candidate else 1
            self._candidate = box
            if self._place_run >= self.placement_dwell_steps:
                self.stage_steps["block_placed"] = step
                if box == self.target_box:
                    self.stage_steps["success"] = step
                    self.success = True
                else:
                    self.failure_code = "wrong_box_placement"
        else:
            self._place_run = 0
            self._candidate = None
        return self

    def to_diagnostics(self):
        return {
            **self._details,
            "success": self.success,
            "terminal": self.terminal,
            "stop_execution": self.terminal,
            "failure_code": self.failure_code,
            "first_placed_box": self.first_placed_box,
            "first_release_step": self.first_release_step,
            "first_choice_correct": None
            if self.first_placed_box is None
            else self.first_placed_box == self.target_box,
            "stage_checks": {key: value is not None for key, value in self.stage_steps.items()},
            "stage_steps": dict(self.stage_steps),
        }

    def to_report_stages(self):
        return {
            "schema_version": "membench.report_stages/v1",
            "task_id": "WR-12",
            "profile_id": REPORT_STAGE_PROFILE_ID,
            "order": list(REPORT_STAGE_ORDER),
            "stages": {
                stage: {
                    "applicable": True,
                    "passed": self.stage_steps[stage] is not None,
                    "first_pass_step": self.stage_steps[stage],
                }
                for stage in REPORT_STAGE_ORDER
            },
        }


def evaluate_initialized_runtime(runtime):
    return runtime.env.evaluator.observe(runtime.env.evaluation_snapshot())


def success_for_runtime(runtime, info=None):
    return evaluate_initialized_runtime(runtime).success


def diagnostics_for_runtime(runtime):
    return evaluate_initialized_runtime(runtime).to_diagnostics()


def report_stages_for_runtime(runtime):
    return evaluate_initialized_runtime(runtime).to_report_stages()
