"""History-aware physical evaluation shared by policies and expert recordings."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from bench.evaluation.report_profiles import WR11_REPORT_PROFILE

from .domain.contracts import ITEM_IDS

REPORT_STAGE_PROFILE_ID = WR11_REPORT_PROFILE.profile_id
STAGE_ORDER = WR11_REPORT_PROFILE.stage_ids


@dataclass(frozen=True)
class WR11EvaluationSnapshot:
    step_id: int
    at_sample: bool = False
    sample_visible: bool = False
    sample_in_any_camera: bool = True
    at_stock: bool = False
    sample_undisturbed: bool = True
    picked_items: tuple[str, ...] = ()
    delivered_items: tuple[str, ...] = ()
    details: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if int(self.step_id) != self.step_id or self.step_id < 0:
            raise ValueError("WR-11 step_id must be a non-negative integer")
        if not {*self.picked_items, *self.delivered_items} <= set(ITEM_IDS):
            raise ValueError("WR-11 snapshot contains an unknown stock item")


class WR11Evaluator:
    def __init__(self, matching_item, *, observation_dwell_steps=3, placement_dwell_steps=3):
        if matching_item not in ITEM_IDS:
            raise ValueError("WR-11 matching_item must identify one stock carton")
        for value in (observation_dwell_steps, placement_dwell_steps):
            if int(value) != value or value < 1:
                raise ValueError("WR-11 dwell steps must be positive integers")
        self.matching_item = matching_item
        self.observation_dwell_steps = int(observation_dwell_steps)
        self.placement_dwell_steps = int(placement_dwell_steps)
        self.reset()

    def reset(self, snapshot=None):
        self.stage_steps = {stage: None for stage in STAGE_ORDER}
        self.first_picked_item = None
        self.first_pick_step = None
        self.failure_code = None
        self.success = False
        self._last = None if snapshot is None else snapshot.step_id
        self._seen_run = self._delivery_run = 0
        self._details = {}
        return self

    @property
    def terminal(self):
        return bool(self.success or self.failure_code)

    def observe(self, snapshot):
        step = snapshot.step_id
        if self._last is not None and step < self._last:
            raise ValueError("WR-11 evaluation step moved backwards")
        if self.terminal or step == self._last:
            return self
        contiguous = self._last is not None and step == self._last + 1
        self._last = step
        self._details = dict(snapshot.details)
        if not contiguous:
            self._seen_run = self._delivery_run = 0
        if not snapshot.sample_undisturbed:
            self.failure_code = "sample_disturbed"
            return self
        if self.stage_steps["sample_observed"] is None:
            self._seen_run = (
                self._seen_run + 1 if snapshot.at_sample and snapshot.sample_visible else 0
            )
            if self._seen_run >= self.observation_dwell_steps:
                self.stage_steps["sample_observed"] = step
        if (
            self.stage_steps["sample_observed"] is not None
            and snapshot.at_stock
            and not snapshot.sample_in_any_camera
            and self.stage_steps["stock_reached"] is None
        ):
            self.stage_steps["stock_reached"] = step
        picked = set(snapshot.picked_items)
        if picked and self.first_picked_item is None:
            self.first_pick_step = step
            if len(picked) != 1:
                self.failure_code = "ambiguous_first_pick"
                return self
            self.first_picked_item = next(iter(picked))
            # The first physical selection is permanent, even if later released.
            if self.first_picked_item != self.matching_item:
                self.failure_code = "wrong_first_pick"
            elif self.stage_steps["stock_reached"] is None or not snapshot.at_stock:
                self.failure_code = "pick_before_memory_barrier"
            elif snapshot.sample_in_any_camera:
                self.failure_code = "sample_visible_at_selection"
            else:
                self.stage_steps["first_pick_correct"] = step
        elif picked - {self.first_picked_item}:
            self.failure_code = "extra_carton_picked"
        if self.failure_code:
            return self
        delivered = (
            self.stage_steps["first_pick_correct"] is not None
            and step > self.first_pick_step
            and self.matching_item in snapshot.delivered_items
        )
        self._delivery_run = self._delivery_run + 1 if delivered else 0
        if self._delivery_run >= self.placement_dwell_steps:
            self.stage_steps["item_delivered"] = step
            self.stage_steps["success"] = step
            self.success = True
        return self

    def to_diagnostics(self):
        return {
            **self._details,
            "success": self.success,
            "terminal": self.terminal,
            "stop_execution": self.terminal,
            "failure_code": self.failure_code,
            "first_picked_item": self.first_picked_item,
            "first_pick_step": self.first_pick_step,
            "first_pick_correct": (
                None
                if self.first_picked_item is None
                else self.first_picked_item == self.matching_item
            ),
            "stage_checks": {key: value is not None for key, value in self.stage_steps.items()},
            "stage_steps": dict(self.stage_steps),
            "placement_run": self._delivery_run,
        }

    def to_report_stages(self):
        return {
            "schema_version": "membench.report_stages/v1",
            "task_id": "WR-11",
            "profile_id": REPORT_STAGE_PROFILE_ID,
            "order": list(STAGE_ORDER),
            "stages": {
                stage: {"applicable": True, "passed": step is not None, "first_pass_step": step}
                for stage, step in self.stage_steps.items()
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
