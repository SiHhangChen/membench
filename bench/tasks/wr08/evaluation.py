"""Physical policy evaluation and ordered expert verification for WR-08.

Policy evaluation follows the same physical contract as WR-07, twice: a
carton must be physically picked, released onto support, and confirmed on its
correct target plate. Color-memory events remain available for expert
recording, but are not policy milestones.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from bench.evaluation.report_profiles import WR08_REPORT_PROFILE
from bench.tasks.wr08.domain.contracts import (
    COLOR_NAMES,
    ITEM_IDS,
    SOURCE_SLOTS,
    TARGET_SLOTS,
)

STAGE_ORDER = WR08_REPORT_PROFILE.stage_ids
EXPERT_STAGE_ORDER = (
    "first_item_picked",
    "first_item_color_latched",
    "first_item_matched",
    "second_item_picked",
    "second_item_color_latched",
    "second_item_matched",
    "success",
)
STAGE_LABELS = {
    "one_item_picked": "拿起一件牛奶盒",
    "one_item_placed": "放下一件牛奶盒",
    "one_item_matched": "一件牛奶盒放对目标盘",
    "two_items_picked": "两件牛奶盒均已拿起",
    "two_items_placed": "两件牛奶盒均已放下",
    "two_items_matched": "两件牛奶盒均已放对目标盘",
    "first_item_picked": "取起第一件牛奶盒",
    "first_item_color_latched": "记住第一件牛奶盒颜色",
    "first_item_matched": "将第一件牛奶盒放到对应糖盘",
    "second_item_picked": "取起第二件牛奶盒",
    "second_item_color_latched": "记住第二件牛奶盒颜色",
    "second_item_matched": "将第二件牛奶盒放到对应糖盘",
    "success": "完成颜色匹配搬运",
}
STAGE_PREREQUISITES: dict[str, str | None] = {
    name: None if index == 0 else STAGE_ORDER[index - 1]
    for index, name in enumerate(STAGE_ORDER)
}
REPORT_STAGE_ORDER = WR08_REPORT_PROFILE.stage_ids
REPORT_STAGE_PROFILE_ID = WR08_REPORT_PROFILE.profile_id
TASK_EVALUATOR_METADATA_KEY = "task_evaluator"
_EXPERT_REPORT_STAGE_SOURCES = {
    "one_item_picked": "first_item_picked",
    "two_items_picked": "second_item_picked",
    "one_item_placed": "first_item_matched",
    "two_items_placed": "second_item_matched",
    "one_item_matched": "first_item_matched",
    "two_items_matched": "second_item_matched",
    "success": "success",
}


def _order(values: Sequence[Any] | None) -> tuple[str, str]:
    result = tuple(ITEM_IDS if values is None else (str(value) for value in values))
    if len(result) != len(ITEM_IDS) or set(result) != set(ITEM_IDS):
        raise ValueError("WR-08 transport_order must contain each item exactly once")
    return result  # type: ignore[return-value]


def _item_map(values: Mapping[str, Any] | None, *, label: str) -> dict[str, str]:
    result = {str(key): str(value) for key, value in dict(values or {}).items()}
    unknown = set(result) - set(ITEM_IDS)
    if unknown:
        raise ValueError(f"unknown WR-08 item ids in {label}: {sorted(unknown)}")
    return result


def _bool_map(values: Mapping[str, Any] | None) -> dict[str, bool]:
    values = dict(values or {})
    unknown = set(values) - set(ITEM_IDS)
    if unknown:
        raise ValueError(f"unknown WR-08 item ids: {sorted(unknown)}")
    return {item: bool(values.get(item, False)) for item in ITEM_IDS}


@dataclass(frozen=True, slots=True)
class WR08EvaluationSnapshot:
    """Current physical facts for policies; recorded events for experts.

Policy pick/placement bits are live observations, not semantic latches.
The evaluator owns their history and ignores expert memory/protocol bits.
New fields are appended to preserve positional legacy snapshot callers.
"""

    step_id: int
    item_picked_by_item: Mapping[str, bool] = field(default_factory=dict)
    memory_latched_by_item: Mapping[str, bool] = field(default_factory=dict)
    item_matched_by_item: Mapping[str, bool] = field(default_factory=dict)
    released_by_item: Mapping[str, bool] = field(default_factory=dict)
    domain_success: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)
    transport_order: tuple[str, str] = ITEM_IDS
    source_slot_by_item: Mapping[str, str] = field(default_factory=dict)
    source_color_by_item: Mapping[str, str] = field(default_factory=dict)
    target_slot_by_item: Mapping[str, str] = field(default_factory=dict)
    target_slot_attempted_by_item: Mapping[str, str] = field(default_factory=dict)
    protocol_violation: bool = False
    item_placed_by_item: Mapping[str, bool] = field(default_factory=dict)
    require_expert_protocol: bool = False

    def __post_init__(self) -> None:
        step = int(self.step_id)
        if step < 0:
            raise ValueError("WR-08 evaluation step_id must be non-negative")
        object.__setattr__(self, "step_id", step)
        object.__setattr__(self, "transport_order", _order(self.transport_order))
        object.__setattr__(
            self,
            "source_slot_by_item",
            _item_map(self.source_slot_by_item, label="source_slot_by_item"),
        )
        object.__setattr__(
            self,
            "source_color_by_item",
            _item_map(self.source_color_by_item, label="source_color_by_item"),
        )
        object.__setattr__(
            self,
            "target_slot_by_item",
            _item_map(self.target_slot_by_item, label="target_slot_by_item"),
        )
        object.__setattr__(
            self,
            "target_slot_attempted_by_item",
            _item_map(
                self.target_slot_attempted_by_item,
                label="target_slot_attempted_by_item",
            ),
        )
        for label, values, allowed in (
            ("source_slot_by_item", self.source_slot_by_item, SOURCE_SLOTS),
            ("target_slot_by_item", self.target_slot_by_item, TARGET_SLOTS),
            (
                "target_slot_attempted_by_item",
                self.target_slot_attempted_by_item,
                TARGET_SLOTS,
            ),
        ):
            invalid = set(values.values()) - set(allowed)
            if invalid:
                raise ValueError(
                    f"invalid WR-08 slot values in {label}: {sorted(invalid)}"
                )
        invalid_colors = set(self.source_color_by_item.values()) - set(COLOR_NAMES)
        if invalid_colors:
            raise ValueError(f"invalid WR-08 source colors: {sorted(invalid_colors)}")
        object.__setattr__(
            self, "item_picked_by_item", _bool_map(self.item_picked_by_item)
        )
        placed = _bool_map(self.item_placed_by_item)
        # Legacy expert snapshots only carried a matched bit. Policies must
        # provide placement evidence explicitly; missing facts fail closed.
        if self.require_expert_protocol and not self.item_placed_by_item:
            placed = _bool_map(self.item_matched_by_item)
        object.__setattr__(self, "item_placed_by_item", placed)
        object.__setattr__(
            self, "memory_latched_by_item", _bool_map(self.memory_latched_by_item)
        )
        object.__setattr__(
            self, "item_matched_by_item", _bool_map(self.item_matched_by_item)
        )
        object.__setattr__(self, "released_by_item", _bool_map(self.released_by_item))
        object.__setattr__(self, "protocol_violation", bool(self.protocol_violation))
        object.__setattr__(self, "require_expert_protocol", bool(self.require_expert_protocol))
        object.__setattr__(self, "domain_success", bool(self.domain_success))
        object.__setattr__(self, "details", dict(self.details or {}))

    def raw_stage_values(self) -> dict[str, bool]:
        if self.require_expert_protocol:
            first, second = self.transport_order
            return {
                "first_item_picked": bool(self.item_picked_by_item[first]),
                "first_item_color_latched": bool(self.memory_latched_by_item[first]),
                "first_item_matched": bool(self.item_matched_by_item[first]),
                "second_item_picked": bool(self.item_picked_by_item[second]),
                "second_item_color_latched": bool(self.memory_latched_by_item[second]),
                "second_item_matched": bool(self.item_matched_by_item[second]),
                "success": bool(self.domain_success),
            }
        placed = {item: self.placed(item) for item in ITEM_IDS}
        matched = {item: self.matched(item) for item in ITEM_IDS}
        return {
            "one_item_picked": any(self.item_picked_by_item.values()),
            "one_item_placed": any(placed.values()),
            "one_item_matched": any(matched.values()),
            "two_items_picked": all(self.item_picked_by_item.values()),
            "two_items_placed": all(placed.values()),
            "two_items_matched": all(matched.values()),
            "success": all(matched.values()),
        }

    def placed(self, item: str) -> bool:
        return bool(self.item_placed_by_item[item] and self.released_by_item[item])

    def matched(self, item: str) -> bool:
        expected = self.target_slot_by_item.get(item)
        return bool(
            self.placed(item)
            and self.item_matched_by_item[item]
            and expected is not None
            and self.target_slot_attempted_by_item.get(item) == expected
        )


@dataclass(frozen=True, slots=True)
class StageEvalResult:
    applicable: bool
    observed: bool
    qualified: bool
    first_observed_step: int | None
    qualified_step: int | None
    invalidated: bool
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "applicable": bool(self.applicable),
            "observed": bool(self.observed),
            "qualified": bool(self.qualified),
            "first_observed_step": self.first_observed_step,
            "qualified_step": self.qualified_step,
            "invalidated": bool(self.invalidated),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class WR08EvalResult:
    success: bool
    terminal: bool
    failure_code: str | None
    stages: Mapping[str, StageEvalResult]
    details: Mapping[str, Any] = field(default_factory=dict)
    stop_execution: bool = False

    def to_diagnostics(self) -> dict[str, Any]:
        payload = dict(self.details or {})
        order = tuple(self.stages)
        labels = {name: STAGE_LABELS[name] for name in order}
        stages = {name: self.stages[name].to_dict() for name in order}
        qualified = {name: bool(values["qualified"]) for name, values in stages.items()}
        observed = {name: bool(values["observed"]) for name, values in stages.items()}
        evaluation = dict(payload.get("evaluation") or {})
        evaluation.update(
            {
                "schema_version": 1,
                "step_id": int(payload.get("step_id", evaluation.get("step_id", 0))),
                "stage_order": list(order),
                "stage_labels": labels,
                "stage_prerequisites": {
                    name: None if index == 0 else order[index - 1]
                    for index, name in enumerate(order)
                },
                "raw_stage_checks": payload.get("raw_stage_checks", {}),
                "rejected_stage_observations": payload.get(
                    "rejected_stage_observations", {}
                ),
            }
        )
        payload["evaluation"] = evaluation
        payload.update(
            {
                "success": bool(self.success),
                "terminal": bool(self.terminal),
                "stop_execution": bool(self.stop_execution),
                "failure_code": self.failure_code,
                "stages": stages,
                "stage_checks": qualified,
                "stage_observed": observed,
                "stage_labels": labels,
                "stage_first_observed_steps": {
                    name: values["first_observed_step"]
                    for name, values in stages.items()
                },
                "stage_qualified_steps": {
                    name: values["qualified_step"] for name, values in stages.items()
                },
            }
        )
        payload.setdefault("success_checks", {})
        return payload

    def to_report_stages(self) -> dict[str, Any]:
        stages = {}
        for name in REPORT_STAGE_ORDER:
            source = name if name in self.stages else _EXPERT_REPORT_STAGE_SOURCES[name]
            internal = self.stages[source]
            passed = (
                bool(self.success) if name == "success" else bool(internal.qualified)
            )
            stages[name] = {
                "applicable": True,
                "passed": passed,
                "first_pass_step": internal.qualified_step if passed else None,
            }
        return {
            "schema_version": "membench.report_stages/v1",
            "task_id": WR08_REPORT_PROFILE.task_id,
            "profile_id": REPORT_STAGE_PROFILE_ID,
            "order": list(REPORT_STAGE_ORDER),
            "stages": stages,
        }


@dataclass
class _MutableStage:
    observed: bool = False
    qualified: bool = False
    first_observed_step: int | None = None
    qualified_step: int | None = None


@dataclass
class _ItemProgress:
    picked_step: int | None = None
    placed_step: int | None = None
    matched_step: int | None = None
    placement_run: int = 0
    matching_run: int = 0
    wrong_target_run: int = 0
    wrong_target_slot: str | None = None


class WR08Evaluator:
    """Track each carton independently and freeze the final policy verdict."""

    def __init__(
        self,
        *,
        transport_order: Sequence[str] | None = None,
        source_slot_by_item: Mapping[str, str] | None = None,
        source_color_by_item: Mapping[str, str] | None = None,
        target_slot_by_item: Mapping[str, str] | None = None,
        placement_dwell_steps: int = 2,
    ) -> None:
        if int(placement_dwell_steps) != placement_dwell_steps or placement_dwell_steps < 1:
            raise ValueError("WR-08 placement_dwell_steps must be a positive integer")
        self.placement_dwell_steps = int(placement_dwell_steps)
        self.transport_order = _order(transport_order)
        self.source_slot_by_item = _item_map(
            source_slot_by_item, label="source_slot_by_item"
        )
        self.source_color_by_item = _item_map(
            source_color_by_item, label="source_color_by_item"
        )
        self.target_slot_by_item = _item_map(
            target_slot_by_item, label="target_slot_by_item"
        )
        self._require_expert_protocol = False
        self._stage_order = STAGE_ORDER
        self._stages = {name: _MutableStage() for name in self._stage_order}
        self._rejected: dict[str, list[dict[str, Any]]] = {
            name: [] for name in STAGE_ORDER
        }
        self._previous_raw = {name: False for name in STAGE_ORDER}
        self._last_step_id: int | None = None
        self._terminal_failure_code: str | None = None
        self._cached_result: WR08EvalResult | None = None
        self._finished = False
        self._items = {item: _ItemProgress() for item in ITEM_IDS}
        self._previous_picked = {item: False for item in ITEM_IDS}
        self._observed_pick_order: list[str] = []
        self._failure_item: str | None = None
        self._configuration_bound = bool(
            self.source_slot_by_item
            or self.source_color_by_item
            or self.target_slot_by_item
        )

    def reset(self, snapshot: WR08EvaluationSnapshot) -> WR08EvalResult:
        self._validate_configuration(snapshot)
        self._require_expert_protocol = snapshot.require_expert_protocol
        self._stage_order = EXPERT_STAGE_ORDER if self._require_expert_protocol else STAGE_ORDER
        self._stages = {name: _MutableStage() for name in self._stage_order}
        self._rejected = {name: [] for name in self._stage_order}
        self._terminal_failure_code = (
            "protocol_violation"
            if self._require_expert_protocol and snapshot.protocol_violation
            else None
        )
        self._finished = False
        self._items = {item: _ItemProgress() for item in ITEM_IDS}
        self._previous_picked = dict(snapshot.item_picked_by_item)
        self._observed_pick_order = []
        self._failure_item = None
        raw = snapshot.raw_stage_values()
        for name, value in raw.items():
            if value:
                self._mark_observed(name, snapshot.step_id)
                self._reject(name, snapshot.step_id, "present_at_reset")
        self._previous_raw = raw
        self._last_step_id = snapshot.step_id
        self._cached_result = self._build_result(snapshot)
        return self._cached_result

    def observe(self, snapshot: WR08EvaluationSnapshot) -> WR08EvalResult:
        self._validate_configuration(snapshot)
        if self._last_step_id is None:
            return self.reset(snapshot)
        if snapshot.require_expert_protocol != self._require_expert_protocol:
            raise ValueError("WR-08 evaluation mode changed without reset")
        if snapshot.step_id == self._last_step_id:
            if self._cached_result is None:
                raise RuntimeError("WR-08 evaluator has no cached result")
            return self._cached_result
        if snapshot.step_id < self._last_step_id:
            raise ValueError(
                f"WR-08 evaluation step moved backwards: {snapshot.step_id} < {self._last_step_id}"
            )
        if self._finished or self._terminal_failure_code is not None:
            return self._cached_result or self._build_result(snapshot)
        if not self._require_expert_protocol:
            return self._observe_policy(snapshot)
        return self._observe_expert(snapshot)

    def _observe_expert(self, snapshot: WR08EvaluationSnapshot) -> WR08EvalResult:
        """Retain recording's ordered pick/color-memory/match event contract."""
        raw = snapshot.raw_stage_values()
        qualified_now = {name: self._stages[name].qualified for name in self._stage_order}
        for name, value in raw.items():
            if value:
                self._mark_observed(name, snapshot.step_id)
        if snapshot.protocol_violation:
            self._terminal_failure_code = "protocol_violation"
            self._previous_raw = raw
            self._last_step_id = snapshot.step_id
            self._cached_result = self._build_result(snapshot)
            self._finished = bool(self._cached_result.stop_execution)
            return self._cached_result
        for name in self._stage_order:
            if (
                not raw[name]
                or self._previous_raw[name]
                or self._stages[name].qualified
            ):
                continue
            prerequisite = self._prerequisite(name)
            if prerequisite is not None and not qualified_now[prerequisite]:
                self._reject(
                    name, snapshot.step_id, f"prerequisite_not_qualified:{prerequisite}"
                )
                continue
            reason = self._stage_validation_failure(name, snapshot)
            if reason is not None:
                self._reject(name, snapshot.step_id, reason)
                if reason in {"wrong_target_placement", "protocol_violation"}:
                    self._terminal_failure_code = self._terminal_failure_code or reason
                continue
            self._qualify(name, snapshot.step_id)
            qualified_now[name] = True
        self._previous_raw = raw
        self._last_step_id = snapshot.step_id
        self._cached_result = self._build_result(snapshot)
        self._finished = bool(self._cached_result.stop_execution)
        return self._cached_result

    def _observe_policy(self, snapshot: WR08EvaluationSnapshot) -> WR08EvalResult:
        continuous = snapshot.step_id == self._last_step_id + 1
        for item in self.transport_order:
            progress = self._items[item]
            picked_before = progress.picked_step is not None
            placed = snapshot.placed(item)
            # The policy must show a real grasp-and-lift observation followed
            # by a later release. Reset poses and simultaneous bits are not history.
            if (
                not picked_before
                and snapshot.item_picked_by_item[item]
                and not self._previous_picked[item]
                and not placed
            ):
                progress.picked_step = snapshot.step_id
                self._observed_pick_order.append(item)

            if not continuous:
                progress.placement_run = progress.matching_run = progress.wrong_target_run = 0
            progress.placement_run = progress.placement_run + 1 if picked_before and placed else 0
            progress.matching_run = (
                progress.matching_run + 1 if picked_before and snapshot.matched(item) else 0
            )
            dwell = self.placement_dwell_steps
            if progress.placement_run >= dwell and progress.placed_step is None:
                progress.placed_step = snapshot.step_id
            if progress.matching_run >= dwell and progress.matched_step is None:
                progress.matched_step = snapshot.step_id

            attempted = snapshot.target_slot_attempted_by_item.get(item)
            expected = self.target_slot_by_item.get(item)
            wrong_slot = (
                attempted if (
                    picked_before and placed and attempted in TARGET_SLOTS
                    and expected is not None and attempted != expected
                ) else None
            )
            progress.wrong_target_run = (
                0 if wrong_slot is None else (
                    progress.wrong_target_run + 1
                    if wrong_slot == progress.wrong_target_slot else 1
                )
            )
            progress.wrong_target_slot = wrong_slot
            # Keep watching already matched cartons while the other carton
            # is being moved. Any confirmed wrong destination ends the task.
            if progress.wrong_target_run >= dwell and self._terminal_failure_code is None:
                self._terminal_failure_code = "wrong_target_placement"
                self._failure_item = item

        values = self._policy_stage_values()
        for name in self._stage_order:
            if not values[name]:
                continue
            self._mark_observed(name, snapshot.step_id)
            prerequisite = self._prerequisite(name)
            if prerequisite is None or self._stages[prerequisite].qualified:
                self._qualify(name, snapshot.step_id)
        self._previous_raw = values
        self._previous_picked = dict(snapshot.item_picked_by_item)
        self._last_step_id = snapshot.step_id
        self._cached_result = self._build_result(snapshot)
        self._finished = self._cached_result.stop_execution
        return self._cached_result

    def _policy_stage_values(self) -> dict[str, bool]:
        picked = sum(progress.picked_step is not None for progress in self._items.values())
        placed = sum(progress.placed_step is not None for progress in self._items.values())
        matched = sum(progress.matched_step is not None for progress in self._items.values())
        return {
            "one_item_picked": picked >= 1,
            "one_item_placed": placed >= 1,
            "one_item_matched": matched >= 1,
            "two_items_picked": picked == len(ITEM_IDS),
            "two_items_placed": placed == len(ITEM_IDS),
            "two_items_matched": matched == len(ITEM_IDS),
            "success": bool(
                matched == len(ITEM_IDS)
                and self._terminal_failure_code is None
                and all(
                    progress.matching_run >= self.placement_dwell_steps
                    for progress in self._items.values()
                )
            ),
        }

    def _prerequisite(self, name: str) -> str | None:
        index = self._stage_order.index(name)
        return None if index == 0 else self._stage_order[index - 1]

    def result(self) -> WR08EvalResult:
        if self._cached_result is None:
            raise RuntimeError("WR-08 evaluator has not been reset")
        return self._cached_result

    def _validate_configuration(self, snapshot: WR08EvaluationSnapshot) -> None:
        if snapshot.transport_order != self.transport_order:
            raise ValueError(
                "WR-08 evaluation snapshot configuration changed within an episode"
            )
        if not self._configuration_bound:
            self.source_slot_by_item = dict(snapshot.source_slot_by_item)
            self.source_color_by_item = dict(snapshot.source_color_by_item)
            self.target_slot_by_item = dict(snapshot.target_slot_by_item)
            self._configuration_bound = True
            return
        if (
            dict(snapshot.source_slot_by_item) != self.source_slot_by_item
            or dict(snapshot.source_color_by_item) != self.source_color_by_item
            or dict(snapshot.target_slot_by_item) != self.target_slot_by_item
        ):
            raise ValueError(
                "WR-08 evaluation snapshot configuration changed within an episode"
            )

    def _stage_validation_failure(
        self, name: str, snapshot: WR08EvaluationSnapshot
    ) -> str | None:
        item = (
            self.transport_order[0]
            if name.startswith("first_")
            else self.transport_order[1] if name.startswith("second_") else None
        )
        if item is None:
            return None if snapshot.domain_success else "domain_success_not_verified"
        if name.endswith("color_latched") and not snapshot.item_picked_by_item[item]:
            return "item_not_picked"
        if name.endswith("matched"):
            if not snapshot.memory_latched_by_item[item]:
                return "source_color_memory_not_latched"
            attempted = snapshot.target_slot_attempted_by_item.get(item)
            expected = snapshot.target_slot_by_item.get(
                item, self.target_slot_by_item.get(item)
            )
            if attempted is None:
                return "target_not_attempted"
            if expected is not None and attempted != expected:
                return "wrong_target_placement"
            if not snapshot.released_by_item[item]:
                return "item_not_released"
            if not snapshot.item_matched_by_item[item]:
                return "target_match_not_verified"
        return None

    def _mark_observed(self, name: str, step_id: int) -> None:
        stage = self._stages[name]
        if not stage.observed:
            stage.observed, stage.first_observed_step = True, int(step_id)

    def _qualify(self, name: str, step_id: int) -> None:
        stage = self._stages[name]
        if not stage.qualified:
            stage.qualified, stage.qualified_step = True, int(step_id)

    def _reject(self, name: str, step_id: int, reason: str) -> None:
        entry = {"step": int(step_id), "reason": str(reason)}
        if entry not in self._rejected[name]:
            self._rejected[name].append(entry)

    def _stage_result(self, name: str) -> StageEvalResult:
        stage = self._stages[name]
        if stage.qualified:
            invalidated, reason = False, None
        elif self._terminal_failure_code is not None:
            invalidated, reason = True, self._terminal_failure_code
        elif self._rejected[name]:
            invalidated, reason = True, str(self._rejected[name][-1]["reason"])
        else:
            prerequisite = self._prerequisite(name)
            invalidated = False
            reason = (
                f"waiting_for_prerequisite:{prerequisite}"
                if prerequisite and not self._stages[prerequisite].qualified
                else "not_observed"
            )
        return StageEvalResult(
            True,
            stage.observed,
            stage.qualified,
            stage.first_observed_step,
            stage.qualified_step,
            invalidated,
            reason,
        )

    def _build_result(self, snapshot: WR08EvaluationSnapshot) -> WR08EvalResult:
        stages = {name: self._stage_result(name) for name in self._stage_order}
        raw = (
            snapshot.raw_stage_values()
            if self._require_expert_protocol else self._policy_stage_values()
        )
        success = bool(
            stages["success"].qualified
            and raw["success"]
            and self._terminal_failure_code is None
        )
        details = dict(snapshot.details)
        details.setdefault("step_id", int(snapshot.step_id))
        details.setdefault("success_checks", {})
        details["raw_stage_checks"] = raw
        details["rejected_stage_observations"] = {
            name: list(values) for name, values in self._rejected.items() if values
        }
        evaluation = dict(details.get("evaluation") or {})
        evaluation.update(
            {
                "schema_version": 1,
                "step_id": int(snapshot.step_id),
                "stage_order": list(self._stage_order),
                "profile_id": REPORT_STAGE_PROFILE_ID,
                "transport_order": list(self.transport_order),
                "stage_bindings": {
                    "source_slot_by_item": dict(snapshot.source_slot_by_item),
                    "source_color_by_item": dict(snapshot.source_color_by_item),
                    "target_slot_by_item": dict(snapshot.target_slot_by_item),
                },
                "mode": (
                    "expert_recording"
                    if snapshot.require_expert_protocol
                    else "policy_physical"
                ),
                "qualification_policy": (
                    "rising_edge_after_prerequisite" if self._require_expert_protocol
                    else "per_item_pick_then_contact_release"
                ),
                "terminal_step": snapshot.step_id if success or self._terminal_failure_code else None,
            }
        )
        if self._require_expert_protocol:
            evaluation["stage_bindings"].update({
                "first_item": self.transport_order[0],
                "second_item": self.transport_order[1],
            })
        else:
            evaluation.update({
                "transport_order_required": False,
                "observed_pick_order": list(self._observed_pick_order),
                "placement_dwell_steps": self.placement_dwell_steps,
                "failure_item": self._failure_item,
                "items": {
                    item: {
                        "picked_step": progress.picked_step,
                        "placed_step": progress.placed_step,
                        "matched_step": progress.matched_step,
                        "placement_run": progress.placement_run,
                        "matching_plate_run": progress.matching_run,
                        "wrong_target_slot": progress.wrong_target_slot,
                        "wrong_target_run": progress.wrong_target_run,
                    }
                    for item, progress in self._items.items()
                },
            })
            details["success_checks"] = {
                "both_items_picked": all(p.picked_step is not None for p in self._items.values()),
                "both_items_placed": all(p.placed_step is not None for p in self._items.values()),
                "both_items_on_matching_target_plates": all(
                    snapshot.matched(item) for item in ITEM_IDS
                ),
                "both_matching_placements_confirmed": all(
                    p.matching_run >= self.placement_dwell_steps for p in self._items.values()
                ),
                "no_wrong_target_placement": self._terminal_failure_code is None,
            }
        details["evaluation"] = evaluation
        terminal = bool(success or self._terminal_failure_code)
        return WR08EvalResult(
            success,
            terminal,
            self._terminal_failure_code,
            stages,
            details,
            stop_execution=bool(not snapshot.require_expert_protocol and terminal),
        )


def evaluate_initialized_runtime(runtime: Any) -> WR08EvalResult:
    initialized = getattr(runtime, "initialized", None) or runtime
    metadata = getattr(initialized, "metadata", {}) or {}
    env = (
        getattr(initialized, "env", None)
        or getattr(runtime, "env", None)
        or initialized
    )
    evaluator = metadata.get(TASK_EVALUATOR_METADATA_KEY) or getattr(
        env, "_wr08_evaluator", None
    )
    if not isinstance(evaluator, WR08Evaluator):
        raise RuntimeError("WR-08 runtime has no initialized task evaluator")
    snapshot_builder = getattr(env, "evaluation_snapshot", None)
    if not callable(snapshot_builder):
        raise AttributeError("WR-08 environment has no evaluation_snapshot()")
    return evaluator.observe(snapshot_builder())


def success_for_runtime(runtime: Any, info: Mapping[str, Any] | None = None) -> bool:
    del info
    return bool(evaluate_initialized_runtime(runtime).success)


def diagnostics_for_runtime(runtime: Any) -> Mapping[str, Any]:
    return evaluate_initialized_runtime(runtime).to_diagnostics()


def report_stages_for_runtime(runtime: Any) -> Mapping[str, Any]:
    return evaluate_initialized_runtime(runtime).to_report_stages()


__all__ = [
    "EXPERT_STAGE_ORDER",
    "REPORT_STAGE_ORDER",
    "REPORT_STAGE_PROFILE_ID",
    "STAGE_LABELS",
    "STAGE_ORDER",
    "STAGE_PREREQUISITES",
    "TASK_EVALUATOR_METADATA_KEY",
    "StageEvalResult",
    "WR08EvalResult",
    "WR08EvaluationSnapshot",
    "WR08Evaluator",
    "diagnostics_for_runtime",
    "evaluate_initialized_runtime",
    "report_stages_for_runtime",
    "success_for_runtime",
]
