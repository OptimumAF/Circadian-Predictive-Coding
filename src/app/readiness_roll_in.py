"""Bounded mixed-behavior control receipts; no learner, simulator, clock or IO.

Own at most fifty attempted controls. Public report copies cannot redefine the
original prefix or preceding paid receipts. Actual raw actions remain separate
from queried teacher targets and proposed student commands.
"""

from dataclasses import asdict
import json
from typing import cast

from src.core.continuous_demonstration import _vector
from src.core.pick_cube_readiness_roll_in import (
    DEFINITION as DEFINITION,
    ReadinessBehaviorDecision,
    ReadinessRollInPlan,
    ReadinessTeacherCompletionPlan,
    ReadinessBehaviorPlan,
    readiness_roll_in_metadata,
    readiness_roll_in_decision,
    validate_readiness_roll_in_plan,
)


def _native_payload(value):
    """Reject callback-capable report objects before serializing receipts."""
    if type(value) in (str, int, float, bool, type(None)):
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("behavior report keys must be native strings")
            _native_payload(item)
    elif type(value) in (tuple, list):
        for item in value:
            _native_payload(item)
    else:
        raise ValueError("behavior reports require exact native payload types")


def _records_seal(records):
    if type(records) is not tuple or len(records) > 50:
        raise ValueError("readiness control receipts require a bounded exact tuple")
    for record in records:
        if (
            type(record) is not tuple
            or len(record) != 7
            or type(record[0]) is not int
            or record[1] not in ("teacher", "student")
            or any(
                type(record[index]) is not tuple or len(record[index]) != 4 for index in (2, 3, 4)
            )
            or type(record[6]) is not bool
        ):
            raise ValueError("readiness control receipt shape changed")
        _vector(record[5], "recorded readiness observation", 1024)
    _native_payload(records)
    return json.dumps(records, sort_keys=True, allow_nan=False)


class ReadinessRollInLedger:
    def __init__(self, plan: ReadinessBehaviorPlan):
        self.plan = validate_readiness_roll_in_plan(plan)
        self._plan_type = type(self.plan)
        self._plan_metadata = json.dumps(readiness_roll_in_metadata(self.plan), sort_keys=True)
        if type(self.plan) is ReadinessRollInPlan:
            self._teacher_steps = self.plan.teacher_steps
        else:
            self._student_steps = cast(ReadinessTeacherCompletionPlan, self.plan).student_steps
        self._records: tuple[tuple, ...] = ()
        # Every member is an owned tuple of native immutable scalar values.
        self._accepted_records: tuple[tuple, ...] = self._records
        self._records_evidence = _records_seal(self._records)
        self._failed = False

    def _check_originals(self):
        try:
            if self._failed:
                raise ValueError("failed readiness behavior sequence cannot resume")
            plan = validate_readiness_roll_in_plan(self.plan)
            if (
                type(plan) is not self._plan_type
                or json.dumps(readiness_roll_in_metadata(plan), sort_keys=True)
                != self._plan_metadata
            ):
                raise ValueError("admitted readiness behavior plan changed")
            if type(plan) is ReadinessRollInPlan:
                valid_prefix = (
                    type(self._teacher_steps) is int and self._teacher_steps == plan.teacher_steps
                )
            else:
                valid_prefix = (
                    type(self._student_steps) is int
                    and self._student_steps
                    == cast(ReadinessTeacherCompletionPlan, plan).student_steps
                )
            if not valid_prefix or (
                _records_seal(self._records) != self._records_evidence
                or _records_seal(self._accepted_records) != self._records_evidence
            ):
                raise ValueError("original readiness behavior prefix or full receipt changed")
        except Exception:
            self._failed = True
            raise

    def report(self):
        controls = [
            dict(
                step=step,
                behavior_source=source,
                student_action=student,
                teacher_target=teacher,
                executed_action=action,
                control_returned=returned,
            )
            for step, source, student, teacher, action, _, returned in self._accepted_records
        ]
        teacher_attempts = sum(row["behavior_source"] == "teacher" for row in controls)
        teacher_returns = sum(
            row["behavior_source"] == "teacher" and row["control_returned"] for row in controls
        )
        student_returns = sum(
            row["behavior_source"] == "student" and row["control_returned"] for row in controls
        )
        metadata = json.loads(self._plan_metadata)
        result = dict(definition=metadata.pop("definition"), controller_encoding="float32")
        result.update(metadata)
        result.update(
            controls=controls,
            teacher_control_attempts=teacher_attempts,
            teacher_control_returns=teacher_returns,
            student_control_attempts=len(controls) - teacher_attempts,
            student_control_returns=student_returns,
        )
        if self._plan_type is ReadinessRollInPlan:
            result["handed_back"] = student_returns > 0
        else:
            result["teacher_takeover_returned"] = teacher_returns > 0
        return result

    def check(self, report):
        try:
            self._check_originals()
            _native_payload(report)
            if json.dumps(report, sort_keys=True, allow_nan=False) != json.dumps(
                self.report(), sort_keys=True, allow_nan=False
            ):
                raise ValueError("preceding readiness behavior receipt or counter changed")
        except Exception:
            self._failed = True
            raise

    def attempt(self, decision, observation, student_action, teacher_target, report):
        self.check(report)
        if type(decision) is not ReadinessBehaviorDecision:
            raise ValueError("expected an exact readiness behavior decision")
        _native_payload(vars(decision))
        if len(self._records) >= 50 or (self._records and not self._records[-1][-1]):
            raise ValueError("roll-in control sequence cannot admit more work")
        expected = readiness_roll_in_decision(
            self.plan, len(self._records), student_action, teacher_target
        )
        if type(decision) is not type(expected) or json.dumps(
            asdict(decision), sort_keys=True
        ) != json.dumps(asdict(expected), sort_keys=True):
            raise ValueError("roll-in decision differs from the fixed prefix policy")
        owned_observation = tuple(observation)
        _vector(owned_observation, "readiness behavior observation", 1024)
        self._check_originals()
        self._records += (
            (
                expected.step,
                expected.behavior_source,
                expected.student_action,
                expected.teacher_target,
                expected.executed_action,
                owned_observation,
                False,
            ),
        )
        self._accepted_records = self._records
        self._records_evidence = _records_seal(self._records)

    def returned(self, row, report):
        admitted = self._accepted_records
        if not admitted or admitted[-1][-1]:
            self._failed = True
            raise ValueError("roll-in control returned without an outstanding attempt")
        checked = False
        try:
            self.check(report)
            checked = True
        finally:
            # Native returned even on a failed check. Retain the original paid
            # receipt, but never bless corrupted prior records with a new seal.
            paid = admitted[:-1] + (admitted[-1][:-1] + (True,),)
            self._accepted_records = paid
            if checked:
                self._records = paid
                self._records_evidence = _records_seal(paid)
            else:
                self._failed = True
        step, _, _, _, action, observation, _ = paid[-1]
        try:
            if row.step != step or row.action != action or row.observation != observation:
                raise ValueError("returned control differs from its selected raw behavior receipt")
            self._check_originals()
        except Exception:
            self._failed = True
            raise

    def verify_episode(self, episode, labels):
        try:
            self._check_originals()
            if len(episode.transitions) != len(self._records):
                raise ValueError("mixed behavior receipts differ from the executed episode")
            for row, label, record in zip(episode.transitions, labels, self._records, strict=True):
                step, _, _, teacher, action, observation, returned = record
                if (
                    not returned
                    or row.step != step
                    or row.action != action
                    or row.observation != observation
                    or label.observation != observation
                    or label.target != teacher
                ):
                    raise ValueError("mixed behavior episode changed after control return")
            self._check_originals()
        except Exception:
            self._failed = True
            raise
