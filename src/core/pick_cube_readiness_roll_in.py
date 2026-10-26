"""Fixed prefix behavior selection for one readiness TRAIN rollout.

Inputs are a frozen prefix length, step and already computed student/teacher
commands. Output records both commands and the selected behavior. No geometry,
teacher query, time, learner, mutable progress, execution or IO occurs here.
"""

from dataclasses import dataclass
import struct
from typing import cast

from src.core.continuous_demonstration import _vector


DEFINITION = "pick_cube_readiness_teacher_prefix_then_student_v1"
COMPLETION_DEFINITION = "pick_cube_readiness_student_prefix_then_teacher_v1"


@dataclass(frozen=True)
class ReadinessRollInPlan:
    teacher_steps: int

    def __post_init__(self) -> None:
        if type(self.teacher_steps) is not int or not 1 <= self.teacher_steps <= 49:
            raise ValueError("readiness roll-in teacher_steps must be a native integer in1..49")


@dataclass(frozen=True)
class ReadinessTeacherCompletionPlan:
    student_steps: int

    def __post_init__(self) -> None:
        if type(self.student_steps) is not int or not 1 <= self.student_steps <= 49:
            raise ValueError("readiness completion student_steps must be a native integer in1..49")


ReadinessBehaviorPlan = ReadinessRollInPlan | ReadinessTeacherCompletionPlan


def validate_readiness_roll_in_plan(value: ReadinessBehaviorPlan) -> ReadinessBehaviorPlan:
    if type(value) is ReadinessRollInPlan:
        if set(vars(value)) != {"teacher_steps"}:
            raise ValueError("readiness roll-in plan fields changed")
        return ReadinessRollInPlan(value.teacher_steps)
    if type(value) is ReadinessTeacherCompletionPlan:
        if set(vars(value)) != {"student_steps"}:
            raise ValueError("readiness completion plan fields changed")
        return ReadinessTeacherCompletionPlan(value.student_steps)
    raise ValueError("expected an exact declared readiness behavior plan")


def readiness_roll_in_metadata(plan: ReadinessBehaviorPlan) -> dict[str, str | int]:
    """Own fixed plan metadata without changing either plan's dataclass schema."""
    owned = validate_readiness_roll_in_plan(plan)
    if type(owned) is ReadinessRollInPlan:
        return dict(definition=DEFINITION, teacher_steps=owned.teacher_steps)
    return dict(
        definition=COMPLETION_DEFINITION,
        student_steps=cast(ReadinessTeacherCompletionPlan, owned).student_steps,
    )


@dataclass(frozen=True)
class ReadinessBehaviorDecision:
    step: int
    behavior_source: str
    student_action: tuple[float, ...]
    teacher_target: tuple[float, ...]
    executed_action: tuple[float, ...]


def readiness_roll_in_decision(
    plan: ReadinessBehaviorPlan,
    step: int,
    student_action: tuple[float, ...],
    teacher_target: tuple[float, ...],
) -> ReadinessBehaviorDecision:
    """Choose the fixed prefix without an outcome-dependent switch or fallback."""
    owned = validate_readiness_roll_in_plan(plan)
    if type(step) is not int or not 0 <= step <= 49:
        raise ValueError("readiness roll-in step must be a native integer in0..49")
    student = _vector(student_action, "roll-in student action", 4)
    teacher = _vector(teacher_target, "roll-in teacher target", 4)
    if any(
        len(command) != 4 or any(not -1.0 <= value <= 1.0 for value in command)
        for command in (student, teacher)
    ):
        raise ValueError("readiness roll-in commands require four normalized bounded axes")
    source = (
        ("teacher" if step < owned.teacher_steps else "student")
        if type(owned) is ReadinessRollInPlan
        else (
            "student"
            if step < cast(ReadinessTeacherCompletionPlan, owned).student_steps
            else "teacher"
        )
    )
    selected = teacher if source == "teacher" else student
    # The existing port sends/stores float32. Preserve the full original
    # proposals and targets, and record the exact native command separately.
    executed = tuple(struct.unpack("!f", struct.pack("!f", value))[0] for value in selected)
    return ReadinessBehaviorDecision(step, source, student, teacher, executed)
