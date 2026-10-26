"""Versioned slow-teacher target data and the affine binary BCE mixture.

Records own bounded probabilities and declared schema/revision metadata. They
provide no model, data-role authority, training, replay selection or provenance.
The mixture is an explicit proposed ablation, not a DER/DER++ reproduction.
"""

from dataclasses import dataclass, fields, replace
from math import isfinite
from typing import Any

import numpy as np

from src.core.coupled_energy import Array

SLOW_TEACHER_MIXTURE_ID = "numpy_slow_teacher_mixed_binary_target_v1"


def _record(value: Any, kind: Any) -> None:
    if type(value) is not kind or vars(value).keys() != {item.name for item in fields(kind)}:
        raise ValueError(f"slow teacher requires an exact complete {kind.__name__}")


def _unit_number(value: float, name: str) -> None:
    if type(value) not in {int, float} or not 0 <= value <= 1 or not isfinite(value):
        raise ValueError(f"{name} must be a finite native number in [0,1]")


def _identifier(value: str, name: str) -> None:
    if type(value) is not str or not value or len(value) > 128:
        raise ValueError(f"{name} must be a nonempty bounded native string")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError as error:
        raise ValueError(f"{name} must be valid UTF-8") from error
    if size > 128:
        raise ValueError(f"{name} exceeds 128 UTF-8 bytes")


@dataclass(frozen=True)
class SlowTeacherPolicy:
    teacher_id: str
    feature_schema: str
    output_schema: str = "binary_probability_v1"
    mixture_weight: float = 0.25
    ema_decay: float = 0.9

    def __post_init__(self) -> None:
        _identifier(self.teacher_id, "teacher_id")
        _identifier(self.feature_schema, "feature_schema")
        if type(self.output_schema) is not str or self.output_schema != "binary_probability_v1":
            raise ValueError("output_schema must be binary_probability_v1")
        _unit_number(self.mixture_weight, "mixture_weight")
        _unit_number(self.ema_decay, "ema_decay")


def copy_slow_teacher_policy(policy: SlowTeacherPolicy) -> SlowTeacherPolicy:
    _record(policy, SlowTeacherPolicy)
    return replace(policy)


def _probabilities(values: Array, name: str) -> None:
    if (
        type(values) is not np.ndarray
        or values.dtype != np.float64
        or values.ndim != 2
        or not 1 <= values.shape[0] <= 128
        or values.shape[1] != 1
    ):
        raise ValueError(f"{name} must be an exact float64 Bx1 array with 1..128 rows")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError(f"{name} must contain finite values in [0,1]")


@dataclass(frozen=True)
class TeacherTargets:
    format_version: int
    policy: SlowTeacherPolicy
    teacher_revision: int
    student_updates: int
    sample_ids: tuple[str, ...]
    probabilities: Array

    def __post_init__(self) -> None:
        if type(self.format_version) is not int or self.format_version != 1:
            raise ValueError("teacher target format_version must be native integer 1")
        policy = copy_slow_teacher_policy(self.policy)
        for name in ("teacher_revision", "student_updates"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a native nonnegative integer")
        if self.teacher_revision > self.student_updates:
            raise ValueError("teacher revision exceeds student updates")
        _probabilities(self.probabilities, "teacher probabilities")
        if (
            type(self.sample_ids) is not tuple
            or len(self.sample_ids) != len(self.probabilities)
            or any(
                type(value) is not str
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
                for value in self.sample_ids
            )
        ):
            raise ValueError("teacher sample IDs must be aligned canonical replay content IDs")
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "probabilities", self.probabilities.copy())


def copy_teacher_targets(targets: TeacherTargets) -> TeacherTargets:
    _record(targets, TeacherTargets)
    return replace(targets)


def blend_teacher_targets(
    targets: Array, teacher_targets: TeacherTargets | None, policy: SlowTeacherPolicy
) -> Array:
    """Return (1-lambda)y+lambda*q, whose BCE gradient is the same affine mixture."""
    policy = copy_slow_teacher_policy(policy)
    _probabilities(targets, "native targets")
    if policy.mixture_weight == 0:
        if teacher_targets is not None:
            raise ValueError("zero mixture requires teacher_targets=None")
        return targets.copy()
    if teacher_targets is None:
        raise ValueError("positive mixture requires teacher targets")
    packet = copy_teacher_targets(teacher_targets)
    if packet.policy != policy or packet.probabilities.shape != targets.shape:
        raise ValueError("teacher target policy or shape differs")
    mixed = (1 - policy.mixture_weight) * targets + policy.mixture_weight * packet.probabilities
    _probabilities(mixed, "mixed targets")
    return mixed
