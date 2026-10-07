"""Validated two-task accuracy arithmetic for prospective continual studies.

Inputs are accuracy on A after A, and accuracy on A/B after B. Outputs are
equal-task final mean, signed forgetting, and an optional retention ratio.
This pure module neither reads data nor selects an experiment setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


TWO_TASK_METRIC_CONTRACT_ID = "phase6_two_task_accuracy_v1"


@dataclass(frozen=True)
class TwoTaskAccuracy:
    a_after_a: float
    a_after_b: float
    b_after_b: float

    def __post_init__(self) -> None:
        for name in ("a_after_a", "a_after_b", "b_after_b"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a finite accuracy in [0, 1]")

    @property
    def final_mean_task_accuracy(self) -> float:
        return (self.a_after_b + self.b_after_b) / 2.0

    @property
    def signed_forgetting_a(self) -> float:
        return self.a_after_a - self.a_after_b

    @property
    def retention_ratio_a(self) -> float | None:
        return self.a_after_b / self.a_after_a if self.a_after_a > 0.0 else None
