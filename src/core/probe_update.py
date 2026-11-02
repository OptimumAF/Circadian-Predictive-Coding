"""Owned scalar receipts for auxiliary updates of an already private probe.

These receipts account for callback applications, not native SGD, arrival,
promotion or replay authority. They contain no examples or learner state.
"""

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ProbeUpdateReceipt:
    definition: str
    work: dict[str, int]
    applied: bool
    loss: float | None = None

    def __post_init__(self):
        if type(self.definition) is not str or not 1 <= len(self.definition) <= 128:
            raise ValueError("auxiliary probe receipt requires a bounded definition")
        if (
            type(self.work) is not dict
            or not 1 <= len(self.work) <= 64
            or any(type(k) is not str or not 1 <= len(k) <= 80 for k in self.work)
            or any(type(v) is not int or not 0 <= v < 2**63 for v in self.work.values())
            or self.work.get("native_sgd_updates") != 0
            or type(self.applied) is not bool
            or self.loss is not None
            and (type(self.loss) is not float or not isfinite(self.loss) or self.loss < 0)
        ):
            raise ValueError("invalid auxiliary probe receipt or native SGD credit")
        object.__setattr__(self, "work", deepcopy(self.work))


class ProbeUpdateFailed(RuntimeError):
    """Keep paid partial work, including an application preceding a late failure."""

    def __init__(self, reason: str, receipt: ProbeUpdateReceipt):
        if type(receipt) is not ProbeUpdateReceipt:
            raise ValueError("auxiliary failure requires an exact receipt")
        super().__init__(reason)
        self.receipt = ProbeUpdateReceipt(
            receipt.definition, receipt.work, receipt.applied, receipt.loss
        )
