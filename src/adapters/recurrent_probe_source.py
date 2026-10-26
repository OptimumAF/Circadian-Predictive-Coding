"""Restore an owned recurrent checkpoint into the existing array-native port.

Inputs are a complete native snapshot and the caller's isolation guard. The
output is an independently owned learner suitable for disposable fresh-task
forks. This bridge does not train, select probes, change record admission, or
extract an owner from a benchmark learner.
"""

from copy import deepcopy
from dataclasses import fields
from typing import Callable

from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.adapters.torch_recurrent_learner import (
    TorchRecurrentLearner,
    TorchRecurrentSnapshot,
)


def recurrent_probe_source(
    state: TorchRecurrentSnapshot, *, check_boundary: Callable[[], None]
) -> TorchRecurrentLearner:
    """Copy and restore complete native state without widening typed training roles."""
    if type(state) is not TorchRecurrentSnapshot or set(vars(state)) != {
        field.name for field in fields(TorchRecurrentSnapshot)
    }:
        raise ValueError("probe source requires an exact complete TorchRecurrentSnapshot")
    if not callable(check_boundary):
        raise ValueError("probe source requires a callable boundary check")
    expected = deepcopy(state)
    owned = deepcopy(expected)
    bindings = {
        name: getattr(state, name)
        for name in ("policy", "parameters", "optimizer_state", "gradients")
    }

    def verify() -> None:
        if (
            any(getattr(state, name, None) is not value for name, value in bindings.items())
            or not _same_state(state, expected)
            or not _same_state(owned, expected)
        ):
            raise ValueError("recurrent probe source snapshot changed across a boundary")

    def boundary() -> None:
        verify()
        check_boundary()
        verify()

    boundary()
    learner = TorchRecurrentLearner(deepcopy(owned.policy))
    boundary()
    learner.restore_state(deepcopy(owned))
    boundary()
    restored = learner.snapshot_state()
    boundary()
    if not _same_state(restored, expected):
        raise ValueError("recurrent probe source restore did not preserve the complete snapshot")
    # Why: a boundary callback follows the first readback. A final owned readback
    # closes that window before handing the new learner to its caller.
    verify()
    final = learner.snapshot_state()
    verify()
    if not _same_state(final, expected):
        raise ValueError("recurrent probe source changed after its restored-state readback")
    return learner
