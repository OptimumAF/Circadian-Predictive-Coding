"""Validate NumPy runner data roles and ordinary baseline snapshots.

App runners use these pure checks before accepting a trusted-file payload.
This module does not read files, score final-test data, or restore models.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Protocol, Sequence

import numpy as np

from src.core.backprop_mlp import BackpropMLP
from src.core.predictive_coding import PredictiveCodingNetwork


class LabeledRole(Protocol):
    @property
    def input(self) -> np.ndarray[Any, Any]: ...

    @property
    def target(self) -> np.ndarray[Any, Any]: ...


def digest_labeled_roles(purpose: str, roles: Sequence[tuple[str, LabeledRole | None]]) -> str:
    """Hash exact role arrays; the caller controls held-out access timing."""
    digest = sha256(purpose.encode("ascii"))
    for name, role in roles:
        digest.update(name.encode("ascii"))
        if role is None:
            digest.update(b"absent")
            continue
        for values in (role.input, role.target):
            canonical = np.ascontiguousarray(values, dtype="<f8")
            digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
            digest.update(canonical.tobytes())
    return digest.hexdigest()


def validate_numpy_baseline_model(
    model: BackpropMLP | PredictiveCodingNetwork,
    hidden_dims: tuple[int, ...],
    expected_steps: int,
) -> None:
    """Reject malformed baseline arrays and report steps before live restore."""
    try:
        if (
            model.input_dim != 2
            or model.hidden_dims != hidden_dims
            or type(model._traffic_steps) is not int
            or model._traffic_steps != expected_steps
            or len(model._hidden_weights) != len(hidden_dims)
            or len(model._hidden_biases) != len(hidden_dims)
            or len(model._traffic_sums) != len(hidden_dims)
            or model.weight_input_hidden is not model._hidden_weights[0]
            or model.bias_hidden is not model._hidden_biases[0]
        ):
            raise ValueError("incompatible NumPy checkpoint baseline state")
        widths = (2, *hidden_dims)
        arrays: list[tuple[np.ndarray, tuple[int, ...]]] = []
        for index, width in enumerate(hidden_dims):
            arrays.extend(
                (
                    (model._hidden_weights[index], (widths[index], width)),
                    (model._hidden_biases[index], (1, width)),
                    (model._traffic_sums[index], (width,)),
                )
            )
        arrays.extend(
            (
                (model.weight_hidden_output, (hidden_dims[-1], 1)),
                (model.bias_output, (1, 1)),
            )
        )
        if any(
            not isinstance(values, np.ndarray)
            or values.dtype != np.float64
            or values.shape != shape
            or not np.all(np.isfinite(values))
            for values, shape in arrays
        ):
            raise ValueError("incompatible NumPy checkpoint baseline arrays")
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError("incompatible NumPy checkpoint baseline arrays") from exc
