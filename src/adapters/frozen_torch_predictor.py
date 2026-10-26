"""Prediction-only copies of logged fixed-feature Torch parameter checkpoints.

Inputs are four bounded, exact float32 tensor value lists. The existing native
head supplies predictions. Fresh valid optimizer/adaptive/RNG state is retained
only to satisfy its native validity checks, not to reconstruct training history.
No training method, parameter selection, IO or publication authority is exposed.
"""

from math import isfinite
from typing import Any

import numpy as np

from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.core.coupled_energy import Array
from src.shared.torch_runtime import require_torch

SHAPES = dict(
    weight_feature_hidden=(2, 3),
    bias_hidden=(1, 3),
    weight_hidden_output=(3, 2),
    bias_output=(1, 2),
)


def validate_parameters(parameters: Any) -> dict[str, tuple[tuple[float, ...], ...]]:
    """Validate the complete small JSON envelope before native construction."""
    if type(parameters) is not dict or set(parameters) != set(SHAPES):
        raise ValueError("frozen prediction requires exactly the four native parameter arrays")
    for name, (rows, columns) in SHAPES.items():
        values = parameters[name]
        if type(values) is not list or len(values) != rows:
            raise ValueError(f"frozen {name} must have shape {(rows, columns)}")
        for row in values:
            if type(row) is not list or len(row) != columns:
                raise ValueError(f"frozen {name} must have shape {(rows, columns)}")
            for value in row:
                if (
                    type(value) not in (int, float)
                    or abs(value) > float(np.finfo(np.float32).max)
                    or not isfinite(value)
                    or float(np.float32(value)) != value
                ):
                    raise ValueError("logged parameters must be finite exact native float32 values")
    return {
        name: tuple(tuple(float(value) for value in row) for row in parameters[name])
        for name in SHAPES
    }


class FrozenTorchPredictor:
    """Own a detached prediction copy; refuse changed state or reuse after a fault."""

    def __init__(self, method: str, parameters: Any, *, seed: int = 17) -> None:
        policy = TorchFeaturePolicy(method, 2, 3, seed)
        values = validate_parameters(parameters)
        self._learner = TorchFeatureLearner(policy)
        state = self._learner.snapshot_state()
        torch = require_torch()
        for name in SHAPES:
            state.head_state[name] = torch.tensor(values[name], dtype=state.head_state[name].dtype)
        self._learner.restore_state(state)
        self._initial = self._learner.snapshot_state()
        self._stopped = False

    def assert_unchanged(self) -> None:
        if not _same_state(self._initial, self._learner.snapshot_state()):
            self._stopped = True
            raise ValueError("frozen predictor complete local state changed")

    def predict(self, features: Array) -> Array:
        if self._stopped:
            raise ValueError("frozen predictor stopped after a previous fault")
        self.assert_unchanged()
        try:
            predictions = self._learner.predict(features)
            self.assert_unchanged()
            return predictions
        except Exception:
            self._stopped = True
            raise
