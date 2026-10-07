"""Model constructors require finite positive integer dimensions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.predictive_coding import PredictiveCodingNetwork
from src.core.resnet50_variants import CircadianPredictiveCodingHead, PredictiveCodingHead


INVALID = [float("nan"), float("inf"), float("-inf"), 1.5, True, 0, -1]
FIELDS = {
    "backprop": ("input_dim", "hidden_dim"),
    "predictive": ("input_dim", "hidden_dim"),
    "circadian": ("input_dim", "hidden_dim", "min_hidden_dim", "max_hidden_dim"),
    "torch_predictive": ("feature_dim", "hidden_dim", "num_classes"),
    "torch_circadian": (
        "feature_dim",
        "hidden_dim",
        "num_classes",
        "min_hidden_dim",
        "max_hidden_dim",
    ),
}


def _construct(kind: str, **changes: Any) -> Any:
    if kind == "backprop":
        options = {"input_dim": 2, "hidden_dim": 2, "seed": 79, **changes}
        return BackpropMLP(**options)
    if kind == "predictive":
        options = {"input_dim": 2, "hidden_dim": 2, "seed": 79, **changes}
        return PredictiveCodingNetwork(**options)
    if kind == "circadian":
        options = {
            "input_dim": 2,
            "hidden_dim": 2,
            "seed": 79,
            "min_hidden_dim": 1,
            "max_hidden_dim": 4,
            **changes,
        }
        return CircadianPredictiveCodingNetwork(**options)
    torch = pytest.importorskip("torch")
    options = {
        "feature_dim": 2,
        "hidden_dim": 2,
        "num_classes": 3,
        "device": torch.device("cpu"),
        "seed": 79,
        **changes,
    }
    if kind == "torch_predictive":
        return PredictiveCodingHead(**options)
    options.update({"min_hidden_dim": 1, "max_hidden_dim": 4})
    options.update(changes)
    return CircadianPredictiveCodingHead(**options)


@pytest.mark.parametrize("kind", list(FIELDS))
@pytest.mark.parametrize("invalid", INVALID)
def test_constructor_rejects_invalid_dimensions(kind: str, invalid: Any) -> None:
    for name in FIELDS[kind]:
        with pytest.raises(ValueError, match=f"{name}.*positive integer"):
            _construct(kind, **{name: invalid})


@pytest.mark.parametrize("kind", ["backprop", "predictive", "circadian"])
@pytest.mark.parametrize("invalid", INVALID)
def test_numpy_hidden_dims_entries_reject_invalid_values(kind: str, invalid: Any) -> None:
    with pytest.raises(ValueError, match="hidden_dims.*positive integer"):
        _construct(kind, hidden_dims=(2, invalid))


@pytest.mark.parametrize("kind", ["backprop", "predictive", "circadian"])
def test_valid_multilayer_dimensions_keep_seeded_initialization(kind: str) -> None:
    first = _construct(kind, hidden_dims=(2, 3))
    second = _construct(kind, hidden_dims=(2, 3))
    np.testing.assert_array_equal(first.weight_hidden_output, second.weight_hidden_output)
    np.testing.assert_array_equal(first.weight_input_hidden, second.weight_input_hidden)


@pytest.mark.parametrize("kind", list(FIELDS))
def test_numpy_integer_dimensions_remain_accepted(kind: str) -> None:
    input_name = "feature_dim" if kind.startswith("torch") else "input_dim"
    model = _construct(kind, **{input_name: np.int64(2), "hidden_dim": np.int64(2)})
    if kind.startswith("torch"):
        assert tuple(model.weight_feature_hidden.shape) == (2, 2)
    else:
        assert model.weight_input_hidden.shape == (2, 2)
