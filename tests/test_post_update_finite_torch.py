"""Torch PC heads reject a nonfinite candidate before changing head state."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from src.core.resnet50_variants import (  # noqa: E402
    CircadianPredictiveCodingHead,
    PredictiveCodingHead,
)


PARAMETERS = (
    "weight_feature_hidden",
    "bias_hidden",
    "weight_hidden_output",
    "bias_output",
)


def _head(kind: str) -> Any:
    if kind == "predictive":
        head = PredictiveCodingHead(2, 1, 2, torch.device("cpu"), seed=43)
    else:
        head = CircadianPredictiveCodingHead(
            2, 1, 2, torch.device("cpu"), seed=43, min_hidden_dim=1
        )
        head._split_cooldown[:] = 3
        head._prune_cooldown[:] = 2
    for name in PARAMETERS:
        setattr(head, name, getattr(head, name).double())
    head.weight_feature_hidden[:] = 0.0
    head.bias_hidden[:] = 0.0
    head.weight_hidden_output[:] = torch.tensor([[-0.5, 0.5]], dtype=torch.float64)
    head.bias_output[:] = 0.0
    return head


def _state(head: Any) -> tuple[dict[str, Any], tuple[Any, ...]]:
    names = [*PARAMETERS, "_traffic_sum"]
    scalars: tuple[Any, ...] = (head._traffic_steps,)
    if isinstance(head, CircadianPredictiveCodingHead):
        names.extend(
            [
                "_chemical",
                "_chemical_fast",
                "_chemical_slow",
                "_importance_ema",
                "_neuron_age",
                "_split_cooldown",
                "_prune_cooldown",
            ]
        )
        scalars += (
            head._steps_since_sleep,
            tuple(head._energy_history),
            head._reward_error_ema,
            head._last_reward_scale,
        )
    return ({name: getattr(head, name).clone() for name in names}, scalars)


def _assert_unchanged(head: Any, before: tuple[dict[str, Any], tuple[Any, ...]]) -> None:
    original_tensors, original_scalars = before
    current_tensors, current_scalars = _state(head)
    assert original_scalars == current_scalars
    for name, original in original_tensors.items():
        torch.testing.assert_close(current_tensors[name], original, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("kind", ["predictive", "circadian"])
def test_overflowing_candidate_rejected_before_torch_head_commit(kind: str) -> None:
    head = _head(kind)
    before = _state(head)
    with pytest.raises(FloatingPointError, match="nonfinite.*(update|diagnostic)"):
        head.train_step(
            torch.tensor([[1e200, 0.0]], dtype=torch.float64),
            torch.tensor([1]),
            1e200,
            1,
            0.2,
        )
    _assert_unchanged(head, before)


@pytest.mark.parametrize("kind", ["predictive", "circadian"])
def test_overflowing_squared_diagnostic_rejected_before_torch_commit(kind: str) -> None:
    head = _head(kind)
    head.weight_hidden_output[:] = torch.tensor([[-1.5e-154, 1.5e-154]], dtype=torch.float64)
    before = _state(head)
    with pytest.raises(FloatingPointError, match="nonfinite.*(update|diagnostic)"):
        head.train_step(torch.zeros((1, 2), dtype=torch.float64), torch.tensor([1]), 0.01, 1, 1e308)
    _assert_unchanged(head, before)


@pytest.mark.parametrize("kind", ["predictive", "circadian"])
def test_nonfinite_model_rejected_before_torch_cooldown_decay(kind: str) -> None:
    head = _head(kind)
    head.weight_hidden_output[0, 0] = float("nan")
    before = _state(head)
    with pytest.raises(FloatingPointError, match="nonfinite"):
        head.train_step(
            torch.tensor([[0.5, 0.2]], dtype=torch.float64), torch.tensor([1]), 0.03, 1, 0.2
        )
    _assert_unchanged(head, before)
