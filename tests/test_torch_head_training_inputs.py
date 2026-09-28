"""Torch PC heads reject malformed training batches before adaptive changes."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from src.core.resnet50_variants import (  # noqa: E402
    CircadianPredictiveCodingHead,
    PredictiveCodingHead,
)


FEATURES = torch.tensor([[0.3, -0.2], [-0.5, 0.4]], dtype=torch.float32)
TARGETS = torch.tensor([0, 2], dtype=torch.long)


def _head(kind: str) -> Any:
    if kind == "predictive":
        return PredictiveCodingHead(2, 3, 3, torch.device("cpu"), seed=29)
    head = CircadianPredictiveCodingHead(2, 3, 3, torch.device("cpu"), seed=29, min_hidden_dim=3)
    head._split_cooldown[:] = torch.tensor([2, 1, 3], dtype=torch.int32)
    head._prune_cooldown[:] = torch.tensor([1, 2, 3], dtype=torch.int32)
    return head


def _state(head: Any) -> tuple[dict[str, Any], tuple[Any, ...]]:
    names = [
        "weight_feature_hidden",
        "bias_hidden",
        "weight_hidden_output",
        "bias_output",
        "_traffic_sum",
    ]
    scalars: tuple[Any, ...] = (head._traffic_steps,)
    if isinstance(head, CircadianPredictiveCodingHead):
        names.extend(
            [
                "_chemical",
                "_chemical_fast",
                "_chemical_slow",
                "_neuron_age",
                "_importance_ema",
                "_split_cooldown",
                "_prune_cooldown",
            ]
        )
        scalars = (
            *scalars,
            head._steps_since_sleep,
            tuple(head._energy_history),
            head._reward_error_ema,
            head._last_reward_scale,
        )
        names.append("_split_generator")
    tensors = {
        name: getattr(head, name).get_state().clone()
        if name == "_split_generator"
        else getattr(head, name).clone()
        for name in names
    }
    return tensors, scalars


@pytest.mark.parametrize("kind", ["predictive", "circadian"])
@pytest.mark.parametrize(
    "case,features,targets,rate,steps,latent_rate,message",
    [
        (
            "empty",
            torch.empty((0, 2)),
            torch.empty((0,), dtype=torch.long),
            0.03,
            2,
            0.2,
            "nonempty",
        ),
        ("feature_rank", FEATURES[0], TARGETS, 0.03, 2, 0.2, "features must be 2D"),
        ("feature_width", torch.zeros((2, 4)), TARGETS, 0.03, 2, 0.2, "feature width"),
        ("feature_dtype", FEATURES.double(), TARGETS, 0.03, 2, 0.2, "dtype"),
        ("feature_integer", FEATURES.to(torch.int32), TARGETS, 0.03, 2, 0.2, "floating"),
        (
            "feature_nan",
            torch.tensor([[float("nan"), 0.0], [0.0, 0.0]]),
            TARGETS,
            0.03,
            2,
            0.2,
            "finite",
        ),
        (
            "feature_inf",
            torch.tensor([[float("inf"), 0.0], [0.0, 0.0]]),
            TARGETS,
            0.03,
            2,
            0.2,
            "finite",
        ),
        ("target_rank", FEATURES, TARGETS[:, None], 0.03, 2, 0.2, "targets must be 1D"),
        ("target_count", FEATURES, TARGETS[:1], 0.03, 2, 0.2, "same batch size"),
        ("target_float", FEATURES, TARGETS.float(), 0.03, 2, 0.2, "int64"),
        ("target_low", FEATURES, torch.tensor([-1, 2]), 0.03, 2, 0.2, "class range"),
        ("target_high", FEATURES, torch.tensor([0, 3]), 0.03, 2, 0.2, "class range"),
        ("target_list", FEATURES, [0, 2], 0.03, 2, 0.2, "targets must be"),
        ("rate_nan", FEATURES, TARGETS, float("nan"), 2, 0.2, "learning_rate.*finite"),
        ("rate_inf", FEATURES, TARGETS, float("inf"), 2, 0.2, "learning_rate.*finite"),
        ("rate_zero", FEATURES, TARGETS, 0.0, 2, 0.2, "learning_rate.*positive"),
        ("steps_zero", FEATURES, TARGETS, 0.03, 0, 0.2, "inference_steps.*positive"),
        (
            "latent_rate_nan",
            FEATURES,
            TARGETS,
            0.03,
            2,
            float("nan"),
            "inference_learning_rate.*finite",
        ),
    ],
)
def test_invalid_torch_head_input_fails_before_state_change(
    kind: str,
    case: str,
    features: Any,
    targets: Any,
    rate: float,
    steps: int,
    latent_rate: float,
    message: str,
) -> None:
    head = _head(kind)
    before_tensors, before_scalars = _state(head)
    with pytest.raises(ValueError, match=message):
        head.train_step(features, targets, rate, steps, latent_rate)
    after_tensors, after_scalars = _state(head)
    assert after_scalars == before_scalars
    for name, previous in before_tensors.items():
        torch.testing.assert_close(after_tensors[name], previous, atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["predictive", "circadian"])
def test_torch_head_accepts_valid_multiclass_batch(kind: str) -> None:
    head = _head(kind)
    before = head.weight_hidden_output.clone()
    metric = head.train_step(FEATURES, TARGETS, 0.03, 2, 0.2)
    assert torch.isfinite(torch.tensor(metric))
    assert not torch.equal(head.weight_hidden_output, before)
