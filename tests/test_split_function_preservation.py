"""Isolated NumPy/Torch splits preserve the represented function."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork
from src.core.resnet50_variants import CircadianHeadConfig, CircadianPredictiveCodingHead


def _numpy_model(noise: float = 0.0) -> CircadianPredictiveCodingNetwork:
    return CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=4,
        seed=701,
        circadian_config=CircadianConfig(
            sleep_mode="components",
            sleep_enable_homeostasis=False,
            sleep_enable_chemical_reset=False,
            sleep_enable_replay=False,
            max_split_per_sleep=1,
            max_prune_per_sleep=0,
            split_weight_norm_mix=0.0,
            split_importance_mix=0.0,
            split_noise_scale=noise,
        ),
        min_hidden_dim=3,
        max_hidden_dim=7,
    )


def _torch_head(noise: float = 0.0) -> CircadianPredictiveCodingHead:
    torch = pytest.importorskip("torch")
    return CircadianPredictiveCodingHead(
        feature_dim=3,
        hidden_dim=4,
        num_classes=2,
        device=torch.device("cpu"),
        seed=703,
        config=CircadianHeadConfig(
            sleep_mode="components",
            sleep_enable_homeostasis=False,
            sleep_enable_chemical_reset=False,
            max_split_per_sleep=1,
            max_prune_per_sleep=0,
            split_weight_norm_mix=0.0,
            split_importance_mix=0.0,
            split_noise_scale=noise,
        ),
        min_hidden_dim=3,
        max_hidden_dim=7,
    )


def test_numpy_single_and_repeated_zero_noise_splits_preserve_predictions() -> None:
    model = _numpy_model()
    inputs = np.array([[0.2, -0.3], [-0.4, 0.5], [1.1, 0.7]], dtype=np.float64)
    initial = model.predict_proba(inputs)

    for source in (0, 4):
        chemical = np.full(model.hidden_dim, 0.5)
        chemical[source] = 1.0
        model.set_chemical_state(chemical)
        before = model.predict_proba(inputs)
        original_row = model.weight_hidden_output[source, :].copy()
        original_column = model.weight_input_hidden[:, source].copy()
        original_bias = model.bias_hidden[0, source]

        event = model.sleep_event(force_sleep=True)

        child = event.old_hidden_dim
        assert event.split_indices == (source,)
        assert event.pruned_indices == ()
        np.testing.assert_allclose(model.weight_input_hidden[:, child], original_column, rtol=0)
        np.testing.assert_allclose(model.bias_hidden[0, child], original_bias, rtol=0)
        np.testing.assert_allclose(
            model.weight_hidden_output[source, :] + model.weight_hidden_output[child, :],
            original_row,
            rtol=0,
            atol=1e-15,
        )
        np.testing.assert_allclose(model.predict_proba(inputs), before, rtol=0, atol=1e-12)
    np.testing.assert_allclose(model.predict_proba(inputs), initial, rtol=0, atol=1e-12)


def test_torch_single_and_repeated_zero_noise_splits_preserve_logits() -> None:
    torch = pytest.importorskip("torch")
    head = _torch_head()
    features = torch.tensor(
        [[0.2, -0.3, 0.4], [-0.4, 0.5, -0.1], [1.1, 0.7, -0.6]], dtype=torch.float32
    )
    initial = head.predict_logits(features)

    for source in (0, 4):
        head._chemical = torch.full((head.hidden_dim,), 0.5)
        head._chemical[source] = 1.0
        before = head.predict_logits(features)
        original_row = head.weight_hidden_output[source, :].clone()
        original_column = head.weight_feature_hidden[:, source].clone()
        original_bias = head.bias_hidden[0, source].clone()

        event = head.sleep_event(force_sleep=True)

        child = event.old_hidden_dim
        assert event.split_indices == (source,)
        assert event.pruned_indices == ()
        torch.testing.assert_close(
            head.weight_feature_hidden[:, child], original_column, rtol=0, atol=0
        )
        torch.testing.assert_close(head.bias_hidden[0, child], original_bias, rtol=0, atol=0)
        torch.testing.assert_close(
            head.weight_hidden_output[source, :] + head.weight_hidden_output[child, :],
            original_row,
            rtol=0,
            atol=2e-7,
        )
        torch.testing.assert_close(head.predict_logits(features), before, rtol=0, atol=2e-6)
    torch.testing.assert_close(head.predict_logits(features), initial, rtol=0, atol=2e-6)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_seeded_noisy_split_conserves_outgoing_row_with_nonzero_perturbation(
    backend: str,
) -> None:
    if backend == "numpy":
        model = _numpy_model(noise=0.2)
        model.set_chemical_state(np.array([1.0, 0.5, 0.5, 0.5]))
        inputs = np.array([[0.2, -0.3], [-0.4, 0.5]], dtype=np.float64)
        before = model.predict_proba(inputs)
        original: Any = model.weight_hidden_output[0, :].copy()
        numpy_event = model.sleep_event(force_sleep=True)
        parent = model.weight_hidden_output[0, :]
        child = model.weight_hidden_output[numpy_event.old_hidden_dim, :]
        np.testing.assert_allclose(parent + child, original, rtol=0, atol=1e-15)
        assert np.max(np.abs(parent - 0.5 * original)) > 1e-8
        np.testing.assert_allclose(model.predict_proba(inputs), before, rtol=0, atol=1e-12)
    else:
        torch = pytest.importorskip("torch")
        head = _torch_head(noise=0.2)
        head._chemical = torch.tensor([1.0, 0.5, 0.5, 0.5])
        features = torch.tensor([[0.2, -0.3, 0.4], [-0.4, 0.5, -0.1]])
        before = head.predict_logits(features)
        original = head.weight_hidden_output[0, :].clone()
        torch_event = head.sleep_event(force_sleep=True)
        parent = head.weight_hidden_output[0, :]
        child = head.weight_hidden_output[torch_event.old_hidden_dim, :]
        torch.testing.assert_close(parent + child, original, rtol=0, atol=2e-7)
        assert torch.max(torch.abs(parent - 0.5 * original)).item() > 1e-8
        torch.testing.assert_close(head.predict_logits(features), before, rtol=0, atol=2e-6)
