"""Fixed train-only probes for the supervised-error modulation signal."""

from __future__ import annotations

from copy import deepcopy
from math import log

import numpy as np
import pytest

from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork


def _fixed_train_batches() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    # Why this: a fixed logistic reference gives 0.2/0.8 clean predictions;
    # the first feature outlier alone makes a confident wrong prediction.
    distance = log(4.0) / 2.0
    clean_features = np.array(
        [[-distance, 0.0], [distance, 0.0], [-distance, 0.0], [distance, 0.0]],
        dtype=np.float64,
    )
    clean_targets = np.array([[0.0], [1.0], [0.0], [1.0]], dtype=np.float64)
    flipped_targets = clean_targets.copy()
    flipped_targets[0, 0] = 1.0
    outlier_features = clean_features.copy()
    outlier_features[0, 0] = 3.0
    return {
        "clean": (clean_features, clean_targets),
        "label_flip": (clean_features.copy(), flipped_targets),
        "feature_outlier": (outlier_features, clean_targets.copy()),
    }


def _probabilities(features: np.ndarray) -> np.ndarray:
    logits = features @ np.array([[2.0], [0.0]], dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-logits))


def _cross_entropy(probabilities: np.ndarray, targets: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-8, 1.0 - 1e-8)
    return float(np.mean(-targets * np.log(clipped) - (1.0 - targets) * np.log(1.0 - clipped)))


def _offline_signals(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    error = np.abs(probabilities - targets)
    after_fixed_correction = probabilities + 0.2 * (targets - probabilities)
    before_loss = _cross_entropy(probabilities, targets)
    after_loss = _cross_entropy(after_fixed_correction, targets)
    return {
        "mean_absolute_error": float(np.mean(error)),
        "clipped_error": float(np.mean(np.minimum(error, 0.5))),
        "loss_improvement": before_loss - after_loss,
        "loss_ratio": after_loss / before_loss,
        "unmodulated_scale": 1.0,
    }


def test_numpy_historical_scale_rises_for_label_flip_and_feature_outlier() -> None:
    config = CircadianConfig(use_reward_modulated_learning=True)
    model = CircadianPredictiveCodingNetwork(2, 4, seed=41, circadian_config=config)
    batches = _fixed_train_batches()
    clean_features, clean_targets = batches["clean"]
    clean_error = _probabilities(clean_features) - clean_targets
    assert model._compute_reward_scale(clean_error) == pytest.approx(1.0)
    baseline = model._reward_error_ema
    assert baseline is not None
    assert baseline == pytest.approx(0.2)

    for name in ("label_flip", "feature_outlier"):
        features, targets = batches[name]
        probe = deepcopy(model)
        batch_error = float(np.mean(np.abs(_probabilities(features) - targets)))
        scale = probe._compute_reward_scale(_probabilities(features) - targets)
        assert scale == pytest.approx(min(config.reward_scale_max, batch_error / baseline))
        assert scale == config.reward_scale_max
        assert probe._reward_error_ema == pytest.approx(
            config.reward_baseline_decay * baseline
            + (1.0 - config.reward_baseline_decay) * batch_error
        )

    unmodulated = CircadianPredictiveCodingNetwork(2, 4, seed=41)
    for features, targets in batches.values():
        assert unmodulated._compute_reward_scale(_probabilities(features) - targets) == 1.0
    assert unmodulated._reward_error_ema is None


def test_torch_historical_scale_matches_numpy_signal_on_cpu() -> None:
    torch = pytest.importorskip("torch")
    from src.core.resnet50_variants import CircadianHeadConfig, CircadianPredictiveCodingHead

    config = CircadianHeadConfig(use_reward_modulated_learning=True)
    head = CircadianPredictiveCodingHead(
        2, 4, 2, torch.device("cpu"), seed=41, config=config, min_hidden_dim=2
    )
    batches = _fixed_train_batches()
    clean_features, clean_targets = batches["clean"]
    clean_error = torch.tensor(_probabilities(clean_features) - clean_targets, dtype=torch.float64)
    assert head._compute_reward_scale(clean_error) == pytest.approx(1.0)
    assert head._reward_error_ema == pytest.approx(0.2)
    for name in ("label_flip", "feature_outlier"):
        features, targets = batches[name]
        probe = CircadianPredictiveCodingHead(
            2, 4, 2, torch.device("cpu"), seed=41, config=config, min_hidden_dim=2
        )
        probe.restore_state(head.snapshot_state())
        error = torch.tensor(_probabilities(features) - targets, dtype=torch.float64)
        assert probe._compute_reward_scale(error) == config.reward_scale_max
        batch_error = float(torch.mean(torch.abs(error)).item())
        assert probe._reward_error_ema == pytest.approx(0.95 * 0.2 + 0.05 * batch_error)

    unmodulated = CircadianPredictiveCodingHead(
        2, 4, 2, torch.device("cpu"), seed=41, min_hidden_dim=2
    )
    for features, targets in batches.values():
        error = torch.tensor(_probabilities(features) - targets, dtype=torch.float64)
        assert unmodulated._compute_reward_scale(error) == 1.0
    assert unmodulated._reward_error_ema is None


def test_offline_comparators_expose_noise_and_lookahead_limits() -> None:
    batches = _fixed_train_batches()
    clean_features, clean_targets = batches["clean"]
    clean = _offline_signals(_probabilities(clean_features), clean_targets)
    flipped_features, flipped_targets = batches["label_flip"]
    flipped = _offline_signals(_probabilities(flipped_features), flipped_targets)
    outlier_features, outlier_targets = batches["feature_outlier"]
    outlier = _offline_signals(_probabilities(outlier_features), outlier_targets)

    assert clean["mean_absolute_error"] == pytest.approx(0.2)
    assert clean["clipped_error"] == pytest.approx(0.2)
    assert flipped["mean_absolute_error"] > flipped["clipped_error"] > clean["clipped_error"]
    assert outlier["mean_absolute_error"] > outlier["clipped_error"]
    assert flipped["clipped_error"] / clean["clipped_error"] == pytest.approx(1.375)
    assert outlier["clipped_error"] / clean["clipped_error"] == pytest.approx(1.375)
    assert clean["loss_improvement"] == pytest.approx(0.048790164169432)
    assert flipped["loss_improvement"] == pytest.approx(0.183539289352604)
    assert outlier["loss_improvement"] == pytest.approx(1.137312541761222)
    assert outlier["loss_improvement"] > flipped["loss_improvement"] > clean["loss_improvement"]
    assert all(item["unmodulated_scale"] == 1.0 for item in (clean, flipped, outlier))


def test_fixed_probe_changes_only_declared_train_row_or_label() -> None:
    clean, flipped, outlier = _fixed_train_batches().values()
    np.testing.assert_array_equal(clean[0], flipped[0])
    np.testing.assert_array_equal(clean[1][1:], flipped[1][1:])
    np.testing.assert_array_equal(clean[1], outlier[1])
    np.testing.assert_array_equal(clean[0][1:], outlier[0][1:])
    assert flipped[1][0, 0] != clean[1][0, 0]
    assert outlier[0][0, 0] != clean[0][0, 0]
