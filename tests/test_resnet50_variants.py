from __future__ import annotations

import pytest

pytest.importorskip("torch")

from src.core.resnet50_variants import CircadianHeadConfig, CircadianPredictiveCodingHead


def test_should_preserve_head_logits_after_function_preserving_split() -> None:
    torch = pytest.importorskip("torch")
    device = torch.device("cpu")
    config = CircadianHeadConfig(
        split_threshold=0.7,
        prune_threshold=-1.0,
        max_split_per_sleep=1,
        max_prune_per_sleep=0,
        split_noise_scale=0.0,
    )
    head = CircadianPredictiveCodingHead(
        feature_dim=8,
        hidden_dim=6,
        num_classes=3,
        device=device,
        seed=17,
        config=config,
        min_hidden_dim=4,
        max_hidden_dim=12,
    )
    head._chemical = torch.tensor([0.95, 0.10, 0.10, 0.20, 0.30, 0.40], dtype=torch.float32)

    features = torch.tensor(
        [[0.3, -0.2, 0.5, 0.1, -0.4, 0.2, 0.0, 0.7], [-0.1, 0.4, 0.2, -0.6, 0.8, 0.3, -0.5, 0.9]],
        dtype=torch.float32,
    )
    pre_sleep_logits = head.predict_logits(features)
    sleep_result = head.sleep_event()
    post_sleep_logits = head.predict_logits(features)

    assert sleep_result.split_indices == (0,)
    assert sleep_result.pruned_indices == ()
    assert head.hidden_dim == 7
    assert torch.allclose(pre_sleep_logits, post_sleep_logits, atol=1e-6)


def test_should_apply_split_cooldown_in_torch_circadian_head() -> None:
    torch = pytest.importorskip("torch")
    device = torch.device("cpu")
    config = CircadianHeadConfig(
        split_threshold=0.5,
        split_hysteresis_margin=0.1,
        split_cooldown_steps=3,
        max_split_per_sleep=1,
        max_prune_per_sleep=0,
        split_noise_scale=0.0,
    )
    head = CircadianPredictiveCodingHead(
        feature_dim=6,
        hidden_dim=4,
        num_classes=2,
        device=device,
        seed=19,
        config=config,
        min_hidden_dim=3,
        max_hidden_dim=8,
    )
    head._chemical = torch.tensor([0.95, 0.2, 0.1, 0.1], dtype=torch.float32)
    head._chemical_fast = head._chemical.clone()
    head._chemical_slow = head._chemical.clone()

    first_sleep = head.sleep_event()
    second_sleep = head.sleep_event()

    assert first_sleep.split_indices == (0,)
    assert second_sleep.split_indices == ()
