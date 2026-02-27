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


def test_should_reduce_plasticity_for_high_importance_in_torch_head() -> None:
    torch = pytest.importorskip("torch")
    device = torch.device("cpu")
    config = CircadianHeadConfig(
        use_adaptive_plasticity_sensitivity=True,
        plasticity_sensitivity_min=0.2,
        plasticity_sensitivity_max=1.0,
        plasticity_importance_mix=1.0,
    )
    head = CircadianPredictiveCodingHead(
        feature_dim=6,
        hidden_dim=4,
        num_classes=2,
        device=device,
        seed=23,
        config=config,
        min_hidden_dim=4,
    )
    head._chemical = torch.full((4,), 0.4, dtype=torch.float32)
    head._importance_ema = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    plasticity = head._plasticity()

    assert float(plasticity[0].item()) < float(plasticity[1].item())


def test_should_trigger_adaptive_sleep_in_torch_circadian_head() -> None:
    torch = pytest.importorskip("torch")
    device = torch.device("cpu")
    config = CircadianHeadConfig(
        use_adaptive_sleep_trigger=True,
        min_sleep_steps=3,
        sleep_energy_window=3,
        sleep_plateau_delta=0.1,
        sleep_chemical_variance_threshold=0.01,
    )
    head = CircadianPredictiveCodingHead(
        feature_dim=6,
        hidden_dim=4,
        num_classes=2,
        device=device,
        seed=29,
        config=config,
        min_hidden_dim=4,
        max_hidden_dim=8,
    )
    head._steps_since_sleep = 3
    head._energy_history = [0.45, 0.43, 0.42]
    head._chemical = torch.tensor([0.1, 0.9, 0.2, 0.8], dtype=torch.float32)

    assert head.should_trigger_sleep() is True


def test_should_restore_snapshot_after_structure_change_in_torch_head() -> None:
    torch = pytest.importorskip("torch")
    device = torch.device("cpu")
    config = CircadianHeadConfig(
        split_threshold=0.7,
        prune_threshold=0.05,
        max_split_per_sleep=1,
        max_prune_per_sleep=0,
        split_noise_scale=0.0,
    )
    head = CircadianPredictiveCodingHead(
        feature_dim=6,
        hidden_dim=5,
        num_classes=2,
        device=device,
        seed=31,
        config=config,
        min_hidden_dim=4,
        max_hidden_dim=8,
    )
    snapshot = head.snapshot_state()
    head._chemical = torch.tensor([0.95, 0.03, 0.1, 0.2, 0.3], dtype=torch.float32)

    _ = head.sleep_event()
    assert head.hidden_dim != int(snapshot["weight_feature_hidden"].shape[1])

    head.restore_state(snapshot)

    assert head.hidden_dim == int(snapshot["weight_feature_hidden"].shape[1])
    assert torch.allclose(head.weight_feature_hidden, snapshot["weight_feature_hidden"])
    assert torch.allclose(head.weight_hidden_output, snapshot["weight_hidden_output"])
    assert torch.allclose(head._chemical, snapshot["chemical"])
