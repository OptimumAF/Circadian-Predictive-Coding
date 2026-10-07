from __future__ import annotations

import pytest

pytest.importorskip("torch")

from src.app.matched_head_benchmark import _hash_trained_head
from src.core.resnet50_variants import (
    BackpropMLPHead,
    BackpropMLPResNet50Classifier,
    CircadianHeadConfig,
    CircadianPredictiveCodingHead,
    PredictiveCodingHead,
)


def test_split_noise_is_model_owned_under_reversed_cpu_execution() -> None:
    torch = pytest.importorskip("torch")
    config = CircadianHeadConfig(
        split_threshold=0.8,
        max_split_per_sleep=1,
        max_prune_per_sleep=0,
        split_noise_scale=0.05,
    )

    def run_order(order: tuple[str, str]) -> dict[str, tuple[tuple[int, ...], str]]:
        observations: dict[str, tuple[tuple[int, ...], str]] = {}
        for name in order:
            head = CircadianPredictiveCodingHead(
                feature_dim=6,
                hidden_dim=4,
                num_classes=2,
                device=torch.device("cpu"),
                seed={"first": 41, "second": 43}[name],
                config=config,
                min_hidden_dim=4,
                max_hidden_dim=6,
            )
            head._chemical = torch.tensor([0.1, 0.95, 0.2, 0.3], dtype=torch.float32)
            decision = head.sleep_event(force_sleep=True)
            observations[name] = (decision.split_indices, _hash_trained_head(head))
            _ = torch.randn(257)
        return observations

    forward = run_order(("first", "second"))
    _ = torch.randn(509)
    reverse = run_order(("second", "first"))

    assert forward == reverse
    assert all(indices == (1,) for indices, _ in forward.values())
    assert forward["first"][1] != forward["second"][1]


def test_should_match_predictive_coding_head_architecture_and_initial_tensors() -> None:
    torch = pytest.importorskip("torch")
    device = torch.device("cpu")
    pc_head = PredictiveCodingHead(5, 4, 3, device, seed=53)
    backprop_head = BackpropMLPHead(5, 4, 3, device, seed=53)
    features = torch.tensor([[0.3, -0.2, 0.7, 0.0, -0.4]], dtype=torch.float32)

    for name, expected_shape in (
        ("weight_feature_hidden", (5, 4)),
        ("bias_hidden", (1, 4)),
        ("weight_hidden_output", (4, 3)),
        ("bias_output", (1, 3)),
    ):
        pc_tensor = getattr(pc_head, name)
        backprop_parameter = getattr(backprop_head, name)
        assert tuple(backprop_parameter.shape) == expected_shape
        assert backprop_parameter.dtype == pc_tensor.dtype == torch.float32
        assert backprop_parameter.requires_grad
        assert torch.equal(backprop_parameter.detach(), pc_tensor)
        assert backprop_parameter.data_ptr() != pc_tensor.data_ptr()

    assert backprop_head.hidden_dim == pc_head.hidden_dim
    assert backprop_head.parameter_count() == pc_head.parameter_count()
    assert torch.equal(backprop_head.forward_logits(features), pc_head.predict_logits(features))


def test_should_update_matched_backprop_head_in_one_cpu_step() -> None:
    torch = pytest.importorskip("torch")
    head = BackpropMLPHead(5, 4, 3, torch.device("cpu"), seed=59)
    features = torch.tensor(
        [[0.3, -0.2, 0.7, 0.0, -0.4], [-0.1, 0.5, 0.2, -0.6, 0.8]],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 2], dtype=torch.long)
    optimizer = torch.optim.SGD(head.trainable_parameters(), lr=0.1)
    initial_tensors = [parameter.detach().clone() for parameter in head.trainable_parameters()]
    initial_loss = torch.nn.functional.cross_entropy(head.forward_logits(features), targets)

    optimizer.zero_grad(set_to_none=True)
    initial_loss.backward()
    optimizer.step()

    updated_loss = torch.nn.functional.cross_entropy(head.forward_logits(features), targets)
    assert float(updated_loss.item()) < float(initial_loss.item())
    assert all(
        not torch.equal(parameter.detach(), initial)
        for parameter, initial in zip(head.trainable_parameters(), initial_tensors)
    )


def test_should_wrap_matched_head_with_frozen_feature_extractor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from src.core import resnet50_variants

    def build_backbone(
        device: object, freeze_backbone: bool, backbone_weights: str
    ) -> tuple[object, int]:
        assert freeze_backbone is True
        assert backbone_weights == "none"
        return torch.nn.Identity().to(device), 5

    monkeypatch.setattr(resnet50_variants, "_build_resnet50_backbone", build_backbone)
    model = BackpropMLPResNet50Classifier(
        num_classes=3,
        device=torch.device("cpu"),
        head_hidden_dim=4,
        seed=61,
        freeze_backbone=True,
    )
    features = torch.ones((2, 5), dtype=torch.float32)

    assert torch.equal(model.forward_logits(features), model.head.forward_logits(features))
    assert tuple(model.forward_logits(features).shape) == (2, 3)
    assert model.trainable_parameter_count() == model.head.parameter_count() == 39
    assert len(model.trainable_parameters()) == 4


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


def test_should_scale_learning_rate_by_reward_signal_in_torch_head() -> None:
    torch = pytest.importorskip("torch")
    device = torch.device("cpu")
    config = CircadianHeadConfig(
        use_reward_modulated_learning=True,
        reward_baseline_decay=0.95,
        reward_scale_min=0.8,
        reward_scale_max=1.6,
    )
    head = CircadianPredictiveCodingHead(
        feature_dim=6,
        hidden_dim=5,
        num_classes=2,
        device=device,
        seed=37,
        config=config,
        min_hidden_dim=4,
        max_hidden_dim=10,
    )
    features = torch.tensor(
        [[0.3, -0.2, 0.5, 0.1, -0.4, 0.2], [-0.1, 0.4, 0.2, -0.6, 0.8, 0.3]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    head._reward_error_ema = 10.0
    head.train_step(
        features=features,
        targets=labels,
        learning_rate=0.03,
        inference_steps=8,
        inference_learning_rate=0.2,
    )
    easy_scale = head.last_reward_scale()

    head._reward_error_ema = 0.01
    head.train_step(
        features=features,
        targets=labels,
        learning_rate=0.03,
        inference_steps=8,
        inference_learning_rate=0.2,
    )
    hard_scale = head.last_reward_scale()

    assert easy_scale <= config.reward_scale_min + 1e-6
    assert hard_scale > 1.0


def test_should_expand_sleep_budget_when_plateau_and_variance_are_high_in_torch_head() -> None:
    torch = pytest.importorskip("torch")
    device = torch.device("cpu")
    config = CircadianHeadConfig(
        use_adaptive_sleep_budget=True,
        max_split_per_sleep=4,
        max_prune_per_sleep=4,
        sleep_energy_window=3,
        sleep_plateau_delta=0.1,
        sleep_chemical_variance_threshold=0.05,
        adaptive_sleep_budget_min_scale=0.25,
        adaptive_sleep_budget_max_scale=1.0,
        adaptive_sleep_budget_plateau_weight=0.5,
        adaptive_sleep_budget_variance_weight=0.5,
    )
    head = CircadianPredictiveCodingHead(
        feature_dim=6,
        hidden_dim=10,
        num_classes=2,
        device=device,
        seed=41,
        config=config,
        min_hidden_dim=4,
        max_hidden_dim=20,
    )

    head._energy_history = [1.0, 0.7, 0.3]
    head._chemical = torch.full((10,), 0.2, dtype=torch.float32)
    low_split_budget, low_prune_budget, _ = head._resolve_sleep_budgets(
        current_step=None, total_steps=None
    )

    head._energy_history = [0.5, 0.5, 0.5]
    head._chemical = torch.tensor([0.0, 1.0] * 5, dtype=torch.float32)
    high_split_budget, high_prune_budget, _ = head._resolve_sleep_budgets(
        current_step=None, total_steps=None
    )

    assert low_split_budget == 1
    assert low_prune_budget == 1
    assert high_split_budget == 4
    assert high_prune_budget == 4
