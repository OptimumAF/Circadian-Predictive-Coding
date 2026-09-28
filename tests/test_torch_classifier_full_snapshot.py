"""Whole-classifier snapshots include backbone and head training state."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from src.core import resnet50_variants  # noqa: E402
from src.core.resnet50_variants import (  # noqa: E402
    CircadianHeadConfig,
    CircadianPredictiveCodingResNet50Classifier,
)


def _model(monkeypatch: pytest.MonkeyPatch) -> CircadianPredictiveCodingResNet50Classifier:
    def build_backbone(
        device: Any, freeze_backbone: bool, backbone_weights: str
    ) -> tuple[Any, int]:
        assert backbone_weights == "none"
        backbone = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3)).to(device)
        if freeze_backbone:
            for parameter in backbone.parameters():
                parameter.requires_grad = False
            backbone.eval()
        return backbone, 3

    monkeypatch.setattr(resnet50_variants, "_build_resnet50_backbone", build_backbone)
    return CircadianPredictiveCodingResNet50Classifier(
        num_classes=2,
        device=torch.device("cpu"),
        head_hidden_dim=4,
        seed=311,
        freeze_backbone=False,
        circadian_config=CircadianHeadConfig(
            sleep_mode="components",
            max_split_per_sleep=1,
            max_prune_per_sleep=0,
            split_threshold=0.8,
            split_weight_norm_mix=0.0,
            split_importance_mix=0.0,
            split_noise_scale=0.1,
        ),
        min_hidden_dim=3,
        max_hidden_dim=6,
    )


def _assert_same_full_state(left: Any, right: Any) -> None:
    assert left.format_version == right.format_version
    assert left.device == right.device
    assert left.freeze_backbone == right.freeze_backbone
    assert left.backbone_module_types == right.backbone_module_types
    assert left.backbone_training == right.backbone_training
    assert left.backbone_requires_grad == right.backbone_requires_grad
    assert left.backbone_state.keys() == right.backbone_state.keys()
    for name, value in left.backbone_state.items():
        assert torch.equal(value, right.backbone_state[name]), name
    assert left.head_state.keys() == right.head_state.keys()
    for name, value in left.head_state.items():
        if torch.is_tensor(value):
            assert torch.equal(value, right.head_state[name]), name
        else:
            assert value == right.head_state[name], name


def test_full_classifier_snapshot_restores_backbone_and_head_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(monkeypatch)
    images = torch.tensor([[0.2, -0.3, 0.1], [-0.1, 0.4, 0.5]])
    targets = torch.tensor([1, 0], dtype=torch.long)
    model.train_step(images, targets, 0.03, 2, 0.2)
    model.head._chemical = torch.tensor([0.95, 0.6, 0.3, 0.1])
    model.backbone[0].eval()
    saved = model.snapshot_full_state()
    expected = deepcopy(saved)
    assert model.snapshot_state().keys() == model.head.snapshot_state().keys()
    assert "backbone_state" not in model.snapshot_state()

    first_energy = model.train_step(images, targets, 0.03, 2, 0.2)
    first_sleep = model.sleep_event(force_sleep=True)
    assert first_sleep.split_indices == (0,)
    first_result = model.snapshot_full_state()

    with torch.no_grad():
        model.backbone[0].weight.add_(3.0)
        model.backbone[1].running_mean.add_(3.0)
    model.backbone.eval()
    model.backbone[0].weight.requires_grad_(False)
    model.restore_full_state(saved)
    _assert_same_full_state(model.snapshot_full_state(), expected)

    saved.backbone_state["0.weight"].zero_()
    saved.head_state["chemical"].zero_()
    _assert_same_full_state(model.snapshot_full_state(), expected)

    repeat_energy = model.train_step(images, targets, 0.03, 2, 0.2)
    repeat_sleep = model.sleep_event(force_sleep=True)
    assert repeat_energy == first_energy
    assert repeat_sleep == first_sleep
    _assert_same_full_state(model.snapshot_full_state(), first_result)


@pytest.mark.parametrize("change", ["version", "backbone_shape", "mode", "head"])
def test_full_classifier_restore_rejects_incompatible_state_without_mutation(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    model = _model(monkeypatch)
    before = model.snapshot_full_state()
    broken = deepcopy(before)
    if change == "version":
        broken = replace(broken, format_version=99)
    elif change == "backbone_shape":
        broken.backbone_state["0.weight"] = broken.backbone_state["0.weight"][:-1]
    elif change == "mode":
        del broken.backbone_training["1"]
    else:
        broken.head_state["format_version"] = 99

    with pytest.raises((TypeError, ValueError), match="snapshot"):
        model.restore_full_state(broken)
    _assert_same_full_state(model.snapshot_full_state(), before)
