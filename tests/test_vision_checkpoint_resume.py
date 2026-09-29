"""Trusted local checkpoints preserve completed unmatched vision models."""

from __future__ import annotations

from dataclasses import fields, replace
import random
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from torch import nn  # noqa: E402

from src.app import resnet50_benchmark as vision  # noqa: E402
from src.app.resnet50_benchmark import (  # noqa: E402
    ResNet50BenchmarkConfig,
    VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL,
    VISION_SEEDED_UNMATCHED_PROTOCOL,
    VISION_VALIDATION_UNMATCHED_PROTOCOL,
)
from src.core import resnet50_variants  # noqa: E402
from src.infra.circadian_checkpoint_files import (  # noqa: E402
    TrustedLocalVisionCheckpointStore,
)


class InterruptedAfterSave(Exception):
    pass


class InterruptingStore:
    def __init__(
        self,
        store: TrustedLocalVisionCheckpointStore,
        after_variants: int = 1,
    ) -> None:
        self.store = store
        self.after_variants = after_variants

    def load(self) -> Any:
        return self.store.load()

    def save(self, checkpoint: Any) -> None:
        self.store.save(checkpoint)
        if checkpoint.next_variant_index == self.after_variants:
            raise InterruptedAfterSave()


class InterruptAtActiveStage:
    def __init__(
        self,
        store: TrustedLocalVisionCheckpointStore,
        stage: str,
        completed_epoch: int,
    ) -> None:
        self.store = store
        self.stage = stage
        self.completed_epoch = completed_epoch

    def load(self) -> Any:
        return self.store.load()

    def save(self, checkpoint: Any) -> None:
        self.store.save(checkpoint)
        active = getattr(checkpoint, "active_circadian", None)
        if (
            active is not None
            and active.stage == self.stage
            and active.completed_epoch == self.completed_epoch
        ):
            raise InterruptedAfterSave()


class StochasticVisionDataset:
    """Raw examples with process-local augmentation for spawned workers."""

    def __init__(self, dataset: Any) -> None:
        self.images = dataset.images
        self.labels = dataset.labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        jitter = 0.01 * (float(torch.rand(())) + float(np.random.random()) + random.random())
        return self.images[index] + jitter, self.labels[index]


def _config() -> ResNet50BenchmarkConfig:
    return ResNet50BenchmarkConfig(
        protocol_id=VISION_SEEDED_UNMATCHED_PROTOCOL,
        train_samples=8,
        guard_samples=8,
        validation_samples=8,
        test_samples=8,
        num_classes=3,
        image_size=32,
        batch_size=4,
        epochs=1,
        seed=73,
        device="cpu",
        target_accuracy=None,
        inference_batches=1,
        warmup_batches=0,
        backprop_freeze_backbone=True,
        backbone_weights="none",
        predictive_head_hidden_dim=16,
        predictive_inference_steps=2,
        circadian_head_hidden_dim=16,
        circadian_min_hidden_dim=16,
        circadian_max_hidden_dim=32,
        circadian_inference_steps=2,
        circadian_sleep_interval=1,
        circadian_force_sleep=True,
        circadian_sleep_warmup_steps=0,
    )


def _sleep_config(reject: bool = False) -> ResNet50BenchmarkConfig:
    return replace(
        _config(),
        epochs=2,
        circadian_sleep_mode="components",
        circadian_use_adaptive_sleep_trigger=False,
        circadian_use_adaptive_sleep_budget=False,
        circadian_use_adaptive_thresholds=False,
        circadian_head_hidden_dim=4,
        circadian_min_hidden_dim=4,
        circadian_max_hidden_dim=6,
        circadian_split_threshold=0.0,
        circadian_split_hysteresis_margin=0.0,
        circadian_split_cooldown_steps=0,
        circadian_sleep_max_change_fraction=1.0,
        circadian_max_prune_per_sleep=0,
        circadian_sleep_enable_prune=False,
        circadian_sleep_enable_homeostasis=False,
        circadian_sleep_enable_chemical_reset=False,
        circadian_sleep_split_only_until_fraction=1.0,
        circadian_sleep_prune_only_after_fraction=1.0,
        circadian_sleep_rollback_tolerance=0.0 if reject else 100.0,
    )


def _same_learning_reports(actual: Any, expected: Any) -> None:
    timed = {
        "train_seconds",
        "train_samples_per_second",
        "mean_train_step_ms",
        "inference_latency_mean_ms",
        "inference_latency_p95_ms",
        "inference_samples_per_second",
    }
    for left, right in zip(actual.reports, expected.reports, strict=True):
        for field in fields(left):
            if field.name not in timed:
                assert getattr(left, field.name) == pytest.approx(getattr(right, field.name))


@pytest.fixture
def tiny_vision_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep repeated sleep-boundary files small while training real heads."""

    class TinyBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.output = torch.nn.Linear(3, 16)

        def forward(self, images: Any) -> Any:
            return self.output(self.pool(images).flatten(1))

    def build(device: Any, freeze_backbone: bool, backbone_weights: str) -> Any:
        assert backbone_weights == "none"
        model = TinyBackbone().to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(not freeze_backbone)
        return model, 16

    monkeypatch.setattr(resnet50_variants, "_build_resnet50_backbone", build)


def test_seeded_vision_file_resume_keeps_completed_model_and_final_test_sealed(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    random.seed(700)
    np.random.seed(701)
    torch.manual_seed(702)
    control = vision.run_resnet50_benchmark(config)
    expected_draw = (random.random(), float(np.random.random()), float(torch.rand(())))

    original_build = vision._build_benchmark_loaders
    original_train = vision._train_seeded_variant
    trained: list[str] = []

    class SealedTestLoader:
        def __init__(self, wrapped: Any) -> None:
            self.wrapped = wrapped

        def __iter__(self) -> Any:
            if len(trained) != 3:
                raise AssertionError("final test opened before all three models trained")
            return iter(self.wrapped)

    def build(config: ResNet50BenchmarkConfig) -> Any:
        loaders = original_build(config)
        return replace(loaders, test_loader=SealedTestLoader(loaders.test_loader))

    def train(name: str, *args: Any, **kwargs: Any) -> Any:
        result = original_train(name, *args, **kwargs)
        trained.append(name)
        return result

    monkeypatch.setattr(vision, "_build_benchmark_loaders", build)
    monkeypatch.setattr(vision, "_train_seeded_variant", train)
    store = TrustedLocalVisionCheckpointStore(tmp_path / "vision.ckpt")
    random.seed(700)
    np.random.seed(701)
    torch.manual_seed(702)
    with pytest.raises(InterruptedAfterSave):
        vision.run_resnet50_benchmark(config, checkpoint_store=InterruptingStore(store))
    assert trained == ["backprop"]

    _ = (random.random(), np.random.random(), torch.rand(()))
    resumed = vision.run_resnet50_benchmark(
        config,
        checkpoint_store=store,
        resume_from_checkpoint=True,
    )
    assert trained == ["backprop", "predictive", "circadian"]
    assert resumed.training_order == control.training_order
    assert resumed.split_hashes == control.split_hashes
    assert resumed.trained_model_hashes == control.trained_model_hashes
    _same_learning_reports(resumed, control)
    assert (random.random(), float(np.random.random()), float(torch.rand(()))) == expected_draw


def test_vision_file_resume_from_all_trained_models_scores_final_test_once(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    tiny_vision_backbone: None,
) -> None:
    config = _config()
    random.seed(731)
    np.random.seed(732)
    torch.manual_seed(733)
    control = vision.run_resnet50_benchmark(config)
    expected_draw = (random.random(), float(np.random.random()), float(torch.rand(())))

    store = TrustedLocalVisionCheckpointStore(tmp_path / "vision.ckpt")
    random.seed(731)
    np.random.seed(732)
    torch.manual_seed(733)
    with pytest.raises(InterruptedAfterSave):
        vision.run_resnet50_benchmark(
            config,
            checkpoint_store=InterruptingStore(store, after_variants=3),
        )
    assert store.load().next_variant_index == 3
    monkeypatch.setattr(
        vision,
        "_train_seeded_variant",
        lambda *args, **kwargs: pytest.fail("terminal checkpoint retrained a model"),
    )
    resumed = vision.run_resnet50_benchmark(
        config,
        checkpoint_store=store,
        resume_from_checkpoint=True,
    )
    assert resumed.trained_model_hashes == control.trained_model_hashes
    _same_learning_reports(resumed, control)
    assert (random.random(), float(np.random.random()), float(torch.rand(()))) == expected_draw


@pytest.mark.parametrize(
    "protocol",
    [VISION_VALIDATION_UNMATCHED_PROTOCOL, VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL],
)
def test_earlier_vision_file_resume_preserves_shared_loader_across_models(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    tiny_vision_backbone: None,
    protocol: str,
) -> None:
    config = replace(_config(), protocol_id=protocol)
    trained: list[tuple[str, str]] = []
    for name in ("backprop", "predictive", "circadian"):
        method_name = f"_train_{name}"
        original_train = getattr(vision, method_name)

        def train(
            *args: Any, _name: str = name, _original: Any = original_train, **kwargs: Any
        ) -> Any:
            outcome = _original(*args, **kwargs)
            trained.append((_name, vision._hash_trained_model(outcome.model)))
            return outcome

        monkeypatch.setattr(vision, method_name, train)

    random.seed(741)
    np.random.seed(742)
    torch.manual_seed(743)
    control = vision.run_resnet50_benchmark(config)
    model_names = {
        "backprop": "BackpropResNet50",
        "predictive": "PredictiveCodingResNet50",
        "circadian": "CircadianPredictiveCodingResNet50",
    }
    control_hashes = {model_names[name]: digest for name, digest in trained}
    expected_draw = (random.random(), float(np.random.random()), float(torch.rand(())))

    original_build = vision._build_benchmark_loaders
    trained.clear()

    class SealedTestLoader:
        def __init__(self, wrapped: Any) -> None:
            self.wrapped = wrapped

        def __iter__(self) -> Any:
            if len(trained) != 3:
                raise AssertionError("earlier protocol final test opened before training")
            return iter(self.wrapped)

    def build(config: ResNet50BenchmarkConfig) -> Any:
        loaders = original_build(config)
        return replace(loaders, test_loader=SealedTestLoader(loaders.test_loader))

    monkeypatch.setattr(vision, "_build_benchmark_loaders", build)
    store = TrustedLocalVisionCheckpointStore(tmp_path / "vision.ckpt")
    random.seed(741)
    np.random.seed(742)
    torch.manual_seed(743)
    with pytest.raises(InterruptedAfterSave):
        vision.run_resnet50_benchmark(config, checkpoint_store=InterruptingStore(store))
    assert [name for name, _ in trained] == ["backprop"]
    _ = (random.random(), np.random.random(), torch.rand(()))
    resumed = vision.run_resnet50_benchmark(
        config,
        checkpoint_store=store,
        resume_from_checkpoint=True,
    )
    assert [name for name, _ in trained] == ["backprop", "predictive", "circadian"]
    assert resumed.split_hashes == control.split_hashes
    assert resumed.trained_model_hashes == control_hashes
    _same_learning_reports(resumed, control)
    assert (random.random(), float(np.random.random()), float(torch.rand(()))) == expected_draw


@pytest.mark.parametrize(
    ("protocol", "stage", "reject"),
    [
        (VISION_VALIDATION_UNMATCHED_PROTOCOL, "wake", False),
        (VISION_VALIDATION_UNMATCHED_PROTOCOL, "before_sleep", False),
        (VISION_VALIDATION_UNMATCHED_PROTOCOL, "after_sleep", False),
        (VISION_VALIDATION_UNMATCHED_PROTOCOL, "after_sleep", True),
        (VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL, "wake", False),
        (VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL, "before_sleep", False),
        (VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL, "after_sleep", False),
        (VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL, "after_sleep", True),
    ],
)
def test_earlier_vision_file_resume_across_guarded_sleep_matches_uninterrupted(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    tiny_vision_backbone: None,
    protocol: str,
    stage: str,
    reject: bool,
) -> None:
    config = replace(_sleep_config(reject), protocol_id=protocol)
    monkeypatch.setattr(
        vision,
        "_compute_pc_metrics",
        lambda _torch, model, _loader, _device, max_batches=None: (
            0.8,
            2.0 if reject and model.head.hidden_dim > 4 else 0.5,
        ),
    )
    random.seed(821)
    np.random.seed(822)
    torch.manual_seed(823)
    control = vision.run_resnet50_benchmark(
        config,
        checkpoint_store=TrustedLocalVisionCheckpointStore(tmp_path / "control.ckpt"),
    )
    expected_draw = (random.random(), float(np.random.random()), float(torch.rand(())))

    store = TrustedLocalVisionCheckpointStore(tmp_path / "vision.ckpt")
    random.seed(821)
    np.random.seed(822)
    torch.manual_seed(823)
    with pytest.raises(InterruptedAfterSave):
        vision.run_resnet50_benchmark(
            config,
            checkpoint_store=InterruptAtActiveStage(
                store,
                stage,
                0 if stage == "wake" else 1,
            ),
        )
    resumed = vision.run_resnet50_benchmark(
        config,
        checkpoint_store=store,
        resume_from_checkpoint=True,
    )
    assert resumed.split_hashes == control.split_hashes
    assert resumed.trained_model_hashes == control.trained_model_hashes
    _same_learning_reports(resumed, control)
    circadian = next(
        report
        for report in resumed.reports
        if report.model_name == "CircadianPredictiveCodingResNet50"
    )
    assert circadian.circadian_sleep_attempts > 0
    assert (
        circadian.circadian_total_rollbacks > 0 if reject else circadian.circadian_total_splits > 0
    )
    assert (random.random(), float(np.random.random()), float(torch.rand(()))) == expected_draw


@pytest.mark.parametrize(
    ("protocol", "workers"),
    [
        (VISION_VALIDATION_UNMATCHED_PROTOCOL, 0),
        (VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL, 2),
    ],
)
def test_earlier_vision_file_resume_replays_shared_augmented_loader(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    tiny_vision_backbone: None,
    protocol: str,
    workers: int,
) -> None:
    config = replace(_config(), protocol_id=protocol)
    original_build = vision._build_benchmark_loaders

    def build(config: ResNet50BenchmarkConfig) -> Any:
        loaders = original_build(config)
        train_loader = torch.utils.data.DataLoader(
            StochasticVisionDataset(loaders.train_loader.dataset),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=workers,
            generator=torch.Generator().manual_seed(99),
        )
        return replace(loaders, train_loader=train_loader)

    monkeypatch.setattr(vision, "_build_benchmark_loaders", build)
    random.seed(941)
    np.random.seed(942)
    torch.manual_seed(943)
    control = vision.run_resnet50_benchmark(
        config,
        checkpoint_store=TrustedLocalVisionCheckpointStore(tmp_path / "control.ckpt"),
    )
    expected_draw = (random.random(), float(np.random.random()), float(torch.rand(())))

    store = TrustedLocalVisionCheckpointStore(tmp_path / "vision.ckpt")
    random.seed(941)
    np.random.seed(942)
    torch.manual_seed(943)
    with pytest.raises(InterruptedAfterSave):
        vision.run_resnet50_benchmark(
            config,
            checkpoint_store=InterruptAtActiveStage(store, "wake", 0),
        )
    resumed = vision.run_resnet50_benchmark(
        config,
        checkpoint_store=store,
        resume_from_checkpoint=True,
    )
    assert resumed.trained_model_hashes == control.trained_model_hashes
    _same_learning_reports(resumed, control)
    assert (random.random(), float(np.random.random()), float(torch.rand(()))) == expected_draw


@pytest.mark.parametrize(
    ("protocol", "damage"),
    [
        (VISION_VALIDATION_UNMATCHED_PROTOCOL, "shared"),
        (VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL, "cursor"),
        (VISION_VALIDATION_UNMATCHED_PROTOCOL, "replay"),
        (VISION_VALIDATION_UNMATCHED_PROTOCOL, "data"),
        (VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL, "data"),
        (VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL, "checksum"),
    ],
)
def test_earlier_vision_file_resume_rejects_incompatible_state_before_training(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    tiny_vision_backbone: None,
    protocol: str,
    damage: str,
) -> None:
    config = replace(_sleep_config(), protocol_id=protocol)
    store = TrustedLocalVisionCheckpointStore(tmp_path / "vision.ckpt")
    with pytest.raises(InterruptedAfterSave):
        vision.run_resnet50_benchmark(
            config,
            checkpoint_store=InterruptAtActiveStage(store, "wake", 0),
        )
    saved = store.load()
    active = saved.active_circadian
    assert active is not None
    if damage == "shared":
        store.save(
            replace(
                saved,
                shared_train_generator_state=torch.Generator().manual_seed(99).get_state(),
            )
        )
    elif damage == "cursor":
        store.save(
            replace(
                saved,
                active_circadian=replace(
                    active,
                    loader_state=replace(active.loader_state, next_batch_index=0),
                ),
            )
        )
    elif damage == "replay":
        store.save(
            replace(
                saved,
                active_circadian=replace(
                    active,
                    loader_state=replace(
                        active.loader_state,
                        epoch_entry_generator_state=(
                            torch.Generator().manual_seed(999).get_state()
                        ),
                    ),
                ),
            )
        )
    elif damage == "data":
        original_build = vision._build_benchmark_loaders

        def changed_data(config: ResNet50BenchmarkConfig) -> Any:
            loaders = original_build(config)
            labels = loaders.guard_loader.dataset.labels
            labels[0] = (labels[0] + 1) % config.num_classes
            return loaders

        monkeypatch.setattr(vision, "_build_benchmark_loaders", changed_data)
    else:
        path = tmp_path / "vision.ckpt"
        content = path.read_bytes()
        path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))

    if damage != "replay":
        monkeypatch.setattr(
            vision,
            "_train_circadian",
            lambda *args, **kwargs: pytest.fail("training began before shared-stream preflight"),
        )
    before_python = random.getstate()
    before_numpy = np.random.get_state()
    before_torch = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="checkpoint"):
        vision.run_resnet50_benchmark(
            config,
            checkpoint_store=store,
            resume_from_checkpoint=True,
        )
    assert random.getstate() == before_python
    np.testing.assert_equal(np.random.get_state(), before_numpy)
    assert torch.equal(torch.get_rng_state(), before_torch)


def test_vision_file_resume_rejects_changed_development_data_before_training(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    tiny_vision_backbone: None,
) -> None:
    config = _config()
    store = TrustedLocalVisionCheckpointStore(tmp_path / "vision.ckpt")
    with pytest.raises(InterruptedAfterSave):
        vision.run_resnet50_benchmark(config, checkpoint_store=InterruptingStore(store))

    original_build = vision._build_benchmark_loaders

    def changed_data(config: ResNet50BenchmarkConfig) -> Any:
        loaders = original_build(config)
        labels = loaders.train_loader.dataset.labels
        labels[0] = (labels[0] + 1) % config.num_classes
        return loaders

    monkeypatch.setattr(vision, "_build_benchmark_loaders", changed_data)
    monkeypatch.setattr(
        vision,
        "_train_seeded_variant",
        lambda *args: pytest.fail("training started before data validation"),
    )
    with pytest.raises(ValueError, match="data"):
        vision.run_resnet50_benchmark(
            config,
            checkpoint_store=store,
            resume_from_checkpoint=True,
        )


@pytest.mark.parametrize("damage", ["samples", "steps", "metric", "hash", "model"])
def test_vision_file_resume_rejects_bad_completed_baseline_before_training(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    tiny_vision_backbone: None,
    damage: str,
) -> None:
    config = _config()
    store = TrustedLocalVisionCheckpointStore(tmp_path / "vision.ckpt")
    with pytest.raises(InterruptedAfterSave):
        vision.run_resnet50_benchmark(config, checkpoint_store=InterruptingStore(store))
    saved = store.load()
    baseline = saved.completed_outcomes[0]
    if damage == "samples":
        changed = replace(baseline, seen_samples=baseline.seen_samples - 1)
    elif damage == "steps":
        changed = replace(baseline, step_times_ms=baseline.step_times_ms[:-1])
    elif damage == "metric":
        changed = replace(baseline, validation_accuracy=float("nan"))
    elif damage == "model":
        model = baseline.model
        state = model.state
        key = next(iter(state.output_state))
        changed = replace(
            baseline,
            model=replace(
                model,
                state=replace(
                    state,
                    output_state={**state.output_state, key: torch.zeros(1)},
                ),
            ),
        )
    else:
        store.save(replace(saved, completed_hashes=("wrong",)))
        changed = baseline
    if damage != "hash":
        store.save(replace(saved, completed_outcomes=(changed,)))

    monkeypatch.setattr(
        vision,
        "_train_seeded_variant",
        lambda *args, **kwargs: pytest.fail("training started before baseline preflight"),
    )
    before_python = random.getstate()
    before_numpy = np.random.get_state()
    before_torch = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="checkpoint"):
        vision.run_resnet50_benchmark(
            config,
            checkpoint_store=store,
            resume_from_checkpoint=True,
        )
    assert random.getstate() == before_python
    np.testing.assert_equal(np.random.get_state(), before_numpy)
    assert torch.equal(torch.get_rng_state(), before_torch)


@pytest.mark.parametrize(
    ("stage", "reject", "model_order"),
    [
        ("wake", False, None),
        ("before_sleep", False, None),
        ("after_sleep", False, None),
        ("after_sleep", True, None),
        ("wake", False, ("circadian", "predictive", "backprop")),
    ],
)
def test_vision_file_resume_across_guarded_sleep_matches_uninterrupted_state(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    tiny_vision_backbone: None,
    stage: str,
    reject: bool,
    model_order: tuple[str, ...] | None,
) -> None:
    config = _sleep_config(reject)
    monkeypatch.setattr(
        vision,
        "_compute_pc_metrics",
        lambda _torch, model, _loader, _device, max_batches=None: (
            0.8,
            2.0 if reject and model.head.hidden_dim > 4 else 0.5,
        ),
    )
    random.seed(811)
    np.random.seed(812)
    torch.manual_seed(813)
    control = vision.run_resnet50_benchmark(config, model_order=model_order)
    expected_draw = (random.random(), float(np.random.random()), float(torch.rand(())))

    store = TrustedLocalVisionCheckpointStore(tmp_path / "vision.ckpt")
    random.seed(811)
    np.random.seed(812)
    torch.manual_seed(813)
    with pytest.raises(InterruptedAfterSave):
        vision.run_resnet50_benchmark(
            config,
            model_order=model_order,
            checkpoint_store=InterruptAtActiveStage(store, stage, 0 if stage == "wake" else 1),
        )
    resumed = vision.run_resnet50_benchmark(
        config,
        model_order=model_order,
        checkpoint_store=store,
        resume_from_checkpoint=True,
    )
    assert resumed.training_order == control.training_order
    assert resumed.trained_model_hashes == control.trained_model_hashes
    _same_learning_reports(resumed, control)
    circadian = next(
        report
        for report in resumed.reports
        if report.model_name == "CircadianPredictiveCodingResNet50"
    )
    assert circadian.circadian_sleep_attempts > 0
    assert (
        circadian.circadian_total_rollbacks > 0 if reject else circadian.circadian_total_splits > 0
    )
    assert (random.random(), float(np.random.random()), float(torch.rand(()))) == expected_draw


@pytest.mark.parametrize("workers", [0, 2])
def test_vision_file_resume_replays_augmented_train_loader(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    tiny_vision_backbone: None,
    workers: int,
) -> None:
    config = _config()
    order = ("circadian", "predictive", "backprop")
    original_build = vision._build_benchmark_loaders

    def build(config: ResNet50BenchmarkConfig) -> Any:
        loaders = original_build(config)
        train_loader = torch.utils.data.DataLoader(
            StochasticVisionDataset(loaders.train_loader.dataset),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=workers,
            generator=torch.Generator().manual_seed(99),
        )
        return replace(loaders, train_loader=train_loader)

    monkeypatch.setattr(vision, "_build_benchmark_loaders", build)
    random.seed(921)
    np.random.seed(922)
    torch.manual_seed(923)
    control = vision.run_resnet50_benchmark(config, model_order=order)
    expected_draw = (random.random(), float(np.random.random()), float(torch.rand(())))

    store = TrustedLocalVisionCheckpointStore(tmp_path / "vision.ckpt")
    random.seed(921)
    np.random.seed(922)
    torch.manual_seed(923)
    with pytest.raises(InterruptedAfterSave):
        vision.run_resnet50_benchmark(
            config,
            model_order=order,
            checkpoint_store=InterruptAtActiveStage(store, "wake", 0),
        )
    resumed = vision.run_resnet50_benchmark(
        config,
        model_order=order,
        checkpoint_store=store,
        resume_from_checkpoint=True,
    )
    assert resumed.trained_model_hashes == control.trained_model_hashes
    _same_learning_reports(resumed, control)
    assert (random.random(), float(np.random.random()), float(torch.rand(()))) == expected_draw


@pytest.mark.parametrize(
    "damage",
    ["cursor", "counter", "classifier", "rng", "checksum", "order", "config", "data"],
)
def test_vision_file_resume_rejects_bad_active_progress_before_training_or_rng_change(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    tiny_vision_backbone: None,
    damage: str,
) -> None:
    config = _sleep_config()
    order = ("circadian", "predictive", "backprop")
    store = TrustedLocalVisionCheckpointStore(tmp_path / "vision.ckpt")
    with pytest.raises(InterruptedAfterSave):
        vision.run_resnet50_benchmark(
            config,
            model_order=order,
            checkpoint_store=InterruptAtActiveStage(store, "wake", 0),
        )

    saved = store.load()
    active = saved.active_circadian
    assert active is not None
    if damage == "cursor":
        store.save(
            replace(
                saved,
                active_circadian=replace(
                    active,
                    loader_state=replace(active.loader_state, next_batch_index=0),
                ),
            )
        )
    elif damage == "counter":
        store.save(
            replace(
                saved,
                active_circadian=replace(active, wake_batches=active.wake_batches + 1),
            )
        )
    elif damage == "classifier":
        classifier = active.classifier_state
        key = next(iter(classifier.backbone_state))
        store.save(
            replace(
                saved,
                active_circadian=replace(
                    active,
                    classifier_state=replace(
                        classifier,
                        backbone_state={**classifier.backbone_state, key: torch.zeros(1)},
                    ),
                ),
            )
        )
    elif damage == "rng":
        store.save(
            replace(
                saved,
                active_circadian=replace(
                    active,
                    outer_entry_torch_state=torch.empty(0, dtype=torch.uint8),
                ),
            )
        )
    elif damage == "checksum":
        path = tmp_path / "vision.ckpt"
        content = path.read_bytes()
        path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    elif damage == "order":
        order = ("backprop", "predictive", "circadian")
    elif damage == "config":
        config = replace(config, circadian_learning_rate=config.circadian_learning_rate / 2)
    else:
        original_build = vision._build_benchmark_loaders

        def changed_data(config: ResNet50BenchmarkConfig) -> Any:
            loaders = original_build(config)
            labels = loaders.guard_loader.dataset.labels
            labels[0] = (labels[0] + 1) % config.num_classes
            return loaders

        monkeypatch.setattr(vision, "_build_benchmark_loaders", changed_data)

    monkeypatch.setattr(
        vision,
        "_train_seeded_variant",
        lambda *args, **kwargs: pytest.fail("training started before checkpoint preflight"),
    )
    before_python = random.getstate()
    before_numpy = np.random.get_state()
    before_torch = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="checkpoint"):
        vision.run_resnet50_benchmark(
            config,
            model_order=order,
            checkpoint_store=store,
            resume_from_checkpoint=True,
        )
    assert random.getstate() == before_python
    np.testing.assert_equal(np.random.get_state(), before_numpy)
    assert torch.equal(torch.get_rng_state(), before_torch)
