from __future__ import annotations

import random
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")
from torchvision import transforms

from src.app import resnet50_benchmark
from src.app.resnet50_benchmark import (
    ResNet50BenchmarkConfig,
    VISION_SEEDED_UNMATCHED_PROTOCOL,
    VISION_VALIDATION_UNMATCHED_PROTOCOL,
    benchmark_validation_candidate,
    format_resnet50_benchmark_result,
    run_resnet50_benchmark,
)
from src.core.resnet50_variants import CircadianPredictiveCodingHead, PredictiveCodingHead
from src.infra.vision_datasets import SyntheticVisionDatasetConfig, build_synthetic_vision_dataloaders


class StochasticImageDataset:
    """Picklable stochastic images for spawned DataLoader workers."""

    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> tuple[Any, int]:
        import torch

        channels = torch.tensor([
            float(index), float(torch.rand(())),
            float(np.random.random()), random.random(),
        ])
        return channels[:, None, None].expand(4, 4, 4).clone(), index % 3


class TorchvisionViewDataset:
    """Picklable, asymmetric tensor images with stochastic torchvision views."""

    def __init__(self) -> None:
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(), transforms.RandomCrop((6, 7)),
        ])

    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> tuple[Any, int]:
        import torch

        image = torch.arange(3 * 8 * 10, dtype=torch.float32).reshape(3, 8, 10)
        return self.transform(image + float(index * 1_000)), index % 3


def test_should_run_resnet50_benchmark_and_return_three_reports() -> None:
    config = ResNet50BenchmarkConfig(
        train_samples=48,
        test_samples=24,
        num_classes=6,
        image_size=64,
        batch_size=8,
        dataset_difficulty="hard",
        dataset_noise_std=0.08,
        epochs=1,
        seed=5,
        device="cpu",
        target_accuracy=None,
        inference_batches=3,
        evaluation_batches=1,
        warmup_batches=1,
        backprop_freeze_backbone=True,
        predictive_head_hidden_dim=64,
        circadian_head_hidden_dim=64,
        circadian_min_hidden_dim=32,
        circadian_max_hidden_dim=128,
    )

    result = run_resnet50_benchmark(config)
    formatted = format_resnet50_benchmark_result(result)
    assert len(result.reports) == 3
    assert "Protocol: vision_guard_separated_unmatched_v2" in formatted
    assert "Repeated stopping and rollback decisions use guard examples" in formatted
    assert set(result.split_hashes) == {"train", "validation", "guard", "test"}
    assert "legacy unmatched heads and backbone state" in formatted
    assert "Legacy linear-head reference: BackpropResNet50." in formatted
    assert "track=unmatched_reference, backbone_trainable=False" in formatted
    assert "vs legacy linear backprop (descriptive)" in formatted
    assert {report.model_name for report in result.reports} == {
        "BackpropResNet50",
        "PredictiveCodingResNet50",
        "CircadianPredictiveCodingResNet50",
    }

    for report in result.reports:
        assert report.benchmark_track == "unmatched_reference"
        assert report.backbone_pretraining == "none"
        assert report.backbone_trainable is False
        assert report.head_type in {
            "linear", "predictive_coding", "circadian_predictive_coding",
        }
        assert report.train_seconds >= 0.0
        assert report.train_samples_per_second >= 0.0
        assert report.inference_samples_per_second >= 0.0
        assert 0.0 <= report.validation_accuracy <= 1.0
        assert 0.0 <= report.test_accuracy <= 1.0
        assert report.total_parameters > 0
        assert report.trainable_parameters > 0
        assert report.circadian_total_rollbacks >= 0
        assert report.final_cross_entropy is not None
        if report.model_name != "BackpropResNet50":
            assert report.final_energy is not None
            assert report.training_energy_id == "torch_pc_half_mean_output_error_sq_plus_half_mean_hidden_error_sq_v1"
        else:
            assert report.training_energy_id is None
    assert "last training diagnostic [torch_pc_half_mean_output_error_sq_plus_half_mean_hidden_error_sq_v1]" in formatted


def test_seeded_unmatched_reference_is_invariant_to_model_order_on_cpu() -> None:
    config = ResNet50BenchmarkConfig(
        protocol_id=VISION_SEEDED_UNMATCHED_PROTOCOL,
        train_samples=8, guard_samples=8, validation_samples=8, test_samples=8,
        num_classes=3, image_size=32, batch_size=4, epochs=1, seed=73,
        device="cpu", target_accuracy=None, inference_batches=1, warmup_batches=0,
        backprop_freeze_backbone=True, backbone_weights="none",
        predictive_head_hidden_dim=16, predictive_inference_steps=2,
        circadian_head_hidden_dim=16, circadian_min_hidden_dim=16,
        circadian_max_hidden_dim=32, circadian_inference_steps=2,
        circadian_sleep_interval=1, circadian_force_sleep=True,
        circadian_sleep_warmup_steps=0,
    )
    forward = run_resnet50_benchmark(config)
    reverse = run_resnet50_benchmark(
        config, model_order=("circadian", "predictive", "backprop"),
    )

    assert forward.training_order == ("backprop", "predictive", "circadian")
    assert reverse.training_order == tuple(reversed(forward.training_order))
    assert forward.split_hashes == reverse.split_hashes
    assert forward.trained_model_hashes is not None
    assert reverse.trained_model_hashes is not None
    assert forward.trained_model_hashes == reverse.trained_model_hashes
    assert set(forward.trained_model_hashes) == {
        "BackpropResNet50", "PredictiveCodingResNet50",
        "CircadianPredictiveCodingResNet50",
    }
    assert "Trained model hashes:" in format_resnet50_benchmark_result(forward)
    forward_reports = {report.model_name: report for report in forward.reports}
    reverse_reports = {report.model_name: report for report in reverse.reports}
    for name in forward_reports:
        assert forward_reports[name].validation_accuracy == pytest.approx(
            reverse_reports[name].validation_accuracy, abs=1e-7,
        )
        assert forward_reports[name].test_accuracy == pytest.approx(
            reverse_reports[name].test_accuracy, abs=1e-7,
        )
        assert forward_reports[name].final_cross_entropy == pytest.approx(
            reverse_reports[name].final_cross_entropy, abs=1e-7,
        )


def test_seeded_reference_rejects_invalid_execution_order_before_loading_data() -> None:
    config = replace(
        ResNet50BenchmarkConfig(), protocol_id=VISION_SEEDED_UNMATCHED_PROTOCOL,
    )
    with pytest.raises(ValueError, match="permutation"):
        run_resnet50_benchmark(
            config, model_order=("backprop", "backprop", "predictive"),
        )


def test_trained_model_hash_covers_adaptive_state_and_structural_rng() -> None:
    torch = pytest.importorskip("torch")
    circadian_head = CircadianPredictiveCodingHead(
        feature_dim=4, hidden_dim=16, num_classes=3,
        device=torch.device("cpu"), seed=73,
    )
    circadian = SimpleNamespace(backbone=torch.nn.Identity(), head=circadian_head)
    initial = resnet50_benchmark._hash_trained_model(circadian)
    circadian_head._chemical[0] = 0.25
    chemical_changed = resnet50_benchmark._hash_trained_model(circadian)
    assert chemical_changed != initial
    torch.rand((1,), generator=circadian_head._split_generator)
    assert resnet50_benchmark._hash_trained_model(circadian) != chemical_changed

    predictive_head = PredictiveCodingHead(
        feature_dim=4, hidden_dim=16, num_classes=3,
        device=torch.device("cpu"), seed=73,
    )
    predictive = SimpleNamespace(backbone=torch.nn.Identity(), head=predictive_head)
    initial_predictive = resnet50_benchmark._hash_trained_model(predictive)
    predictive_head._traffic_sum[0] = 0.25
    assert resnet50_benchmark._hash_trained_model(predictive) != initial_predictive


def test_seeded_epoch_loader_replays_shuffle_and_random_views() -> None:
    torch = pytest.importorskip("torch")

    class RandomViewDataset:
        def __len__(self) -> int:
            return 6

        def __getitem__(self, index: int) -> tuple[Any, Any]:
            return torch.tensor([float(index), torch.rand(()).item()]), index % 3

    generator = torch.Generator().manual_seed(3)
    loader = torch.utils.data.DataLoader(
        RandomViewDataset(), batch_size=2, shuffle=True, generator=generator,
    )

    def collect_epochs() -> list[list[Any]]:
        replay = resnet50_benchmark._SeededEpochTrainLoader(torch, loader, 79)
        return [
            [images.clone() for images, _ in replay]
            for _ in range(2)
        ]

    before = torch.random.get_rng_state().clone()
    first = collect_epochs()
    assert torch.equal(before, torch.random.get_rng_state())
    _ = torch.rand((101,))
    second = collect_epochs()

    for first_epoch, second_epoch in zip(first, second):
        assert all(torch.equal(left, right) for left, right in zip(first_epoch, second_epoch))
    assert any(not torch.equal(left, right) for left, right in zip(first[0], first[1]))


def test_seeded_epoch_loader_replays_multiworker_stochastic_images() -> None:
    torch = pytest.importorskip("torch")
    loader = torch.utils.data.DataLoader(
        StochasticImageDataset(), batch_size=2, shuffle=True,
        num_workers=2, generator=torch.Generator().manual_seed(3),
    )

    def collect_epochs() -> list[list[Any]]:
        replay = resnet50_benchmark._SeededEpochTrainLoader(torch, loader, 79)
        return [[images.clone() for images, _ in replay] for _ in range(2)]

    before = torch.random.get_rng_state().clone()
    first = collect_epochs()
    assert torch.equal(before, torch.random.get_rng_state())
    _ = torch.rand((101,))
    _ = np.random.random(101)
    _ = [random.random() for _ in range(101)]
    second = collect_epochs()

    for first_epoch, second_epoch in zip(first, second):
        assert all(torch.equal(left, right) for left, right in zip(first_epoch, second_epoch))
    assert any(not torch.equal(left, right) for left, right in zip(first[0], first[1]))


def test_seeded_epoch_loader_replays_torchvision_transforms_with_workers() -> None:
    torch = pytest.importorskip("torch")
    loader = torch.utils.data.DataLoader(
        TorchvisionViewDataset(), batch_size=2, shuffle=True,
        num_workers=2, generator=torch.Generator().manual_seed(3),
    )

    def collect_epochs() -> list[list[Any]]:
        replay = resnet50_benchmark._SeededEpochTrainLoader(torch, loader, 131)
        return [[images.clone() for images, _ in replay] for _ in range(2)]

    first = collect_epochs()
    _ = torch.rand((101,))
    second = collect_epochs()

    for first_epoch, second_epoch in zip(first, second):
        assert all(torch.equal(left, right) for left, right in zip(first_epoch, second_epoch))
    assert any(not torch.equal(left, right) for left, right in zip(first[0], first[1]))


@pytest.mark.parametrize(
    ("num_workers", "dataset_type"),
    [(0, TorchvisionViewDataset), (0, StochasticImageDataset), (2, StochasticImageDataset)],
)
def test_seeded_epoch_loader_resumes_mid_epoch_with_identical_future_batches(
    num_workers: int,
    dataset_type: type[Any],
) -> None:
    torch = pytest.importorskip("torch")

    def build_loader() -> Any:
        return torch.utils.data.DataLoader(
            dataset_type(),
            batch_size=2,
            shuffle=True,
            num_workers=num_workers,
            generator=torch.Generator().manual_seed(3),
        )

    original = resnet50_benchmark._SeededEpochTrainLoader(torch, build_loader(), 131)
    original_batches = iter(original)
    next(original_batches)
    _ = (torch.rand(()), np.random.random(), random.random())
    next(original_batches)
    _ = (torch.rand(()), np.random.random(), random.random())
    cursor = original.snapshot_state()
    if num_workers == 2:
        changed_prefetch = torch.utils.data.DataLoader(
            dataset_type(),
            batch_size=2,
            shuffle=True,
            num_workers=2,
            prefetch_factor=3,
            generator=torch.Generator().manual_seed(3),
        )
        with pytest.raises(ValueError, match="configuration"):
            resnet50_benchmark._SeededEpochTrainLoader(
                torch,
                changed_prefetch,
                131,
            ).restore_state(cursor)

    def collect_remaining(batches: Any) -> list[Any]:
        collected = []
        for batch in batches:
            collected.append(batch)
            _ = (torch.rand(()), np.random.random(), random.random())
        return collected

    expected_remainder = collect_remaining(original_batches)
    expected_next_epoch = list(original)
    expected_draw = (float(torch.rand(())), float(np.random.random()), random.random())

    _ = torch.rand((101,))
    _ = np.random.random(101)
    _ = [random.random() for _ in range(101)]
    resumed = resnet50_benchmark._SeededEpochTrainLoader(torch, build_loader(), 131)
    resumed.restore_state(cursor)
    actual_remainder = collect_remaining(resumed)
    actual_next_epoch = list(resumed)
    actual_draw = (float(torch.rand(())), float(np.random.random()), random.random())

    for expected, actual in zip(expected_remainder, actual_remainder, strict=True):
        assert torch.equal(expected[0], actual[0])
        assert torch.equal(expected[1], actual[1])
    for expected, actual in zip(expected_next_epoch, actual_next_epoch, strict=True):
        assert torch.equal(expected[0], actual[0])
        assert torch.equal(expected[1], actual[1])
    assert actual_draw == expected_draw


def test_seeded_epoch_loader_rejects_invalid_resume_cursor_before_replay() -> None:
    torch = pytest.importorskip("torch")
    loader = torch.utils.data.DataLoader(
        TorchvisionViewDataset(),
        batch_size=2,
        shuffle=True,
        generator=torch.Generator().manual_seed(3),
    )
    original = resnet50_benchmark._SeededEpochTrainLoader(torch, loader, 131)
    batches = iter(original)
    next(batches)
    cursor = original.snapshot_state()
    batches.close()
    resumed = resnet50_benchmark._SeededEpochTrainLoader(torch, loader, 131)
    before = loader.generator.get_state().clone()

    with pytest.raises(ValueError, match="batch index"):
        resumed.restore_state(replace(cursor, next_batch_index=len(loader) + 1))
    with pytest.raises(ValueError, match="generator state"):
        resumed.restore_state(replace(cursor, generator_state=torch.zeros(1)))
    with pytest.raises(ValueError, match="configuration"):
        resumed.restore_state(replace(cursor, batch_size=1))
    assert resumed.epoch == 0
    assert torch.equal(loader.generator.get_state(), before)

    persistent_loader = torch.utils.data.DataLoader(
        StochasticImageDataset(),
        batch_size=2,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        generator=torch.Generator().manual_seed(3),
    )
    with pytest.raises(ValueError, match="ordered, resettable"):
        resnet50_benchmark._SeededEpochTrainLoader(
            torch,
            persistent_loader,
            131,
        ).snapshot_state()

    changed_generator = torch.Generator().manual_seed(999).get_state()
    resumed.restore_state(replace(cursor, generator_state=changed_generator))
    before_torch = torch.get_rng_state().clone()
    before_python = random.getstate()
    before_numpy = np.random.get_state()
    with pytest.raises(ValueError, match="changed during replay"):
        next(iter(resumed))
    assert resumed.epoch == 0
    assert torch.equal(torch.get_rng_state(), before_torch)
    assert random.getstate() == before_python
    actual_numpy = np.random.get_state()
    assert isinstance(before_numpy, tuple) and isinstance(actual_numpy, tuple)
    assert actual_numpy[0] == before_numpy[0]
    assert np.array_equal(actual_numpy[1], before_numpy[1])
    assert actual_numpy[2:] == before_numpy[2:]
    assert torch.equal(loader.generator.get_state(), before)


def test_seeded_reference_keeps_final_loader_sealed_until_all_models_train(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_build = resnet50_benchmark._build_benchmark_loaders
    original_train = resnet50_benchmark._train_seeded_variant
    trained: list[str] = []

    class SealedTestLoader:
        def __init__(self, original: Any) -> None:
            self.original = original

        def __iter__(self) -> Any:
            if len(trained) != 3:
                raise AssertionError("Final test opened before all seeded models trained")
            return iter(self.original)

    def build_loaders(config: ResNet50BenchmarkConfig) -> Any:
        loaders = original_build(config)
        return replace(loaders, test_loader=SealedTestLoader(loaders.test_loader))

    def train(variant: str, *args: Any) -> Any:
        outcome = original_train(variant, *args)
        trained.append(variant)
        return outcome

    monkeypatch.setattr(resnet50_benchmark, "_build_benchmark_loaders", build_loaders)
    monkeypatch.setattr(resnet50_benchmark, "_train_seeded_variant", train)
    config = ResNet50BenchmarkConfig(
        protocol_id=VISION_SEEDED_UNMATCHED_PROTOCOL,
        train_samples=8, guard_samples=8, validation_samples=8, test_samples=8,
        num_classes=3, image_size=32, batch_size=4, epochs=1, seed=79,
        device="cpu", target_accuracy=None, inference_batches=1, warmup_batches=0,
        backprop_freeze_backbone=True, predictive_head_hidden_dim=16,
        predictive_inference_steps=2, circadian_head_hidden_dim=16,
        circadian_min_hidden_dim=16, circadian_max_hidden_dim=32,
        circadian_inference_steps=2, circadian_sleep_interval=0,
    )
    result = run_resnet50_benchmark(config)

    assert trained == ["backprop", "predictive", "circadian"]
    assert len(result.reports) == 3


def test_seeded_validation_candidate_uses_train_and_guard_roles_only() -> None:
    torch = pytest.importorskip("torch")
    config = ResNet50BenchmarkConfig(
        protocol_id=VISION_SEEDED_UNMATCHED_PROTOCOL,
        train_samples=8, guard_samples=8, validation_samples=8, test_samples=8,
        num_classes=3, image_size=32, batch_size=4, epochs=1, seed=83,
        device="cpu", target_accuracy=None, inference_batches=1, warmup_batches=0,
        backprop_freeze_backbone=True,
    )
    loaders = build_synthetic_vision_dataloaders(
        SyntheticVisionDatasetConfig(
            train_samples=8, guard_samples=8, validation_samples=8,
            test_samples=8, num_classes=3, image_size=32, batch_size=4, seed=83,
        )
    )

    report = benchmark_validation_candidate(
        variant="backprop", torch=torch, device=torch.device("cpu"),
        loaders=resnet50_benchmark._training_loaders(loaders), config=config,
    )

    assert report.model_name == "BackpropResNet50"
    assert report.benchmark_track == "unmatched_reference"
    assert 0.0 <= report.validation_accuracy <= 1.0


def test_should_validate_new_sleep_config_values() -> None:
    config = ResNet50BenchmarkConfig(
        circadian_sleep_energy_window=1,
    )

    with pytest.raises(ValueError):
        _ = run_resnet50_benchmark(config)


def test_should_validate_rollback_metric_name() -> None:
    config = ResNet50BenchmarkConfig(circadian_sleep_rollback_metric="invalid")

    with pytest.raises(ValueError):
        _ = run_resnet50_benchmark(config)


def test_should_validate_hidden_dim_bounds() -> None:
    config = ResNet50BenchmarkConfig(
        circadian_head_hidden_dim=64,
        circadian_min_hidden_dim=96,
    )

    with pytest.raises(ValueError):
        _ = run_resnet50_benchmark(config)


def test_should_validate_dataset_name() -> None:
    config = ResNet50BenchmarkConfig(dataset_name="invalid")

    with pytest.raises(ValueError):
        _ = run_resnet50_benchmark(config)


def test_should_validate_num_classes_for_torchvision_dataset() -> None:
    config = ResNet50BenchmarkConfig(dataset_name="cifar100", num_classes=10)

    with pytest.raises(ValueError):
        _ = run_resnet50_benchmark(config)


def test_should_reject_unknown_vision_protocol() -> None:
    with pytest.raises(ValueError, match="Unknown vision benchmark protocol"):
        run_resnet50_benchmark(ResNet50BenchmarkConfig(protocol_id="unknown"))


def test_should_keep_explicit_legacy_vision_split_route() -> None:
    config = ResNet50BenchmarkConfig(
        protocol_id=VISION_VALIDATION_UNMATCHED_PROTOCOL,
        train_samples=8, validation_samples=8, test_samples=8,
        num_classes=3, image_size=32, batch_size=4,
    )
    loaders = resnet50_benchmark._build_benchmark_loaders(config)

    assert loaders.guard_loader is loaders.validation_loader
    assert set(loaders.split_hashes) == {"train", "validation", "test"}


@pytest.mark.parametrize("variant", ["backprop", "predictive", "circadian"])
def test_validation_candidate_cannot_read_final_test_loader(variant: str) -> None:
    torch = pytest.importorskip("torch")
    config = ResNet50BenchmarkConfig(
        train_samples=8, validation_samples=8, test_samples=8,
        num_classes=3, image_size=32, batch_size=4, epochs=1, seed=17,
        device="cpu", target_accuracy=None, inference_batches=1,
        evaluation_batches=1, warmup_batches=0,
        predictive_head_hidden_dim=32, circadian_head_hidden_dim=32,
        circadian_min_hidden_dim=16, circadian_max_hidden_dim=64,
    )
    loaders = build_synthetic_vision_dataloaders(
        SyntheticVisionDatasetConfig(
            train_samples=8, validation_samples=8, guard_samples=8, test_samples=8,
            num_classes=3, image_size=32, batch_size=4, seed=17,
        )
    )

    class SealedLoaders:
        train_loader = loaders.train_loader
        validation_loader = loaders.validation_loader
        guard_loader = loaders.guard_loader
        num_classes = loaders.num_classes

        @property
        def test_loader(self) -> Any:
            raise AssertionError("Candidate evaluation accessed the final test loader")

    report = benchmark_validation_candidate(
        variant=variant, torch=torch, device=torch.device("cpu"),
        loaders=resnet50_benchmark._training_loaders(SealedLoaders()), config=config,
    )
    assert 0 <= report.validation_accuracy <= 1
    assert report.validation_cross_entropy >= 0
    assert not hasattr(report, "test_accuracy")


def test_should_use_disjoint_guard_for_stopping_and_sleep_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    loaders = build_synthetic_vision_dataloaders(
        SyntheticVisionDatasetConfig(
            train_samples=8,
            validation_samples=8,
            guard_samples=8,
            test_samples=8,
            num_classes=3,
            image_size=32,
            batch_size=4,
            seed=17,
        )
    )
    monkeypatch.setattr(resnet50_benchmark, "_build_benchmark_loaders", lambda config: loaders)
    observed: list[tuple[str, object, int | None]] = []
    original_backprop = resnet50_benchmark._compute_backprop_metrics
    original_pc = resnet50_benchmark._compute_pc_metrics

    def check_backprop(torch_module: object, model: object, loader: object,
                       device: object, max_batches: int | None) -> tuple[float, float]:
        observed.append(("backprop", loader, max_batches))
        return original_backprop(torch_module, model, loader, device, max_batches)

    def check_pc(torch_module: object, model: object, loader: object,
                 device: object, max_batches: int | None) -> tuple[float, float]:
        observed.append(("pc", loader, max_batches))
        return original_pc(torch_module, model, loader, device, max_batches)

    monkeypatch.setattr(resnet50_benchmark, "_compute_backprop_metrics", check_backprop)
    monkeypatch.setattr(resnet50_benchmark, "_compute_pc_metrics", check_pc)
    result = run_resnet50_benchmark(
        ResNet50BenchmarkConfig(
            train_samples=8,
            validation_samples=8,
            test_samples=8,
            num_classes=3,
            image_size=32,
            batch_size=4,
            epochs=1,
            seed=17,
            device="cpu",
            target_accuracy=None,
            evaluation_batches=1,
            inference_batches=1,
            warmup_batches=0,
            backprop_freeze_backbone=True,
            predictive_head_hidden_dim=32,
            circadian_head_hidden_dim=32,
            circadian_min_hidden_dim=16,
            circadian_max_hidden_dim=64,
            circadian_sleep_interval=1,
            circadian_force_sleep=True,
            circadian_min_sleep_steps=0,
        )
    )

    assert result.split_hashes == dict(loaders.split_hashes)
    assert len(observed) == 11
    assert all(loader is loaders.guard_loader for _, loader, batches in observed
               if batches is not None)
    assert all(loader is not loaders.test_loader or batches is None
               for _, loader, batches in observed)
    assert sum(loader is loaders.test_loader for _, loader, _ in observed) == 3
    assert sum(loader is loaders.validation_loader for _, loader, _ in observed) == 3
    assert sum(loader is loaders.guard_loader for _, loader, _ in observed) == 5
    assert torch.device(result.device).type == "cpu"


def test_should_restore_training_mode_after_backprop_validation() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel:
        def __init__(self) -> None:
            self.backbone = torch.nn.Linear(2, 2)
            self.classifier = torch.nn.Linear(2, 2)

        def forward_logits(self, images: object) -> object:
            return self.classifier(self.backbone(images))

    model = TinyModel()
    model.backbone.train()
    model.classifier.train()
    loader = [(torch.zeros((2, 2)), torch.tensor([0, 1]))]
    resnet50_benchmark._compute_backprop_metrics(
        torch, model, loader, torch.device("cpu"), max_batches=1
    )
    assert model.backbone.training
    assert model.classifier.training


@pytest.mark.parametrize("changed_role", ["validation", "test"])
def test_should_keep_circadian_state_when_outer_labels_change(
    changed_role: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    original_model_class = resnet50_benchmark.CircadianPredictiveCodingResNet50Classifier
    created_models: list[Any] = []

    def capture_model(**kwargs: Any) -> Any:
        model = original_model_class(**kwargs)
        created_models.append(model)
        return model

    monkeypatch.setattr(
        resnet50_benchmark, "CircadianPredictiveCodingResNet50Classifier", capture_model
    )
    config = ResNet50BenchmarkConfig(
        train_samples=8,
        validation_samples=8,
        test_samples=8,
        num_classes=3,
        image_size=32,
        batch_size=4,
        epochs=2,
        seed=29,
        device="cpu",
        target_accuracy=None,
        evaluation_batches=1,
        inference_batches=1,
        warmup_batches=0,
        circadian_head_hidden_dim=32,
        circadian_min_hidden_dim=16,
        circadian_max_hidden_dim=64,
        circadian_sleep_interval=1,
        circadian_force_sleep=True,
        circadian_min_sleep_steps=0,
    )

    def run_with_label_shift(shift: int) -> tuple[Any, dict[str, Any]]:
        loaders = build_synthetic_vision_dataloaders(
            SyntheticVisionDatasetConfig(
                train_samples=config.train_samples,
                validation_samples=config.validation_samples,
                guard_samples=config.guard_samples,
                test_samples=config.test_samples,
                num_classes=config.num_classes,
                image_size=config.image_size,
                batch_size=config.batch_size,
                seed=config.seed,
            )
        )
        changed_dataset = getattr(loaders, f"{changed_role}_loader").dataset
        changed_dataset.labels = (
            changed_dataset.labels + shift
        ) % config.num_classes
        resnet50_benchmark._set_seed(torch, config.seed)
        report = resnet50_benchmark._benchmark_circadian(
            torch=torch, device=torch.device("cpu"), loaders=loaders, config=config
        )
        return report, created_models[-1].snapshot_state()

    original_report, original_state = run_with_label_shift(0)
    altered_report, altered_state = run_with_label_shift(1)

    assert original_report.epochs_ran == altered_report.epochs_ran
    if changed_role == "test":
        assert original_report.validation_accuracy == altered_report.validation_accuracy
    assert original_report.circadian_total_rollbacks == altered_report.circadian_total_rollbacks
    assert original_report.circadian_total_splits == altered_report.circadian_total_splits
    assert original_report.circadian_total_prunes == altered_report.circadian_total_prunes
    assert original_state.keys() == altered_state.keys()
    for key, value in original_state.items():
        if torch.is_tensor(value):
            assert torch.equal(value, altered_state[key]), key
        else:
            assert value == altered_state[key], key


@pytest.mark.parametrize(
    ("variant", "class_name"),
    [
        ("backprop", "BackpropResNet50Classifier"),
        ("predictive", "PredictiveCodingResNet50Classifier"),
    ],
)
@pytest.mark.parametrize("changed_role", ["validation", "test"])
def test_other_vision_models_ignore_outer_labels_during_training(
    variant: str, class_name: str, changed_role: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    original_class = getattr(resnet50_benchmark, class_name)
    created_models: list[Any] = []

    def capture_model(**kwargs: Any) -> Any:
        model = original_class(**kwargs)
        created_models.append(model)
        return model

    monkeypatch.setattr(resnet50_benchmark, class_name, capture_model)
    config = ResNet50BenchmarkConfig(
        train_samples=8, validation_samples=8, test_samples=8,
        num_classes=3, image_size=32, batch_size=4, epochs=1, seed=31,
        device="cpu", target_accuracy=None, evaluation_batches=1,
        inference_batches=1, warmup_batches=0, backprop_freeze_backbone=True,
        predictive_head_hidden_dim=32,
    )

    def run_with_shift(shift: int) -> tuple[Any, dict[str, Any]]:
        loaders = build_synthetic_vision_dataloaders(
            SyntheticVisionDatasetConfig(
                train_samples=8, validation_samples=8, guard_samples=8, test_samples=8,
                num_classes=3, image_size=32, batch_size=4, seed=31,
            )
        )
        changed_dataset = getattr(loaders, f"{changed_role}_loader").dataset
        changed_dataset.labels = (
            changed_dataset.labels + shift
        ) % config.num_classes
        resnet50_benchmark._set_seed(torch, config.seed)
        report = getattr(resnet50_benchmark, f"_benchmark_{variant}")(
            torch=torch, device=torch.device("cpu"), loaders=loaders, config=config,
        )
        model = created_models[-1]
        state = {
            f"backbone.{key}": value.clone()
            for key, value in model.backbone.state_dict().items()
        }
        if variant == "backprop":
            state.update({
                f"classifier.{key}": value.clone()
                for key, value in model.classifier.state_dict().items()
            })
        else:
            state.update({
                name: getattr(model.head, name).clone()
                for name in (
                    "weight_feature_hidden", "bias_hidden", "weight_hidden_output",
                    "bias_output", "_traffic_sum",
                )
            })
            state["traffic_steps"] = model.head._traffic_steps
        return report, state

    original_report, original_state = run_with_shift(0)
    altered_report, altered_state = run_with_shift(1)
    assert original_report.epochs_ran == altered_report.epochs_ran
    if changed_role == "test":
        assert original_report.validation_accuracy == altered_report.validation_accuracy
    assert original_state.keys() == altered_state.keys()
    for key, value in original_state.items():
        if torch.is_tensor(value):
            assert torch.equal(value, altered_state[key]), key
        else:
            assert value == altered_state[key], key


def test_should_keep_final_loader_sealed_until_every_model_is_trained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trained_names: list[str] = []

    class SealedTestLoader:
        def __iter__(self) -> Any:
            if len(trained_names) != 3:
                raise AssertionError("Final test data were opened before all models trained.")
            return iter(())

    test_loader = SealedTestLoader()
    loaders = SimpleNamespace(
        train_loader=object(),
        validation_loader=object(),
        guard_loader=object(),
        test_loader=test_loader,
        num_classes=3,
        split_hashes={"train": "a", "validation": "b", "guard": "d", "test": "c"},
    )
    monkeypatch.setattr(resnet50_benchmark, "_build_benchmark_loaders", lambda config: loaders)

    def fake_train(name: str) -> Any:
        def train(torch: Any, device: Any, loaders: Any, config: Any) -> Any:
            del torch, device, config
            assert not hasattr(loaders, "test_loader")
            trained_names.append(name)
            return SimpleNamespace(model_name=name)

        return train

    monkeypatch.setattr(resnet50_benchmark, "_train_backprop", fake_train("BackpropResNet50"))
    monkeypatch.setattr(
        resnet50_benchmark, "_train_predictive", fake_train("PredictiveCodingResNet50")
    )
    monkeypatch.setattr(
        resnet50_benchmark,
        "_train_circadian",
        fake_train("CircadianPredictiveCodingResNet50"),
    )

    def fake_finalize(torch: Any, device: Any, outcome: Any,
                      test_loader: Any, config: Any) -> Any:
        del torch, device, config
        assert list(test_loader) == []
        return outcome

    monkeypatch.setattr(resnet50_benchmark, "_finalize_test_report", fake_finalize)
    result = run_resnet50_benchmark(ResNet50BenchmarkConfig())
    assert len(result.reports) == 3
    assert trained_names == [
        "BackpropResNet50",
        "PredictiveCodingResNet50",
        "CircadianPredictiveCodingResNet50",
    ]
