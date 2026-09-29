"""Process-isolated observed-memory gates for fixed-width matched heads."""

from __future__ import annotations

from dataclasses import replace
from os import getpid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.app import isolated_head_memory as isolated  # noqa: E402
from src.app.resnet50_benchmark import ResNet50BenchmarkConfig  # noqa: E402
from src.shared.process_memory import read_process_rss_bytes  # noqa: E402


def _config() -> ResNet50BenchmarkConfig:
    return ResNet50BenchmarkConfig(
        train_samples=8,
        guard_samples=8,
        validation_samples=8,
        test_samples=8,
        num_classes=3,
        image_size=32,
        batch_size=4,
        epochs=1,
        seed=47,
        device="cpu",
        target_accuracy=None,
        backprop_freeze_backbone=True,
        backbone_weights="none",
        predictive_head_hidden_dim=16,
        predictive_inference_steps=2,
        circadian_head_hidden_dim=16,
        circadian_min_hidden_dim=16,
        circadian_max_hidden_dim=16,
        circadian_inference_steps=2,
        circadian_sleep_interval=1,
        circadian_force_sleep=True,
        circadian_sleep_warmup_steps=0,
        circadian_use_adaptive_sleep_trigger=False,
        circadian_enable_sleep_rollback=True,
    )


def test_fresh_processes_preserve_features_and_initial_tensors() -> None:
    result = isolated.run_process_isolated_fixed_width_memory(_config(), timeout_seconds=60)

    assert result.protocol_id == "vision_three_head_fixed_width_process_memory_v1"
    assert result.source_protocol_id == "vision_guard_separated_unmatched_v2"
    assert result.memory_scope == "setup_and_trainer_observed_process_rss"
    assert set(result.reports) == set(isolated.HEAD_NAMES)
    reports = tuple(result.reports.values())
    assert all(report.pid != getpid() for report in reports)
    assert len({report.backbone_hash for report in reports}) == 1
    assert len({report.initial_head_hash for report in reports}) == 1
    assert len({report.feature_bytes for report in reports}) == 1
    assert len({report.head_parameters for report in reports}) == 1
    assert all(report.head_parameters == report.final_head_parameters for report in reports)
    assert all(report.split_hashes == reports[0].split_hashes for report in reports)
    assert all(report.feature_hashes == reports[0].feature_hashes for report in reports)
    assert set(reports[0].split_hashes) == {"train", "guard", "validation"}
    assert set(reports[0].feature_hashes) == {"train", "guard", "validation"}
    assert all(report.feature_bytes > 0 for report in reports)
    assert all(report.seen_samples == 8 and report.wake_batches == 2 for report in reports)
    assert result.reports["circadian_predictive_coding"].sleep_attempts == 1
    assert result.reports["circadian_predictive_coding"].guard_examples_scored == 24
    assert result.reports["backprop_mlp"].guard_examples_scored == 8
    if read_process_rss_bytes() is not None:
        for report in reports:
            assert report.setup_rss_start_bytes is not None
            assert report.setup_rss_peak_observed_bytes is not None
            assert report.pretrain_rss_bytes is not None
            assert report.train_rss_start_bytes is not None
            assert report.train_rss_peak_observed_bytes is not None
            assert report.setup_rss_peak_observed_bytes >= report.setup_rss_start_bytes
            assert report.train_rss_peak_observed_bytes >= report.train_rss_start_bytes
            assert report.setup_rss_samples >= 2
            assert report.train_rss_samples >= 2
    assert all(report.cuda_allocated_peak_bytes is None for report in reports)


def test_worker_never_reads_final_test_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    features = torch.ones((2, 5), dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    role_loader = [(features, labels)]

    class Loaders:
        train_loader = role_loader
        guard_loader = list(role_loader)
        validation_loader = list(role_loader)
        num_classes = 3
        split_hashes = {role: role for role in ("train", "guard", "validation", "test")}

        @property
        def test_loader(self) -> Any:
            raise AssertionError("Isolated memory worker opened final test")

    monkeypatch.setattr(isolated.matched, "_build_benchmark_loaders", lambda config: Loaders())
    monkeypatch.setattr(
        isolated.matched,
        "_build_resnet50_backbone",
        lambda **kwargs: (torch.nn.Identity(), 5),
    )
    report = isolated._measure_one_head(_config(), "circadian_predictive_coding")
    assert set(report.feature_hashes) == {"train", "guard", "validation"}
    assert report.sleep_attempts == 1


def test_invalid_request_is_rejected_before_spawning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        isolated,
        "_run_one_head_process",
        lambda *args: pytest.fail("Invalid request spawned a child"),
    )
    with pytest.raises(ValueError, match="Fixed-width"):
        isolated.run_process_isolated_fixed_width_memory(
            replace(_config(), circadian_max_hidden_dim=32)
        )
    with pytest.raises(ValueError, match="timeout"):
        isolated.run_process_isolated_fixed_width_memory(_config(), timeout_seconds=0)


def test_local_cifar_memory_has_distinct_protocol_and_rejects_unsafe_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cifar-10-batches-py"
    cache.mkdir()
    (cache / "data_batch_1").touch()
    (cache / "test_batch").touch()
    config = replace(
        _config(),
        dataset_name="cifar10",
        dataset_data_root=str(tmp_path),
        dataset_download=False,
        dataset_num_workers=0,
        num_classes=10,
    )
    spawned: list[str] = []

    def run_one(candidate: Any, head_name: str, timeout: float) -> Any:
        assert candidate == config
        assert timeout == 5.0
        spawned.append(head_name)
        return SimpleNamespace(head_name=head_name)

    monkeypatch.setattr(isolated, "_run_one_head_process", run_one)
    monkeypatch.setattr(isolated, "_verify_isolated_reports", lambda reports: None)
    result = isolated.run_process_isolated_fixed_width_memory(config, timeout_seconds=5.0)
    assert result.protocol_id == isolated.PROCESS_ISOLATED_CIFAR_MEMORY_PROTOCOL
    assert spawned == list(isolated.HEAD_NAMES)

    for invalid in (
        replace(config, dataset_download=True),
        replace(config, dataset_num_workers=2),
        replace(config, dataset_name="cifar100"),
        replace(config, dataset_data_root=str(tmp_path / "missing")),
    ):
        spawned.clear()
        with pytest.raises((ValueError, FileNotFoundError)):
            isolated.run_process_isolated_fixed_width_memory(invalid, timeout_seconds=5.0)
        assert spawned == []


def test_parent_rejects_mismatched_feature_hashes(monkeypatch: pytest.MonkeyPatch) -> None:
    base = isolated.IsolatedHeadMemoryReport(
        head_name="backprop_mlp",
        pid=getpid() + 1,
        backbone_hash="backbone",
        initial_head_hash="initial",
        trained_head_hash="trained",
        split_hashes={role: role for role in ("train", "guard", "validation")},
        feature_hashes={role: role for role in ("train", "guard", "validation")},
        feature_bytes=100,
        head_parameters=20,
        final_head_parameters=20,
        seen_samples=8,
        wake_batches=2,
        latent_relaxation_steps=0,
        guard_examples_scored=8,
        sleep_attempts=0,
        setup_rss_start_bytes=10,
        setup_rss_peak_observed_bytes=20,
        setup_rss_samples=2,
        pretrain_rss_bytes=20,
        train_rss_start_bytes=20,
        train_rss_peak_observed_bytes=30,
        train_rss_samples=2,
        cuda_allocated_start_bytes=None,
        cuda_allocated_peak_bytes=None,
        cuda_reserved_peak_bytes=None,
    )
    reports = {
        "backprop_mlp": base,
        "predictive_coding": replace(base, head_name="predictive_coding", pid=getpid() + 2),
        "circadian_predictive_coding": replace(
            base,
            head_name="circadian_predictive_coding",
            pid=getpid() + 3,
            feature_hashes={**base.feature_hashes, "guard": "changed"},
        ),
    }
    with pytest.raises(AssertionError, match="feature"):
        isolated._verify_isolated_reports(reports)
