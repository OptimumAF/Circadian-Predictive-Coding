"""Observe matched-head memory with one fresh process per head.

Each child recreates the same frozen train/guard/validation feature bank and
trains one head. The parent verifies common hashes before returning any
descriptive memory report. No child reads the final-test loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from multiprocessing import get_context
from multiprocessing.connection import Connection
from os import getpid
from typing import Any

from src.app import matched_head_benchmark as matched
from src.app.resnet50_benchmark import (
    ResNet50BenchmarkConfig,
    _build_circadian_head_config,
    _resolve_device,
    _set_seed,
)
from src.core.resnet50_variants import (
    BackpropMLPHead,
    CircadianPredictiveCodingHead,
    PredictiveCodingHead,
)
from src.shared.process_memory import ProcessRssSampler
from src.shared.torch_runtime import require_torch

PROCESS_ISOLATED_FIXED_WIDTH_MEMORY_PROTOCOL = "vision_three_head_fixed_width_process_memory_v1"
HEAD_NAMES = ("backprop_mlp", "predictive_coding", "circadian_predictive_coding")
ROLES = ("train", "guard", "validation")


@dataclass(frozen=True)
class IsolatedHeadMemoryReport:
    head_name: str
    pid: int
    backbone_hash: str
    initial_head_hash: str
    trained_head_hash: str
    split_hashes: dict[str, str]
    feature_hashes: dict[str, str]
    feature_bytes: int
    head_parameters: int
    final_head_parameters: int
    seen_samples: int
    wake_batches: int
    latent_relaxation_steps: int
    guard_examples_scored: int
    sleep_attempts: int
    setup_rss_start_bytes: int | None
    setup_rss_peak_observed_bytes: int | None
    setup_rss_samples: int
    pretrain_rss_bytes: int | None
    train_rss_start_bytes: int | None
    train_rss_peak_observed_bytes: int | None
    train_rss_samples: int
    cuda_allocated_start_bytes: int | None
    cuda_allocated_peak_bytes: int | None
    cuda_reserved_peak_bytes: int | None


@dataclass(frozen=True)
class ProcessIsolatedMemoryResult:
    protocol_id: str
    source_protocol_id: str
    benchmark_track: str
    memory_scope: str
    config: ResNet50BenchmarkConfig
    reports: dict[str, IsolatedHeadMemoryReport]


def run_process_isolated_fixed_width_memory(
    config: ResNet50BenchmarkConfig,
    *,
    timeout_seconds: float = 120.0,
) -> ProcessIsolatedMemoryResult:
    """Measure one matched head per spawned child, without final-test access."""
    matched._validate_fixed_width_capacity_config(config)
    if not isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be positive finite seconds.")
    if config.device not in {"cpu", "cuda"}:
        raise ValueError("Process-isolated memory requires an explicit cpu or cuda device.")
    if config.dataset_name != "synthetic" or config.dataset_num_workers != 0:
        raise ValueError(
            "The local process-isolated gate requires synthetic data and zero workers."
        )
    reports = {
        head_name: _run_one_head_process(config, head_name, timeout_seconds)
        for head_name in HEAD_NAMES
    }
    _verify_isolated_reports(reports)
    return ProcessIsolatedMemoryResult(
        protocol_id=PROCESS_ISOLATED_FIXED_WIDTH_MEMORY_PROTOCOL,
        source_protocol_id=config.protocol_id,
        benchmark_track=matched.FROZEN_SHARED_REPRESENTATION_TRACK,
        memory_scope="setup_and_trainer_observed_process_rss",
        config=config,
        reports=reports,
    )


def _run_one_head_process(
    config: ResNet50BenchmarkConfig,
    head_name: str,
    timeout_seconds: float,
) -> IsolatedHeadMemoryReport:
    context = get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_worker_entry, args=(config, head_name, child))
    started = False
    try:
        process.start()
        started = True
        child.close()
        if not parent.poll(timeout_seconds):
            raise TimeoutError(f"Isolated {head_name} worker exceeded {timeout_seconds} seconds.")
        try:
            status, payload = parent.recv()
        except EOFError as error:
            raise RuntimeError(f"Isolated {head_name} worker closed without a report.") from error
        process.join(timeout=5.0)
        if process.is_alive():
            raise RuntimeError(f"Isolated {head_name} worker did not exit after reporting.")
        if process.exitcode != 0:
            raise RuntimeError(f"Isolated {head_name} worker exited with {process.exitcode}.")
        if status != "ok":
            raise RuntimeError(f"Isolated {head_name} worker failed: {payload}")
        if not isinstance(payload, IsolatedHeadMemoryReport):
            raise RuntimeError(f"Isolated {head_name} worker returned an invalid report.")
        if payload.pid != process.pid:
            raise RuntimeError(f"Isolated {head_name} worker reported the wrong process ID.")
        return payload
    finally:
        parent.close()
        child.close()
        if started:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5.0)


def _worker_entry(
    config: ResNet50BenchmarkConfig,
    head_name: str,
    connection: Connection,
) -> None:
    try:
        report = _measure_one_head(config, head_name)
        connection.send(("ok", report))
    except Exception as error:
        connection.send(("error", f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


def _measure_one_head(
    config: ResNet50BenchmarkConfig,
    head_name: str,
) -> IsolatedHeadMemoryReport:
    if head_name not in HEAD_NAMES:
        raise ValueError(f"Unknown matched head: {head_name}")
    matched._validate_fixed_width_capacity_config(config)
    torch = require_torch()
    _set_seed(torch, config.seed)
    device = _resolve_device(torch, config.device)
    with ProcessRssSampler(interval_seconds=matched.PROCESS_RSS_SAMPLE_INTERVAL_SECONDS) as setup:
        loaders = matched._build_benchmark_loaders(config)
        if "guard" not in loaders.split_hashes or loaders.guard_loader is loaders.validation_loader:
            raise ValueError("Isolated memory requires a distinct guard loader.")
        backbone, feature_dim = matched._build_resnet50_backbone(
            device=device,
            freeze_backbone=True,
            backbone_weights=config.backbone_weights,
        )
        backbone.eval()
        backbone_hash = matched._hash_named_tensors(tuple(backbone.state_dict().items()))
        batches = {
            "train": matched._materialize_role_features(
                torch,
                backbone,
                loaders.train_loader,
                device,
                config.seed + 101,
            ),
            "guard": matched._materialize_role_features(
                torch,
                backbone,
                loaders.guard_loader,
                device,
                config.seed + 102,
            ),
            "validation": matched._materialize_role_features(
                torch,
                backbone,
                loaders.validation_loader,
                device,
                config.seed + 103,
            ),
        }
        head = _build_head(head_name, config, feature_dim, loaders.num_classes, device)
        initial_hash = matched._hash_head(head)
        head_parameters = head.parameter_count()
        pretrain_rss = setup.sample()

    outcome = matched._train_with_memory_telemetry(
        torch,
        device,
        lambda: _train_head(head_name, torch, device, head, batches, config),
    )
    final_parameters = head.parameter_count()
    if final_parameters != head_parameters or outcome.parameter_count != head_parameters:
        raise AssertionError("Isolated fixed-width head changed parameter capacity.")
    if head_name == "circadian_predictive_coding" and (
        outcome.sleep_attempts < 1
        or outcome.total_splits != 0
        or outcome.total_prunes != 0
        or outcome.hidden_dim_end != config.circadian_head_hidden_dim
    ):
        raise AssertionError("Isolated circadian guarded-sleep capacity invariant failed.")
    return IsolatedHeadMemoryReport(
        head_name=head_name,
        pid=getpid(),
        backbone_hash=backbone_hash,
        initial_head_hash=initial_hash,
        trained_head_hash=matched._hash_trained_head(head),
        split_hashes={role: loaders.split_hashes[role] for role in ROLES},
        feature_hashes={role: matched._hash_batches(batches[role]) for role in ROLES},
        feature_bytes=sum(
            tensor.numel() * tensor.element_size()
            for role in ROLES
            for pair in batches[role]
            for tensor in pair
        ),
        head_parameters=head_parameters,
        final_head_parameters=final_parameters,
        seen_samples=outcome.seen_samples,
        wake_batches=outcome.wake_batches,
        latent_relaxation_steps=outcome.latent_relaxation_steps,
        guard_examples_scored=outcome.guard_examples_scored,
        sleep_attempts=outcome.sleep_attempts,
        setup_rss_start_bytes=setup.start_bytes,
        setup_rss_peak_observed_bytes=setup.peak_bytes,
        setup_rss_samples=setup.sample_count,
        pretrain_rss_bytes=pretrain_rss,
        train_rss_start_bytes=outcome.process_rss_start_bytes,
        train_rss_peak_observed_bytes=outcome.process_rss_peak_observed_bytes,
        train_rss_samples=outcome.process_rss_samples,
        cuda_allocated_start_bytes=outcome.cuda_allocated_start_bytes,
        cuda_allocated_peak_bytes=outcome.cuda_allocated_peak_bytes,
        cuda_reserved_peak_bytes=outcome.cuda_reserved_peak_bytes,
    )


def _build_head(
    head_name: str,
    config: ResNet50BenchmarkConfig,
    feature_dim: int,
    num_classes: int,
    device: Any,
) -> Any:
    head_seed = config.seed + 11
    if head_name == "backprop_mlp":
        return BackpropMLPHead(
            feature_dim,
            config.predictive_head_hidden_dim,
            num_classes,
            device,
            head_seed,
        )
    if head_name == "predictive_coding":
        return PredictiveCodingHead(
            feature_dim,
            config.predictive_head_hidden_dim,
            num_classes,
            device,
            head_seed,
        )
    return CircadianPredictiveCodingHead(
        feature_dim=feature_dim,
        hidden_dim=config.circadian_head_hidden_dim,
        num_classes=num_classes,
        device=device,
        seed=head_seed,
        config=_build_circadian_head_config(config),
        min_hidden_dim=config.circadian_min_hidden_dim,
        max_hidden_dim=config.circadian_max_hidden_dim,
    )


def _train_head(
    head_name: str,
    torch: Any,
    device: Any,
    head: Any,
    batches: dict[str, matched.FeatureBatches],
    config: ResNet50BenchmarkConfig,
) -> matched._TrainedHead:
    common = (
        torch,
        device,
        head,
        batches["train"],
        batches["guard"],
        batches["validation"],
        config,
    )
    if head_name == "backprop_mlp":
        return matched._train_backprop_head(*common)
    if head_name == "predictive_coding":
        return matched._train_predictive_head(*common)
    return matched._train_circadian_head(*common)


def _verify_isolated_reports(reports: dict[str, IsolatedHeadMemoryReport]) -> None:
    if set(reports) != set(HEAD_NAMES):
        raise AssertionError("Isolated memory requires exactly three head reports.")
    reference = reports[HEAD_NAMES[0]]
    if any(report.pid == getpid() for report in reports.values()):
        raise AssertionError("Isolated memory ran a head in the parent process.")
    if set(reference.split_hashes) != set(ROLES) or set(reference.feature_hashes) != set(ROLES):
        raise AssertionError("Isolated memory accessed an unexpected data role.")
    for head_name, report in reports.items():
        if report.head_name != head_name:
            raise AssertionError("Isolated memory head report identity mismatch.")
        if report.split_hashes != reference.split_hashes:
            raise AssertionError("Isolated memory split hashes differ between heads.")
        if report.feature_hashes != reference.feature_hashes:
            raise AssertionError("Isolated memory feature hashes differ between heads.")
        if report.backbone_hash != reference.backbone_hash:
            raise AssertionError("Isolated memory backbone hashes differ between heads.")
        if report.initial_head_hash != reference.initial_head_hash:
            raise AssertionError("Isolated memory initial head tensors differ.")
        if report.feature_bytes != reference.feature_bytes:
            raise AssertionError("Isolated memory feature cache sizes differ.")
        if (
            report.head_parameters != reference.head_parameters
            or report.final_head_parameters != report.head_parameters
        ):
            raise AssertionError("Isolated memory head capacity differs.")
