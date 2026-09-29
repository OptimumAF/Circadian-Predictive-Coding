"""Audit separately saved representative confirmation scopes as one study.

The scope workers own training and file writes. This module only restores
their JSON and applies the existing app identity checks plus work, deadline,
and memory coverage before a combined result can be published.
"""

from __future__ import annotations

from dataclasses import replace
import json
from math import isfinite
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_cifar_representative_confirmation as runner  # noqa: E402
from scripts.restore_cifar_representative_selection import (  # noqa: E402
    RestoredRepresentativeSelection,
)
from src.app import isolated_head_memory as isolated  # noqa: E402
from src.app import matched_head_benchmark as matched  # noqa: E402
from src.app import matched_head_tuning as tuning  # noqa: E402
from src.app import repeated_head_confirmation as repeated  # noqa: E402
from src.app.resnet50_benchmark import ResNet50BenchmarkConfig  # noqa: E402


def _read_scope(
    path: Path,
    scope: str,
    seed: int | None,
    restored: RestoredRepresentativeSelection,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"representative {scope} result missing: {path}")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if (
        set(payload)
        != {
            "schema",
            "scope",
            "seed",
            "request_sha256",
            "freeze_digest",
            "manifest_digest",
            "report",
        }
        or payload["schema"] != "cifar_representative_confirmation_scope_v1"
        or payload["scope"] != scope
        or payload["seed"] != seed
        or payload["request_sha256"] != runner.selection.REQUEST_SHA256
        or payload["freeze_digest"] != restored.freeze["freeze_digest"]
        or payload["manifest_digest"] != restored.confirmation_manifest.manifest_digest
    ):
        raise ValueError(f"representative {scope} result changed scope or provenance")
    return payload["report"]


def _restore_fixed(value: dict[str, Any]) -> tuning.MatchedHeadTuningResult:
    return tuning.MatchedHeadTuningResult(
        protocol_id=value["protocol_id"],
        benchmark_track=value["benchmark_track"],
        base_config=ResNet50BenchmarkConfig(**value["base_config"]),
        seeds=tuple(value["seeds"]),
        candidates_per_head=value["candidates_per_head"],
        trials_per_head=value["trials_per_head"],
        attempts=tuple(
            tuning.HeadTuningAttempt(**{**row, "config": ResNet50BenchmarkConfig(**row["config"])})
            for row in value["attempts"]
        ),
        trials=tuple(
            tuning.HeadTuningTrial(**{**row, "config": ResNet50BenchmarkConfig(**row["config"])})
            for row in value["trials"]
        ),
        selections=tuple(
            tuning.HeadTuningSelection(**{**row, "seeds": tuple(row["seeds"])})
            for row in value["selections"]
        ),
        confirmations=tuple(tuning.HeadTuningConfirmation(**row) for row in value["confirmations"]),
    )


def _restore_wall(value: dict[str, Any]) -> matched.ThreeHeadFixedFeatureResult:
    capacity = value["capacity_control"]
    return matched.ThreeHeadFixedFeatureResult(
        **{
            **value,
            "training_order": tuple(value["training_order"]),
            "backprop": matched.FixedFeatureHeadReport(**value["backprop"]),
            "predictive_coding": matched.FixedFeatureHeadReport(**value["predictive_coding"]),
            "circadian": matched.FixedFeatureHeadReport(**value["circadian"]),
            "capacity_control": matched.FixedWidthCapacityControl(**capacity)
            if capacity is not None
            else None,
        }
    )


def _restore_memory(value: dict[str, Any]) -> isolated.ProcessIsolatedMemoryResult:
    return isolated.ProcessIsolatedMemoryResult(
        **{
            **value,
            "config": ResNet50BenchmarkConfig(**value["config"]),
            "reports": {
                name: isolated.IsolatedHeadMemoryReport(**row)
                for name, row in value["reports"].items()
            },
        }
    )


def _verify_fixed_work(
    fixed: tuning.MatchedHeadTuningResult,
    restored: RestoredRepresentativeSelection,
) -> None:
    config = restored.confirmation_manifest.base_config
    expected_batches = config.dataset_train_subset_size // config.batch_size
    if fixed.base_config != config or fixed.trials_per_head != len(fixed.seeds):
        raise ValueError("fixed-data result changed its frozen config or trial budget")
    for row in fixed.trials:
        expected_relax = (
            0
            if row.head_name == "backprop_mlp"
            else row.wake_batches
            * (
                config.predictive_inference_steps
                if row.head_name == "predictive_coding"
                else config.circadian_inference_steps
            )
        )
        if (
            row.epochs_ran != config.epochs
            or row.seen_samples != config.dataset_train_subset_size
            or row.wake_batches != expected_batches
            or row.latent_relaxation_steps != expected_relax
            or row.validation_examples_scored != config.dataset_validation_subset_size
            or row.guard_examples_scored <= 0
            or row.replay_examples != 0
        ):
            raise ValueError("fixed-data trial omitted declared work or roles")
    for confirmation in fixed.confirmations:
        if (
            not isfinite(confirmation.test_accuracy)
            or not 0 <= confirmation.test_accuracy <= 1
            or not isfinite(confirmation.test_cross_entropy)
            or confirmation.test_cross_entropy < 0
        ):
            raise ValueError("fixed-data final score is not finite and bounded")


def _verify_wall_work(
    wall: matched.ThreeHeadFixedFeatureResult,
    restored: RestoredRepresentativeSelection,
) -> None:
    budget = restored.confirmation_manifest.wall_time_budget_seconds
    if (
        wall.wall_time_budget_seconds != budget
        or set(wall.training_order) != set(tuning.HEAD_NAMES)
        or len(wall.training_order) != 3
    ):
        raise ValueError("wall-time result changed its head order or deadline")
    for name in tuning.HEAD_NAMES:
        head = getattr(wall, repeated._report_field(name))
        if (
            head.stop_reason != "deadline"
            or head.train_seconds < budget
            or head.deadline_overshoot_seconds < 0
            or head.wake_batches <= 0
            or head.seen_samples <= 0
            or head.guard_examples_scored <= 0
            or head.replay_examples != 0
            or not isfinite(head.test_accuracy)
            or not 0 <= head.test_accuracy <= 1
        ):
            raise ValueError(f"wall-time {name} omitted deadline work or score")


def _verify_memory_work(
    memory: isolated.ProcessIsolatedMemoryResult,
    restored: RestoredRepresentativeSelection,
    seed: int,
) -> None:
    manifest = restored.confirmation_manifest
    expected_config = replace(repeated._combine_selected_config(manifest), seed=seed)
    if memory.config != expected_config or set(memory.reports) != set(tuning.HEAD_NAMES):
        raise ValueError("memory scope changed its frozen config or head set")
    for name, head in memory.reports.items():
        expected_relax = (
            0
            if name == "backprop_mlp"
            else head.wake_batches
            * (
                expected_config.predictive_inference_steps
                if name == "predictive_coding"
                else expected_config.circadian_inference_steps
            )
        )
        if (
            head.seen_samples != expected_config.dataset_train_subset_size
            or head.wake_batches
            != expected_config.dataset_train_subset_size // expected_config.batch_size
            or head.latent_relaxation_steps != expected_relax
            or head.guard_examples_scored <= 0
            or head.setup_rss_peak_observed_bytes is None
            or head.train_rss_peak_observed_bytes is None
            or head.cuda_allocated_peak_bytes is None
            or head.cuda_reserved_peak_bytes is None
            or head.head_parameters != head.final_head_parameters
        ):
            raise ValueError(f"memory {name} omitted work, capacity, or scoped telemetry")


def audit_fixed_scope(
    data_dir: Path, restored: RestoredRepresentativeSelection
) -> tuning.MatchedHeadTuningResult:
    """Reject an incomplete fixed-data scope before wall-time work begins."""
    manifest = restored.confirmation_manifest
    fixed = _restore_fixed(_read_scope(data_dir / runner.FIXED_NAME, "fixed_data", None, restored))
    repeated._verify_fixed_data(manifest, fixed)
    _verify_fixed_work(fixed, restored)
    return fixed


def audit_wall_scope(
    data_dir: Path,
    restored: RestoredRepresentativeSelection,
    fixed: tuning.MatchedHeadTuningResult,
) -> tuple[matched.ThreeHeadFixedFeatureResult, ...]:
    """Reject missing or mismatched deadline heads before memory work."""
    wall: list[matched.ThreeHeadFixedFeatureResult] = []
    for seed in restored.confirmation_manifest.confirmation_seeds:
        report = _restore_wall(
            _read_scope(runner._seed_path(data_dir, "wall-time", seed), "wall_time", seed, restored)
        )
        repeated._verify_wall_time_pair(fixed, seed, report)
        _verify_wall_work(report, restored)
        wall.append(report)
    return tuple(wall)


def audit_memory_scope(
    data_dir: Path,
    restored: RestoredRepresentativeSelection,
    fixed: tuning.MatchedHeadTuningResult,
) -> tuple[isolated.ProcessIsolatedMemoryResult, ...]:
    """Require nine distinct children and matched scoped telemetry."""
    memory: list[isolated.ProcessIsolatedMemoryResult] = []
    pids: set[int] = set()
    for seed in restored.confirmation_manifest.confirmation_seeds:
        memory_report = _restore_memory(
            _read_scope(
                runner._seed_path(data_dir, "memory", seed), "capacity_memory", seed, restored
            )
        )
        repeated._verify_memory_pair(fixed, seed, memory_report)
        _verify_memory_work(memory_report, restored, seed)
        for head in memory_report.reports.values():
            if head.pid in pids:
                raise ValueError("memory scope reused a process across head/seed reports")
            pids.add(head.pid)
        memory.append(memory_report)
    if len(pids) != 9:
        raise ValueError("memory scope omitted a fresh head child")
    return tuple(memory)


def audit_saved_scopes(
    data_dir: Path, restored: RestoredRepresentativeSelection
) -> repeated.RepeatedConfirmationResult:
    """Require every declared seed/head/scope and paired identity before success."""
    manifest = restored.confirmation_manifest
    fixed = audit_fixed_scope(data_dir, restored)
    wall = audit_wall_scope(data_dir, restored, fixed)
    memory = audit_memory_scope(data_dir, restored, fixed)
    return repeated.RepeatedConfirmationResult(
        protocol_id=repeated.REPEATED_CONFIRMATION_PROTOCOL,
        manifest=manifest,
        fixed_data=fixed,
        wall_time=wall,
        capacity_memory=memory,
        fixed_data_accuracy=repeated._accuracy_summaries(
            manifest.confirmation_seeds,
            {(row.head_name, row.seed): row.test_accuracy for row in fixed.confirmations},
        ),
        wall_time_accuracy=repeated._accuracy_summaries(
            manifest.confirmation_seeds,
            {
                (name, seed): getattr(report, repeated._report_field(name)).test_accuracy
                for seed, report in zip(manifest.confirmation_seeds, wall, strict=True)
                for name in tuning.HEAD_NAMES
            },
        ),
        observed_train_rss=repeated._memory_summaries(manifest.confirmation_seeds, memory),
    )
