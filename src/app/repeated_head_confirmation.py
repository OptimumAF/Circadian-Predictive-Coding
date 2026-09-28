"""Predeclared, paired confirmation of validation-selected matched heads.

The manifest is derived from a validation-only tuning result before final
test is opened. Fixed-data, wall-time, and isolated-memory observations stay
separate, and every declared confirmation seed is retained.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from hashlib import sha256
from json import dumps
from math import isfinite
from statistics import mean, pstdev

from src.app import isolated_head_memory as isolated
from src.app import matched_head_benchmark as matched
from src.app import matched_head_tuning as tuning
from src.app.resnet50_benchmark import ResNet50BenchmarkConfig

CONFIRMATION_MANIFEST_PROTOCOL = "vision_matched_head_confirmation_manifest_v1"
REPEATED_CONFIRMATION_PROTOCOL = "vision_matched_head_repeated_confirmation_v1"
CONFIRMATION_SCOPES = ("fixed_data_epoch", "fixed_wall_time", "isolated_capacity_memory")
METRIC_NAMES = ("accuracy", "cross_entropy")


@dataclass(frozen=True)
class SelectedHeadConfiguration:
    head_name: str
    candidate_id: str
    config: ResNet50BenchmarkConfig


@dataclass(frozen=True)
class RepeatedConfirmationManifest:
    protocol_id: str
    source_selection_protocol_id: str
    source_selection_digest: str
    selection_seeds: tuple[int, ...]
    confirmation_seeds: tuple[int, ...]
    base_config: ResNet50BenchmarkConfig
    selected_heads: tuple[SelectedHeadConfiguration, ...]
    metric_names: tuple[str, ...]
    scopes: tuple[str, ...]
    wall_time_budget_seconds: float
    wall_time_epoch_cap: int
    manifest_digest: str


@dataclass(frozen=True)
class DescriptiveSummary:
    metric_name: str
    seeds: tuple[int, ...]
    values: tuple[float, ...]
    mean: float
    population_std: float


@dataclass(frozen=True)
class RepeatedConfirmationResult:
    protocol_id: str
    manifest: RepeatedConfirmationManifest
    fixed_data: tuning.MatchedHeadTuningResult
    wall_time: tuple[matched.ThreeHeadFixedFeatureResult, ...]
    capacity_memory: tuple[isolated.ProcessIsolatedMemoryResult, ...]
    fixed_data_accuracy: dict[str, DescriptiveSummary]
    wall_time_accuracy: dict[str, DescriptiveSummary]
    observed_train_rss: dict[str, DescriptiveSummary]


def create_confirmation_manifest(
    selection: tuning.MatchedHeadTuningResult,
    *,
    confirmation_seeds: tuple[int, ...],
    wall_time_budget_seconds: float,
    wall_time_epoch_cap: int,
) -> RepeatedConfirmationManifest:
    """Freeze selected configs and all confirmation choices before test access."""
    if (
        selection.protocol_id != tuning.MATCHED_HEAD_VALIDATION_SELECTION_PROTOCOL
        or selection.confirmations
    ):
        raise ValueError("Confirmation requires a validation-only selection result.")
    _validate_confirmation_seeds(selection.seeds, confirmation_seeds)
    matched._validate_fixed_width_capacity_config(selection.base_config)
    _validate_wall_time_budget(wall_time_budget_seconds, wall_time_epoch_cap)
    if len(selection.selections) != len(tuning.HEAD_NAMES):
        raise ValueError("Validation selection must contain every matched head.")
    if any(attempt.status != "complete" for attempt in selection.attempts):
        raise ValueError("Validation selection contains an incomplete attempt.")

    selected: list[SelectedHeadConfiguration] = []
    for head_name in tuning.HEAD_NAMES:
        matches = [item for item in selection.selections if item.head_name == head_name]
        if len(matches) != 1:
            raise ValueError(f"Validation selection is incomplete for {head_name}.")
        choice = matches[0]
        rows = [
            trial
            for trial in selection.trials
            if trial.head_name == head_name and trial.candidate_id == choice.candidate_id
        ]
        if len(rows) != len(selection.seeds) or {row.seed for row in rows} != set(selection.seeds):
            raise ValueError(f"Selected {head_name} candidate lacks a complete seed set.")
        normalized = [replace(row.config, seed=selection.base_config.seed) for row in rows]
        if any(config != normalized[0] for config in normalized):
            raise ValueError(f"Selected {head_name} config changed between selection seeds.")
        selected.append(SelectedHeadConfiguration(head_name, choice.candidate_id, normalized[0]))

    candidate_map = _candidate_map(tuple(selected))
    tuning._validate_tuning_request(
        selection.base_config,
        candidate_map,
        confirmation_seeds,
        1,
    )
    manifest = RepeatedConfirmationManifest(
        protocol_id=CONFIRMATION_MANIFEST_PROTOCOL,
        source_selection_protocol_id=selection.protocol_id,
        source_selection_digest=_digest(asdict(selection)),
        selection_seeds=selection.seeds,
        confirmation_seeds=confirmation_seeds,
        base_config=selection.base_config,
        selected_heads=tuple(selected),
        metric_names=METRIC_NAMES,
        scopes=CONFIRMATION_SCOPES,
        wall_time_budget_seconds=wall_time_budget_seconds,
        wall_time_epoch_cap=wall_time_epoch_cap,
        manifest_digest="",
    )
    return replace(manifest, manifest_digest=_manifest_digest(manifest))


def run_repeated_confirmation(
    manifest: RepeatedConfirmationManifest,
) -> RepeatedConfirmationResult:
    """Run every declared seed in each scope, without test-based choices."""
    _validate_manifest(manifest)
    candidates = _candidate_map(manifest.selected_heads)
    fixed_data = tuning.run_matched_head_tuning(
        manifest.base_config,
        candidates,
        seeds=manifest.confirmation_seeds,
        candidates_per_head=1,
    )
    _verify_fixed_data(manifest, fixed_data)

    combined = _combine_selected_config(manifest)
    wall_time = tuple(
        matched.run_three_head_fixed_feature_wall_time_benchmark(
            replace(combined, seed=seed, epochs=manifest.wall_time_epoch_cap),
            wall_time_budget_seconds=manifest.wall_time_budget_seconds,
        )
        for seed in manifest.confirmation_seeds
    )
    for seed, report in zip(manifest.confirmation_seeds, wall_time, strict=True):
        _verify_wall_time_pair(fixed_data, seed, report)

    capacity_memory = tuple(
        isolated.run_process_isolated_fixed_width_memory(
            replace(combined, seed=seed),
            timeout_seconds=60.0,
        )
        for seed in manifest.confirmation_seeds
    )
    for seed, memory_report in zip(manifest.confirmation_seeds, capacity_memory, strict=True):
        _verify_memory_pair(fixed_data, seed, memory_report)

    return RepeatedConfirmationResult(
        protocol_id=REPEATED_CONFIRMATION_PROTOCOL,
        manifest=manifest,
        fixed_data=fixed_data,
        wall_time=wall_time,
        capacity_memory=capacity_memory,
        fixed_data_accuracy=_accuracy_summaries(
            manifest.confirmation_seeds,
            {(row.head_name, row.seed): row.test_accuracy for row in fixed_data.confirmations},
        ),
        wall_time_accuracy=_accuracy_summaries(
            manifest.confirmation_seeds,
            {
                (head_name, seed): getattr(report, _report_field(head_name)).test_accuracy
                for seed, report in zip(manifest.confirmation_seeds, wall_time, strict=True)
                for head_name in tuning.HEAD_NAMES
            },
        ),
        observed_train_rss=_memory_summaries(manifest.confirmation_seeds, capacity_memory),
    )


def _validate_confirmation_seeds(
    selection_seeds: tuple[int, ...],
    confirmation_seeds: tuple[int, ...],
) -> None:
    if (
        type(confirmation_seeds) is not tuple
        or not 3 <= len(confirmation_seeds) <= 4
        or any(type(seed) is not int for seed in confirmation_seeds)
        or len(set(confirmation_seeds)) != len(confirmation_seeds)
        or set(confirmation_seeds) & set(selection_seeds)
    ):
        raise ValueError("Use 3–4 distinct confirmation seeds disjoint from selection seeds.")


def _validate_wall_time_budget(seconds: float, epoch_cap: int) -> None:
    if not isfinite(seconds) or seconds <= 0.0:
        raise ValueError("wall_time_budget_seconds must be positive finite seconds.")
    if type(epoch_cap) is not int or epoch_cap <= 0:
        raise ValueError("wall_time_epoch_cap must be a positive integer.")


def _validate_manifest(manifest: RepeatedConfirmationManifest) -> None:
    if manifest.protocol_id != CONFIRMATION_MANIFEST_PROTOCOL:
        raise ValueError("Unknown confirmation manifest protocol.")
    if manifest.source_selection_protocol_id != tuning.MATCHED_HEAD_VALIDATION_SELECTION_PROTOCOL:
        raise ValueError("Confirmation must derive from validation-only selection.")
    if not manifest.source_selection_digest:
        raise ValueError("Confirmation manifest needs selection evidence.")
    _validate_confirmation_seeds(manifest.selection_seeds, manifest.confirmation_seeds)
    _validate_wall_time_budget(manifest.wall_time_budget_seconds, manifest.wall_time_epoch_cap)
    if manifest.metric_names != METRIC_NAMES or manifest.scopes != CONFIRMATION_SCOPES:
        raise ValueError("Confirmation metrics and scopes must remain predeclared.")
    matched._validate_fixed_width_capacity_config(manifest.base_config)
    tuning._validate_tuning_request(
        manifest.base_config,
        _candidate_map(manifest.selected_heads),
        manifest.confirmation_seeds,
        1,
    )
    if manifest.manifest_digest != _manifest_digest(manifest):
        raise ValueError("Confirmation manifest changed after freezing.")


def _candidate_map(
    selected: tuple[SelectedHeadConfiguration, ...],
) -> dict[str, tuple[tuning.HeadTuningCandidate, ...]]:
    if len(selected) != len(tuning.HEAD_NAMES) or {item.head_name for item in selected} != set(
        tuning.HEAD_NAMES
    ):
        raise ValueError("Selected configs must name each matched head exactly once.")
    return {
        item.head_name: (tuning.HeadTuningCandidate(item.candidate_id, item.config),)
        for item in selected
    }


def _combine_selected_config(manifest: RepeatedConfirmationManifest) -> ResNet50BenchmarkConfig:
    selected = {item.head_name: item.config for item in manifest.selected_heads}
    base = manifest.base_config
    return replace(
        base,
        backprop_learning_rate=selected["backprop_mlp"].backprop_learning_rate,
        backprop_momentum=selected["backprop_mlp"].backprop_momentum,
        predictive_learning_rate=selected["predictive_coding"].predictive_learning_rate,
        predictive_inference_steps=selected["predictive_coding"].predictive_inference_steps,
        predictive_inference_learning_rate=selected[
            "predictive_coding"
        ].predictive_inference_learning_rate,
        circadian_learning_rate=selected["circadian_predictive_coding"].circadian_learning_rate,
        circadian_inference_steps=selected["circadian_predictive_coding"].circadian_inference_steps,
        circadian_inference_learning_rate=selected[
            "circadian_predictive_coding"
        ].circadian_inference_learning_rate,
    )


def _verify_fixed_data(
    manifest: RepeatedConfirmationManifest,
    result: tuning.MatchedHeadTuningResult,
) -> None:
    if result.protocol_id != tuning.MATCHED_HEAD_TUNING_PROTOCOL:
        raise AssertionError("Fixed-data confirmation used the wrong protocol.")
    if result.seeds != manifest.confirmation_seeds:
        raise AssertionError("Fixed-data confirmation changed the declared seed set.")
    expected = {(name, seed) for name in tuning.HEAD_NAMES for seed in manifest.confirmation_seeds}
    trial_pairs = {(row.head_name, row.seed) for row in result.trials}
    test_pairs = {(row.head_name, row.seed) for row in result.confirmations}
    if len(result.trials) != len(expected) or trial_pairs != expected:
        raise AssertionError("Fixed-data confirmation has missing or duplicate training runs.")
    if len(result.attempts) != len(expected) or any(
        row.status != "complete" for row in result.attempts
    ):
        raise AssertionError("Fixed-data confirmation has missing or failed attempts.")
    if len(result.confirmations) != len(expected) or test_pairs != expected:
        raise AssertionError("Fixed-data confirmation has missing or duplicate final tests.")
    selected_ids = {item.head_name: item.candidate_id for item in manifest.selected_heads}
    if any(row.candidate_id != selected_ids[row.head_name] for row in result.confirmations):
        raise AssertionError("Fixed-data confirmation changed a selected candidate.")
    if any(row.candidate_id != selected_ids[row.head_name] for row in result.trials):
        raise AssertionError("Fixed-data training changed a selected candidate.")
    for seed in manifest.confirmation_seeds:
        rows = [row for row in result.trials if row.seed == seed]
        confirmed = [row for row in result.confirmations if row.seed == seed]
        if (
            len({row.backbone_hash for row in rows}) != 1
            or len({row.initial_head_hash for row in rows}) != 1
            or any(row.feature_hashes != rows[0].feature_hashes for row in rows)
            or any(row.split_hashes != rows[0].split_hashes for row in rows)
            or len({row.initial_trainable_parameters for row in rows}) != 1
            or any(row.trainable_parameters != row.initial_trainable_parameters for row in rows)
            or len({row.test_feature_hash for row in confirmed}) != 1
            or len({row.test_split_hash for row in confirmed}) != 1
        ):
            raise AssertionError("Fixed-data heads lost input or capacity matching.")
        circadian = next(row for row in rows if row.head_name == "circadian_predictive_coding")
        if (
            circadian.hidden_dim_start != manifest.base_config.circadian_head_hidden_dim
            or circadian.hidden_dim_end != manifest.base_config.circadian_head_hidden_dim
            or circadian.total_splits != 0
            or circadian.total_prunes != 0
            or circadian.sleep_attempts < 1
        ):
            raise AssertionError("Fixed-data circadian capacity or sleep changed.")


def _verify_wall_time_pair(
    fixed_data: tuning.MatchedHeadTuningResult,
    seed: int,
    report: matched.ThreeHeadFixedFeatureResult,
) -> None:
    reference = next(row for row in fixed_data.trials if row.seed == seed)
    if report.protocol_id != matched.THREE_HEAD_FIXED_FEATURE_WALL_TIME_PROTOCOL:
        raise AssertionError("Wall-time scope used the wrong protocol.")
    if report.backbone_hash != reference.backbone_hash:
        raise AssertionError("Wall-time backbone differs from fixed-data scope.")
    if any(
        report.feature_hashes[role] != reference.feature_hashes[role]
        for role in reference.feature_hashes
    ):
        raise AssertionError("Wall-time features differ from fixed-data scope.")
    if any(
        report.split_hashes[role] != reference.split_hashes[role] for role in reference.split_hashes
    ):
        raise AssertionError("Wall-time splits differ from fixed-data scope.")
    confirmation = next(row for row in fixed_data.confirmations if row.seed == seed)
    if (
        report.feature_hashes["test"] != confirmation.test_feature_hash
        or report.split_hashes["test"] != confirmation.test_split_hash
    ):
        raise AssertionError("Wall-time final test differs from fixed-data scope.")
    if any(value != reference.initial_head_hash for value in report.initial_head_hashes.values()):
        raise AssertionError("Wall-time heads differ at initialization.")
    if any(
        getattr(report, _report_field(name)).trainable_parameters
        != reference.initial_trainable_parameters
        for name in tuning.HEAD_NAMES
    ):
        raise AssertionError("Wall-time head capacity differs from fixed-data scope.")
    if any(
        getattr(report, _report_field(name)).stop_reason != "deadline" for name in tuning.HEAD_NAMES
    ):
        raise AssertionError("Wall-time head stopped before its declared deadline.")


def _verify_memory_pair(
    fixed_data: tuning.MatchedHeadTuningResult,
    seed: int,
    report: isolated.ProcessIsolatedMemoryResult,
) -> None:
    reference = next(row for row in fixed_data.trials if row.seed == seed)
    if report.protocol_id != isolated.PROCESS_ISOLATED_FIXED_WIDTH_MEMORY_PROTOCOL:
        raise AssertionError("Capacity-memory scope used the wrong protocol.")
    for head in report.reports.values():
        if head.backbone_hash != reference.backbone_hash:
            raise AssertionError("Capacity-memory backbone differs from fixed-data scope.")
        if head.feature_hashes != reference.feature_hashes:
            raise AssertionError("Capacity-memory features differ from fixed-data scope.")
        if head.split_hashes != reference.split_hashes:
            raise AssertionError("Capacity-memory splits differ from fixed-data scope.")
        if head.initial_head_hash != reference.initial_head_hash:
            raise AssertionError("Capacity-memory heads differ at initialization.")
        if head.head_parameters != reference.initial_trainable_parameters:
            raise AssertionError("Capacity-memory head parameter count differs.")
        if head.final_head_parameters != head.head_parameters:
            raise AssertionError("Capacity-memory head changed parameter count.")


def _accuracy_summaries(
    seeds: tuple[int, ...],
    scores: dict[tuple[str, int], float],
) -> dict[str, DescriptiveSummary]:
    expected = {(name, seed) for name in tuning.HEAD_NAMES for seed in seeds}
    if set(scores) != expected:
        raise AssertionError("Accuracy summary requires every declared head and seed.")
    return {
        name: _summarize("test_accuracy", seeds, tuple(scores[(name, seed)] for seed in seeds))
        for name in tuning.HEAD_NAMES
    }


def _memory_summaries(
    seeds: tuple[int, ...],
    reports: tuple[isolated.ProcessIsolatedMemoryResult, ...],
) -> dict[str, DescriptiveSummary]:
    if len(reports) != len(seeds):
        raise AssertionError("Capacity-memory summary requires every declared seed.")
    summaries: dict[str, DescriptiveSummary] = {}
    for name in tuning.HEAD_NAMES:
        values = tuple(report.reports[name].train_rss_peak_observed_bytes for report in reports)
        if all(value is not None for value in values):
            summaries[name] = _summarize(
                "train_rss_peak_observed_bytes",
                seeds,
                tuple(float(value) for value in values if value is not None),
            )
    return summaries


def _summarize(
    metric_name: str,
    seeds: tuple[int, ...],
    values: tuple[float, ...],
) -> DescriptiveSummary:
    if len(values) != len(seeds) or any(not isfinite(value) for value in values):
        raise AssertionError("Summary values must cover every seed and be finite.")
    return DescriptiveSummary(
        metric_name=metric_name,
        seeds=seeds,
        values=values,
        mean=mean(values),
        population_std=pstdev(values),
    )


def _report_field(head_name: str) -> str:
    if head_name == "backprop_mlp":
        return "backprop"
    if head_name == "circadian_predictive_coding":
        return "circadian"
    return head_name


def _digest(value: object) -> str:
    return sha256(
        dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default).encode()
    ).hexdigest()


def _manifest_digest(manifest: RepeatedConfirmationManifest) -> str:
    return _digest(
        {
            field.name: getattr(manifest, field.name)
            for field in fields(manifest)
            if field.name != "manifest_digest"
        }
    )


def _json_default(value: object) -> object:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)  # type: ignore[call-overload]
    raise TypeError(f"Cannot hash confirmation value of type {type(value).__name__}.")
