"""Validation-selected tuning for the frozen, shared-feature head track.

Every head receives the same declared candidate and seed budget. Candidate
training sees only cached train, guard, and outer-validation features. Final
test features are materialized once per seed after selections are frozen.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from math import isfinite
from typing import Any, Callable

from src.app import matched_head_benchmark as matched
from src.app.resnet50_benchmark import (
    ResNet50BenchmarkConfig,
    VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL,
    _build_circadian_head_config,
    _resolve_device,
    _set_seed,
    _validate_benchmark_config,
)
from src.core.resnet50_variants import (
    BackpropMLPHead,
    CircadianPredictiveCodingHead,
    PredictiveCodingHead,
)
from src.shared.torch_runtime import require_torch

MATCHED_HEAD_TUNING_PROTOCOL = "vision_matched_head_equal_trial_tuning_v1"
MATCHED_HEAD_VALIDATION_SELECTION_PROTOCOL = "vision_matched_head_validation_selection_v1"
HEAD_NAMES = ("backprop_mlp", "predictive_coding", "circadian_predictive_coding")
TUNABLE_FIELDS = {
    "backprop_mlp": ("backprop_learning_rate", "backprop_momentum"),
    "predictive_coding": (
        "predictive_learning_rate",
        "predictive_inference_steps",
        "predictive_inference_learning_rate",
    ),
    "circadian_predictive_coding": (
        "circadian_learning_rate",
        "circadian_inference_steps",
        "circadian_inference_learning_rate",
    ),
}
MAX_TRIALS_PER_HEAD = 8


@dataclass(frozen=True)
class HeadTuningCandidate:
    candidate_id: str
    config: ResNet50BenchmarkConfig


@dataclass(frozen=True)
class HeadTuningAttempt:
    head_name: str
    candidate_id: str
    seed: int
    config: ResNet50BenchmarkConfig
    status: str
    error: str | None = None


@dataclass(frozen=True)
class HeadTuningTrial:
    head_name: str
    candidate_id: str
    seed: int
    config: ResNet50BenchmarkConfig
    validation_accuracy: float
    initial_head_hash: str
    trained_head_hash: str
    epochs_ran: int
    seen_samples: int
    wake_batches: int
    latent_relaxation_steps: int
    guard_examples_scored: int
    validation_examples_scored: int
    sleep_attempts: int
    replay_examples: int
    initial_trainable_parameters: int
    trainable_parameters: int
    hidden_dim_start: int | None
    hidden_dim_end: int | None
    total_splits: int
    total_prunes: int
    total_rollbacks: int
    train_seconds: float
    split_hashes: dict[str, str]
    feature_hashes: dict[str, str]
    backbone_hash: str


@dataclass(frozen=True)
class HeadTuningSelection:
    head_name: str
    candidate_id: str
    mean_validation_accuracy: float
    trial_count: int
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class HeadTuningConfirmation:
    head_name: str
    candidate_id: str
    seed: int
    trained_head_hash: str
    test_accuracy: float
    test_cross_entropy: float
    test_split_hash: str
    test_feature_hash: str


@dataclass(frozen=True)
class MatchedHeadTuningResult:
    protocol_id: str
    benchmark_track: str
    base_config: ResNet50BenchmarkConfig
    seeds: tuple[int, ...]
    candidates_per_head: int
    trials_per_head: int
    attempts: tuple[HeadTuningAttempt, ...]
    trials: tuple[HeadTuningTrial, ...]
    selections: tuple[HeadTuningSelection, ...]
    confirmations: tuple[HeadTuningConfirmation, ...]


@dataclass(frozen=True)
class _SeedBank:
    loaders: Any
    backbone: Any
    train: matched.FeatureBatches
    guard: matched.FeatureBatches
    validation: matched.FeatureBatches
    backbone_hash: str
    split_hashes: dict[str, str]
    feature_hashes: dict[str, str]
    feature_dim: int
    num_classes: int


class MatchedHeadTuningError(RuntimeError):
    """An attempted trial or confirmation failed; completed evidence is retained."""

    def __init__(
        self,
        message: str,
        attempts: tuple[HeadTuningAttempt, ...],
        trials: tuple[HeadTuningTrial, ...],
        selections: tuple[HeadTuningSelection, ...] = (),
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.trials = trials
        self.selections = selections


def run_matched_head_tuning(
    base_config: ResNet50BenchmarkConfig,
    candidates: dict[str, tuple[HeadTuningCandidate, ...]],
    *,
    seeds: tuple[int, ...],
    candidates_per_head: int,
    confirm_test: bool = True,
    development_only_source: bool = False,
    attempt_observer: Callable[[HeadTuningAttempt, HeadTuningTrial | None], None] | None = None,
) -> MatchedHeadTuningResult:
    """Run equal-trial selection; optionally defer every final-test read."""
    if type(confirm_test) is not bool:
        raise ValueError("confirm_test must be a boolean.")
    if type(development_only_source) is not bool:
        raise ValueError("development_only_source must be a boolean.")
    if development_only_source and (
        confirm_test or base_config.dataset_name not in {"cifar10", "cifar100"}
    ):
        raise ValueError("development-only selection requires CIFAR and confirm_test=False")
    _validate_tuning_request(base_config, candidates, seeds, candidates_per_head)
    torch = require_torch()
    device = _resolve_device(torch, base_config.device)
    attempts: list[HeadTuningAttempt] = []
    trials: list[HeadTuningTrial] = []
    banks: dict[int, _SeedBank] = {}
    trained: dict[tuple[str, str, int], matched._TrainedHead] = {}
    for seed in seeds:
        seed_config = replace(base_config, seed=seed)
        _set_seed(torch, seed)
        bank = (
            _build_seed_bank(torch, device, seed_config, include_final_test=False)
            if development_only_source
            else _build_seed_bank(torch, device, seed_config)
        )
        banks[seed] = bank
        initial_hash: str | None = None
        for head_name in HEAD_NAMES:
            for candidate in candidates[head_name]:
                config = replace(candidate.config, seed=seed)
                if attempt_observer is not None:
                    attempt_observer(
                        HeadTuningAttempt(
                            head_name, candidate.candidate_id, seed, config, "started"
                        ),
                        None,
                    )
                try:
                    outcome, trial = _run_candidate_trial(
                        torch,
                        device,
                        bank,
                        config,
                        head_name,
                        candidate.candidate_id,
                        initial_hash,
                    )
                except Exception as error:
                    attempts.append(
                        HeadTuningAttempt(
                            head_name,
                            candidate.candidate_id,
                            seed,
                            config,
                            "failed",
                            f"{type(error).__name__}: {error}",
                        )
                    )
                    if attempt_observer is not None:
                        attempt_observer(attempts[-1], None)
                    raise MatchedHeadTuningError(
                        "Matched-head tuning trial failed before final test.",
                        tuple(attempts),
                        tuple(trials),
                    ) from error
                attempts.append(
                    HeadTuningAttempt(
                        head_name,
                        candidate.candidate_id,
                        seed,
                        config,
                        "complete",
                    )
                )
                initial_hash = trial.initial_head_hash
                trained[(head_name, candidate.candidate_id, seed)] = outcome
                trials.append(trial)
                if attempt_observer is not None:
                    attempt_observer(attempts[-1], trial)

    frozen_trials = tuple(trials)
    try:
        selections = _select_validation_candidates(
            frozen_trials,
            candidates,
            seeds,
            candidates_per_head,
        )
    except Exception as error:
        raise MatchedHeadTuningError(
            "Matched-head validation selection failed before final test.",
            tuple(attempts),
            frozen_trials,
        ) from error
    if not confirm_test:
        return MatchedHeadTuningResult(
            protocol_id=MATCHED_HEAD_VALIDATION_SELECTION_PROTOCOL,
            benchmark_track=matched.FROZEN_SHARED_REPRESENTATION_TRACK,
            base_config=base_config,
            seeds=seeds,
            candidates_per_head=candidates_per_head,
            trials_per_head=candidates_per_head * len(seeds),
            attempts=tuple(attempts),
            trials=frozen_trials,
            selections=selections,
            confirmations=(),
        )
    # No test-loader attribute is read above this line. Selection uses only
    # HeadTuningTrial, whose type has no final-test metric or label field.
    confirmations: list[HeadTuningConfirmation] = []
    try:
        for seed in seeds:
            bank = banks[seed]
            test = matched._materialize_role_features(
                torch,
                bank.backbone,
                bank.loaders.test_loader,
                device,
                seed + 104,
            )
            test_hash = matched._hash_batches(test)
            for selection in selections:
                outcome = trained[(selection.head_name, selection.candidate_id, seed)]
                logits = (
                    outcome.head.forward_logits
                    if selection.head_name == "backprop_mlp"
                    else outcome.head.predict_logits
                )
                accuracy, cross_entropy = matched._evaluate_head(
                    torch,
                    device,
                    logits,
                    test,
                    None,
                )
                confirmations.append(
                    HeadTuningConfirmation(
                        head_name=selection.head_name,
                        candidate_id=selection.candidate_id,
                        seed=seed,
                        trained_head_hash=matched._hash_trained_head(outcome.head),
                        test_accuracy=accuracy,
                        test_cross_entropy=cross_entropy,
                        test_split_hash=bank.loaders.split_hashes["test"],
                        test_feature_hash=test_hash,
                    )
                )
    except Exception as error:
        raise MatchedHeadTuningError(
            "Matched-head final confirmation failed after selection.",
            tuple(attempts),
            frozen_trials,
            selections,
        ) from error
    return MatchedHeadTuningResult(
        protocol_id=MATCHED_HEAD_TUNING_PROTOCOL,
        benchmark_track=matched.FROZEN_SHARED_REPRESENTATION_TRACK,
        base_config=base_config,
        seeds=seeds,
        candidates_per_head=candidates_per_head,
        trials_per_head=candidates_per_head * len(seeds),
        attempts=tuple(attempts),
        trials=frozen_trials,
        selections=selections,
        confirmations=tuple(confirmations),
    )


def _run_candidate_trial(
    torch: Any,
    device: Any,
    bank: _SeedBank,
    config: ResNet50BenchmarkConfig,
    head_name: str,
    candidate_id: str,
    expected_initial_hash: str | None,
) -> tuple[matched._TrainedHead, HeadTuningTrial]:
    # Candidate order cannot alter stochastic sleep or initialization.
    _set_seed(torch, config.seed + 11)
    head = _build_head(head_name, config, bank, device, config.seed + 11)
    initial_hash = matched._hash_head(head)
    if expected_initial_hash is not None and initial_hash != expected_initial_hash:
        raise AssertionError("Tuning heads must share identical initial tensors.")
    initial_parameters = head.parameter_count()
    outcome = _train_head(head_name, torch, device, head, bank, config)
    trial = HeadTuningTrial(
        head_name=head_name,
        candidate_id=candidate_id,
        seed=config.seed,
        config=config,
        validation_accuracy=outcome.validation_accuracy,
        initial_head_hash=initial_hash,
        trained_head_hash=matched._hash_trained_head(head),
        epochs_ran=outcome.epochs_ran,
        seen_samples=outcome.seen_samples,
        wake_batches=outcome.wake_batches,
        latent_relaxation_steps=outcome.latent_relaxation_steps,
        guard_examples_scored=outcome.guard_examples_scored,
        validation_examples_scored=matched._count_evaluated_examples(bank.validation, None),
        sleep_attempts=outcome.sleep_attempts,
        replay_examples=0,
        initial_trainable_parameters=initial_parameters,
        trainable_parameters=outcome.parameter_count,
        hidden_dim_start=outcome.hidden_dim_start,
        hidden_dim_end=outcome.hidden_dim_end,
        total_splits=outcome.total_splits,
        total_prunes=outcome.total_prunes,
        total_rollbacks=outcome.total_rollbacks,
        train_seconds=outcome.train_seconds,
        split_hashes=dict(bank.split_hashes),
        feature_hashes=dict(bank.feature_hashes),
        backbone_hash=bank.backbone_hash,
    )
    return outcome, trial


def _validate_tuning_request(
    base: ResNet50BenchmarkConfig,
    candidates: dict[str, tuple[HeadTuningCandidate, ...]],
    seeds: tuple[int, ...],
    candidates_per_head: int,
) -> None:
    _validate_benchmark_config(base)
    if base.protocol_id != VISION_GUARD_SEPARATED_UNMATCHED_PROTOCOL:
        raise ValueError("Tuning requires the guard-separated vision protocol.")
    if not base.backprop_freeze_backbone or base.target_accuracy is not None:
        raise ValueError("Tuning requires a frozen backbone and target_accuracy=None.")
    if base.predictive_head_hidden_dim != base.circadian_head_hidden_dim:
        raise ValueError("Tuning requires the same initial hidden width for all heads.")
    if (
        type(seeds) is not tuple
        or not seeds
        or any(type(seed) is not int for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds must be a nonempty tuple of distinct integers.")
    if (
        type(candidates_per_head) is not int
        or candidates_per_head <= 0
        or candidates_per_head * len(seeds) > MAX_TRIALS_PER_HEAD
    ):
        raise ValueError(f"Tuning allows 1–{MAX_TRIALS_PER_HEAD} trials per head.")
    if set(candidates) != set(HEAD_NAMES):
        raise ValueError("Exactly the three matched head families must have candidates.")
    config_fields = {field.name for field in fields(ResNet50BenchmarkConfig)}
    _validate_optimization_settings(base)
    for head_name in HEAD_NAMES:
        options = candidates[head_name]
        if len(options) != candidates_per_head:
            raise ValueError("Each head must have the declared equal candidate count.")
        ids: set[str] = set()
        settings: set[tuple[Any, ...]] = set()
        allowed = set(TUNABLE_FIELDS[head_name])
        for candidate in options:
            if not candidate.candidate_id or candidate.candidate_id in ids:
                raise ValueError("Candidate IDs must be nonempty and unique per head.")
            ids.add(candidate.candidate_id)
            _validate_benchmark_config(candidate.config)
            _validate_optimization_settings(candidate.config)
            changed = {
                name
                for name in config_fields
                if getattr(candidate.config, name) != getattr(base, name)
            }
            if not changed <= allowed:
                raise ValueError(
                    f"{head_name} candidate changes fixed settings: {sorted(changed - allowed)}"
                )
            values = tuple(getattr(candidate.config, name) for name in TUNABLE_FIELDS[head_name])
            if values in settings:
                raise ValueError("Duplicate per-head tuning settings waste an equal trial.")
            settings.add(values)


def _validate_optimization_settings(config: ResNet50BenchmarkConfig) -> None:
    rates = (
        config.backprop_learning_rate,
        config.predictive_learning_rate,
        config.predictive_inference_learning_rate,
        config.circadian_learning_rate,
        config.circadian_inference_learning_rate,
    )
    if any(not isfinite(rate) or rate <= 0.0 for rate in rates):
        raise ValueError("Tuning learning rates must be positive and finite.")
    if not isfinite(config.backprop_momentum) or not 0.0 <= config.backprop_momentum < 1.0:
        raise ValueError("Tuning backprop momentum must be finite and in [0, 1).")
    steps = (config.predictive_inference_steps, config.circadian_inference_steps)
    if any(type(step) is not int or step <= 0 for step in steps):
        raise ValueError("Tuning inference steps must be positive integers.")


def _build_seed_bank(
    torch: Any,
    device: Any,
    config: ResNet50BenchmarkConfig,
    *,
    include_final_test: bool = True,
) -> _SeedBank:
    if type(include_final_test) is not bool:
        raise ValueError("include_final_test must be a boolean")
    loaders = (
        matched._build_benchmark_loaders(config)
        if include_final_test
        else matched._build_benchmark_loaders(config, include_final_test=False)
    )
    if "guard" not in loaders.split_hashes or loaders.guard_loader is loaders.validation_loader:
        raise ValueError("Tuning requires a distinct guard loader.")
    backbone, feature_dim = matched._build_resnet50_backbone(
        device=device,
        freeze_backbone=True,
        backbone_weights=config.backbone_weights,
    )
    backbone.eval()
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
    return _SeedBank(
        loaders=loaders,
        backbone=backbone,
        train=batches["train"],
        guard=batches["guard"],
        validation=batches["validation"],
        backbone_hash=matched._hash_named_tensors(tuple(backbone.state_dict().items())),
        split_hashes={role: loaders.split_hashes[role] for role in batches},
        feature_hashes={role: matched._hash_batches(value) for role, value in batches.items()},
        feature_dim=feature_dim,
        num_classes=loaders.num_classes,
    )


def _build_head(
    name: str,
    config: ResNet50BenchmarkConfig,
    bank: _SeedBank,
    device: Any,
    head_seed: int,
) -> Any:
    if name == "backprop_mlp":
        return BackpropMLPHead(
            bank.feature_dim,
            config.predictive_head_hidden_dim,
            bank.num_classes,
            device,
            head_seed,
        )
    if name == "predictive_coding":
        return PredictiveCodingHead(
            bank.feature_dim,
            config.predictive_head_hidden_dim,
            bank.num_classes,
            device,
            head_seed,
        )
    return CircadianPredictiveCodingHead(
        feature_dim=bank.feature_dim,
        hidden_dim=config.circadian_head_hidden_dim,
        num_classes=bank.num_classes,
        device=device,
        seed=head_seed,
        config=_build_circadian_head_config(config),
        min_hidden_dim=config.circadian_min_hidden_dim,
        max_hidden_dim=config.circadian_max_hidden_dim,
    )


def _train_head(
    name: str,
    torch: Any,
    device: Any,
    head: Any,
    bank: _SeedBank,
    config: ResNet50BenchmarkConfig,
) -> matched._TrainedHead:
    if name == "backprop_mlp":
        return matched._train_backprop_head(
            torch,
            device,
            head,
            bank.train,
            bank.guard,
            bank.validation,
            config,
        )
    if name == "predictive_coding":
        return matched._train_predictive_head(
            torch,
            device,
            head,
            bank.train,
            bank.guard,
            bank.validation,
            config,
        )
    return matched._train_circadian_head(
        torch,
        device,
        head,
        bank.train,
        bank.guard,
        bank.validation,
        config,
    )


def _select_validation_candidates(
    trials: tuple[HeadTuningTrial, ...],
    candidates: dict[str, tuple[HeadTuningCandidate, ...]],
    seeds: tuple[int, ...],
    candidates_per_head: int,
) -> tuple[HeadTuningSelection, ...]:
    selections: list[HeadTuningSelection] = []
    for head_name in HEAD_NAMES:
        scores: list[tuple[float, str]] = []
        for candidate in candidates[head_name]:
            rows = tuple(
                trial
                for trial in trials
                if trial.head_name == head_name and trial.candidate_id == candidate.candidate_id
            )
            if len(rows) != len(seeds) or {row.seed for row in rows} != set(seeds):
                raise AssertionError("Every candidate must complete the same seeds.")
            if any(
                not isfinite(row.validation_accuracy) or not 0.0 <= row.validation_accuracy <= 1.0
                for row in rows
            ):
                raise ValueError("Validation accuracy must be finite and in [0, 1].")
            scores.append(
                (sum(row.validation_accuracy for row in rows) / len(rows), candidate.candidate_id)
            )
        if len(scores) != candidates_per_head:
            raise AssertionError("Selection must use the declared trial budget.")
        # max() keeps the first predeclared candidate on a validation tie.
        best_score, best_id = max(scores, key=lambda item: item[0])
        selections.append(
            HeadTuningSelection(
                head_name=head_name,
                candidate_id=best_id,
                mean_validation_accuracy=best_score,
                trial_count=candidates_per_head * len(seeds),
                seeds=seeds,
            )
        )
    return tuple(selections)
