"""Release common final roles after all fixed v14 trigger trials preflight.

Inputs are the prospective full-stack manifest and six train-only trials.
Outputs are every seed/arm/method metric and paired contrast. This module
does not tune triggers, choose a model, checkpoint, or write files.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.app.continual_matched_replay_runner import AppliedMethodReplay
from src.app.continual_trigger_replay_runner import MethodWakeWork, TriggerReplayTrainingResult
from src.app.continual_trigger_replay_schedule import (
    TriggerReplayOpportunityManifest,
    _validate_manifest,
)
from src.app.continual_trigger_replay_training_study import (
    TriggerReplayTrainingStudy,
    preflight_trigger_replay_training_study,
    run_trigger_replay_training_study,
)
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import (
    CircadianPredictiveCodingNetwork,
    ReplayRetentionSnapshot,
)
from src.core.predictive_coding import PredictiveCodingNetwork
from src.core.replay_retention import ReplayExposureSnapshot
from src.infra.continual_roles import PhaseDecisionRoles, release_final_test
from src.infra.datasets import LabeledData


TRIGGER_REPLAY_OUTCOMES_PROTOCOL = "continual_trigger_replay_outcomes_v14"
Model = BackpropMLP | PredictiveCodingNetwork | CircadianPredictiveCodingNetwork


@dataclass(frozen=True)
class MethodTriggerOutcome:
    """One method's fixed final-role metrics, exposure, and capacity."""

    method: str
    a_after_a_accuracy: float
    a_after_a_bce: float
    a_after_b_accuracy: float
    a_after_b_bce: float
    b_after_b_accuracy: float
    b_after_b_bce: float
    signed_forgetting: float
    balanced_score: float
    wake_work: MethodWakeWork
    replay_work: AppliedMethodReplay
    width_after_a: int
    width_after_b: int
    parameters_after_a: int
    parameters_after_b: int


@dataclass(frozen=True)
class TriggerTrialOutcome:
    """All three methods on common A/B final roles for one arm and seed."""

    seed: int
    arm: str
    final_role_hashes: tuple[tuple[str, str], ...]
    final_role_ids: tuple[tuple[str, tuple[str, ...]], ...]
    methods: tuple[MethodTriggerOutcome, ...]
    sleep_attempts: int
    sleep_accepted: int
    sleep_rolled_back: int
    applied_splits: int
    applied_prunes: int
    phase_a_replay_retention: ReplayRetentionSnapshot
    phase_b_replay_retention: ReplayRetentionSnapshot
    circadian_replay_exposure: ReplayExposureSnapshot


@dataclass(frozen=True)
class TriggerPairContrast:
    """Signed left-minus-right effect within the same seed and method."""

    seed: int
    method: str
    left_arm: str
    right_arm: str
    a_after_b_accuracy_delta: float
    a_after_b_bce_delta: float
    b_after_b_accuracy_delta: float
    b_after_b_bce_delta: float
    signed_forgetting_delta: float
    balanced_score_delta: float
    replay_updates_delta: int
    final_parameters_delta: int


@dataclass(frozen=True)
class TriggerReplayComparison:
    protocol_id: str
    manifest: TriggerReplayOpportunityManifest
    manifest_digest: str
    outcomes: tuple[TriggerTrialOutcome, ...]
    contrasts: tuple[TriggerPairContrast, ...]


def run_trigger_replay_outcomes(
    manifest: TriggerReplayOpportunityManifest,
) -> TriggerReplayComparison:
    """Freeze all six trials, release common finals, then score every cell."""
    _validate_manifest(manifest)
    study = run_trigger_replay_training_study(manifest)
    return score_trigger_replay_training_study(study)


def score_trigger_replay_training_study(
    study: TriggerReplayTrainingStudy,
) -> TriggerReplayComparison:
    """Score an existing unscored study only after repeating its global gate."""
    manifest = study.manifest
    _validate_manifest(manifest)
    preflight_trigger_replay_training_study(study)
    released = _release_all_final_roles(study)
    _validate_common_final_roles(study, released)
    outcomes = tuple(
        _score_trial(trial, phases) for trial, phases in zip(study.trials, released, strict=True)
    )
    return TriggerReplayComparison(
        TRIGGER_REPLAY_OUTCOMES_PROTOCOL,
        manifest,
        study.manifest_digest,
        outcomes,
        _pair_contrasts(manifest, outcomes),
    )


def _release_all_final_roles(
    study: TriggerReplayTrainingStudy,
) -> tuple[tuple[PhaseDecisionRoles, PhaseDecisionRoles], ...]:
    # Why this: a missing/failed trial must stop before *any* final label.
    return tuple(
        (release_final_test(trial.pending.phase_a), release_final_test(trial.pending.phase_b))
        for trial in study.trials
    )


def _validate_common_final_roles(
    study: TriggerReplayTrainingStudy,
    released: tuple[tuple[PhaseDecisionRoles, PhaseDecisionRoles], ...],
) -> None:
    if len(released) != len(study.trials):
        raise ValueError("v14 final roles require all fixed trials")
    by_seed: dict[int, tuple[PhaseDecisionRoles, PhaseDecisionRoles]] = {}
    for trial, phases in zip(study.trials, released, strict=True):
        if any(bound.seed != trial.seed or not bound.final_released for bound in phases):
            raise ValueError("v14 released final role seed or state differs")
        previous = by_seed.setdefault(trial.seed, phases)
        for actual, expected in zip(phases, previous, strict=True):
            if (
                actual.phase != expected.phase
                or actual.sample_ids["final_test"] != expected.sample_ids["final_test"]
                or actual.split_hashes["final_test"] != expected.split_hashes["final_test"]
            ):
                raise ValueError("v14 common final roles differ across arms")


def _score_trial(
    trial: TriggerReplayTrainingResult,
    phases: tuple[PhaseDecisionRoles, PhaseDecisionRoles],
) -> TriggerTrialOutcome:
    bound_a, bound_b = phases
    if bound_a.final_test is None or bound_b.final_test is None:
        raise ValueError("v14 final role release is incomplete")
    trial.pending.audit.record_final_release("a")
    trial.pending.audit.record_final_release("b")
    state = trial.pending.state
    models = (
        ("backprop", state.backprop_after_a, state.backprop_model),
        ("predictive_coding", state.predictive_after_a, state.predictive_model),
        ("circadian_predictive_coding", state.circadian_after_a, state.circadian_model),
    )
    methods = tuple(
        _score_method(trial, method, after_a, after_b, bound_a.final_test, bound_b.final_test)
        for method, after_a, after_b in models
    )
    events = trial.pending.sleep_events
    return TriggerTrialOutcome(
        trial.seed,
        trial.arm,
        (("a", bound_a.split_hashes["final_test"]), ("b", bound_b.split_hashes["final_test"])),
        (("a", bound_a.sample_ids["final_test"]), ("b", bound_b.sample_ids["final_test"])),
        methods,
        sum(event.outcome in {"accepted", "rolled_back"} for event in events),
        sum(event.outcome == "accepted" for event in events),
        sum(event.outcome == "rolled_back" for event in events),
        state.total_splits,
        state.total_prunes,
        state.circadian_after_a.get_replay_retention(),
        state.circadian_model.get_replay_retention(),
        state.circadian_model.get_replay_exposure(),
    )


def _score_method(
    trial: TriggerReplayTrainingResult,
    method: str,
    after_a: Model,
    after_b: Model,
    final_a: LabeledData,
    final_b: LabeledData,
) -> MethodTriggerOutcome:
    a_after_a, a_after_a_bce = _score(after_a, final_a)
    a_after_b, a_after_b_bce = _score(after_b, final_a)
    b_after_b, b_after_b_bce = _score(after_b, final_b)
    wake = next(work for work in trial.wake_work if work.method == method)
    replay = _sum_replay(trial, method)
    width_a, width_b = _width(after_a), _width(after_b)
    return MethodTriggerOutcome(
        method,
        a_after_a,
        a_after_a_bce,
        a_after_b,
        a_after_b_bce,
        b_after_b,
        b_after_b_bce,
        a_after_a - a_after_b,
        0.5 * (a_after_b + b_after_b),
        wake,
        replay,
        width_a,
        width_b,
        _parameter_count(after_a),
        _parameter_count(after_b),
    )


def _score(model: Model, batch: LabeledData) -> tuple[float, float]:
    probabilities = np.asarray(model.predict_proba(batch.input), dtype=np.float64)
    if probabilities.shape != batch.target.shape or not np.all(np.isfinite(probabilities)):
        raise ValueError("v14 final prediction shape or values are invalid")
    accuracy = float(np.mean((probabilities >= 0.5) == batch.target))
    clipped = np.clip(probabilities, 1e-8, 1.0 - 1e-8)
    bce = float(
        np.mean(-batch.target * np.log(clipped) - (1.0 - batch.target) * np.log(1.0 - clipped))
    )
    return accuracy, bce


def _sum_replay(trial: TriggerReplayTrainingResult, method: str) -> AppliedMethodReplay:
    work = tuple(
        item
        for opportunity in trial.opportunities
        for item in opportunity.applied_by_method
        if item.method == method
    )
    return AppliedMethodReplay(
        method,
        tuple(sample_id for item in work for sample_id in item.sample_ids),
        sum(item.examples for item in work),
        sum(item.optimizer_updates for item in work),
        sum(item.inference_iterations for item in work),
    )


def _width(model: Model) -> int:
    return (
        model.hidden_dim
        if isinstance(model, CircadianPredictiveCodingNetwork)
        else model.hidden_dims[-1]
    )


def _parameter_count(model: Model) -> int:
    return int(
        sum(
            values.size
            for values in (
                model.weight_input_hidden,
                model.bias_hidden,
                model.weight_hidden_output,
                model.bias_output,
            )
        )
    )


def _pair_contrasts(
    manifest: TriggerReplayOpportunityManifest,
    outcomes: tuple[TriggerTrialOutcome, ...],
) -> tuple[TriggerPairContrast, ...]:
    cells = {(item.seed, item.arm): item for item in outcomes}
    pairs = (("periodic", "no_sleep"), ("adaptive", "no_sleep"), ("periodic", "adaptive"))
    return tuple(
        _contrast(seed, method, left, right, cells[(seed, left)], cells[(seed, right)])
        for seed in manifest.seeds
        for method in manifest.arrived.training.model_order
        for left, right in pairs
    )


def _contrast(
    seed: int,
    method: str,
    left_arm: str,
    right_arm: str,
    left_trial: TriggerTrialOutcome,
    right_trial: TriggerTrialOutcome,
) -> TriggerPairContrast:
    left = next(item for item in left_trial.methods if item.method == method)
    right = next(item for item in right_trial.methods if item.method == method)
    return TriggerPairContrast(
        seed,
        method,
        left_arm,
        right_arm,
        left.a_after_b_accuracy - right.a_after_b_accuracy,
        left.a_after_b_bce - right.a_after_b_bce,
        left.b_after_b_accuracy - right.b_after_b_accuracy,
        left.b_after_b_bce - right.b_after_b_bce,
        left.signed_forgetting - right.signed_forgetting,
        left.balanced_score - right.balanced_score,
        left.replay_work.optimizer_updates - right.replay_work.optimizer_updates,
        left.parameters_after_b - right.parameters_after_b,
    )
