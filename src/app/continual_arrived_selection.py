"""Ordinary, outer-selected continual comparison with one global final seal.

Inputs are two to four predeclared v6 candidates and ordered seeds. Outputs
retain every outer-role trial, an immutable per-method choice, and selected
final scores. Checkpoint continuation is delegated to a separate app module;
this module does not read files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
import json
from typing import Any, Protocol

import numpy as np

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_shift_benchmark as base
from src.app.continual_arrived_selection_checkpoint import ArrivedSelectionCheckpointStore
from src.core.circadian_predictive_coding import ReplayRetentionSnapshot
from src.core.sleep_telemetry import SleepEventTelemetry
from src.infra.continual_roles import release_final_test


ARRIVED_SELECTION_PROTOCOL = "continual_arrived_outer_selection_v7"
_SELECTION_OBJECTIVE = "mean_0.5_phase_a_post_plus_0.5_phase_b_post_outer_accuracy"
_MAX_CANDIDATES = 4
_MAX_TRIALS_PER_METHOD = 8
_LEARNING_RATE_FIELDS = (
    "backprop_learning_rate",
    "pc_learning_rate",
    "circadian_learning_rate",
)
_METHOD_RATE_FIELD = dict(zip(base.CONTINUAL_MODEL_ORDER, _LEARNING_RATE_FIELDS, strict=True))


class _AccuracyModel(Protocol):
    def compute_accuracy(self, input_batch: np.ndarray, target_batch: np.ndarray) -> float: ...


@dataclass(frozen=True)
class ArrivedSelectionCandidate:
    candidate_id: str
    config: arrived.ContinualArrivedRolesConfig


@dataclass(frozen=True)
class ArrivedOuterTrial:
    candidate_id: str
    seed: int
    method: str
    phase_a_pre_accuracy: float
    phase_a_post_accuracy: float
    phase_b_post_accuracy: float
    balanced_score: float
    phase_a_outer_hash: str
    phase_b_outer_hash: str
    development_role_ids: tuple[tuple[str, tuple[str, ...]], ...]
    development_role_hashes: tuple[tuple[str, str], ...]
    train_updates: int
    train_examples_seen: int
    inner_guard_examples_scored: int
    outer_examples_scored: int
    sleep_event_count: int
    replay_phase_a: ReplayRetentionSnapshot | None
    replay_phase_b: ReplayRetentionSnapshot | None
    role_accesses: tuple[arrived.RoleAccessEvent, ...]
    outer_accesses: tuple[arrived.RoleAccessEvent, ...]
    guard_decisions: tuple[arrived.GuardDecision, ...]
    method_task_information: tuple[arrived.MethodTaskInformation, ...]
    sleep_events: tuple[SleepEventTelemetry, ...] = field(compare=False)
    sleep_history_digest: str = field(compare=False)


@dataclass(frozen=True)
class ArrivedSelectionChoice:
    method: str
    candidate_id: str
    mean_outer_balanced_score: float
    trial_count: int


@dataclass(frozen=True)
class ArrivedSelectionFreeze:
    objective: str
    candidate_manifest_digest: str
    trial_digest: str
    choices: tuple[ArrivedSelectionChoice, ...]
    freeze_digest: str
    sleep_history_digest: str = field(compare=False)


@dataclass(frozen=True)
class ArrivedSelectedSeedResult:
    seed: int
    metrics: base.ContinualBoundedReplaySeedResult
    role_ids: dict[str, tuple[str, ...]]
    role_hashes: dict[str, str]
    final_role_accesses: tuple[arrived.RoleAccessEvent, ...]


@dataclass(frozen=True)
class ArrivedCandidateSleepHistory:
    """Observed v6 sleep attempts for one candidate and seed before selection."""

    candidate_id: str
    seed: int
    # Why this: measured elapsed time is evidence, not a selection input.
    sleep_events: tuple[SleepEventTelemetry, ...] = field(compare=False)


@dataclass(frozen=True)
class ArrivedOuterSelectionResult:
    protocol_id: str
    candidates: tuple[ArrivedSelectionCandidate, ...]
    candidate_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    trials: tuple[ArrivedOuterTrial, ...]
    selections: tuple[ArrivedSelectionChoice, ...]
    freeze: ArrivedSelectionFreeze
    final_seed_results: tuple[ArrivedSelectedSeedResult, ...]
    aggregate: base.ContinualShiftAggregate
    candidate_sleep_histories: tuple[ArrivedCandidateSleepHistory, ...] = field(compare=False)


def run_arrived_outer_selection(
    candidates: tuple[ArrivedSelectionCandidate, ...],
    seeds: list[int],
    *,
    checkpoint_store: ArrivedSelectionCheckpointStore | None = None,
    resume_from_checkpoint: bool = False,
    sleep_error_retries: int = 0,
) -> ArrivedOuterSelectionResult:
    """Select every method from arrived outer roles before releasing final fields."""
    _validate_selection_request(candidates, seeds)
    if type(sleep_error_retries) is not int or sleep_error_retries < 0:
        raise ValueError("v7 sleep error retries must be a nonnegative integer")
    if checkpoint_store is not None and sleep_error_retries:
        raise ValueError("checkpointed v7 sleep errors require explicit resume")
    if resume_from_checkpoint and checkpoint_store is None:
        raise ValueError("v7 resume requires a candidate checkpoint store")
    if checkpoint_store is None:
        pending = _train_all_candidates(candidates, seeds, sleep_error_retries)
        _validate_shared_roles(candidates, seeds, pending)
        trials = tuple(
            _score_outer_trial(
                candidate.candidate_id, pending[candidate.candidate_id, seed], method
            )
            for candidate in candidates
            for seed in seeds
            for method in base.CONTINUAL_MODEL_ORDER
        )
        freeze = _freeze_selection(candidates, tuple(seeds), trials)
    else:
        from src.app.continual_arrived_selection_resume import run_or_resume_selection

        pending, trials, freeze = run_or_resume_selection(
            candidates, seeds, checkpoint_store, resume_from_checkpoint
        )
    final = tuple(_score_selected_seed(candidates, freeze.choices, seed, pending) for seed in seeds)
    return ArrivedOuterSelectionResult(
        protocol_id=ARRIVED_SELECTION_PROTOCOL,
        candidates=candidates,
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        seeds=tuple(seeds),
        trials=trials,
        selections=freeze.choices,
        freeze=freeze,
        final_seed_results=final,
        aggregate=_aggregate_selected(final),
        candidate_sleep_histories=tuple(
            ArrivedCandidateSleepHistory(
                candidate.candidate_id,
                seed,
                pending[candidate.candidate_id, seed].sleep_events,
            )
            for candidate in candidates
            for seed in seeds
        ),
    )


def _validate_selection_request(
    candidates: tuple[ArrivedSelectionCandidate, ...], seeds: list[int]
) -> None:
    if type(candidates) is not tuple or not 2 <= len(candidates) <= _MAX_CANDIDATES:
        raise ValueError("v7 needs two to four predeclared candidates")
    if type(seeds) is not list or len(candidates) * len(seeds) > _MAX_TRIALS_PER_METHOD:
        raise ValueError("v7 exceeds its eight-trial-per-method local budget")
    if any(type(item) is not ArrivedSelectionCandidate for item in candidates):
        raise ValueError("v7 candidates must have typed IDs and configs")
    ids = tuple(item.candidate_id for item in candidates)
    if any(type(identifier) is not str or not identifier.strip() for identifier in ids) or len(
        set(ids)
    ) != len(ids):
        raise ValueError("v7 candidate IDs must be nonempty and unique")
    for candidate in candidates:
        arrived._validate_arrived_config(candidate.config, seeds)
    first = candidates[0].config
    for candidate in candidates[1:]:
        config = candidate.config
        if (
            config.inner_guard_fraction != first.inner_guard_fraction
            or config.outer_selection_fraction != first.outer_selection_fraction
            or config.guard_drop_tolerance != first.guard_drop_tolerance
            or replace(
                config.training,
                **{name: getattr(first.training, name) for name in _LEARNING_RATE_FIELDS},
            )
            != first.training
        ):
            raise ValueError("v7 candidates must share fixed work, roles, and guard settings")
    for method, rate_field in _METHOD_RATE_FIELD.items():
        rates = tuple(getattr(item.config.training, rate_field) for item in candidates)
        if len(set(rates)) != len(rates) or any(not np.isfinite(rate) for rate in rates):
            raise ValueError(f"v7 {method} candidates need distinct learning rates")


def _train_all_candidates(
    candidates: tuple[ArrivedSelectionCandidate, ...],
    seeds: list[int],
    sleep_error_retries: int,
) -> dict[tuple[str, int], arrived._PendingSeed]:
    pending: dict[tuple[str, int], arrived._PendingSeed] = {}
    for candidate in candidates:
        for seed in seeds:
            # Why this: zero keeps the existing ordinary call boundary;
            # retries opt into v6's bounded, typed error-only behavior.
            pending[candidate.candidate_id, seed] = (
                arrived._train_arrived_seed(candidate.config, seed, sleep_error_retries)
                if sleep_error_retries
                else arrived._train_arrived_seed(candidate.config, seed)
            )
    return pending


def _validate_shared_roles(
    candidates: tuple[ArrivedSelectionCandidate, ...],
    seeds: list[int],
    pending: dict[tuple[str, int], arrived._PendingSeed],
) -> None:
    for seed in seeds:
        reference = pending[candidates[0].candidate_id, seed]
        expected = arrived._development_identity(reference.phase_a, reference.phase_b)
        for candidate in candidates[1:]:
            item = pending[candidate.candidate_id, seed]
            if arrived._development_identity(item.phase_a, item.phase_b) != expected:
                raise ValueError("v7 candidates received different development roles")


def _score_outer_trial(
    candidate_id: str, pending: arrived._PendingSeed, method: str
) -> ArrivedOuterTrial:
    state = pending.state
    pre_a, post_a, post_b = _outer_accuracies(pending, method)
    accesses = tuple(
        arrived.RoleAccessEvent(
            pending.seed, phase, "outer_selection", "outer_selection", "candidate_selection", method
        )
        for phase in ("a", "a", "b")
    )
    role_ids, role_hashes = arrived._development_identity(pending.phase_a, pending.phase_b)
    if {item.method for item in pending.audit.task_information} != set(base.CONTINUAL_MODEL_ORDER):
        raise ValueError("v7 candidate training ledger is incomplete")
    updates, train_examples, guard_examples = _trial_work_counts(pending, method)
    circadian = method == "circadian_predictive_coding"
    sleep_events = pending.sleep_events if circadian else ()
    sleep_history_digest = (
        arrived.arrived_sleep_history_digest(
            sleep_events,
            arrived.arrived_event_digest(
                tuple(pending.audit.accesses),
                tuple(pending.audit.guard_decisions),
                tuple(pending.audit.task_information),
            ),
        )
        if circadian
        else ""
    )
    return ArrivedOuterTrial(
        candidate_id=candidate_id,
        seed=pending.seed,
        method=method,
        phase_a_pre_accuracy=pre_a,
        phase_a_post_accuracy=post_a,
        phase_b_post_accuracy=post_b,
        balanced_score=0.5 * (post_a + post_b),
        phase_a_outer_hash=pending.phase_a.split_hashes["outer_selection"],
        phase_b_outer_hash=pending.phase_b.split_hashes["outer_selection"],
        development_role_ids=role_ids,
        development_role_hashes=role_hashes,
        train_updates=updates,
        train_examples_seen=train_examples,
        inner_guard_examples_scored=guard_examples,
        outer_examples_scored=(
            2 * len(pending.phase_a.outer_selection.input)
            + len(pending.phase_b.outer_selection.input)
        ),
        sleep_event_count=state.sleep_event_count if circadian else 0,
        replay_phase_a=state.circadian_after_a.get_replay_retention() if circadian else None,
        replay_phase_b=state.circadian_model.get_replay_retention() if circadian else None,
        role_accesses=tuple(pending.audit.accesses) + accesses,
        outer_accesses=accesses,
        guard_decisions=tuple(pending.audit.guard_decisions),
        method_task_information=tuple(
            item for item in pending.audit.task_information if item.method == method
        ),
        sleep_events=sleep_events,
        sleep_history_digest=sleep_history_digest,
    )


def _outer_accuracies(pending: arrived._PendingSeed, method: str) -> tuple[float, float, float]:
    state = pending.state
    after_a: _AccuracyModel
    final: _AccuracyModel
    if method == "backprop":
        after_a, final = state.backprop_after_a, state.backprop_model
    elif method == "predictive_coding":
        after_a, final = state.predictive_after_a, state.predictive_model
    else:
        after_a, final = state.circadian_after_a, state.circadian_model
    outer_a = pending.phase_a.outer_selection
    outer_b = pending.phase_b.outer_selection
    pre_a = float(after_a.compute_accuracy(outer_a.input, outer_a.target))
    post_a = float(final.compute_accuracy(outer_a.input, outer_a.target))
    post_b = float(final.compute_accuracy(outer_b.input, outer_b.target))
    if any(not np.isfinite(score) or not 0.0 <= score <= 1.0 for score in (pre_a, post_a, post_b)):
        raise ValueError("v7 outer accuracy must be finite and in [0, 1]")
    return pre_a, post_a, post_b


def _trial_work_counts(pending: arrived._PendingSeed, method: str) -> tuple[int, int, int]:
    guard_exposures = (
        2
        * sum(
            len(pending.phase_a.inner_guard.input)
            if decision.phase == "a"
            else len(pending.phase_b.inner_guard.input)
            for decision in pending.audit.guard_decisions
        )
        if method == "circadian_predictive_coding"
        else 0
    )
    train_epochs_a = sum(
        event.action == "train" and event.phase == "a" and event.method == method
        for event in pending.audit.accesses
    )
    train_epochs_b = sum(
        event.action == "train" and event.phase == "b" and event.method == method
        for event in pending.audit.accesses
    )
    return (
        train_epochs_a + train_epochs_b,
        (
            train_epochs_a * len(pending.phase_a.train.input)
            + train_epochs_b * len(pending.phase_b.train.input)
        ),
        guard_exposures,
    )


def _freeze_selection(
    candidates: tuple[ArrivedSelectionCandidate, ...],
    seeds: tuple[int, ...],
    trials: tuple[ArrivedOuterTrial, ...],
) -> ArrivedSelectionFreeze:
    expected = len(candidates) * len(seeds) * len(base.CONTINUAL_MODEL_ORDER)
    if len(trials) != expected:
        raise ValueError("v7 cannot freeze an incomplete trial grid")
    choices = tuple(
        _choose_method(method, candidates, seeds, trials) for method in base.CONTINUAL_MODEL_ORDER
    )
    manifest = _candidate_manifest_digest(candidates, seeds)
    trial_digest = _trial_digest(trials)
    return ArrivedSelectionFreeze(
        objective=_SELECTION_OBJECTIVE,
        candidate_manifest_digest=manifest,
        trial_digest=trial_digest,
        choices=choices,
        freeze_digest=_hash_json((manifest, trial_digest, [asdict(item) for item in choices])),
        sleep_history_digest=_sleep_history_digest(trials),
    )


def _trial_digest(trials: tuple[ArrivedOuterTrial, ...]) -> str:
    """Keep the historical score/work trial identity independent of telemetry."""
    rows = [asdict(item) for item in trials]
    for row in rows:
        row.pop("sleep_events")
        row.pop("sleep_history_digest")
    return _hash_json(rows)


def _sleep_history_digest(trials: tuple[ArrivedOuterTrial, ...]) -> str:
    """Bind the ordered candidate/seed sleep histories beside trial identity."""
    rows = [
        (item.candidate_id, item.seed, item.sleep_history_digest)
        for item in trials
        if item.method == "circadian_predictive_coding"
    ]
    if not rows or any(len(digest) != 64 for _, _, digest in rows):
        raise ValueError("v7 cannot freeze incomplete candidate sleep history")
    return _hash_json(rows)


def _candidate_manifest_digest(
    candidates: tuple[ArrivedSelectionCandidate, ...], seeds: tuple[int, ...]
) -> str:
    return _hash_json(
        {
            "protocol": ARRIVED_SELECTION_PROTOCOL,
            "objective": _SELECTION_OBJECTIVE,
            "seeds": seeds,
            "candidates": [
                (item.candidate_id, arrived.arrived_config_digest(item.config, seeds))
                for item in candidates
            ],
        }
    )


def _choose_method(
    method: str,
    candidates: tuple[ArrivedSelectionCandidate, ...],
    seeds: tuple[int, ...],
    trials: tuple[ArrivedOuterTrial, ...],
) -> ArrivedSelectionChoice:
    scores: list[tuple[float, str]] = []
    for candidate in candidates:
        rows = [
            item
            for item in trials
            if item.method == method and item.candidate_id == candidate.candidate_id
        ]
        if tuple(item.seed for item in rows) != seeds or any(
            not np.isfinite(item.balanced_score) or not 0.0 <= item.balanced_score <= 1.0
            for item in rows
        ):
            raise ValueError("v7 trial grid or outer score is invalid")
        scores.append(
            (sum(item.balanced_score for item in rows) / len(seeds), candidate.candidate_id)
        )
    # Why this: the first predeclared candidate wins an exact outer-score tie.
    best_score, best_id = max(scores, key=lambda item: item[0])
    return ArrivedSelectionChoice(method, best_id, best_score, len(scores) * len(seeds))


def _score_selected_seed(
    candidates: tuple[ArrivedSelectionCandidate, ...],
    choices: tuple[ArrivedSelectionChoice, ...],
    seed: int,
    pending: dict[tuple[str, int], arrived._PendingSeed],
) -> ArrivedSelectedSeedResult:
    selected = {item.method: pending[item.candidate_id, seed].state for item in choices}
    reference = pending[candidates[0].candidate_id, seed]
    releases: list[arrived.RoleAccessEvent] = []
    bound_a = release_final_test(reference.phase_a)
    releases.extend(_final_release_events(seed, "a"))
    bound_b = release_final_test(reference.phase_b)
    releases.extend(_final_release_events(seed, "b"))
    assert bound_a.final_test is not None and bound_b.final_test is not None
    state = _combine_selected_models(reference.state, selected)
    role_ids = {
        f"phase_{phase}_{role}": ids
        for phase, bound in (("a", bound_a), ("b", bound_b))
        for role, ids in bound.sample_ids.items()
    }
    role_hashes = {
        f"phase_{phase}_{role}": digest
        for phase, bound in (("a", bound_a), ("b", bound_b))
        for role, digest in bound.split_hashes.items()
    }
    circadian_candidate_id = next(
        item.candidate_id for item in choices if item.method == "circadian_predictive_coding"
    )
    metrics = base._score_seed_models(
        candidates[0].config.training,
        seed,
        state,
        bound_a.final_test,
        bound_b.final_test,
        role_hashes,
        sleep_events=pending[circadian_candidate_id, seed].sleep_events,
    )
    assert isinstance(metrics, base.ContinualBoundedReplaySeedResult)
    return ArrivedSelectedSeedResult(seed, metrics, role_ids, role_hashes, tuple(releases))


def _combine_selected_models(
    reference: base._ContinualTrainingState,
    selected: dict[str, base._ContinualTrainingState],
) -> base._ContinualTrainingState:
    circadian = selected["circadian_predictive_coding"]
    return replace(
        reference,
        backprop_model=selected["backprop"].backprop_model,
        backprop_after_a=selected["backprop"].backprop_after_a,
        predictive_model=selected["predictive_coding"].predictive_model,
        predictive_after_a=selected["predictive_coding"].predictive_after_a,
        circadian_model=circadian.circadian_model,
        circadian_after_a=circadian.circadian_after_a,
        sleep_event_count=circadian.sleep_event_count,
        total_splits=circadian.total_splits,
        total_prunes=circadian.total_prunes,
        hidden_dim_start=circadian.hidden_dim_start,
    )


def _final_release_events(seed: int, phase: str) -> tuple[arrived.RoleAccessEvent, ...]:
    return tuple(
        arrived.RoleAccessEvent(seed, phase, "final_test", action, "global_freeze")
        for action in ("source_release", "label_release")
    )


def _aggregate_selected(
    scored: tuple[ArrivedSelectedSeedResult, ...],
) -> base.ContinualShiftAggregate:
    return base.ContinualShiftAggregate(
        run_count=len(scored),
        backprop=base._aggregate_model_stats([item.metrics.backprop for item in scored]),
        predictive_coding=base._aggregate_model_stats(
            [item.metrics.predictive_coding for item in scored]
        ),
        circadian_predictive_coding=base._aggregate_circadian_stats(
            [item.metrics.circadian_predictive_coding for item in scored]
        ),
    )


def _hash_json(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
