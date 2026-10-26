"""Measure fixed fresh-task learning on a disposable, owned learner fork.

Feature widths retain the toy 1..16 contract and admit exactly148 public MiniGrid
coordinates or296 public initial-plus-current coordinates within the original
aggregate storage bound.
Probe wakes use only probe_train labels; probe_test labels stay in application
metrics. Opt-in fixed rehearsal uses separately admitted TRAIN episodes; an
explicit optional return uses separate old-TRAIN update templates.
The caller chooses checkpoints in advance and supplies a separate fresh
budget. Results describe complete or partial work, not benchmark updates or a
scientific advantage. Source isolation relies on the owned-fork learner contract.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields, replace
from math import isfinite
from time import monotonic
from typing import Any, Callable, TypedDict, cast

import numpy as np

from src.app.coupled_attribution import (
    AttributionBatch,
    AttributionStream,
    _copy_batch,
    _probability_scores,
    _same_value,
    _validate_batch,
)
from src.app.stream_rehearsal import StreamRehearsal, _continuation_budget
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.app.probe_training_readouts import (
    TrainingReadoutProgress,
    score_training_readouts,
    validate_training_readout_envelope,
    validate_training_readouts,
)
from src.app.probe_return_phase import (
    budget_binding as _return_budget_binding,
    paid_report_guard,
    validate_return_envelope,
    validate_return_values,
    validate_return_zero,
)
from src.app.probe_endpoint_observer import EndpointObserver, observe_endpoint
from src.core.actor_ports import ForkableLearner
from src.core.categorical_probability_scores import probability_scores as _categorical_scores
from src.core.categorical_training_diagnostic import (
    CategoricalPCTrainingDiagnostic,
    ScheduledCategoricalUpdateFailed,
)
from src.core.coupled_energy import Array
from src.core.learner_ports import NativeLearner, TrainingDiagnostic, TrainingUpdateFailed
from src.core.probe_decision_diagnostics import decision_class_diagnostics


def _validate_batches(
    training: tuple[AttributionBatch, ...],
    evaluation: tuple[AttributionBatch, ...],
    training_readouts: tuple[AttributionBatch, ...] = (),
    return_training: tuple[AttributionBatch, ...] = (),
) -> None:
    if type(training) is not tuple or not 1 <= len(training) <= 16:
        raise ValueError("probe training must be a tuple containing 1..16 batches")
    if type(evaluation) is not tuple or not 1 <= len(evaluation) <= 4:
        raise ValueError("probe evaluation must be a tuple containing 1..4 batches")
    seen: set[str] = set()
    arrays: list[Array] = []
    entries = 0
    for batches, role in ((training, "probe_train"), (evaluation, "probe_test")):
        for batch in batches:
            if type(batch) is not AttributionBatch:
                raise ValueError("probe batches must contain exact AttributionBatch records")
            if type(batch.role) is not str or batch.role != role:
                raise ValueError(f"batch role must be {role}")
            if (
                type(batch.features) is not np.ndarray
                or batch.features.dtype != np.float64
                or batch.features.ndim != 2
                or not 1 <= batch.features.shape[0] <= 128
                or not (1 <= batch.features.shape[1] <= 16 or batch.features.shape[1] in (148, 296))
                or type(batch.targets) is not np.ndarray
                or batch.targets.dtype != np.float64
                or batch.targets.shape != (batch.features.shape[0], 1)
            ):
                raise ValueError(
                    "probe data must be float64 matrices with 1..128 rows and "
                    "1..16, exactly148 or296 features, with targets (B, 1)"
                )
            entries += batch.features.size + batch.targets.size
            if entries > 65536:
                raise ValueError("probe data exceed 65536 aggregate entries")
            arrays.extend((batch.features, batch.targets))
    # Admit the whole input envelope before delegates allocate value-check arrays.
    if type(training_readouts) is not tuple or training_readouts:
        validate_training_readout_envelope(training, evaluation, training_readouts)
    if type(return_training) is not tuple or return_training:
        validate_return_envelope(training, evaluation, training_readouts, return_training)
    for batches, role in ((training, "probe_train"), (evaluation, "probe_test")):
        for batch in batches:
            _validate_batch(batch, role, seen)
    contexts = tuple(batch.context for batch in evaluation)
    if len(set(contexts)) != len(contexts):
        raise ValueError("probe evaluation contexts must be unique")
    if any(batch.context not in contexts for batch in training):
        raise ValueError("every probe training context requires an evaluation context")
    width = training[0].features.shape[1]
    if any(batch.features.shape[1] != width for batch in (*training, *evaluation)):
        raise ValueError("all probe feature widths must match")
    if any(batch.num_classes != training[0].num_classes for batch in (*training, *evaluation)):
        raise ValueError("all probe training and evaluation class counts must match")
    episodes = [batch.episode for batch in (*training, *evaluation)]
    if any(episode is not None for episode in episodes) and any(
        episode is None for episode in episodes
    ):
        raise ValueError("episode probes require complete metadata for every batch")
    if any(
        np.shares_memory(left, right)
        for index, left in enumerate(arrays)
        for right in arrays[index + 1 :]
    ):
        raise ValueError("all probe feature and target arrays must have disjoint storage")


def _validate_controls(
    budget: ToyBudgetSession,
    checkpoints: tuple[int, ...],
    updates: int,
    timer: Callable[[], float],
) -> None:
    if type(budget) is not ToyBudgetSession or type(budget.budget) is not ToyExecutionBudget:
        raise ValueError("probe requires an exact ToyBudgetSession and budget")
    replace(budget.budget)  # Revalidate mutable contents of the frozen limits record.
    if type(budget.updates_completed) is not int or budget.updates_completed != 0:
        raise ValueError("probe budget must be fresh with zero completed updates")
    if budget.budget.max_training_updates != updates:
        raise ValueError("probe budget max_training_updates must equal the training batch count")
    if (
        type(checkpoints) is not tuple
        or not 2 <= len(checkpoints) <= 17
        or any(type(point) is not int for point in checkpoints)
        or checkpoints[0] != 0
        or checkpoints[-1] != updates
        or any(left >= right for left, right in zip(checkpoints, checkpoints[1:]))
    ):
        raise ValueError("probe checkpoints must be strictly increasing native counts from 0 to n")
    if not callable(timer):
        raise ValueError("probe timer must be callable")


class _Elapsed:
    def __init__(self, timer: Callable[[], float]) -> None:
        self.timer = timer
        self.started: float | None = None
        self.last: float | None = None

    def read(self) -> float:
        value = self.timer()
        if type(value) not in {int, float}:
            raise ValueError("probe timer must return finite native numbers")
        try:
            current = float(value)
        except OverflowError as error:
            raise ValueError("probe timer must return finite native numbers") from error
        if not isfinite(current) or (self.last is not None and current < self.last):
            raise ValueError("probe timer must be finite and nondecreasing")
        if self.started is not None and not isfinite(current - self.started):
            raise ValueError("probe elapsed seconds must remain finite")
        if self.started is None:
            self.started = current
        self.last = current
        return current

    def elapsed(self) -> float | None:
        if self.started is None or self.last is None:
            return None
        return self.last - self.started


@dataclass
class _Progress:
    attempted_updates: int = 0
    completed_updates: int = 0
    attempted_training_examples: int = 0
    completed_training_examples: int = 0
    score_forward_calls_attempted: int = 0
    score_forward_calls_completed: int = 0
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    _failed_native_work: dict[str, Any] | None = None


class _ProbeOptions(TypedDict, total=False):
    check_boundary: Callable[[], None]
    training_readouts: tuple[AttributionBatch, ...]
    readout_progress: TrainingReadoutProgress | None


def _probe_options(check_boundary, training_readouts, readout_progress) -> _ProbeOptions:
    # Preserve omitted default keywords while making the optional port types explicit.
    options: _ProbeOptions = {}
    if check_boundary is not None:
        options["check_boundary"] = check_boundary
    if training_readouts:
        options["training_readouts"] = training_readouts
        options["readout_progress"] = readout_progress
    return options


def _score(
    learner: NativeLearner[Array, Array, Array, Any],
    evaluation: tuple[AttributionBatch, ...],
    updates: int,
    budget: ToyBudgetSession,
    elapsed: _Elapsed,
    progress: _Progress,
    *,
    decision_diagnostics: bool = False,
    check_boundary: Callable[[], None] | None = None,
    training_readouts: tuple[AttributionBatch, ...] = (),
    readout_progress: TrainingReadoutProgress | None = None,
) -> None:
    scores = {}
    details = {}
    for batch in evaluation:
        budget.before_final()
        elapsed.read()
        if check_boundary is not None:
            check_boundary()
        progress.score_forward_calls_attempted += 1
        prediction = learner.predict(batch.features.copy())
        progress.score_forward_calls_completed += 1
        if check_boundary is not None:
            check_boundary()
        scores[batch.context] = (
            _probability_scores(prediction, batch.targets)
            if batch.num_classes == 2
            else _categorical_scores(prediction, batch.targets, batch.num_classes)
        )
        if decision_diagnostics:
            diagnostic = decision_class_diagnostics(prediction, batch.targets, batch.num_classes)
            diagnostic.update(
                row_ids=list(batch.row_ids),
                episode_id=None if batch.episode is None else batch.episode.episode_id,
                sequence_kind="batch_row_order" if batch.episode is None else "complete_episode",
            )
            details[batch.context] = diagnostic
        budget.before_final()
        elapsed.read()
    if check_boundary is not None:
        check_boundary()
    # A failed context or boundary must not publish a partial checkpoint.
    point: dict[str, Any] = {"updates": updates, "scores": scores}
    if decision_diagnostics:
        point["decision_diagnostics"] = details
    progress.checkpoints.append(point)
    if training_readouts:
        if readout_progress is None:
            raise ValueError("training readouts require separate work accounting")
        score_training_readouts(
            learner,
            training_readouts,
            updates,
            budget,
            elapsed,
            readout_progress,
            decision_diagnostics=decision_diagnostics,
            **({} if check_boundary is None else {"check_boundary": check_boundary}),
        )


def _train(
    learner: NativeLearner[Array, Array, Array, Any],
    training: tuple[AttributionBatch, ...],
    evaluation: tuple[AttributionBatch, ...],
    checkpoints: tuple[int, ...],
    budget: ToyBudgetSession,
    elapsed: _Elapsed,
    progress: _Progress,
    *,
    decision_diagnostics: bool = False,
    check_boundary: Callable[[], None] | None = None,
    training_readouts: tuple[AttributionBatch, ...] = (),
    readout_progress: TrainingReadoutProgress | None = None,
    stream: AttributionStream | None = None,
) -> None:
    for batch in training:
        budget.before_update()
        elapsed.read()
        if check_boundary is not None:
            check_boundary()
        progress.attempted_updates += 1
        progress.attempted_training_examples += len(batch.features)
        try:
            if stream is None:
                diagnostic = learner.train_batch(batch.features.copy(), batch.targets.copy())
            else:
                diagnostic = _stream_update(stream, batch, progress)
        except TrainingUpdateFailed as error:
            # Only this call can report native update work; external callbacks
            # raising the same type must not fabricate an optimizer charge.
            if type(error) is ScheduledCategoricalUpdateFailed:
                # Why: the scheduled receipt adds publication costs independently
                # of native work. Revalidate both owned scalar namespaces here.
                validated_failure = ScheduledCategoricalUpdateFailed(
                    str(error), error.definition, error.work, error.scheduling_work
                )
                progress._failed_native_work = dict(
                    definition=validated_failure.definition,
                    work=validated_failure.work,
                    scheduling_work=validated_failure.scheduling_work,
                )
            else:
                progress._failed_native_work = dict(definition=error.definition, work=error.work)
            raise
        # Count the actual native commit before any subsequent callback can fail.
        if stream is None:
            progress.completed_updates += 1
            progress.completed_training_examples += len(batch.features)
            budget.record_update()
        if type(diagnostic) not in (TrainingDiagnostic, CategoricalPCTrainingDiagnostic):
            raise ValueError("probe training must return an exact supported TrainingDiagnostic")
        if type(diagnostic) is CategoricalPCTrainingDiagnostic and set(vars(diagnostic)) != {
            item.name for item in fields(CategoricalPCTrainingDiagnostic)
        }:
            raise ValueError("probe training requires a complete categorical diagnostic")
        validated = replace(diagnostic)
        if stream is None:
            progress.diagnostics.append(asdict(validated))
        if check_boundary is not None:
            check_boundary()
        budget.before_final()
        elapsed.read()
        if progress.completed_updates in checkpoints:
            _score(
                learner,
                evaluation,
                progress.completed_updates,
                budget,
                elapsed,
                progress,
                decision_diagnostics=decision_diagnostics,
                **_probe_options(check_boundary, training_readouts, readout_progress),
            )


def _stream_update(
    stream: AttributionStream, batch: AttributionBatch, progress: _Progress
) -> TrainingDiagnostic:
    """Keep a paid wake receipt even when its subsequent rehearsal fails."""
    completed = progress.completed_updates
    try:
        stream.run_until(completed + 1)
    finally:
        receipts = stream._runtime.applied_updates
        if len(receipts) == completed + 1:
            progress.completed_updates += 1
            progress.completed_training_examples += len(batch.features)
            progress.diagnostics.append(asdict(receipts[-1].diagnostic))
    return receipts[-1].diagnostic


def run_fresh_task_probe(
    source: ForkableLearner[Array, Array, Array, Any],
    state: Any,
    training: tuple[AttributionBatch, ...],
    evaluation: tuple[AttributionBatch, ...],
    *,
    budget: ToyBudgetSession,
    checkpoint_updates: tuple[int, ...],
    timer: Callable[[], float] = monotonic,
    decision_diagnostics: bool = False,
    check_boundary: Callable[[], None] | None = None,
    training_readouts: tuple[AttributionBatch, ...] = (),
    return_training: tuple[AttributionBatch, ...] = (),
    return_budget: ToyBudgetSession | None = None,
    endpoint_observer: EndpointObserver | None = None,
    rehearsal_factory: Callable[[Any, ToyBudgetSession, int], StreamRehearsal] | None = None,
    extra_f_exposure: bool = False,
) -> dict[str, Any]:
    """Train fixed probe batches on an owned fork and retain measured partial work.

    Invalid public inputs raise before callbacks. Operational exceptions produce
    failed rows with completed checkpoints retained and no later scoring/retry.
    A failed row's elapsed time covers the last successful timer reading only
    (timing_status='partial'); unavailable initial readings produce None.
    Inner elapsed time starts after data/state detachment; a caller measuring
    total probe cost must include preflight and detachment around this call.
    reserved_data_bytes is unavailable (None) if data detachment is incomplete.
    Arrays and opaque state are detached before external fork/restore/budget or
    timer calls. The source's fork must own its native state and training policy.
    Optional decision/class reports use the same returned predictions, preserving
    aggregate scores and native calls. CPU/report allocations add measured cost;
    reserved_data_bytes continues to count only the original detached input arrays.
    An optional boundary guard observes external native calls without renewing
    budgets. Successful native work is charged before its post-call guard runs.
    Optional training_readouts measure old TRAIN fit on the same disposable fork
    after each original checkpoint. Their predictions, arrays, timing and partial
    work are separate; those labels never reach train_batch. The original entry
    limit includes these arrays, and empty readouts preserve the original report.
    Optional return_training supplies two TRAIN templates equal to the monitors
    with fresh IDs/storage. Their fixed sixteen updates reuse this same learner
    with a separate fresh return_budget after F completion; original F accounting
    and timing remain separate from the nested return_phase report.
    Optional endpoint_observer reads this same disposable learner after complete
    probe/return phases. It requires the full return recipe, owns its native-state
    checks and readout accounting, and must use the supplied protected boundary.
    Its failures retain paid native work and prevent subsequent phase admission.
    Optional rehearsal_factory constructs a fixed TRAIN bank for an explicitly
    probe_train-scoped existing stream after this owner restores its disposable
    fork. F roles remain unchanged; final data never enter the stream. Rehearsal
    work/storage and stream receipts are reported separately; A return reuses the
    F-adapted candidate with no further replay. Default direct updates are unchanged.
    Explicit extra_f_exposure repeats the original owned sixteen F batches three
    times by reference, using a fresh48-update budget and fixed0/12/24/48 readouts.
    It requires the full separate A16 return recipe and admits no rehearsal; the
    original input envelope, array storage bound and role assignments remain fixed.
    """
    if type(extra_f_exposure) is not bool:
        raise ValueError("extra_f_exposure must be an exact bool")
    if extra_f_exposure and (
        rehearsal_factory is not None
        or any(
            type(batches) is not tuple or len(batches) != count
            for batches, count in (
                (training, 16),
                (evaluation, 2),
                (training_readouts, 2),
                (return_training, 2),
            )
        )
    ):
        raise ValueError(
            "extra F exposure requires the full original16 F/A16 recipe without rehearsal"
        )
    original_probe_updates = 48 if extra_f_exposure else 16
    original_checkpoints = (0, 12, 24, 48) if extra_f_exposure else (0, 4, 8, 16)
    if type(decision_diagnostics) is not bool:
        raise ValueError("decision_diagnostics must be an exact bool")
    if check_boundary is not None and not callable(check_boundary):
        raise ValueError("check_boundary must be callable or None")
    if rehearsal_factory is not None and not callable(rehearsal_factory):
        raise ValueError("rehearsal_factory must be callable or None")
    if endpoint_observer is not None and (
        not callable(endpoint_observer) or type(return_training) is not tuple or not return_training
    ):
        raise ValueError("endpoint_observer requires a callable and the full return recipe")
    return_enabled = type(return_training) is not tuple or bool(return_training)
    return_limits = None
    expected_return_limits = None
    if return_enabled:
        _validate_batches(training, evaluation, training_readouts, return_training)
        validate_training_readouts(training, evaluation, training_readouts)
        validate_return_values(training, evaluation, training_readouts, return_training)
        if return_budget is budget:
            raise ValueError("A return requires a distinct fresh budget")
        if type(return_budget) is not ToyBudgetSession:
            raise ValueError("A return requires an exact fresh ToyBudgetSession")
        if (
            type(return_budget.replay_examples_completed) is not int
            or return_budget.replay_examples_completed != 0
        ):
            raise ValueError("A return requires a fresh budget with zero replay work")
        if checkpoint_updates != original_checkpoints or any(
            type(point) is not int for point in checkpoint_updates
        ):
            raise ValueError(
                "extra F exposure requires fixed original checkpoints0/12/24/48"
                if extra_f_exposure
                else "A return requires fixed original checkpoints0/4/8/16"
            )
        _validate_controls(return_budget, (0, 4, 8, 16), 16, timer)
        return_limits = return_budget.budget
        expected_return_limits = deepcopy(return_limits)
    elif return_budget is not None:
        raise ValueError("return_budget requires two return_training templates")
    elif type(training_readouts) is not tuple or training_readouts:
        _validate_batches(training, evaluation, training_readouts)
        validate_training_readouts(training, evaluation, training_readouts)
    else:
        _validate_batches(training, evaluation)
    _validate_controls(
        budget,
        checkpoint_updates,
        original_probe_updates if extra_f_exposure else len(training),
        timer,
    )
    if extra_f_exposure and (
        type(budget.replay_examples_completed) is not int or budget.replay_examples_completed != 0
    ):
        raise ValueError("extra F exposure requires a fresh budget with zero replay work")
    planned_f_training_rows = (
        sum(len(batch.features) for batch in training) * 3 if extra_f_exposure else None
    )
    original_probe_authority = None if rehearsal_factory is None else _continuation_budget(budget)
    original_probe_limits = None if rehearsal_factory is None else deepcopy(budget.budget)
    original_probe_binding = None if rehearsal_factory is None else _return_budget_binding(budget)
    original_probe_last_clock = budget.last_clock
    if rehearsal_factory is not None and any(batch.episode is None for batch in training):
        raise ValueError("rehearsed probes require complete probe_train episode metadata")
    fixed_budgets = None
    if return_enabled:
        fixed_budgets = (
            budget,
            _return_budget_binding(budget),
            deepcopy(budget.budget),
            _return_budget_binding(return_budget),
        )
    num_classes = training[0].num_classes
    episode_mode = training[0].episode is not None
    progress = _Progress()
    readout_progress = TrainingReadoutProgress() if training_readouts else None
    elapsed = _Elapsed(timer)
    reserved_bytes = None
    readout_bytes = None
    return_bytes = None
    owned_return: tuple[AttributionBatch, ...] = ()
    episode_scope = None
    reason = None
    stream: AttributionStream | None = None
    helper: StreamRehearsal | None = None
    replay_owner: ToyBudgetSession | None = None
    replay_limits: ToyExecutionBudget | None = None
    replay_binding: tuple[Any, ...] | None = None
    stream_report: dict[str, Any] | None = None

    def rehearsal_boundary() -> None:
        nonlocal original_probe_last_clock
        if check_boundary is not None:
            check_boundary()
        if (
            not _same_value(budget.budget, original_probe_limits)
            or _return_budget_binding(budget) != original_probe_binding
            or type(budget.updates_completed) is not int
            or budget.updates_completed != progress.completed_updates
            or type(budget.replay_examples_completed) is not int
            or budget.replay_examples_completed != 0
            or type(budget.last_clock) not in (int, float)
            or not isfinite(budget.last_clock)
            or budget.last_clock < original_probe_last_clock
        ):
            raise ValueError("original probe budget or observed counters changed during rehearsal")
        original_probe_last_clock = budget.last_clock
        if helper is not None:
            if (
                helper._budget is not budget
                or helper._replay_budget is not replay_owner
                or replay_owner is None
                or not _same_value(replay_owner.budget, replay_limits)
                or _return_budget_binding(replay_owner) != replay_binding
                or type(replay_owner.updates_completed) is not int
                or replay_owner.updates_completed != 0
                or type(replay_owner.replay_examples_completed) is not int
                or replay_owner.replay_examples_completed
                != sum(event["charged_exposures"] for event in helper._events)
            ):
                raise ValueError("original independent probe replay budget or charges changed")
            helper._check_schema()

    boundary = check_boundary if rehearsal_factory is None else rehearsal_boundary
    try:
        owned_training = tuple(map(_copy_batch, training))
        owned_evaluation = tuple(map(_copy_batch, evaluation))
        reserved_bytes = sum(
            batch.features.nbytes + batch.targets.nbytes
            for batch in (*owned_training, *owned_evaluation)
        )
        owned_readouts = tuple(map(_copy_batch, training_readouts)) if training_readouts else ()
        if training_readouts:
            readout_bytes = sum(
                batch.features.nbytes + batch.targets.nbytes for batch in owned_readouts
            )
        if return_enabled:
            owned_return = tuple(map(_copy_batch, return_training))
            return_bytes = sum(
                batch.features.nbytes + batch.targets.nbytes for batch in owned_return
            )
        if owned_training[0].episode is not None:
            episode_scope = _episode_scope(owned_training, owned_evaluation)
        owned_state = deepcopy(state)
        elapsed.read()
        budget.before_final()
        if check_boundary is not None:
            check_boundary()
        learner = source.fork()
        if check_boundary is not None:
            check_boundary()
        if learner is source:
            raise ValueError("probe fork must be distinct from its source learner")
        budget.before_final()
        elapsed.read()
        if check_boundary is not None:
            check_boundary()
        learner.restore_state(owned_state)
        if check_boundary is not None:
            check_boundary()
        budget.before_final()
        elapsed.read()
        if rehearsal_factory is not None:
            helper = rehearsal_factory(learner, budget, len(owned_training))
            if _continuation_budget(budget) != original_probe_authority:
                raise ValueError("rehearsal construction changed original probe budget authority")
            if type(helper) is not StreamRehearsal or helper._wake_role != "probe_train":
                raise ValueError("probe rehearsal requires its explicit probe_train helper")
            if not _same_value(learner.snapshot_state(), owned_state):
                raise ValueError("rehearsal construction changed the restored disposable state")
            replay_owner = helper._replay_budget
            replay_limits = deepcopy(replay_owner.budget)
            replay_binding = _return_budget_binding(replay_owner)
            # Establish native clocks/counters using the existing original session;
            # this observation neither restores nor renews its limits.
            _continuation_budget(replay_owner)
            rehearsal_boundary()
            if not callable(getattr(learner, "fork", None)):
                raise ValueError(
                    "rehearsed disposable learner requires owned actor/candidate forks"
                )
            stream = AttributionStream(
                cast(ForkableLearner[Array, Array, Array, Any], learner),
                owned_training,
                (),
                checkpoint_updates=(0, len(owned_training)),
                label_delay=0,
                budget=budget,
                rehearsal=helper,
                training_only=True,
                training_role="probe_train",
            )
            learner = stream._runtime._candidate
            rehearsal_boundary()
        _score(
            learner,
            owned_evaluation,
            0,
            budget,
            elapsed,
            progress,
            decision_diagnostics=decision_diagnostics,
            **_probe_options(boundary, owned_readouts, readout_progress),
        )
        _train(
            learner,
            owned_training * 3 if extra_f_exposure else owned_training,
            owned_evaluation,
            checkpoint_updates,
            budget,
            elapsed,
            progress,
            decision_diagnostics=decision_diagnostics,
            **_probe_options(boundary, owned_readouts, readout_progress),
            **({} if stream is None else {"stream": stream}),
        )
        if stream is not None:
            stream_report = stream.finish()
            # Keep the one detached rehearsal report below, without duplicating
            # complete selected-episode records in both output branches.
            stream_report.pop("rehearsal", None)
        budget.before_final()
        elapsed.read()
        if boundary is not None:
            boundary()
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
    measured = elapsed.elapsed()
    report = {
        "status": "complete" if reason is None else "failed",
        "reason": reason,
        **asdict(progress),
        "elapsed_seconds": measured,
        "timing_status": "unavailable"
        if measured is None
        else "complete"
        if reason is None
        else "partial",
        "reserved_data_bytes": reserved_bytes,
    }
    failed_native_work = report.pop("_failed_native_work")
    if extra_f_exposure:
        report["extra_f_exposure"] = dict(
            input_training_batches=16,
            schedule_repetitions=3,
            native_training_updates=48,
            planned_training_rows=planned_f_training_rows,
            checkpoint_mapping={
                "original_f16_updates": [0, 4, 8, 16],
                "native_updates": [0, 12, 24, 48],
            },
            replay_native_calls=0,
            replay_training_rows=0,
            scope="three ordered passes over the same owned probe_train batches; no additional data or old-A replay; original input arrays detached once; repeated tuple references add no feature/target arrays; native calls, diagnostics and temporary copies remain charged separately",
        )
    if num_classes > 2:
        report["num_classes"] = num_classes
    if rehearsal_factory is not None:
        report["probe_stream"] = stream_report
        report["rehearsal"] = None
        report["rehearsal_scope"] = (
            "explicit probe_train stream on a disposable candidate; complete old TRAIN replay "
            "with a separate original budget; F counters exclude replay work; no replay during "
            "A return; stream/native/copy/event storage additional to reserved_data_bytes"
        )
        if stream is not None and stream_report is None:
            report["probe_stream"] = dict(
                training_role="probe_train",
                completed_updates=stream.completed_updates,
                wake_budget_updates_charged=budget.updates_completed,
                applied_updates=[asdict(row) for row in stream._runtime.applied_updates],
                applied_consolidations=[
                    asdict(row) for row in stream._runtime.applied_consolidations
                ],
            )
        if helper is not None:
            try:
                rehearsal_boundary()
                report["rehearsal"] = helper.report
            except Exception as error:
                report.update(
                    status="failed", reason=f"rehearsal readback: {type(error).__name__}: {error}"
                )
                reason = report["reason"]
                report["rehearsal_readback_error"] = report["reason"]
                report["replay_budget_charged_exposures"] = (
                    None if replay_owner is None else replay_owner.replay_examples_completed
                )
    if failed_native_work is not None:
        report["failed_native_work"] = failed_native_work
    if episode_mode:
        report["episode_scope"] = episode_scope
    if decision_diagnostics:
        report.update(
            decision_diagnostics_enabled=True,
            decision_diagnostics_scope="reserved probe_test predictions already returned; no extra native calls. At most17checkpoints/4contexts/128rows/16classes per context; confusion rows=true, columns=predicted. Metadata-free indices are batch rows, not temporal steps. Additional Python report/temp NumPy allocations excluded from reserved_data_bytes; included in invocation time/process RSS.",
        )
    if readout_progress is not None:
        report["training_readouts"] = _training_readout_report(
            readout_progress, reason, readout_bytes
        )
        if rehearsal_factory is not None:
            report["training_readouts"]["scope"] = (
                "readonly old TRAIN fit on the disposable candidate after fixed F checkpoints; "
                "probe_train wakes and explicitly admitted TRAIN bank replay update separately; "
                "readout operations perform no updates. Native/copy/event/report/RSS costs are "
                "additional to reserved_data_bytes; no acquired-skill retention claim"
            )
    if endpoint_observer is not None and reason is None:
        try:
            observe_endpoint(
                endpoint_observer,
                learner,
                "after_probe",
                report,
                budget=budget,
                return_budget=return_budget,
                fixed_budgets=fixed_budgets,
                return_limits=expected_return_limits,
                check_boundary=boundary,
                **({"original_probe_updates": 48} if extra_f_exposure else {}),
            )
        except Exception as error:
            reason = f"endpoint observer after_probe: {type(error).__name__}: {error}"
            report.update(
                status="failed",
                reason=reason,
                timing_status="unavailable" if measured is None else "partial",
            )
    if return_enabled:
        report["return_phase"] = _run_return_phase(
            learner if reason is None else None,
            owned_return,
            owned_evaluation if reason is None else (),
            owned_readouts if reason is None else (),
            report,
            return_budget,
            timer,
            return_bytes,
            decision_diagnostics,
            boundary,
            return_limits,
            expected_return_limits,
            fixed_budgets,
            **({"original_probe_updates": 48} if extra_f_exposure else {}),
        )
        if (
            endpoint_observer is not None
            and report["status"] == "complete"
            and report["return_phase"]["status"] == "complete"
        ):
            try:
                observe_endpoint(
                    endpoint_observer,
                    learner,
                    "after_return",
                    report,
                    budget=budget,
                    return_budget=return_budget,
                    fixed_budgets=fixed_budgets,
                    return_limits=expected_return_limits,
                    check_boundary=boundary,
                    **({"original_probe_updates": 48} if extra_f_exposure else {}),
                )
            except Exception as error:
                report.update(
                    status="failed",
                    reason=f"endpoint observer after_return: {type(error).__name__}: {error}",
                    timing_status="unavailable" if measured is None else "partial",
                )
    return report


def _training_readout_report(readout_progress, reason, readout_bytes) -> dict[str, Any]:
    monitor_elapsed = readout_progress.elapsed_seconds
    return {
        **asdict(readout_progress),
        "status": "complete"
        if reason is None
        else "failed"
        if readout_progress.checkpoints
        else "not_run",
        "reason": reason,
        "reserved_data_bytes": readout_bytes,
        "timing_status": "unavailable"
        if monitor_elapsed is None
        else "complete"
        if reason is None
        else "partial",
        "scope": "readonly TRAIN fit on the same disposable probe learner after original probe_test checkpoints; only probe_train batches update; no acquired-skill retention/generalization claim. Detached input arrays counted separately; Python/episode/native-state/report/temp copies and failed work are additional or unknown; elapsed covers returned timing intervals only.",
    }


def _run_return_phase(
    learner,
    templates,
    evaluation,
    monitors,
    original,
    budget,
    timer,
    reserved_bytes,
    decision_diagnostics,
    check_boundary,
    limits,
    expected_limits,
    fixed_budgets,
    *,
    original_probe_updates: int = 16,
) -> dict[str, Any]:
    """Reuse the existing fixed scoring/training loop on the same native learner."""
    if type(original_probe_updates) is not int or original_probe_updates not in (16, 48):
        raise ValueError("A return supports only explicit original F16 or F48 authority")
    progress = _Progress()
    monitor_progress = TrainingReadoutProgress()
    elapsed = _Elapsed(timer)
    reason = None
    started = False
    paid_guard, expected = paid_report_guard(original)
    last_return_clock = None

    def budget_guard() -> None:
        nonlocal last_return_clock
        original_budget, original_binding, original_limits, return_binding = fixed_budgets
        if (
            budget.budget is not limits
            or not _same_value(budget.budget, expected_limits)
            or type(budget.updates_completed) is not int
            or budget.updates_completed != progress.completed_updates
            or type(budget.replay_examples_completed) is not int
            or budget.replay_examples_completed != 0
            or _return_budget_binding(budget) != return_binding
            or type(budget.last_clock) is not float
            or not isfinite(budget.last_clock)
            or (last_return_clock is not None and budget.last_clock < last_return_clock)
            or _return_budget_binding(original_budget) != original_binding
            or not _same_value(original_budget.budget, original_limits)
            or type(original_budget.updates_completed) is not int
            or original_budget.updates_completed != original_probe_updates
            or type(original_budget.replay_examples_completed) is not int
            or original_budget.replay_examples_completed != 0
        ):
            raise ValueError("A return budget configuration or observed counters changed")
        last_return_clock = budget.last_clock

    def guard() -> None:
        paid_guard()
        budget_guard()
        if check_boundary is not None:
            check_boundary()
        paid_guard()
        budget_guard()

    if original["status"] == "complete":
        started = True
        try:
            _validate_controls(budget, (0, 4, 8, 16), 16, timer)
            guard()
            _score(
                learner,
                evaluation,
                0,
                budget,
                elapsed,
                progress,
                decision_diagnostics=decision_diagnostics,
                **_probe_options(guard, monitors, monitor_progress),
            )
            validate_return_zero(expected, progress, monitor_progress)
            guard()
            schedule = tuple(templates[index] for index in (0, 1) * 8)
            _train(
                learner,
                schedule,
                evaluation,
                (0, 4, 8, 16),
                budget,
                elapsed,
                progress,
                decision_diagnostics=decision_diagnostics,
                **_probe_options(guard, monitors, monitor_progress),
            )
            budget.before_final()
            elapsed.read()
            guard()
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
    else:
        reason = "original F probe or TRAIN readouts did not complete; no return work admitted"
    # Rejected edits cannot rewrite the already paid original result.
    try:
        paid_guard()
    except ValueError as error:
        reason = f"{type(error).__name__}: {error}"
        original.clear()
        original.update(expected)
    measured = elapsed.elapsed()
    report = dict(
        status="complete" if started and reason is None else "failed" if started else "not_run",
        reason=reason,
        **asdict(progress),
        elapsed_seconds=measured,
        timing_status="unavailable"
        if measured is None
        else "complete"
        if reason is None
        else "partial",
        reserved_data_bytes=reserved_bytes,
        checkpoint_updates=[0, 4, 8, 16],
        schedule=[0, 1] * 8,
        scope="same disposable F16 learner and optimizer; explicit old TRAIN return templates only; relative return counts; no acquired-skill recovery/generalization claim",
    )
    if original_probe_updates == 48:
        report["scope"] = (
            "same disposable F48 learner and optimizer; explicit old TRAIN return templates only; relative return counts; no acquired-skill recovery/generalization claim"
        )
    failed = report.pop("_failed_native_work")
    if failed is not None:
        report["failed_native_work"] = failed
    if original.get("num_classes") is not None:
        report["num_classes"] = original["num_classes"]
    report["training_readouts"] = _training_readout_report(monitor_progress, reason, 0)
    report["training_readouts"]["scope"] = (
        "readonly old TRAIN fit on the same F-adapted disposable learner during explicit A return; "
        "only the separate return TRAIN templates update. Original detached monitor arrays reused "
        "without recopy; Python/episode/native-state/report/temp copies and failed work additional "
        "or unknown; elapsed covers returned timing intervals only; no acquired-skill recovery claim"
    )
    return report


def _episode_scope(
    training: tuple[AttributionBatch, ...], evaluation: tuple[AttributionBatch, ...]
) -> dict[str, Any]:
    """Record detached complete packet identities without inventing record byte sizes."""
    return dict(
        training_episode_ids=[
            batch.episode.episode_id for batch in training if batch.episode is not None
        ],
        evaluation_episode_ids=[
            batch.episode.episode_id for batch in evaluation if batch.episode is not None
        ],
        training_transitions=sum(len(batch.features) for batch in training),
        evaluation_transitions=sum(len(batch.features) for batch in evaluation),
        memory_scope="reserved_data_bytes counts detached feature/target arrays; excludes episode record/Python/native-state/copy payloads and process RSS",
    )
