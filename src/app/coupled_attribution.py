"""Train a bounded delayed stream, then score its detached model checkpoints.

Inputs are declared train/final-test batches, an owned-fork learner, fixed
checkpoint counts, and an execution budget. Labels from heldout batches are used
only by application metrics after all training completes. Explicit training-only
streams admit no heldout data and finish without evaluation. This module does not
promote candidates, tune policies, establish provenance, or release scientific
claims. Exceptions leave the invocation incomplete and propagate to the caller.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, is_dataclass
from copy import deepcopy
from math import isfinite
import pickle
from threading import Lock
from typing import Any, Callable, Protocol, Iterator

import numpy as np

from src.app.actor_shadow import ActorShadowRuntime, StableActor
from src.app.counterfactual_controller import CounterfactualController
from src.app.stream_capacity import StreamCapacity
from src.app.stream_rehearsal import StreamRehearsal
from src.app.stream_unit_renewal import StreamUnitRenewal
from src.app.reference_prequential import ReferencePrequential
from src.app.toy_execution_budget import ToyBudgetSession
from src.core.actor_ports import AppliedConsolidation, CandidateState, ForkableLearner
from src.core.binary_probability_scores import probability_scores as _probability_scores
from src.core.categorical_probability_scores import (
    probability_scores as _categorical_scores,
    validate_class_count,
    validate_class_targets,
)
from src.core.coupled_energy import Array
from src.core.contextual_controller import FittedController, choose_contextual
from src.core.demonstration_episode import DemonstrationEpisode, copy_episode, episode_arrays
from src.core.reference_controller import ReferencePolicy, ReferenceObservation, choose_reference
from src.core.experience import (
    AppliedExperience,
    Experience,
    ExperiencePermissions,
    LabelArrival,
    LogicalClock,
    require_identifier,
    require_training_role,
)


@dataclass(frozen=True)
class AttributionBatch:
    role: str
    context: str
    row_ids: tuple[str, ...]
    features: Array
    targets: Array
    episode: DemonstrationEpisode | None = None
    num_classes: int = 2


@dataclass(frozen=True)
class PrearrivalObservation:
    """Owned features available before this tick's labels, without task metadata."""

    tick: int
    row_ids: tuple[str, ...]
    features: Array


def _identifier(value: str, name: str) -> None:
    require_identifier(value, name)


def _matrix(value: Array, name: str) -> None:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.float64
        or value.ndim != 2
        or min(value.shape) < 1
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be a nonempty finite float64 matrix")


def _validate_batch(batch: AttributionBatch, role: str, seen_rows: set[str]) -> None:
    if type(batch) is not AttributionBatch:
        raise ValueError("batches must contain exact AttributionBatch records")
    if type(batch.role) is not str or batch.role != role:
        raise ValueError(f"batch role must be {role}")
    _identifier(batch.context, "batch context")
    _matrix(batch.features, "batch features")
    _matrix(batch.targets, "batch targets")
    classes = validate_class_count(batch.num_classes)
    rows = len(batch.features)
    if role in ("train", "probe_train") and rows > 1024:
        raise ValueError("training batches must contain at most 1024 rows")
    if classes == 2:
        if batch.targets.shape != (rows, 1) or np.any((batch.targets < 0) | (batch.targets > 1)):
            raise ValueError("batch targets must have shape (batch, 1) and values in [0, 1]")
    else:
        validate_class_targets(batch.targets, rows, classes)
    if type(batch.row_ids) is not tuple or len(batch.row_ids) != rows:
        raise ValueError("row_ids must be a tuple matching the batch row count")
    for row_id in batch.row_ids:
        _identifier(row_id, "row ID")
        if row_id in seen_rows:
            raise ValueError("row IDs must be globally unique across training and heldout data")
        seen_rows.add(row_id)
    if batch.episode is not None:
        episode = copy_episode(batch.episode)
        features, targets = episode_arrays(episode)
        if (
            episode.role != role
            or episode.num_classes != classes
            or episode.context != batch.context
            or batch.row_ids
            != tuple(f"{episode.episode_id}:step:{t.step}" for t in episode.transitions)
            or not np.array_equal(batch.features, features)
            or not np.array_equal(batch.targets, targets)
        ):
            raise ValueError(
                "episode batch metadata, ordered features and action targets must agree"
            )


def _copy_batch(batch: AttributionBatch) -> AttributionBatch:
    return AttributionBatch(
        batch.role,
        batch.context,
        batch.row_ids,
        batch.features.copy(),
        batch.targets.copy(),
        None if batch.episode is None else copy_episode(batch.episode),
        batch.num_classes,
    )


def _validated_batches(
    training: tuple[AttributionBatch, ...],
    heldout: tuple[AttributionBatch, ...],
    *,
    training_only: bool = False,
    training_role: str = "train",
) -> tuple[tuple[AttributionBatch, ...], tuple[AttributionBatch, ...]]:
    role = require_training_role(training_role)
    if type(training_only) is not bool:
        raise ValueError("training_only must be an exact bool")
    if type(training) is not tuple or not 1 <= len(training) <= 512:
        raise ValueError("training must be a tuple containing 1..512 batches")
    if training_only and (type(heldout) is not tuple or heldout):
        raise ValueError("training_only requires an exact empty heldout tuple")
    if not training_only and (type(heldout) is not tuple or not heldout):
        raise ValueError("heldout must be a nonempty tuple")
    if role == "probe_train" and (not training_only or len(training) > 16):
        raise ValueError("probe training requires training_only with at most 16 complete episodes")
    seen_rows: set[str] = set()
    for batch in training:
        _validate_batch(batch, role, seen_rows)
        if role == "probe_train" and batch.episode is None:
            raise ValueError("probe training requires complete episode metadata")
    for batch in heldout:
        _validate_batch(batch, "final_test", seen_rows)
    if any(batch.num_classes != training[0].num_classes for batch in (*training, *heldout)):
        raise ValueError("all training and heldout class counts must match")
    episodes = [batch.episode for batch in (*training, *heldout)]
    if any(episode is not None for episode in episodes):
        if any(episode is None for episode in episodes):
            raise ValueError("episode streams require complete episode metadata for every batch")
        episode_ids = [episode.episode_id for episode in episodes if episode is not None]
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError("episode IDs must be globally unique across training and final data")
    contexts = tuple(batch.context for batch in heldout)
    if len(contexts) != len(set(contexts)):
        raise ValueError("heldout contexts must be unique")
    if not training_only and any(batch.context not in contexts for batch in training):
        raise ValueError("every training context must have a heldout context")
    width = training[0].features.shape[1]
    if any(batch.features.shape[1] != width for batch in (*training, *heldout)):
        raise ValueError("all training and heldout feature widths must match")
    for train_batch in training:
        for test_batch in heldout:
            if any(
                np.shares_memory(train_array, test_array)
                for train_array in (train_batch.features, train_batch.targets)
                for test_array in (test_batch.features, test_batch.targets)
            ):
                raise ValueError("training and heldout arrays must not share memory")
    # Copy before any external learner/budget call can mutate caller-owned data.
    return tuple(map(_copy_batch, training)), tuple(map(_copy_batch, heldout))


def _validate_controls(
    checkpoint_updates: tuple[int, ...], label_delay: int, budget: ToyBudgetSession, updates: int
) -> None:
    if (
        type(checkpoint_updates) is not tuple
        or not 2 <= len(checkpoint_updates) <= 32
        or any(type(point) is not int for point in checkpoint_updates)
        or checkpoint_updates[0] != 0
        or checkpoint_updates[-1] != updates
        or any(left >= right for left, right in zip(checkpoint_updates, checkpoint_updates[1:]))
    ):
        raise ValueError("checkpoints must be 2..32 strictly increasing native counts from 0 to n")
    if type(label_delay) is not int or not 0 <= label_delay <= 512:
        raise ValueError("label_delay must be a native integer in 0..512")
    if type(budget) is not ToyBudgetSession:
        raise ValueError("attribution stream requires an exact ToyBudgetSession")


def _record_tick(
    runtime: ActorShadowRuntime[Array, Array, Array, Any],
    training: tuple[AttributionBatch, ...],
    tick: int,
    delay: int,
) -> None:
    if tick < len(training):
        episode = training[tick].episode
        runtime.record_experience(
            Experience(
                f"batch-{tick}",
                "attribution" if episode is None else episode.episode_id,
                tick,
                runtime._base_actor_version,
                training[tick].features,
                require_training_role(training[tick].role),
                ExperiencePermissions(training=True),
            )
        )
    label_index = tick - delay
    if 0 <= label_index < len(training):
        episode = training[label_index].episode
        runtime.record_label(
            LabelArrival(
                f"label-{label_index}",
                f"batch-{label_index}",
                "attribution" if episode is None else episode.episode_id,
                tick,
                runtime._base_actor_version,
                training[label_index].targets,
                role=require_training_role(training[label_index].role),
            )
        )


class _CheckpointPredictor(Protocol):
    def predict(self, features: Array) -> Array: ...

    def restore_state(self, state: Any) -> None: ...


class _CheckpointScorer(Protocol):
    def fork(self) -> _CheckpointPredictor: ...


def _score_checkpoints(
    learner: _CheckpointScorer,
    states: list[tuple[int, Any]],
    training: tuple[AttributionBatch, ...],
    heldout: tuple[AttributionBatch, ...],
    budget: ToyBudgetSession,
    *,
    training_contexts: tuple[str, ...] | None = None,
    verify_state: bool = False,
    check_boundary: Callable[[], None] | None = None,
    score_progress: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if type(verify_state) is not bool:
        raise ValueError("verify_state must be an exact bool")
    if check_boundary is not None and not callable(check_boundary):
        raise ValueError("check_boundary must be callable or None")
    if score_progress is not None:
        if type(score_progress) is not dict or score_progress:
            raise ValueError("score_progress must be a fresh empty exact dict")
        score_progress.update(
            status="running",
            reason=None,
            prediction_calls_attempted=0,
            prediction_calls_completed=0,
            prediction_rows_attempted=0,
            prediction_rows_completed=0,
            checkpoints=[],
        )
    if training_contexts is not None:
        if (
            training != ()
            or type(training_contexts) is not tuple
            or not 1 <= len(training_contexts) <= 512
            or any(
                type(c) is not str or not c or c != c.strip() or len(c) > 32
                for c in training_contexts
            )
            or any(type(n) is not int or not 0 <= n <= len(training_contexts) for n, _ in states)
        ):
            raise ValueError(
                "explicit scoring contexts require empty training and bounded context entries"
            )

    def boundary() -> None:
        budget.before_final()
        if check_boundary is not None:
            check_boundary()

    def verify(expected: Any) -> None:
        if not verify_state:
            return
        if check_boundary is not None:
            check_boundary()
        actual = getattr(scorer, "snapshot_state")()
        if check_boundary is not None:
            check_boundary()
        if not _same_value(expected, actual):
            raise ValueError("checkpoint evaluation native state differs from its captured state")

    try:
        boundary()
        scorer = learner.fork()
        boundary()
        if scorer is learner:
            raise ValueError("scorer fork must be distinct from the source learner")
        if verify_state and not callable(getattr(scorer, "snapshot_state", None)):
            raise ValueError("verified checkpoint evaluation requires snapshot_state")
        checkpoints = []
        for updates, state in states:
            expected = deepcopy(state) if verify_state else state
            context = (
                None
                if updates == 0
                else (
                    training[updates - 1].context
                    if training_contexts is None
                    else training_contexts[updates - 1]
                )
            )
            scores: dict[str, Any] = {}
            progress_point = dict(
                updates=updates, training_context=context, scores=scores, status="incomplete"
            )
            if score_progress is not None:
                score_progress["checkpoints"].append(progress_point)
            boundary()
            scorer.restore_state(deepcopy(expected) if verify_state else state)
            boundary()
            verify(expected)
            for batch in heldout:
                boundary()
                verify(expected)
                if score_progress is not None:
                    score_progress["prediction_calls_attempted"] += 1
                    score_progress["prediction_rows_attempted"] += len(batch.features)
                predictions = scorer.predict(batch.features.copy())
                if score_progress is not None:
                    score_progress["prediction_calls_completed"] += 1
                    score_progress["prediction_rows_completed"] += len(batch.features)
                else:
                    # Preserve the established guard/metric order for old callers.
                    boundary()
                    verify(expected)
                # Record a valid returned metric before a later clock/state guard
                # can fail. Paid partial results must not disappear on failure.
                scores[batch.context] = (
                    _probability_scores(predictions, batch.targets)
                    if batch.num_classes == 2
                    else _categorical_scores(predictions, batch.targets, batch.num_classes)
                )
                boundary()
                if score_progress is not None:
                    verify(expected)
            progress_point["status"] = "complete"
            checkpoints.append(dict(updates=updates, training_context=context, scores=scores))
        if check_boundary is not None:
            check_boundary()
        if score_progress is not None:
            score_progress["status"] = "complete"
        return checkpoints
    except Exception as error:
        if score_progress is not None:
            score_progress.update(status="failed", reason=f"{type(error).__name__}: {error}")
        raise


def _same_value(left: Any, right: Any) -> bool:
    """Exact owned-state values, including native Torch tensors without importing Torch."""
    if type(left) is not type(right):
        return False
    # Why: owned snapshots contain millions of primitive coefficient/history
    # leaves. Check them before reflective dataclass/tensor dispatch, preserving
    # exact types and NaN inequality without caching or skipping nested values.
    if type(left) in (str, int, float, bool, type(None)):
        return bool(left == right)
    if type(left) is np.ndarray:
        return (
            left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
        )
    if type(left).__module__ == "torch" and type(left).__name__ == "Tensor":
        return left.dtype == right.dtype and left.device == right.device and bool(left.equal(right))
    if is_dataclass(left) and not isinstance(left, type):
        return all(_same_value(getattr(left, f.name), getattr(right, f.name)) for f in fields(left))
    if type(left) is dict:
        return left.keys() == right.keys() and all(_same_value(left[k], right[k]) for k in left)
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(_same_value(a, b) for a, b in zip(left, right))
    return False


class AttributionPause:
    """One original owner's consumable local pause; observations carry no authority."""

    def __init__(self, owner: "AttributionStream", snapshot: dict[str, Any]) -> None:
        self._owner: AttributionStream | None = owner
        self._snapshot = deepcopy(snapshot)

    def snapshot_state(self) -> dict[str, Any]:
        if self._owner is None:
            raise ValueError("attribution pause has been consumed")
        return deepcopy(self._snapshot)


class AttributionStream:
    """The ordinary delayed stream loop with bounded owner-local continuation.

    A pause reconstructs candidate/inbox/replay payloads from owned snapshots on
    resume. Original runtime, actor, budgets, clocks and receipts remain binding.
    No process restart, managed lifecycle, renewal continuation or disk authority.
    """

    def __init__(
        self,
        learner: ForkableLearner[Array, Array, Array, Any],
        training: tuple[AttributionBatch, ...],
        heldout: tuple[AttributionBatch, ...],
        *,
        checkpoint_updates: tuple[int, ...],
        label_delay: int,
        budget: ToyBudgetSession,
        observe_checkpoint: Callable[[int, Any, AttributionBatch | None], None] | None = None,
        observe_features: Callable[[PrearrivalObservation, CandidateState[Any]], None]
        | None = None,
        rehearsal: StreamRehearsal | None = None,
        renewal: StreamUnitRenewal | None = None,
        capacity: StreamCapacity | None = None,
        training_only: bool = False,
        training_role: str = "train",
        actor: StableActor[Array, Array, Array, Any] | None = None,
    ) -> None:
        role = require_training_role(training_role)
        if actor is not None and (
            not isinstance(actor, StableActor) or role != "train" or not training_only
        ):
            raise ValueError("supplied actor requires a TRAIN-only stream")
        training, heldout = _validated_batches(
            training, heldout, training_only=training_only, training_role=role
        )
        _validate_controls(checkpoint_updates, label_delay, budget, len(training))
        if role == "probe_train" and (
            budget.budget.max_training_updates != len(training)
            or budget.updates_completed != 0
            or budget.replay_examples_completed != 0
            or renewal is not None
            or capacity is not None
            or observe_checkpoint is not None
            or observe_features is not None
        ):
            raise ValueError(
                "isolated probe streams require a fresh exact update budget without observers, "
                "renewal or capacity"
            )
        if observe_checkpoint is not None and not callable(observe_checkpoint):
            raise ValueError("observe_checkpoint must be callable or None")
        if observe_features is not None and not callable(observe_features):
            raise ValueError("observe_features must be callable or None")
        if rehearsal is not None and (renewal is not None or capacity is not None):
            raise ValueError("simultaneous rehearsal, unit renewal or capacity is not supported")
        if capacity is not None and type(capacity) is not StreamCapacity:
            raise ValueError("capacity must be an exact StreamCapacity or None")
        if renewal is not None and type(renewal) is not StreamUnitRenewal:
            raise ValueError("renewal must be an exact StreamUnitRenewal or None")
        if capacity is not None and renewal is not None and not capacity.observation_only:
            raise ValueError(
                "simultaneous scheduled capacity split and unit renewal is not supported"
            )
        maximum = 0
        if capacity is not None:
            if type(capacity) is not StreamCapacity:
                raise ValueError("capacity must be an exact StreamCapacity or None")
            maximum = capacity.bind(learner, budget, len(training))
        if renewal is not None:
            if type(renewal) is not StreamUnitRenewal:
                raise ValueError("renewal must be an exact StreamUnitRenewal or None")
            maximum = renewal.bind(learner, budget, len(training))
        if rehearsal is not None:
            if type(rehearsal) is not StreamRehearsal:
                raise ValueError("rehearsal must be an exact StreamRehearsal or None")
            schema = rehearsal.episode_schema
            if (schema is None) != (training[0].episode is None):
                raise ValueError("episode replay requires its declared episode schema and metadata")
            if training[0].num_classes > 2 and schema is None:
                raise ValueError(
                    "categorical rehearsal requires complete episodes and class schema"
                )
            if schema is not None and any(
                batch.episode is None
                or len(batch.episode.transitions) != schema.steps_per_episode
                or batch.num_classes != schema.num_classes
                for batch in training
            ):
                raise ValueError(
                    "episode training lengths/classes must match original replay schema"
                )
            if role == "train":
                maximum = rehearsal.bind(
                    learner, budget, len(training), training[0].features.shape[1]
                )
            else:
                maximum = rehearsal.bind(
                    learner, budget, len(training), training[0].features.shape[1], wake_role=role
                )
        self._learner, self._budget = learner, budget
        self._training, self._heldout = training, heldout
        # Why this: evaluation authority is fixed at admission, never inferred
        # from mutable data or a later reporting choice.
        self._training_role = self._original_training_role = role
        self._training_only = self._original_training_only = training_only
        self._checkpoints, self._delay = checkpoint_updates, label_delay
        self._observer, self._rehearsal, self._renewal = observe_checkpoint, rehearsal, renewal
        self._feature_observer = self._original_feature_observer = observe_features
        self._prearrival_counts = dict(
            calls_attempted=0,
            calls_completed=0,
            feature_rows_attempted=0,
            feature_rows_completed=0,
        )
        self._original_helper = rehearsal
        self._original_renewal = renewal
        self._capacity, self._original_capacity = capacity, capacity
        self._original_bindings = self._immutable_bindings()
        self._starting_updates = budget.updates_completed
        self._starting_wake_replay = budget.replay_examples_completed
        self._starting_replay_work = (
            None
            if rehearsal is None
            else (
                rehearsal._replay_budget.updates_completed,
                rehearsal._replay_budget.replay_examples_completed,
            )
        )
        self._completed, self._next_tick = 0, 0
        self._clock = LogicalClock()
        self._runtime = ActorShadowRuntime(
            learner,
            actor_version="a0" if actor is None else actor.version,
            candidate_version="c0",
            clock=self._clock,
            budget=budget,
            max_experiences=len(training),
            max_consolidations=maximum,
            training_role=role,
            actor=actor,
        )
        self._check_original()
        self._states = [(0, self._runtime.candidate_snapshot().state)]
        if capacity is not None:
            self._budget.before_final()
            capacity.start(self._runtime)
            self._budget.before_final()
        self._failed, self._finished = False, False
        self._pause: AttributionPause | None = None
        self._paused_state: dict[str, Any] | None = None
        self._paused_data: (
            tuple[tuple[AttributionBatch, ...], tuple[AttributionBatch, ...]] | None
        ) = None
        self._gate = Lock()
        if observe_checkpoint is not None:
            budget.before_final()
            observe_checkpoint(0, deepcopy(self._states[0][1]), None)
            budget.before_final()

    @property
    def completed_updates(self) -> int:
        return self._completed

    @property
    def next_tick(self) -> int:
        return self._next_tick

    @property
    def applied_updates(self) -> tuple[AppliedExperience, ...]:
        """Detach committed feedback receipts at a quiescent stream boundary."""
        with self._exclusive():
            self._check_original()
            return deepcopy(self._runtime.applied_updates)

    @property
    def applied_consolidations(self) -> tuple[AppliedConsolidation, ...]:
        """Detach published consolidation receipts at the original stream boundary."""
        with self._exclusive():
            self._check_original()
            return deepcopy(self._runtime.applied_consolidations)

    @property
    def prearrival_report(self) -> dict[str, Any] | None:
        """Owned paid callback counts, available even when the stream has failed.

        Forward, row and memory costs inside the observer require its own report;
        callback counts alone do not measure native prediction work.
        """
        if self._original_feature_observer is None:
            return None
        return dict(
            **self._prearrival_counts,
            prediction_subject="current_shadow_candidate",
            label_order="before_same_tick_label_arrival",
        )

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        if not self._gate.acquire(blocking=False):
            raise ValueError("attribution stream is busy")
        try:
            yield
        finally:
            self._gate.release()

    def _require_open(self, *, paused: bool = False) -> None:
        if self._failed or self._finished:
            raise ValueError("attribution stream is failed or finished")
        if not paused and self._pause is not None:
            raise ValueError("attribution stream is paused")

    def _immutable_bindings(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {key: value for key, value in binding.items() if key not in ("updates", "replay")}
            for binding in self._budget_bindings()
        )

    def _check_original(self) -> None:
        if (
            require_training_role(self._training_role) != self._original_training_role
            or self._runtime._training_role != self._original_training_role
            or self._runtime._original_training_role != self._original_training_role
            or self._runtime._inbox._training_role != self._original_training_role
            or any(
                batch.role != self._original_training_role
                or (
                    batch.episode is not None and batch.episode.role != self._original_training_role
                )
                for batch in self._training
            )
        ):
            raise ValueError("attribution requires its original training role")
        self._runtime._inbox._require_training_scope()
        if self._feature_observer is not self._original_feature_observer:
            raise ValueError("attribution requires its original feature observer")
        if (
            type(self._training_only) is not bool
            or self._training_only is not self._original_training_only
            or (self._training_only and (type(self._heldout) is not tuple or self._heldout))
        ):
            raise ValueError("attribution requires its original evaluation scope and heldout data")
        if any(
            type(count) is not int or count < 0
            for session in self._sessions()
            for count in (session.updates_completed, session.replay_examples_completed)
        ):
            raise ValueError("attribution cumulative charges must be nonnegative native integers")
        if (
            self._rehearsal is not self._original_helper
            or self._renewal is not self._original_renewal
            or self._capacity is not self._original_capacity
            or not _same_value(self._original_bindings, self._immutable_bindings())
            or self._runtime._budget is not self._budget
            or self._runtime._inbox._budget is not self._budget
            or self._runtime._inbox._clock is not self._clock
        ):
            raise ValueError(
                "attribution continuation requires its original budget, clock and helper authority"
            )
        if (
            self._budget.updates_completed != self._starting_updates + self._completed
            or self._budget.replay_examples_completed != self._starting_wake_replay
            or len(self._runtime._inbox._applied) != self._completed
        ):
            raise ValueError(
                "attribution cumulative wake charges differ from actual completed receipts"
            )
        if self._rehearsal is not None and self._starting_replay_work is not None:
            replay = self._rehearsal._replay_budget
            charged = sum(event["charged_exposures"] for event in self._rehearsal._events)
            if (
                replay.updates_completed != self._starting_replay_work[0]
                or replay.replay_examples_completed != self._starting_replay_work[1] + charged
                or self._rehearsal._observed != self._completed
            ):
                raise ValueError(
                    "attribution cumulative replay charges differ from actual event work"
                )
            self._rehearsal.check_incident_progress(self._completed)
        if self._capacity is not None:
            self._capacity._check_binding()
            if (
                type(self._capacity._calls) is not int
                or self._capacity._calls != self._completed
                or type(self._capacity._rows) is not int
                or self._capacity._rows
                != sum(len(b.features) for b in self._training[: self._completed])
            ):
                raise ValueError("attribution capacity work differs from actual completed wakes")
        if self._renewal is not None:
            self._renewal.check_progress(self._runtime, self._completed)

    def _checkpoint(self, *, batch_index: int | None = None) -> None:
        if self._completed not in self._checkpoints or self._completed == self._states[-1][0]:
            return
        self._states.append((self._completed, self._runtime.candidate_snapshot().state))
        if self._observer is not None:
            self._budget.before_final()
            self._observer(
                self._completed,
                deepcopy(self._states[-1][1]),
                _copy_batch(
                    self._training[self._completed - 1 if batch_index is None else batch_index]
                ),
            )
            self._budget.before_final()

    def _observe_features(self, tick: int) -> None:
        observer = self._feature_observer
        if observer is None or tick >= len(self._training):
            return
        self._budget.before_final()
        self._check_original()
        batch = self._training[tick]
        features = batch.features.copy()
        runtime = self._runtime
        candidate_owner, inbox_owner = runtime._candidate, runtime._inbox
        learner_owner, actor_owner = self._learner, runtime.actor
        source = deepcopy(self._learner.snapshot_state())
        actor = deepcopy(actor_owner.snapshot_state())

        def observe(candidate: CandidateState[Any]) -> None:
            expected = deepcopy(candidate)
            observation = PrearrivalObservation(tick, batch.row_ids, features.copy())
            self._budget.before_final()
            self._prearrival_counts["calls_attempted"] += 1
            self._prearrival_counts["feature_rows_attempted"] += len(features)
            try:
                observer(observation, deepcopy(candidate))
                self._prearrival_counts["calls_completed"] += 1
                self._prearrival_counts["feature_rows_completed"] += len(features)
            finally:
                # Why this: diagnostics spend the original session's wall/RSS
                # allowance even when their callback raises before returning.
                self._budget.before_final()
            if (
                self._feature_observer is not self._original_feature_observer
                or self._runtime is not runtime
                or runtime._candidate is not candidate_owner
                or runtime._inbox is not inbox_owner
                or self._learner is not learner_owner
                or runtime.actor is not actor_owner
                or not _same_value(source, learner_owner.snapshot_state())
                or not _same_value(actor, actor_owner.snapshot_state())
                or not _same_value(features, batch.features)
                or not _same_value(expected, runtime._snapshot_candidate())
            ):
                raise ValueError("prearrival observer changed its source, actor or shadow state")
            self._budget.before_final()

        # The existing runtime lease refuses training from inside the callback.
        # Prediction uses only the detached shadow state, never actor deployment.
        runtime.with_candidate_snapshot(observe)
        self._check_original()

    def run_until(self, completed_updates: int) -> None:
        """Advance to an absolute committed-wake boundary without any final scoring."""
        with self._exclusive():
            self._require_open()
            if type(completed_updates) is not int or not self._completed < completed_updates <= len(
                self._training
            ):
                raise ValueError("run_until requires a strictly later wake count within the stream")
            try:
                while self._completed < completed_updates:
                    self._check_original()
                    if self._next_tick >= len(self._training) + self._delay:
                        raise RuntimeError("attribution stream did not commit every delayed batch")
                    tick = self._next_tick
                    self._clock.advance_to(tick)
                    self._observe_features(tick)
                    _record_tick(self._runtime, self._training, tick, self._delay)
                    newly_completed = len(self._runtime.train_ready(max_updates=1))
                    self._completed += newly_completed
                    if self._rehearsal is not None and newly_completed:
                        batch = self._training[self._completed - 1]
                        if batch.episode is None:
                            self._rehearsal.after_update(
                                self._runtime, self._completed, batch.features, batch.targets
                            )
                        else:
                            self._rehearsal.after_update(
                                self._runtime,
                                self._completed,
                                batch.features,
                                batch.targets,
                                episode=batch.episode,
                            )
                    if self._capacity is not None and newly_completed:
                        self._capacity.after_update(
                            self._runtime,
                            self._completed,
                            len(self._training[self._completed - 1].features),
                        )
                    if self._renewal is not None and newly_completed:
                        self._observe_renewal()
                    self._checkpoint()
                    self._next_tick += 1
                    self._check_original()
            except BaseException:
                self._failed = True
                raise

    @property
    def reference_report(self) -> dict[str, Any] | None:
        """Detached finite-window progress, including paid failed attempts."""
        return deepcopy(getattr(self, "_reference_report", None))

    def _observe_reference_arrival(self, tick, index, batch, observer, boundary):
        """Read one admitted packet on a detached candidate under its existing lease."""
        self._budget.before_final()
        self._check_original()
        runtime, clock = self._runtime, self._clock
        source, actor = self._learner, runtime.actor
        inbox, candidate_owner = runtime._inbox, runtime._candidate
        source_state, actor_state = source.snapshot_state(), actor.snapshot_state()
        original_batch = _copy_batch(batch)

        def observe(candidate):
            expected = deepcopy(candidate)
            try:
                observer(tick, index, _copy_batch(batch), deepcopy(candidate))
            finally:
                self._budget.before_final()
            boundary()
            if (
                self._runtime is not runtime
                or self._clock is not clock
                or clock.now() != tick
                or self._learner is not source
                or runtime.actor is not actor
                or runtime._inbox is not inbox
                or runtime._candidate is not candidate_owner
                or not _same_value(source_state, source.snapshot_state())
                or not _same_value(actor_state, actor.snapshot_state())
                or not _same_value(original_batch, batch)
                or not _same_value(expected, runtime._snapshot_candidate())
            ):
                raise ValueError(
                    "arrived observer changed source, actor, packet, clock or candidate"
                )
            self._check_original()

        runtime.with_candidate_snapshot(observe)

    def run_reference_window(
        self,
        policy: ReferencePolicy,
        *,
        prequential_feedback: bool = False,
        ready_feedback: bool = False,
        acknowledge_correct: bool = False,
        observe_arrived: Callable[[int, int, AttributionBatch, CandidateState[Any]], None]
        | None = None,
    ) -> dict[str, Any]:
        if type(policy) is not ReferencePolicy:
            raise ValueError("reference window requires an exact fixed ReferencePolicy")
        return self._run_policy_window(
            policy,
            prequential_feedback=prequential_feedback,
            ready_feedback=ready_feedback,
            acknowledge_correct=acknowledge_correct,
            observe_arrived=observe_arrived,
        )

    def run_contextual_window(
        self, controller: CounterfactualController, *, acknowledge_correct: bool = False
    ) -> dict[str, Any]:
        """Apply an existing source-bound fit with internal arrived TRAIN feedback.

        No fit, arbitrary chooser callback, final/probe access or new budget is
        granted. Original fit measures and application costs remain separate.
        """
        if type(controller) is not CounterfactualController:
            raise ValueError("contextual window requires the original counterfactual owner")
        return self._run_policy_window(
            controller, ready_feedback=True, acknowledge_correct=acknowledge_correct
        )

    def _run_policy_window(
        self,
        selection: ReferencePolicy | CounterfactualController,
        *,
        prequential_feedback: bool = False,
        ready_feedback: bool = False,
        acknowledge_correct: bool = False,
        observe_arrived: Callable[[int, int, AttributionBatch, CandidateState[Any]], None]
        | None = None,
    ) -> dict[str, Any]:
        """Seal one ordinary TRAIN arrival window with fixed wake/wait choices.

        Unlike finish(), completing this window need not consume every packet.
        No final evaluation or implicit tail drain follows a waiting reference.
        Checkpoint observers receive the original detached committed states.
        """
        with self._exclusive():
            self._require_open()
            controller = selection if type(selection) is CounterfactualController else None
            policy = selection if type(selection) is ReferencePolicy else None
            if (
                type(acknowledge_correct) is not bool
                or (acknowledge_correct and (not ready_feedback or prequential_feedback))
                or type(prequential_feedback) is not bool
                or type(ready_feedback) is not bool
                or (observe_arrived is not None and not callable(observe_arrived))
                or (observe_arrived is not None and controller is not None)
                or (prequential_feedback and ready_feedback)
                or (controller is not None and (prequential_feedback or not ready_feedback))
                or (
                    policy is not None
                    and policy.mode == "drift"
                    and not (prequential_feedback or ready_feedback)
                )
                or not self._training_only
                or self._training_role != "train"
                or self._next_tick != 0
                or self._completed != 0
                or self._checkpoints != tuple(range(len(self._training) + 1))
                or self._budget.budget.max_training_updates is None
                or any(
                    helper is not None
                    for helper in (
                        self._rehearsal,
                        self._renewal,
                        self._capacity,
                        self._feature_observer,
                    )
                )
            ):
                raise ValueError(
                    "reference window requires a pristine ordinary TRAIN-only stream, "
                    "all wake checkpoints, finite credit and a supported fixed policy"
                )
            if policy is not None:
                policy.__post_init__()
            if (
                ready_feedback
                and len(self._training) * max(len(b.features) for b in self._training) > 4096
            ):
                raise ValueError("ready feedback worst-case diagnostic rows exceed4096")
            feedback_reader = ReferencePrequential(self._training) if prequential_feedback else None
            ready_reader = ReferencePrequential(self._training) if ready_feedback else None
            pinned: ReferencePolicy | FittedController | None = deepcopy(policy)
            report: dict[str, Any] = dict(
                status="running",
                window_complete=False,
                training_complete=False,
                policy=asdict(policy) if policy is not None else None,
                decisions=[],
                attempted_wakes=0,
            )
            self._reference_report = report
            arrival_attempts = arrival_returns = arrival_rows = returned_arrival_rows = 0
            arrival_bytes = 0
            arrival_record = None

            def arrival_counts():
                return dict(
                    calls_attempted=arrival_attempts,
                    calls_completed=arrival_returns,
                    rows_attempted=arrival_rows,
                    rows_completed=returned_arrival_rows,
                    input_array_bytes_attempted=arrival_bytes,
                    unreturned_work_unknown=arrival_attempts != arrival_returns,
                )

            if observe_arrived is not None:
                report["arrived_observation"] = arrival_record = arrival_counts()

            def observe_arrival(tick):
                nonlocal arrival_attempts, arrival_returns, arrival_rows, returned_arrival_rows
                nonlocal arrival_bytes, arrival_record
                index = tick - self._delay
                if observe_arrived is None or not 0 <= index < len(self._training):
                    return
                if report.get("arrived_observation") is not arrival_record or not _same_value(
                    arrival_record, arrival_counts()
                ):
                    raise ValueError("preceding arrived diagnostic charges changed")
                batch = self._training[index]
                previous_counts = arrival_counts()
                arrival_attempts += 1
                arrival_rows += len(batch.features)
                arrival_bytes += batch.features.nbytes + batch.targets.nbytes
                try:
                    self._observe_reference_arrival(tick, index, batch, observe_arrived, boundary)
                    if report.get("arrived_observation") is not arrival_record or not _same_value(
                        arrival_record, previous_counts
                    ):
                        raise ValueError("arrived diagnostic charges changed during callback")
                    arrival_returns += 1
                    returned_arrival_rows += len(batch.features)
                finally:
                    report["arrived_observation"] = arrival_record = arrival_counts()

            if feedback_reader is not None:
                report["prequential"] = feedback_reader.report
            if ready_reader is not None:
                report["ready_feedback"] = ready_reader.report
                ready_reader.report["feedback_mode"] = "oldest_ready"
            # Why: a verified consumption is not an optimizer update.
            # Keep packet progress separate from native fit/checkpoint counts.
            processed = 0
            ack_pins: list[tuple[Any, Any, Any]] = []
            ack_ledger = None
            ack_attempts = ack_returns = 0
            proof_reader = ReferencePrequential(self._training) if acknowledge_correct else None
            if acknowledge_correct:
                report["acknowledgement"] = dict(enabled=True)

            binding = None
            fitted: FittedController | None = None
            application: dict[str, Any] = dict(binding_reads_attempted=0, binding_reads_completed=0)
            if controller is not None:
                report["controller_application"] = application

            def read_binding():
                assert controller is not None
                self._check_original()
                self._budget.before_final()
                runtime, clock = self._runtime, self._clock
                tick, candidate_owner = clock.now(), runtime._candidate

                def observe(candidate):
                    if not _same_value(candidate.state, self._states[-1][1]):
                        raise ValueError(
                            "contextual candidate differs from its last owned checkpoint"
                        )
                    expected = deepcopy(candidate)
                    application["binding_reads_attempted"] += 1
                    try:
                        observed = controller.application_snapshot()
                        application["binding_reads_completed"] += 1
                    finally:
                        self._budget.before_final()
                    if (
                        self._runtime is not runtime
                        or self._clock is not clock
                        or runtime._inbox._clock is not clock
                        or clock.now() != tick
                        or runtime._candidate is not candidate_owner
                        or not _same_value(expected, runtime._snapshot_candidate())
                    ):
                        raise ValueError(
                            "contextual binding callback changed original clock or candidate"
                        )
                    return observed

                observed = runtime.with_candidate_snapshot(observe)
                self._check_original()
                return observed

            def check_acknowledgements():
                ledger = getattr(self._runtime._inbox, "_acknowledged", {})
                if ack_ledger is not None and ledger is not ack_ledger:
                    raise ValueError("original acknowledgement ledger was replaced")
                if len(ledger) != len(ack_pins) or processed != self._completed + len(ack_pins):
                    raise ValueError("processed cursor differs from original fit/ack receipts")
                for key, receipt, saved in ack_pins:
                    if ledger.get(key) is not receipt or not _same_value(receipt, saved):
                        raise ValueError("preceding acknowledgement changed or was replaced")

            def boundary():
                if acknowledge_correct:
                    check_acknowledgements()
                self._reference_boundary(
                    fitted if fitted is not None else policy, pinned, source, actor
                )
                if controller is not None:
                    observed = read_binding()
                    if not _same_value(observed, binding):
                        raise ValueError("original contextual fit or source binding changed")
                    for versioned in (observed["source"], observed["current_source"]):
                        if (
                            versioned.actor_version != self._runtime._base_actor_version
                            or not _same_value(versioned.state, source)
                            or not _same_value(versioned.state, actor.state)
                        ):
                            raise ValueError(
                                "fitted controller belongs to a different complete source or actor"
                            )
                    self._reference_boundary(fitted, pinned, source, actor)

            try:
                self._check_original()
                source = deepcopy(self._learner.snapshot_state())
                actor = self._runtime.actor.snapshot_state()
                if controller is not None:
                    binding = read_binding()
                    fitted = binding["policy"]
                    assert type(fitted) is FittedController
                    pinned = deepcopy(fitted)
                    report["policy"] = asdict(fitted)
                    application["fit"] = deepcopy(binding["fit"])
                for tick in range(len(self._training) + self._delay):
                    boundary()
                    self._clock.advance_to(tick)
                    if feedback_reader is not None and tick < len(self._training):
                        self._predict_reference_packet(tick, feedback_reader)
                        boundary()
                    _record_tick(self._runtime, self._training, tick, self._delay)
                    observe_arrival(tick)
                    feedback = None
                    label_index = tick - self._delay
                    if feedback_reader is not None and 0 <= label_index < len(self._training):
                        feedback = feedback_reader.arrive(
                            label_index,
                            tick,
                            self._delay,
                            self._training[label_index],
                            self._runtime.candidate_snapshot(),
                        )
                    limit = self._budget.budget.max_training_updates
                    assert limit is not None
                    arrived = max(0, min(len(self._training), tick - self._delay + 1))
                    index = processed if acknowledge_correct else self._completed
                    if ready_reader is not None and index < arrived:
                        feedback = self._predict_reference_packet(
                            index,
                            ready_reader,
                            scored_at=tick,
                            processed_index=index if acknowledge_correct else None,
                        )
                        boundary()
                    observation = ReferenceObservation(
                        tick,
                        len(self._training) + self._delay - tick,
                        arrived - index,
                        max(0, limit - self._budget.updates_completed),
                        self._runtime._base_actor_version,
                        feedback,
                    )
                    if fitted is not None:
                        choice = choose_contextual(fitted, observation)
                    else:
                        assert policy is not None
                        choice = choose_reference(policy, observation)
                    row = dict(
                        observation=asdict(observation), choice=asdict(choice), returned=False
                    )
                    report["decisions"].append(row)
                    if choice.action_id == "wake1":
                        report["attempted_wakes"] += 1
                        applied = self._runtime.train_ready(max_updates=1)
                        self._completed += len(applied)
                        if acknowledge_correct:
                            processed += len(applied)
                            self._checkpoint(batch_index=index)
                        else:
                            self._checkpoint()
                    elif acknowledge_correct and feedback is not None and feedback.mistakes == 0:
                        assert proof_reader is not None
                        ack_attempts += 1
                        receipt = self._acknowledge_reference_packet(
                            index, proof_reader, check_acknowledgements
                        )
                        ack_returns += 1
                        if receipt is not None:
                            key = (receipt.episode_id, receipt.sample_id)
                            if ack_ledger is None:
                                ack_ledger = self._runtime._inbox._acknowledged
                            ack_pins.append((key, receipt, deepcopy(receipt)))
                            processed += 1
                    self._next_tick += 1
                    boundary()
                    row.update(returned=True, completed_updates=self._completed)
                report.update(status="complete", window_complete=True)
                self._finished = True
            except BaseException as error:
                self._failed = True
                report.update(status="failed", error=f"{type(error).__name__}: {error}"[:512])
                raise
            finally:
                self._capture_reference_progress(report)
                if acknowledge_correct:
                    assert proof_reader is not None
                    proof_reader.report.update(
                        unreturned_prediction_unknown=proof_reader.report[
                            "prediction_calls_attempted"
                        ]
                        > proof_reader.report["prediction_calls_completed"],
                        unreturned_fork_unknown=proof_reader.report["fork_attempts"]
                        > proof_reader.report["fork_completed"],
                    )
                    report["acknowledgement"].update(
                        attempts=ack_attempts,
                        returned=ack_returns,
                        unreturned_work_unknown=ack_attempts > ack_returns,
                        proof=deepcopy(proof_reader.report),
                    )
                    report["known_acknowledgements"] = [asdict(saved) for _, _, saved in ack_pins]
                    try:
                        receipts = self._runtime._inbox.acknowledged_experiences
                        applied = self._runtime._inbox.applied_updates
                        report.update(
                            acknowledged_experiences=[asdict(r) for r in receipts],
                            processed_packets=len(applied) + len(receipts),
                            loop_processed_packets=processed,
                            pending_unapplied=len(self._training) - len(applied) - len(receipts),
                            unfitted_packets=len(self._training) - len(applied),
                            consumption_complete=report["window_complete"]
                            and len(applied) + len(receipts) == len(self._training),
                            training_complete=report["window_complete"]
                            and len(applied) == len(self._training),
                            all_packets_fitted=len(applied) == len(self._training),
                            training_examples=sum(
                                len(self._training[r.observed_at].features) for r in applied
                            ),
                            applied_packet_indices=[r.observed_at for r in applied],
                        )
                    except BaseException as error:
                        self._failed = True
                        report.update(
                            status="failed",
                            training_complete=False,
                            acknowledgement_capture_error=f"{type(error).__name__}: {error}"[:512],
                        )
            return deepcopy(report)

    def _predict_reference_packet(
        self,
        tick,
        reader: ReferencePrequential,
        *,
        scored_at=None,
        processed_index=None,
        owned_candidate=None,
    ):
        """Predict under a candidate lease without exposing labels to the fork."""
        if scored_at is not None and (
            type(scored_at) is not int
            or scored_at != self._next_tick
            or tick != (self._completed if processed_index is None else processed_index)
            or tick + self._delay > scored_at
        ):
            raise ValueError("ready feedback requires the original oldest eligible FIFO packet")

        def check_ready_clock():
            if scored_at is not None and self._clock.now() != scored_at:
                raise ValueError("original ready-feedback logical clock changed")

        check_ready_clock()
        batch = self._training[tick]
        runtime = self._runtime
        candidate_owner = runtime._candidate
        features = batch.features.copy()
        source = deepcopy(self._learner.snapshot_state())
        actor = deepcopy(runtime.actor.snapshot_state())

        def predict(candidate):
            expected = deepcopy(candidate)
            self._budget.before_final()
            reader.report["fork_attempts"] += 1
            predictor = self._learner.fork()
            reader.report["fork_completed"] += 1
            self._budget.before_final()
            check_ready_clock()
            if (
                predictor is self._learner
                or predictor is candidate_owner
                or runtime.actor._owns(predictor)
            ):
                raise ValueError("prequential fork must be independently owned")
            predictor.restore_state(deepcopy(candidate.state))
            self._budget.before_final()
            check_ready_clock()
            if (
                not _same_value(predictor.snapshot_state(), candidate.state)
                or not _same_value(source, self._learner.snapshot_state())
                or not _same_value(actor, runtime.actor.snapshot_state())
                or not _same_value(expected, runtime._snapshot_candidate())
            ):
                raise ValueError("prequential fork differs from captured candidate")
            reader.report["prediction_calls_attempted"] += 1
            reader.report["prediction_rows_attempted"] += len(features)
            try:
                predictions = predictor.predict(features.copy())
                reader.report["prediction_calls_completed"] += 1
                reader.report["prediction_rows_completed"] += len(features)
            finally:
                self._budget.before_final()
            check_ready_clock()
            if (
                runtime is not self._runtime
                or runtime._candidate is not candidate_owner
                or not _same_value(expected, runtime._snapshot_candidate())
                or not _same_value(predictor.snapshot_state(), expected.state)
                or not _same_value(features, batch.features)
                or (
                    owned_candidate is not None
                    and (
                        not _same_value(source, self._learner.snapshot_state())
                        or not _same_value(actor, runtime.actor.snapshot_state())
                    )
                )
            ):
                raise ValueError("prequential prediction changed captured or native state")
            reader.observe(tick, tick, batch, expected, predictions)
            self._budget.before_final()
            if scored_at is not None:
                feedback = reader.arrive(tick, tick + self._delay, self._delay, batch, expected)
                reader.report["events"][-1].update(
                    scored_at=scored_at, feedback_source="oldest_ready"
                )
                return feedback

        if owned_candidate is not None:
            if not runtime._inbox._draining or not _same_value(
                owned_candidate, runtime._snapshot_candidate()
            ):
                raise ValueError("acknowledgement proof requires original candidate lease")
            result = predict(owned_candidate)
        else:
            result = runtime.with_candidate_snapshot(predict)
        self._check_original()
        return result

    def _acknowledge_reference_packet(self, index, reader, check_acknowledgements):
        """Fresh verification inside the existing exclusive acknowledgement port."""
        runtime, batch = self._runtime, self._training[index]
        episode = "attribution" if batch.episode is None else batch.episode.episode_id
        key = (episode, f"batch-{index}")

        def verify(candidate, source, label):
            check_acknowledgements()
            if (
                source.key != key
                or source.observed_at != index
                or label.arrived_at != index + self._delay
                or not _same_value(source.features, batch.features)
                or not _same_value(label.targets, batch.targets)
            ):
                raise ValueError("acknowledgement differs from original arrived TRAIN packet")
            feedback = self._predict_reference_packet(
                index,
                reader,
                scored_at=self._clock.now(),
                processed_index=index,
                owned_candidate=candidate,
            )
            check_acknowledgements()
            return feedback is not None and feedback.mistakes == 0

        return runtime.acknowledge_ready(
            key,
            actor_version=runtime._base_actor_version,
            learner_version=runtime._candidate_version,
            wake_updates=self._budget.updates_completed,
            verify=verify,
        )

    def _reference_boundary(self, policy, pinned, source, actor) -> None:
        self._check_original()
        self._budget.before_final()
        if policy != pinned or not _same_value(self._learner.snapshot_state(), source):
            raise ValueError("original reference policy or source learner changed")
        if not _same_value(self._runtime.actor.snapshot_state(), actor):
            raise ValueError("original reference actor changed")
        self._check_original()

    def _capture_reference_progress(self, report) -> None:
        """A failed native return can leave receipts ahead of the loop counter."""
        application = report.get("controller_application")
        if application is not None:
            application["unreturned_binding_unknown"] = (
                application["binding_reads_attempted"] > application["binding_reads_completed"]
            )
        for key in ("prequential", "ready_feedback"):
            if key not in report:
                continue
            diagnostics = report[key]
            diagnostics["unreturned_prediction_unknown"] = (
                diagnostics["prediction_calls_attempted"]
                > diagnostics["prediction_calls_completed"]
            )
            diagnostics["unreturned_fork_unknown"] = (
                diagnostics["fork_attempts"] > diagnostics["fork_completed"]
            )
        report.update(
            completed_updates=self._completed,
            next_tick=self._next_tick,
            budget_updates=self._budget.updates_completed,
            replay_examples=self._budget.replay_examples_completed,
            score_forward_calls=0,
        )
        try:
            receipts = self._runtime.applied_updates
            report.update(
                applied_updates=[asdict(receipt) for receipt in receipts],
                training_examples=sum(len(b.features) for b in self._training[: len(receipts)]),
                pending_unapplied=len(self._training) - len(receipts),
                training_complete=report["window_complete"]
                and len(receipts) == len(self._training),
                unreturned_work_unknown=report["attempted_wakes"] > len(receipts),
            )
        except BaseException as error:
            self._failed = True
            report.update(
                status="failed",
                window_complete=False,
                training_complete=False,
                receipt_capture_error=f"{type(error).__name__}: {error}"[:512],
                unreturned_work_unknown=True,
            )

    def _observe_renewal(self) -> None:
        renewal = self._renewal
        assert renewal is not None
        if renewal is not self._original_renewal:
            raise ValueError("attribution requires its original unit renewal helper")
        renewal._check_binding()
        # Capacity has already recorded the state used for this wake. Renewal
        # owns its selection/charge; its after-state gets no additional wake vote.
        scheduled = self._completed % renewal._every == 0
        event = renewal._events[self._completed // renewal._every - 1] if scheduled else None
        try:
            renewal.after_update(self._runtime, self._completed)
        except Exception:
            if self._capacity is not None and scheduled:
                try:
                    self._capacity.after_consolidation(
                        self._runtime, self._completed, check_budget=False
                    )
                except Exception as error:
                    assert event is not None
                    event["capacity_observation_error"] = f"{type(error).__name__}: {error}"
            raise
        if self._capacity is not None and scheduled:
            self._capacity.after_consolidation(self._runtime, self._completed)

    def candidate_snapshot(self) -> Any:
        with self._exclusive():
            self._require_open(paused=True)
            return self._runtime.candidate_snapshot().state

    def _sessions(self) -> tuple[ToyBudgetSession, ...]:
        return (
            (self._budget,)
            if self._rehearsal is None
            else (self._budget, self._rehearsal._replay_budget)
        )

    def _budget_bindings(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            dict(
                session=id(session),
                limit_id=id(session.budget),
                limits=deepcopy(session.budget),
                clock=id(session.clock),
                started_at=session.started_at,
                progress=id(session.progress),
                sampler=id(session.process_rss_sampler),
                updates=session.updates_completed,
                replay=session.replay_examples_completed,
            )
            for session in self._sessions()
        )

    def _record(self) -> dict[str, Any]:
        """Caller holds the runtime lease; copy complete payload without its authority."""
        runtime = self._runtime
        self._check_original()
        runtime._require_open()
        if runtime.actor._payload_registry._lifecycle is not None:
            raise ValueError("attribution continuation does not support managed lifecycle owners")
        record = dict(
            format_version=1,
            completed_updates=self._completed,
            next_tick=self._next_tick,
            candidate_state=deepcopy(runtime._candidate.snapshot_state()),
            inbox_cursor=runtime._inbox.capture_cursor(),
            consolidation_cursor=deepcopy(runtime._read_consolidation_cursor()),
            checkpoint_states=deepcopy(self._states),
            event_tick=self._clock.now(),
            runtime_revision=runtime._revision,
            rehearsal_state=None
            if self._rehearsal is None
            else self._rehearsal._capture_continuation(),
            budget_bindings=self._budget_bindings(),
            authority=dict(
                runtime=id(runtime),
                actor=id(runtime.actor),
                learner=id(self._learner),
                candidate=id(runtime._candidate),
                inbox=id(runtime._inbox),
                event_clock=id(self._clock),
                observer=id(self._observer),
                checkpoints=self._checkpoints,
                delay=self._delay,
            ),
        )
        if self._training_role != "train":
            record["authority"]["training_role"] = self._training_role
        if self._training_only:
            record["authority"]["training_only"] = True
        if self._original_feature_observer is not None:
            record["authority"]["feature_observer"] = id(self._feature_observer)
            record["prearrival_observation"] = self.prearrival_report
        return record

    def pause(self) -> AttributionPause:
        """Capture one current complete local checkpoint; no history chain is retained."""
        with self._exclusive():
            self._require_open()
            if self._renewal is not None:
                raise ValueError("attribution renewal continuation is not supported")
            if self._capacity is not None:
                raise ValueError("attribution capacity continuation is not supported")
            if self._completed == 0:
                raise ValueError("attribution pause requires a committed wake boundary")
            try:
                for session in self._sessions():
                    session.before_final()
                with self._runtime._exclusive():
                    state = self._record()
                    state["clock_floors"] = tuple(
                        session.last_clock for session in self._sessions()
                    )
                    self._paused_data = deepcopy((self._training, self._heldout))
                    token = AttributionPause(self, state)
                    self._paused_state = deepcopy(state)
                for session in self._sessions():
                    session.before_final()
                self._pause = token
                return token
            except BaseException:
                self._failed = True
                raise

    def _check_paused(self, expected: dict[str, Any], token: AttributionPause) -> None:
        if type(token._snapshot) is not dict or token._snapshot.keys() != expected.keys():
            raise ValueError("attribution pause requires its exact original snapshot fields")
        ordinary = {
            key: value
            for key, value in expected.items()
            if key not in ("rehearsal_state", "clock_floors")
        }
        incoming = {
            key: value
            for key, value in token._snapshot.items()
            if key not in ("rehearsal_state", "clock_floors")
        }
        if not _same_value(ordinary, incoming) or not _same_value(
            expected["clock_floors"], token._snapshot.get("clock_floors")
        ):
            raise ValueError("attribution pause observation was corrupted or replaced")
        actual = self._record()
        helper = actual.pop("rehearsal_state")
        if not _same_value(ordinary, actual):
            raise ValueError("attribution live state changed since pause")
        if not _same_value(self._paused_data, (self._training, self._heldout)):
            raise ValueError("attribution source data changed since pause")
        # Only the helper's bounded NumPy payload uses pickle value comparison;
        # Torch storage serialization identifiers are intentionally never compared.
        if pickle.dumps(helper) != pickle.dumps(expected["rehearsal_state"]) or pickle.dumps(
            token._snapshot.get("rehearsal_state")
        ) != pickle.dumps(expected["rehearsal_state"]):
            raise ValueError("attribution replay state changed since pause")
        for session, floor in zip(self._sessions(), expected["clock_floors"], strict=True):
            if (
                type(session.last_clock) is not float
                or not isfinite(session.last_clock)
                or session.last_clock < floor
            ):
                raise ValueError("attribution original wall clock moved backwards")

    def resume(self, token: AttributionPause) -> None:
        """Reconstruct payloads against the unchanged original owner and spent authority."""
        with self._exclusive():
            self._require_open(paused=True)
            if (
                type(token) is not AttributionPause
                or token is not self._pause
                or token._owner is not self
                or self._paused_state is None
            ):
                raise ValueError("attribution resume requires its original unconsumed pause")
            runtime = self._runtime
            try:
                with runtime._exclusive():
                    expected = self._paused_state
                    self._check_paused(expected, token)
                    for session in self._sessions():
                        session.before_final()
                    candidate = self._learner.fork()
                    if (
                        candidate is runtime._candidate
                        or candidate is self._learner
                        or runtime.actor._owns(candidate)
                    ):
                        raise ValueError(
                            "attribution resumed candidate must be independently owned"
                        )
                    candidate.restore_state(deepcopy(expected["candidate_state"]))
                    if not _same_value(candidate.snapshot_state(), expected["candidate_state"]):
                        raise ValueError(
                            "attribution staged candidate differs from its complete pause state"
                        )
                    inbox = runtime._inbox._materialize_cursor(
                        candidate, deepcopy(expected["inbox_cursor"])
                    )
                    if not _same_value(inbox._borrow_cursor(), expected["inbox_cursor"]):
                        raise ValueError(
                            "attribution staged inbox differs from its complete pause cursor"
                        )
                    helper = (
                        None
                        if self._rehearsal is None
                        else self._rehearsal._prepare_continuation(expected["rehearsal_state"])
                    )
                    states = deepcopy(expected["checkpoint_states"])
                    for session in self._sessions():
                        session.before_final()
                    self._check_paused(expected, token)
                    # Publication contains plain assignments only. Original budget,
                    # actor, runtime, logical/wall clocks and receipts never reset.
                    old_inbox = runtime._inbox
                    runtime._candidate, runtime._inbox = candidate, inbox
                    if self._rehearsal is not None and helper is not None:
                        self._rehearsal._commit_continuation(helper)
                    runtime._revision += 1
                    old_inbox._historical_completed_updates = self._budget.updates_completed
                    self._states = states
                    self._completed, self._next_tick = (
                        expected["completed_updates"],
                        expected["next_tick"],
                    )
                    self._pause, self._paused_state = None, None
                    token._owner, token._snapshot = None, {}
                    self._paused_data = None
            except BaseException:
                self._failed = True
                raise

    def finish(self, *, score_progress: dict[str, Any] | None = None) -> dict[str, Any]:
        """Seal the complete stream, scoring only its admitted heldout evaluation."""
        with self._exclusive():
            self._require_open()
            if score_progress is not None and (
                self._training_only or type(score_progress) is not dict or score_progress
            ):
                raise ValueError(
                    "finish score_progress requires heldout evaluation and a fresh empty dict"
                )
            if (
                self._completed != len(self._training)
                or self._next_tick != len(self._training) + self._delay
            ):
                raise ValueError("attribution finish requires every delayed training tick")
            try:
                self._check_original()
                receipts = self._runtime.applied_updates
                if (
                    len(receipts) != len(self._training)
                    or tuple(point for point, _ in self._states) != self._checkpoints
                ):
                    raise RuntimeError("attribution stream has incomplete receipts or checkpoints")
                checkpoints = []
                if not self._training_only:
                    if score_progress is None:
                        checkpoints = _score_checkpoints(
                            self._learner, self._states, self._training, self._heldout, self._budget
                        )
                    else:
                        checkpoints = _score_checkpoints(
                            self._learner,
                            self._states,
                            self._training,
                            self._heldout,
                            self._budget,
                            score_progress=score_progress,
                            verify_state=True,
                        )
                self._budget.before_final()
                self._check_original()
                result = dict(
                    completed_updates=self._completed,
                    training_examples=sum(len(batch.features) for batch in self._training),
                    score_forward_calls=len(self._states) * len(self._heldout),
                    checkpoints=checkpoints,
                    applied_updates=[asdict(receipt) for receipt in receipts],
                )
                if self._training_role != "train":
                    result["training_role"] = self._training_role
                if self._training_only:
                    result["evaluation_scope"] = "training_only"
                if self._original_feature_observer is not None:
                    result["prearrival_observation"] = self.prearrival_report
                if self._training[0].num_classes > 2:
                    result["num_classes"] = self._training[0].num_classes
                if self._rehearsal is not None:
                    result["rehearsal"] = self._rehearsal.report
                if self._renewal is not None:
                    result["unit_renewal"] = self._renewal.report
                if self._capacity is not None:
                    result["capacity"] = self._capacity.report
                if any(
                    helper is not None
                    for helper in (self._rehearsal, self._renewal, self._capacity)
                ):
                    result["applied_consolidations"] = [
                        asdict(receipt) for receipt in self._runtime.applied_consolidations
                    ]
                self._finished = True
                return result
            except BaseException as error:
                self._failed = True
                if score_progress is not None and score_progress:
                    score_progress.update(
                        status="failed", reason=f"{type(error).__name__}: {error}"
                    )
                raise


def run_attribution_stream(
    learner: ForkableLearner[Array, Array, Array, Any],
    training: tuple[AttributionBatch, ...],
    heldout: tuple[AttributionBatch, ...],
    *,
    checkpoint_updates: tuple[int, ...],
    label_delay: int,
    budget: ToyBudgetSession,
    observe_checkpoint: Callable[[int, Any, AttributionBatch | None], None] | None = None,
    observe_features: Callable[[PrearrivalObservation, CandidateState[Any]], None] | None = None,
    rehearsal: StreamRehearsal | None = None,
    renewal: StreamUnitRenewal | None = None,
    capacity: StreamCapacity | None = None,
    training_only: bool = False,
) -> dict[str, Any]:
    """Train all delayed batches, then score fixed detached checkpoint states.

    Checkpoint counts must include zero and the complete stream. Accuracy uses
    threshold 0.5 for predictions and targets; BCE always clips at 1e-12.
    Declared roles/row IDs and disjoint storage are checked without claiming
    physical provenance. Opaque model state never appears in the returned report.
    Optional checkpoint observers receive separate state copies and only the
    last arrived training batch; initial observation has no labels or batch.
    Optional feature observers receive owned features and a versioned current
    shadow state before same-tick labels arrive, including with zero delay.
    They receive no targets or context, cannot train through the runtime lease,
    and do not deploy or promote the shadow candidate. Observer work consumes
    the original wall/RSS allowance; callbacks account for their native work.
    Optional shared rehearsal runs after newly committed wakes and before their
    checkpoint capture; it uses a separate explicitly charged replay budget.
    Explicit training_only requires an empty heldout tuple and seals receipts
    without scoring or returning checkpoint states; observers remain train-only.
    """
    stream = AttributionStream(
        learner,
        training,
        heldout,
        checkpoint_updates=checkpoint_updates,
        label_delay=label_delay,
        budget=budget,
        observe_checkpoint=observe_checkpoint,
        observe_features=observe_features,
        rehearsal=rehearsal,
        renewal=renewal,
        capacity=capacity,
        training_only=training_only,
    )
    stream.run_until(len(training))
    return stream.finish()
