"""Complete same-process candidate ownership transfer with cumulative resources.

Inputs: a trusted shared runtime, independent native builder and complete state/
policy digests. Outputs: bounded single-use checkpoints, detached observations and
an atomically published replacement owner. Native policy, original budget/time/
RSS sampler, sharing gate/resource probe and serving actor stay owned in process.
No disk serialization, process-restart recovery, model tuning or scientific seal.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import pickle
import re
from threading import Lock
from typing import Callable, ContextManager, Generic, Iterator, TypeVar

from src.app.actor_shadow import ActorShadowRuntime, StableActor
from src.app.resource_sharing import ResourceSharedRuntime
from src.app.serving_promotion import PromotableActor
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionProgress
from src.core.actor_ports import AppliedConsolidation
from src.core.experience import LogicalClock, require_identifier, require_tick
from src.core.inbox_cursor import InboxCursor, validate_inbox_cursor
from src.core.learner_ports import NativeLearner, TrainingDiagnostic
from src.core.resource_sharing import SharingSnapshot
from src.core.payload_ownership import PayloadReferences
from src.core.native_graph_copy import copy_graph
from src.app.checkpoint_replay_handoff import (
    checkpoint_publication_lease,
    commit_replay_transition,
)
from src.shared.process_memory import ProcessRssSampler

Features = TypeVar("Features")
Targets = TypeVar("Targets")
Prediction = TypeVar("Prediction")
State = TypeVar("State")


@dataclass(frozen=True, eq=False)
class CandidateCheckpoint:
    """Identity token; copied/serialized tokens never acquire restore authority."""


@dataclass(frozen=True)
class CandidateCheckpointView(Generic[Features, Targets, State]):
    actor_version: str
    actor_generation: int
    learner_version: str
    state: State
    inbox: InboxCursor[Features, Targets]
    attempted_ids: tuple[str, ...]
    consolidations: tuple[AppliedConsolidation, ...]
    consolidation_limit: int
    stopped: bool
    budget_state: dict[str, object]
    sharing: SharingSnapshot
    event_tick: int


@dataclass
class _Pending(Generic[Features, Targets, Prediction, State]):
    owner: ActorShadowRuntime[Features, Targets, Prediction, State]
    revision: int
    view: CandidateCheckpointView[Features, Targets, State]
    integrity: str
    native_digest: str
    policy_digest: str
    build: Callable[[State], NativeLearner[Features, Targets, Prediction, State]]
    policy: Callable[[NativeLearner[Features, Targets, Prediction, State]], str]
    state_probe: object
    budget: ToyBudgetSession
    wall_clock: object
    event_clock: LogicalClock
    sampler: object
    progress: object
    resource: object
    sharing_gate: object
    actor: object
    sampler_state: object


def _digest(probe, value) -> str:
    result = probe(value)
    if type(result) is not str or re.fullmatch("[0-9a-f]{64}", result) is None:
        raise ValueError("checkpoint digest probe must return lowercase SHA-256")
    return result


def _integrity(view) -> str:
    # Trusted local payloads only, like deepcopy; never unpickle caller input.
    # This detects changed owned records and grants no graph/source certificate.
    return hashlib.sha256(pickle.dumps(view)).hexdigest()


def _budget_state(budget):
    raw = {
        k: v
        for k, v in vars(budget).items()
        if k not in {"clock", "process_rss_sampler", "progress"}
    }
    raw["progress_state"] = None if budget.progress is None else vars(budget.progress).copy()
    return deepcopy(raw)


def _sampler_state(sampler):
    if sampler is None:
        return None
    observed = sampler.snapshot()
    return sampler.read_rss_bytes, observed


def _actor_stamp(actor):
    # Caller holds the serving read gate; no native snapshot/callback is needed.
    if type(actor) is PromotableActor:
        return actor._slot.bundle.version, actor._slot.generation
    if type(actor) is StableActor:
        return actor._version, 0
    raise ValueError("checkpoint requires a supported exact stable/promotable actor")


def _validate_view(view) -> None:
    if type(view) is not CandidateCheckpointView:
        raise ValueError("checkpoint observation requires exact supported format")
    validate_inbox_cursor(view.inbox)
    for value in (view.actor_version, view.learner_version):
        require_identifier(value, "checkpoint version")
    for tick in (view.actor_generation, view.consolidation_limit, view.event_tick):
        require_tick(tick, "checkpoint cursor")
    if view.learner_version != view.inbox.learner_version or view.event_tick < view.inbox.last_tick:
        raise ValueError("checkpoint inbox version/time is inconsistent")
    if type(view.stopped) is not bool or (view.inbox.stopped and not view.stopped):
        raise ValueError("checkpoint stopped history cannot be reopened")
    if type(view.attempted_ids) is not tuple or type(view.consolidations) is not tuple:
        raise ValueError("checkpoint consolidation histories require immutable tuples")
    for event in view.attempted_ids:
        require_identifier(event, "checkpoint attempted consolidation")
    if (
        len(set(view.attempted_ids)) != len(view.attempted_ids)
        or len(view.attempted_ids) > view.consolidation_limit
    ):
        raise ValueError("checkpoint consolidation attempt quota/identity is invalid")
    seen, previous = set(), 0
    for receipt in view.consolidations:
        if (
            type(receipt) is not AppliedConsolidation
            or type(receipt.diagnostic) is not TrainingDiagnostic
        ):
            raise ValueError("checkpoint consolidation receipt/diagnostic is invalid")
        replace(receipt.diagnostic)
        require_tick(receipt.attempt_number, "consolidation attempt_number")
        if (
            receipt.event_id not in view.attempted_ids
            or receipt.event_id in seen
            or not previous < receipt.attempt_number <= len(view.attempted_ids)
            or receipt.actor_version != view.actor_version
            or receipt.learner_version != view.learner_version
        ):
            raise ValueError("checkpoint consolidation references/numbering are invalid")
        seen.add(receipt.event_id)
        previous = receipt.attempt_number


class CandidateCheckpointController(Generic[Features, Targets, Prediction, State]):
    def __init__(
        self,
        shared: ResourceSharedRuntime[Features, Targets, Prediction, State],
        *,
        build_learner: Callable[[State], NativeLearner[Features, Targets, Prediction, State]],
        state_digest: Callable[[State], str],
        policy_digest: Callable[[NativeLearner[Features, Targets, Prediction, State]], str],
        max_pending: int = 2,
        max_prepare_attempts: int = 4,
    ) -> None:
        if type(shared) is not ResourceSharedRuntime or not all(
            map(callable, (build_learner, state_digest, policy_digest))
        ):
            raise ValueError(
                "checkpoint controller requires shared runtime and trusted native probes"
            )
        for value in (max_pending, max_prepare_attempts):
            require_tick(value, "checkpoint capacity")
            if value == 0:
                raise ValueError("checkpoint capacities must be positive")
        self._shared, self._build, self._digest, self._policy = (
            shared,
            build_learner,
            state_digest,
            policy_digest,
        )
        self._limit, self._attempt_limit, self._attempts = max_pending, max_prepare_attempts, 0
        self._pending: dict[
            CandidateCheckpoint, _Pending[Features, Targets, Prediction, State]
        ] = {}
        self._models: list[NativeLearner[Features, Targets, Prediction, State]] = []
        self._gate = Lock()
        self._payload_ready = True
        shared._runtime.actor._payload_registry.enroll("checkpoint", self)

    def _payload_exclusive(self) -> ContextManager[None]:
        return self._exclusive()

    def _payload_references(self) -> PayloadReferences:
        return PayloadReferences(
            models=tuple(self._models), snapshots=tuple(p.view for p in self._pending.values())
        )

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        acquired_gate = self._gate
        if not acquired_gate.acquire(blocking=False):
            raise ValueError("checkpoint controller is busy; concurrent/nested operation refused")
        try:
            yield
        finally:
            acquired_gate.release()

    def _require(self, token):
        if type(token) is not CandidateCheckpoint or token not in self._pending:
            raise ValueError("checkpoint is foreign, copied, discarded or already used")
        return self._pending[token]

    def inspect(
        self, checkpoint: CandidateCheckpoint
    ) -> CandidateCheckpointView[Features, Targets, State]:
        with self._exclusive():
            pending = self._require(checkpoint)
            if pending.owner.actor._payload_registry._lifecycle is not None:
                pending.owner.actor._payload_registry._lifecycle._require_access()
            _validate_view(pending.view)
            return deepcopy(pending.view)

    def discard(self, checkpoint: CandidateCheckpoint) -> None:
        with self._exclusive():
            self._require(checkpoint)
            del self._pending[checkpoint]

    @staticmethod
    def _supported(owner):
        if type(owner) is not ActorShadowRuntime or owner._retired:
            raise ValueError("checkpoint requires a supported current candidate owner")
        if type(owner._inbox._clock) is not LogicalClock:
            raise ValueError("checkpoint supports the original exact LogicalClock only")
        budget = owner._budget
        if type(budget) is not ToyBudgetSession or (
            budget.progress is not None and type(budget.progress) is not ToyExecutionProgress
        ):
            raise ValueError("checkpoint requires original supported cumulative budget/progress")
        if (
            budget.process_rss_sampler is not None
            and type(budget.process_rss_sampler) is not ProcessRssSampler
        ):
            raise ValueError("checkpoint requires original supported ProcessRssSampler")
        owner._validate_budget(budget)

    def capture(self) -> CandidateCheckpoint:
        with self._exclusive():
            if len(self._pending) >= self._limit:
                raise ValueError(
                    "checkpoint pending capacity exhausted; discard obsolete checkpoints"
                )
            owner, gate = self._shared._runtime, self._shared._sharing
            with owner._exclusive(), gate._checkpoint_lease():
                self._supported(owner)
                if owner.actor._payload_registry._lifecycle is not None:
                    owner.actor._payload_registry._lifecycle._require_access()
                    owner.actor._payload_registry._lifecycle._before_checkpoint_capture(owner)
                owner._budget.before_final()
                with owner.actor._read_gate:
                    actor_version, generation = _actor_stamp(owner.actor)
                if actor_version != owner._base_actor_version:
                    raise ValueError("checkpoint actor changed from candidate base")
                view = CandidateCheckpointView(
                    actor_version,
                    generation,
                    owner._candidate_version,
                    copy_graph(self, "checkpoint_capture", owner._candidate.snapshot_state()),
                    owner._inbox.capture_cursor(),
                    tuple(sorted(owner._attempted_ids)),
                    tuple(deepcopy(owner._consolidations)),
                    owner._consolidation_limit,
                    owner._stopped,
                    _budget_state(owner._budget),
                    gate.snapshot(),
                    owner._inbox._clock.now(),
                )
                _validate_view(view)
                token = CandidateCheckpoint()
                budget = owner._budget
                event_clock = owner._inbox._clock
                if type(event_clock) is not LogicalClock:
                    raise ValueError("checkpoint supports the original exact LogicalClock only")
                pending = _Pending(
                    owner,
                    owner._revision,
                    view,
                    _integrity(view),
                    _digest(self._digest, view.state),
                    _digest(self._policy, owner._candidate),
                    self._build,
                    self._policy,
                    self._digest,
                    budget,
                    budget.clock,
                    event_clock,
                    budget.process_rss_sampler,
                    budget.progress,
                    gate._resource,
                    gate,
                    owner.actor,
                    _sampler_state(budget.process_rss_sampler),
                )
                if owner.actor._payload_registry._lifecycle is not None:
                    owner.actor._payload_registry._lifecycle._require_access()
                self._pending[token] = pending
                return token

    def _check_current(self, pending):
        owner, gate, view = pending.owner, self._shared._sharing, pending.view
        self._supported(owner)
        if owner.actor._payload_registry._lifecycle is not None:
            owner.actor._payload_registry._lifecycle._require_access()
        _validate_view(view)
        if _integrity(view) != pending.integrity:
            raise ValueError("checkpoint complete observation changed")
        if (
            self._shared._runtime is not owner
            or owner._revision != pending.revision
            or self._build is not pending.build
            or self._policy is not pending.policy
            or self._digest is not pending.state_probe
            or owner._budget is not pending.budget
            or owner._inbox._clock is not pending.event_clock
            or owner._budget.clock is not pending.wall_clock
            or owner._budget.progress is not pending.progress
            or owner._budget.process_rss_sampler is not pending.sampler
            or gate._resource is not pending.resource
            or gate is not pending.sharing_gate
            or owner.actor is not pending.actor
        ):
            raise ValueError(
                "checkpoint owner, work, builder or resource identity changed; stale checkpoint"
            )
        if (
            owner._inbox._clock.now() < view.event_tick
            or _digest(self._digest, owner._candidate.snapshot_state()) != pending.native_digest
            or _digest(self._policy, owner._candidate) != pending.policy_digest
        ):
            raise ValueError("checkpoint native policy/state or event chronology changed")
        if (
            _integrity(owner._inbox.capture_cursor()) != _integrity(view.inbox)
            or tuple(sorted(owner._attempted_ids)) != view.attempted_ids
            or _integrity(tuple(owner._consolidations)) != _integrity(view.consolidations)
            or owner._consolidation_limit != view.consolidation_limit
            or owner._stopped != view.stopped
        ):
            raise ValueError("checkpoint complete candidate history/limits changed")
        if pending.sampler is not None:
            reader, captured_rss = pending.sampler_state
            current_reader, current_rss = _sampler_state(pending.sampler)
            if (
                reader is not current_reader
                or current_rss.pid != captured_rss.pid
                or current_rss.start_bytes != captured_rss.start_bytes
                or current_rss.interval_seconds != captured_rss.interval_seconds
                or current_rss.peak_bytes < captured_rss.peak_bytes
                or current_rss.sample_count < captured_rss.sample_count
            ):
                raise ValueError(
                    "checkpoint original RSS resource state changed or moved backwards"
                )
        # These cumulative observations may advance, never reset. Remaining
        # budget/config/work/progress identity must match the captured owner.
        current = _budget_state(owner._budget)
        captured = deepcopy(view.budget_state)
        if current["last_clock"] < captured["last_clock"]:
            raise ValueError("checkpoint budget clock moved backwards")
        for key in ("last_clock", "process_rss_segment"):
            current.pop(key)
            captured.pop(key)
        for state in (current, captured):
            if state["progress_state"] is not None:
                state["progress_state"].pop("process_rss_segment", None)
        if current != captured:
            raise ValueError("checkpoint cumulative budget/work/configuration changed")
        if gate.snapshot() != view.sharing:
            raise ValueError("checkpoint sharing quota/pause/deferral state changed")

    def restore(
        self, checkpoint: CandidateCheckpoint
    ) -> ActorShadowRuntime[Features, Targets, Prediction, State]:
        with self._exclusive():
            pending = self._require(checkpoint)
            owner, gate = pending.owner, self._shared._sharing
            with owner._exclusive(), gate._checkpoint_lease():
                self._check_current(pending)
                if self._attempts >= self._attempt_limit:
                    raise ValueError("checkpoint preparation attempt quota exhausted")
                self._attempts += 1
                owner._budget.before_final()
                if owner.actor._payload_registry._lifecycle is not None:
                    owner.actor._payload_registry._lifecycle._before_checkpoint_restore(
                        self._build, pending.view
                    )
                model = self._build(copy_graph(self, "checkpoint_build_state", pending.view.state))
                if not all(
                    callable(getattr(model, name, None))
                    for name in ("train_batch", "predict", "snapshot_state", "restore_state")
                ):
                    raise ValueError("checkpoint builder must return a complete native learner")
                if (
                    owner._owns_candidate(model)
                    or owner.actor._owns(model)
                    or any(model is old for old in self._models)
                ):
                    raise ValueError("checkpoint prepared native model must be independently owned")
                self._models.append(model)
                if _digest(self._policy, model) != pending.policy_digest:
                    raise ValueError("checkpoint prepared native policy changed")
                if owner.actor._payload_registry._lifecycle is not None:
                    owner.actor._payload_registry._lifecycle._require_access()
                model.restore_state(
                    copy_graph(self, "checkpoint_restore_state", pending.view.state)
                )
                if (
                    _digest(self._digest, model.snapshot_state()) != pending.native_digest
                    or _digest(self._policy, model) != pending.policy_digest
                ):
                    raise ValueError("checkpoint native restore failed complete state equality")
                if owner.actor._payload_registry._lifecycle is not None:
                    owner.actor._payload_registry._lifecycle._require_access()
                prepared = owner._prepare_handoff(model, pending.view.inbox)
                owner._budget.before_final()
                self._check_current(pending)
                if (
                    _digest(self._digest, model.snapshot_state()) != pending.native_digest
                    or _digest(self._policy, model) != pending.policy_digest
                ):
                    raise ValueError(
                        "checkpoint prepared model state/policy changed before publication"
                    )
                with owner.actor._read_gate:
                    if _actor_stamp(owner.actor) != (
                        pending.view.actor_version,
                        pending.view.actor_generation,
                    ):
                        raise ValueError("checkpoint actor generation changed; stale checkpoint")
                    with checkpoint_publication_lease(self, owner, prepared, pending) as transition:
                        # Single publication point; no callback/copy follows retirement.
                        owner._inbox._retire_ledger()
                        owner._retired = True
                        self._shared._runtime = prepared
                        self._pending.clear()
                        if transition is not None:
                            commit_replay_transition(transition)
                return prepared
