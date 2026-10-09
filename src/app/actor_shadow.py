"""Stable local serving with exclusively mutated, detached candidate state.

Inputs: a trusted owned-fork learner, versions, clocks/budget and arrived training
events or detached-state consolidation transforms. Outputs: versioned owned actor
predictions, candidate snapshots and committed-work receipts. No promotion,
rollback, background worker, latency scheduler or scientific authority lives here.
"""

from contextlib import contextmanager
from copy import deepcopy
from threading import Lock
from typing import Callable, ContextManager, Generic, Iterator, TypeVar

from src.app.experience_inbox import ExperienceInbox
from src.core.native_update_origin import NativeUpdateObserver
from src.app.toy_execution_budget import ToyBudgetSession
from src.app.payload_ownership import PayloadOwnershipRegistry, lease_payload_lock
from src.core.payload_ownership import PayloadReferences
from src.core.actor_ports import (
    AppliedConsolidation,
    CandidateState,
    ConsolidatedState,
    ForkableLearner,
    VersionedPrediction,
    VersionedState,
)
from src.core.experience import (
    AppliedExperience,
    EventClock,
    Experience,
    LabelArrival,
    require_identifier,
)
from src.core.consolidation_cursor import ConsolidationCursor

Features = TypeVar("Features")
Targets = TypeVar("Targets")
Prediction = TypeVar("Prediction")
State = TypeVar("State")
Result = TypeVar("Result")


class StableActor(Generic[Features, Targets, Prediction, State]):
    """Own a serving fork; serialize prediction internals and detach results."""

    def __init__(
        self, learner: ForkableLearner[Features, Targets, Prediction, State], *, version: str
    ) -> None:
        require_identifier(version, "actor_version")
        self._learner = learner.fork()
        if self._learner is learner:
            raise ValueError("actor fork must be distinct from its source")
        self._version = version
        self._read_gate = Lock()
        self._payload_ready = True
        self._payload_registry = PayloadOwnershipRegistry()
        self._payload_registry.enroll("actor", self)

    def _payload_exclusive(self) -> ContextManager[None]:
        return lease_payload_lock(self._read_gate, "actor")

    def _payload_references(self) -> PayloadReferences:
        return PayloadReferences(models=(self._learner,))

    @property
    def version(self) -> str:
        return self._version

    def _owns(self, learner: object) -> bool:
        return self._learner is learner

    def predict(self, features: Features) -> VersionedPrediction[Prediction]:
        with self._read_gate:
            prediction = self._learner.predict(deepcopy(features))
            return VersionedPrediction(self._version, deepcopy(prediction))

    def snapshot_state(self) -> VersionedState[State]:
        with self._read_gate:
            if self._payload_registry._lifecycle is not None:
                self._payload_registry._lifecycle._require_access()
            return VersionedState(self._version, deepcopy(self._learner.snapshot_state()))


class ActorShadowRuntime(Generic[Features, Targets, Prediction, State]):
    def __init__(
        self,
        learner: ForkableLearner[Features, Targets, Prediction, State],
        *,
        actor_version: str,
        candidate_version: str,
        clock: EventClock,
        budget: ToyBudgetSession,
        max_experiences: int = 1024,
        max_consolidations: int = 8,
        actor: StableActor[Features, Targets, Prediction, State] | None = None,
    ) -> None:
        self._payload_ready = False
        self._payload_lineage = object()
        require_identifier(actor_version, "actor_version")
        require_identifier(candidate_version, "candidate_version")
        if actor_version == candidate_version:
            raise ValueError("actor and candidate versions must differ")
        if type(max_consolidations) is not int or max_consolidations < 0:
            raise ValueError("max_consolidations must be a nonnegative integer")
        self._validate_budget(budget)
        budget.before_sleep()
        if actor is not None and actor.version != actor_version:
            raise ValueError("supplied actor version must match candidate base")
        self._actor = actor if actor is not None else StableActor(learner, version=actor_version)
        # Why: charge ownership enrollment before a new candidate native fork.
        self._actor._payload_registry.enroll("candidate", self)
        self._base_actor_version = actor_version
        self._candidate = learner.fork()
        if self._candidate is learner or self._actor._owns(self._candidate):
            raise ValueError("candidate fork must be distinct from source and actor")
        self._inbox = ExperienceInbox(
            self._candidate,
            clock=clock,
            budget=budget,
            learner_version=candidate_version,
            max_experiences=max_experiences,
        )
        self._budget, self._candidate_version = budget, candidate_version
        self._consolidation_limit = max_consolidations
        self._attempted_ids: set[str] = set()
        self._consolidations: list[AppliedConsolidation] = []
        self._stopped = False
        self._retired, self._revision = False, 0
        self._write_gate = Lock()
        budget.before_final()
        self._payload_ready = True

    def _payload_exclusive(self) -> ContextManager[None]:
        return self._exclusive()

    def _payload_references(self) -> PayloadReferences:
        return PayloadReferences(models=(self._candidate,), inboxes=(self._inbox,))

    @staticmethod
    def _validate_budget(budget: ToyBudgetSession) -> None:
        if type(budget) is not ToyBudgetSession:
            raise ValueError("actor/shadow runtime requires ToyBudgetSession")
        if (
            budget.budget.max_hidden_width is not None
            or budget.budget.max_replay_examples is not None
        ):
            raise ValueError("actor/shadow runtime does not enforce width or replay limits")
        if budget.budget.max_process_rss_bytes is not None and budget.process_rss_sampler is None:
            raise ValueError("actor/shadow RSS budget requires an attached sampler")

    @property
    def actor(self) -> StableActor[Features, Targets, Prediction, State]:
        return self._actor

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        # Why: candidate contention/reentrance is an explicit refusal; serving
        # never waits on this lock, even during native update/restore failures.
        acquired_gate = self._write_gate
        if not acquired_gate.acquire(blocking=False):
            raise ValueError("candidate is busy; nested or concurrent operations are unsupported")
        try:
            yield
        finally:
            acquired_gate.release()

    def _require_open(self) -> None:
        if self._retired:
            raise ValueError("candidate owner is retired after checkpoint handoff")
        if self._stopped:
            raise ValueError("candidate is stopped after uncertain native mutation")
        if self._actor._payload_registry._lifecycle is not None:
            self._actor._payload_registry._lifecycle._require_access()

    def record_experience(self, source: Experience[Features]) -> None:
        with self._exclusive():
            self._require_open()
            self._revision += 1
            self._inbox.record_experience(source)

    def record_label(self, label: LabelArrival[Targets]) -> None:
        with self._exclusive():
            self._require_open()
            self._revision += 1
            self._inbox.record_label(label)

    def train_ready(
        self,
        *,
        max_updates: int | None = None,
        before_each_update: Callable[[], ContextManager[bool]] | None = None,
        native_observer: NativeUpdateObserver[Features, Targets] | None = None,
    ) -> tuple[AppliedExperience, ...]:
        with self._exclusive():
            self._require_open()
            self._revision += 1
            try:
                return self._inbox.drain(
                    max_updates=max_updates,
                    before_each_update=before_each_update,
                    native_observer=native_observer,
                )
            finally:
                self._stopped = self._inbox.stopped

    @property
    def applied_updates(self) -> tuple[AppliedExperience, ...]:
        with self._exclusive():
            return self._inbox.applied_updates

    @property
    def applied_consolidations(self) -> tuple[AppliedConsolidation, ...]:
        with self._exclusive():
            return tuple(self._consolidations)

    def capture_consolidation_cursor(self) -> ConsolidationCursor:
        """Observe complete metadata under the original lease,including closed owners.

        Why:failed transforms consume IDs;uncertain mutation and retirement flags
        must survive observation. No native payload or live authority is copied.
        """
        if type(self) is not ActorShadowRuntime:
            raise ValueError("consolidation capture requires an exact supported owner")
        with self._exclusive():
            return deepcopy(self._read_consolidation_cursor())

    def _read_consolidation_cursor(self) -> ConsolidationCursor:
        """Internal trusted read while the original candidate lease is held.

        Why: common-owner capture already holds this nonreentrant lease. Bound
        original histories before sorting/copying; no native operation runs.
        """
        if type(self) is not ActorShadowRuntime:
            raise ValueError("consolidation capture requires an exact supported owner")
        if type(self._attempted_ids) is not set or type(self._consolidations) is not list:
            raise ValueError("consolidation owner histories differ from supported storage")
        if (
            type(self._consolidation_limit) is not int
            or not 0 <= self._consolidation_limit < 2**63
            or len(self._attempted_ids) > self._consolidation_limit
            or len(self._consolidations) > len(self._attempted_ids)
        ):
            raise ValueError("consolidation histories exceed original consumed allowance")
        for event in self._attempted_ids:
            require_identifier(event, "attempted consolidation")
        return ConsolidationCursor(
            1,
            self._base_actor_version,
            self._candidate_version,
            self._consolidation_limit,
            tuple(sorted(self._attempted_ids)),
            tuple(self._consolidations),
            self._stopped,
            self._retired,
            self._revision,
            self._payload_ready,
        )

    def candidate_snapshot(self) -> CandidateState[State]:
        with self._exclusive():
            self._require_open()
            return self._snapshot_candidate()

    def _snapshot_candidate(self) -> CandidateState[State]:
        return CandidateState(
            self._base_actor_version,
            self._candidate_version,
            deepcopy(self._candidate.snapshot_state()),
            self._budget.updates_completed,
            len(self._attempted_ids),
            len(self._consolidations),
        )

    def with_candidate_snapshot(
        self, operation: Callable[[CandidateState[State]], Result]
    ) -> Result:
        """Keep the candidate unchanged through a trusted approval/commit callback.

        Why: snapshot-then-commit without this lease races candidate training.
        The callback receives detached state and cannot access a native handle.
        """
        if not callable(operation):
            raise ValueError("candidate snapshot operation must be callable")
        with self._exclusive():
            self._require_open()
            return operation(self._snapshot_candidate())

    def _owns_candidate(self, learner: object) -> bool:
        return self._candidate is learner

    def _prepare_handoff(self, candidate, cursor):
        """Materialize an independent owner under the caller's candidate lease.

        Why: constructors fork models and initialize cursors. This private path
        transfers complete validated history and retains the spent budget/clock.
        Publication and old-owner retirement belong to the checkpoint controller.
        """
        prepared = object.__new__(type(self))
        prepared._actor, prepared._base_actor_version = self._actor, self._base_actor_version
        prepared._candidate, prepared._candidate_version = candidate, self._candidate_version
        prepared._inbox = self._inbox._materialize_cursor(candidate, cursor)
        prepared._budget, prepared._consolidation_limit = self._budget, self._consolidation_limit
        prepared._attempted_ids = self._attempted_ids.copy()
        prepared._consolidations = deepcopy(self._consolidations)
        prepared._stopped, prepared._retired = self._stopped, False
        prepared._revision, prepared._write_gate = self._revision, Lock()
        prepared._payload_ready = False
        prepared._payload_lineage = self._payload_lineage
        self._actor._payload_registry.enroll("candidate", prepared)
        prepared._payload_ready = True
        return prepared

    def consolidate(
        self, event_id: str, operation: Callable[[State], ConsolidatedState[State]]
    ) -> AppliedConsolidation:
        with self._exclusive():
            self._require_open()
            self._revision += 1
            require_identifier(event_id, "consolidation event_id")
            if self._actor._payload_registry._lifecycle is not None:
                raise ValueError(
                    "managed retention does not support arbitrary consolidation transforms"
                )
            if not callable(operation):
                raise ValueError("consolidation operation must be callable")
            if event_id in self._attempted_ids:
                raise ValueError("duplicate consolidation event_id")
            if len(self._attempted_ids) >= self._consolidation_limit:
                raise ValueError("consolidation attempt quota exhausted")
            self._budget.before_sleep()
            # Charge before trusted code/copy, including failed transforms.
            self._attempted_ids.add(event_id)
            state = deepcopy(self._candidate.snapshot_state())
            result = operation(state)
            if type(result) is not ConsolidatedState:
                raise ValueError("consolidation must return ConsolidatedState")
            prepared = deepcopy(result.state)
            receipt = self._commit_consolidation(event_id, prepared, result)
            self._budget.before_final()
            return receipt

    def _commit_consolidation(
        self, event_id: str, prepared: State, result: ConsolidatedState[State]
    ) -> AppliedConsolidation:
        # Transform failures have not touched the private candidate. An
        # arbitrary restore failure may have; stop rather than retry it.
        try:
            self._candidate.restore_state(prepared)
        except BaseException:
            self._stopped = True
            raise
        receipt = AppliedConsolidation(
            event_id,
            self._base_actor_version,
            self._candidate_version,
            len(self._attempted_ids),
            result.diagnostic,
        )
        self._consolidations.append(receipt)
        return receipt


from src.app.expiry_release_proof import pin_release_methods as _pin_release_methods

_EXPIRY_ACTOR_RELEASE_PINS = _pin_release_methods(
    StableActor, ("_payload_exclusive", "_payload_references")
)
_EXPIRY_PROMOTABLE_LEASE_PINS = (_EXPIRY_ACTOR_RELEASE_PINS[0],)
_EXPIRY_CANDIDATE_RELEASE_PINS = _pin_release_methods(
    ActorShadowRuntime, ("_payload_exclusive", "_payload_references", "_exclusive")
)
