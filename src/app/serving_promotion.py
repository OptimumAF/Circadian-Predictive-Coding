"""Guard-issued complete serving swaps and one-step rollback.

Trusted native builders/probes own independent models; this module neither
certifies arbitrary Python graphs nor grants global final-label authority.
Preparation holds the candidate gate, never the serving gate during native
restore/evaluation. Commit and rollback publish one complete slot under the same
gate used by every read; no callbacks run after its single publication point.
"""

from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass
import re
from threading import Lock
from typing import Callable, ContextManager, Generic, Iterator, TypeVar

from src.app.actor_shadow import ActorShadowRuntime, StableActor
from src.app.promotion_guard_evaluation import PromotionGuardEvaluator
from src.core.actor_ports import (
    CandidateState,
    ForkableLearner,
    VersionedPrediction,
    VersionedState,
)
from src.core.experience import SampleKey, require_tick
from src.core.learner_ports import NativeLearner
from src.core.promotion_guard import (
    GuardBatch,
    PromotionGuardReport,
    PromotionPolicy,
    decide_promotion,
)
from src.core.serving_ports import (
    CachedPrediction,
    PreparedPromotion,
    PromotionReceipt,
    ServingConfiguration,
    ServingPrediction,
    ServingSnapshot,
)
from src.core.payload_ownership import PayloadReferences

Features = TypeVar("Features")
Targets = TypeVar("Targets")
Prediction = TypeVar("Prediction")
State = TypeVar("State")


def _digest(probe: Callable[[State], str], state: State) -> str:
    value = probe(deepcopy(state))
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("configured digest must return lowercase SHA256")
    return value


@dataclass(frozen=True)
class _Bundle(Generic[Features, Targets, Prediction, State]):
    learner: NativeLearner[Features, Targets, Prediction, State]
    version: str
    configuration: ServingConfiguration
    cache: dict[str, CachedPrediction[Prediction]]
    metadata: dict[str, object]
    last_cache_tick: int


@dataclass(frozen=True)
class _Slot(Generic[Features, Targets, Prediction, State]):
    bundle: _Bundle[Features, Targets, Prediction, State]
    generation: int
    previous: _Bundle[Features, Targets, Prediction, State] | None = None


class PromotableActor(StableActor[Features, Targets, Prediction, State]):
    def __init__(
        self,
        learner: ForkableLearner[Features, Targets, Prediction, State],
        *,
        version: str,
        configuration: ServingConfiguration,
        feature_digest: Callable[[Features], str],
        metadata: dict[str, object],
    ) -> None:
        if type(configuration) is not ServingConfiguration or not callable(feature_digest):
            raise ValueError("serving requires declared cache configuration and feature digest")
        if type(metadata) is not dict or any(type(key) is not str for key in metadata):
            raise ValueError("serving metadata must be a dictionary with string keys")
        super().__init__(learner, version=version)
        self._payload_ready = False
        self._feature_digest = feature_digest
        self._slot = _Slot(
            _Bundle(self._learner, version, configuration, {}, deepcopy(metadata), 0), 0
        )
        del self._learner  # ownership transferred; do not retain an obsolete third model
        self._payload_ready = True

    def _payload_references(self) -> PayloadReferences:
        bundles = (
            (self._slot.bundle,)
            if self._slot.previous is None
            else (self._slot.bundle, self._slot.previous)
        )
        return PayloadReferences(
            models=tuple(b.learner for b in bundles),
            auxiliary=tuple(v for b in bundles for v in (b.metadata, b.cache)),
        )

    @property
    def version(self) -> str:
        with self._read_gate:
            return self._slot.bundle.version

    def _owns(self, learner: object) -> bool:
        with self._read_gate:
            slot = self._slot
            return learner is slot.bundle.learner or (
                slot.previous is not None and learner is slot.previous.learner
            )

    def predict(self, features: Features) -> VersionedPrediction[Prediction]:
        with self._read_gate:
            bundle = self._slot.bundle
            prediction = deepcopy(bundle.learner.predict(deepcopy(features)))
            return VersionedPrediction(bundle.version, prediction)

    def snapshot_state(self) -> VersionedState[State]:
        with self._read_gate:
            if self._payload_registry._lifecycle is not None:
                self._payload_registry._lifecycle._require_access()
            bundle = self._slot.bundle
            return VersionedState(bundle.version, deepcopy(bundle.learner.snapshot_state()))

    def serving_snapshot(self) -> ServingSnapshot[State, Prediction]:
        with self._read_gate:
            if self._payload_registry._lifecycle is not None:
                self._payload_registry._lifecycle._require_access()
            slot, bundle = self._slot, self._slot.bundle
            return ServingSnapshot(
                slot.generation,
                bundle.version,
                deepcopy(bundle.learner.snapshot_state()),
                bundle.configuration,
                deepcopy(bundle.cache),
                deepcopy(bundle.metadata),
                bundle.last_cache_tick,
            )

    def serve(self, features: Features, *, now: int) -> ServingPrediction[Prediction]:
        require_tick(now, "serving cache tick")
        with self._read_gate:
            slot, bundle = self._slot, self._slot.bundle
            lifecycle = self._payload_registry._lifecycle
            if lifecycle is not None:
                lifecycle._validate_cache_input(features)
            if now < bundle.last_cache_tick:
                raise ValueError("serving cache tick moved backwards")
            key = _digest(self._feature_digest, features)
            cached = bundle.cache.get(key)
            hit = cached is not None and now < cached.expires_at
            bound = (
                None
                if hit or lifecycle is None
                else lifecycle._before_cache_write(bundle.learner, features)
            )
            prediction = (
                deepcopy(cached.prediction)
                if hit and cached is not None
                else deepcopy(bundle.learner.predict(deepcopy(features)))
            )
            if lifecycle is not None:
                lifecycle._validate_cache_output(prediction, bound)
            # Allocate/copy everything before updating the observable cache slot.
            result = ServingPrediction(
                slot.generation,
                bundle.version,
                deepcopy(prediction),
                deepcopy(bundle.metadata),
                hit,
            )
            cache = {k: v for k, v in bundle.cache.items() if now < v.expires_at}
            if not hit:
                cache[key] = CachedPrediction(
                    now + bundle.configuration.cache_ttl_ticks, deepcopy(prediction)
                )
                while len(cache) > bundle.configuration.max_cache_entries:
                    del cache[next(iter(cache))]
            updated = _Bundle(
                bundle.learner, bundle.version, bundle.configuration, cache, bundle.metadata, now
            )
            self._slot = _Slot(updated, slot.generation, slot.previous)
            return result

    def _commit(
        self,
        bundle: _Bundle[Features, Targets, Prediction, State],
        *,
        generation: int,
        actor_digest: str,
        state_digest: Callable[[State], str],
    ) -> PromotionReceipt:
        with self._read_gate:
            slot = self._slot
            if slot.generation != generation:
                raise ValueError("actor generation is stale")
            if _digest(state_digest, slot.bundle.learner.snapshot_state()) != actor_digest:
                raise ValueError("actor model state changed after guard evaluation")
            if self._payload_registry._lifecycle is not None:
                self._payload_registry._lifecycle._require_access()
            receipt = PromotionReceipt(slot.generation + 1, slot.bundle.version, bundle.version)
            prepared = _Slot(bundle, receipt.generation, slot.bundle)
            self._slot = prepared
            return receipt

    def _rollback(self, receipt: PromotionReceipt) -> int:
        with self._read_gate:
            if self._payload_registry._lifecycle is not None:
                self._payload_registry._lifecycle._require_access()
            slot = self._slot
            if slot.generation != receipt.generation or slot.previous is None:
                raise ValueError("rollback generation is stale or already consumed")
            generation = slot.generation + 1
            prepared = _Slot(slot.previous, generation)
            self._slot = prepared
            return generation


@dataclass(frozen=True)
class _Pending(Generic[Features, Targets, Prediction, State]):
    runtime: ActorShadowRuntime[Features, Targets, Prediction, State]
    report: PromotionGuardReport
    bundle: _Bundle[Features, Targets, Prediction, State]
    actor_generation: int
    builder: Callable[[State], NativeLearner[Features, Targets, Prediction, State]]


class ServingPromotionController(Generic[Features, Targets, Prediction, State]):
    def __init__(
        self,
        actor: PromotableActor[Features, Targets, Prediction, State],
        *,
        policy: PromotionPolicy,
        evaluator: PromotionGuardEvaluator[Features, Targets, Prediction, State],
        build_learner: Callable[[State], NativeLearner[Features, Targets, Prediction, State]],
        state_digest: Callable[[State], str],
        max_pending: int = 4,
    ) -> None:
        if not isinstance(actor, PromotableActor) or type(policy) is not PromotionPolicy:
            raise ValueError("promotion controller requires promotable actor and declared policy")
        if type(evaluator) is not PromotionGuardEvaluator or not all(
            map(callable, (build_learner, state_digest))
        ):
            raise ValueError("promotion controller requires configured evaluator/builder/digest")
        require_tick(max_pending, "max_pending")
        if evaluator._build is not build_learner:
            raise ValueError("guard and serving preparation must use the identical native builder")
        if max_pending == 0:
            raise ValueError("pending ticket capacity must be positive")
        self._actor, self._policy, self._evaluator = actor, policy, evaluator
        self._build, self._digest, self._limit = build_learner, state_digest, max_pending
        self._pending: dict[PreparedPromotion, _Pending[Features, Targets, Prediction, State]] = {}
        self._latest: tuple[PromotionReceipt, PromotionReceipt] | None = None
        self._gate = Lock()
        self._payload_ready = True
        actor._payload_registry.enroll("promotion", self)

    def _payload_exclusive(self) -> ContextManager[None]:
        return self._exclusive()

    def _payload_references(self) -> PayloadReferences:
        bundles = tuple(p.bundle for p in self._pending.values())
        return PayloadReferences(
            models=tuple(b.learner for b in bundles),
            auxiliary=tuple(v for b in bundles for v in (b.metadata, b.cache)),
        )

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        if not self._gate.acquire(blocking=False):
            raise ValueError("promotion controller is busy; nested/concurrent operations refused")
        try:
            yield
        finally:
            self._gate.release()

    def _require_runtime(
        self, runtime: ActorShadowRuntime[Features, Targets, Prediction, State]
    ) -> None:
        if not isinstance(runtime, ActorShadowRuntime) or runtime.actor is not self._actor:
            raise ValueError("promotion runtime must own this controller's actor")

    def prepare(
        self,
        runtime: ActorShadowRuntime[Features, Targets, Prediction, State],
        new: GuardBatch[Features, Targets],
        old: GuardBatch[Features, Targets],
        *,
        now: int,
        training_ids: frozenset[SampleKey],
        metadata: dict[str, object] | None = None,
    ) -> PreparedPromotion:
        self._require_runtime(runtime)
        with self._exclusive():
            if self._build is not self._evaluator._build:
                raise ValueError("configured guard/serving native builder changed")
            if len(self._pending) >= self._limit:
                raise ValueError(
                    "pending promotion ticket quota exhausted; discard obsolete tickets"
                )

            # One lock order: controller -> candidate -> serving. Callbacks cannot
            # reenter either operation gate; serving never takes the outer gates.
            def prepare_owned(candidate: CandidateState[State]) -> PreparedPromotion:
                lifecycle = self._actor._payload_registry._lifecycle
                if lifecycle is not None:
                    context = self._actor._slot.bundle.metadata if metadata is None else metadata
                    lifecycle._before_promotion(self._build, candidate.state, context)
                return self._prepare_owned(
                    runtime, candidate, new, old, now, training_ids, metadata
                )

            return runtime.with_candidate_snapshot(prepare_owned)

    def _prepare_owned(
        self,
        runtime: ActorShadowRuntime[Features, Targets, Prediction, State],
        candidate: CandidateState[State],
        new: GuardBatch[Features, Targets],
        old: GuardBatch[Features, Targets],
        now: int,
        training_ids: frozenset[SampleKey],
        metadata: dict[str, object] | None,
    ) -> PreparedPromotion:
        if type(training_ids) is not frozenset:
            raise ValueError("training IDs must be an immutable set")
        actual_ids = frozenset((u.episode_id, u.sample_id) for u in runtime._inbox.applied_updates)
        actor = self._actor.serving_snapshot()
        report = self._evaluator.evaluate(
            self._policy,
            VersionedState(actor.actor_version, actor.model_state),
            candidate,
            new,
            old,
            now=now,
            training_ids=training_ids | actual_ids,
        )
        if (
            report.decision != decide_promotion(self._policy, report.evidence)
            or not report.decision.accepted
        ):
            raise ValueError("candidate rejected: " + ", ".join(report.decision.reasons))
        if _digest(self._digest, actor.model_state) != report.actor_snapshot_digest:
            raise ValueError("controller and evaluator actor state digests disagree")
        if _digest(self._digest, candidate.state) != report.candidate_snapshot_digest:
            raise ValueError("controller and evaluator candidate state digests disagree")
        bundle = self._build_bundle(runtime, candidate, actor, report, metadata)
        ticket = PreparedPromotion(actor.generation, deepcopy(report))
        self._pending[ticket] = _Pending(
            runtime, deepcopy(report), bundle, actor.generation, self._build
        )
        return ticket

    def _build_bundle(
        self,
        runtime: ActorShadowRuntime[Features, Targets, Prediction, State],
        candidate: CandidateState[State],
        actor: ServingSnapshot[State, Prediction],
        report: PromotionGuardReport,
        metadata: dict[str, object] | None,
    ) -> _Bundle[Features, Targets, Prediction, State]:
        lifecycle = self._actor._payload_registry._lifecycle
        if lifecycle is not None:
            lifecycle._require_access()
        context = actor.metadata if metadata is None else metadata
        if type(context) is not dict or any(type(key) is not str for key in context):
            raise ValueError("serving metadata must be a dictionary with string keys")
        owned_context = deepcopy(context)
        model = self._build(deepcopy(candidate.state))
        if lifecycle is not None:
            lifecycle._require_access()
        if (
            self._actor._owns(model)
            or runtime._owns_candidate(model)
            or any(p.bundle.learner is model for p in self._pending.values())
        ):
            raise ValueError("prepared native model must be independently owned")
        model.restore_state(deepcopy(candidate.state))
        if lifecycle is not None:
            lifecycle._require_access()
        if _digest(self._digest, model.snapshot_state()) != report.candidate_snapshot_digest:
            raise ValueError("prepared model restore does not reproduce complete candidate state")
        if lifecycle is not None:
            lifecycle._require_access()
        return _Bundle(model, candidate.learner_version, actor.configuration, {}, owned_context, 0)

    def commit(
        self,
        runtime: ActorShadowRuntime[Features, Targets, Prediction, State],
        ticket: PreparedPromotion,
    ) -> PromotionReceipt:
        with self._exclusive():
            pending = self._require_ticket(ticket)
            if pending.runtime is not runtime:
                raise ValueError("ticket belongs to a different candidate runtime")
            self._require_runtime(runtime)

            def commit_owned(candidate: CandidateState[State]) -> PromotionReceipt:
                self._check_candidate(candidate, pending.report)
                if (
                    _digest(self._digest, pending.bundle.learner.snapshot_state())
                    != pending.report.candidate_snapshot_digest
                ):
                    raise ValueError("prepared model changed since guard evaluation")
                if self._actor._payload_registry._lifecycle is not None:
                    self._actor._payload_registry._lifecycle._require_access()
                return self._actor._commit(
                    pending.bundle,
                    generation=ticket.actor_generation,
                    actor_digest=pending.report.actor_snapshot_digest,
                    state_digest=self._digest,
                )

            receipt = runtime.with_candidate_snapshot(commit_owned)
            del self._pending[ticket]
            self._latest = (receipt, deepcopy(receipt))
            return receipt

    def _require_ticket(
        self, ticket: PreparedPromotion
    ) -> _Pending[Features, Targets, Prediction, State]:
        if type(ticket) is not PreparedPromotion or ticket not in self._pending:
            raise ValueError("promotion ticket is foreign, forged, discarded or consumed")
        pending = self._pending[ticket]
        if self._build is not pending.builder or self._evaluator._build is not pending.builder:
            raise ValueError("issued report native builder configuration changed")
        if ticket.actor_generation != pending.actor_generation:
            raise ValueError("issued ticket actor generation changed")
        if ticket.report != pending.report or pending.report.policy != self._policy:
            raise ValueError("issued promotion report or configured policy changed")
        if not decide_promotion(self._policy, pending.report.evidence).accepted:
            raise ValueError("issued report no longer satisfies configured guards")
        return pending

    def _check_candidate(
        self, candidate: CandidateState[State], report: PromotionGuardReport
    ) -> None:
        revision = (
            candidate.wake_updates,
            candidate.consolidation_attempts,
            candidate.consolidations_completed,
        )
        if (
            candidate.actor_version != report.actor_version
            or candidate.learner_version != report.learner_version
            or revision != report.candidate_revision
            or _digest(self._digest, candidate.state) != report.candidate_snapshot_digest
        ):
            raise ValueError(
                "candidate version/revision/model state changed after guard evaluation"
            )

    def discard(self, ticket: PreparedPromotion) -> None:
        with self._exclusive():
            self._require_ticket(ticket)
            del self._pending[ticket]

    def rollback(self, receipt: PromotionReceipt) -> int:
        with self._exclusive():
            if self._latest is None or receipt is not self._latest[0]:
                raise ValueError("rollback receipt is foreign, stale or consumed")
            original = self._latest[1]
            if (receipt.generation, receipt.previous_version, receipt.actor_version) != (
                original.generation,
                original.previous_version,
                original.actor_version,
            ):
                raise ValueError("rollback receipt was modified")
            generation = self._actor._rollback(receipt)
            self._latest = None
            return generation
