"""Cooperative serving priority over new candidate native updates.

The gate bounds requests and admitted attempts. It never preempts an in-flight
native call, creates background work, measures p50/p95 or claims a hard RSS cap.
All supported shared predict/serve/train calls must pass through the wrapper.
"""

from contextlib import contextmanager
from copy import deepcopy
from threading import Lock
from typing import Callable, Generic, Iterator, TypeVar

from src.app.actor_shadow import ActorShadowRuntime
from src.app.serving_promotion import PromotableActor
from src.app.payload_ownership import PayloadOwnershipBusy
from src.core.actor_ports import VersionedPrediction
from src.core.native_update_origin import NativeUpdateObserver
from src.core.resource_sharing import (
    DeferReason,
    SharingLimits,
    SharingSnapshot,
    TrainingAdmission,
    TrainingPoll,
)
from src.core.serving_ports import ServingPrediction

Features = TypeVar("Features")
Targets = TypeVar("Targets")
Prediction = TypeVar("Prediction")
State = TypeVar("State")


class ServingPriorityGate:
    def __init__(self, limits: SharingLimits, *, resource_available: Callable[[], bool]) -> None:
        if type(limits) is not SharingLimits or not callable(resource_available):
            raise ValueError("sharing gate requires declared limits and callable resource probe")
        self._limits, self._resource = deepcopy(limits), resource_available
        self._serving, self._admitted = 0, 0
        self._training, self._paused = False, False
        self._checkpointing = False
        self._retention_hold: object | None = None
        self._deferrals: dict[DeferReason, int] = {}
        self._gate = Lock()

    def snapshot(self) -> SharingSnapshot:
        with self._gate:
            return SharingSnapshot(
                deepcopy(self._limits),
                self._serving,
                self._training,
                self._paused or self._retention_hold is not None,
                self._admitted,
                tuple(sorted(self._deferrals.items())),
            )

    def pause(self) -> None:
        with self._gate:
            self._paused = True

    def resume(self) -> None:
        with self._gate:
            if self._checkpointing:
                raise ValueError("cannot resume during checkpoint preparation")
            self._paused = False

    def _hold_retention(self, token: object) -> None:
        with self._gate:
            if self._retention_hold is not None and self._retention_hold is not token:
                raise PayloadOwnershipBusy("sharing retention hold belongs to another owner")
            self._retention_hold = token

    def _release_retention(self, token: object) -> None:
        with self._gate:
            if self._retention_hold is not token:
                raise ValueError("foreign retention hold")
            self._retention_hold = None

    @contextmanager
    def _checkpoint_lease(self) -> Iterator[None]:
        """Require paused quiescence; keep new serving available during restore."""
        acquired_gate = self._gate
        with acquired_gate:
            if self._checkpointing or self._training or self._serving:
                raise PayloadOwnershipBusy("sharing gate is busy during checkpoint handoff")
            if not self._paused and self._retention_hold is None:
                raise ValueError("checkpoint handoff requires paused training")
            self._checkpointing = True
        try:
            yield
        finally:
            with acquired_gate:
                self._checkpointing = False

    @contextmanager
    def serving(self) -> Iterator[None]:
        with self._gate:
            if self._serving >= self._limits.max_serving_requests:
                raise ValueError("serving request capacity exceeded; defer the request")
            self._serving += 1
        try:
            yield
        finally:
            with self._gate:
                self._serving -= 1

    def _reason(self) -> DeferReason | None:
        if self._paused or self._retention_hold is not None:
            return "paused"
        if self._serving:
            return "serving_active"
        if self._training:
            return "training_active"
        if self._admitted >= self._limits.max_admitted_updates:
            return "work_quota"
        return None

    def _deny(self, reason: DeferReason) -> TrainingAdmission:
        self._deferrals[reason] = self._deferrals.get(reason, 0) + 1
        return TrainingAdmission(False, reason)

    def _admit(self) -> TrainingAdmission:
        with self._gate:
            reason = self._reason()
            if reason is not None:
                return self._deny(reason)
        # Why: never hold the gate while an external resource probe runs. A
        # serving arrival/pause during that probe wins the second check below.
        available = self._resource()
        if type(available) is not bool:
            raise ValueError("resource probe must return an exact boolean")
        with self._gate:
            reason = self._reason()
            if reason is None and not available:
                reason = "resource_unavailable"
            if reason is not None:
                return self._deny(reason)
            self._training = True
            self._admitted += 1  # charge attempts before callbacks/native work
            return TrainingAdmission(True, None)

    @contextmanager
    def training(self) -> Iterator[TrainingAdmission]:
        decision = self._admit()
        try:
            yield decision
        finally:
            if decision.allowed:
                with self._gate:
                    self._training = False


class ResourceSharedRuntime(Generic[Features, Targets, Prediction, State]):
    def __init__(
        self,
        runtime: ActorShadowRuntime[Features, Targets, Prediction, State],
        gate: ServingPriorityGate,
    ) -> None:
        if not isinstance(runtime, ActorShadowRuntime) or type(gate) is not ServingPriorityGate:
            raise ValueError("resource sharing requires an actor/shadow runtime and priority gate")
        self._runtime, self._sharing = runtime, gate

    def predict(self, features: Features) -> VersionedPrediction[Prediction]:
        with self._sharing.serving():
            return self._runtime.actor.predict(features)

    def serve(self, features: Features, *, now: int) -> ServingPrediction[Prediction]:
        actor = self._runtime.actor
        if not isinstance(actor, PromotableActor):
            raise ValueError("cached shared serving requires a PromotableActor")
        with self._sharing.serving():
            return actor.serve(features, now=now)

    def train_ready(
        self, *, native_observer: NativeUpdateObserver[Features, Targets] | None = None
    ) -> TrainingPoll:
        deferred: list[DeferReason] = []

        @contextmanager
        def admission() -> Iterator[bool]:
            with self._sharing.training() as decision:
                if decision.reason is not None:
                    deferred.append(decision.reason)
                yield decision.allowed

        updates = self._runtime.train_ready(
            max_updates=self._sharing.snapshot().limits.max_updates_per_poll,
            before_each_update=admission,
            native_observer=native_observer,
        )
        return TrainingPoll(updates, deferred[-1] if deferred else None)


from src.app.expiry_release_proof import pin_release_methods as _pin_release_methods

_EXPIRY_RELEASE_PINS = _pin_release_methods(ServingPriorityGate, ("_checkpoint_lease",))
