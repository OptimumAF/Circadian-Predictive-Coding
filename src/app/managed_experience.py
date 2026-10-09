"""Permanent opt-in consent admission around a trusted local candidate owner.

Inputs: metadata declarations and opaque events. Outputs: bounded metadata and
existing inbox receipts. An optional original-authority coordinator supplies
conservative cleanup; this admission module itself provides no erasure, hard
byte quota, unlearning, portable authority or provenance attestation.
"""

from contextlib import contextmanager
from threading import Lock
from typing import Any, Generic, Iterator, TypeVar

from src.app.actor_shadow import ActorShadowRuntime
from src.app.resource_sharing import ResourceSharedRuntime
from src.app.payload_ownership import PayloadOwnershipBusy
from src.core.data_lifecycle import LifecycleDeclaration, LifecycleLimits
from src.core.native_update_origin import NativeUpdateObserver
from src.core.resource_sharing import TrainingPoll
from src.core.experience import Experience, LabelArrival, SampleKey, require_identifier

Features = TypeVar("Features")
Targets = TypeVar("Targets")
Prediction = TypeVar("Prediction")
State = TypeVar("State")


class ManagedExperienceOwner(Generic[Features, Targets, Prediction, State]):
    def __init__(
        self,
        shared: ResourceSharedRuntime[Features, Targets, Prediction, State],
        *,
        limits: LifecycleLimits,
    ) -> None:
        if type(shared) is not ResourceSharedRuntime or type(limits) is not LifecycleLimits:
            raise ValueError("managed admission requires a shared runtime and typed limits")
        self._shared, self._limits = shared, limits
        self._catalog: dict[SampleKey, LifecycleDeclaration] = {}
        self._opted_out: set[str] = set()
        self._issuing: object | None = None
        self._gate = Lock()
        self._lifecycle: Any = None
        self._revoked_keys: set[SampleKey] = set()
        self._declaration_ticks: dict[SampleKey, int] = {}
        self._declaration_seconds: dict[SampleKey, float] = {}
        runtime = shared._runtime
        if type(runtime) is not ActorShadowRuntime:
            raise ValueError("managed admission requires an exact supported candidate owner")
        with runtime._exclusive():
            runtime._require_open()
            inbox = runtime._inbox
            if (
                inbox._experiences
                or inbox._labels
                or inbox._applied
                or runtime._attempted_ids
                or runtime._budget.updates_completed
                or inbox._registration_guard is not None
                or inbox._training_guard is not None
            ):
                raise ValueError("managed admission requires a fresh uninstalled inbox")
            inbox._registration_guard = self._require_registration
            inbox._training_guard = self._eligible
            runtime._revision += 1

    @contextmanager
    def _operation(self) -> Iterator[None]:
        acquired_gate = self._gate
        if not acquired_gate.acquire(blocking=False):
            raise PayloadOwnershipBusy("managed admission is busy; reentrance is unsupported")
        try:
            yield
        finally:
            acquired_gate.release()

    @property
    def declarations(self) -> tuple[LifecycleDeclaration, ...]:
        with self._operation():
            return tuple(self._catalog.values())

    def declare(self, declaration: LifecycleDeclaration) -> None:
        with self._operation():
            runtime = self._shared._runtime
            with runtime._exclusive():
                self._declare(runtime, declaration)

    def _declare(self, runtime, declaration: LifecycleDeclaration) -> None:
        runtime._require_open()
        self._limits.require_supported(declaration)
        if declaration.provenance.subject_id in self._opted_out:
            raise ValueError("subject has opted out")
        if declaration.key in self._catalog:
            raise ValueError("duplicate lifecycle declaration")
        # Why: lifetime grant counts never renew when consent is revoked.
        if len(self._catalog) >= min(
            self._limits.max_approved_records, self._limits.max_replay_records
        ):
            raise ValueError("lifetime approved or replay record quota exhausted")
        tick, seconds = (
            self._lifecycle._declaration_anchor() if self._lifecycle is not None else (None, None)
        )
        self._catalog[declaration.key] = declaration
        if self._lifecycle is not None:
            assert tick is not None
            self._declaration_ticks[declaration.key] = tick
            if seconds is not None:
                self._declaration_seconds[declaration.key] = seconds
        runtime._revision += 1

    def opt_out(self, subject_id: str) -> None:
        require_identifier(subject_id, "subject_id")
        if self._lifecycle is not None:
            self._lifecycle.opt_out(subject_id)
            return
        with self._operation():
            runtime = self._shared._runtime
            with runtime._exclusive():
                runtime._require_open()
                if not any(d.provenance.subject_id == subject_id for d in self._catalog.values()):
                    raise ValueError("unknown subject cannot grow the bounded opt-out catalog")
                self._opted_out.add(subject_id)
                runtime._revision += 1

    def _require_live(self, key: SampleKey) -> LifecycleDeclaration:
        if key not in self._catalog:
            raise ValueError("unknown lifecycle declaration")
        if key in self._revoked_keys:
            raise ValueError("lifecycle identity has been revoked")
        declaration = self._catalog[key]
        self._limits.require_supported(declaration)
        if declaration.provenance.subject_id in self._opted_out:
            raise ValueError("subject has opted out")
        if self._lifecycle is not None:
            self._lifecycle._require_live(key)
        return declaration

    def _require_registration(self, event: object) -> None:
        if event is not self._issuing:
            raise ValueError("registration requires the original managed admission owner")
        if type(event) is Experience:
            source = event
            self._require_live(source.key)
            if (
                source.role != "train"
                or not source.permissions.training
                or not source.permissions.replay
            ):
                raise ValueError("source requires training and replay permission")
        elif type(event) is LabelArrival:
            self._require_live(event.key)
            if event.role != "train":
                raise ValueError("label requires train role")
        else:
            raise ValueError("managed registration requires an exact source or label")
        if event.model_version != self._shared._runtime._base_actor_version:
            raise ValueError("event actor version differs from the original candidate base")
        if self._lifecycle is not None:
            self._lifecycle._admit_event(event)

    def _eligible(self, source: Experience[Features], label: LabelArrival[Targets]) -> bool:
        declaration = self._catalog.get(source.key)
        return (
            declaration is not None
            and source.key not in self._revoked_keys
            and declaration.provenance.subject_id not in self._opted_out
            and declaration.consent.training
            and declaration.consent.replay
            and declaration.retention == "replay"
            and source.key == label.key
            and source.role == label.role == "train"
            and source.permissions.training
            and source.permissions.replay
        )

    def record_experience(self, source: Experience[Features]) -> None:
        with self._operation():
            self._issuing = source
            try:
                self._shared._runtime.record_experience(source)
            finally:
                self._issuing = None

    def record_label(self, label: LabelArrival[Targets]) -> None:
        with self._operation():
            self._issuing = label
            try:
                self._shared._runtime.record_label(label)
            finally:
                self._issuing = None

    def train_ready(
        self, *, native_observer: NativeUpdateObserver[Features, Targets] | None = None
    ) -> TrainingPoll:
        """Observe native inputs under the original managed and sharing gates."""
        with self._operation():
            return self._shared.train_ready(native_observer=native_observer)
