"""Weak holder enrollment and nonblocking quiescence for future data cleanup.

Owns only metadata/weak references. It neither mutates retained data nor proves
arbitrary object graph/caller/callback ownership, byte limits or deletion.
"""

from contextlib import contextmanager, ExitStack
from threading import Lock
from typing import Any, Iterator
from weakref import ref, ReferenceType

from src.core.payload_ownership import (
    OwnedPayloadGroup,
    OwnershipKind,
    PayloadHolder,
    PayloadHolderMetadata,
    PayloadOwnershipLimits,
    PayloadOwnershipSnapshot,
    PayloadReferences,
    require_ownership_kind,
)


class PayloadOwnershipBusy(ValueError):
    """A supported operation lease could not acquire quiescent ownership."""


@contextmanager
def lease_payload_lock(lock, name: str) -> Iterator[None]:
    if not lock.acquire(blocking=False):
        raise PayloadOwnershipBusy(f"{name} payload owner is busy")
    try:
        yield
    finally:
        lock.release()


class PayloadOwnershipRegistry:
    def __init__(self) -> None:
        self._holders: dict[int, tuple[OwnershipKind, ReferenceType[PayloadHolder]]] = {}
        self._total = 0
        self._limits: PayloadOwnershipLimits | None = None
        self._gate = Lock()
        self._lifecycle: Any = None

    def _live(self) -> list[tuple[int, OwnershipKind, PayloadHolder]]:
        live = []
        for number, (kind, reference) in list(self._holders.items()):
            owner = reference()
            if owner is None:
                del self._holders[number]
            else:
                live.append((number, kind, owner))
        return live

    def configure(self, limits: PayloadOwnershipLimits) -> None:
        if type(limits) is not PayloadOwnershipLimits:
            raise ValueError("ownership registry requires typed limits")
        with lease_payload_lock(self._gate, "ownership registry"):
            if self._limits is not None:
                raise ValueError("ownership registry limits are already configured")
            if len(self._live()) > limits.max_live_holders:
                raise ValueError("current live holders exceed ownership limits")
            if self._total > limits.max_lifetime_enrollments:
                raise ValueError("consumed lifetime enrollments exceed ownership limits")
            self._limits = limits

    def enroll(self, kind: OwnershipKind, holder: PayloadHolder) -> None:
        require_ownership_kind(kind)
        if type(getattr(holder, "_payload_ready", None)) is not bool or not all(
            callable(getattr(holder, name, None))
            for name in ("_payload_exclusive", "_payload_references")
        ):
            raise ValueError("enrollment requires a complete payload holder port")
        with lease_payload_lock(self._gate, "ownership registry"):
            live = self._live()
            if any(owner is holder for _, _, owner in live):
                raise ValueError("payload holder is already enrolled")
            if self._lifecycle is not None:
                self._lifecycle._require_holder(kind, holder)
            if self._limits is not None:
                if len(live) >= self._limits.max_live_holders:
                    raise ValueError("live payload holder quota exhausted")
                if self._total >= self._limits.max_lifetime_enrollments:
                    raise ValueError("lifetime payload enrollment quota exhausted")
            try:
                reference = ref(holder)
            except TypeError as error:
                raise ValueError("payload holder must support weak ownership references") from error
            self._total += 1
            self._holders[self._total] = kind, reference

    def snapshot(self) -> PayloadOwnershipSnapshot:
        with lease_payload_lock(self._gate, "ownership registry"):
            return PayloadOwnershipSnapshot(
                tuple(
                    PayloadHolderMetadata(n, kind, owner._payload_ready)
                    for n, kind, owner in self._live()
                ),
                self._total,
                self._limits,
            )

    @contextmanager
    def _lease(self) -> Iterator[tuple[OwnedPayloadGroup, ...]]:
        """Internal trusted coordinator: freeze all holders before enumeration.

        Why: try every lock without waiting. This safely coexists with existing
        controller->candidate->actor orders and avoids partial cleanup on refusal.
        Strong references live only in this lease; metadata snapshots retain none.
        """
        primary: BaseException | None = None
        try:
            with lease_payload_lock(self._gate, "ownership registry"), ExitStack() as stack:
                try:
                    live = self._live()
                    for _, _, owner in live:
                        if type(owner._payload_ready) is not bool or not owner._payload_ready:
                            raise ValueError("payload holder initialization is incomplete")
                        try:
                            stack.enter_context(owner._payload_exclusive())
                        except ValueError as error:
                            raise PayloadOwnershipBusy(str(error)) from error
                    groups = []
                    for number, kind, owner in live:
                        references = owner._payload_references()
                        if type(references) is not PayloadReferences:
                            raise ValueError("holder enumeration requires exact PayloadReferences")
                        groups.append(
                            OwnedPayloadGroup(PayloadHolderMetadata(number, kind, True), references)
                        )
                    yield tuple(groups)
                except BaseException as error:
                    primary = error
                    raise
        except BaseException:
            if primary is None:
                raise
        # Preserve the first visible entry/body error across unsafe old exits.
        if primary is not None:
            raise primary


from src.app.expiry_release_proof import pin_release_methods as _pin_release_methods

_EXPIRY_RELEASE_PINS = _pin_release_methods(PayloadOwnershipRegistry, ("_lease", "_live"))
_EXPIRY_LOCK_HELPER = lease_payload_lock
_EXPIRY_LOCK_CODE = lease_payload_lock.__code__
_EXPIRY_LOCK_GENERATOR: Any = lease_payload_lock.__dict__["__wrapped__"]
_EXPIRY_LOCK_GENERATOR_CODE = _EXPIRY_LOCK_GENERATOR.__code__
_EXPIRY_EXIT_STACK = ExitStack
