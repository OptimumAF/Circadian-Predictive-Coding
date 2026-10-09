"""One monotonic original-authority allowance for attempted owned payload copies.

Reserve before copying. Failures, disposal and cleanup do not refund capacity;
this conservative accounting avoids reclaiming bytes while another owner retains
them. No payload references, GC callbacks, timing, native operations or IO.
"""

from contextlib import contextmanager
from threading import Lock, get_ident
from typing import Callable, Iterator
from src.app.payload_ownership import lease_payload_lock
from src.core.payload_bytes import PayloadByteSnapshot, PayloadCopyLimits
from src.core.experience import require_tick


class PayloadCopyBudget:
    def __init__(self, limits: PayloadCopyLimits) -> None:
        if type(limits) is not PayloadCopyLimits:
            raise ValueError("copy budget requires typed limits")
        self._limits, self._charged, self._gate = limits, 0, Lock()

    def reserve(self, size: int) -> None:
        require_tick(size, "owned payload copy bytes")
        with self._lease() as reserve:
            reserve(size)

    @contextmanager
    def _lease(self) -> Iterator[Callable[[int], None]]:
        """Internal source coordinator: reserve without reacquiring this gate.

        Why: source sizing and admission must share one original owner interval.
        The capability is local to that interval and thread; it is not portable
        authority. Attempted copies remain charged even if the body raises.
        """
        gate, limits, thread = self._gate, self._limits, get_ident()
        active = False

        def reserve(size: int) -> None:
            if not active or get_ident() != thread:
                raise ValueError("payload reservation is outside its original lease/thread")
            if self._gate is not gate or self._limits is not limits:
                raise ValueError("payload reservation original gate/policy changed")
            require_tick(size, "owned payload copy bytes")
            if self._charged + size > self._limits.max_lifetime_owned_bytes:
                raise ValueError("original lifetime owned payload byte allowance exhausted")
            self._charged += size

        with lease_payload_lock(gate, "payload byte budget"):
            active = True
            try:
                yield reserve
            finally:
                active = False

    def snapshot(self, observed: int) -> PayloadByteSnapshot:
        with lease_payload_lock(self._gate, "payload byte budget"):
            return PayloadByteSnapshot(self._limits, self._charged, observed)
