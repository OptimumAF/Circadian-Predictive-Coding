"""Windows time/RSS/registered-process observations under a live epoch anchor.

No coordinator, transactional fence, native payload IO or automatic retry. An
ended/unknown anchor or failed observation terminally disables this observer.
"""

from threading import Lock

from src.core.recovery_observation import (
    RecoveryHostObservation,
    RecoveryProcessProbe,
    anchored_clock_epoch,
)
from src.infra.windows_process_handles import WindowsProcessHandle


class WindowsRecoveryObserver:
    def __init__(self, anchor: object, *, peak_rss_bytes: int = 0) -> None:
        if type(peak_rss_bytes) is not int or not 0 <= peak_rss_bytes < 2**63:
            raise ValueError("observer RSS peak seed requires a bounded exact nonnegative integer")
        if type(anchor) is not WindowsProcessHandle:
            raise ValueError("Windows recovery requires a registered process anchor")
        if anchor.is_ended():
            raise ValueError("Windows recovery anchor has ended")
        self._anchor, self._api = anchor, anchor._api
        self._epoch = anchored_clock_epoch(anchor.identity)
        with WindowsProcessHandle.pin(self._api.current_pid(), api=self._api) as current:
            self._identity = current.identity
        self._last_ns, self._peak_rss = 0, peak_rss_bytes
        self._failed = False
        self._gate = Lock()

    def _require_anchor(self) -> None:
        if self._anchor.is_ended():
            raise ValueError("recovery anchor has ended")
        if anchored_clock_epoch(self._anchor.identity) != self._epoch:
            raise ValueError("recovery anchor identity/epoch changed")

    def observe(
        self, previous_owner: RecoveryProcessProbe | None = None
    ) -> RecoveryHostObservation:
        if not self._gate.acquire(blocking=False):
            raise ValueError("recovery observer is busy; nested/concurrent observation refused")
        try:
            if self._failed:
                raise ValueError("recovery observer previously failed")
            return self._observe(previous_owner)
        except BaseException:
            self._failed = True
            raise
        finally:
            self._gate.release()

    def _observe(self, previous_owner: RecoveryProcessProbe | None) -> RecoveryHostObservation:
        self._require_anchor()
        if previous_owner is not None and (
            type(previous_owner) is not WindowsProcessHandle
            or previous_owner._api is not self._api
            or previous_owner.identity == self._anchor.identity
        ):
            raise ValueError("previous owner requires an independent registered process handle")
        now = self._api.interrupt_ns()
        if type(now) is not int or not 0 < now < 2**63 or now < self._last_ns:
            raise ValueError("recovery clock is invalid or went backwards")
        self._last_ns = now
        rss = self._api.rss_bytes()
        if type(rss) is not int or not 0 < rss < 2**63:
            raise ValueError("recovery RSS observation is unavailable or invalid")
        self._peak_rss = max(self._peak_rss, rss)
        ended = None if previous_owner is None else previous_owner.is_ended()
        self._require_anchor()  # Fail if anchor ended during a probe.
        return RecoveryHostObservation(
            self._epoch,
            now,
            rss,
            self._peak_rss,
            self._identity,
            None if previous_owner is None else previous_owner.identity,
            ended,
        )
