"""Bounded opt-in automatic expiry under original local ownership authority.

No new clock authority or allowance renewal. The worker waits interruptibly and
tries quiescent cleanup;it cannot preempt native callbacks or promise real-time
physical erasure. Stop/fault/exhaustion closes raw admission;unfinished cleanup
retains its independent sharing hold. Observations never retain tracebacks.
"""

from logging import getLogger
from threading import Event, Lock, RLock, Thread, current_thread
from time import monotonic
from src.app.managed_data_lifecycle import ManagedDataLifecycle
from src.app.payload_ownership import PayloadOwnershipBusy, lease_payload_lock
from src.core.retention_driver import (
    RetentionDriverLimits,
    RetentionDriverSnapshot,
    RetentionPollResult,
    RetentionDriverState,
    require_seconds,
)

logger = getLogger(__name__)


class RetentionExpiryDriver:
    def __init__(self, lifecycle: ManagedDataLifecycle, *, limits: RetentionDriverLimits) -> None:
        if (
            type(lifecycle) is not ManagedDataLifecycle
            or type(limits) is not RetentionDriverLimits
            or lifecycle._policy.max_retention_seconds is None
        ):
            raise ValueError(
                "retention driver requires original lifecycle and declared elapsed policy/limits"
            )
        with lifecycle._owner._operation():
            if lifecycle._retention_driver is not None:
                raise ValueError("original retention driver allowance cannot be renewed")
            lifecycle._tick()
            lifecycle._elapsed()
            self._lifecycle, self._limits = lifecycle, limits
            self._state: RetentionDriverState = "ready"
            self._polls = self._purges = self._cleanup_attempts = 0
            self._pending = False
            self._error_type: str | None = None
            self._token = object()
            self._held = False
            self._operation_gate = Lock()
            # Short state transitions only: callbacks and joins must not block stop.
            self._state_gate = RLock()
            self._stop = Event()
            self._wake = Event()
            self._purged = Event()
            self._thread: Thread | None = None
            self._created_at = monotonic()
            lifecycle._retention_driver = self

    def snapshot(self) -> RetentionDriverSnapshot:
        with self._state_gate:
            return RetentionDriverSnapshot(
                self._limits,
                self._state,
                self._polls,
                self._purges,
                self._cleanup_attempts,
                self._thread is not None and self._thread.is_alive(),
                self._pending,
                self._error_type,
            )

    def _hold(self) -> None:
        with self._state_gate:
            self._lifecycle._sharing._hold_retention(self._token)
            self._held = True
            self._pending = True

    def _release(self) -> None:
        with self._state_gate:
            if self._held:
                self._lifecycle._sharing._release_retention(self._token)
            self._held = False
            self._pending = False

    def _failure(self, error: BaseException) -> RetentionPollResult:
        with self._state_gate:
            self._state = "failed"
            self._error_type = type(error).__name__
            self._lifecycle._retention_fault = True
            try:
                self._hold()
            except PayloadOwnershipBusy:
                self._pending = True
        logger.error(
            "retention_cleanup_failed", extra={"error_type": self._error_type, "polls": self._polls}
        )
        return RetentionPollResult("failed")

    def _attempt(self, *, all_data: bool) -> RetentionPollResult:
        with self._state_gate:
            if self._cleanup_attempts >= self._limits.max_polls + 2:
                return self._failure(ValueError("retention cleanup attempt allowance exhausted"))
            self._cleanup_attempts += 1
        try:
            self._hold()
            report = self._lifecycle._expire_all() if all_data else self._lifecycle.expire()
        except PayloadOwnershipBusy:
            return RetentionPollResult("busy")
        except BaseException as error:
            return self._failure(error)
        with self._state_gate:
            self._purges += 1
            self._release()
            self._purged.set()
        logger.info(
            "retention_cleanup_complete",
            extra={
                "polls": self._polls,
                "purges": self._purges,
                "inboxes_cleared": report.inboxes_cleared,
            },
        )
        return RetentionPollResult("purged", report)

    def poll_once(self) -> RetentionPollResult:
        with lease_payload_lock(self._operation_gate, "retention driver"):
            with self._state_gate:
                if self._state not in ("ready", "running"):
                    raise ValueError("retention driver terminal;poll allowance cannot renew")
                self._polls += 1
                exhausted = (
                    self._polls >= self._limits.max_polls
                    or monotonic() - self._created_at >= self._limits.max_run_seconds
                )
                if exhausted:
                    self._state = "exhausted"
            if exhausted:
                return self._attempt(all_data=True)
            try:
                keys, auxiliary = self._lifecycle._due()
            except PayloadOwnershipBusy:
                self._hold()
                return RetentionPollResult("busy")
            except BaseException as error:
                return self._failure(error)
            with self._state_gate:
                terminal = self._state in ("stopping", "stopped", "exhausted", "failed")
            if terminal:
                return self._attempt(all_data=True)
            if keys or auxiliary or self._lifecycle._failed:
                return self._attempt(all_data=True if self._lifecycle._failed else False)
            self._release()
            return RetentionPollResult("idle")

    def start(self) -> None:
        with lease_payload_lock(self._operation_gate, "retention driver"), self._state_gate:
            if self._state != "ready" or self._thread is not None:
                raise ValueError("retention driver cannot restart or renew allowance")
            self._state = "running"
            self._thread = Thread(target=self._run, name="circadian-retention", daemon=False)
            self._thread.start()

    def wake(self) -> None:
        with self._state_gate:
            self._wake.set()

    def wait_for_purge(self, timeout_seconds: float) -> bool:
        require_seconds(timeout_seconds, "purge wait")
        if timeout_seconds > self._limits.join_timeout_seconds:
            raise ValueError("purge wait exceeds declared driver join bound")
        return self._purged.wait(timeout_seconds)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except PayloadOwnershipBusy:
                pass
            except BaseException as error:
                self._failure(error)
                return
            with self._state_gate:
                if self._state in ("failed", "exhausted"):
                    return
                remaining = self._limits.max_run_seconds - (monotonic() - self._created_at)
                if remaining <= 0:
                    self._state = "exhausted"
                    self._hold()
                    return
            self._wake.wait(max(0.0, min(self._limits.poll_interval_seconds, remaining)))
            with self._state_gate:
                self._wake.clear()
        try:
            with lease_payload_lock(self._operation_gate, "retention driver"):
                with self._state_gate:
                    self._state = "stopping"
                self._attempt(all_data=True)
                with self._state_gate:
                    if self._state != "failed":
                        self._state = "stopped"
        except PayloadOwnershipBusy:
            with self._state_gate:
                self._hold()
                self._state = "stopped"

    def stop(self) -> bool:
        with self._state_gate:
            if self._state in ("ready", "running"):
                self._state = "stopping"
            self._stop.set()
            self._wake.set()
            thread = self._thread
            stopping = self._state == "stopping"
        if thread is not None:
            if thread is current_thread():
                raise ValueError("retention worker cannot join itself")
            thread.join(self._limits.join_timeout_seconds)
            if thread.is_alive():
                return False
        elif stopping:
            with lease_payload_lock(self._operation_gate, "retention driver"):
                with self._state_gate:
                    self._state = "stopping"
                self._attempt(all_data=True)
                with self._state_gate:
                    if self._state != "failed":
                        self._state = "stopped"
        with self._state_gate:
            return not self._pending

    def finish_cleanup(self) -> RetentionPollResult:
        with lease_payload_lock(self._operation_gate, "retention driver"):
            with self._state_gate:
                if self._thread is not None and self._thread.is_alive():
                    raise ValueError("stop/join retention worker before manual terminal cleanup")
                if not self._pending:
                    raise ValueError("no unfinished retention cleanup")
            return self._attempt(all_data=True)
