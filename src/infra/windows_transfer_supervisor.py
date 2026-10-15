"""Retained physical Windows supervision for one explicit coordinator transfer.

The caller supplies independently completed native-only authority and original
handles registered while alive. An owned trusted child observes its own worker;
the supervisor brackets that response with genuine local observations. Why keep
both facts: a remote epoch tag cannot account for later supervisor time or RSS.
This adapter owns registrations, never processes. No checkpoint restore, child
launch, automatic retry, disk bootstrap or authentication is implemented here.
"""

from copy import deepcopy
from dataclasses import asdict
from threading import Lock
from typing import Any

from src.core.coordinator_transfer import (
    CoordinatorTransfer,
    CoordinatorTransferFailure,
    CoordinatorTransferObservation,
    SupervisorRecoveryWitness,
    _authority,
    plan_coordinator_transfer,
    validate_coordinator_transfer,
)
from src.core.recovery_authority import AuthorityRecord
from src.core.recovery_admission import _identifier
from src.core.recovery_observation import (
    RecoveryHostObservation,
    RecoveryProcessIdentity,
    anchored_clock_epoch,
)
from src.infra.bounded_json_child import BoundedJsonChild
from src.infra.sqlite_recovery_journal import SqliteRecoveryJournal
from src.infra.windows_process_handles import WindowsProcessHandle
from src.infra.windows_recovery_observer import WindowsRecoveryObserver


def _object(value: object, names: set[str], context: str) -> dict[str, Any]:
    if type(value) is not dict or value.keys() != names:
        raise ValueError(f"transfer reply requires exact {context} fields")
    return value


def _process(value: object) -> RecoveryProcessIdentity:
    record = _object(value, {"pid", "created_filetime"}, "process identity")
    return RecoveryProcessIdentity(record["pid"], record["created_filetime"])


def _reply(value: object) -> RecoveryHostObservation:
    envelope = _object(value, {"event", "observation"}, "envelope")
    if type(envelope["event"]) is not str or envelope["event"] != "observed":
        raise ValueError("transfer reply requires the observed event")
    record = _object(
        envelope["observation"],
        {
            "clock_epoch",
            "now_ns",
            "rss_bytes",
            "peak_rss_bytes",
            "observer",
            "previous_owner",
            "previous_owner_ended",
        },
        "host observation",
    )
    return RecoveryHostObservation(
        record["clock_epoch"],
        record["now_ns"],
        record["rss_bytes"],
        record["peak_rss_bytes"],
        _process(record["observer"]),
        _process(record["previous_owner"]),
        record["previous_owner_ended"],
    )


class WindowsTransferSupervisor:
    """Single-owner retained supervisor; all failed operations disable live transfer."""

    def __init__(
        self,
        known_authority: AuthorityRecord,
        supervisor: object,
        old_anchor: object,
        old_worker: object,
    ) -> None:
        _authority(known_authority)
        if any(
            type(handle) is not WindowsProcessHandle
            for handle in (supervisor, old_anchor, old_worker)
        ):
            raise ValueError("transfer supervisor requires exact retained Windows handles")
        assert isinstance(supervisor, WindowsProcessHandle)
        assert isinstance(old_anchor, WindowsProcessHandle)
        assert isinstance(old_worker, WindowsProcessHandle)
        witness = SupervisorRecoveryWitness(deepcopy(known_authority), supervisor.identity)
        api = supervisor._api
        if old_anchor._api is not api or old_worker._api is not api:
            raise ValueError("transfer supervisor registrations require one native API instance")
        if (
            old_anchor.identity != known_authority.anchor
            or old_worker.identity != known_authority.worker
        ):
            raise ValueError(
                "transfer supervisor original registrations differ from known authority"
            )
        if len({supervisor.identity.pid, old_anchor.identity.pid, old_worker.identity.pid}) != 3:
            raise ValueError("original simultaneously live registrations require distinct PIDs")
        current_pid = api.current_pid()
        if type(current_pid) is not int or current_pid != supervisor.identity.pid:
            raise ValueError("transfer supervisor must be the physical current process")
        if any(handle.is_ended() is not False for handle in (supervisor, old_anchor, old_worker)):
            raise ValueError("original transfer registrations must all be retained while live")
        # Only now does this instance own registrations; invalid preflight leaves
        # ownership with the caller, including their original process lifetimes.
        self._owned = [supervisor, old_anchor, old_worker]
        self._supervisor, self._old_anchor, self._old_worker = supervisor, old_anchor, old_worker
        self._api, self._witness = api, witness
        self._known = deepcopy(known_authority)
        self._peak = known_authority.metadata.usage.peak_rss_bytes
        self._last_ns = known_authority.metadata.observed_ns
        self._closed, self._failed, self._planned_once, self._commit_once = (
            False,
            False,
            False,
            False,
        )
        self._gate = Lock()
        self._change: CoordinatorTransfer | None = None
        self._binding: CoordinatorTransfer | None = None
        self._peer: BoundedJsonChild | None = None
        self._new_anchor: WindowsProcessHandle | None = None
        self._new_worker: WindowsProcessHandle | None = None
        try:
            self._observer = WindowsRecoveryObserver(supervisor, peak_rss_bytes=self._peak)
            initial = self._observer.observe(old_anchor)
            if initial.previous_owner_ended is not False:
                raise ValueError("original coordinator ended during supervisor registration")
            self._retain(initial)
            self._within_limits(self._last_ns, self._peak)
        except BaseException as error:
            self._failed = True
            self._cleanup_primary(error)
            raise

    def _retain(self, observation: RecoveryHostObservation) -> None:
        if observation.now_ns < self._last_ns:
            raise ValueError("transfer native observation time regressed")
        self._last_ns = observation.now_ns
        self._peak = max(self._peak, observation.peak_rss_bytes)

    def _within_limits(self, now: int, peak: int) -> None:
        metadata = self._known.metadata
        if (
            now - metadata.started_ns >= metadata.limits.max_elapsed_ns
            or peak > metadata.limits.max_rss_bytes
        ):
            raise ValueError("original transfer elapsed/RSS allowance exhausted")

    def _current_supervisor(self) -> None:
        pid = self._api.current_pid()
        if type(pid) is not int or pid != self._witness.supervisor.pid:
            raise ValueError("transfer supervisor physical process changed")
        if (
            self._supervisor.identity != self._witness.supervisor
            or self._supervisor.is_ended() is not False
        ):
            raise ValueError("retained transfer supervisor is no longer live")

    def _live_replacement(self) -> None:
        self._current_supervisor()
        assert (
            self._peer is not None and self._new_anchor is not None and self._new_worker is not None
        )
        if self._peer.pid != self._new_anchor.identity.pid or self._peer.returncode is not None:
            raise ValueError("owned transfer coordinator channel is no longer live/bound")
        if self._new_anchor.is_ended() is not False or self._new_worker.is_ended() is not False:
            raise ValueError("replacement transfer registrations must both remain live")
        if self._failed:
            raise ValueError("transfer supervisor previously failed")

    def _enter_operation(self, *, terminal: bool = False) -> None:
        if not self._gate.acquire(blocking=False):
            self._failed = True
            raise ValueError("transfer supervisor is busy; nested/concurrent operation refused")
        try:
            if self._closed or (self._failed and not terminal):
                raise ValueError("transfer supervisor is closed or previously failed")
        except BaseException:
            self._gate.release()
            raise

    def _registrations(self, peer: object, anchor: object, worker: object) -> None:
        if (
            type(peer) is not BoundedJsonChild
            or type(anchor) is not WindowsProcessHandle
            or type(worker) is not WindowsProcessHandle
        ):
            raise ValueError(
                "transfer requires an exact owned child and retained replacement handles"
            )
        assert isinstance(peer, BoundedJsonChild)
        assert isinstance(anchor, WindowsProcessHandle)
        assert isinstance(worker, WindowsProcessHandle)
        self._owned.extend((anchor, worker))
        self._peer, self._new_anchor, self._new_worker = peer, anchor, worker
        if anchor._api is not self._api or worker._api is not self._api:
            raise ValueError("replacement registrations require the original native API instance")
        identities = (
            self._known.anchor,
            self._known.worker,
            self._witness.supervisor,
            anchor.identity,
            worker.identity,
        )
        if (
            len(set(identities)) != 5
            or len({self._witness.supervisor.pid, anchor.identity.pid, worker.identity.pid}) != 3
        ):
            raise ValueError("transfer requires distinct full identities and distinct live PIDs")
        self._live_replacement()

    def plan_transfer(
        self, peer: object, new_anchor: object, new_worker: object, owner_id: str
    ) -> CoordinatorTransfer:
        self._enter_operation()
        try:
            if self._planned_once:
                raise ValueError("transfer may be planned only once")
            self._planned_once = True
            _identifier(owner_id, "new owner_id")
            if owner_id == self._known.metadata.owner_id:
                raise ValueError("transfer requires a distinct new owner_id")
            self._registrations(peer, new_anchor, new_worker)
            assert (
                self._peer is not None
                and self._new_anchor is not None
                and self._new_worker is not None
            )
            anchor_ended = self._observer.observe(self._old_anchor)
            self._retain(anchor_ended)
            if anchor_ended.previous_owner_ended is not True:
                raise ValueError("original coordinator has not ended")
            lower = self._observer.observe(self._old_worker)
            self._retain(lower)
            if lower.previous_owner_ended is not True:
                raise ValueError("original worker has not ended")
            self._within_limits(self._last_ns, self._peak)
            self._live_replacement()
            self._peer.send(
                dict(
                    op="observe",
                    worker=asdict(self._new_worker.identity),
                    peak_rss_bytes=lower.peak_rss_bytes,
                )
            )
            remote = _reply(self._peer.receive())
            if (
                remote.observer != self._new_anchor.identity
                or remote.previous_owner != self._new_worker.identity
                or remote.previous_owner_ended is not False
                or remote.clock_epoch != anchored_clock_epoch(self._new_anchor.identity)
                or remote.now_ns < lower.now_ns
                or remote.peak_rss_bytes < lower.peak_rss_bytes
            ):
                raise ValueError(
                    "owned transfer reply differs from native replacement bindings/bracket"
                )
            self._retain(remote)
            self._live_replacement()
            after = self._observer.observe(self._new_anchor)
            self._retain(after)
            self._live_replacement()
            change = plan_coordinator_transfer(
                self._witness,
                CoordinatorTransferObservation(anchor_ended, lower, remote, after),
                owner_id,
            )
            self._change, self._binding = change, deepcopy(change)
            return change
        except BaseException:
            self._failed = True
            raise
        finally:
            self._gate.release()

    def _planned(self) -> CoordinatorTransfer:
        if self._change is None or self._binding is None:
            raise ValueError("transfer requires a retained completed plan")
        validate_coordinator_transfer(self._change)
        if self._change != self._binding or self._change.expected != self._known:
            raise ValueError("retained transfer plan differs from independent original binding")
        return self._change

    def commit_transfer(self, journal: SqliteRecoveryJournal) -> CoordinatorTransfer:
        self._enter_operation()
        try:
            if self._commit_once:
                raise ValueError("transfer commit may be attempted only once")
            self._commit_once = True
            if type(journal) is not SqliteRecoveryJournal:
                raise ValueError("transfer requires an exact SQLite recovery journal")
            change = self._planned()
            self._live_replacement()
            now = self._api.interrupt_ns()
            if type(now) is not int or not 0 < now < 2**63 or now < self._last_ns:
                raise ValueError("transfer precommit native clock is invalid or regressed")
            # Even a later cap/CAS failure must not let terminal facts regress.
            self._last_ns = now
            self._within_limits(now, self._peak)
            self._live_replacement()
            if journal.transfer_coordinator(change) is not True:
                raise ValueError("coordinator transfer compare-and-swap was refused")
            return change
        except BaseException:
            self._failed = True
            raise
        finally:
            self._gate.release()

    def failure_witness(self) -> CoordinatorTransferFailure:
        self._enter_operation(terminal=True)
        try:
            self._failed = True  # Requesting terminal reconciliation never admits a later commit.
            change = self._planned()
            self._current_supervisor()
            assert self._new_anchor is not None
            observer = WindowsRecoveryObserver(self._supervisor, peak_rss_bytes=self._peak)
            observation = observer.observe(self._new_anchor)
            self._retain(observation)
            return CoordinatorTransferFailure(change, observation)
        finally:
            self._gate.release()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        seen: set[int] = set()
        for handle in reversed(self._owned):
            if id(handle) in seen:
                continue
            seen.add(id(handle))
            try:
                handle.close()
            except BaseException as error:
                errors.append(error)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("transfer registration cleanup failed", errors)

    def _cleanup_primary(self, primary: BaseException) -> None:
        try:
            self.close()
        except BaseException as cleanup:
            BaseException.add_note(primary, f"transfer registration cleanup failed: {cleanup!r}")

    def __enter__(self) -> "WindowsTransferSupervisor":
        acquired = False
        try:
            self._enter_operation()
            acquired = True
            self._current_supervisor()
            return self
        except BaseException as error:
            self._failed = True
            self._cleanup_primary(error)
            raise
        finally:
            if acquired:
                self._gate.release()

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc is None:
            self.close()
        else:
            self._cleanup_primary(exc)
