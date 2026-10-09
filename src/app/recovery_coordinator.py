"""Surviving coordinator sequencing over trusted registrations/observer/journal.

Owns original high-water witness and retained probes; commits before preparation,
checks host/persisted facts around trusted callbacks. Publication uses an inward
writer lease; no native completion protocol or rollback of side effects.
"""

from contextlib import contextmanager
from threading import Lock
from typing import Callable, Iterator, TypeVar

from src.core.recovery_authority import (
    AuthorityRecord,
    AuthorityChange,
    plan_handoff,
    plan_reservation,
    validate_authority_record,
)
from src.core.recovery_coordination import (
    RecoveryCosts,
    RecoveryRegistration,
    validate_coordination_observation,
)
from src.core.recovery_observation import (
    RecoveryHostObservation,
    RecoveryObservationPort,
    validate_process_identity,
)
from src.core.recovery_reporting import (
    AuthorityReport,
    FailureWitness,
    TerminalReportingFailure,
    plan_report,
    select_report_observation,
)
from src.core.recovery_publication import RecoveryPublicationLease, RecoveryLeasedPublicationPort

Value = TypeVar("Value")


class RecoveryCoordinator:
    def __init__(
        self,
        known_authority: AuthorityRecord,
        authority: RecoveryLeasedPublicationPort,
        observer: RecoveryObservationPort,
        anchor: RecoveryRegistration,
        worker: RecoveryRegistration,
    ) -> None:
        validate_authority_record(known_authority)
        self._current, self._authority, self._observer = known_authority, authority, observer
        self._anchor, self._worker = anchor, worker
        self._owned = [anchor, worker]
        self._gate = Lock()
        self._closed = self._failed = False
        self._now = known_authority.metadata.observed_ns
        self._peak = known_authority.metadata.usage.peak_rss_bytes
        self._attempt: AuthorityChange | AuthorityReport | None = None
        self._failure_observation: RecoveryHostObservation | None = None
        self._failure_witness: FailureWitness | None = None
        self._terminal_confirmed = False
        try:
            self._check()
        except BaseException as error:
            self._failed = True
            terminal_error = self._record_terminal_failure()
            if terminal_error is not None:
                error.__cause__ = terminal_error
            try:
                self._close_owned()
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "coordinator initialization and cleanup failed", [error, cleanup]
                )
            raise

    def _require_ready(self) -> None:
        if self._closed:
            raise ValueError("coordinator closed")
        if self._failed:
            raise ValueError("coordinator previously failed; independent reconciliation required")

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        if not self._gate.acquire(blocking=False):
            raise ValueError("coordinator busy; nested/concurrent lane refused")
        try:
            self._require_ready()
            try:
                yield
            except BaseException as error:
                self._failed = True
                terminal_error = self._record_terminal_failure()
                if terminal_error is not None:
                    raise error from terminal_error
                raise
        finally:
            self._gate.release()

    def _read_exact(self, lease: RecoveryPublicationLease | None = None) -> None:
        current = self._authority.read() if lease is None else lease.read()
        validate_authority_record(current)
        if current != self._current:
            raise ValueError("coordinator persisted authority differs from independent witness")

    def _bindings(self, *, ended: bool) -> None:
        if (
            self._anchor.identity != self._current.anchor
            or self._worker.identity != self._current.worker
        ):
            raise ValueError("coordinator original registered process identity changed")
        if self._anchor.is_ended() is not False or self._worker.is_ended() is not ended:
            raise ValueError("coordinator registered process is ended or unsupported")

    def _sample(self, *, ended: bool = False, pending: bool = False) -> RecoveryHostObservation:
        self._bindings(ended=ended)
        observation = self._observer.observe(self._worker)
        # Keep bound over-cap facts for terminal evidence; invalid facts still fail below.
        captured = select_report_observation(self._current, observation)
        if captured is not None:
            self._failure_observation = captured
        validate_coordination_observation(
            self._current, observation, self._now, self._peak, ended=ended, pending=pending
        )
        self._bindings(ended=ended)
        self._now, self._peak = observation.now_ns, observation.peak_rss_bytes
        return observation

    def _check(
        self, *, pending: bool = False, lease: RecoveryPublicationLease | None = None
    ) -> None:
        self._read_exact(lease)
        observation = self._sample(pending=pending)
        self._commit(plan_report(self._current, observation), lease)

    def _commit(
        self,
        change: AuthorityChange | AuthorityReport,
        lease: RecoveryPublicationLease | None = None,
    ) -> None:
        self._attempt = change
        if isinstance(change, AuthorityReport):
            committed = self._authority.report(change) if lease is None else lease.report(change)
        else:
            if lease is not None:
                raise ValueError("publication lease cannot reserve native work")
            committed = self._authority.advance(change)
        if committed is not True:
            raise ValueError("coordinator authority CAS stale or outcome unsupported")
        self._current = change.proposed
        self._read_exact(lease)
        self._attempt = None

    @property
    def terminal_confirmed(self) -> bool:
        return self._terminal_confirmed

    @property
    def failure_witness(self) -> FailureWitness | None:
        return self._failure_witness

    def _record_terminal_failure(self) -> TerminalReportingFailure | None:
        expected = self._current if self._attempt is None else self._attempt.expected
        observation = select_report_observation(expected, self._failure_observation)
        if observation is None and self._attempt is not None:
            observation = select_report_observation(
                self._attempt.proposed, self._failure_observation
            )
        terminal = plan_report(
            self._current,
            select_report_observation(self._current, self._failure_observation),
            terminal=True,
        )
        witness = FailureWitness(expected, self._attempt, terminal, observation)
        self._failure_witness = witness
        try:
            if self._authority.report(terminal) is not True:
                raise ValueError("terminal authority CAS stale or outcome unsupported")
            self._current = terminal.proposed
            self._read_exact()
            self._terminal_confirmed = True
            return None
        except BaseException as error:
            self._terminal_confirmed = False
            failure = TerminalReportingFailure(witness)
            failure.__cause__ = error
            return failure

    def execute(
        self,
        costs: RecoveryCosts,
        prepare: Callable[[], Value],
        publish: Callable[[Value], None] | None = None,
    ) -> Value:
        with self._exclusive():
            if type(costs) is not RecoveryCosts:
                raise ValueError("coordinator requires exact costs")
            RecoveryCosts.__post_init__(costs)
            if not callable(prepare) or (publish is not None and not callable(publish)):
                raise ValueError("coordinator requires trusted callable actions")
            self._read_exact()
            observation = self._sample()
            change = plan_reservation(
                self._current,
                observation,
                updates=costs.updates,
                copy_bytes=costs.copy_bytes,
                grants=costs.grants,
                checkpoint_attempts=costs.checkpoint_attempts,
            )
            self._commit(change)
            pending = costs.updates == 1
            self._check(pending=pending)
            value = prepare()
            self._check(pending=pending)
            if publish is not None:
                # No native completion codec exists: pending work cannot publish.
                self._check()
                with self._authority.publication_lease(self._current) as lease:
                    self._check(lease=lease)
                    publish(value)
                    self._check(lease=lease)
                self._check()
            return value

    def handoff(self, replacement: RecoveryRegistration, owner_id: str) -> None:
        with self._exclusive():
            if not any(replacement is probe for probe in self._owned):
                self._owned.append(replacement)
            validate_process_identity(replacement.identity)
            if (
                replacement.identity in (self._current.anchor, self._current.worker)
                or replacement.is_ended() is not False
            ):
                raise ValueError("coordinator handoff requires independent live registered worker")
            self._read_exact()
            observation = self._sample(ended=True)
            change = plan_handoff(self._current, observation, replacement.identity, owner_id)
            old = self._worker
            self._commit(change)
            self._worker = replacement
            self._check()
            old.close()
            self._owned = [probe for probe in self._owned if probe is not old]

    def _close_owned(self) -> None:
        self._closed = self._failed = True
        probes, self._owned = self._owned, []
        errors: list[BaseException] = []
        seen: set[int] = set()
        for probe in probes:
            if id(probe) in seen:
                continue
            seen.add(id(probe))
            try:
                probe.close()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise BaseExceptionGroup("coordinator registration cleanup failed", errors)

    def close(self) -> None:
        if self._closed:
            return
        if not self._gate.acquire(blocking=False):
            raise ValueError("coordinator busy; close during operation refused")
        try:
            terminal_error = None
            if not self._failed:
                self._failed = True
                terminal_error = self._record_terminal_failure()
            try:
                self._close_owned()
            except BaseException as cleanup:
                if terminal_error is not None:
                    raise BaseExceptionGroup(
                        "terminal reporting and cleanup failed", [terminal_error, cleanup]
                    )
                raise
            if terminal_error is not None:
                raise terminal_error
        finally:
            self._gate.release()

    def __enter__(self) -> "RecoveryCoordinator":
        self._require_ready()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
