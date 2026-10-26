"""Three ordered outer trials with fixed utility and sealed final callback.

Application owners supply existing execution/readout callbacks. This boundary
does not run models, store labels, checkpoint, select policies or grant learner
permissions. One study holds at most three scalar rows and one identity receipt.
Trusted local Python is not a hostile-code sandbox.
"""

from contextlib import contextmanager
from copy import deepcopy
from typing import Callable, Iterator

from src.core.controller_objective import (
    ControllerFreeze,
    ControllerProtocol,
    ControllerTrial,
    ControllerTrialMetadata,
    ControllerStream,
    ScoredControllerTrial,
    controller_utility,
)
from src.core.experience import LogicalClock, require_tick
from src.core.reference_controller import ReferencePolicy


class ControllerChronology:
    def __init__(self, protocol: ControllerProtocol, clock: LogicalClock):
        if type(protocol) is not ControllerProtocol or type(clock) is not LogicalClock:
            raise ValueError("chronology requires a declared protocol and logical clock")
        protocol.__post_init__()
        if clock.now() != protocol.declared_at:
            raise ValueError("declare the protocol at its original clock before trial execution")
        self._protocol, self._pin = deepcopy(protocol), deepcopy(protocol)
        self._clock, self._original_clock = clock, clock
        self._rows: tuple[ScoredControllerTrial, ...] = ()
        self._attempts = 0
        self._busy = False
        self._stopped = False
        self._receipt: ControllerFreeze | None = None
        self._frozen_at: int | None = None

    def _require_owner(self):
        if self._protocol != self._pin or self._clock is not self._original_clock:
            raise ValueError(
                "chronology requires its original objective, policy, streams and clock"
            )
        self._protocol.__post_init__()
        if self._stopped:
            raise ValueError("chronology stopped after failed paid trial")

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        if self._busy:
            raise ValueError("chronology is busy; reentrant trial/release refused")
        self._busy = True
        try:
            yield
        finally:
            self._busy = False

    def run_development(
        self, stream_id: str, operation: Callable[[ReferencePolicy], ControllerTrial]
    ) -> ScoredControllerTrial:
        with self._exclusive():
            self._require_owner()
            if self._receipt is not None or len(self._rows) >= 2:
                raise ValueError("development is closed; no repeated validation/tuning")
            stream = self._protocol.streams[len(self._rows)]
            if stream.stream_id != stream_id:
                raise ValueError(
                    "development requires the next declared stream before payload access"
                )
            return self._run(stream, operation)

    def _run(self, stream, operation):
        if not callable(operation):
            raise ValueError("trial operation must be callable")
        started = self._clock.now()
        self._attempts += 1
        try:
            result = operation(deepcopy(self._protocol.policy))
            self._require_owner()
            if (
                type(result) is not ControllerTrial
                or type(result.metadata) is not ControllerTrialMetadata
            ):
                raise ValueError("trial must return declared controller metadata before metrics")
            metadata = result.metadata
            if (
                type(metadata.stream) is not ControllerStream
                or type(metadata.policy) is not ReferencePolicy
            ):
                raise ValueError("trial requires exact stream and policy metadata before metrics")
            metadata.stream.__post_init__()
            metadata.policy.__post_init__()
            if metadata.stream != stream or metadata.policy != self._protocol.policy:
                raise ValueError("trial stream role/group or policy differs before metrics access")
            for v in (metadata.started_at, metadata.labels_arrived_at, metadata.completed_at):
                require_tick(v, "trial chronology")
            if (
                not started
                <= metadata.started_at
                <= metadata.labels_arrived_at
                <= metadata.completed_at
                <= self._clock.now()
            ):
                raise ValueError("trial labels/returns have not arrived causally")
            row = ScoredControllerTrial(
                deepcopy(metadata), controller_utility(self._protocol.objective, result.measures)
            )
            self._rows = (*self._rows, row)
            return deepcopy(row)
        except BaseException:
            self._stopped = True
            raise

    def training_rows(self) -> tuple[ScoredControllerTrial, ...]:
        with self._exclusive():
            self._require_owner()
            if len(self._rows) != 1 or self._receipt is not None:
                raise ValueError("controller training rows are available only before validation")
            return deepcopy(self._rows[:1])

    def validation_rows(self) -> tuple[ScoredControllerTrial, ...]:
        with self._exclusive():
            self._require_owner()
            if len(self._rows) != 2 or self._receipt is not None:
                raise ValueError("validation feedback is available only before freeze")
            return deepcopy(self._rows[1:2])

    def freeze(self) -> ControllerFreeze:
        with self._exclusive():
            self._require_owner()
            if len(self._rows) != 2 or self._receipt is not None:
                raise ValueError("freeze requires complete train/validation and occurs once")
            self._receipt = ControllerFreeze(deepcopy(self._protocol), self._clock.now())
            self._frozen_at = self._clock.now()
            return self._receipt

    def evaluate_final(
        self, receipt: ControllerFreeze, operation: Callable[[ReferencePolicy], ControllerTrial]
    ) -> ScoredControllerTrial:
        with self._exclusive():
            self._require_owner()
            if (
                self._receipt is None
                or receipt is not self._receipt
                or len(self._rows) != 2
                or receipt.protocol != self._pin
                or receipt.frozen_at != self._frozen_at
                or receipt.frozen_at > self._clock.now()
            ):
                raise ValueError(
                    "sealed evaluation requires the original current freeze before source access"
                )
            return self._run(self._protocol.streams[2], operation)

    def snapshot(self) -> dict:
        # Metadata stays observable after a paid failure; no retry/reset authority.
        return dict(
            protocol=deepcopy(self._protocol),
            attempts=self._attempts,
            rows=deepcopy(self._rows),
            frozen_at=self._frozen_at,
            stopped=self._stopped,
            clock=self._clock.now(),
        )
