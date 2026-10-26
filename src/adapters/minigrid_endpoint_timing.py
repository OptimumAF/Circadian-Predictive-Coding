"""Bounded elapsed spans for the existing MiniGrid endpoint evaluator.

Inputs are a diagnostic clock and four named spans. Output is detached JSON
metadata, never budget authority, scientific qualification or exclusive CPU time.
No model, environment, filesystem or network work is performed here.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from math import isfinite
from time import perf_counter
from typing import Any

PHASES = ("preservation", "input_preparation", "prediction", "physics")


class MiniGridEndpointTiming:
    """Keep four aggregate counters; retain no per-call or success-record chain."""

    def __init__(self, clock: Callable[[], float] = perf_counter) -> None:
        if not callable(clock):
            raise ValueError("endpoint timing requires a callable diagnostic clock")
        self._clock = clock
        self._last: float | None = None
        self._reads = 0
        self._error: str | None = None
        self._phases: dict[str, dict[str, Any]] = {
            name: dict(attempted=0, returned=0, failed=0, timed_calls=0, known_seconds=0.0)
            for name in PHASES
        }

    def _sample_clock(self) -> float:
        return self._clock()

    def _read(self) -> float | None:
        if self._error is not None:
            return None
        self._reads += 1
        try:
            raw = self._sample_clock()
            if type(raw) not in (int, float):
                raise ValueError("diagnostic clock must return native finite seconds")
            value = float(raw)
            if not isfinite(value) or (self._last is not None and value < self._last):
                raise ValueError("diagnostic clock is nonfinite or moved backwards")
            self._last = value
            return value
        except BaseException as error:
            # Why: diagnostic failure cannot replace a paid execution fault or
            # renew its deadline. Its missing interval stays explicitly unknown.
            self._error = f"{type(error).__name__}: {error}"[:256]
            if not isinstance(error, Exception):
                raise
            return None

    @contextmanager
    def measure(self, phase: str) -> Iterator[None]:
        if phase not in self._phases:
            raise ValueError("unsupported MiniGrid endpoint timing phase")
        row = self._phases[phase]
        row["attempted"] += 1
        start = self._read()
        operation_failed = False
        try:
            yield
        except BaseException:
            operation_failed = True
            row["failed"] += 1
            raise
        else:
            row["returned"] += 1
        finally:
            try:
                end = self._read()
            except BaseException:
                if not operation_failed:
                    raise
                # A diagnostic cancellation cannot hide the original paid
                # operation failure/cancellation already propagating outward.
                end = None
            if start is not None and end is not None:
                seconds = end - start
                total = row["known_seconds"] + seconds
                if isfinite(seconds) and isfinite(total):
                    row["known_seconds"] = total
                    row["timed_calls"] += 1
                else:
                    self._error = "ValueError: diagnostic elapsed interval overflowed"

    def report(self) -> dict[str, Any]:
        """Read detached aggregates without another diagnostic or budget read."""
        phases = {
            name: {
                **row,
                "seconds": row["known_seconds"] if row["timed_calls"] == row["attempted"] else None,
            }
            for name, row in self._phases.items()
        }
        return dict(
            status="complete" if self._error is None else "incomplete",
            clock_reads=self._reads,
            clock_error=self._error,
            phases=phases,
            scope=(
                "elapsed inclusive spans, including failed calls; not exclusive CPU/kernel time; "
                "prediction includes adapter projection/validation; input preparation covers "
                "public observation/history/array construction; preservation covers evaluator "
                "checks including qualified caller checks; physics covers step only; "
                "reset/fork/setup/serialization/uncovered checks, timing bookkeeping and Python "
                "storage are additional; spans may overlap and must not be summed as total time; "
                "unknown intervals excluded from known_seconds; no budget authority"
            ),
        )


class NativeMiniGridEndpointTiming(MiniGridEndpointTiming):
    """Production spans use the system clock, never a caller-supplied callback.

    Why: an injected clock could mutate state after a completed preservation
    check. Keep injection in the isolated recorder/tests, outside the H32 port.
    """

    def __init__(self) -> None:
        super().__init__()

    def _sample_clock(self) -> float:
        # Do not read the inherited callback field: replacing it after admission
        # must not introduce another callback between state proof and prediction.
        return perf_counter()
