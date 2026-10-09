"""Sample process resident memory for local benchmark telemetry.

Inputs are the current process and an optional RSS reader. Outputs are a
start value, an observed high-water RSS, and a typed invocation snapshot.
This module does not attribute memory to a model or replace a process-isolated
profiler.
"""

from __future__ import annotations

import ctypes
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from math import isfinite
from os import getpid
from threading import Event, Lock, Thread, get_ident
from types import TracebackType
from typing import Any, Callable, Iterator


@dataclass(frozen=True)
class ProcessRssSegment:
    """One invocation's observed absolute RSS, starting at the trainer boundary."""

    pid: int
    start_bytes: int
    peak_bytes: int
    sample_count: int
    interval_seconds: float


class _WindowsMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


@lru_cache(maxsize=1)
def _windows_memory_apis() -> tuple[Any, Any]:
    # ctypes exposes WinDLL only on Windows, but this module is type-checked on Linux too.
    windows_dll = getattr(ctypes, "WinDLL", None)
    if windows_dll is None:
        raise OSError("Windows process memory APIs are unavailable.")
    kernel32 = windows_dll("kernel32", use_last_error=True)
    psapi = windows_dll("psapi", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    get_process_memory_info = psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_WindowsMemoryCounters),
        ctypes.c_ulong,
    ]
    get_process_memory_info.restype = ctypes.c_int
    return get_current_process, get_process_memory_info


def read_process_rss_bytes() -> int | None:
    """Return current process RSS on Windows/Linux; None on unsupported hosts."""
    if sys.platform == "win32":
        get_current_process, get_process_memory_info = _windows_memory_apis()
        counters = _WindowsMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not get_process_memory_info(
            get_current_process(),
            ctypes.byref(counters),
            counters.cb,
        ):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return int(counters.WorkingSetSize)
    if sys.platform.startswith("linux"):
        with open("/proc/self/statm", encoding="ascii") as status:
            resident_pages = int(status.read().split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    return None


class ProcessRssSampler:
    """Observe a process RSS high-water value while a benchmark section runs."""

    def __init__(
        self,
        *,
        interval_seconds: float = 0.005,
        read_rss_bytes: Callable[[], int | None] = read_process_rss_bytes,
    ) -> None:
        if not isfinite(interval_seconds) or interval_seconds <= 0.0:
            raise ValueError("interval_seconds must be positive and finite.")
        self.interval_seconds = interval_seconds
        self.read_rss_bytes = read_rss_bytes
        self.start_bytes: int | None = None
        self.peak_bytes: int | None = None
        self.sample_count = 0
        self._lock = Lock()
        self._stop = Event()
        self._thread: Thread | None = None
        self._error: Exception | None = None

    def sample(self) -> int | None:
        """Observe RSS now, including allocations deliberately held by a caller."""
        value = self.read_rss_bytes()
        if value is not None:
            with self._lock:
                if self.start_bytes is None:
                    self.start_bytes = value
                self.peak_bytes = value if self.peak_bytes is None else max(self.peak_bytes, value)
                self.sample_count += 1
        return value

    def snapshot(self) -> ProcessRssSegment:
        """Capture a consistent observation through the latest explicit sample."""
        with self._lock:
            return self._snapshot_leased()

    def _snapshot_leased(self) -> ProcessRssSegment:
        if self.start_bytes is None or self.peak_bytes is None or self.sample_count < 1:
            raise RuntimeError("Process RSS is unavailable for checkpoint telemetry.")
        return ProcessRssSegment(
            getpid(), self.start_bytes, self.peak_bytes, self.sample_count, self.interval_seconds
        )

    @contextmanager
    def _lease_observation(self) -> Iterator[Callable[[], ProcessRssSegment | None]]:
        """Internal capture port: freeze original counters before reading RSS.

        Why: public sample/snapshot reacquire a nonreentrant lock. Keep this
        capability within one original thread/source interval, without renewing
        baseline or allowing a blocked capture to invoke its reader first.
        """
        gate, reader, interval, thread = (
            self._lock,
            self.read_rss_bytes,
            self.interval_seconds,
            get_ident(),
        )
        active = False
        reading = False
        stop, worker = self._stop, self._thread

        def observe() -> ProcessRssSegment | None:
            nonlocal reading
            if not active or get_ident() != thread:
                raise ValueError("RSS observation outside original lease/thread")
            if reading:
                self._error = ValueError("reentrant original RSS reader")
                stop.set()
                raise self._error
            if (
                self._lock is not gate
                or self.read_rss_bytes is not reader
                or self.interval_seconds != interval
                or self._stop is not stop
                or self._thread is not worker
            ):
                raise ValueError("original RSS gate/reader/interval changed")
            self._require_capture_open()
            assert self.peak_bytes is not None
            counters = (self.start_bytes, self.peak_bytes, self.sample_count)
            try:
                reading = True
                value = reader()
                if value is None:
                    self._error = RuntimeError("original RSS observation unavailable")
                    self._stop.set()
                    return None
                if type(value) is not int or value < 0:
                    raise ValueError("original RSS observation requires nonnegative exact bytes")
                if (
                    self._lock is not gate
                    or self.read_rss_bytes is not reader
                    or self.interval_seconds != interval
                    or self._stop is not stop
                    or self._thread is not worker
                    or (self.start_bytes, self.peak_bytes, self.sample_count) != counters
                ):
                    raise ValueError("original RSS source changed during reader callback")
                self._require_capture_open()
                self.peak_bytes = max(self.peak_bytes, value)  # baseline was checked, never reset
                self.sample_count += 1
                return self._snapshot_leased()
            except Exception as error:
                if self._error is None:
                    self._error = error
                stop.set()
                raise
            finally:
                reading = False

        if not gate.acquire(blocking=False):
            raise ValueError("original RSS sampler is busy")
        try:
            active = True
            yield observe
        finally:
            active = False
            gate.release()

    def _require_capture_open(self) -> None:
        if self._error is not None:
            raise RuntimeError("original RSS sampler failed") from self._error
        if self._stop.is_set() or (self._thread is not None and not self._thread.is_alive()):
            raise ValueError("original RSS sampler is terminal")
        if (
            type(self.start_bytes) is not int
            or type(self.peak_bytes) is not int
            or self.start_bytes < 0
            or self.peak_bytes < self.start_bytes
            or type(self.sample_count) is not int
            or self.sample_count < 1
        ):
            raise ValueError("original RSS baseline/peak/count is unavailable or corrupt")

    def __enter__(self) -> ProcessRssSampler:
        self.sample()
        if self.start_bytes is not None:
            self._thread = Thread(target=self._sample_until_stopped, daemon=True)
            self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.sample()
        if self._error is not None and exc is None:
            raise RuntimeError("Process RSS sampler failed") from self._error

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.sample()
            except Exception as error:
                with self._lock:
                    self._error = error
                    self._stop.set()
                return
