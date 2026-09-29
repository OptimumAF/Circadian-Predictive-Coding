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
from dataclasses import dataclass
from functools import lru_cache
from math import isfinite
from os import getpid
from threading import Event, Lock, Thread
from types import TracebackType
from typing import Any, Callable


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
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
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
            if self.start_bytes is None or self.peak_bytes is None or self.sample_count < 1:
                raise RuntimeError("Process RSS is unavailable for checkpoint telemetry.")
            return ProcessRssSegment(
                pid=getpid(),
                start_bytes=self.start_bytes,
                peak_bytes=self.peak_bytes,
                sample_count=self.sample_count,
                interval_seconds=self.interval_seconds,
            )

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
                self._error = error
                self._stop.set()
                return
