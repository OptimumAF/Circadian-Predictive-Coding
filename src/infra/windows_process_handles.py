"""Documented Windows observation APIs and retained registered process handles.

No process termination, blocking waits, owner lease, disk/native restore or boot
GUID attestation. Handles are registered while alive and explicitly closed.
"""

from dataclasses import replace
import ctypes
import sys
from threading import Lock
from typing import Any, Callable

from src.core.recovery_observation import RecoveryProcessIdentity, validate_process_identity
from src.shared.process_memory import read_process_rss_bytes

_REGISTRATION = object()


class _FileTime(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]


def _api_error(name: str) -> OSError:
    code = getattr(ctypes, "get_last_error", lambda: 0)()
    return OSError(code, f"Windows recovery {name} failed")


def _load_kernel(name: str = "kernel32"):
    loader = getattr(ctypes, "WinDLL", None)
    if sys.platform != "win32" or loader is None:
        raise OSError("Windows recovery observation APIs are unavailable on this host")
    return loader(name, use_last_error=True)


class WindowsRecoveryApi:
    """Narrow documented APIs; private injected kernel/RSS hooks are trusted tests."""

    filetime_type = _FileTime

    def __init__(
        self,
        *,
        _kernel: Any = None,
        rss_reader: Callable[[], int | None] = read_process_rss_bytes,
    ) -> None:
        kernel = _load_kernel() if _kernel is None else _kernel
        # The documented realtime API contract resolves exports absent in kernel32.
        timing = _load_kernel("api-ms-win-core-realtime-l1-1-1.dll") if _kernel is None else kernel
        self._rss = rss_reader
        signatures = {
            "OpenProcess": ([ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32], ctypes.c_void_p),
            "CloseHandle": ([ctypes.c_void_p], ctypes.c_int),
            "GetProcessTimes": ([ctypes.c_void_p] + [ctypes.POINTER(_FileTime)] * 4, ctypes.c_int),
            "WaitForSingleObject": ([ctypes.c_void_p, ctypes.c_uint32], ctypes.c_uint32),
            "GetCurrentProcessId": ([], ctypes.c_uint32),
            "QueryInterruptTimePrecise": ([ctypes.POINTER(ctypes.c_uint64)], None),
        }
        self._calls: dict[str, Any] = {}
        try:
            for name, (arguments, result) in signatures.items():
                function = getattr(timing if name == "QueryInterruptTimePrecise" else kernel, name)
                if not callable(function):
                    raise AttributeError(name)
                function.argtypes, function.restype = arguments, result
                self._calls[name] = function
        except AttributeError as error:
            raise OSError("Windows recovery required API is unavailable") from error

    def open_process(self, pid: int) -> int:
        handle = self._calls["OpenProcess"](0x101000, False, pid)
        if (
            type(handle) is not int
            or not 0 < handle < 2 ** (ctypes.sizeof(ctypes.c_void_p) * 8) - 1
        ):
            raise _api_error("OpenProcess")
        return handle

    def close_handle(self, handle: int) -> None:
        if not self._calls["CloseHandle"](handle):
            raise _api_error("CloseHandle")

    def creation_time(self, handle: int) -> int:
        creation, exit_time, kernel, user = (_FileTime() for _ in range(4))
        if not self._calls["GetProcessTimes"](
            handle, *(ctypes.byref(value) for value in (creation, exit_time, kernel, user))
        ):
            raise _api_error("GetProcessTimes")
        return (int(creation.high) << 32) | int(creation.low)

    def process_ended(self, handle: int) -> bool:
        status = self._calls["WaitForSingleObject"](handle, 0)
        if type(status) is not int or status not in (0, 258):
            raise _api_error("WaitForSingleObject")
        return status == 0

    def current_pid(self) -> int:
        return self._calls["GetCurrentProcessId"]()

    def interrupt_ns(self) -> int:
        counter = ctypes.c_uint64()
        self._calls["QueryInterruptTimePrecise"](ctypes.byref(counter))
        return int(counter.value) * 100

    def rss_bytes(self) -> int | None:
        return self._rss()


class WindowsProcessHandle:
    """Owned noninherited kernel reference, retained across registered worker exit."""

    def __init__(
        self,
        handle: int,
        identity: RecoveryProcessIdentity,
        api: WindowsRecoveryApi,
        *,
        _registration: object = None,
    ):
        if _registration is not _REGISTRATION:
            raise ValueError("process handles require live registration through pin")
        self._handle, self._identity, self._api = handle, identity, api
        self._closed = False
        self._gate = Lock()

    @classmethod
    def pin(
        cls,
        pid: int,
        *,
        expected: RecoveryProcessIdentity | None = None,
        api: WindowsRecoveryApi | None = None,
    ) -> "WindowsProcessHandle":
        if cls is not WindowsProcessHandle:
            raise ValueError("unsupported process handle subclass")
        RecoveryProcessIdentity(pid, 1)  # Reject bad PIDs before native calls.
        if expected is not None:
            validate_process_identity(expected)
            if expected.pid != pid:
                raise ValueError("registered process identity PID differs")
        api = WindowsRecoveryApi() if api is None else api
        if type(api) is not WindowsRecoveryApi:
            raise ValueError("process registration requires supported observation APIs")
        handle = api.open_process(pid)
        try:
            identity = RecoveryProcessIdentity(pid, api.creation_time(handle))
            if expected is not None and expected != identity:
                raise ValueError("registered process identity differs; possible PID reuse")
            if api.process_ended(handle):
                raise ValueError("process must be registered while alive")
            return cls(handle, identity, api, _registration=_REGISTRATION)
        except BaseException:
            api.close_handle(handle)
            raise

    @property
    def identity(self) -> RecoveryProcessIdentity:
        validate_process_identity(self._identity)
        return replace(self._identity)

    def is_ended(self) -> bool:
        with self._gate:
            if self._closed:
                raise ValueError("registered process handle is closed")
            if self._api.creation_time(self._handle) != self.identity.created_filetime:
                raise ValueError("pinned process identity changed")
            return self._api.process_ended(self._handle)

    def close(self) -> None:
        with self._gate:
            if self._closed:
                return
            # A failed close is reported; never reuse an uncertain handle.
            self._closed = True
            self._api.close_handle(self._handle)

    def __enter__(self) -> "WindowsProcessHandle":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
