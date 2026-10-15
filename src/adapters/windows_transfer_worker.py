"""Fixed trusted child for the physical, zero-native transfer example.

Inputs are bounded commands on a supervisor-owned private pipe. Authority is
adopted only from that trusted supervisor after its CAS acknowledgment, never
discovered from disk. Holds its physical process and worker registrations; no
model, checkpoint restoration, child launch or general RPC service.
"""

from contextlib import ExitStack
from dataclasses import asdict, fields
import json
import os
import sys
from typing import Any

from src.core.recovery_coordination import RecoveryCosts
from src.core.recovery_observation import RecoveryProcessIdentity
from src.infra.bounded_json_child import _json_object
from src.infra.recovery_authority_codec import decode_authority, encode_authority
from src.infra.sqlite_recovery_journal import SqliteRecoveryJournal
from src.infra.windows_process_handles import WindowsProcessHandle, WindowsRecoveryApi
from src.infra.windows_recovery_composition import compose_windows_recovery
from src.infra.windows_recovery_observer import WindowsRecoveryObserver


def _fields(value: Any, expected: set[str]) -> None:
    if type(value) is not dict or set(value) != expected:
        raise ValueError("transfer worker command fields differ")


def _identity(value: Any) -> RecoveryProcessIdentity:
    _fields(value, {field.name for field in fields(RecoveryProcessIdentity)})
    return RecoveryProcessIdentity(**value)


def _emit(value: dict[str, Any]) -> None:
    raw = json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
    if len(raw) > 65536:
        raise ValueError("transfer worker response exceeds line bound")
    sys.stdout.buffer.write(raw)
    sys.stdout.buffer.flush()


def _adopt(request, own, worker, last):
    _fields(request, {"op", "authority_utf8", "journal_path"})
    raw, path = request["authority_utf8"], request["journal_path"]
    if type(raw) is not str or type(path) is not str or not path or len(path) > 4096:
        raise ValueError("transfer adoption requires bounded authority and private journal path")
    known = decode_authority(raw.encode("utf-8"))
    if (
        known.anchor != own.identity
        or known.worker != worker.identity
        or known.metadata.observed_ns < last.now_ns
    ):
        raise ValueError("transferred authority differs from physical retained registration")
    journal = SqliteRecoveryJournal(path, known)
    coordinator = compose_windows_recovery(known, path, own, worker)
    return coordinator, journal


def _continue(request, own, worker, last, native_calls) -> dict[str, Any]:
    coordinator, journal = _adopt(request, own, worker, last)
    callbacks = []

    def prepare():
        callbacks.append("prepare")
        return "prepared"

    def publish(value):
        if value != "prepared":
            raise ValueError("unexpected fixed prepared value")
        callbacks.append("published")

    def forbidden():
        callbacks.append("forbidden")

    try:
        result = coordinator.execute(
            RecoveryCosts(copy_bytes=1, checkpoint_attempts=1), prepare, publish
        )
        accepted = journal.read()
        try:
            coordinator.execute(RecoveryCosts(checkpoint_attempts=1), forbidden)
        except ValueError as error:
            if str(error) != "original authority allowance exceeded":
                raise
        else:
            raise ValueError("fixed original attempt allowance was not exhausted")
        terminal = journal.read()
        if (
            result != "prepared"
            or callbacks != ["prepare", "published"]
            or not terminal.metadata.stopped
        ):
            raise ValueError("fixed continuation/stop callbacks differ")
        return dict(
            event="continued",
            accepted_utf8=encode_authority(accepted).decode("utf-8"),
            terminal_utf8=encode_authority(terminal).decode("utf-8"),
            callbacks=callbacks,
            native_calls=native_calls[0],
        )
    finally:
        coordinator.close()


def _serve(
    stack: ExitStack, own: WindowsProcessHandle, api: WindowsRecoveryApi, native_calls
) -> None:
    _emit(dict(event="hello", pid=own.identity.pid, version=list(sys.version_info[:3])))
    worker = last = None
    continued = False
    bound = None
    for _ in range(4):
        raw = sys.stdin.buffer.readline(65537)
        if len(raw) > 65536 or not raw.endswith(b"\n"):
            raise ValueError("transfer worker requires a bounded complete command line")
        request = _json_object(raw)
        if request == {"op": "exit"}:
            return
        if request == {"op": "crash"} and bound is not None:
            os._exit(17)
        if request.get("op") == "observe" and not continued and bound is None:
            _fields(request, {"op", "worker", "peak_rss_bytes"})
            expected = _identity(request["worker"])
            if worker is None:
                worker = stack.enter_context(
                    WindowsProcessHandle.pin(expected.pid, expected=expected, api=api)
                )
            elif worker.identity != expected:
                raise ValueError("transfer worker cannot change its retained worker")
            seed = request["peak_rss_bytes"]
            if type(seed) is not int or not 0 < seed < 2**63:
                raise ValueError("transfer worker peak seed requires a positive bounded integer")
            observer = WindowsRecoveryObserver(
                own, peak_rss_bytes=max(seed, 0 if last is None else last.peak_rss_bytes)
            )
            observation = observer.observe(worker)
            if observation.previous_owner_ended is not False or (
                last is not None and observation.now_ns < last.now_ns
            ):
                raise ValueError("transfer worker observation is no longer live/monotone")
            last = observation
            _emit(dict(event="observed", observation=asdict(observation)))
        elif (
            request.get("op") == "bind"
            and worker is not None
            and last is not None
            and bound is None
            and not continued
        ):
            bound, journal = _adopt(request, own, worker, last)
            stack.callback(bound.close)
            callbacks = []

            def bind_callback():
                callbacks.append("bound")
                return "bound"

            if bound.execute(RecoveryCosts(), bind_callback) != "bound":
                raise ValueError("original coordinator bind callback differs")
            _emit(
                dict(
                    event="bound",
                    authority_utf8=encode_authority(journal.read()).decode("utf-8"),
                    callbacks=callbacks,
                    native_calls=native_calls[0],
                )
            )
        elif (
            request.get("op") == "continue"
            and worker is not None
            and last is not None
            and not continued
            and bound is None
        ):
            continued = True
            _emit(_continue(request, own, worker, last, native_calls))
        else:
            raise ValueError("unsupported transfer worker command or chronology")
    raise ValueError("transfer worker command limit exhausted")


def main() -> int:
    if sys.platform != "win32":
        raise OSError("physical transfer worker requires Windows")
    native_calls = [0]

    def forbid_native(frame, event, arg):
        # Filename/code observation is alias safe and requires no model imports.
        path = frame.f_code.co_filename.replace("\\", "/")
        if event == "call" and (
            "/src/core/coupled_" in path
            or "/src/app/serving_promotion.py" in path
            or "/src/app/promotion_guard_evaluation.py" in path
        ):
            native_calls[0] += 1
            raise RuntimeError("zero-native transfer worker budget exceeded before work")

    sys.setprofile(forbid_native)
    try:
        api = WindowsRecoveryApi()
        with ExitStack() as stack:
            own = stack.enter_context(WindowsProcessHandle.pin(api.current_pid(), api=api))
            _serve(stack, own, api, native_calls)
        return 0
    finally:
        sys.setprofile(None)


if __name__ == "__main__":
    raise SystemExit(main())
