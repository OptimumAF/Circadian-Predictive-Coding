"""Finite entry measurement for disposable test subprocesses.

Inputs: original raw Python functions/C callables, labels, limits and a fresh
breach receipt path. Outputs: scalar entry counts or process exit73 on breach.
This preserves dispatch identities and covers the main thread and future workers.
It does not cover pre-existing workers, argument evaluation or work dispatched
inside trusted prior profile callbacks (Python suppresses profiling there).
No model, provider or scheduler is replaced; this is measurement, not a sandbox.
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import threading
from types import BuiltinFunctionType, CodeType, FrameType, FunctionType
from typing import Callable, Iterator, NoReturn

_EXIT = os._exit
_SET_PROFILE = sys.setprofile


class ProviderEntryProfile:
    """One-use, identity-bound counters; prior hooks must be trusted observers."""

    def __init__(
        self,
        entries: tuple[tuple[Callable, str, int], ...],
        receipt: Path,
    ) -> None:
        if not 0 < len(entries) <= 64:
            raise ValueError("entry profile requires one to64 bindings")
        table: list[tuple[object, str, int, str]] = []
        for function, label, limit in entries:
            if (
                type(label) is not str
                or not label.isascii()
                or not label.isidentifier()
                or len(label) > 64
            ):
                raise ValueError(
                    "entry counter label must be an ASCII identifier of <=64 characters"
                )
            if type(limit) is not int or not 0 <= limit <= 1_000_000:
                raise ValueError("entry limit must be an integer in0..1000000")
            token: object
            if type(function) is FunctionType:
                token, event = function.__code__, "call"
            elif type(function) is BuiltinFunctionType:
                token, event = function, "c_call"
            else:
                raise TypeError("entry binding requires a raw Python function or C callable")
            if any(token is prior or label == key for prior, key, _, _ in table):
                raise ValueError("entry bindings require distinct identities and labels")
            table.append((token, label, limit, event))
        self._table = tuple(table)
        self._counts = dict.fromkeys((row[1] for row in table), 0)
        self._receipt = receipt
        self._lock = threading.Lock()
        self._callback = self._profile
        self._used = self._active = False
        self._main = threading.get_ident()
        self._prior_main = sys.getprofile()
        self._prior_thread = threading.getprofile()
        self._profile_setters: tuple[CodeType, ...] = tuple(
            function.__code__
            for function in (
                threading.setprofile,
                getattr(threading, "setprofile_all_threads", None),
            )
            if type(function) is FunctionType
        )

    @property
    def counts(self) -> dict[str, int]:
        with self._lock:
            return self._counts.copy()

    def _abort(
        self, reason: str, label: str = "profile", attempt: int = 0, limit: int = 0
    ) -> NoReturn:
        # Why process exit: callback exceptions detach profiling and may be caught
        # by the original driver. Exit also ends every other worker in this child.
        try:
            raw = json.dumps(
                dict(reason=reason, counter=label, attempt=attempt, limit=limit)
            ).encode()
            if len(raw) <= 1024:
                with self._receipt.open("xb") as stream:
                    stream.write(raw)
        finally:
            _EXIT(73)

    def _profile(self, frame: FrameType, event: str, arg: object) -> None:
        try:
            if self._active:
                if (event == "c_call" and arg is _SET_PROFILE) or (
                    event == "call" and any(frame.f_code is code for code in self._profile_setters)
                ):
                    self._abort("profile_mutation")
                token = frame.f_code if event == "call" else arg
                for original, label, limit, kind in self._table:
                    if event == kind and token is original:
                        with self._lock:
                            self._counts[label] += 1
                            attempt = self._counts[label]
                        if attempt > limit:
                            self._abort("entry_limit", label, attempt, limit)
                        break
            prior = self._prior_main if threading.get_ident() == self._main else self._prior_thread
            if prior is not None:
                try:
                    prior(frame, event, arg)
                except BaseException:
                    self._abort("prior_profile_failure")
        except BaseException:
            self._abort("measurement_failure")

    @contextmanager
    def measure(self) -> Iterator[None]:
        if self._used:
            raise RuntimeError("entry profile cannot renew its allowance")
        if (
            threading.current_thread() is not threading.main_thread()
            or threading.active_count() != 1
        ):
            raise RuntimeError("entry profile requires main thread with no existing workers")
        self._used = True
        self._main = threading.get_ident()
        self._prior_main = sys.getprofile()
        self._prior_thread = threading.getprofile()
        try:
            threading.setprofile(self._callback)
            _SET_PROFILE(self._callback)
        except BaseException:
            self._abort("profile_install_failure")
        self._active = True
        try:
            yield
        finally:
            if threading.active_count() != 1:
                self._abort("unjoined_worker")
            if (
                sys.getprofile() is not self._callback
                or threading.getprofile() is not self._callback
            ):
                self._abort("profile_coverage_lost")
            self._active = False
            try:
                _SET_PROFILE(self._prior_main)
                threading.setprofile(self._prior_thread)
            except BaseException:
                self._abort("profile_restore_failure")
