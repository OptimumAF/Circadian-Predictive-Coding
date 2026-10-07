"""Live current-process runtime proof adapter with import/code-generation denial.

Freeze every observed Python/native code binding while the full V2/native owner
context is active. Audit hooks become inactive after exit (Python cannot remove
them). This binds actual observations; original-version attestation and scientific
admission remain separate. No source/model/RNG/array/final work occurs here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
import sys
from time import monotonic
from typing import Any
from uuid import uuid4

from src.app.continual_confirmation_json import same_json
from src.app.prospective_runtime_closure import validate_runtime_entrypoints
from src.core.prospective_generation_bundles import GenerationBundleSnapshot
from src.core.prospective_generation_ownership import (
    generation_ownership_scope,
    generation_ownership_utc,
)
from src.core.prospective_runtime_closure import LiveRuntimeCodeLease, RuntimeCodeObservation
from src.core.seed_stream_screening import EvidenceIdentity
from src.infra.runtime_native_images import WindowsRuntimeImages
from src.infra.runtime_python_objects import (
    bind_runtime_entrypoints,
    capture_python_runtime,
    retain_python_executable_objects,
)


class _ProcessRuntimeLease:
    def __init__(
        self,
        observer: ProcessGenerationRuntimeObserver,
        snapshot: GenerationBundleSnapshot,
        entrypoints: tuple[str, ...],
    ) -> None:
        self._observer, self._scope, self._entrypoints = (
            observer,
            generation_ownership_scope(snapshot),
            entrypoints,
        )
        self._nonce, self._sequence, self._last_utc = uuid4().hex, 0, snapshot.observed_utc
        self._baseline: str | None = None
        self._active = True

    def observe(self, snapshot: GenerationBundleSnapshot) -> RuntimeCodeObservation:
        if not self._active or not self._observer._active:
            raise RuntimeError("runtime code lease is inactive")
        same_json(
            asdict(self._scope),
            asdict(generation_ownership_scope(snapshot)),
            "complete live runtime scope",
        )
        started = monotonic()
        python = capture_python_runtime()
        python_done = monotonic()
        native = self._observer._native.capture()
        native_done = monotonic()
        body = {
            "schema_id": "p67_observed_process_runtime_v1",
            "python": python,
            "native": native,
            "entrypoints": bind_runtime_entrypoints(self._entrypoints),
            "source_attestation": False,
        }
        raw = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        self._observer.phase_timings.append(
            {
                "python_seconds": python_done - started,
                "native_seconds": native_done - python_done,
                "serialization_seconds": monotonic() - native_done,
                "python_modules": len(python["modules"]),
                "python_functions": len(python["functions"]),
                "native_images": len(native["images"]),
                "executable_regions": len(native["executable_regions"]),
                "complete_runtime_bytes": len(raw.encode()),
            }
        )
        if self._baseline is not None and raw != self._baseline:
            raise ValueError(
                "complete runtime code/object/file/native membership drifted while held"
            )
        now = self._observer._now()
        if generation_ownership_utc(now) < generation_ownership_utc(self._last_utc):
            raise ValueError("runtime observation clock moved backward")
        result = RuntimeCodeObservation(
            self._scope,
            os.getpid(),
            self._nonce,
            self._sequence,
            now,
            self._entrypoints,
            raw,
            EvidenceIdentity(len(raw.encode()), sha256(raw.encode()).hexdigest()),
        )
        self._baseline, self._last_utc = raw, now
        self._sequence += 1
        return result


class ProcessGenerationRuntimeObserver:
    """Trusted observer enumerates actual process state, never a declared subset."""

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._native = WindowsRuntimeImages()
        self._clock = clock if clock is not None else lambda: datetime.now(timezone.utc)
        self._active = False
        self._installed = False
        self.phase_timings: list[dict[str, Any]] = []

    def _now(self) -> str:
        value = self._clock()
        if type(value) is not datetime or value.tzinfo != timezone.utc:
            raise ValueError("runtime clock must return an aware UTC datetime")
        result = value.isoformat()
        generation_ownership_utc(result)
        return result

    def _audit(self, event: str, arguments: tuple[Any, ...]) -> None:
        if not self._active:
            return
        if event in {
            "import",
            "exec",
            "compile",
            "code.__new__",
            "function.__new__",
            "ctypes.dlopen",
            "ctypes.dlsym",
            "sys.settrace",
            "sys.setprofile",
        } or (event == "object.__setattr__" and len(arguments) >= 2 and arguments[1] == "__code__"):
            raise RuntimeError(f"runtime freeze denied executable drift before execution: {event}")

    @contextmanager
    def freeze(
        self, snapshot: GenerationBundleSnapshot, entrypoints: tuple[str, ...]
    ) -> Iterator[LiveRuntimeCodeLease]:
        generation_ownership_scope(snapshot)
        validate_runtime_entrypoints(entrypoints)
        bind_runtime_entrypoints(entrypoints)
        if self._active:
            raise RuntimeError("runtime observer already has a live freeze")
        if not self._installed:
            sys.addaudithook(self._audit)
            self._installed = True
        lease = _ProcessRuntimeLease(self, snapshot, entrypoints)
        retained = retain_python_executable_objects()
        self._active = True
        try:
            yield lease
        finally:
            self._active = False
            lease._active = False
            # Keep baseline objects alive until after all final checks/releases.
            del retained
