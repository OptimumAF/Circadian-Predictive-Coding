"""Observe complete V2 request ownership with a permanent local native lock file.

Inputs are trusted bundle/registry roots and an aware UTC clock. Outputs are held
leases and exact point-in-time observations. Matching files bind metadata only;
this adapter never executes code/source/arrays/models, claims runtime closure or
cross-host ownership, resolves prior usage/resource/repeat, or authorizes science.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import import_module
import logging
import os
from pathlib import Path
import stat
from typing import IO
from uuid import uuid4

from src.app.continual_confirmation_json import same_json
from src.core.prospective_generation_bundles import GenerationBundleSnapshot
from src.core.prospective_generation_ownership import (
    GenerationOwnershipObservation,
    LiveGenerationRequestLease,
    generation_owner_lock_name,
    generation_ownership_scope,
    generation_ownership_utc,
)
from src.infra.prospective_generation_bundles import FileProspectiveGenerationBundleReader

LOGGER = logging.getLogger(__name__)


def _deny_links(path: Path) -> None:
    if any(
        part.is_symlink() or getattr(part, "is_junction", lambda: False)()
        for part in (path, *path.parents)
    ):
        raise ValueError("generation owner registry cannot use symlinks or junctions")


def _native_lock(stream: IO[bytes], *, release: bool = False) -> None:
    stream.seek(0)
    # Why this: optional standard platform modules are loaded only in infra;
    # existing V14 locking/source closure stays unchanged on both platforms.
    if os.name == "nt":
        native = import_module("msvcrt")
        native.locking(stream.fileno(), native.LK_UNLCK if release else native.LK_NBLCK, 1)
    else:
        native = import_module("fcntl")
        native.flock(
            stream.fileno(), native.LOCK_UN if release else native.LOCK_EX | native.LOCK_NB
        )


class _NativeGenerationLease:
    def __init__(
        self,
        owner: FileGenerationRequestOwner,
        stream: IO[bytes],
        snapshot: GenerationBundleSnapshot,
    ) -> None:
        self._owner, self._stream = owner, stream
        self._scope = generation_ownership_scope(snapshot)
        self._name = generation_owner_lock_name(self._scope)
        self._physical = os.fstat(stream.fileno())
        self._nonce = uuid4().hex  # Non-scientific handle identity; never a model/source seed.
        self._acquired = owner._now()
        self._last_utc = self._acquired
        self._active, self._sequence = True, 0

    def observe(self, snapshot: GenerationBundleSnapshot) -> GenerationOwnershipObservation:
        if not self._active:
            raise RuntimeError("generation owner lease is inactive")
        scope = generation_ownership_scope(snapshot)
        same_json(asdict(scope), asdict(self._scope), "complete live generation owner scope")
        self._owner._check_lock(self._stream, self._name, self._physical)
        try:
            self._owner._files.recheck_bundle(snapshot)
        finally:
            self._owner._check_lock(self._stream, self._name, self._physical)
        now = self._owner._now()
        if generation_ownership_utc(now) < generation_ownership_utc(self._last_utc):
            raise ValueError("generation owner observation clock moved backward")
        result = GenerationOwnershipObservation(
            scope,
            self._owner._registry.as_posix(),
            self._name,
            self._physical.st_dev,
            self._physical.st_ino,
            self._nonce,
            self._acquired,
            now,
            self._sequence,
        )
        self._sequence += 1
        self._last_utc = now
        return result

    def close(self) -> None:
        self._active = False


class FileGenerationRequestOwner:
    """Cooperative local owners share a permanent request key in one registry."""

    def __init__(
        self, bundle_root: Path, registry_root: Path, *, clock: Callable[[], datetime] | None = None
    ) -> None:
        if not registry_root.is_absolute() or ".." in registry_root.parts:
            raise ValueError("generation owner registry must be canonical and absolute")
        _deny_links(registry_root)
        self._registry = registry_root.resolve(strict=True)
        if not self._registry.is_dir():
            raise ValueError("generation owner registry must be an existing directory")
        root = self._registry.stat()
        self._registry_identity = (root.st_dev, root.st_ino)
        self._files = FileProspectiveGenerationBundleReader(bundle_root)
        self._clock = clock if clock is not None else lambda: datetime.now(timezone.utc)

    def _now(self) -> str:
        now = self._clock()
        if type(now) is not datetime or now.tzinfo != timezone.utc:
            raise ValueError("generation owner clock must return an aware UTC datetime")
        value = now.isoformat()
        generation_ownership_utc(value)
        return value

    def _check_registry(self) -> None:
        _deny_links(self._registry)
        root = self._registry.stat()
        if not stat.S_ISDIR(root.st_mode) or (root.st_dev, root.st_ino) != self._registry_identity:
            raise ValueError("generation owner registry directory changed")

    def _open_lock(self, name: str) -> IO[bytes]:
        self._check_registry()
        path = self._registry / name
        _deny_links(path)
        if path.exists() and not path.is_file():
            raise ValueError("generation owner lock path must be regular")
        flags = os.O_RDWR | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            descriptor = os.open(path, flags)
        else:
            try:
                os.write(descriptor, b"0")
                os.fsync(descriptor)
            except BaseException:
                os.close(descriptor)
                raise
        # Why this: never unlink an existing lock pathname. A second inode at the
        # same logical key could otherwise let two cooperative owners proceed.
        return os.fdopen(descriptor, "r+b", buffering=0)

    def _check_lock(self, stream: IO[bytes], name: str, expected: os.stat_result) -> None:
        self._check_registry()
        path = self._registry / name
        _deny_links(path)
        actual = path.stat()
        handle = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(actual.st_mode)
            or actual.st_nlink != 1
            or handle.st_nlink != 1
            or (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino)
            or (handle.st_dev, handle.st_ino) != (expected.st_dev, expected.st_ino)
        ):
            raise ValueError("generation owner lock physical identity or alias count changed")
        stream.seek(0)
        if actual.st_size != 1 or stream.read(2) != b"0":
            raise ValueError("generation owner permanent lock bytes changed")

    @contextmanager
    def claim(self, snapshot: GenerationBundleSnapshot) -> Iterator[LiveGenerationRequestLease]:
        scope = generation_ownership_scope(snapshot)
        self._files.recheck_bundle(snapshot)
        name = generation_owner_lock_name(scope)
        try:
            stream = self._open_lock(name)
        except OSError as error:
            raise ValueError("generation owner registry file is unavailable") from error
        with stream:
            try:
                _native_lock(stream)
            except OSError as error:
                raise RuntimeError("generation request owner is already active") from error
            lease = None
            try:
                self._check_lock(stream, name, os.fstat(stream.fileno()))
                lease = _NativeGenerationLease(self, stream, snapshot)
                LOGGER.info(
                    "generation_request_owner_acquired",
                    extra={"generation_request_sha256": scope.generation_request_identity.sha256},
                )
                yield lease
            finally:
                if lease is not None:
                    lease.close()
                _native_lock(stream, release=True)
                LOGGER.info(
                    "generation_request_owner_released",
                    extra={"generation_request_sha256": scope.generation_request_identity.sha256},
                )
