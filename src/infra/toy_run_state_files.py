"""Exclusively claim and atomically advance a local toy CLI run state.

Inputs are finite JSON records and a local path. This module owns only file
integrity and cooperative single-writer exclusion; it does not train, score,
or decide whether a checkpoint is scientifically compatible.
"""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Iterator, Mapping


class ToyRunStateFile:
    """Create once, then replace only the exact state this process read."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_name(f"{self.path.name}.lock")
        self._digest: str | None = None

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        """Refuse a second local writer, including a stale interrupted lock."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        acquired = False
        try:
            with self.lock_path.open("x", encoding="utf-8") as lock:
                acquired = True
                lock.write("toy run state writer\n")
                lock.flush()
                os.fsync(lock.fileno())
            yield
        finally:
            if acquired:
                self.lock_path.unlink(missing_ok=True)

    def create(self, record: Mapping[str, object]) -> None:
        """Claim the state path before starting any training work."""
        payload = _encode(record)
        with self.path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        self._digest = sha256(payload).hexdigest()

    def load(self) -> dict[str, object]:
        """Read a finite object record and remember its exact byte identity."""
        try:
            raw = self.path.read_bytes()
            record = json.loads(raw, parse_constant=_reject_constant)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"toy run state cannot be read: {self.path}") from exc
        if type(record) is not dict:
            raise ValueError("toy run state must contain a JSON object")
        self._digest = sha256(raw).hexdigest()
        return record

    def replace(self, record: Mapping[str, object]) -> None:
        """Atomically advance the file only if its prior bytes still match."""
        if self._digest is None:
            raise ValueError("toy run state must be created or loaded before replacement")
        try:
            current = self.path.read_bytes()
        except OSError as exc:
            raise ValueError(f"toy run state cannot be read: {self.path}") from exc
        if sha256(current).hexdigest() != self._digest:
            raise ValueError("toy run state changed during execution")
        payload = _encode(record)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, self.path)
        finally:
            Path(temporary_path).unlink(missing_ok=True)
        self._digest = sha256(payload).hexdigest()


def _encode(record: Mapping[str, object]) -> bytes:
    return (json.dumps(record, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _reject_constant(value: str) -> object:
    raise ValueError(f"nonfinite toy run state token: {value}")
