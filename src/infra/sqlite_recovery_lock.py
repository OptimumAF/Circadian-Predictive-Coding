"""Permanent one-byte private journal writer lock; no authority or model logic."""

from contextlib import contextmanager
from importlib import import_module
import os
from pathlib import Path
import sqlite3
from typing import IO, Iterator


def _native_lock(stream: IO[bytes], *, release: bool = False) -> None:
    stream.seek(0)
    if os.name == "nt":
        native = import_module("msvcrt")
        native.locking(stream.fileno(), native.LK_UNLCK if release else native.LK_NBLCK, 1)
    elif os.name == "posix":
        native = import_module("fcntl")
        native.flock(
            stream.fileno(), native.LOCK_UN if release else native.LOCK_EX | native.LOCK_NB
        )
    else:
        raise ValueError("private authority writer lock unsupported on this platform")


@contextmanager
def authority_writer_lock(path: Path) -> Iterator[None]:
    """Nonblocking lock shared by all supported adapters on the canonical private path.

    Why this: publication must retain ownership while SQLite observation reports
    commit. Never unlink this permanent file: replacing it would split ownership.
    """
    lock_path = path.with_name(path.name + ".writer.lock")
    if lock_path.is_symlink():
        raise ValueError("authority writer lock cannot be a symlink")
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "r+b") as stream:
        try:
            _native_lock(stream)
        except OSError as error:
            raise sqlite3.OperationalError("authority writer locked") from error
        try:
            size = os.fstat(stream.fileno()).st_size
            if size > 1:
                raise ValueError("authority writer lock exceeds its one-byte bound")
            if size == 0:
                stream.write(b"1")
                stream.flush()
            yield
        finally:
            _native_lock(stream, release=True)
