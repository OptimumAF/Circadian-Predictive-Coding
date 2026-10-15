"""Bounded codec bytes in replaceable local checkpoint files.

Why stage beside the target: one atomic replace publishes only fully written,
flushed, fsynced and closed bytes. This requires a trusted parent directory,
quiescent input and one cooperating writer. It is not a lock/CAS, hostile-file
race defense, power-loss guarantee or runtime ownership/recovery protocol.
Codec ports own semantic state and returned-state detachment; this infrastructure
owns byte bounds, digest checks and only its temporary file/descriptor cleanup.
"""

from hashlib import sha256
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import BinaryIO, Generic, TypeVar

from src.core.checkpoint_codec import (
    CheckpointCodec,
    CodecBinding,
    CodecLimits,
    require_codec_digest,
)

State = TypeVar("State")


def _controls(binding: CodecBinding, limits: CodecLimits) -> tuple[CodecBinding, CodecLimits]:
    if type(binding) is not CodecBinding or type(limits) is not CodecLimits:
        raise ValueError("checkpoint files require exact CodecBinding and CodecLimits")
    if vars(binding).keys() != {"source_sha256", "policy_sha256"} or vars(limits).keys() != {
        "max_encoded_bytes",
        "max_array_bytes",
        "max_layers",
        "max_dimension",
    }:
        raise ValueError("checkpoint controls differ from their complete supported schema")
    return (
        CodecBinding(binding.source_sha256, binding.policy_sha256),
        CodecLimits(
            limits.max_encoded_bytes,
            limits.max_array_bytes,
            limits.max_layers,
            limits.max_dimension,
        ),
    )


def _target_path(path: str | Path) -> Path:
    if not isinstance(path, (str, Path)):
        raise ValueError("checkpoint path must be a string or Path")
    original = os.fspath(path)
    if "\x00" in original:
        raise ValueError("checkpoint path must not contain a null character")
    filename = os.path.basename(original)
    if not filename or not filename.strip("."):
        raise ValueError("checkpoint path requires a nonempty, non-dot target filename")
    # abspath binds cwd lexically; resolve would perform forbidden constructor IO.
    return Path(os.path.abspath(original))


def _inspect(path: Path, *, missing_ok: bool) -> os.stat_result | None:
    try:
        status = path.lstat()
    except FileNotFoundError:
        if missing_ok:
            return None
        raise
    if not stat.S_ISREG(status.st_mode):
        raise ValueError(f"checkpoint target must be a regular file, not a link/directory: {path}")
    return status


def _identity(status: os.stat_result, *, include_change_time: bool = True) -> tuple[object, ...]:
    names: tuple[str, ...] = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if include_change_time:
        names += ("st_ctime_ns",)
    return tuple(getattr(status, name, None) for name in names)


def _read_size(status: os.stat_result, maximum: int) -> int:
    size = status.st_size
    if not stat.S_ISREG(status.st_mode):
        raise ValueError("opened checkpoint must be a regular file")
    if type(size) is not int or not 0 <= size <= maximum:
        raise ValueError("checkpoint file exceeds the encoded byte bound")
    if size >= sys.maxsize:
        raise ValueError("checkpoint actual size cannot support a bounded size+1 read")
    return size


def _note(error: BaseException, message: str) -> None:
    BaseException.add_note(error, message)


def _cleanup(
    error: BaseException,
    stream: BinaryIO | None,
    descriptor: int | None,
    temporary: Path | None,
) -> None:
    if stream is not None:
        try:
            if not stream.closed:
                stream.close()
        except BaseException as cleanup_error:
            _note(error, f"checkpoint owned-stream cleanup failed: {cleanup_error!r}")
    # Save streams use closefd=False: the store remains the sole raw-fd owner,
    # including when fdopen fails after constructing an internal stream object.
    if descriptor is not None:
        try:
            os.close(descriptor)
        except BaseException as cleanup_error:
            _note(
                error, f"checkpoint owned descriptor {descriptor} cleanup failed: {cleanup_error!r}"
            )
    if temporary is not None:
        try:
            temporary.unlink(missing_ok=True)
        except BaseException as cleanup_error:
            _note(
                error, f"checkpoint orphan temporary may remain at {temporary}: {cleanup_error!r}"
            )


class BoundedCheckpointFileStore(Generic[State]):
    """Transport trusted-codec state using independently supplied bounds/bindings."""

    def __init__(self, path: str | Path, codec: CheckpointCodec[State]) -> None:
        self._path = _target_path(path)
        self._codec = codec

    @property
    def path(self) -> Path:
        return self._path

    def save(self, state: State, *, binding: CodecBinding, limits: CodecLimits) -> str:
        owned_binding, owned_limits = _controls(binding, limits)
        maximum = owned_limits.max_encoded_bytes
        payload = self._codec.encode(state, binding=owned_binding, limits=owned_limits)
        if type(payload) is not bytes or len(payload) > maximum:
            raise ValueError("checkpoint encoder must return exact bounded bytes")
        digest = sha256(payload).hexdigest()
        stream: BinaryIO | None = None
        descriptor: int | None = None
        temporary: Path | None = None
        try:
            _inspect(self._path, missing_ok=True)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(
                prefix=f".{self._path.name}.", suffix=".tmp", dir=self._path.parent
            )
            temporary = Path(name)
            # Buffer construction can close an internal raw stream and then
            # raise. closefd=False keeps that from closing/recycling our fd.
            stream = os.fdopen(descriptor, "wb", closefd=False)
            written = stream.write(payload)
            if type(written) is not int or written != len(payload):
                raise OSError(
                    "checkpoint temporary write did not write the complete exact byte count"
                )
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            stream = None
            owned_descriptor = descriptor
            # A raw close failure has an ambiguous ownership outcome. Clear
            # before attempting it, so cleanup never retries a recycled number.
            descriptor = None
            os.close(owned_descriptor)
            _inspect(self._path, missing_ok=True)
            os.replace(temporary, self._path)
            # Successful replacement is the last fallible operation. No cleanup
            # after commit can misreport a successful publication as a failure.
            return digest
        except BaseException as error:
            _note(error, f"checkpoint save failed for {self._path}")
            _cleanup(error, stream, descriptor, temporary)
            raise

    def _read(self, maximum: int) -> bytes:
        stream: BinaryIO | None = None
        try:
            admitted = _inspect(self._path, missing_ok=False)
            assert admitted is not None
            size = _read_size(admitted, maximum)
            stream = self._path.open("rb")
            opened = os.fstat(stream.fileno())
            _read_size(opened, maximum)
            # Windows path-stat ctime can report creation time while fd-stat
            # reports metadata change time (observed on Python 3.14). Compare
            # only comparable fields across APIs, retaining full fd-stat ctime
            # comparison below to detect changes during the bounded read.
            cross_api_ctime = os.name != "nt"
            if _identity(opened, include_change_time=cross_api_ctime) != _identity(
                admitted, include_change_time=cross_api_ctime
            ):
                raise ValueError("checkpoint changed between inspection and open")
            payload = stream.read(size + 1)
            after = os.fstat(stream.fileno())
            _read_size(after, maximum)
            if (
                type(payload) is not bytes
                or len(payload) != size
                or _identity(after) != _identity(opened)
            ):
                raise ValueError("checkpoint read was short, oversized or changed")
            stream.close()
            stream = None
            return payload
        except BaseException as error:
            _note(error, f"checkpoint load failed for {self._path}")
            _cleanup(error, stream, None, None)
            raise

    def load(self, *, binding: CodecBinding, expected_sha256: str, limits: CodecLimits) -> State:
        owned_binding, owned_limits = _controls(binding, limits)
        require_codec_digest(expected_sha256)
        maximum = owned_limits.max_encoded_bytes
        payload = self._read(maximum)
        if sha256(payload).hexdigest() != expected_sha256:
            raise ValueError(
                f"checkpoint content differs from independently expected digest: {self._path}"
            )
        return self._codec.decode(
            payload, binding=owned_binding, expected_sha256=expected_sha256, limits=owned_limits
        )
