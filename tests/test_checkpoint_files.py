"""Bounded byte transport and atomic replacement under native IO faults.

The injected codec owns state semantics. File tests exercise OS boundaries and
count their dedicated fixture files; only six controls execute native training.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import os
from pathlib import Path
import pickle
import stat
from typing import Any

import numpy as np
import pytest

import src.infra.checkpoint_files as files
from src.adapters.coupled_checkpoint_codec import CoupledCheckpointCodec
from src.core.checkpoint_codec import CodecBinding, CodecLimits
from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.infra.checkpoint_files import BoundedCheckpointFileStore
from test_coupled_checkpoint_codec import arrays, assert_same, make_model

BINDING = CodecBinding("a" * 64, "b" * 64)
LIMITS = CodecLimits(32768, 8192, 8, 64)
DISK_WORK = {"created_files": 0, "payload_bytes": 0}


class Stream:
    def __init__(self, native):
        self.native = native

    def __getattr__(self, name):
        return getattr(self.native, name)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        return self.native.close()


@pytest.fixture(autouse=True)
def count_owned_file_work(monkeypatch, tmp_path):
    native_write, native_temp, native_fdopen, native_link = (
        Path.write_bytes,
        files.tempfile.mkstemp,
        files.os.fdopen,
        Path.symlink_to,
    )
    owned_fds = set()

    def direct_write(path, raw):
        existed = path.exists()
        result = native_write(path, raw)
        if path.is_relative_to(tmp_path):
            DISK_WORK["created_files"] += int(not existed)
            DISK_WORK["payload_bytes"] += len(raw)
        return result

    def temporary(*args, **kwargs):
        fd, name = native_temp(*args, **kwargs)
        if Path(name).is_relative_to(tmp_path):
            DISK_WORK["created_files"] += 1
            owned_fds.add(fd)
        return fd, name

    class CountWrite(Stream):
        def write(self, raw):
            DISK_WORK["payload_bytes"] += len(raw)
            return self.native.write(raw)

    def fdopen(fd, *args, **kwargs):
        stream = native_fdopen(fd, *args, **kwargs)
        return CountWrite(stream) if fd in owned_fds else stream

    def link(path, *args, **kwargs):
        result = native_link(path, *args, **kwargs)
        if path.is_relative_to(tmp_path):
            DISK_WORK["created_files"] += 1
        return result

    monkeypatch.setattr(Path, "write_bytes", direct_write)
    monkeypatch.setattr(files.tempfile, "mkstemp", temporary)
    monkeypatch.setattr(files.os, "fdopen", fdopen)
    monkeypatch.setattr(Path, "symlink_to", link)
    yield
    assert DISK_WORK["created_files"] <= 384
    assert DISK_WORK["payload_bytes"] <= 12 * 1024**2


class Codec:
    def __init__(self, raw=b"new\x00bytes"):
        self.raw = raw
        self.encodes = []
        self.decodes = []
        self.before_encode = None
        self.before_decode = None
        self.decoded = object()

    def encode(self, state, *, binding, limits):
        self.encodes.append((state, binding, limits))
        if self.before_encode is not None:
            self.before_encode(state, binding, limits)
        return self.raw

    def decode(self, raw, *, binding, expected_sha256, limits):
        self.decodes.append((raw, binding, expected_sha256, limits))
        if self.before_decode is not None:
            self.before_decode(raw, binding, expected_sha256, limits)
        return self.decoded


def save(store, state=None, **options):
    return store.save(
        state,
        binding=options.pop("binding", BINDING),
        limits=options.pop("limits", LIMITS),
        **options,
    )


def load(store, raw=b"new\x00bytes", **options):
    return store.load(
        binding=options.pop("binding", BINDING),
        expected_sha256=options.pop("expected_sha256", sha256(raw).hexdigest()),
        limits=options.pop("limits", LIMITS),
        **options,
    )


@pytest.mark.parametrize("raw", (b"", b"a\x00b", bytes(range(256))))
def test_should_transport_exact_opaque_bytes_and_inclusive_caps_without_pickle(
    monkeypatch, tmp_path, raw
):
    codec = Codec(raw)
    store = BoundedCheckpointFileStore(tmp_path / "state.bin", codec)
    limits = replace(LIMITS, max_encoded_bytes=max(1, len(raw)))
    state = object()
    monkeypatch.setattr(pickle, "dumps", lambda *args, **kwargs: pytest.fail("pickle encode"))
    monkeypatch.setattr(pickle, "loads", lambda *args, **kwargs: pytest.fail("pickle decode"))
    digest = save(store, state, limits=limits)
    assert digest == sha256(raw).hexdigest()
    assert store.path.read_bytes() == raw
    assert load(store, raw, limits=limits) is codec.decoded
    assert codec.encodes[0][0] is state
    assert codec.decodes[0][0] == raw and codec.decodes[0][2] == digest


def test_should_replace_regular_file_and_bind_relative_path_at_construction(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    codec = Codec(b"first")
    store = BoundedCheckpointFileStore("nested/state.bin", codec)
    expected = tmp_path / "nested" / "state.bin"
    assert store.path == expected and store.path.is_absolute()
    with pytest.raises(AttributeError):
        setattr(store, "path", tmp_path / "other.bin")
    save(store)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    codec.raw = b"second"
    save(store)
    assert store.path == expected and load(store, b"second") is codec.decoded
    assert expected.read_bytes() == b"second"


@pytest.mark.parametrize("path", ("", ".", "..", Path("."), Path(".."), b"state.bin", None, True))
def test_should_refuse_invalid_or_dot_only_target_paths(path):
    with pytest.raises(ValueError):
        BoundedCheckpointFileStore(path, Codec())


def test_should_capture_lexical_absolute_path_without_any_filesystem_inspection(monkeypatch):
    expected = Path(os.path.abspath("parent/state.bin"))

    def forbidden(*args, **kwargs):
        pytest.fail("constructor performed IO or resolution")

    for name in ("resolve", "stat", "lstat", "exists", "mkdir", "open"):
        monkeypatch.setattr(Path, name, forbidden)
    monkeypatch.setattr(files.tempfile, "mkstemp", forbidden)
    store = BoundedCheckpointFileStore("parent/state.bin", Codec())
    assert store.path == expected


@pytest.mark.parametrize(
    "fault",
    (
        "binding",
        "limits",
        "mutated_binding",
        "mutated_limits",
        "digest_bool",
        "digest_upper",
        "digest_short",
    ),
)
def test_should_validate_owned_public_controls_before_io_or_codec(monkeypatch, tmp_path, fault):
    codec = Codec()
    store = BoundedCheckpointFileStore(tmp_path / "absent.bin", codec)
    binding: Any = deepcopy(BINDING)
    limits: Any = deepcopy(LIMITS)
    digest: Any = "a" * 64
    if fault == "binding":
        binding = asdict(binding)
    elif fault == "limits":
        limits = asdict(limits)
    elif fault == "mutated_binding":
        object.__setattr__(binding, "source_sha256", True)
    elif fault == "mutated_limits":
        object.__setattr__(limits, "max_encoded_bytes", True)
    else:
        digest = {"digest_bool": True, "digest_upper": "A" * 64, "digest_short": "a" * 63}[fault]
    monkeypatch.setattr(Path, "lstat", lambda *args: pytest.fail("IO before admission"))
    with pytest.raises(ValueError):
        store.load(binding=binding, limits=limits, expected_sha256=digest)
    if not fault.startswith("digest"):
        with pytest.raises(ValueError):
            store.save(object(), binding=binding, limits=limits)
    assert codec.encodes == codec.decodes == []


@pytest.mark.parametrize("present", (False, True))
@pytest.mark.parametrize(
    "fault", ("bytearray", "text", "none", "oversize", "codec_error", "codec_cancel")
)
def test_should_encode_before_mutation_and_preserve_target_on_encoder_refusal(
    tmp_path, present, fault
):
    target = tmp_path / "nested" / "state.bin"
    if present:
        target.parent.mkdir()
        target.write_bytes(b"old")
    codec = Codec()
    error: BaseException | None = None
    if fault in ("codec_error", "codec_cancel"):
        error = (
            RuntimeError("encoder fault")
            if fault == "codec_error"
            else KeyboardInterrupt("encoder cancellation")
        )

        def fail(*args):
            raise error

        codec.before_encode = fail
    else:
        codec.raw = {
            "bytearray": bytearray(b"new"),
            "text": "new",
            "none": None,
            "oversize": b"x" * 9,
        }[fault]
    store = BoundedCheckpointFileStore(target, codec)
    with pytest.raises(type(error) if error is not None else ValueError) as captured:
        save(store, limits=replace(LIMITS, max_encoded_bytes=8))
    if error is not None:
        assert captured.value is error
    assert target.exists() is present
    if present:
        assert target.read_bytes() == b"old" and list(target.parent.iterdir()) == [target]
    else:
        assert not target.parent.exists()


def test_should_capture_immutable_enforcement_bound_before_codec_can_relax_private_limits(tmp_path):
    codec = Codec(b"123456789")
    binding, limits = deepcopy(BINDING), replace(LIMITS, max_encoded_bytes=8)

    def relax(state, owned_binding, owned_limits):
        assert owned_binding is not binding and owned_limits is not limits
        object.__setattr__(owned_limits, "max_encoded_bytes", 1000)
        object.__setattr__(owned_binding, "source_sha256", "c" * 64)

    codec.before_encode = relax
    store = BoundedCheckpointFileStore(tmp_path / "absent" / "state.bin", codec)
    with pytest.raises(ValueError):
        save(store, binding=binding, limits=limits)
    assert limits.max_encoded_bytes == 8 and binding == BINDING
    assert not store.path.parent.exists()


def test_should_own_controls_in_both_codec_directions_and_close_before_decode(
    tmp_path, monkeypatch
):
    codec = Codec()
    store = BoundedCheckpointFileStore(tmp_path / "state.bin", codec)
    binding, limits = deepcopy(BINDING), deepcopy(LIMITS)
    save(store, binding=binding, limits=limits)
    opened = []
    native_open = Path.open

    def track(path, *args, **kwargs):
        stream = native_open(path, *args, **kwargs)
        opened.append(stream)
        return stream

    monkeypatch.setattr(Path, "open", track)

    def decode_callback(raw, owned_binding, digest, owned_limits):
        assert opened[-1].closed
        assert owned_binding is not binding and owned_limits is not limits
        object.__setattr__(owned_binding, "policy_sha256", "c" * 64)
        object.__setattr__(owned_limits, "max_encoded_bytes", 1)

    codec.before_decode = decode_callback
    assert load(store, binding=binding, limits=limits) is codec.decoded
    assert binding == BINDING and limits == LIMITS


class WriteFault(Stream):
    def __init__(self, native, fault, error):
        super().__init__(native)
        self.fault, self.error = fault, error

    def write(self, raw):
        if self.fault in ("write", "cancel"):
            raise self.error
        if self.fault == "short":
            return self.native.write(raw[:-1])
        result = self.native.write(raw)
        return {"none": None, "bool": True, "extra": result + 1, "numpyint": np.int64(result)}.get(
            self.fault, result
        )

    def flush(self):
        if self.fault == "flush":
            raise self.error
        return self.native.flush()

    def close(self):
        self.native.close()
        if self.fault == "close":
            raise self.error


@pytest.mark.parametrize("present", (False, True))
@pytest.mark.parametrize(
    "fault",
    (
        "short",
        "none",
        "bool",
        "extra",
        "numpyint",
        "write",
        "flush",
        "fsync",
        "close",
        "fdopen",
        "replace",
        "cancel",
    ),
)
def test_should_preserve_old_or_absent_target_and_clean_owned_staging_on_precommit_failure(
    monkeypatch, tmp_path, present, fault
):
    target = tmp_path / "state.bin"
    if present:
        target.write_bytes(b"old")
    error = (
        KeyboardInterrupt("native cancellation")
        if fault == "cancel"
        else OSError("injected " + fault)
    )
    native_fdopen = files.os.fdopen
    raw_fds, stages = [], []
    native_temp = files.tempfile.mkstemp

    def temporary(*args, **kwargs):
        fd, name = native_temp(*args, **kwargs)
        raw_fds.append(fd)
        stages.append(Path(name))
        assert Path(name).parent == target.parent
        return fd, name

    def fdopen(fd, *args, **kwargs):
        if fault == "fdopen":
            raise error
        return WriteFault(native_fdopen(fd, *args, **kwargs), fault, error)

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(files.tempfile, "mkstemp", temporary)
    monkeypatch.setattr(files.os, "fdopen", fdopen)
    if fault in ("fsync", "replace"):
        monkeypatch.setattr(files.os, fault, fail)
    expectation = (
        (ValueError, OSError)
        if fault in ("short", "none", "bool", "extra", "numpyint")
        else type(error)
    )
    with pytest.raises(expectation) as captured:
        save(BoundedCheckpointFileStore(target, Codec()))
    if fault not in ("short", "none", "bool", "extra", "numpyint"):
        assert captured.value is error
    assert target.exists() is present
    if present:
        assert target.read_bytes() == b"old"
    assert stages and all(not path.exists() for path in stages)
    for fd in raw_fds:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_should_preserve_primary_error_and_note_owned_orphan_if_cleanup_fails(
    monkeypatch, tmp_path
):
    target = tmp_path / "state.bin"
    target.write_bytes(b"old")
    primary = OSError("primary fsync failure")
    native_unlink = Path.unlink

    def fsync(fd):
        raise primary

    def unlink(path, *args, **kwargs):
        assert path != target and path.parent == tmp_path
        raise PermissionError("staging cleanup denied")

    monkeypatch.setattr(files.os, "fsync", fsync)
    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(OSError) as captured:
        save(BoundedCheckpointFileStore(target, Codec()))
    assert captured.value is primary and target.read_bytes() == b"old"
    orphan = next(path for path in tmp_path.iterdir() if path != target)
    notes = " ".join(getattr(primary, "__notes__", ()))
    assert str(orphan) in notes and ("cleanup" in notes.lower() or "orphan" in notes.lower())
    native_unlink(orphan)


def test_should_not_retry_owned_raw_close_after_error_recycles_the_descriptor(
    monkeypatch, tmp_path
):
    target, unrelated = tmp_path / "state.bin", tmp_path / "unrelated.bin"
    target.write_bytes(b"old")
    unrelated.write_bytes(b"unrelated")
    native_close = files.os.close
    replacement = []
    primary = OSError("close reported failure after releasing descriptor")

    def close_then_recycle(fd):
        native_close(fd)
        replacement.append(os.open(unrelated, os.O_RDONLY))
        assert replacement[-1] == fd
        raise primary

    monkeypatch.setattr(files.os, "close", close_then_recycle)
    try:
        with pytest.raises(OSError) as captured:
            save(BoundedCheckpointFileStore(target, Codec()))
        assert captured.value is primary
        assert replacement and os.fstat(replacement[0]).st_size == len(b"unrelated")
        assert target.read_bytes() == b"old" and unrelated.read_bytes() == b"unrelated"
        assert set(tmp_path.iterdir()) == {target, unrelated}
    finally:
        for fd in set(replacement):
            try:
                native_close(fd)
            except OSError:
                # Preserve the assertion if a broken store already closed this
                # descriptor; cleanup must not hide the ownership failure.
                pass


def test_should_keep_owned_raw_descriptor_when_fdopen_partially_constructs_then_fails(
    monkeypatch, tmp_path
):
    target, unrelated = tmp_path / "state.bin", tmp_path / "unrelated.bin"
    target.write_bytes(b"old")
    unrelated.write_bytes(b"unrelated")
    native_fdopen = files.os.fdopen
    primary = OSError("fdopen buffer construction failed")
    borrowed = os.open(unrelated, os.O_RDONLY)
    owned = []

    def partially_construct(fd, *args, **kwargs):
        assert kwargs["closefd"] is False
        stream = native_fdopen(fd, *args, **kwargs)
        stream.close()
        assert os.fstat(fd).st_size == 0
        owned.append(fd)
        raise primary

    monkeypatch.setattr(files.os, "fdopen", partially_construct)
    try:
        with pytest.raises(OSError) as captured:
            save(BoundedCheckpointFileStore(target, Codec()))
        assert captured.value is primary
        assert os.fstat(borrowed).st_size == len(b"unrelated")
        for fd in owned:
            with pytest.raises(OSError):
                os.fstat(fd)
        assert target.read_bytes() == b"old"
        assert set(tmp_path.iterdir()) == {target, unrelated}
    finally:
        os.close(borrowed)


def test_should_finish_write_flush_fsync_close_before_replace_and_do_no_postcommit_cleanup(
    monkeypatch, tmp_path
):
    target = tmp_path / "state.bin"
    target.write_bytes(b"old")
    trace = []
    native_fdopen, native_fsync, native_close, native_replace = (
        files.os.fdopen,
        files.os.fsync,
        files.os.close,
        files.os.replace,
    )

    class Ordered(Stream):
        def write(self, raw):
            trace.append("write")
            return self.native.write(raw)

        def flush(self):
            trace.append("flush")
            return self.native.flush()

        def close(self):
            trace.append("close")
            return self.native.close()

    def fsync(fd):
        trace.append("fsync")
        return native_fsync(fd)

    def replace_file(source, destination):
        trace.append("replace")
        assert trace == ["write", "flush", "fsync", "close", "raw_close", "replace"]
        return native_replace(source, destination)

    def close(fd):
        trace.append("raw_close")
        return native_close(fd)

    monkeypatch.setattr(
        files.os, "fdopen", lambda fd, *args, **kwargs: Ordered(native_fdopen(fd, *args, **kwargs))
    )
    monkeypatch.setattr(files.os, "fsync", fsync)
    monkeypatch.setattr(files.os, "close", close)
    monkeypatch.setattr(files.os, "replace", replace_file)
    monkeypatch.setattr(
        Path,
        "unlink",
        lambda *args, **kwargs: pytest.fail("fallible cleanup after successful commit"),
    )
    assert save(BoundedCheckpointFileStore(target, Codec())) == sha256(b"new\x00bytes").hexdigest()
    assert target.read_bytes() == b"new\x00bytes"


def stat_with(original, **changes):
    values = dict(
        (name, getattr(original, name)) for name in dir(original) if name.startswith("st_")
    )
    values.update(changes)
    from types import SimpleNamespace

    return SimpleNamespace(**values)


class PlatformOS:
    """Select one store policy branch without changing pathlib's global OS."""

    def __init__(self, name, fstat):
        self.native = files.os
        self.name = name
        self.fstat = fstat

    def __getattr__(self, name):
        return getattr(self.native, name)


@pytest.mark.parametrize("field", ("st_ctime_ns", "st_dev", "st_ino", "st_size", "st_mtime_ns"))
def test_should_accept_only_windows_cross_api_ctime_difference_without_relaxing_identity(
    monkeypatch, tmp_path, field
):
    target = tmp_path / "state.bin"
    target.write_bytes(b"old")
    codec = Codec()
    store = BoundedCheckpointFileStore(target, codec)
    native_fstat = files.os.fstat

    def opened_stat(fd):
        result = native_fstat(fd)
        return stat_with(result, **{field: getattr(result, field) + 1})

    with monkeypatch.context() as local:
        local.setattr(files, "os", PlatformOS("nt", opened_stat))
        if field == "st_ctime_ns":
            assert load(store, b"old") is codec.decoded
        else:
            with pytest.raises(ValueError):
                load(store, b"old")
            assert codec.decodes == []


def test_should_keep_non_windows_cross_api_ctime_comparison_strict(monkeypatch, tmp_path):
    target = tmp_path / "state.bin"
    target.write_bytes(b"old")
    codec = Codec()
    store = BoundedCheckpointFileStore(target, codec)
    native_fstat = files.os.fstat

    def opened_stat(fd):
        result = native_fstat(fd)
        return stat_with(result, st_ctime_ns=result.st_ctime_ns + 1)

    with monkeypatch.context() as local:
        local.setattr(files, "os", PlatformOS("posix", opened_stat))
        with pytest.raises(ValueError):
            load(store, b"old")
    assert codec.decodes == []


def test_should_refuse_ctime_change_between_two_opened_descriptor_stats_on_windows(
    monkeypatch, tmp_path
):
    target = tmp_path / "state.bin"
    target.write_bytes(b"old")
    codec = Codec()
    store = BoundedCheckpointFileStore(target, codec)
    native_fstat = files.os.fstat
    calls = 0

    def opened_stat(fd):
        nonlocal calls
        calls += 1
        result = native_fstat(fd)
        return stat_with(result, st_ctime_ns=result.st_ctime_ns + int(calls == 2))

    with monkeypatch.context() as local:
        local.setattr(files, "os", PlatformOS("nt", opened_stat))
        with pytest.raises(ValueError):
            load(store, b"old")
    assert calls == 2 and codec.decodes == []


@pytest.mark.parametrize(
    "fault",
    (
        "short",
        "long",
        "size_before",
        "identity_before",
        "nonregular_open",
        "size_after",
        "identity_after",
    ),
)
def test_should_bound_actual_read_and_refuse_changed_or_nonregular_opened_file_before_decode(
    monkeypatch, tmp_path, fault
):
    target = tmp_path / "state.bin"
    target.write_bytes(b"old")
    codec = Codec()
    store = BoundedCheckpointFileStore(target, codec)
    native_open, native_fstat = Path.open, files.os.fstat
    reads = []
    stats = 0

    class Reading(Stream):
        def read(self, count):
            reads.append(count)
            assert count == 4
            value = self.native.read(count)
            return value[:-1] if fault == "short" else value + b"x" if fault == "long" else value

    def fstat(fd):
        nonlocal stats
        stats += 1
        result = native_fstat(fd)
        if fault == "nonregular_open":
            return stat_with(result, st_mode=stat.S_IFIFO | 0o600)
        if fault == "size_before" or fault == "size_after" and stats == 2:
            return stat_with(result, st_size=4)
        if fault == "identity_before" or fault == "identity_after" and stats == 2:
            return stat_with(result, st_ino=result.st_ino + 1)
        return result

    monkeypatch.setattr(
        Path, "open", lambda path, *args, **kwargs: Reading(native_open(path, *args, **kwargs))
    )
    monkeypatch.setattr(files.os, "fstat", fstat)
    with pytest.raises((ValueError, OSError)):
        load(store, b"old")
    assert codec.decodes == []
    assert reads == ([] if fault in ("size_before", "identity_before", "nonregular_open") else [4])


def test_should_read_actual_size_plus_one_under_huge_allowance_and_close_before_decode(
    monkeypatch, tmp_path
):
    target = tmp_path / "state.bin"
    target.write_bytes(b"old")
    codec = Codec()
    opened, requested = [], []
    native_open = Path.open

    class Reading(Stream):
        def read(self, count):
            requested.append(count)
            return self.native.read(count)

    def open_file(path, *args, **kwargs):
        stream = Reading(native_open(path, *args, **kwargs))
        opened.append(stream)
        return stream

    monkeypatch.setattr(Path, "open", open_file)
    monkeypatch.setattr(Path, "read_bytes", lambda *args: pytest.fail("unbounded read_bytes"))
    codec.before_decode = lambda *args: (
        None if opened[0].closed else pytest.fail("decode before close")
    )
    assert (
        load(
            BoundedCheckpointFileStore(target, codec),
            b"old",
            limits=replace(LIMITS, max_encoded_bytes=2**63 - 1),
        )
        is codec.decoded
    )
    assert requested == [4] and opened[0].closed


@pytest.mark.parametrize("fault", ("oversize", "digest", "corrupt"))
def test_should_refuse_size_or_digest_before_semantic_decoder(monkeypatch, tmp_path, fault):
    target = tmp_path / "state.bin"
    target.write_bytes(b"old" if fault != "corrupt" else b"bad")
    codec = Codec()
    if fault == "oversize":
        monkeypatch.setattr(
            Path, "open", lambda *args, **kwargs: pytest.fail("open before size bound")
        )
    options = (
        {"limits": replace(LIMITS, max_encoded_bytes=2)}
        if fault == "oversize"
        else {"expected_sha256": "c" * 64}
        if fault == "digest"
        else {}
    )
    with pytest.raises(ValueError):
        load(BoundedCheckpointFileStore(target, codec), b"old", **options)
    assert codec.decodes == []


def test_should_refuse_static_directory_without_mutation(tmp_path):
    target = tmp_path / "directory"
    target.mkdir()
    codec = Codec()
    store = BoundedCheckpointFileStore(target, codec)
    for operation in (lambda: save(store), lambda: load(store)):
        with pytest.raises((ValueError, OSError)):
            operation()
    assert target.is_dir() and list(target.iterdir()) == [] and codec.decodes == []


@pytest.mark.parametrize("dangling", (False, True))
def test_should_refuse_actual_static_symlink_when_host_capability_allows(tmp_path, dangling):
    target, destination = tmp_path / "link.bin", tmp_path / "destination.bin"
    if not dangling:
        destination.write_bytes(b"untouched")
    try:
        target.symlink_to(destination)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"host cannot create symbolic links: {type(error).__name__}: {error}")
    store = BoundedCheckpointFileStore(target, Codec())
    for operation in (lambda: save(store), lambda: load(store)):
        with pytest.raises((ValueError, OSError)):
            operation()
    assert target.is_symlink()
    if not dangling:
        assert destination.read_bytes() == b"untouched"


@pytest.mark.parametrize("mode", (stat.S_IFLNK, stat.S_IFIFO))
def test_should_refuse_native_nonregular_mode_even_without_host_link_privilege(
    monkeypatch, tmp_path, mode
):
    target = tmp_path / "state.bin"
    target.write_bytes(b"old")
    native_lstat = Path.lstat
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda path, *args, **kwargs: (
            stat_with(native_lstat(path, *args, **kwargs), st_mode=mode | 0o600)
            if path == target
            else native_lstat(path, *args, **kwargs)
        ),
    )
    store = BoundedCheckpointFileStore(target, Codec())
    for operation in (lambda: save(store), lambda: load(store)):
        with pytest.raises((ValueError, OSError)):
            operation()
    assert target.read_bytes() == b"old"


def test_should_recheck_target_type_before_publication_and_keep_prior_regular_bytes(
    monkeypatch, tmp_path
):
    target = tmp_path / "state.bin"
    target.write_bytes(b"old")
    native_lstat = Path.lstat
    checks = 0

    def lstat(path, *args, **kwargs):
        nonlocal checks
        result = native_lstat(path, *args, **kwargs)
        if path == target:
            checks += 1
            if checks > 1:
                return stat_with(result, st_mode=stat.S_IFLNK | 0o600)
        return result

    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises((ValueError, OSError)):
        save(BoundedCheckpointFileStore(target, Codec()))
    assert target.read_bytes() == b"old" and list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("family", ("pc", "neutral", "active"))
@pytest.mark.parametrize("guarded", (False, True))
def test_should_restore_and_continue_complete_native_file_state_without_live_mutation(
    tmp_path, family, guarded
):
    policy = CoupledInferencePolicy() if guarded else None
    config = (
        CoupledCircadianConfig.matched_pc_control()
        if family == "neutral"
        else CoupledCircadianConfig()
    )
    source = (
        CoupledPredictiveCodingNetwork(2, (3, 2), 23)
        if family == "pc"
        else CoupledCircadianPredictiveCodingNetwork(2, (3, 2), 23, config=config)
    )
    x, y = np.array([[0.2, -0.3], [0.7, 0.1]]), np.array([[0.0], [1.0]])
    options = {} if policy is None else {"inference_policy": policy}
    for _ in range(2):
        source.train_epoch(x, y, 0.02, 2, 0.05, **options)
    state_a = source.snapshot_state()
    store = BoundedCheckpointFileStore(tmp_path / "coupled.bin", CoupledCheckpointCodec())
    digest_a = save(store, state_a)
    decoded = store.load(binding=BINDING, expected_sha256=digest_a, limits=LIMITS)
    assert_same(decoded, state_a)
    restored = make_model(decoded)
    restored.restore_state(decoded)
    np.testing.assert_array_equal(restored.predict_proba(x), source.predict_proba(x))
    for array in arrays(decoded):
        array[:] = 999
    assert_same(source.snapshot_state(), state_a)
    left = source.train_epoch(x, y, 0.02, 2, 0.05, **options)
    right = restored.train_epoch(x, y, 0.02, 2, 0.05, **options)
    assert left == right
    state_b = source.snapshot_state()
    assert_same(restored.snapshot_state(), state_b)
    digest_b = save(store, state_b)
    assert digest_b != digest_a
    assert_same(store.load(binding=BINDING, expected_sha256=digest_b, limits=LIMITS), state_b)
    with pytest.raises(ValueError):
        store.load(binding=BINDING, expected_sha256=digest_a, limits=LIMITS)
    assert_same(source.snapshot_state(), state_b)
