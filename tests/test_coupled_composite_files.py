"""Complete data checkpoints through the existing bounded atomic file store.

Synthetic exact native records exercise storage and CLI boundaries without any
model construction, training, prediction or restored runtime authority.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any

import numpy as np
import pytest

import scripts.run_coupled_composite_files as cli
import src.infra.checkpoint_files as files
from src.adapters.coupled_composite_codec import CoupledCompositeCheckpointCodec
from src.core.checkpoint_codec import CodecBinding
from src.infra.checkpoint_files import BoundedCheckpointFileStore
from test_coupled_composite_codec import (
    BINDING,
    CAPTURE_LIMITS,
    LIMITS as WIRE_LIMITS,
    assert_graph,
    metadata,
    state_factory as state_factory,
)
from test_coupled_managed_composite_capture import graph_factory as graph_factory


DISK_WORK = dict(created_files=0, payload_bytes=0, maximum_payload_bytes=0)


@pytest.fixture(autouse=True)
def account_owned_file_work(monkeypatch, tmp_path):
    write_bytes, temporary, fdopen = Path.write_bytes, files.tempfile.mkstemp, files.os.fdopen
    owned_fds: set[int] = set()

    def charge(raw):
        assert len(raw) <= 262144
        DISK_WORK["payload_bytes"] += len(raw)
        DISK_WORK["maximum_payload_bytes"] = max(DISK_WORK["maximum_payload_bytes"], len(raw))
        assert DISK_WORK["payload_bytes"] <= 8 * 1024**2

    def direct(path, raw):
        if path.is_relative_to(tmp_path):
            DISK_WORK["created_files"] += int(not path.exists())
            charge(raw)
        return write_bytes(path, raw)

    def staging(*args, **kwargs):
        fd, name = temporary(*args, **kwargs)
        if Path(name).is_relative_to(tmp_path):
            DISK_WORK["created_files"] += 1
            owned_fds.add(fd)
        return fd, name

    class CountWrite:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def write(self, raw):
            charge(raw)
            return self.stream.write(raw)

    def opened(fd, *args, **kwargs):
        stream = fdopen(fd, *args, **kwargs)
        return CountWrite(stream) if fd in owned_fds else stream

    monkeypatch.setattr(Path, "write_bytes", direct)
    monkeypatch.setattr(files.tempfile, "mkstemp", staging)
    monkeypatch.setattr(files.os, "fdopen", opened)
    yield
    assert DISK_WORK["created_files"] <= 256


def options(raw):
    return dict(
        binding=BINDING,
        expected_sha256=sha256(raw).hexdigest(),
        capture_limits=CAPTURE_LIMITS,
        wire_limits=replace(WIRE_LIMITS, max_layers=4),
    )


def source_file(tmp_path, state):
    raw = CoupledCompositeCheckpointCodec(capture_limits=CAPTURE_LIMITS).encode(
        state, binding=BINDING, limits=WIRE_LIMITS
    )
    source = tmp_path / "source.json"
    source.write_bytes(raw)
    return source, raw


@pytest.mark.parametrize("family", ("pc", "neutral", "active"))
@pytest.mark.parametrize("guarded", (False, True))
def test_should_copy_complete_graph_with_independent_digest_and_data_only_report(
    state_factory, tmp_path, family, guarded
):
    state = state_factory(family, guarded)
    typed = np.array([[0.25, -0.0, 0.5], [0.75, -0.5, 1.0]], dtype=">f8", order="F")
    typed.flags.writeable = False
    metadata(state)["typed_file_payload"] = typed
    source, raw = source_file(tmp_path, state)
    directory = tmp_path / "new-checkpoint"
    report = cli.copy_checkpoint(source, directory, **options(raw))
    target = directory / "checkpoint.json"
    assert report == dict(
        source=str(source.absolute()),
        directory=str(directory.absolute()),
        checkpoint=str(target.absolute()),
        digest=sha256(raw).hexdigest(),
        data_only=True,
    )
    assert target.read_bytes() == source.read_bytes() == raw
    assert sorted(path.name for path in directory.iterdir()) == ["checkpoint.json"]
    restored = BoundedCheckpointFileStore(
        target, CoupledCompositeCheckpointCodec(capture_limits=CAPTURE_LIMITS)
    ).load(binding=BINDING, expected_sha256=sha256(raw).hexdigest(), limits=WIRE_LIMITS)
    assert_graph(restored, state)
    assert not hasattr(restored, "authority")
    assert len(json.dumps(report, allow_nan=False).encode()) < 65536


@pytest.mark.parametrize(
    "fault",
    (
        "digest",
        "source_binding",
        "policy_binding",
        "capture_limits",
        "wire_bytes",
        "array_bytes",
        "native_layers",
    ),
)
def test_should_refuse_independent_control_mismatch_before_creating_output(
    state_factory, tmp_path, fault
):
    source, raw = source_file(tmp_path, state_factory())
    kwargs: dict[str, Any] = options(raw)
    if fault == "digest":
        kwargs["expected_sha256"] = "c" * 64
    elif fault == "source_binding":
        kwargs["binding"] = CodecBinding("c" * 64, BINDING.policy_sha256)
    elif fault == "policy_binding":
        kwargs["binding"] = CodecBinding(BINDING.source_sha256, "c" * 64)
    elif fault == "capture_limits":
        kwargs["capture_limits"] = replace(CAPTURE_LIMITS, max_nodes=CAPTURE_LIMITS.max_nodes + 1)
    elif fault == "wire_bytes":
        kwargs["wire_limits"] = replace(WIRE_LIMITS, max_encoded_bytes=len(raw) - 1)
    elif fault == "array_bytes":
        kwargs["wire_limits"] = replace(WIRE_LIMITS, max_array_bytes=1)
    else:
        kwargs["wire_limits"] = replace(WIRE_LIMITS, max_layers=1)
    directory = tmp_path / "refused"
    with pytest.raises(ValueError):
        cli.copy_checkpoint(source, directory, **kwargs)
    assert not directory.exists() and source.read_bytes() == raw


@pytest.mark.parametrize("present", (False, True))
def test_should_leave_output_absent_when_source_is_missing_or_malformed(tmp_path, present):
    source = tmp_path / "source.json"
    raw = b"malformed complete checkpoint"
    if present:
        source.write_bytes(raw)
    directory = tmp_path / "refused"
    with pytest.raises(ValueError if present else FileNotFoundError):
        cli.copy_checkpoint(source, directory, **options(raw))
    assert not directory.exists()


@pytest.mark.parametrize("kind", ("directory", "file"))
def test_should_preserve_existing_output_before_source_decoder(monkeypatch, tmp_path, kind):
    source, directory = tmp_path / "missing-source.json", tmp_path / "existing"
    marker = b"user-owned marker"
    if kind == "directory":
        directory.mkdir()
        (directory / "marker").write_bytes(marker)
    else:
        directory.write_bytes(marker)
    monkeypatch.setattr(
        BoundedCheckpointFileStore,
        "load",
        lambda *a, **k: pytest.fail("source decoded before output refusal"),
    )
    with pytest.raises((ValueError, FileExistsError)):
        cli.copy_checkpoint(source, directory, **options(b"unused"))
    assert (directory / "marker" if kind == "directory" else directory).read_bytes() == marker


def test_should_refuse_a_dangling_link_entry_reported_only_by_lstat(monkeypatch, tmp_path):
    directory = tmp_path / "dangling-output"
    assert not directory.exists()
    original = Path.lstat

    def inspect(path, *args, **kwargs):
        if path == directory:
            return os.stat_result((stat.S_IFLNK, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", inspect)
    monkeypatch.setattr(
        BoundedCheckpointFileStore,
        "load",
        lambda *a, **k: pytest.fail("source decoded before dangling-output refusal"),
    )
    with pytest.raises(FileExistsError):
        cli.copy_checkpoint(tmp_path / "source.json", directory, **options(b"unused"))
    assert not directory.exists()


@pytest.mark.parametrize("present", (False, True))
@pytest.mark.parametrize("fault", ("fsync", "replace"))
def test_should_preserve_prior_bytes_and_remove_only_owned_temp_on_atomic_failure(
    state_factory, monkeypatch, tmp_path, present, fault
):
    state = state_factory()
    codec = CoupledCompositeCheckpointCodec(capture_limits=CAPTURE_LIMITS)
    old = codec.encode(state, binding=BINDING, limits=WIRE_LIMITS)
    candidate = deepcopy(state)
    metadata(candidate)["new_data"] = "candidate"
    target = tmp_path / "checkpoint.json"
    if present:
        target.write_bytes(old)
    marker = tmp_path / "unrelated.txt"
    marker.write_bytes(b"leave alone")
    error = OSError("injected atomic boundary failure")
    calls: list[str] = []
    original_fsync, original_replace = files.os.fsync, files.os.replace

    def sync(fd):
        calls.append("fsync")
        if fault == "fsync":
            raise error
        return original_fsync(fd)

    def publish(source, destination):
        calls.append("replace")
        if fault == "replace":
            raise error
        return original_replace(source, destination)

    monkeypatch.setattr(files.os, "fsync", sync)
    monkeypatch.setattr(files.os, "replace", publish)
    with pytest.raises(OSError) as caught:
        BoundedCheckpointFileStore(target, codec).save(
            candidate, binding=BINDING, limits=WIRE_LIMITS
        )
    assert caught.value is error
    assert calls == (["fsync"] if fault == "fsync" else ["fsync", "replace"])
    assert marker.read_bytes() == b"leave alone"
    assert target.read_bytes() == old if present else not target.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == (
        ["checkpoint.json", "unrelated.txt"] if present else ["unrelated.txt"]
    )


@pytest.mark.parametrize("fault", ("saved_digest", "verification"))
def test_should_report_late_failure_without_deleting_successfully_published_data(
    state_factory, monkeypatch, tmp_path, fault
):
    source, raw = source_file(tmp_path, state_factory())
    directory = tmp_path / "new-checkpoint"
    original_save, original_load = BoundedCheckpointFileStore.save, BoundedCheckpointFileStore.load
    error = RuntimeError("injected verification failure")
    loads: list[Path] = []

    def save(self, *args, **kwargs):
        digest = original_save(self, *args, **kwargs)
        return "c" * 64 if fault == "saved_digest" else digest

    def load(self, *args, **kwargs):
        loads.append(self.path)
        if fault == "verification" and len(loads) == 2:
            raise error
        return original_load(self, *args, **kwargs)

    monkeypatch.setattr(BoundedCheckpointFileStore, "save", save)
    monkeypatch.setattr(BoundedCheckpointFileStore, "load", load)
    with pytest.raises((ValueError, RuntimeError)) as caught:
        cli.copy_checkpoint(source, directory, **options(raw))
    if fault == "verification":
        assert caught.value is error
    assert (directory / "checkpoint.json").read_bytes() == source.read_bytes() == raw


def cli_args():
    return [
        "--source",
        "source.json",
        "--directory",
        "new-output",
        "--source-sha256",
        BINDING.source_sha256,
        "--policy-sha256",
        BINDING.policy_sha256,
        "--expected-sha256",
        "c" * 64,
    ]


def test_should_describe_without_filesystem_or_native_work(monkeypatch, capsys):
    expected = cli.describe_example()
    monkeypatch.setattr(sys, "argv", ["run_coupled_composite_files.py", "--describe"])
    monkeypatch.setattr(
        cli, "copy_checkpoint", lambda *a, **k: pytest.fail("description copied files")
    )
    for name in ("lstat", "open", "mkdir", "write_bytes"):
        monkeypatch.setattr(
            Path, name, lambda *a, **k: pytest.fail("description touched filesystem")
        )
    cli.main()
    assert json.loads(capsys.readouterr().out) == json.loads(json.dumps(expected))
    assert (
        expected["protocol_id"] == "coupled_composite_files_engineering_v1"
        and expected["data_only"] is True
    )
    assert expected["operations"] == ["load", "save", "load"]


@pytest.mark.parametrize(
    "missing",
    ("--source", "--directory", "--source-sha256", "--policy-sha256", "--expected-sha256"),
)
def test_should_require_every_independent_operation_argument_before_copy(monkeypatch, missing):
    args = cli_args()
    index = args.index(missing)
    del args[index : index + 2]
    monkeypatch.setattr(sys, "argv", ["run_coupled_composite_files.py", *args])
    monkeypatch.setattr(
        cli, "copy_checkpoint", lambda *a, **k: pytest.fail("incomplete request copied files")
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


@pytest.mark.parametrize("args", (["--describe", "--source", "source.json"], ["--unknown-control"]))
def test_should_refuse_ambiguous_or_unknown_cli_requests_before_copy(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["run_coupled_composite_files.py", *args])
    monkeypatch.setattr(
        cli, "copy_checkpoint", lambda *a, **k: pytest.fail("invalid request copied files")
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


def test_should_pass_all_explicit_bounds_and_bindings_to_reusable_callable(monkeypatch, capsys):
    calls: list[Any] = []

    def copied(source, directory, **kwargs):
        calls.append((source, directory, kwargs))
        return {"data_only": True, "digest": "c" * 64}

    bounds = dict(
        record_max_records=65,
        record_max_identifier_bytes=129,
        capture_max_nodes=8193,
        capture_max_depth=65,
        capture_max_array_bytes=65537,
        capture_max_dimension=33,
        wire_max_encoded_bytes=262145,
        wire_max_array_bytes=65538,
        wire_max_layers=5,
        wire_max_dimension=34,
    )
    args = cli_args()
    for key, value in bounds.items():
        args.extend(("--" + key.replace("_", "-"), str(value)))
    monkeypatch.setattr(sys, "argv", ["run_coupled_composite_files.py", *args])
    monkeypatch.setattr(cli, "copy_checkpoint", copied)
    cli.main()
    assert len(calls) == 1
    source, directory, kwargs = calls[0]
    assert source == Path("source.json") and directory == Path("new-output")
    assert kwargs["binding"] == BINDING and kwargs["expected_sha256"] == "c" * 64
    assert asdict(kwargs["capture_limits"]) == dict(
        records=dict(max_records=65, max_identifier_bytes=129),
        max_nodes=8193,
        max_depth=65,
        max_array_bytes=65537,
        max_dimension=33,
    )
    assert asdict(kwargs["wire_limits"]) == dict(
        max_encoded_bytes=262145, max_array_bytes=65538, max_layers=5, max_dimension=34
    )
    assert json.loads(capsys.readouterr().out) == {"data_only": True, "digest": "c" * 64}


def test_should_route_default_cli_bounds_without_inventing_controls(monkeypatch, capsys):
    calls: list[dict[str, Any]] = []

    def copied(source, directory, **kwargs):
        calls.append(kwargs)
        return {"data_only": True}

    monkeypatch.setattr(sys, "argv", ["run_coupled_composite_files.py", *cli_args()])
    monkeypatch.setattr(cli, "copy_checkpoint", copied)
    cli.main()
    assert len(calls) == 1 and calls[0]["capture_limits"] == CAPTURE_LIMITS
    assert calls[0]["wire_limits"] == replace(WIRE_LIMITS, max_layers=4)
    assert json.loads(capsys.readouterr().out) == {"data_only": True}


def test_should_count_platform_newlines_and_terminal_newline_in_output_allowance(
    monkeypatch, capsys
):
    report = {"data_only": True, "digest": "a" * 64}
    encoded = json.dumps(report, allow_nan=False, sort_keys=True, indent=2)
    required = len((encoded + "\n").replace("\n", "\r\n").encode("utf8"))
    monkeypatch.setattr(cli.os, "linesep", "\r\n")
    monkeypatch.setattr(cli, "MAX_OUTPUT_BYTES", required - 1)
    with pytest.raises(ValueError):
        cli._print_report(report)
    assert not capsys.readouterr().out
    monkeypatch.setattr(cli, "MAX_OUTPUT_BYTES", required)
    cli._print_report(report)
    assert json.loads(capsys.readouterr().out) == report
