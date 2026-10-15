"""Deterministic pipe-boundary failures plus two declared owned child probes."""

from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Event
from typing import Any

import numpy as np
import pytest

import src.infra.bounded_json_child as child_module
from src.infra.bounded_json_child import BoundedJsonChild

CHILD_WORK = {"actual_launches": 0}


class Input(BytesIO):
    def __init__(self):
        super().__init__()
        self.writes = []

    def write(self, raw):
        self.writes.append(raw)
        return super().write(raw)


class FakeProcess:
    def __init__(self, output=b'{"ok":1}\n', diagnostic=b""):
        self.pid = 1234
        self.returncode = None
        self.stdin, self.stdout, self.stderr = Input(), BytesIO(output), BytesIO(diagnostic)
        self.trace = []
        self.exit_code = 0
        self.wait_error = None

    def poll(self):
        return self.returncode

    def kill(self):
        self.trace.append("kill")
        peer_ended = getattr(self.stdin, "peer_ended", None)
        if callable(peer_ended):
            peer_ended()
        self.returncode = -9

    def wait(self, timeout=None):
        assert type(timeout) in (float, int) and timeout > 0
        self.trace.append("wait")
        if self.wait_error is not None and self.returncode is None:
            raise self.wait_error
        if self.returncode is None:
            self.returncode = self.exit_code
        return self.returncode


def fake(monkeypatch, tmp_path, output=b'{"ok":1}\n', diagnostic=b"", input_stream=None, **options):
    process = FakeProcess(output, diagnostic)
    if input_stream is not None:
        process.stdin = input_stream
    captured = []

    def launch(command, **kwargs):
        captured.append((command, kwargs))
        return process

    monkeypatch.setattr(child_module.subprocess, "Popen", launch)
    child = BoundedJsonChild(
        ["fixture-python", "fixture"],
        cwd=tmp_path,
        env={"CPU": "1"},
        timeout_seconds=options.pop("timeout_seconds", 0.5),
        **options,
    )
    return child, process, captured


def assert_closed(process):
    assert process.poll() is not None
    assert all(pipe.closed for pipe in (process.stdin, process.stdout, process.stderr))


def test_should_launch_binary_owned_pipes_without_shell_and_return_owned_json(
    monkeypatch, tmp_path
):
    child, process, captured = fake(monkeypatch, tmp_path, output='{"text":"λ"}\n'.encode())
    try:
        assert child.pid == process.pid and child.returncode is None and child.stderr == b""
        child.send({"input": [1, "λ"]})
        assert json.loads(process.stdin.getvalue()) == {"input": [1, "λ"]}
        assert child.receive() == {"text": "λ"}
        assert child.wait() == 0
        command, kwargs = captured[0]
        assert command == ["fixture-python", "fixture"]
        assert kwargs.get("shell", False) is False and kwargs.get("text", False) is False
        assert all(kwargs[name] == subprocess.PIPE for name in ("stdin", "stdout", "stderr"))
        for name in ("pid", "returncode", "stderr"):
            with pytest.raises(AttributeError):
                setattr(child, name, 1)
    finally:
        child.close()
    child.close()
    assert_closed(process)


@pytest.mark.parametrize(
    "options",
    (
        {"command": []},
        {"command": ("python",)},
        {"command": [True]},
        {"command": [""]},
        {"command": ["bad\x00argument"]},
        {"command": ["é" * 32769]},
        {"timeout_seconds": True},
        {"timeout_seconds": np.float64(1)},
        {"timeout_seconds": 0},
        {"timeout_seconds": float("nan")},
        {"timeout_seconds": float("inf")},
        {"max_line_bytes": True},
        {"max_output_bytes": 0},
        {"max_stderr_bytes": -1},
        {"max_messages": True},
        {"max_messages": 0},
        {"cwd": None},
        {"env": {"x": 1}},
    ),
)
def test_should_refuse_invalid_native_configuration_before_process_launch(
    monkeypatch, tmp_path, options
):
    monkeypatch.setattr(
        child_module.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("launch before preflight"),
    )
    config: dict[str, Any] = dict(command=["python"], cwd=tmp_path, env={})
    config.update(options)
    with pytest.raises(ValueError):
        BoundedJsonChild(**config)


@pytest.mark.parametrize(
    "raw",
    (
        b"[]\n",
        b"null\n",
        b"1\n",
        b"\n",
        b'{"a":1,"a":2}\n',
        b'{"x":{"a":1,"a":2}}\n',
        b'{"x":NaN}\n',
        b'{"x":Infinity}\n',
        b"\xff\n",
        b'{"ok":1}',
    ),
)
def test_should_terminally_refuse_malformed_nonobject_nonfinite_or_partial_json(
    monkeypatch, tmp_path, raw
):
    child, process, _ = fake(monkeypatch, tmp_path, output=raw)
    try:
        with pytest.raises(ValueError):
            child.receive()
        with pytest.raises((ValueError, RuntimeError)):
            child.send({"retry": True})
    finally:
        child.close()
    assert_closed(process)


@pytest.mark.parametrize("limit,accepted", ((9, True), (8, False)))
def test_should_measure_line_limit_in_bytes_including_terminal_newline(
    monkeypatch, tmp_path, limit, accepted
):
    child, process, _ = fake(monkeypatch, tmp_path, max_line_bytes=limit)
    try:
        if accepted:
            assert child.receive() == {"ok": 1}
        else:
            with pytest.raises(ValueError):
                child.receive()
    finally:
        child.close()
    assert_closed(process)


@pytest.mark.parametrize("failure", ("aggregate", "messages", "stderr"))
def test_should_refuse_cumulative_output_message_and_stderr_excess(monkeypatch, tmp_path, failure):
    raw = b'{"ok":1}\n' * 3
    options = (
        {"max_output_bytes": 23}
        if failure == "aggregate"
        else {"max_messages": 2}
        if failure == "messages"
        else {"max_stderr_bytes": 2}
    )
    child, process, _ = fake(
        monkeypatch,
        tmp_path,
        output=raw,
        diagnostic=b"err" if failure == "stderr" else b"",
        **options,
    )
    try:
        with pytest.raises(ValueError):
            for _ in range(3):
                child.receive()
            child.wait()
    finally:
        child.close()
    assert_closed(process)


@pytest.mark.parametrize(
    "value", ([], None, {"x": float("nan")}, {"x": float("inf")}, {"x": object()})
)
def test_should_refuse_invalid_send_before_any_pipe_write(monkeypatch, tmp_path, value):
    child, process, _ = fake(monkeypatch, tmp_path)
    try:
        with pytest.raises(ValueError):
            child.send(value)
        assert process.stdin.writes == []
    finally:
        child.close()


@pytest.mark.parametrize("kind", ("partial", "none", "bool", "numpyint"))
def test_should_refuse_nonexact_or_partial_native_stdin_write_and_reap(monkeypatch, tmp_path, kind):
    class Input(BytesIO):
        def write(self, raw):
            result = super().write(raw)
            return {
                "partial": result - 1,
                "none": None,
                "bool": True,
                "numpyint": np.int64(result),
            }[kind]

    child, process, _ = fake(monkeypatch, tmp_path, input_stream=Input())
    try:
        with pytest.raises(OSError):
            child.send({"value": 1})
    finally:
        child.close()
    assert_closed(process)


def test_should_bound_writer_deadline_and_release_owned_writer_before_close_returns(
    monkeypatch, tmp_path
):
    started, released, done = Event(), Event(), Event()

    class Input(BytesIO):
        def peer_ended(self):
            released.set()

        def write(self, raw):
            started.set()
            try:
                if not released.wait(0.5):
                    raise TimeoutError("fixture writer not released")
                return len(raw)
            finally:
                done.set()

        def close(self):
            released.set()
            return super().close()

    child, process, _ = fake(monkeypatch, tmp_path, timeout_seconds=0.02, input_stream=Input())
    try:
        with pytest.raises(TimeoutError):
            child.send({"value": 1})
        assert started.is_set()
    finally:
        child.close()
    assert done.wait(0.5)
    assert_closed(process)


@pytest.mark.parametrize("failure", ("exit", "stderr"))
def test_should_reject_unsuccessful_wait_and_retain_bounded_diagnostics(
    monkeypatch, tmp_path, failure
):
    child, process, _ = fake(
        monkeypatch, tmp_path, diagnostic=b"diagnostic" if failure == "stderr" else b""
    )
    process.exit_code = 17 if failure == "exit" else 0
    try:
        with pytest.raises(RuntimeError):
            child.wait()
        if failure == "stderr":
            assert child.stderr == b"diagnostic"
    finally:
        child.close()
    assert_closed(process)


def test_should_kill_only_owned_process_and_make_close_idempotent(monkeypatch, tmp_path):
    child, process, _ = fake(monkeypatch, tmp_path, output=b'{"pid":999999}\n')
    assert child.receive() == {"pid": 999999}
    assert child.kill() == -9
    child.close()
    child.close()
    assert process.trace.count("kill") == 1
    assert_closed(process)


def test_should_reap_and_close_owned_process_when_reader_failed_before_context_entry(
    monkeypatch, tmp_path
):
    threads = []
    native_thread = child_module.Thread

    def track_thread(*args, **kwargs):
        thread = native_thread(*args, **kwargs)
        threads.append((kwargs["name"].rsplit("-", 1)[-1], thread))
        return thread

    monkeypatch.setattr(child_module, "Thread", track_thread)
    child, process, _ = fake(monkeypatch, tmp_path, output=b"[]\n")
    assert {role for role, _ in threads} == {"stdout", "stderr", "stdin"}
    for role, thread in threads:
        if role in ("stdout", "stderr"):
            thread.join(timeout=1)
            assert not thread.is_alive()
    try:
        with pytest.raises(ValueError):
            with child:
                pytest.fail("failed reader admitted a context body")
        assert_closed(process)
        assert all(not thread.is_alive() for _, thread in threads)
    finally:
        child.close()


def direct_python():
    value = getattr(sys, "_base_executable", None)
    if type(value) is not str or not Path(value).is_file():
        raise ValueError("actual child probe requires existing physical base interpreter")
    return value


@pytest.mark.parametrize("mode", ("echo", "stall"))
def test_should_run_one_bounded_physical_echo_or_timeout_probe(mode):
    code = (
        "import json,sys; value=json.loads(sys.stdin.buffer.readline(4097)); print(json.dumps(value),flush=True)"
        if mode == "echo"
        else "import time; time.sleep(10)"
    )
    environment = dict(
        os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )
    CHILD_WORK["actual_launches"] += 1
    child = BoundedJsonChild(
        [direct_python(), "-B", "-X", "utf8", "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        timeout_seconds=2.0 if mode == "echo" else 0.2,
        max_line_bytes=4096,
        max_output_bytes=4096,
        max_stderr_bytes=4096,
    )
    try:
        if mode == "echo":
            child.send({"text": "λ"})
            assert child.receive() == {"text": "λ"}
            assert child.wait() == 0 and child.stderr == b""
        else:
            with pytest.raises(TimeoutError):
                child.receive()
    finally:
        child.close()
    assert child.returncode is not None
