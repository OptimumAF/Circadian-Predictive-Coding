"""One bounded real worker exit observation, not durable model crash recovery."""

from dataclasses import asdict
import json
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys
from threading import Thread

import pytest

from src.core.recovery_observation import RecoveryProcessIdentity
from src.infra.windows_process_handles import WindowsProcessHandle, WindowsRecoveryApi
from src.infra.windows_recovery_observer import WindowsRecoveryObserver


CHILD = """
import json,os,sys
from dataclasses import asdict
from src.core.recovery_observation import RecoveryProcessIdentity
from src.infra.windows_process_handles import WindowsProcessHandle,WindowsRecoveryApi
from src.infra.windows_recovery_observer import WindowsRecoveryObserver
api=WindowsRecoveryApi()
version=tuple(int(part) for part in sys.argv[3].split('.'))
if sys.version_info[:3]!=version:raise RuntimeError("worker interpreter version differs")
expected=RecoveryProcessIdentity(int(sys.argv[1]),int(sys.argv[2]))
with WindowsProcessHandle.pin(expected.pid,expected=expected,api=api) as anchor:
    observation=WindowsRecoveryObserver(anchor).observe()
    print(json.dumps(asdict(observation)),flush=True)
    if sys.stdin.readline(32)!="crash\\n":raise RuntimeError("expected bounded fixture command")
    os._exit(17)
"""


def direct_worker_executable() -> str:
    # Venv redirectors start a second physical process; pin the actual worker.
    executable = getattr(sys, "_base_executable", None)
    if type(executable) is not str or not Path(executable).is_file():
        raise ValueError("bounded worker fixture requires its existing base interpreter")
    return executable


def test_should_select_existing_base_interpreter_without_launching_venv_redirector():
    executable = direct_worker_executable()
    assert executable == getattr(sys, "_base_executable")
    if sys.prefix != sys.base_prefix:
        assert Path(executable).resolve() != Path(sys.executable).resolve()


def test_should_refuse_missing_base_interpreter_before_starting_worker(monkeypatch):
    monkeypatch.setattr(sys, "_base_executable", None)
    with pytest.raises(ValueError, match="base interpreter"):
        direct_worker_executable()


@pytest.mark.skipif(sys.platform != "win32", reason="supported Windows observation only")
def test_should_observe_one_registered_child_exit_in_same_live_anchor_epoch():
    api = WindowsRecoveryApi()
    with WindowsProcessHandle.pin(os.getpid(), api=api) as anchor:
        observer = WindowsRecoveryObserver(anchor)
        before = observer.observe()
        command = [
            direct_worker_executable(),
            "-B",
            "-X",
            "utf8",
            "-c",
            CHILD,
            str(anchor.identity.pid),
            str(anchor.identity.created_filetime),
            ".".join(str(part) for part in sys.version_info[:3]),
        ]
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        assert process.stdin is not None and process.stdout is not None
        stdout_pipe = process.stdout
        inbox: Queue[str] = Queue()
        reader = Thread(target=lambda: inbox.put(stdout_pipe.readline(4097)), daemon=True)
        reader.start()
        registered = None
        try:
            line = inbox.get(timeout=5)
            assert len(line) <= 4096 and line.endswith("\n")
            child = json.loads(line)
            registered = WindowsProcessHandle.pin(process.pid, api=api)
            observed_child = RecoveryProcessIdentity(**child["observer"])
            assert registered.identity == observed_child
            alive = observer.observe(registered)
            assert alive.previous_owner_ended is False
            assert before.now_ns <= child["now_ns"] <= alive.now_ns
            assert child["clock_epoch"] == before.clock_epoch == alive.clock_epoch
            assert 0 < child["rss_bytes"] <= 512 * 1024 * 1024
            assert 0 < alive.rss_bytes <= 512 * 1024 * 1024
            process.stdin.write("crash\n")
            process.stdin.flush()
            stdout, stderr = process.communicate(timeout=5)
            assert process.returncode == 17 and stdout == stderr == ""
            ended = observer.observe(registered)
            assert ended.previous_owner_ended is True and ended.previous_owner == observed_child
            assert ended.clock_epoch == before.clock_epoch
            assert ended.now_ns >= alive.now_ns and ended.peak_rss_bytes >= alive.peak_rss_bytes
            receipt = {
                "PASS": True,
                "before": asdict(before),
                "child": child,
                "alive": asdict(alive),
                "ended": asdict(ended),
                "returncode": 17,
                "owned_children": 1,
                "NEW_native_work": 0,
                "actual_durable_recovery_proven": False,
            }
            destination = os.environ.get("CIRCADIAN_RECOVERY_PROCESS_RECEIPT")
            if destination:
                with Path(destination).open("x", encoding="utf8") as stream:
                    json.dump(receipt, stream, indent=2)
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)
            reader.join(timeout=5)
            assert not reader.is_alive()
            if registered is not None:
                registered.close()
            for pipe in (process.stdin, process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()
