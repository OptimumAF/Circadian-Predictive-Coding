"""Twelve pure guard controls; seven disposable negative children, no native work."""

import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import FunctionType

import pytest

from provider_entry_profile import ProviderEntryProfile


def write_work(path, **values):
    path.write_text(json.dumps(values, indent=2), encoding="utf8")


def test_should_count_original_alias_recursive_and_error_entries_before_bodies(tmp_path):
    observed = []

    def provider(depth=0, fail=False):
        observed.append(guard.counts["calls"])
        if depth:
            provider(depth - 1)
        if fail:
            raise RuntimeError("original failure")

    code = provider.__code__
    guard = ProviderEntryProfile(((provider, "calls", 5),), tmp_path / "breach.json")
    before_main, before_thread = sys.getprofile(), threading.getprofile()
    with guard.measure():
        alias = provider
        alias()
        provider(2)
        with pytest.raises(RuntimeError, match="original failure"):
            provider(fail=True)
    assert guard.counts == {"calls": 5} and observed == [1, 2, 3, 4, 5]
    assert provider.__code__ is code and alias is provider
    assert sys.getprofile() is before_main and threading.getprofile() is before_thread
    with pytest.raises(RuntimeError, match="cannot renew"):
        with guard.measure():
            pytest.fail("spent guard reused")
    write_work(tmp_path / "work.json", calls=5, bodies=len(observed), metadata=True, restored=True)


def test_should_ignore_foreign_equal_code_and_refuse_invalid_bindings(tmp_path):
    def provider():
        return 7

    foreign = FunctionType(provider.__code__.replace(), provider.__globals__)
    assert foreign.__code__ == provider.__code__ and foreign.__code__ is not provider.__code__
    for entries in (
        (),
        ((provider, "calls", True),),
        ((provider, "calls", -1),),
        ((provider, "calls", 1), (provider, "other", 1)),
    ):
        with pytest.raises(ValueError):
            ProviderEntryProfile(entries, tmp_path / "invalid.json")
    guard = ProviderEntryProfile(
        ((provider, "calls", 1), (time.monotonic, "monotonic", 2)), tmp_path / "breach.json"
    )
    with guard.measure():
        assert foreign() == provider() == 7
        first, second = time.monotonic(), time.monotonic()
    assert second >= first and guard.counts == {"calls": 1, "monotonic": 2}
    write_work(tmp_path / "work.json", calls=1, bodies=2, monotonic=2, foreign_ignored=True)


def test_should_forward_and_restore_main_profile_without_replacing_dispatch(tmp_path):
    forwarded = []

    def provider():
        return 9

    def prior(frame, event, arg):
        if event == "call" and frame.f_code is provider.__code__:
            forwarded.append(True)

    previous_main, previous_thread = sys.getprofile(), threading.getprofile()
    try:
        sys.setprofile(prior)
        guard = ProviderEntryProfile(((provider, "calls", 1),), tmp_path / "breach.json")
        with guard.measure():
            assert provider() == 9
        assert sys.getprofile() is prior and threading.getprofile() is previous_thread
    finally:
        sys.setprofile(previous_main)
    assert forwarded == [True] and guard.counts == {"calls": 1}
    write_work(tmp_path / "work.json", calls=1, bodies=1, forwarded=1, restored=True)


def test_should_cover_future_worker_and_restore_both_profiles_after_join(tmp_path):
    forwarded, observed = [], []

    def provider():
        observed.append(guard.counts["calls"])

    def prior(frame, event, arg):
        if event == "call" and frame.f_code is provider.__code__:
            forwarded.append(True)

    previous_main, previous_thread = sys.getprofile(), threading.getprofile()
    try:
        threading.setprofile(prior)
        guard = ProviderEntryProfile(((provider, "calls", 1),), tmp_path / "breach.json")
        with guard.measure():
            worker = threading.Thread(target=provider, daemon=False)
            worker.start()
            worker.join(1.0)
            assert not worker.is_alive()
        assert sys.getprofile() is previous_main and threading.getprofile() is prior
    finally:
        threading.setprofile(previous_thread)
    assert observed == [1] and forwarded == [True] and guard.counts == {"calls": 1}
    write_work(
        tmp_path / "work.json", calls=1, bodies=1, threads=1, alive=0, forwarded=1, restored=True
    )


def run_negative_case(mode, directory):
    """Only called by seven disposable control children; exit73 is expected."""
    path = Path(directory)

    def provider():
        (path / "body").write_text("must never run", encoding="utf8")

    def invoke():
        try:
            provider()
        except BaseException:
            (path / "caught").write_text("must never continue", encoding="utf8")

    def prior(frame, event, arg):
        if (
            mode == "setup-error"
            and event == "call"
            and frame.f_code is threading.setprofile.__code__
        ):
            raise RuntimeError("install hook failure")
        if event == "call" and frame.f_code is provider.__code__:
            raise RuntimeError("prior hook failure")

    receipt = path / "breach.json"
    if mode == "receipt-failure":
        receipt = path / "missing-parent" / "breach.json"
    if mode in ("prior-error", "setup-error"):
        sys.setprofile(prior)
    guard = ProviderEntryProfile(((provider, "calls", int(mode == "prior-error")),), receipt)
    with guard.measure():
        if mode == "sys-mutation":
            sys.setprofile(None)
        elif mode == "thread-mutation":
            threading.setprofile(None)
        elif mode == "worker-cap":
            worker = threading.Thread(target=invoke, daemon=False)
            worker.start()
            worker.join(1.0)
        else:
            invoke()
    (path / "continued").write_text("must never continue", encoding="utf8")


def assert_negative(mode, tmp_path):
    args = [
        sys.executable,
        "-B",
        "-X",
        "utf8",
        "-c",
        "import sys;sys.path.insert(0,'tests');from test_provider_entry_profile import run_negative_case;run_negative_case(sys.argv[1],sys.argv[2])",
        mode,
        str(tmp_path),
    ]
    result = subprocess.run(
        args, cwd=Path(__file__).resolve().parents[1], capture_output=True, timeout=5
    )
    assert result.returncode == 73 and len(result.stdout) + len(result.stderr) <= 4096
    assert not any((tmp_path / name).exists() for name in ("body", "caught", "continued"))
    receipt = None
    if mode == "receipt-failure":
        assert (
            not (tmp_path / "missing-parent").exists() and not (tmp_path / "breach.json").exists()
        )
    else:
        raw = (tmp_path / "breach.json").read_bytes()
        assert len(raw) <= 1024
        receipt = json.loads(raw)
        reasons = {
            "prior-error": "prior_profile_failure",
            "setup-error": "profile_install_failure",
            "sys-mutation": "profile_mutation",
            "thread-mutation": "profile_mutation",
        }
        assert receipt == (
            dict(reason=reasons[mode], counter="profile", attempt=0, limit=0)
            if mode in reasons
            else dict(reason="entry_limit", counter="calls", attempt=1, limit=0)
        )
    write_work(
        tmp_path / "child.json",
        mode=mode,
        exit=result.returncode,
        stdout_bytes=len(result.stdout),
        stderr_bytes=len(result.stderr),
        receipt=receipt,
        body=0,
        caught=0,
        continued=0,
        threads=int(mode == "worker-cap"),
    )


def test_should_terminate_main_cap_before_body_despite_baseexception_catch(tmp_path):
    assert_negative("main-cap", tmp_path)


def test_should_terminate_entire_child_on_future_worker_cap(tmp_path):
    assert_negative("worker-cap", tmp_path)


def test_should_terminate_even_when_breach_receipt_cannot_be_written(tmp_path):
    assert_negative("receipt-failure", tmp_path)


def test_should_terminate_on_prior_hook_failure_before_original_body(tmp_path):
    assert_negative("prior-error", tmp_path)


def test_should_terminate_on_profile_install_failure_before_continuation(tmp_path):
    assert_negative("setup-error", tmp_path)


def test_should_terminate_before_original_main_profile_setter_mutates_coverage(tmp_path):
    assert_negative("sys-mutation", tmp_path)


def test_should_terminate_before_future_thread_profile_setter_mutates_coverage(tmp_path):
    assert_negative("thread-mutation", tmp_path)


def test_should_refuse_existing_worker_before_installation(tmp_path):
    ready, stop = threading.Event(), threading.Event()

    def idle():
        ready.set()
        stop.wait()

    def provider():
        pytest.fail("refused measurement must not dispatch")

    worker = threading.Thread(target=idle, daemon=False)
    worker.start()
    before_main, before_thread = sys.getprofile(), threading.getprofile()
    try:
        assert ready.wait(1.0)
        guard = ProviderEntryProfile(((provider, "calls", 0),), tmp_path / "breach.json")
        with pytest.raises(RuntimeError, match="no existing workers"):
            with guard.measure():
                pytest.fail("uncovered worker admitted")
    finally:
        stop.set()
        worker.join(1.0)
    assert not worker.is_alive() and guard.counts == {"calls": 0}
    assert sys.getprofile() is before_main and threading.getprofile() is before_thread
    write_work(
        tmp_path / "work.json",
        calls=0,
        bodies=1,
        threads=1,
        alive=0,
        admission_refused=True,
        restored=True,
    )
