"""Thirteen original-generator controls; six disposable children, no native work.

These distinguish object construction from profile call events at each resume.
They preserve original functions/code and test the existing guard unchanged.
They do not qualify async generators, pre-existing workers or live providers.
"""

from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import CodeType
from typing import Callable

import pytest

from provider_entry_profile import ProviderEntryProfile


def write_generator_work(path, **values):
    path.write_text(json.dumps(values, indent=2), encoding="utf8")


def finish_positive(tmp_path, mode, guard, before, observed, selected_entries):
    assert sum(guard.counts.values()) == selected_entries
    restored_main = sys.getprofile() is before[0]
    restored_future = threading.getprofile() is before[1]
    assert restored_main and restored_future
    assert not (tmp_path / "breach.json").exists()
    write_generator_work(
        tmp_path / "work.json",
        mode=mode,
        counters=guard.counts,
        observed=observed,
        phases=observed,
        body_phases=len(observed),
        wrapper_entries=guard.counts.get("wrapper", 0),
        selected_entries=selected_entries,
        metadata=True,
        restored_main=restored_main,
        restored_future=restored_future,
        threads=0,
    )


def test_should_construct_original_generator_without_entering_body(tmp_path):
    observed = []

    def provider():
        observed.append(guard.counts["generator"])
        yield 7

    code = provider.__code__
    guard = ProviderEntryProfile(((provider, "generator", 0),), tmp_path / "breach.json")
    before = sys.getprofile(), threading.getprofile()
    with guard.measure():
        generator = provider()
        assert generator.gi_code is code and generator.gi_frame is not None
        assert observed == [] and guard.counts == {"generator": 0}
    assert provider.__code__ is code and generator.gi_code is code
    finish_positive(tmp_path, "construction", guard, before, observed, 0)
    generator.close()


def test_should_count_next_send_and_exhaustion_before_each_original_phase(tmp_path):
    observed = []

    def provider():
        observed.append(["first", guard.counts["generator"]])
        sent = yield "first"
        observed.append(["sent", guard.counts["generator"], sent])
        yield "second"
        observed.append(["exhausted", guard.counts["generator"]])

    code = provider.__code__
    guard = ProviderEntryProfile(((provider, "generator", 3),), tmp_path / "breach.json")
    before = sys.getprofile(), threading.getprofile()
    generator = provider()
    with guard.measure():
        assert next(generator) == "first"
        assert generator.send(17) == "second"
        with pytest.raises(StopIteration):
            next(generator)
        assert guard.counts == {"generator": 3}
        with pytest.raises(StopIteration):
            next(generator)
        assert guard.counts == {"generator": 3}
    assert observed == [["first", 1], ["sent", 2, 17], ["exhausted", 3]]
    assert provider.__code__ is code and generator.gi_code is code
    assert generator.gi_frame is None
    finish_positive(tmp_path, "next-send-exhaustion", guard, before, observed, 3)


def test_should_count_handled_throw_before_original_handler(tmp_path):
    observed = []
    failure = ValueError("original handled throw")

    def provider():
        observed.append(["first", guard.counts["generator"]])
        try:
            yield "ready"
        except ValueError as caught:
            assert caught is failure
            observed.append(["handler", guard.counts["generator"]])
            yield "handled"
        observed.append(["exhausted", guard.counts["generator"]])

    code = provider.__code__
    guard = ProviderEntryProfile(((provider, "generator", 3),), tmp_path / "breach.json")
    before = sys.getprofile(), threading.getprofile()
    generator = provider()
    with guard.measure():
        assert next(generator) == "ready"
        assert generator.throw(failure) == "handled"
        with pytest.raises(StopIteration):
            next(generator)
    assert observed == [["first", 1], ["handler", 2], ["exhausted", 3]]
    assert guard.counts == {"generator": 3}
    assert provider.__code__ is code and generator.gi_code is code
    assert generator.gi_frame is None
    finish_positive(tmp_path, "handled-throw", guard, before, observed, 3)


def test_should_count_unhandled_throw_and_preserve_original_exception(tmp_path):
    observed = []
    failure = RuntimeError("original unhandled throw")

    def provider():
        observed.append(["first", guard.counts["generator"]])
        yield "ready"
        observed.append(["resumed", guard.counts["generator"]])

    code = provider.__code__
    guard = ProviderEntryProfile(((provider, "generator", 2),), tmp_path / "breach.json")
    before = sys.getprofile(), threading.getprofile()
    generator = provider()
    with guard.measure():
        assert next(generator) == "ready"
        with pytest.raises(RuntimeError) as caught:
            generator.throw(failure)
        assert caught.value is failure
    assert observed == [["first", 1]] and guard.counts == {"generator": 2}
    assert provider.__code__ is code and generator.gi_code is code
    assert generator.gi_frame is None
    finish_positive(tmp_path, "unhandled-throw", guard, before, observed, 2)


def test_should_count_close_before_original_finally(tmp_path):
    observed = []

    def provider():
        observed.append(["first", guard.counts["generator"]])
        try:
            yield "ready"
        finally:
            observed.append(["finally", guard.counts["generator"]])

    code = provider.__code__
    guard = ProviderEntryProfile(((provider, "generator", 2),), tmp_path / "breach.json")
    before = sys.getprofile(), threading.getprofile()
    generator = provider()
    with guard.measure():
        assert next(generator) == "ready"
        generator.close()
    assert observed == [["first", 1], ["finally", 2]]
    assert guard.counts == {"generator": 2}
    assert provider.__code__ is code and generator.gi_code is code
    assert generator.gi_frame is None
    finish_positive(tmp_path, "close-finally", guard, before, observed, 2)


def test_should_bind_shared_wrapper_once_and_original_context_generators(tmp_path):
    observed = []

    @contextmanager
    def first():
        observed.append(["first-enter", guard.counts["wrapper"], guard.counts["first"]])
        yield 1
        observed.append(["first-exit", guard.counts["wrapper"], guard.counts["first"]])

    @contextmanager
    def second():
        observed.append(["second-enter", guard.counts["wrapper"], guard.counts["second"]])
        yield 2
        observed.append(["second-exit", guard.counts["wrapper"], guard.counts["second"]])

    original_first, original_second = vars(first)["__wrapped__"], vars(second)["__wrapped__"]
    wrapper_code = first.__code__
    assert second.__code__ is wrapper_code
    assert original_first.__code__ is not wrapper_code
    assert original_second.__code__ is not wrapper_code
    with pytest.raises(ValueError, match="distinct identities"):
        ProviderEntryProfile(
            ((first, "first_wrapper", 1), (second, "second_wrapper", 1)),
            tmp_path / "invalid.json",
        )
    guard = ProviderEntryProfile(
        (
            (wrapper_code, "wrapper", 2),
            (original_first, "first", 2),
            (original_second, "second", 2),
        ),
        tmp_path / "breach.json",
    )
    before = sys.getprofile(), threading.getprofile()
    with guard.measure():
        with first() as value:
            assert value == 1
        with second() as value:
            assert value == 2
    assert observed == [
        ["first-enter", 1, 1],
        ["first-exit", 1, 2],
        ["second-enter", 2, 1],
        ["second-exit", 2, 2],
    ]
    assert guard.counts == {"wrapper": 2, "first": 2, "second": 2}
    assert first.__code__ is second.__code__ is wrapper_code
    assert (
        vars(first)["__wrapped__"] is original_first
        and vars(second)["__wrapped__"] is original_second
    )
    finish_positive(tmp_path, "shared-context-normal", guard, before, observed, 6)


def test_should_count_exceptional_context_exit_before_handler_and_finally(tmp_path):
    observed = []
    failure = ValueError("original context failure")

    @contextmanager
    def context():
        observed.append(["enter", guard.counts["wrapper"], guard.counts["generator"]])
        try:
            yield "ready"
        except ValueError as caught:
            assert caught is failure
            observed.append(["handler", guard.counts["wrapper"], guard.counts["generator"]])
            raise
        finally:
            observed.append(["finally", guard.counts["wrapper"], guard.counts["generator"]])

    original = vars(context)["__wrapped__"]
    wrapper_code, generator_code = context.__code__, original.__code__
    assert wrapper_code is not generator_code
    guard = ProviderEntryProfile(
        ((wrapper_code, "wrapper", 1), (original, "generator", 2)), tmp_path / "breach.json"
    )
    before = sys.getprofile(), threading.getprofile()
    caught_failure = None
    with guard.measure():
        try:
            with context() as value:
                assert value == "ready"
                raise failure
        except ValueError as caught:
            caught_failure = caught
    assert caught_failure is failure
    assert observed == [["enter", 1, 1], ["handler", 1, 2], ["finally", 1, 2]]
    assert guard.counts == {"wrapper": 1, "generator": 2}
    assert context.__code__ is wrapper_code and vars(context)["__wrapped__"] is original
    assert original.__code__ is generator_code
    finish_positive(tmp_path, "context-exception", guard, before, observed, 3)


def run_generator_negative_case(mode, directory):
    """Six disposable children; hard exit must preempt caller exception catches."""
    path = Path(directory)
    modes = {"first-next", "send", "throw", "close", "context-enter", "context-exit"}
    assert mode in modes

    def provider():
        write_generator_work(path / "initial.json", counters=guard.counts)
        try:
            yield "ready"
            (path / "resumed").write_text("forbidden resumed body", encoding="utf8")
        except BaseException:
            (path / "handler").write_text("forbidden handler", encoding="utf8")
            raise
        finally:
            (path / "finally").write_text("forbidden finally", encoding="utf8")

    limit = int(mode not in ("first-next", "context-enter"))
    context = contextmanager(provider)
    entries: tuple[tuple[Callable | CodeType, str, int], ...] = ((provider, "generator", limit),)
    if mode.startswith("context-"):
        assert (
            vars(context)["__wrapped__"] is provider and context.__code__ is not provider.__code__
        )
        entries += ((context.__code__, "wrapper", 1),)
    guard = ProviderEntryProfile(entries, path / "breach.json")
    with guard.measure():
        try:
            if mode.startswith("context-"):
                manager = context()
                write_generator_work(path / "prepared.json", counters=guard.counts)
                with manager:
                    if mode == "context-enter":
                        (path / "resumed").write_text("forbidden context body", encoding="utf8")
            else:
                generator = provider()
                write_generator_work(path / "prepared.json", counters=guard.counts)
                if mode == "first-next":
                    next(generator)
                else:
                    assert next(generator) == "ready"
                    if mode == "send":
                        generator.send(7)
                    elif mode == "throw":
                        generator.throw(ValueError("forbidden injected throw"))
                    else:
                        generator.close()
        except BaseException:
            (path / "caught").write_text("forbidden caller catch", encoding="utf8")
    (path / "continued").write_text("forbidden continuation", encoding="utf8")


def assert_generator_negative(mode, tmp_path):
    args = [
        sys.executable,
        "-B",
        "-X",
        "utf8",
        "-c",
        "import sys;sys.path.insert(0,'tests');from test_provider_generator_profile import run_generator_negative_case;run_generator_negative_case(sys.argv[1],sys.argv[2])",
        mode,
        str(tmp_path),
    ]
    result = subprocess.run(
        args, cwd=Path(__file__).resolve().parents[1], capture_output=True, timeout=5
    )
    assert result.returncode == 73 and len(result.stdout) + len(result.stderr) <= 4096
    absent = ("resumed", "handler", "finally", "caught", "continued")
    assert not any((tmp_path / name).exists() for name in absent)
    raw = (tmp_path / "breach.json").read_bytes()
    assert len(raw) <= 1024
    receipt = json.loads(raw)
    limit = int(mode not in ("first-next", "context-enter"))
    assert receipt == dict(
        reason="entry_limit", counter="generator", attempt=limit + 1, limit=limit
    )
    prepared = json.loads((tmp_path / "prepared.json").read_bytes())["counters"]
    expected_prepared = {"generator": 0}
    if mode.startswith("context-"):
        expected_prepared["wrapper"] = 1
    assert prepared == expected_prepared
    initial = None
    if limit:
        initial = json.loads((tmp_path / "initial.json").read_bytes())["counters"]
        assert initial == {**expected_prepared, "generator": 1}
    else:
        assert not (tmp_path / "initial.json").exists()
    counters = {**prepared, "generator": receipt["attempt"]}
    write_generator_work(
        tmp_path / "child.json",
        mode=mode,
        exit=result.returncode,
        stdout_bytes=len(result.stdout),
        stderr_bytes=len(result.stderr),
        receipt=receipt,
        counters=counters,
        prepared=prepared,
        initial=initial,
        permitted_initial_bodies=limit,
        phases=dict(initial=limit, resumed=0, handler=0, finally_body=0),
        resumed=0,
        handler=0,
        finally_body=0,
        caught=0,
        continued=0,
        threads=0,
    )


def test_should_terminate_before_first_next_body_at_zero_cap(tmp_path):
    assert_generator_negative("first-next", tmp_path)


def test_should_terminate_before_send_resumes_original_body(tmp_path):
    assert_generator_negative("send", tmp_path)


def test_should_terminate_before_throw_enters_original_handler(tmp_path):
    assert_generator_negative("throw", tmp_path)


def test_should_terminate_before_close_enters_original_finally(tmp_path):
    assert_generator_negative("close", tmp_path)


def test_should_terminate_before_context_enter_original_generator(tmp_path):
    assert_generator_negative("context-enter", tmp_path)


def test_should_terminate_before_context_exit_resumes_original_generator(tmp_path):
    assert_generator_negative("context-exit", tmp_path)
