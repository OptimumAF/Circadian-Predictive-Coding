"""Seven bounded profiler controls; no native fixture or graph execution."""

import json
import sys
from types import FunctionType
from typing import cast

import pytest

from native_work_profile import bind_original_calls, count_original_calls
import test_native_managed_replay_checkpoints as native
from src.app.expiry_validation_proof import require_original_metadata_dispatch


def test_should_count_normal_recursive_and_failed_entries_before_their_bodies():
    work = {"calls": 0}
    observed = []

    class Calls:
        def run(self, depth=0, fail=False):
            observed.append(work["calls"])
            if depth:
                self.run(depth - 1)
            if fail:
                raise RuntimeError("original body failure")

    table = bind_original_calls(((Calls, "run", "calls"),))
    previous = sys.getprofile()

    def profile(frame, event, arg):
        count_original_calls(table, work, {"calls": 5}, frame, event)

    try:
        sys.setprofile(profile)
        instance = Calls()
        instance.run()
        instance.run(2)
        with pytest.raises(RuntimeError, match="original body failure"):
            instance.run(fail=True)
    finally:
        sys.setprofile(previous)
    assert work == {"calls": 5}
    assert observed == [1, 2, 3, 4, 5]


def test_should_ignore_equal_foreign_code_and_count_a_direct_original_alias():
    class Calls:
        def run(self):
            return 17

    original = Calls.__dict__["run"]
    foreign = FunctionType(original.__code__.replace(), original.__globals__)
    assert foreign.__code__ == original.__code__
    assert foreign.__code__ is not original.__code__
    table = bind_original_calls(((Calls, "run", "calls"),))
    work = {"calls": 0}
    previous = sys.getprofile()

    def profile(frame, event, arg):
        count_original_calls(table, work, {"calls": 1}, frame, event)

    try:
        sys.setprofile(profile)
        assert foreign(None) == 17
        alias = original
        assert alias(None) == 17
    finally:
        sys.setprofile(previous)
    assert work == {"calls": 1}


def test_should_refuse_duplicate_code_before_measurement():
    class Calls:
        def run(self):
            raise AssertionError("body must not run")

        alias = run

    previous = sys.getprofile()
    with pytest.raises(ValueError, match="distinct code identities"):
        bind_original_calls(((Calls, "run", "calls"), (Calls, "alias", "other")))
    assert sys.getprofile() is previous


def test_should_charge_a_zero_cap_attempt_before_the_original_body():
    bodies = []

    class Calls:
        def run(self):
            bodies.append(True)

    table = bind_original_calls(((Calls, "run", "calls"),))
    work = {"calls": 0}
    previous = sys.getprofile()

    def profile(frame, event, arg):
        count_original_calls(table, work, {"calls": 0}, frame, event)

    try:
        sys.setprofile(profile)
        with pytest.raises(AssertionError, match="original call cap exceeded: calls"):
            Calls().run()
    finally:
        sys.setprofile(previous)
    assert work == {"calls": 1}
    assert bodies == []


def zero_native_limits():
    return dict.fromkeys(
        (
            "models",
            "learners",
            "forks",
            "wakes",
            "steps",
            "stores",
            "predicts",
            "array_copies",
            "graph_events",
            "source_array_bytes",
            "captures",
            "preparations",
            "native_restores",
            "handoffs",
            "cleanup_attempts",
        ),
        0,
    )


def raw_dispatch():
    return tuple(
        owner.__dict__[method]
        for owner, method in (
            (native.CandidateCheckpointController, "capture"),
            (native.CandidateCheckpointController, "restore"),
            (native.ExperienceInbox, "_retire_ledger"),
            (native.ManagedDataLifecycle, "_cleanup"),
        )
    )


def new_ledger(tmp_path_factory, before):
    paths = set(tmp_path_factory.getbasetemp().glob("managed-handoff-work[0-9]*/work.json"))
    added = paths - before
    assert len(added) == 1
    return json.loads(added.pop().read_text(encoding="utf8"))


def test_should_preserve_dispatch_attestation_and_forward_and_restore_profile(tmp_path_factory):
    before = set(tmp_path_factory.getbasetemp().glob("managed-handoff-work[0-9]*/work.json"))
    dispatch = raw_dispatch()
    forwarded = []
    previous = sys.getprofile()

    def sentinel():
        return 1

    def prior(frame, event, arg):
        if event == "call" and frame.f_code is sentinel.__code__:
            forwarded.append(True)

    observer = None
    try:
        sys.setprofile(prior)
        observer = native.observe_native_work(tmp_path_factory, zero_native_limits())
        next(observer)
        assert all(a is b for a, b in zip(raw_dispatch(), dispatch, strict=True))
        require_original_metadata_dispatch()
        assert sentinel() == 1
        observer.close()
        assert sys.getprofile() is prior
    finally:
        if observer is not None:
            observer.close()
        sys.setprofile(previous)
    assert forwarded == [True]
    assert all(value == 0 for value in new_ledger(tmp_path_factory, before)["actual"].values())


def test_should_preserve_primary_error_and_reset_fault_and_restore_profile(tmp_path_factory):
    before = set(tmp_path_factory.getbasetemp().glob("managed-handoff-work[0-9]*/work.json"))
    previous = sys.getprofile()
    primary = RuntimeError("original observer body failure")
    observer = native.observe_native_work(tmp_path_factory, zero_native_limits())
    try:
        next(observer)
        native.FAIL_RESTORE_COPY[0] = True
        with pytest.raises(RuntimeError) as raised:
            observer.throw(primary)
        assert raised.value is primary
        assert sys.getprofile() is previous
        assert native.FAIL_RESTORE_COPY == [False]
    finally:
        observer.close()
        sys.setprofile(previous)
    assert all(value == 0 for value in new_ledger(tmp_path_factory, before)["actual"].values())


def test_should_charge_original_cleanup_before_body_and_record_failed_cap(tmp_path_factory):
    before = set(tmp_path_factory.getbasetemp().glob("managed-handoff-work[0-9]*/work.json"))
    previous = sys.getprofile()
    observer = native.observe_native_work(tmp_path_factory, zero_native_limits())
    try:
        next(observer)
        with pytest.raises(AssertionError, match="original call cap exceeded: cleanup_attempts"):
            native.ManagedDataLifecycle._cleanup(
                cast(native.ManagedDataLifecycle, None), (), "expired"
            )
    finally:
        observer.close()
        sys.setprofile(previous)
    ledger = new_ledger(tmp_path_factory, before)
    assert ledger["actual"]["cleanup_attempts"] == 1
    assert ledger["limits"]["cleanup_attempts"] == 0
    assert all(value == 0 for key, value in ledger["actual"].items() if key != "cleanup_attempts")
