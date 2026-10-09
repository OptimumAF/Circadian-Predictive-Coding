"""Sixteen original timed graphs prove final dispatch, scalars and time lease.

Birth-installed ports and the original once-only training helper are preserved.
Outer assertions prove actual callback chronology; no final provider read occurs.
"""

from dataclasses import asdict, dataclass
import json

import pytest

import src.app.managed_data_lifecycle as lifecycle
import src.app.expiry_inbox_observation as observation
from src.core.data_lifecycle import DataConsent
from src.core.data_retention import DataCleanupReport
from src.core.inbox_origin import InboxOriginData
from src.app.expiry_validation_proof import require_original_metadata_dispatch
from src.app.managed_replay_origins import ManagedReplayOrigins
from test_automatic_inbox_expiry import (
    AutomaticWork,
    FIRST,
    _assert_successful_raw_cleanup,
    _assert_unpublished,
    _authority,
    _fresh_automatic,
)
from test_expiry_clock_prior import ClockPort
from test_expiry_release_proof import _assert_released


@pytest.fixture(scope="module")
def time_work(tmp_path_factory):
    work = AutomaticWork()
    try:
        yield work
    finally:
        (tmp_path_factory.mktemp("time-proof-work") / "work.json").write_text(
            json.dumps(asdict(work), indent=2), encoding="utf8"
        )
    assert work.graphs == work.updates == 16
    assert work.learners == 48 and work.forks == 32
    assert work.arrays == 96 and work.array_bytes == 1152
    assert work.consumed_arrays == work.native_copy_arrays == 32
    assert work.consumed_bytes == work.native_copy_bytes == 384
    assert work.cleanup_calls == 32 and work.post_erase_probes == 16
    assert work.pending_consumed is None


@dataclass
class Calls:
    calls: int = 0
    fired: int = 0
    unlocked: bool = True
    unsafe: int = 0
    probes: int = 0
    acquired: bool = False


def _timed(work):
    provider = ClockPort()
    graph, ports = _fresh_automatic(work, wall_clock=provider, max_retention_seconds=1000)
    assert provider.calls == 31 and provider.armed_calls == 0
    provider.action = lambda: 11.0
    graph.clock.advance_to(122)
    return graph, ports, provider, _authority(graph)


def _refused(graph):
    with pytest.raises(lifecycle.ExpiryObservationError) as error:
        graph.life.expire()
    assert error.value.code == "expiry-observation-refused"
    return error.value.report


def _assert_refused_raw(graph, report, before, provider, *, busy=False):
    _assert_successful_raw_cleanup(graph, report, before)
    _assert_unpublished(graph, before)
    assert provider.calls == 32 and provider.armed_calls == 1
    assert graph.life._time_gate.locked() is busy
    _assert_released(graph)


def _delegate(work, name, count):
    graph, ports, provider, before = _timed(work)
    original = getattr(lifecycle.ManagedDataLifecycle, name)
    calls = Calls()

    def delegate():
        calls.calls += 1
        return original(graph.life)

    graph.life.__dict__[name] = delegate
    report = _refused(graph)
    assert calls.calls == count and ports.fired == 0
    _assert_refused_raw(graph, report, before, provider)


def test_should_refuse_tick_instance_delegate_while_original_raw_path_runs(time_work):
    _delegate(time_work, "_tick", 2)


def test_should_refuse_elapsed_instance_delegate_while_original_raw_path_runs(time_work):
    _delegate(time_work, "_elapsed", 1)


def test_should_refuse_elapsed_leased_instance_delegate_while_original_raw_path_runs(time_work):
    _delegate(time_work, "_elapsed_leased", 1)


def _late_helper(work, monkeypatch, name):
    graph, ports, provider, before = _timed(work)
    calls = Calls()

    def poison(self):
        calls.unsafe += 1
        raise RuntimeError("late time helper must not be invoked")

    def change_after_probe():
        calls.fired += 1
        monkeypatch.setattr(lifecycle.ManagedDataLifecycle, name, poison)

    ports.on_last_footprint = change_after_probe
    report = _refused(graph)
    assert ports.fired == calls.fired == 1 and calls.unsafe == 0
    _assert_refused_raw(graph, report, before, provider)


def test_should_refuse_late_tick_class_change_without_invoking_it(time_work, monkeypatch):
    _late_helper(time_work, monkeypatch, "_tick")


def test_should_refuse_late_elapsed_class_change_without_invoking_it(time_work, monkeypatch):
    _late_helper(time_work, monkeypatch, "_elapsed")


def test_should_refuse_late_elapsed_leased_class_change_without_invoking_it(time_work, monkeypatch):
    _late_helper(time_work, monkeypatch, "_elapsed_leased")


def _report_boundary(graph, monkeypatch, action):
    calls = Calls()

    def report(*args, **kwargs):
        result = DataCleanupReport(*args, **kwargs)
        calls.calls += 1
        assert not graph.runtime._inbox._experiences and not graph.runtime._inbox._labels
        assert not graph.runtime._candidate.model.rows
        assert graph.life._failed is False and graph.life._retention_fault is False
        assert graph.life._auxiliary_started_at is None
        action(calls)
        calls.fired += 1
        return result

    monkeypatch.setattr(lifecycle, "DataCleanupReport", report)
    return calls


def _second_validation(graph, monkeypatch, action):
    calls = Calls()
    original = DataCleanupReport.__post_init__

    def validate(report):
        calls.calls += 1
        calls.unlocked = calls.unlocked and not graph.life._time_gate.locked()
        original(report)
        if calls.calls == 2:
            action(calls)
            calls.fired += 1
            monkeypatch.setattr(DataCleanupReport, "__post_init__", original)

    monkeypatch.setattr(DataCleanupReport, "__post_init__", validate)
    return calls


def test_should_refuse_elapsed_nan_from_second_report_validation(time_work, monkeypatch):
    graph, ports, provider, before = _timed(time_work)
    calls = _second_validation(
        graph, monkeypatch, lambda _: setattr(graph.life, "_last_seconds", float("nan"))
    )
    report = _refused(graph)
    assert calls.calls == 2 and calls.fired == 1 and calls.unlocked and ports.fired == 0
    assert graph.life._last_seconds != graph.life._last_seconds
    graph.life._last_seconds = 11.0
    _assert_refused_raw(graph, report, before, provider)


def test_should_refuse_changed_last_tick_after_report_construction(time_work, monkeypatch):
    graph, ports, provider, before = _timed(time_work)
    calls = _report_boundary(graph, monkeypatch, lambda _: setattr(graph.life, "_last_tick", 121))
    report = _refused(graph)
    assert calls.calls == calls.fired == 1 and ports.fired == 0 and graph.life._last_tick == 121
    graph.life._last_tick = 122
    _assert_refused_raw(graph, report, before, provider)


def test_should_refuse_restored_auxiliary_anchor_after_report_construction(time_work, monkeypatch):
    graph, ports, provider, before = _timed(time_work)
    calls = _report_boundary(
        graph, monkeypatch, lambda _: setattr(graph.life, "_auxiliary_started_at", 10.0)
    )
    report = _refused(graph)
    assert (
        calls.calls == calls.fired == 1
        and ports.fired == 0
        and graph.life._auxiliary_started_at == 10.0
    )
    graph.life._auxiliary_started_at = None
    _assert_refused_raw(graph, report, before, provider)


def test_should_refuse_nonboolean_healthy_flag_after_report_construction(time_work, monkeypatch):
    graph, ports, provider, before = _timed(time_work)
    calls = _report_boundary(graph, monkeypatch, lambda _: setattr(graph.life, "_failed", 0))
    report = _refused(graph)
    assert calls.calls == calls.fired == 1 and ports.fired == 0 and type(graph.life._failed) is int
    graph.life._failed = False
    _assert_refused_raw(graph, report, before, provider)


def test_should_refuse_genuinely_busy_original_final_time_gate_without_releasing_it(
    time_work, monkeypatch
):
    graph, ports, provider, before = _timed(time_work)
    gate = graph.life._time_gate

    def acquire(calls):
        calls.acquired = gate.acquire(blocking=False)

    calls = _report_boundary(graph, monkeypatch, acquire)
    try:
        report = _refused(graph)
        assert calls.calls == calls.fired == 1 and calls.acquired and ports.fired == 0
        assert graph.life._time_gate is gate and gate.locked()
        _assert_refused_raw(graph, report, before, provider, busy=True)
    finally:
        if calls.acquired:
            gate.release()
    assert not gate.locked()


def test_should_publish_original_timed_graph_after_footprint_without_extra_provider_read(time_work):
    graph, ports, provider, before = _timed(time_work)
    calls = Calls()

    def observe():
        calls.fired += 1
        calls.unlocked = calls.unlocked and not graph.life._time_gate.locked()

    ports.on_last_footprint = observe
    report = graph.life.expire()
    _assert_successful_raw_cleanup(graph, report, before)
    assert not graph.history._records and FIRST in graph.history._erased_records
    assert ports.fired == calls.fired == 1 and calls.unlocked
    assert provider.calls == 32 and provider.armed_calls == 1
    assert graph.life._last_seconds == 11.0 and not graph.life._time_gate.locked()
    _assert_released(graph)


def _late_metadata(work, monkeypatch, *, getter):
    graph, ports, provider, before = _timed(work)

    def install(calls):
        def forbidden(*args):
            calls.unsafe += 1
            acquired = graph.life._time_gate.acquire(blocking=False)
            calls.probes += int(acquired)
            if acquired:
                graph.life._time_gate.release()
            raise RuntimeError("late metadata dispatch must not execute")

        if getter:
            monkeypatch.setattr(DataConsent, "training", property(forbidden), raising=False)
        else:
            monkeypatch.setattr(InboxOriginData, "__post_init__", forbidden)

    calls = _second_validation(graph, monkeypatch, install)
    report = _refused(graph)
    assert calls.calls == 2 and calls.fired == 1 and calls.unlocked
    assert calls.unsafe == calls.probes == ports.fired == 0
    # Restore only after the observed refusal, for the existing raw oracle.
    monkeypatch.undo()
    _assert_refused_raw(graph, report, before, provider)


def test_should_refuse_late_deep_metadata_validator_without_calling_it_under_time_gate(
    time_work, monkeypatch
):
    _late_metadata(time_work, monkeypatch, getter=False)


def test_should_refuse_late_consumed_consent_getter_without_calling_it_under_time_gate(
    time_work, monkeypatch
):
    _late_metadata(time_work, monkeypatch, getter=True)


class ForeignKey:
    """A collision must be rejected before its equality code can execute."""

    def __init__(self, original):
        self.original = original
        self.calls = 0

    def __hash__(self):
        return hash(self.original)

    def __eq__(self, other):
        self.calls += 1
        return self.original == other


def _foreign_key_refusal(mapping, name):
    original = tuple(mapping.items())
    value = mapping.pop(name)
    foreign = ForeignKey(name)
    mapping[foreign] = value
    try:
        with pytest.raises(ValueError, match="original validator|original dataclass"):
            require_original_metadata_dispatch()
        assert foreign.calls == 0
    finally:
        mapping.clear()
        mapping.update(original)
    require_original_metadata_dispatch()


def test_should_refuse_foreign_keyword_default_key_before_invoking_equality():
    _foreign_key_refusal(ManagedReplayOrigins.__init__.__kwdefaults__, "retain_inbox_origins")


def test_should_refuse_foreign_dataclass_field_key_before_invoking_equality():
    _foreign_key_refusal(DataConsent.__dataclass_fields__, "training")


def test_should_refuse_late_observer_method_without_invoking_it_under_time_gate(
    time_work, monkeypatch
):
    graph, ports, provider, before = _timed(time_work)

    def install(calls):
        def forbidden(*args):
            calls.unsafe += 1
            acquired = graph.life._time_gate.acquire(blocking=False)
            calls.probes += int(acquired)
            if acquired:
                graph.life._time_gate.release()
            raise RuntimeError("late observer method must not execute")

        monkeypatch.setattr(observation._ExpiryObservation, "_require_original", forbidden)

    calls = _second_validation(graph, monkeypatch, install)
    report = _refused(graph)
    assert calls.calls == 2 and calls.fired == 1 and calls.unlocked
    assert calls.unsafe == calls.probes == ports.fired == 0
    monkeypatch.undo()
    _assert_refused_raw(graph, report, before, provider)


def test_should_refuse_cleared_commit_marker_from_second_report_validation(time_work, monkeypatch):
    graph, ports, provider, before = _timed(time_work)

    def clear(calls):
        scope = observation._CURRENT.get()
        assert scope.committed is True
        scope.committed = False

    calls = _second_validation(graph, monkeypatch, clear)
    report = _refused(graph)
    assert calls.calls == 2 and calls.fired == 1 and calls.unlocked and ports.fired == 0
    _assert_refused_raw(graph, report, before, provider)


def test_should_mark_refusal_without_invoking_a_foreign_scope_setter(monkeypatch):
    scope = object.__new__(observation._ExpiryObservation)
    calls = Calls()

    def forbidden(*args):
        calls.unsafe += 1
        raise RuntimeError("refusal must not invoke a foreign setter")

    monkeypatch.setattr(observation._ExpiryObservation, "__setattr__", forbidden)
    token = observation._CURRENT.set(scope)
    try:
        assert observation.observe_expiry_time_dispatch(None) is True
        assert scope.fault is True and calls.unsafe == 0
    finally:
        observation._CURRENT.reset(token)


def test_should_refuse_mutated_or_replaced_original_slot_name_cache_without_callbacks(monkeypatch):
    from src.app.expiry_validation_proof import require_validation_types

    cache = type.__getattribute__(observation._ExpiryObservation, "__dict__")["__slotnames__"]
    original = tuple(cache)
    foreign = ForeignKey(original[-1])
    try:
        cache[-1] = foreign
        with pytest.raises(ValueError, match="original slot-name cache"):
            require_validation_types(observation._OBSERVER_GETTER_PINS)
        assert foreign.calls == 0
    finally:
        cache[:] = original
    require_validation_types(observation._OBSERVER_GETTER_PINS)
    with monkeypatch.context() as patch:
        patch.setattr(observation._ExpiryObservation, "__slotnames__", list(original))
        with pytest.raises(ValueError, match="original validator/getter"):
            require_validation_types(observation._OBSERVER_GETTER_PINS)
    require_validation_types(observation._OBSERVER_GETTER_PINS)
