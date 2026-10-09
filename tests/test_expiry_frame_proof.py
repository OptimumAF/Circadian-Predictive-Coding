"""Original active expiry frames survive mutable late method dispatch.

Four original numeric graphs use the same birth-installed cleanup ports. Actual
callbacks record only scalars and the current scope; outer assertions establish
refusal, primary-error identity, mandatory raw cleanup and original release.
"""

from dataclasses import asdict, dataclass
import json
from typing import Any

import pytest

import src.app.expiry_inbox_observation as observation
from src.app.managed_data_lifecycle import ManagedDataLifecycle
from test_automatic_inbox_expiry import (
    AutomaticWork,
    FIRST,
    _assert_authority,
    _assert_unpublished,
    _authority,
    _fresh_automatic,
    _observer_refusal,
)
from test_expiry_release_proof import _assert_released


@pytest.fixture(scope="module")
def frame_work(tmp_path_factory):
    work = AutomaticWork()
    try:
        yield work
    finally:
        (tmp_path_factory.mktemp("frame-work") / "work.json").write_text(
            json.dumps(asdict(work), indent=2), encoding="utf8"
        )
    assert work.graphs == work.updates == 4
    assert work.learners == 12 and work.forks == 8
    assert work.arrays == 24 and work.array_bytes == 288
    assert work.consumed_arrays == work.native_copy_arrays == 8
    assert work.consumed_bytes == work.native_copy_bytes == 96
    assert work.cleanup_calls == 8 and work.post_erase_probes == 4
    assert work.pending_consumed is None


@dataclass
class FrameRecord:
    entered: int = 0
    exited: int = 0
    lexical: bool = False
    foreign_refused: bool = False
    scope: Any = None
    unchanged: bool = False
    released: bool = False


def _replacement(self):
    raise RuntimeError("replacement expiry must not run from the active original frame")


def _observe_original_frame(record):
    scope = observation._CURRENT.get()
    record.scope = scope
    record.lexical = observation._lexical_proof(scope) is scope.proof
    saved = (scope.proof, scope.context, scope.access, scope.close_access, scope.gate, scope.thread)
    # Direct callback/manual close is not the original lexical caller, even
    # though it can see the scope. A refusal must leave the genuine scope live.
    try:
        scope.close(scope.gate, scope.context, scope.close_access, scope.thread)
    except ValueError:
        record.foreign_refused = True
    current = (
        scope.proof,
        scope.context,
        scope.access,
        scope.close_access,
        scope.gate,
        scope.thread,
    )
    record.unchanged = (
        all(value is previous for value, previous in zip(current, saved))
        and observation._CURRENT.get() is scope
        and scope.gate.locked()
    )


def _assert_scope_closed(graph, record):
    scope = record.scope
    assert scope is not None and scope.closed
    assert all(
        getattr(scope, name) is None
        for name in (
            "ledger",
            "owner",
            "runtime",
            "history",
            "gate",
            "before",
            "proof",
            "staged",
            "context",
            "access",
            "close_access",
        )
    )
    assert observation._CURRENT.get() is None and observation._ACCESS.get() is None
    _assert_released(graph)


def test_should_clean_and_close_original_frame_called_by_changed_class_wrapper(
    frame_work, monkeypatch
):
    graph, ports = _fresh_automatic(frame_work)
    before = _authority(graph)
    original = ManagedDataLifecycle.expire
    record = FrameRecord()

    def wrapper(life):
        record.entered += 1
        try:
            return original(life)
        finally:
            record.exited += 1
            record.released = observation._CURRENT.get() is None and not graph.ledger._gate.locked()

    monkeypatch.setattr(ManagedDataLifecycle, "expire", wrapper)
    graph.clock.advance_to(122)
    _observer_refusal(graph, before)
    assert record.entered == record.exited == 1 and record.released
    assert ports.fired == 0
    _assert_released(graph)


def test_should_refuse_late_expiry_class_change_without_losing_original_lexical_scope(
    frame_work, monkeypatch
):
    graph, ports = _fresh_automatic(frame_work)
    before = _authority(graph)
    record = FrameRecord()

    def change_after_probe():
        monkeypatch.setattr(ManagedDataLifecycle, "expire", _replacement)
        _observe_original_frame(record)

    ports.on_last_footprint = change_after_probe
    graph.clock.advance_to(122)
    _observer_refusal(graph, before)
    assert ports.fired == 1 and record.lexical and record.foreign_refused and record.unchanged
    _assert_scope_closed(graph, record)


def test_should_refuse_late_expiry_code_change_without_losing_original_lexical_scope(
    frame_work, monkeypatch
):
    graph, ports = _fresh_automatic(frame_work)
    before = _authority(graph)
    original = ManagedDataLifecycle.expire
    record = FrameRecord()

    def change_after_probe():
        monkeypatch.setattr(original, "__code__", _replacement.__code__)
        _observe_original_frame(record)

    ports.on_last_footprint = change_after_probe
    graph.clock.advance_to(122)
    _observer_refusal(graph, before)
    assert ManagedDataLifecycle.expire is original
    assert original.__code__ is _replacement.__code__
    assert ports.fired == 1 and record.lexical and record.foreign_refused and record.unchanged
    _assert_scope_closed(graph, record)


def test_should_preserve_footprint_primary_after_late_class_change_and_close_original_gate(
    frame_work, monkeypatch
):
    graph, ports = _fresh_automatic(frame_work)
    before = _authority(graph)
    record = FrameRecord()
    primary = RuntimeError("original birth footprint callback failure")
    setter_calls = []

    def forbidden_setter(*args):
        setter_calls.append(True)
        raise RuntimeError("late scope setter must not replace primary")

    def fail_after_probe():
        monkeypatch.setattr(ManagedDataLifecycle, "expire", _replacement)
        _observe_original_frame(record)
        monkeypatch.setattr(observation._ExpiryObservation, "__setattr__", forbidden_setter)
        raise primary

    ports.on_last_footprint = fail_after_probe
    graph.clock.advance_to(122)
    with pytest.raises(RuntimeError) as failure:
        graph.life.expire()
    assert failure.value is primary
    assert not setter_calls
    assert ports.fired == 1 and record.lexical and record.foreign_refused and record.unchanged
    assert graph.life._failed and graph.runtime._stopped and not graph.ledger._poisoned
    assert not graph.runtime._candidate.model.rows
    assert graph.runtime._inbox._experiences and graph.runtime._inbox._labels
    assert graph.owner._revoked_keys == {FIRST} and not graph.runtime._inbox._erased
    _assert_authority(graph, before)
    _assert_unpublished(graph, before)
    _assert_scope_closed(graph, record)
