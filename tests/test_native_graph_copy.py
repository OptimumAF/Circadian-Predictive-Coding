"""Pure sequence identity/expiry/caps across changing roots and copy channels."""

from contextvars import copy_context
import gc
from threading import Thread
from typing import Any
from weakref import ref

import pytest

from src.core.native_graph_copy import copy_graph, observe_graph_copies
from src.core.native_model_copy import ModelCopyLimits, observe_model_copies
from src.core.native_state_copy import copy_native_state, observe_state_copies
from test_native_model_copy import Graph, copy_once

LIMITS = ModelCopyLimits(3, 6, 64)


def test_should_follow_actual_memos_across_roots_and_expire_earlier_readers():
    producer = Graph()
    source = {"row": producer.child, "alias": producer.child}
    handles: list[Any] = []
    events = []

    def observe(owner, kind, stage, read, lookup):
        assert owner is producer
        for old_read, old_lookup in handles:
            with pytest.raises(ValueError, match="synchronous"):
                old_read()
            with pytest.raises(ValueError, match="synchronous"):
                old_lookup(None)
        original = read().source
        assert isinstance(original, dict)
        if stage == "copied":
            target = read().target
            assert isinstance(target, dict)
            assert lookup(original) is target
            assert lookup(original["row"]) is target["row"] is target["alias"]
            assert target["row"] is not original["row"]
        handles.append((read, lookup))
        events.append((kind, stage))

    with observe_graph_copies(observe, LIMITS) as sequence:
        snapshot = copy_graph(producer, "checkpoint_capture", source)
        restored = copy_graph(producer, "checkpoint_restore_state", snapshot)
        cursor = copy_graph(producer, "inbox_materialize", restored)
        assert sequence._window is not None and sequence._window._copies == 3
        assert sequence._window._source is observe
    assert cursor["row"] is not source["row"]
    assert [kind for kind, stage in events if stage == "copied"] == [
        "checkpoint_capture",
        "checkpoint_restore_state",
        "inbox_materialize",
    ]
    assert sequence._anchor is sequence._observer is sequence._event is None


@pytest.mark.parametrize("state_first", [True, False])
def test_should_share_exact_copier_memo_between_graph_and_state_observers(state_first):
    producer = Graph()
    source = {"row": producer.child}
    seen = []

    def state(stage, read, lookup):
        if stage == "copied":
            seen.append((read().target, lookup(producer.child)))

    def graph(owner, kind, stage, read, lookup):
        assert owner is producer and kind == "native_snapshot"
        if stage == "copied":
            assert seen[-1] == (read().target, lookup(producer.child))

    graph_scope = observe_graph_copies(graph, LIMITS)
    state_scope = observe_state_copies(source, state, LIMITS)
    first, second = (state_scope, graph_scope) if state_first else (graph_scope, state_scope)
    with first:
        with second:
            target = copy_native_state(source, producer=producer, kind="native_snapshot")
    assert target is seen[0][0]


def test_should_keep_model_and_graph_channels_independent_when_active_together():
    producer = Graph()
    source = {"row": producer.child}
    with observe_model_copies(producer, lambda *_: None, LIMITS):
        with observe_graph_copies(lambda *_: None, LIMITS):
            copy_once(producer)
            copy_graph(producer, "checkpoint_capture", source)
        copy_once(producer)


@pytest.mark.parametrize("operation", ["copy", "read", "lookup", "close"])
def test_should_refuse_inherited_same_thread_context_without_losing_original(operation):
    producer = Graph()
    source = {"row": producer.child}

    def observer(owner, kind, stage, read, lookup):
        context = copy_context()
        if operation == "read":
            with pytest.raises(ValueError, match="original context"):
                context.run(read)
        if operation == "lookup" and stage == "copied":
            with pytest.raises(ValueError, match="original context"):
                context.run(lookup, source)
        assert read().source is source

    scope = observe_graph_copies(observer, LIMITS)
    with scope:
        if operation == "copy":
            with pytest.raises(ValueError, match="original context"):
                copy_context().run(copy_graph, producer, "checkpoint_capture", source)
        if operation == "close":
            with pytest.raises(ValueError, match="original context"):
                copy_context().run(scope.__exit__, None, None, None)
        copy_graph(producer, "checkpoint_capture", source)


def test_should_refuse_inherited_foreign_thread_and_join_before_original_copy():
    producer = Graph()
    errors = []
    with observe_graph_copies(lambda *_: None, LIMITS):
        context = copy_context()

        def foreign():
            try:
                copy_graph(producer, "checkpoint_capture", {})
            except ValueError as error:
                errors.append(str(error))

        worker = Thread(target=lambda: context.run(foreign))
        worker.start()
        worker.join(1)
        assert not worker.is_alive() and len(errors) == 1
        copy_graph(producer, "checkpoint_capture", {})


@pytest.mark.parametrize("fault", ["before_copy", "copied", "copier"])
def test_should_poison_sequence_after_fault_without_renewing_for_new_root(fault):
    class Broken:
        def __deepcopy__(self, memo):
            raise RuntimeError("copier fault")

    producer = Graph()
    source = {"row": Broken() if fault == "copier" else producer.child}

    def observer(owner, kind, stage, read, lookup):
        if stage == fault:
            raise RuntimeError("observer fault")

    with observe_graph_copies(observer, LIMITS) as sequence:
        with pytest.raises(RuntimeError, match="fault"):
            copy_graph(producer, "checkpoint_capture", source)
        assert sequence._window is not None and sequence._window._copies == 1
        assert sequence._window._source is observer and sequence._event is None
        with pytest.raises(ValueError, match="failed"):
            copy_graph(producer, "inbox_capture", {})


@pytest.mark.parametrize("cap", ["copies", "notifications", "reads"])
def test_should_enforce_one_original_allowance_across_new_roots_before_allocation(cap, monkeypatch):
    from src.core import native_graph_copy

    producer = Graph()
    source = {"row": producer.child}
    calls = []
    original = native_graph_copy.deepcopy

    def copy(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(native_graph_copy, "deepcopy", copy)
    limits = ModelCopyLimits(1, 1 if cap == "notifications" else 2, 1 if cap == "reads" else 8)

    def observer(owner, kind, stage, read, lookup):
        read()
        if cap == "reads":
            read()

    with observe_graph_copies(observer, limits):
        if cap == "copies":
            copy_graph(producer, "checkpoint_capture", source)
            with pytest.raises(ValueError, match="attempt limit"):
                copy_graph(producer, "inbox_capture", {})
            assert len(calls) == 1
        else:
            with pytest.raises(ValueError, match="notification limit|read limit"):
                copy_graph(producer, "checkpoint_capture", source)
            assert calls == []


def test_should_refuse_nested_reopen_reentry_and_preserve_original_scope():
    producer = Graph()

    def observer(owner, kind, stage, read, lookup):
        with pytest.raises(ValueError, match="reenter"):
            copy_graph(producer, "inbox_capture", {})
        with pytest.raises(ValueError, match="quiescent"):
            scope.__exit__(None, None, None)

    scope = observe_graph_copies(observer, LIMITS)
    with scope:
        with pytest.raises(ValueError, match="nest"):
            with observe_graph_copies(lambda *_: None, LIMITS):
                pytest.fail("nested")
        copy_graph(producer, "checkpoint_capture", {})
    with pytest.raises(ValueError, match="reopen"):
        scope.__enter__()


@pytest.mark.parametrize("bad", ["producer", "kind", "source"])
def test_should_require_actual_producer_supported_kind_and_source_before_copy(bad):
    values: list[Any] = [Graph(), "checkpoint_capture", {}]
    values[{"producer": 0, "kind": 1, "source": 2}[bad]] = None if bad != "kind" else "foreign"
    with observe_graph_copies(lambda *_: pytest.fail("callback entered"), LIMITS):
        with pytest.raises(ValueError, match="actual producer"):
            copy_graph(*values)


def test_should_refuse_foreign_actual_producer_via_original_observer_before_copy(monkeypatch):
    from src.core import native_graph_copy

    producer = Graph()

    def observe(owner, kind, stage, read, lookup):
        if owner is not producer:
            raise ValueError("original producer differs")

    monkeypatch.setattr(native_graph_copy, "deepcopy", lambda *_: pytest.fail("copier entered"))
    with observe_graph_copies(observe, LIMITS) as sequence:
        with pytest.raises(ValueError, match="producer differs"):
            copy_graph(Graph(), "checkpoint_capture", {})
        assert sequence._window is not None and sequence._window._copies == 1


def test_should_release_producer_source_and_target_while_saved_readers_survive():
    producer = Graph()
    source = {"row": Graph()}
    weak_producer, weak_source = ref(producer), ref(source["row"])
    handles: list[Any] = []
    targets: list[Any] = []

    def observer(owner, kind, stage, read, lookup):
        handles.append((read, lookup))
        if stage == "copied":
            target = read().target
            assert isinstance(target, dict)
            targets.append(ref(target["row"]))

    with observe_graph_copies(observer, LIMITS) as scope:
        target = copy_graph(producer, "checkpoint_capture", source)
        del target
        gc.collect()
        assert all(weak() is None for weak in targets)
    del producer, source, observer
    gc.collect()
    assert weak_producer() is weak_source() is None
    assert scope._observer is scope._anchor is scope._event is None
    for read, lookup in handles:
        with pytest.raises(ValueError, match="synchronous"):
            read()


def test_should_keep_unobserved_copy_behavior_and_default_state_helper():
    source = {"row": Graph()}
    target = copy_graph(Graph(), "checkpoint_capture", source)
    assert target["row"] is not source["row"]
    copied = copy_native_state(source)
    assert copied["row"] is not source["row"]


@pytest.mark.parametrize("observer", [None, object()])
def test_should_refuse_noncallable_observer_before_scope_creation(observer):
    with pytest.raises(ValueError, match="trusted observer"):
        observe_graph_copies(observer, LIMITS)


def test_should_refuse_premature_close_without_invalidating_future_original_entry():
    scope = observe_graph_copies(lambda *_: None, LIMITS)
    with pytest.raises(ValueError, match="original live scope"):
        scope.__exit__(None, None, None)
    with scope:
        copy_graph(Graph(), "checkpoint_capture", {})
