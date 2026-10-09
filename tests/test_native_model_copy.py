"""Pure graph copies qualify borrowed identity windows without native work."""

from contextvars import copy_context
from copy import deepcopy
from threading import Thread
import gc
import weakref
from typing import Any, Callable

import pytest

from src.core.native_model_copy import (
    ModelCopy,
    ModelCopyLimits,
    begin_model_copy,
    observe_model_copies,
)

LIMITS = ModelCopyLimits(2, 4, 32)


class Graph:
    def __init__(self):
        self.child = GraphLeaf()
        self.alias = self.child


class GraphLeaf:
    pass


def copy_once(source):
    window = begin_model_copy(source)
    assert window is not None
    try:
        memo: dict[int, Any] = {}
        target = deepcopy(source, memo)
        window.copied(target, memo)
        return target
    finally:
        window.end()


def test_should_bind_actual_source_target_and_alias_memo_without_reviving_old_readers():
    source = Graph()
    handles: list[tuple[Callable[[], ModelCopy], Callable[[object], object]]] = []
    stages = []

    def observer(stage, read, lookup):
        for old_read, old_lookup in handles:
            with pytest.raises(ValueError, match="synchronous"):
                old_read()
            with pytest.raises(ValueError, match="synchronous"):
                old_lookup(source)
        observed = read()
        assert observed.source is source
        if stage == "before_copy":
            assert observed.target is None
            with pytest.raises(ValueError, match="memo"):
                lookup(source)
        else:
            assert isinstance(observed.target, Graph)
            assert observed.target is not source
            assert lookup(source) is observed.target
            assert lookup(source.child) is observed.target.child
            assert observed.target.child is observed.target.alias
            with pytest.raises(ValueError, match="memo"):
                lookup(GraphLeaf())
        handles.append((read, lookup))
        stages.append(stage)

    with observe_model_copies(source, observer, LIMITS):
        first = copy_once(source)
        second = copy_once(source)
    assert first is not second and first.child is not source.child
    assert stages == ["before_copy", "copied", "before_copy", "copied"]
    for read, lookup in handles:
        with pytest.raises(ValueError, match="synchronous"):
            read()
        with pytest.raises(ValueError, match="synchronous"):
            lookup(source)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, 2**63])
@pytest.mark.parametrize("field", ["max_copies", "max_notifications", "max_reads"])
def test_should_refuse_invalid_limits_before_window_creation(value, field):
    values: dict[str, Any] = dict(max_copies=1, max_notifications=2, max_reads=8)
    values[field] = value
    with pytest.raises(ValueError, match="positive"):
        ModelCopyLimits(**values)


def test_should_refuse_wrong_source_and_nested_reopen_without_disturbing_original():
    source = Graph()
    scope = observe_model_copies(source, lambda *_: None, LIMITS)
    with scope as window:
        with pytest.raises(ValueError, match="source differs"):
            begin_model_copy(Graph())
        with pytest.raises(ValueError, match="nest"):
            with observe_model_copies(source, lambda *_: None, LIMITS):
                pytest.fail("nested")
        copy_once(source)
    assert window._source is None and window._observer is None
    with pytest.raises(ValueError, match="reopen"):
        scope.__enter__()
    assert begin_model_copy(source) is None


@pytest.mark.parametrize("fault_stage", ["before_copy", "copied"])
def test_should_spend_failed_attempt_and_release_callback_roots(fault_stage):
    source = Graph()
    weak_source, weak_child = weakref.ref(source), weakref.ref(source.child)
    handles = []

    def observer(stage, read, lookup):
        handles.append((read, lookup))
        if stage == fault_stage:
            raise RuntimeError("observer fault")

    with observe_model_copies(source, observer, LIMITS) as window:
        with pytest.raises(RuntimeError, match="observer fault"):
            copy_once(source)
        assert window._copies == 1
        with pytest.raises(ValueError, match="failed"):
            copy_once(source)
    del source, observer
    gc.collect()
    assert weak_source() is None and weak_child() is None
    for read, _ in handles:
        with pytest.raises(ValueError, match="synchronous"):
            read()


def test_should_refuse_copy_notification_and_read_caps_without_refunding():
    source = Graph()
    with observe_model_copies(source, lambda *_: None, ModelCopyLimits(1, 2, 8)) as window:
        copy_once(source)
        with pytest.raises(ValueError, match="attempt"):
            copy_once(source)
        assert window._copies == 1 and window._notifications == 2
    with observe_model_copies(source, lambda *_: pytest.fail("entered"), ModelCopyLimits(1, 1, 8)):
        with pytest.raises(ValueError, match="before allocation"):
            copy_once(source)

    def observer(stage, read, lookup):
        read()
        read()

    with observe_model_copies(source, observer, ModelCopyLimits(1, 2, 1)) as window:
        with pytest.raises(ValueError, match="read limit"):
            copy_once(source)
        assert window._reads == 1 and window._copies == 1


def test_should_refuse_callback_reentry_and_close_without_losing_original_window():
    source = Graph()

    def observer(stage, read, lookup):
        read()
        with pytest.raises(ValueError, match="reenter"):
            begin_model_copy(source)
        with pytest.raises(ValueError, match="quiescent"):
            scope.__exit__(None, None, None)

    scope = observe_model_copies(source, observer, LIMITS)
    with scope:
        copy_once(source)


def test_should_refuse_inherited_foreign_thread_and_foreign_context_close():
    source = Graph()
    errors = []
    scope = observe_model_copies(source, lambda *_: None, LIMITS)
    with scope as window:
        context = copy_context()

        def foreign():
            try:
                begin_model_copy(source)
            except ValueError as error:
                errors.append(error)

        worker = Thread(target=lambda: context.run(foreign))
        worker.start()
        worker.join(1)
        assert not worker.is_alive() and len(errors) == 1
        with pytest.raises(ValueError):
            context.run(scope.__exit__, None, None, None)
        assert window._source is source
        copy_once(source)


@pytest.mark.parametrize("bad", ["same", "none", "foreign_memo", "missing_memo"])
def test_should_refuse_unobserved_target_and_poison_aborted_copy(bad):
    source = Graph()
    with observe_model_copies(source, lambda *_: None, LIMITS) as window:
        window.begin(source)
        target = source if bad == "same" else None if bad == "none" else Graph()
        memo = (
            [] if bad == "foreign_memo" else {} if bad == "missing_memo" else {id(source): target}
        )
        try:
            with pytest.raises(ValueError, match="actual distinct"):
                window.copied(target, memo)
        finally:
            window.end()
        with pytest.raises(ValueError, match="failed"):
            begin_model_copy(source)


def test_should_abort_failed_copier_and_release_window_roots():
    class Broken:
        def __deepcopy__(self, memo):
            raise RuntimeError("copier fault")

    source = Broken()
    with observe_model_copies(source, lambda *_: None, LIMITS) as window:
        with pytest.raises(RuntimeError, match="copier fault"):
            copy_once(source)
        assert window._failed and window._copies == 1 and not window._inflight
    assert window._source is None


def test_should_release_copied_graph_while_window_and_saved_readers_remain():
    source = Graph()
    handles: list[Any] = []
    targets: list[Any] = []

    def observer(stage, read, lookup):
        if stage == "copied":
            target = read().target
            assert isinstance(target, Graph)
            targets.append(weakref.ref(target))
            targets.append(weakref.ref(target.child))
        handles.append((read, lookup))

    with observe_model_copies(source, observer, LIMITS):
        target = copy_once(source)
        del target
        gc.collect()
        assert all(reference() is None for reference in targets)
        for read, lookup in handles:
            with pytest.raises(ValueError, match="synchronous"):
                read()


def test_should_refuse_before_copy_without_invoking_copier(monkeypatch):
    import sys

    source = Graph()
    calls = []

    def copier(*args):
        calls.append(args)
        pytest.fail("copier must not run after before-copy refusal")

    def observer(stage, read, lookup):
        assert stage == "before_copy" and read().target is None
        raise ValueError("original admission refused")

    monkeypatch.setattr(sys.modules[__name__], "deepcopy", copier)
    with observe_model_copies(source, observer, LIMITS):
        with pytest.raises(ValueError, match="admission refused"):
            copy_once(source)
    assert calls == []


@pytest.mark.parametrize("operation", ["begin", "read", "lookup", "close"])
def test_should_refuse_inherited_same_thread_context_and_preserve_original(operation):
    source = Graph()

    def observer(stage, read, lookup):
        context = copy_context()
        if operation == "read":
            with pytest.raises(ValueError, match="original context"):
                context.run(read)
        elif operation == "lookup" and stage == "copied":
            with pytest.raises(ValueError, match="original context"):
                context.run(lookup, source)
        assert read().source is source
        if stage == "copied":
            assert lookup(source) is read().target

    scope = observe_model_copies(source, observer, LIMITS)
    with scope as window:
        context = copy_context()
        if operation == "begin":
            with pytest.raises(ValueError, match="original context"):
                context.run(begin_model_copy, source)
        if operation == "close":
            with pytest.raises(ValueError, match="original context"):
                context.run(scope.__exit__, None, None, None)
        assert window._source is source
        copy_once(source)
    assert window._source is None and window._context_token is None
