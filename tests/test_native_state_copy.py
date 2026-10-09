"""Scalar graphs test independent state observation without native allocations."""

from contextvars import copy_context
from copy import deepcopy
import gc
from threading import Thread
from typing import Any
import weakref

import pytest

from src.core.native_model_copy import ModelCopyLimits, observe_model_copies
from src.core.native_state_copy import copy_native_state, observe_state_copies
from test_native_model_copy import Graph, copy_once

LIMITS = ModelCopyLimits(2, 4, 32)


@pytest.mark.parametrize("state_first", [True, False])
def test_should_keep_actual_model_and_state_memos_separate_when_scopes_overlap(state_first):
    model = Graph()
    state = {"row": model.child, "alias": model.child}
    events = []

    def model_observer(stage, read, lookup):
        assert read().source is model
        if stage == "copied":
            assert lookup(model) is read().target
        events.append(("model", stage))

    def state_observer(stage, read, lookup):
        assert read().source is state
        if stage == "copied":
            target = read().target
            assert isinstance(target, dict)
            assert lookup(state) is target
            assert lookup(model.child) is target["row"] is target["alias"]
            assert target["row"] is not model.child
        events.append(("state", stage))

    model_scope = observe_model_copies(model, model_observer, LIMITS)
    state_scope = observe_state_copies(state, state_observer, LIMITS)
    first, second = (state_scope, model_scope) if state_first else (model_scope, state_scope)
    with first:
        with second:
            copy_once(model)
            copy_native_state(state)
        if state_first:
            copy_native_state(state)
        else:
            copy_once(model)
    assert len(events) == 6


def test_should_refuse_wrong_root_nesting_reopen_and_preserve_original_channel():
    state = {"row": Graph()}
    scope = observe_state_copies(state, lambda *_: None, LIMITS)
    with scope as window:
        with pytest.raises(ValueError, match="source differs"):
            copy_native_state(dict(state))
        with pytest.raises(ValueError, match="nest"):
            with observe_state_copies(state, lambda *_: None, LIMITS):
                pytest.fail("nested state observation")
        copy_native_state(state)
    assert window._source is None and window._observer is None
    with pytest.raises(ValueError, match="reopen"):
        scope.__enter__()
    assert copy_native_state(state)["row"] is not state["row"]


@pytest.mark.parametrize("source", [None, [], (), Graph()])
def test_should_require_exact_original_dictionary_for_explicit_observation(source):
    with pytest.raises(ValueError, match="original dictionary"):
        observe_state_copies(source, lambda *_: None, LIMITS)


@pytest.mark.parametrize("stage_fault", ["before_copy", "copied", "copier"])
def test_should_spend_failed_state_copy_without_poisoning_model_channel(stage_fault):
    class Broken:
        def __deepcopy__(self, memo):
            raise RuntimeError("copier fault")

    model = Graph()
    state = {"row": Broken() if stage_fault == "copier" else model.child}

    def observer(stage, read, lookup):
        if stage == stage_fault:
            raise RuntimeError("observer fault")

    with observe_model_copies(model, lambda *_: None, LIMITS):
        with observe_state_copies(state, observer, LIMITS) as window:
            with pytest.raises(RuntimeError, match="fault"):
                copy_native_state(state)
            assert window._copies == 1 and window._failed and not window._inflight
            with pytest.raises(ValueError, match="failed"):
                copy_native_state(state)
            copy_once(model)
        copy_once(model)


@pytest.mark.parametrize("operation", ["begin", "read", "lookup", "close"])
def test_should_refuse_inherited_context_and_keep_original_state_reader(operation):
    state = {"row": Graph()}

    def observer(stage, read, lookup):
        context = copy_context()
        if operation == "read":
            with pytest.raises(ValueError, match="original context"):
                context.run(read)
        if operation == "lookup" and stage == "copied":
            with pytest.raises(ValueError, match="original context"):
                context.run(lookup, state)
        assert read().source is state
        if stage == "copied":
            assert lookup(state) is read().target

    scope = observe_state_copies(state, observer, LIMITS)
    with scope:
        context = copy_context()
        if operation == "begin":
            with pytest.raises(ValueError, match="original context"):
                context.run(copy_native_state, state)
        if operation == "close":
            with pytest.raises(ValueError, match="original context"):
                context.run(scope.__exit__, None, None, None)
        copy_native_state(state)


def test_should_refuse_inherited_thread_and_join_before_original_copy():
    state = {"row": Graph()}
    errors = []
    with observe_state_copies(state, lambda *_: None, LIMITS):
        context = copy_context()

        def foreign():
            try:
                copy_native_state(state)
            except ValueError as error:
                errors.append(str(error))

        worker = Thread(target=lambda: context.run(foreign))
        worker.start()
        worker.join(1)
        assert not worker.is_alive() and len(errors) == 1
        copy_native_state(state)


def test_should_expire_saved_readers_between_callbacks_and_release_raw_graphs():
    state = {"row": Graph()}
    original = weakref.ref(state["row"])
    handles: list[Any] = []
    targets: list[Any] = []

    def observer(stage, read, lookup):
        for old_read, old_lookup in handles:
            with pytest.raises(ValueError, match="synchronous"):
                old_read()
            with pytest.raises(ValueError, match="synchronous"):
                old_lookup(None)
        if stage == "copied":
            target = read().target
            assert isinstance(target, dict)
            targets.append(weakref.ref(target["row"]))
        handles.append((read, lookup))

    with observe_state_copies(state, observer, LIMITS):
        target = copy_native_state(state)
        del target
        gc.collect()
        assert all(reference() is None for reference in targets)
    del state, observer
    gc.collect()
    assert original() is None
    for read, lookup in handles:
        with pytest.raises(ValueError, match="synchronous"):
            read()


@pytest.mark.parametrize("cap", ["copies", "notifications", "reads"])
def test_should_enforce_state_limits_before_copier_or_read_without_refunding(cap, monkeypatch):
    from src.core import native_state_copy

    state = {"row": Graph()}
    calls = []

    def copier(*args):
        calls.append(args)
        return deepcopy(*args)

    monkeypatch.setattr(native_state_copy, "deepcopy", copier)
    limits = ModelCopyLimits(1, 1 if cap == "notifications" else 2, 1 if cap == "reads" else 8)

    def observer(stage, read, lookup):
        read()
        if cap == "reads":
            read()

    with observe_state_copies(state, observer, limits) as window:
        if cap == "copies":
            copy_native_state(state)
            with pytest.raises(ValueError, match="attempt limit"):
                copy_native_state(state)
            assert len(calls) == 1 and window._copies == 1
        else:
            with pytest.raises(ValueError, match="notification limit|read limit"):
                copy_native_state(state)
            assert calls == []


def test_should_refuse_observer_reentry_and_admission_before_actual_copier(monkeypatch):
    from src.core import native_state_copy

    state = {"row": Graph()}

    def observer(stage, read, lookup):
        assert stage == "before_copy" and read().target is None
        with pytest.raises(ValueError, match="reenter"):
            copy_native_state(state)
        raise ValueError("admission refused")

    monkeypatch.setattr(native_state_copy, "deepcopy", lambda *_: pytest.fail("copier entered"))
    with observe_state_copies(state, observer, LIMITS):
        with pytest.raises(ValueError, match="admission refused"):
            copy_native_state(state)
