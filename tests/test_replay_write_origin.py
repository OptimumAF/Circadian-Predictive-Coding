"""Fresh fake replay windows, original producer composition and reference expiry."""

from contextvars import copy_context
from threading import Thread
import gc
import weakref
from typing import Any

import pytest

from src.app.replay_write_origin import ManagedReplayWriteAccess
from src.core.replay_write_origin import (
    ReplayWriteLimits,
    observe_replay_writes,
    begin_replay_write,
)
from test_managed_experience import setup, declaration, record

LIMITS = ReplayWriteLimits(16, 16, 64)


def test_should_report_actual_ranges_snapshots_and_retained_identities():
    model, features, targets, old, new = [object() for _ in range(5)]
    reports: list[Any] = []
    handles = []

    def observer(stage, read):
        reports.append((stage, read()))
        handles.append(read)

    with observe_replay_writes(model, features, targets, observer, LIMITS) as window:
        assert begin_replay_write(model, features, targets, [old], 2, "rows") is window
        window.before_copy(1, 1)
        window.copied(new)
        window.finish([new])
        for read in handles:
            with pytest.raises(ValueError, match="synchronous"):
                read()
    assert [r[0] for r in reports] == ["begin", "before_copy", "copied", "retained"]
    assert reports[0][1].retained == (old,) and reports[-1][1].retained == (new,)
    assert reports[2][1].snapshot is new and reports[2][1].row_start == 1
    assert all(
        r[1].model is model and r[1].features is features and r[1].targets is targets
        for r in reports
    )
    with pytest.raises(ValueError, match="expired"):
        window.finish([])


@pytest.mark.parametrize("foreign", ["model", "features", "targets"])
def test_should_refuse_foreign_reference_without_observer_entry(foreign):
    originals = dict(model=object(), features=object(), targets=object())
    with observe_replay_writes(
        **originals, observer=lambda *_: pytest.fail("entered"), limits=LIMITS
    ):
        changed = {**originals, foreign: object()}
        with pytest.raises(ValueError, match="differs"):
            begin_replay_write(**changed, retained=[], rows=1, mode="batch")


@pytest.mark.parametrize("rows,retained", [(17, 0), (True, 0), (0, 0), (1, 17)])
def test_should_refuse_input_and_retained_limits_before_reference_report(rows, retained):
    model, features, targets = object(), object(), object()
    with observe_replay_writes(model, features, targets, lambda *_: pytest.fail("entered"), LIMITS):
        with pytest.raises(ValueError, match="limit"):
            begin_replay_write(model, features, targets, [None] * retained, rows, "rows")


def test_should_preserve_charged_notifications_on_fault_and_refuse_repeat_write():
    model, features, targets = object(), object(), object()
    reports = []
    with observe_replay_writes(
        model,
        features,
        targets,
        lambda stage, read: reports.append(stage),
        ReplayWriteLimits(1, 1, 1),
    ) as window:
        begin_replay_write(model, features, targets, [], 1, "batch")
        with pytest.raises(ValueError, match="notification"):
            window.before_copy(0, 1)
        with pytest.raises(ValueError, match="already began"):
            begin_replay_write(model, features, targets, [], 1, "batch")
    assert reports == ["begin"] and window._notifications == 1


def test_should_refuse_nested_and_callback_reentry_with_original_scope_intact():
    model, features, targets = object(), object(), object()

    def observer(stage, read):
        read()
        with pytest.raises(ValueError, match="reenter"):
            window.before_copy(0, 1)
        with pytest.raises(ValueError, match="quiescent"):
            window.close()

    with observe_replay_writes(model, features, targets, observer, LIMITS) as window:
        with pytest.raises(ValueError, match="nest"):
            with observe_replay_writes(model, features, targets, observer, LIMITS):
                pytest.fail("nested")
        begin_replay_write(model, features, targets, [], 1, "batch")
        window.finish([])


def test_should_refuse_foreign_thread_with_inherited_context_and_join_it():
    model, features, targets = object(), object(), object()
    errors = []
    with observe_replay_writes(model, features, targets, lambda *_: None, LIMITS):
        inherited = copy_context()

        def foreign():
            try:
                begin_replay_write(model, features, targets, [], 1, "batch")
            except ValueError as error:
                errors.append(error)

        worker = Thread(target=lambda: inherited.run(foreign))
        worker.start()
        worker.join(1)
        assert not worker.is_alive()
        assert len(errors) == 1
        begin_replay_write(model, features, targets, [], 1, "batch")


def test_should_release_window_owned_payload_references_after_failure():
    class Payload:
        pass

    payload = Payload()
    reference = weakref.ref(payload)
    handles = []

    def observer(stage, read):
        handles.append(read)
        raise RuntimeError("observer fault")

    with pytest.raises(RuntimeError, match="observer fault"):
        with observe_replay_writes(payload, payload, payload, observer, LIMITS) as window:
            begin_replay_write(payload, payload, payload, [], 1, "batch")
    del payload
    gc.collect()
    assert reference() is None
    assert window._source is None and window._observer is None
    with pytest.raises(ValueError):
        handles[0]()


@pytest.mark.parametrize("fault", [False, True])
def test_should_bind_original_managed_producer_and_close_after_native_outcome(fault):
    owner, _, runtime, clock, gate, _, budget = setup()
    owner.declare(declaration())
    record(owner)
    clock.advance_to(3)
    candidate = runtime._candidate
    original = candidate.train_batch
    reports: list[Any] = []
    stages = []
    handles = []

    def observer(stage, read):
        value = read()
        reports.append((stage, value))
        handles.append(read)
        assert value.origin.learner is candidate
        assert value.origin.source is runtime._inbox._experiences[("e1", "s1")]
        assert value.origin.label is runtime._inbox._labels[("e1", "s1")]
        assert value.write.features is value.origin.features
        if fault and stage == "copied":
            raise RuntimeError("replay observer fault")

    access = ManagedReplayWriteAccess(
        lambda learner: learner,
        observer,
        LIMITS,
        update_observer=lambda stage, read: stages.append((stage, read())),
    )

    def train(features, targets):
        window = begin_replay_write(candidate, features, targets, [], 1, "batch")
        assert window is not None
        window.before_copy(0, 1)
        snapshot = object()
        window.copied(snapshot)
        window.finish([snapshot])
        return original(features, targets)

    candidate.train_batch = train
    if fault:
        with pytest.raises(RuntimeError, match="replay observer fault"):
            owner.train_ready(native_observer=access)
        assert runtime._stopped and budget.updates_completed == 0
        assert [stage for stage, _ in stages] == ["started", "uncertain"]
    else:
        poll = owner.train_ready(native_observer=access)
        assert stages[-1][1].receipt is poll.updates[0] and budget.updates_completed == 1
        assert [stage for stage, _ in stages] == ["started", "completed"]
    assert gate.snapshot().admitted_updates == 1 and access._scope is None
    for read in handles:
        with pytest.raises(ValueError, match="synchronous"):
            read()
    assert begin_replay_write(object(), object(), object(), [], 1, "batch") is None


def test_should_preserve_original_token_and_refs_when_close_is_refused():
    model, features, targets = object(), object(), object()
    scope = observe_replay_writes(model, features, targets, lambda *_: None, LIMITS)
    window = scope.__enter__()
    inherited = copy_context()
    with pytest.raises(ValueError):
        inherited.run(scope.__exit__, None, None, None)
    assert window._source is not None
    begin_replay_write(model, features, targets, [], 1, "batch")
    scope.__exit__(None, None, None)
    assert window._source is None
    assert begin_replay_write(model, features, targets, [], 1, "batch") is None


@pytest.mark.parametrize("start,count", [(True, 1), (-1, 1), (0, 0), (1, 1), (0, 2)])
def test_should_refuse_corrupt_copy_range_before_observation(start, count):
    model, features, targets = object(), object(), object()
    stages = []
    with observe_replay_writes(
        model, features, targets, lambda stage, read: stages.append(stage), LIMITS
    ) as window:
        begin_replay_write(model, features, targets, [], 1, "rows")
        with pytest.raises(ValueError, match="range"):
            window.before_copy(start, count)
    assert stages == ["begin"]
