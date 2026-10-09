"""Fresh bounded fake inputs; no native models, persistence or replay training."""

from contextlib import nullcontext
from dataclasses import replace
import gc
from threading import Thread
from typing import Any
import weakref

import pytest

from src.app.native_update_origin import NativeUpdateAccess
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionStopped
from src.core.native_update_origin import NativeUpdateOrigin
from src.app.payload_ownership import PayloadOwnershipBusy
from test_experience_inbox import setup, experience, label
from test_managed_experience import setup as managed_setup, declaration, record


def ready(budget=None):
    inbox, clock, learner, original_budget = setup(budget=budget)
    inbox.record_experience(experience())
    inbox.record_label(label())
    clock.advance_to(3)
    return inbox, clock, learner, original_budget


def test_should_observe_original_records_and_actual_detached_inputs_without_extra_copies():
    inbox, _, learner, budget = ready()
    source, target = inbox._experiences[("e1", "s1")], inbox._labels[("e1", "s1")]
    observations: list[Any] = []
    handles = []

    def observe(stage, read):
        value = read()
        handles.append(read)
        observations.append((stage, value))
        assert value.learner is learner and value.source is source and value.label is target
        assert value.features is not source.features and value.targets is not target.targets

    (receipt,) = inbox.drain(native_observer=observe)
    started, completed = observations
    assert [row[0] for row in observations] == ["started", "completed"]
    assert started[1].features is completed[1].features
    assert started[1].targets is completed[1].targets
    assert completed[1].features == ["s1", "learner mutation"]
    assert source.features == ["s1"] and target.targets == ["target-s1"]
    assert started[1].receipt is None and started[1].completed_updates == 0
    assert completed[1].receipt is receipt is inbox.applied_updates[0]
    assert completed[1].completed_updates == budget.updates_completed == 1
    assert completed[1].learner_version == "candidate-0"
    for read in handles:
        with pytest.raises(ValueError, match="synchronous"):
            read()
    assert inbox.drain(native_observer=observe) == () and len(observations) == 2


def test_should_refuse_foreign_thread_and_access_between_notifications():
    inbox, _, learner, _ = ready()
    handles, errors = [], []

    def observe(stage, read):
        handles.append(read)

        def foreign():
            try:
                read()
            except ValueError as error:
                errors.append(error)

        worker = Thread(target=foreign)
        worker.start()
        worker.join(1)
        assert not worker.is_alive()

    original = learner.train_batch

    def train(features, targets):
        with pytest.raises(ValueError, match="synchronous"):
            handles[0]()
        return original(features, targets)

    learner.train_batch = train
    inbox.drain(native_observer=observe)
    assert len(errors) == 2


@pytest.mark.parametrize("observed", [False, True])
def test_should_preserve_default_fields_and_outcomes(observed):
    inbox, _, learner, budget = ready()
    fields = tuple(vars(inbox))
    result = inbox.drain(native_observer=(lambda stage, read: read()) if observed else None)
    assert tuple(vars(inbox)) == fields
    assert len(result) == budget.updates_completed == len(learner.calls) == 1
    assert not inbox.stopped


def test_should_refuse_invalid_observer_before_clock_and_copy(monkeypatch):
    inbox, _, learner, budget = ready()
    monkeypatch.setattr(type(inbox), "_read_clock", lambda self: pytest.fail("clock accessed"))
    with pytest.raises(ValueError, match="native_observer"):
        inbox.drain(native_observer=object())
    assert learner.calls == [] and budget.updates_completed == 0


def test_should_observe_original_pre_update_refusal_without_spending_work():
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 0.0)
    inbox, _, learner, _ = ready(budget)
    observations = []
    with pytest.raises(ToyExecutionStopped):
        inbox.drain(native_observer=lambda stage, read: observations.append((stage, read())))
    assert [stage for stage, _ in observations] == ["refused"]
    assert observations[0][1].receipt is None and observations[0][1].completed_updates == 0
    assert not inbox.stopped and not inbox.applied_updates and not learner.calls


def test_should_preserve_uncertain_native_failure_and_original_receipt_absence():
    inbox, _, learner, budget = ready()
    failure = RuntimeError("partial fake mutation")

    def train(features, targets):
        learner.calls.append((features, targets))
        features.append("partial")
        raise failure

    learner.train_batch = train
    observations = []
    with pytest.raises(RuntimeError) as caught:
        inbox.drain(native_observer=lambda stage, read: observations.append((stage, read())))
    assert caught.value is failure and [row[0] for row in observations] == ["started", "uncertain"]
    assert observations[-1][1].features[-1] == "partial"
    assert inbox.stopped and budget.updates_completed == 0 and not inbox.applied_updates
    with pytest.raises(ValueError):
        inbox.drain()
    assert len(learner.calls) == 1


def test_should_observe_committed_failure_without_retry_or_budget_reset():
    seconds = [0.0]
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=2, max_wall_seconds=1), lambda: seconds[0]
    )
    inbox, _, learner, _ = ready(budget)
    original = learner.train_batch

    def train(features, targets):
        value = original(features, targets)
        seconds[0] = 2.0
        return value

    learner.train_batch = train
    observations = []
    with pytest.raises(ToyExecutionStopped):
        inbox.drain(native_observer=lambda stage, read: observations.append((stage, read())))
    assert [row[0] for row in observations] == ["started", "completed", "committed_failure"]
    receipt = inbox.applied_updates[0]
    assert observations[-1][1].receipt is receipt and budget.updates_completed == 1
    assert budget.started_at == 0.0 and budget.last_clock == 2.0 and not inbox.stopped
    assert inbox.drain() == () and len(learner.calls) == 1


@pytest.mark.parametrize("fault_stage", ["started", "completed"])
def test_should_close_observer_fault_with_preserved_work_and_identity(fault_stage):
    inbox, _, learner, budget = ready()
    failure = RuntimeError("observer fault")
    observations, handles = [], []

    def observe(stage, read):
        handles.append(read)
        observations.append((stage, read()))
        if stage == fault_stage:
            raise failure

    with pytest.raises(RuntimeError) as caught:
        inbox.drain(native_observer=observe)
    assert caught.value is failure and inbox.stopped
    committed = fault_stage == "completed"
    assert observations[-1][0] == ("committed_failure" if committed else "uncertain")
    assert budget.updates_completed == len(learner.calls) == int(committed)
    assert len(inbox.applied_updates) == int(committed)
    if committed:
        assert observations[-1][1].receipt is inbox.applied_updates[0]
    for read in handles:
        with pytest.raises(ValueError):
            read()


def test_should_preserve_primary_and_failure_observer_errors_together():
    inbox, _, learner, budget = ready()
    primary, secondary = RuntimeError("native fault"), LookupError("observer fault")

    def train(features, targets):
        learner.calls.append((features, targets))
        raise primary

    def observe(stage, read):
        read()
        if stage == "uncertain":
            raise secondary

    learner.train_batch = train
    with pytest.raises(ExceptionGroup) as caught:
        inbox.drain(native_observer=observe)
    assert caught.value.exceptions == (primary, secondary)
    assert inbox.stopped and budget.updates_completed == 0 and not inbox.applied_updates


@pytest.mark.parametrize("defer", ["pause", "serving", "consent"])
def test_should_keep_original_managed_admission_before_observer(defer):
    owner, _, runtime, clock, gate, _, budget = managed_setup()
    owner.declare(declaration())
    record(owner)
    clock.advance_to(3)
    if defer == "pause":
        gate.pause()
    elif defer == "consent":
        owner._opted_out.add("person")
    context = gate.serving() if defer == "serving" else nullcontext()
    with context:
        result = owner.train_ready(native_observer=lambda *_: pytest.fail("observer entered"))
    assert not result.updates and not budget.updates_completed and not runtime.applied_updates
    assert gate.snapshot().admitted_updates == 0


def test_should_propagate_original_managed_inputs_and_refuse_reentrant_owner():
    owner, _, runtime, clock, gate, _, budget = managed_setup()
    owner.declare(declaration())
    record(owner)
    clock.advance_to(3)
    fields = tuple(vars(owner)), tuple(vars(runtime)), tuple(vars(runtime._inbox))
    observations = []

    def observe(stage, read):
        observations.append((stage, read()))
        with pytest.raises(PayloadOwnershipBusy):
            owner.train_ready()
        with pytest.raises(ValueError, match="candidate is busy"):
            runtime.train_ready()
        assert read().learner is runtime._candidate

    result = owner.train_ready(native_observer=observe)
    assert observations[-1][1].receipt is result.updates[0]
    assert budget.updates_completed == gate.snapshot().admitted_updates == 1
    assert fields == (tuple(vars(owner)), tuple(vars(runtime)), tuple(vars(runtime._inbox)))


def test_should_release_access_owned_references_when_closed():
    class Payload:
        pass

    payload = Payload()
    reference = weakref.ref(payload)
    source = replace(experience(), features=payload)
    target = label()
    observation = NativeUpdateOrigin(
        object(), source, target, payload, target.targets, "v", None, 0
    )

    def observe(stage, read) -> None:
        read()

    def original(value: NativeUpdateOrigin[Any, Any] = observation):
        return value

    access = NativeUpdateAccess(observe, original)
    read = access.read
    access.notify("started")
    access.close()
    del observation, source, payload, original
    gc.collect()
    assert reference() is None
    with pytest.raises(ValueError):
        read()
    with pytest.raises(ValueError):
        access.notify("completed")
