"""Compose bounded external reads without losing paid proofs or refusals.

All learners, evaluator receipts and clocks are scripted engineering ports. No
native learner updates, predictions or environment steps run in this file.
"""

from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest

import src.app.fresh_task_probe as probe
from src.adapters import minigrid_probe_rollout as live
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from test_fresh_task_probe import session
from test_minigrid_probe_rollout import Learner, ports as ports
from test_probe_return_phase import ReturnLearner, run


def test_should_check_parent_between_original_clocks_and_supplied_reads():
    events = []
    active = [False]

    def parent():
        if active[0]:
            events.append("parent")

    def observe(learner, phase, boundary):
        if phase == "after_probe":
            active[0] = True
            try:
                boundary(lambda: events.append("read ports"))
            finally:
                active[0] = False

    report = run(ReturnLearner(), check_boundary=parent, endpoint_observer=observe)
    assert report["status"] == report["return_phase"]["status"] == "complete"
    assert events == ["parent", "parent", "read ports", "parent", "parent"]


def test_should_keep_no_argument_and_explicit_none_boundary_calls_compatible():
    calls = []

    def observe(learner, phase, boundary):
        boundary()
        boundary(None)
        calls.append(phase)

    assert run(ReturnLearner(), endpoint_observer=observe) == run(ReturnLearner())
    assert calls == ["after_probe", "after_return"]


def test_should_always_check_the_parent_after_a_supplied_operation_raises():
    events = []
    active = [False]

    def parent():
        if active[0]:
            events.append("parent")

    def read_ports():
        events.append("read ports")
        raise RuntimeError("scripted read failed")

    def observe(learner, phase, boundary):
        active[0] = True
        try:
            boundary(read_ports)
        finally:
            active[0] = False

    report = run(ReturnLearner(), check_boundary=parent, endpoint_observer=observe)
    assert report["status"] == "failed" and "scripted read failed" in report["reason"]
    assert events == ["parent", "parent", "read ports", "parent"]
    assert report["completed_updates"] == 16 and report["return_phase"]["status"] == "not_run"


@pytest.mark.parametrize("phase", ["after_probe", "after_return"])
@pytest.mark.parametrize("fault", ["value", "equal_container"])
def test_should_restore_paid_evidence_changed_by_a_composed_read(monkeypatch, phase, fault):
    delegate = probe.observe_endpoint
    owners: dict[str, dict[str, Any]] = {}
    originals: dict[str, dict[str, Any]] = {}

    def capture(observer, learner, current, report, **options):
        owners[current], originals[current] = report, deepcopy(report)
        return delegate(observer, learner, current, report, **options)

    def observe(learner, current, boundary):
        if current == phase:

            def mutate():
                paid = (
                    owners[current] if current == "after_probe" else owners[current]["return_phase"]
                )
                if fault == "value":
                    paid["checkpoints"][-1]["scores"].clear()
                else:
                    paid["checkpoints"] = deepcopy(paid["checkpoints"])

            boundary(mutate)

    monkeypatch.setattr(probe, "observe_endpoint", capture)
    report = run(ReturnLearner(), endpoint_observer=observe)
    assert report["status"] == "failed" and phase in report["reason"]
    assert report["checkpoints"] == originals[phase]["checkpoints"]
    assert report["training_readouts"] == originals[phase]["training_readouts"]
    if phase == "after_return":
        assert report["return_phase"] == originals[phase]["return_phase"]
    else:
        assert report["return_phase"]["status"] == "not_run"


@pytest.mark.parametrize("fault", ["updates", "clock", "equal_limits", "return_origin"])
def test_should_refuse_original_budget_changes_inside_supplied_reads(fault):
    original, returned = session(16), session(16)

    def observe(learner, phase, boundary):
        assert phase == "after_probe"

        def mutate():
            if fault == "updates":
                original.updates_completed = 15
            elif fault == "clock":
                original.clock = lambda: 0.0
            elif fault == "equal_limits":
                returned.budget = replace(returned.budget)
            else:
                returned.started_at += 0.25

        boundary(mutate)

    source = ReturnLearner()
    report = run(source, budget=original, return_budget=returned, endpoint_observer=observe)
    assert report["status"] == "failed" and report["completed_updates"] == 16
    assert report["completed_training_examples"] == 144
    assert report["return_phase"]["status"] == "not_run"
    assert sum(event[0] == "train" for event in source.trace) == 16


@pytest.mark.parametrize("mutation", [False, True])
def test_should_keep_a_caught_operation_failure_sticky_even_after_local_repair(mutation):
    original, returned = session(16), session(16)
    reads = []
    caught = []

    def observe(learner, phase, boundary):
        def fail():
            reads.append("failed read")
            if mutation:
                original.updates_completed = 15
            raise RuntimeError("first operation failed")

        try:
            boundary(fail)
        except Exception as error:
            caught.append(str(error))
        finally:
            original.updates_completed = 16
        try:
            boundary(lambda: reads.append("forbidden later read"))
        except Exception as error:
            caught.append(str(error))

    report = run(
        ReturnLearner(), budget=original, return_budget=returned, endpoint_observer=observe
    )
    assert reads == ["failed read"] and len(caught) == 2
    assert "first operation failed" in caught[0]
    assert report["status"] == "failed" and report["return_phase"]["status"] == "not_run"
    assert returned.updates_completed == 0


def test_should_not_call_supplied_ports_when_the_opening_parent_proof_refuses():
    active, reads = [False], []

    def parent():
        if active[0]:
            raise ValueError("parent proof refused before local reads")

    def observe(learner, phase, boundary):
        active[0] = True
        boundary(lambda: reads.append("not admitted"))

    report = run(ReturnLearner(), endpoint_observer=observe, check_boundary=parent)
    assert report["status"] == "failed" and reads == []
    assert report["return_phase"]["status"] == "not_run"


def test_should_qualify_elapsed_original_budget_time_before_the_composed_call_returns():
    now = [0.0]
    clock = lambda: now[0]
    limits = ToyExecutionBudget(max_training_updates=16, max_wall_seconds=1.0)
    original, returned = ToyBudgetSession(limits, clock), ToyBudgetSession(limits, clock)
    stopped = []

    def observe(learner, phase, boundary):
        def slow_read():
            now[0] = 2.0

        try:
            boundary(slow_read)
        except Exception as error:
            stopped.append(str(error))
        assert stopped, "closing deadline must refuse before control returns to the observer"

    report = run(
        ReturnLearner(), budget=original, return_budget=returned, endpoint_observer=observe
    )
    assert report["status"] == "failed" and "max_wall_seconds" in report["reason"]
    assert original.last_clock == 2.0 and returned.updates_completed == 0


def test_should_not_discard_an_operation_result_instead_of_its_none_contract():
    reading = dict(status="failed", reason="retained caller reading")

    def observe(learner, phase, boundary):
        boundary(lambda: reading)

    report = run(ReturnLearner(), endpoint_observer=observe)
    assert report["status"] == "failed" and "return None" in report["reason"]
    assert report["return_phase"]["status"] == "not_run"
    assert reading == dict(status="failed", reason="retained caller reading")


@pytest.mark.parametrize("invalid", [False, 0, "not callable", []])
def test_should_refuse_invalid_composed_read_ports_without_admitting_return_work(invalid):
    value: Any = invalid

    def observe(learner, phase, boundary):
        boundary(value)

    report = run(ReturnLearner(), endpoint_observer=observe)
    assert report["status"] == "failed" and report["completed_updates"] == 16
    assert report["return_phase"]["status"] == "not_run"


def test_should_check_parent_state_changes_even_when_the_read_port_itself_failed():
    changed, postchecks = [False], []

    def parent():
        if changed[0]:
            postchecks.append("observed changed owner")
            raise ValueError("parent state changed")

    def observe(learner, phase, boundary):
        def fail_and_mutate():
            changed[0] = True
            raise RuntimeError("port failed after mutation")

        boundary(fail_and_mutate)

    report = run(ReturnLearner(), endpoint_observer=observe, check_boundary=parent)
    assert report["status"] == "failed" and postchecks
    assert "port failed after mutation" in report["reason"]
    assert report["return_phase"]["status"] == "not_run"


def test_should_qualify_each_local_helper_read_once_and_preserve_legacy_receipts(
    monkeypatch, ports
):
    original_budget = live._live_budget
    inside = [False]
    local_reads, enclosures, legacy_calls = [], [], []

    def budget():
        session_budget, binding = original_budget()
        before = session_budget.before_final

        def read():
            assert inside[0], "local clock reads must be inside the qualified boundary"
            local_reads.append(session_budget)
            before()

        monkeypatch.setattr(session_budget, "before_final", read)
        return session_budget, binding

    def qualified(read_ports=None):
        assert callable(read_ports)
        enclosures.append(read_ports)
        assert not inside[0]
        inside[0] = True
        try:
            read_ports()
        finally:
            inside[0] = False

    monkeypatch.setattr(live, "_live_budget", budget)
    observer = live.ProbeRollouts(0)
    observer.observe_qualified(Learner(), "after_probe", qualified)
    observer.check()
    assert len(enclosures) == len(local_reads) > 0
    assert observer.records[0]["status"] == "complete"

    monkeypatch.setattr(live, "_live_budget", original_budget)
    legacy = live.ProbeRollouts(0)

    def old_zero_argument_callback():
        legacy_calls.append(True)

    legacy.observe(Learner(), "after_probe", old_zero_argument_callback)
    assert legacy_calls and legacy.records == observer.records
    assert len(ports["calls"]) == 2


@pytest.mark.parametrize("fault", ["opening_clock", "supplied_read"])
def test_should_refuse_parent_changes_before_admitting_the_next_external_port(fault):
    active, changed = [False], [False]
    calls = []

    def clock():
        if active[0]:
            calls.append("clock")
            if fault == "opening_clock":
                changed[0] = True
        return 0.0

    def parent():
        if changed[0]:
            raise ValueError("parent cohort changed inside a port")

    first = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=16, max_wall_seconds=1.0), clock
    )

    def observe(learner, phase, boundary):
        active[0] = True

        def supplied():
            calls.append("supplied")
            if fault == "supplied_read":
                changed[0] = True

        boundary(supplied)

    report = run(
        ReturnLearner(),
        budget=first,
        return_budget=session(16),
        endpoint_observer=observe,
        check_boundary=parent,
    )
    assert report["status"] == "failed" and "parent cohort" in report["reason"]
    assert calls == (["clock"] if fault == "opening_clock" else ["clock", "supplied"])
    assert report["return_phase"]["status"] == "not_run"
