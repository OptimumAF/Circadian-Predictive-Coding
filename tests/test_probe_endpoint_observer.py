"""Paid endpoint callbacks through scripted ports, without native computation."""

from copy import deepcopy
from dataclasses import replace
import pickle
from typing import Any

import numpy as np
import pytest

import src.app.fresh_task_probe as probe
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionProgress
from test_fresh_task_probe import session
from test_probe_return_phase import ReturnLearner, run
from test_probe_training_readouts import cohorts


def progress_budget():
    return ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=16, max_wall_seconds=1.0),
        lambda: 0.0,
        ToyExecutionProgress(),
    )


def test_should_observe_the_same_paid_f_and_return_endpoints_without_changing_native_reports():
    source = ReturnLearner()
    first, second = session(16), session(16)
    data = cohorts()
    before = pickle.dumps((source.value, data))
    readings = []

    def observe(learner, phase, check_boundary):
        assert learner is source.child
        assert first.updates_completed == 16
        assert second.updates_completed == (0 if phase == "after_probe" else 16)
        assert learner.value[0] == (16 if phase == "after_probe" else 32)
        check_boundary()
        probabilities = learner.predict(np.array([[0.25, 0.25]]))
        readings.append((learner, phase, float(learner.value[0]), probabilities.copy()))
        check_boundary()

    report = run(source, data=data, budget=first, return_budget=second, endpoint_observer=observe)
    assert report == run(ReturnLearner(), data=data)
    assert [phase for _, phase, _, _ in readings] == ["after_probe", "after_return"]
    assert readings[0][0] is readings[1][0]
    assert sum(event[0] == "train" for event in source.trace) == 32
    assert (
        sum(event[0] == "fork" for event in source.trace)
        == sum(event[0] == "restore" for event in source.trace)
        == 1
    )
    assert pickle.dumps((source.value, data)) == before


def test_should_preserve_omitted_callback_report_exactly_when_none_is_explicit():
    assert run(ReturnLearner()) == run(ReturnLearner(), endpoint_observer=None)


def test_should_not_silently_discard_a_report_returned_instead_of_the_none_contract():
    reading = {"status": "failed", "reason": "caller retains this reading"}
    report = run(ReturnLearner(), endpoint_observer=lambda *args: reading)
    assert report["status"] == "failed" and "return None" in report["reason"]
    assert report["completed_updates"] == 16 and report["return_phase"]["status"] == "not_run"
    assert reading == {"status": "failed", "reason": "caller retains this reading"}


@pytest.mark.parametrize("value", [False, 0, 1, "callback", [], object()])
def test_should_refuse_noncallable_observers_before_owned_copies_or_ports(monkeypatch, value):
    source = ReturnLearner()

    def forbidden_copy(batch):
        pytest.fail("invalid observer reached owned copies")

    monkeypatch.setattr(probe, "_copy_batch", forbidden_copy)
    with pytest.raises(ValueError, match="endpoint_observer"):
        run(source, endpoint_observer=value)
    assert source.trace == []


def test_should_require_the_complete_return_recipe_before_any_observer_or_learner_call():
    source = ReturnLearner()
    training, evaluation, monitors = cohorts()
    calls = []
    with pytest.raises(ValueError, match="full return recipe"):
        probe.run_fresh_task_probe(
            source,
            np.array([0.0]),
            training,
            evaluation,
            budget=session(16),
            checkpoint_updates=(0, 4, 8, 16),
            training_readouts=monitors,
            endpoint_observer=lambda *args: calls.append(args),
        )
    assert calls == source.trace == []


def test_should_admit_clock_progress_only_through_the_supplied_boundary():
    now = [0.0]
    clock = lambda: now[0]
    limits = ToyExecutionBudget(max_training_updates=16, max_wall_seconds=1.0)
    first, second = ToyBudgetSession(limits, clock), ToyBudgetSession(limits, clock)
    phases = []

    def observe(learner, phase, boundary):
        now[0] += 0.1
        boundary()
        assert first.last_clock == second.last_clock == now[0]
        phases.append(phase)

    report = run(ReturnLearner(), budget=first, return_budget=second, endpoint_observer=observe)
    assert report["status"] == report["return_phase"]["status"] == "complete"
    assert phases == ["after_probe", "after_return"]
    assert first.started_at == second.started_at == 0.0


@pytest.mark.parametrize("phase", ["after_probe", "after_return"])
def test_should_keep_failed_observer_readings_and_stop_later_phase_work(phase):
    source = ReturnLearner()
    first, second = session(16), session(16)
    readings = []

    def observe(learner, current, boundary):
        boundary()
        readings.append((current, learner.predict(np.array([[0.25, 0.25]])).copy()))
        if current == phase:
            raise RuntimeError("scripted endpoint reading failed")

    report = run(source, budget=first, return_budget=second, endpoint_observer=observe)
    assert report["status"] == "failed" and phase in report["reason"]
    assert report["completed_updates"] == first.updates_completed == 16
    expected = 0 if phase == "after_probe" else 16
    assert report["return_phase"]["completed_updates"] == second.updates_completed == expected
    assert report["return_phase"]["status"] == ("not_run" if phase == "after_probe" else "complete")
    assert len(readings) == (1 if phase == "after_probe" else 2)
    assert sum(event[0] == "train" for event in source.trace) == 16 + expected


@pytest.mark.parametrize("phase", ["after_probe", "after_return"])
@pytest.mark.parametrize("fault", ["score_value", "equal_phase", "equal_monitor"])
def test_should_restore_paid_values_and_refuse_equal_container_replacements(
    monkeypatch, phase, fault
):
    actual = probe.observe_endpoint
    owners: dict[str, dict[str, Any]] = {}
    trusted: dict[str, dict[str, Any]] = {}

    def capture(observer, learner, current, report, **options):
        owners[current] = report
        trusted[current] = deepcopy(report)
        return actual(observer, learner, current, report, **options)

    def observe(learner, current, boundary):
        if current != phase:
            return
        report = owners[current]
        paid = report if current == "after_probe" else report["return_phase"]
        if fault == "score_value":
            paid["checkpoints"][-1]["scores"].clear()
        elif fault == "equal_phase":
            if current == "after_probe":
                report["checkpoints"] = deepcopy(report["checkpoints"])
            else:
                report["return_phase"] = deepcopy(report["return_phase"])
        else:
            paid["training_readouts"] = deepcopy(paid["training_readouts"])

    monkeypatch.setattr(probe, "observe_endpoint", capture)
    source = ReturnLearner()
    report = run(source, endpoint_observer=observe)
    assert report["status"] == "failed" and phase in report["reason"]
    assert report["checkpoints"] == trusted[phase]["checkpoints"]
    assert report["training_readouts"] == trusted[phase]["training_readouts"]
    if phase == "after_return":
        assert report["return_phase"] == trusted[phase]["return_phase"]
    else:
        assert report["return_phase"]["status"] == "not_run"
    assert sum(event[0] == "train" for event in source.trace) == (
        16 if phase == "after_probe" else 32
    )


@pytest.mark.parametrize(
    "fault",
    [
        "probe_clock",
        "return_clock",
        "return_started_at",
        "equal_limits",
        "return_counter",
        "progress_value",
        "progress_owner",
        "unobserved_clock",
        "replay_bool",
    ],
)
def test_should_refuse_changed_original_controls_or_progress_before_return_updates(fault):
    first, second = progress_budget(), progress_budget()
    source = ReturnLearner()

    def observe(learner, phase, boundary):
        assert phase == "after_probe"
        if fault == "probe_clock":
            first.clock = lambda: 0.0
        elif fault == "return_clock":
            second.clock = lambda: 0.0
        elif fault == "return_started_at":
            second.started_at += 0.25
        elif fault == "equal_limits":
            first.budget = replace(first.budget)
        elif fault == "return_counter":
            second.updates_completed = 1
        elif fault == "progress_value":
            assert first.progress is not None
            first.progress.updates_completed = 15
        elif fault == "progress_owner":
            first.progress = deepcopy(first.progress)
        elif fault == "unobserved_clock":
            first.last_clock = 0.25
        else:
            first.replay_examples_completed = False

    report = run(source, budget=first, return_budget=second, endpoint_observer=observe)
    assert report["status"] == "failed" and "after_probe" in report["reason"]
    assert report["completed_updates"] == 16 and report["completed_training_examples"] == 144
    assert report["return_phase"]["status"] == "not_run"
    assert sum(event[0] == "train" for event in source.trace) == 16


def test_should_refuse_repaired_counter_values_after_a_boundary_has_already_failed():
    first, second = progress_budget(), progress_budget()
    refused = []

    def observe(learner, phase, boundary):
        assert phase == "after_probe"
        first.updates_completed = 15
        try:
            boundary()
        except ValueError as error:
            refused.append(str(error))
        finally:
            first.updates_completed = 16

    source = ReturnLearner()
    report = run(source, budget=first, return_budget=second, endpoint_observer=observe)
    assert refused and report["status"] == "failed"
    assert report["return_phase"]["status"] == "not_run" and second.updates_completed == 0
    assert sum(event[0] == "train" for event in source.trace) == 16


def test_should_recheck_the_parent_after_a_budget_clock_port_returns():
    active, corrupted = [False], [False]

    def clock():
        if active[0]:
            corrupted[0] = True
        return 0.0

    first = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=16, max_wall_seconds=1.0), clock
    )
    second = session(16)

    def parent():
        if corrupted[0]:
            raise ValueError("native owner changed during clock readback")

    def observe(learner, phase, boundary):
        assert phase == "after_probe"
        active[0] = True
        boundary()

    report = run(
        ReturnLearner(),
        budget=first,
        return_budget=second,
        endpoint_observer=observe,
        check_boundary=parent,
    )
    assert report["status"] == "failed" and "clock readback" in report["reason"]
    assert report["completed_updates"] == 16 and second.updates_completed == 0


def test_should_leave_native_state_checks_with_the_callers_endpoint_guard():
    dirty = [False]

    def parent():
        if dirty[0]:
            raise ValueError("caller refused changed disposable native state")

    def observe(learner, phase, boundary):
        learner.value[0] += 0.25
        dirty[0] = True
        boundary()

    source = ReturnLearner()
    report = run(source, endpoint_observer=observe, check_boundary=parent)
    assert report["status"] == "failed" and "native state" in report["reason"]
    assert report["completed_updates"] == 16 and report["return_phase"]["status"] == "not_run"
    assert source.child.value[0] == 16.25
    np.testing.assert_array_equal(source.value, [99.0])


def test_should_skip_observers_when_original_f_work_did_not_complete():
    def fail(learner, features, probabilities):
        raise RuntimeError("original prediction failed")

    calls = []
    report = run(
        ReturnLearner(hooks={"predict": fail}), endpoint_observer=lambda *args: calls.append(args)
    )
    assert report["status"] == "failed" and calls == []
    assert report["return_phase"]["status"] == "not_run"


def test_should_skip_the_return_observer_after_a_paid_partial_return_failure():
    dirty, calls = [False], []

    def train(learner, features, targets):
        if learner.value[0] == 17:
            dirty[0] = True

    def parent():
        if dirty[0]:
            raise ValueError("return commit guard failed")

    report = run(
        ReturnLearner(hooks={"train": train}),
        check_boundary=parent,
        endpoint_observer=lambda learner, phase, boundary: calls.append(phase),
    )
    assert calls == ["after_probe"]
    assert report["return_phase"]["status"] == "failed"
    assert report["return_phase"]["completed_updates"] == 1
