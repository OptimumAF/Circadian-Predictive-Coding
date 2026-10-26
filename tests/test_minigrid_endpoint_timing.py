"""Diagnostic span accounting and scripted readouts; no native learner or physics."""

from copy import deepcopy
from decimal import Decimal
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.adapters.minigrid_endpoint_timing import (
    PHASES,
    MiniGridEndpointTiming,
    NativeMiniGridEndpointTiming,
)
from src.adapters import minigrid_policy_evaluation as evaluation
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.minigrid_key_door import predefined_layouts
from test_minigrid_learning_pilot import Environment


class Clock:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        value = next(self.values)
        if isinstance(value, BaseException):
            raise value
        return value


def test_should_keep_an_empty_detached_report_without_reading_any_clock():
    clock = Clock([])
    recorder = MiniGridEndpointTiming(clock)
    before = recorder.report()
    assert before["status"] == "complete" and before["clock_error"] is None
    assert before["clock_reads"] == clock.calls == 0
    assert tuple(before["phases"]) == PHASES
    for row in before["phases"].values():
        assert row == dict(
            attempted=0, returned=0, failed=0, timed_calls=0, known_seconds=0.0, seconds=0.0
        )
    before["phases"]["physics"]["attempted"] = 99
    assert recorder.report()["phases"]["physics"]["attempted"] == 0
    assert clock.calls == 0


def test_should_distinguish_all_four_elapsed_spans_and_accumulate_known_calls():
    clock = Clock([10, 11, 13, 15, 17, 20, 25, 29, 30, 31])
    recorder = MiniGridEndpointTiming(clock)
    for phase in (*PHASES, "physics"):
        with recorder.measure(phase):
            pass
    result = recorder.report()
    assert result["status"] == "complete"
    assert result["clock_reads"] == clock.calls == 10
    for phase, seconds in zip(PHASES, (1.0, 2.0, 3.0, 5.0), strict=True):
        row = result["phases"][phase]
        calls = 2 if phase == "physics" else 1
        assert row == dict(
            attempted=calls,
            returned=calls,
            failed=0,
            timed_calls=calls,
            known_seconds=seconds,
            seconds=seconds,
        )


@pytest.mark.parametrize("error", [RuntimeError("paid failure"), KeyboardInterrupt("stop")])
@pytest.mark.parametrize("phase", PHASES)
def test_should_preserve_the_same_primary_fault_and_time_the_failed_call(error, phase):
    recorder = MiniGridEndpointTiming(Clock([2.0, 2.75]))
    with pytest.raises(type(error)) as raised:
        with recorder.measure(phase):
            raise error
    assert raised.value is error
    assert recorder.report()["phases"][phase] == dict(
        attempted=1, returned=0, failed=1, timed_calls=1, known_seconds=0.75, seconds=0.75
    )


@pytest.mark.parametrize(
    "invalid",
    [True, None, "1", Decimal("1"), float("nan"), float("inf"), -float("inf"), 10**400],
)
def test_should_mark_bad_clock_intervals_unknown_without_blocking_operations(invalid):
    clock = Clock([0.0, 0.5, invalid, 50.0])
    recorder = MiniGridEndpointTiming(clock)
    completed = []
    for index in range(3):
        with recorder.measure("input_preparation"):
            completed.append(index)
    result = recorder.report()
    assert completed == [0, 1, 2]
    assert result["status"] == "incomplete" and result["clock_error"]
    assert result["clock_reads"] == clock.calls == 3
    assert result["phases"]["input_preparation"] == dict(
        attempted=3, returned=3, failed=0, timed_calls=1, known_seconds=0.5, seconds=None
    )
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("end", [-1.0, RuntimeError("diagnostic read failed")])
def test_should_keep_a_primary_operation_error_when_its_end_clock_is_bad(end):
    recorder = MiniGridEndpointTiming(Clock([0.0, end]))
    original = ValueError("native operation failed first")
    with pytest.raises(ValueError) as raised:
        with recorder.measure("physics"):
            raise original
    assert raised.value is original
    result = recorder.report()
    assert result["status"] == "incomplete"
    assert result["clock_reads"] == 2
    assert result["phases"]["physics"] == dict(
        attempted=1, returned=0, failed=1, timed_calls=0, known_seconds=0.0, seconds=None
    )


@pytest.mark.parametrize("primary", [RuntimeError("paid fault"), SystemExit("paid cancellation")])
def test_should_preserve_an_active_paid_error_when_exit_clock_is_cancelled(primary):
    recorder = MiniGridEndpointTiming(Clock([0.0, KeyboardInterrupt("diagnostic cancellation")]))
    with pytest.raises(type(primary)) as raised:
        with recorder.measure("physics"):
            raise primary
    assert raised.value is primary
    result = recorder.report()
    assert result["status"] == "incomplete"
    assert "KeyboardInterrupt" in result["clock_error"]
    assert result["phases"]["physics"]["failed"] == 1
    assert result["phases"]["physics"]["seconds"] is None


@pytest.mark.parametrize("entry", [True, False])
def test_should_propagate_diagnostic_cancellation_without_a_prior_operation_error(entry):
    cancellation = KeyboardInterrupt("cancel timing")
    recorder = MiniGridEndpointTiming(Clock([cancellation] if entry else [0.0, cancellation]))
    operated = []
    with pytest.raises(KeyboardInterrupt) as raised:
        with recorder.measure("prediction"):
            operated.append(True)
    assert raised.value is cancellation
    assert operated == ([] if entry else [True])
    result = recorder.report()
    assert result["status"] == "incomplete"
    assert result["phases"]["prediction"]["seconds"] is None


def test_should_refuse_clock_regression_between_otherwise_positive_intervals():
    recorder = MiniGridEndpointTiming(Clock([5.0, 6.0, 4.0, 7.0]))
    with recorder.measure("prediction"):
        pass
    with recorder.measure("physics"):
        pass
    result = recorder.report()
    assert result["status"] == "incomplete" and result["clock_reads"] == 3
    assert result["phases"]["prediction"]["seconds"] == 1.0
    assert result["phases"]["physics"]["seconds"] is None


def test_should_keep_overflowed_elapsed_time_unknown_and_json_finite():
    recorder = MiniGridEndpointTiming(Clock([-1e308, 1e308]))
    with recorder.measure("preservation"):
        pass
    result = recorder.report()
    assert result["status"] == "incomplete" and "overflow" in result["clock_error"]
    assert result["phases"]["preservation"]["known_seconds"] == 0.0
    assert result["phases"]["preservation"]["seconds"] is None
    json.dumps(result, allow_nan=False)


def test_should_keep_a_known_lower_bound_when_overlapping_aggregate_overflows():
    recorder = MiniGridEndpointTiming(Clock([0.0, 0.0, 1e308, 1e308]))
    with recorder.measure("prediction"):
        with recorder.measure("prediction"):
            pass
    result = recorder.report()
    assert result["status"] == "incomplete"
    assert result["phases"]["prediction"] == dict(
        attempted=2, returned=2, failed=0, timed_calls=1, known_seconds=1e308, seconds=None
    )
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("phase", ["training", "", None, 1])
def test_should_refuse_unknown_spans_before_clock_or_operation(phase):
    clock = Clock([])
    recorder = MiniGridEndpointTiming(clock)
    with pytest.raises(ValueError, match="phase"):
        with recorder.measure(phase):
            pytest.fail("invalid span admitted work")
    assert clock.calls == recorder.report()["clock_reads"] == 0
    assert all(row["attempted"] == 0 for row in recorder.report()["phases"].values())


@pytest.mark.parametrize("clock", [None, 1, False, "clock"])
def test_should_require_a_callable_clock(clock):
    with pytest.raises(ValueError, match="callable"):
        MiniGridEndpointTiming(clock)


def test_should_keep_bounded_aggregates_and_no_success_record_chain():
    clock = Clock(range(4000))
    recorder = MiniGridEndpointTiming(clock)
    initial_keys = set(vars(recorder))
    for _ in range(2000):
        with recorder.measure("prediction"):
            pass
    result = recorder.report()
    assert set(vars(recorder)) == initial_keys
    assert len(result["phases"]) == 4
    assert result["phases"]["prediction"]["attempted"] == 2000
    assert result["phases"]["prediction"]["seconds"] == 2000.0
    assert len(json.dumps(result)) < 2400
    result["phases"]["prediction"]["seconds"] = -1
    assert recorder.report()["phases"]["prediction"]["seconds"] == 2000.0
    assert clock.calls == 4000


class ScriptedLearner:
    """One scripted probability row, with no optimizer or neural backend."""

    def __init__(self, work):
        self.work = work
        self.state = SimpleNamespace(policy="scripted", value=0)

    def snapshot_state(self):
        return deepcopy(self.state)

    def fork(self):
        child = ScriptedLearner(self.work)
        child.state = self.snapshot_state()
        return child

    def restore_state(self, state):
        self.state = deepcopy(state)

    def predict(self, features):
        self.work["predict_attempts"] += 1
        if self.work.get("prediction_error"):
            raise RuntimeError("scripted prediction failed")
        self.work["predict_returned"] += 1
        return np.tile(np.array([[1.0, 0, 0, 0, 0, 0, 0]]), (len(features), 1))


def scripted_readout(monkeypatch, *, profiled, fault=None, broken_clock=False):
    work = dict(predict_attempts=0, predict_returned=0)
    if fault is not None:
        work[fault] = True
    monkeypatch.setattr(Environment, "work", work)
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", Environment)
    source = ScriptedLearner(work)
    state = source.snapshot_state()
    budget_reads = []

    def budget_clock():
        budget_reads.append(1)
        return 0.0

    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), budget_clock)
    binding = (budget.budget, budget.clock, budget.started_at, budget.progress)
    guard_calls = []

    def guard():
        guard_calls.append(1)
        assert (budget.budget, budget.clock, budget.started_at, budget.progress) == binding
        assert budget.updates_completed == budget.replay_examples_completed == 0

    clock_values = [RuntimeError("diagnostic failed")] if broken_clock else range(10000)
    recorder = MiniGridEndpointTiming(Clock(clock_values)) if profiled else None
    options = dict(timings=recorder) if profiled else {}
    try:
        report = evaluation._run_checkpoint(
            source,
            state,
            method="observation_memory",
            updates=0,
            layouts=(predefined_layouts()[0],),
            role="train",
            sources=(),
            budget=budget,
            check_boundary=guard,
            public_history=True,
            memory_copier=deepcopy,
            **options,
        )
    except evaluation.MiniGridEvaluationFailed as error:
        report = error.report
    assert source.snapshot_state() == state
    assert all(game.closed for game in work.get("environments", []))
    trace = (
        len(guard_calls),
        len(budget_reads),
        work["predict_attempts"],
        work["predict_returned"],
    )
    return report, recorder, trace


@pytest.mark.parametrize("fault", [None, "prediction_error", "step_error"])
def test_should_profile_scripted_execution_without_changing_paid_work_guards_or_errors(
    monkeypatch, fault
):
    plain, _, plain_trace = scripted_readout(monkeypatch, profiled=False, fault=fault)
    profiled, recorder, profiled_trace = scripted_readout(monkeypatch, profiled=True, fault=fault)
    assert recorder is not None
    assert profiled_trace == plain_trace
    assert "phase_timings" not in plain
    saved_timing = profiled.pop("phase_timings")
    assert profiled == plain
    assert saved_timing == recorder.report()
    rows = saved_timing["phases"]
    assert saved_timing["status"] == "complete"
    assert rows["preservation"]["attempted"] > 0
    assert rows["input_preparation"]["attempted"] > 0
    assert rows["prediction"]["attempted"] == (1 if fault else 4)
    assert rows["prediction"]["failed"] == int(fault == "prediction_error")
    assert rows["physics"]["attempted"] == (0 if fault == "prediction_error" else 1 if fault else 4)
    assert rows["physics"]["failed"] == int(fault == "step_error")
    for row in rows.values():
        assert row["attempted"] == row["returned"] + row["failed"]
        assert row["timed_calls"] == row["attempted"]
        assert row["seconds"] == row["known_seconds"]
    assert plain["status"] == ("complete" if fault is None else "failed")


@pytest.mark.parametrize("fault", [None, "prediction_error", "step_error"])
def test_should_keep_scripted_execution_and_original_error_when_diagnostic_clock_fails(
    monkeypatch, fault
):
    plain, _, plain_trace = scripted_readout(monkeypatch, profiled=False, fault=fault)
    profiled, recorder, actual_trace = scripted_readout(
        monkeypatch, profiled=True, fault=fault, broken_clock=True
    )
    assert recorder is not None
    timings = profiled.pop("phase_timings")
    assert profiled == plain and actual_trace == plain_trace
    assert timings == recorder.report()
    assert timings["status"] == "incomplete" and timings["clock_reads"] == 1
    for row in timings["phases"].values():
        assert row["known_seconds"] == 0.0
        expected_seconds = None if row["attempted"] else 0.0
        assert row["seconds"] == expected_seconds


class TimingSubclass(MiniGridEndpointTiming):
    pass


@pytest.mark.parametrize("recorder", [object(), TimingSubclass(Clock([]))])
@pytest.mark.parametrize("public", [True, False])
def test_should_refuse_nonexact_recorders_before_snapshot_budget_or_callbacks(recorder, public):
    def forbidden():
        pytest.fail("invalid recorder admitted callback")

    common = dict(budget=object(), check_boundary=forbidden, timings=recorder)
    with pytest.raises(ValueError, match="exact .*MiniGridEndpointTiming"):
        if public:
            evaluation.evaluate_wide_neural_probe_checkpoint(
                object(), object(), source_updates=0, phase="after_probe", method="bp", **common
            )
        else:
            evaluation._run_checkpoint(
                object(),
                object(),
                method="observation_memory",
                updates=0,
                layouts=(),
                role="train",
                sources=(),
                public_history=True,
                **common,
            )


def test_should_ignore_a_replaced_callback_field_in_the_production_clock():
    recorder = NativeMiniGridEndpointTiming()
    called = []

    def untrusted():
        called.append(True)
        raise AssertionError("diagnostic callback gained production authority")

    recorder._clock = untrusted
    with recorder.measure("preservation"):
        pass
    with recorder.measure("prediction"):
        pass
    assert called == []
    result = recorder.report()
    assert result["status"] == "complete"
    assert result["clock_reads"] == 4
    assert result["phases"]["prediction"]["seconds"] >= 0


def test_should_refuse_injected_clock_at_public_h32_entry_before_any_work():
    called = []
    recorder = MiniGridEndpointTiming(lambda: called.append(True))
    with pytest.raises(ValueError, match="exact NativeMiniGridEndpointTiming"):
        evaluation.evaluate_wide_neural_probe_checkpoint(
            object(),
            object(),
            source_updates=0,
            phase="after_probe",
            method="bp",
            budget=object(),
            check_boundary=lambda: None,
            timings=recorder,
        )
    assert called == []


class NativeTimingSubclass(NativeMiniGridEndpointTiming):
    def _sample_clock(self):
        pytest.fail("subclass callback admitted production authority")


def test_should_refuse_native_clock_subclasses_before_any_work():
    with pytest.raises(ValueError, match="exact NativeMiniGridEndpointTiming"):
        evaluation.evaluate_wide_neural_probe_checkpoint(
            object(),
            object(),
            source_updates=0,
            phase="after_probe",
            method="bp",
            budget=object(),
            check_boundary=lambda: None,
            timings=NativeTimingSubclass(),
        )


def test_should_forward_only_native_clock_recorder_through_public_h32_route(monkeypatch):
    recorder = NativeMiniGridEndpointTiming()
    calls = []
    state = SimpleNamespace(
        policy=SimpleNamespace(method="bp"), completed_updates=16, training_examples=144
    )
    monkeypatch.setattr(
        evaluation, "_copy_wide_neural_probe_snapshot", lambda value, **kw: deepcopy(state)
    )
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=0, max_replay_examples=0), lambda: 0.0
    )

    def run(source, checkpoint, **options):
        assert options["timings"] is recorder

        def untrusted():
            calls.append(True)
            raise AssertionError("replaced diagnostic callback ran")

        recorder._clock = untrusted
        with recorder.measure("prediction"):
            pass
        return dict(status="complete", phase_timings=recorder.report())

    monkeypatch.setattr(evaluation, "_run_checkpoint", run)
    result = evaluation.evaluate_wide_neural_probe_checkpoint(
        object(),
        state,
        source_updates=0,
        phase="after_probe",
        method="bp",
        budget=budget,
        check_boundary=lambda: None,
        timings=recorder,
    )
    assert result["status"] == result["phase_timings"]["status"] == "complete"
    assert calls == []
    assert budget.updates_completed == budget.replay_examples_completed == 0
