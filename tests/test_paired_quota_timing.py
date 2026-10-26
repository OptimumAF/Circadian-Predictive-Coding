"""Deterministic wall telemetry with literal paired orchestration, no native work.

The imported fixture stubs all learners, collections and evaluations. Direct
timer cases call only literal Python callbacks. No arrays, models, RNG, forwards,
SGD or physical steps execute; budget monotonic time is never replaced.

Per invocation: 24 cases; five fake pilots, seven stub arm entries, eight literal
final packets and two stub evaluations. Direct timer cases make 20 timed entries
and 35 clock reads. Actual numerical/native work is zero.
"""

import json
from typing import Any

import pytest

from test_paired_quota_control import orchestration as orchestration
from src.adapters import key_door_quota_comparison as paired


def install_clock(monkeypatch, readings):
    values = iter(readings)

    def read():
        value = next(values)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(paired, "perf_counter", read)


def assert_record(record, *, attempts, returns, raised, seconds, clock_error=False):
    assert record["attempts"] == attempts
    assert record["returns"] == returns
    assert record["raised"] == raised
    assert record["inclusive_seconds"] == seconds
    assert (record["clock_error"] is not None) is clock_error
    json.dumps(record, allow_nan=False)


def test_should_create_independent_unstarted_records():
    first, second = paired._timing_record(), paired._timing_record()
    assert (
        first
        == second
        == dict(attempts=0, returns=0, raised=0, inclusive_seconds=0.0, clock_error=None)
    )
    first["attempts"] = 1
    assert second["attempts"] == 0


def test_should_accumulate_known_spans_and_preserve_return_identity(monkeypatch):
    install_clock(monkeypatch, [10.0, 12.5, 20.0, 20.75])
    record = paired._timing_record()
    result = object()
    assert paired._time_call(record, lambda: result) is result
    assert paired._time_call(record, lambda: None) is None
    assert_record(record, attempts=2, returns=2, raised=0, seconds=3.25)


def test_should_keep_nested_spans_inclusive_without_claiming_additive_time(monkeypatch):
    install_clock(monkeypatch, [0.0, 1.0, 3.0, 5.0])
    outer, inner = paired._timing_record(), paired._timing_record()
    paired._time_call(outer, lambda: paired._time_call(inner, lambda: "inner"))
    assert_record(outer, attempts=1, returns=1, raised=0, seconds=5.0)
    assert_record(inner, attempts=1, returns=1, raised=0, seconds=2.0)
    assert outer["inclusive_seconds"] + inner["inclusive_seconds"] > 5.0


@pytest.mark.parametrize("error", [RuntimeError("paid callback"), SystemExit("stop")])
def test_should_rethrow_original_body_failure_and_keep_its_elapsed_span(monkeypatch, error):
    install_clock(monkeypatch, [2.0, 6.0])
    record = paired._timing_record()

    def fail():
        raise error

    with pytest.raises(type(error)) as caught:
        paired._time_call(record, fail)
    assert caught.value is error
    assert_record(record, attempts=1, returns=0, raised=1, seconds=4.0)


@pytest.mark.parametrize("reading", [True, "1.0", float("nan"), float("inf")])
def test_should_refuse_invalid_start_without_admitting_callback(monkeypatch, reading):
    install_clock(monkeypatch, [reading])
    record = paired._timing_record()
    with pytest.raises(ValueError, match="finite"):
        paired._time_call(record, lambda: pytest.fail("invalid clock admitted work"))
    assert_record(record, attempts=0, returns=0, raised=0, seconds=0.0, clock_error=True)


@pytest.mark.parametrize("reading", [True, "2.0", float("nan"), float("inf"), 0.5])
def test_should_preserve_returned_work_but_refuse_invalid_end_interval(monkeypatch, reading):
    install_clock(monkeypatch, [1.0, reading])
    record = paired._timing_record()
    calls = []
    with pytest.raises(ValueError, match="finite"):
        paired._time_call(record, lambda: calls.append("paid"))
    assert calls == ["paid"]
    assert_record(record, attempts=1, returns=1, raised=0, seconds=0.0, clock_error=True)


@pytest.mark.parametrize("where", ["start", "end"])
def test_should_keep_clock_accessor_failures_and_observed_call_boundary(monkeypatch, where):
    error = OSError("timer unavailable")
    install_clock(monkeypatch, [error] if where == "start" else [0.0, error])
    record = paired._timing_record()
    calls = []
    with pytest.raises(OSError) as caught:
        paired._time_call(record, lambda: calls.append("paid"))
    assert caught.value is error
    expected = int(where == "end")
    assert len(calls) == expected
    assert_record(
        record, attempts=expected, returns=expected, raised=0, seconds=0.0, clock_error=True
    )


@pytest.mark.parametrize("ending", [float("nan"), OSError("timer also failed")])
def test_should_preserve_primary_body_exception_when_end_clock_also_fails(monkeypatch, ending):
    install_clock(monkeypatch, [0.0, ending])
    record = paired._timing_record()
    primary = RuntimeError("original paid failure")

    def fail():
        raise primary

    with pytest.raises(RuntimeError) as caught:
        paired._time_call(record, fail)
    assert caught.value is primary
    assert_record(record, attempts=1, returns=0, raised=1, seconds=0.0, clock_error=True)


def test_should_refuse_finite_clock_subtraction_overflow_without_nonfinite_report(monkeypatch):
    install_clock(monkeypatch, [-1e308, 1e308])
    record = paired._timing_record()
    with pytest.raises(ValueError, match="overflow"):
        paired._time_call(record, lambda: None)
    assert_record(record, attempts=1, returns=1, raised=0, seconds=0.0, clock_error=True)


@pytest.fixture
def timed_orchestration(monkeypatch, orchestration):
    clock: dict[str, Any] = {"now": 100.0}
    monkeypatch.setattr(paired, "perf_counter", lambda: clock["now"])

    def train(arm):
        clock["now"] += {"frozen": 3.0, "trained": 5.0}[arm]

    def evaluate(arm):
        clock["now"] += {"frozen": 7.0, "trained": 11.0}[arm]

    orchestration["on_arm"] = train
    orchestration["on_finish"] = evaluate
    return orchestration, clock


def test_should_time_both_phases_and_all_guard_categories_outside_sealed_rows(timed_orchestration):
    evidence, _ = timed_orchestration
    result = paired.run_pilot()
    assert result["status"] == "complete"
    timing = result["timing"]
    assert timing["definition"] == "paired_quota_inclusive_wall_v1"
    assert timing["clock"] == "perf_counter"
    assert "overlap" in timing["scope"] and "must not be summed" in timing["scope"]
    for arm, durations in (("frozen", (3.0, 7.0)), ("trained", (5.0, 11.0))):
        for phase, seconds in zip(("train", "evaluation"), durations, strict=True):
            assert_record(
                timing["arms"][arm][phase], attempts=1, returns=1, raised=0, seconds=seconds
            )
    for guard in timing["guards"].values():
        assert guard["attempts"] > 0
        assert guard["returns"] == guard["attempts"] and guard["raised"] == 0
        assert guard["inclusive_seconds"] == 0.0 and guard["clock_error"] is None
    assert [row["wake_updates"] for row in result["runs"]] == [0, 16]
    assert len(evidence["final_packets"]) == 8
    assert all("timing" not in row for row in result["runs"])
    assert result["scientific_advantage_verified"] is False
    json.dumps(result, allow_nan=False)


def test_should_distinguish_returned_failed_train_from_accepted_phase(timed_orchestration):
    evidence, _ = timed_orchestration
    evidence["failure"] = "frozen"
    result = paired.run_pilot()
    assert result["status"] == "incomplete" and evidence["final_packets"] == []
    assert result["runs"][0]["status"] == "failed"
    timing = result["timing"]
    assert_record(timing["arms"]["frozen"]["train"], attempts=1, returns=1, raised=0, seconds=3.0)
    for record in (
        timing["arms"]["trained"]["train"],
        timing["arms"]["frozen"]["evaluation"],
        timing["arms"]["trained"]["evaluation"],
    ):
        assert record == paired._timing_record()
    assert all(record == paired._timing_record() for record in timing["guards"].values())
    assert result["runs"][0]["training"][0]["collection"]["environment_steps"] == 12


def test_should_retain_raised_train_time_and_unreturned_paid_native_work(timed_orchestration):
    evidence, _ = timed_orchestration
    evidence["failure"] = "unreturned_trained"
    result = paired.run_pilot()
    assert result["status"] == "incomplete" and evidence["final_packets"] == []
    assert result["runs"][1]["wake_updates"] is None
    assert result["runs"][1]["training"][0]["collection"]["environment_steps"] == 6
    assert_record(
        result["timing"]["arms"]["trained"]["train"], attempts=1, returns=0, raised=1, seconds=5.0
    )
    assert result["timing"]["arms"]["trained"]["evaluation"] == paired._timing_record()


def test_should_time_failed_predecessor_guard_without_admitting_second_work(timed_orchestration):
    evidence, clock = timed_orchestration

    def mutate(arm):
        clock["now"] += 2.0
        if arm == "trained":
            evidence["rows"]["frozen"]["work"]["physical_steps"] = 0

    evidence["on_arm"] = mutate
    result = paired.run_pilot()
    assert result["status"] == "incomplete"
    assert evidence["physics"] == {"frozen": 96} and evidence["final_packets"] == []
    assert_record(
        result["timing"]["guards"]["preceding"], attempts=1, returns=0, raised=1, seconds=0.0
    )
    assert_record(
        result["timing"]["guards"]["pending"], attempts=1, returns=0, raised=1, seconds=0.0
    )
    assert_record(
        result["timing"]["arms"]["trained"]["train"], attempts=1, returns=0, raised=1, seconds=2.0
    )


def test_should_refuse_invalid_phase_start_clock_before_any_arm_work(
    monkeypatch, timed_orchestration
):
    evidence, _ = timed_orchestration
    monkeypatch.setattr(paired, "perf_counter", lambda: float("nan"))
    result = paired.run_pilot()
    assert result["status"] == "incomplete" and evidence["arms"] == []
    assert evidence["final_packets"] == []
    assert_record(
        result["timing"]["arms"]["frozen"]["train"],
        attempts=0,
        returns=0,
        raised=0,
        seconds=0.0,
        clock_error=True,
    )
    assert result["timing"]["arms"]["trained"]["train"] == paired._timing_record()
