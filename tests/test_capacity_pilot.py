"""Capacity through the existing delayed runner and one tiny native case."""

import json
import pickle
import sys
from typing import Any, cast

import numpy as np
import pytest

import scripts.run_stream_rehearsal_pilot as cli
import src.adapters.capacity_pilot as pilot
import src.app.toy_execution_budget as budget_module
from src.app.coupled_attribution import AttributionBatch, AttributionStream, run_attribution_stream
from src.app.stream_capacity import StreamCapacity
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.capacity_cost import CapacityUsage
from src.core.binary_probability_scores import probability_scores
from src.core.learner_ports import TrainingDiagnostic
from src.shared.process_memory import ProcessRssSegment


class FakeLearner:
    def __init__(self, trace=None, state=0):
        self.trace = [] if trace is None else trace
        self.state = state

    def fork(self):
        self.trace.append(("fork", self.state))
        return FakeLearner(self.trace, self.state)

    def snapshot_state(self):
        return np.array([float(self.state)])

    def restore_state(self, state):
        self.state = int(state[0])

    def train_batch(self, x, y):
        self.trace.append(("wake", self.state, x.copy(), y.copy()))
        self.state += 1
        return TrainingDiagnostic("fake_capacity_wake_v1", 0.5)

    def predict(self, x):
        self.trace.append(("score", self.state, x.copy()))
        return np.full((len(x), 1), 0.4 + self.state / 100)


def budget():
    return ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=4, max_wall_seconds=10), lambda: 0.0
    )


def fake_data():
    training, _ = pilot.build_stream()
    heldout = tuple(
        AttributionBatch(
            "final_test", c, (f"fake-final-{c}",), np.array([[99.0, 98.0]]), np.array([[float(k)]])
        )
        for k, c in enumerate(("A", "B"))
    )
    return training, heldout


def test_should_observe_every_returned_wake_without_training_or_scoring_from_observation():
    source = FakeLearner()
    wake = budget()
    measured = []

    def measure(state):
        measured.append(state.copy())
        return CapacityUsage(9, 72, 0)

    helper = StreamCapacity(
        source,
        wake,
        total_updates=4,
        measure_state=measure,
        max_parameters=17,
        max_state_array_bytes=200,
    )
    training, final = fake_data()
    result = run_attribution_stream(
        source,
        training,
        final,
        checkpoint_updates=(0, 2, 4),
        label_delay=2,
        budget=wake,
        capacity=helper,
    )
    assert [float(s[0]) for s in measured] == [0, 1, 2, 3, 4]
    assert helper.report["completed_native_calls"] == 4
    assert helper.report["completed_training_rows"] == 8
    assert helper.report["mean_parameters"] == helper.report["peak_parameters"] == 9
    assert result["capacity"] == helper.report and result["applied_consolidations"] == []
    assert [r[0] for r in source.trace if r[0] in ("wake", "score")] == ["wake"] * 4 + ["score"] * 6
    assert source.state == 0


def test_should_keep_default_schema_and_scores_when_capacity_is_explicitly_none():
    training, final = fake_data()
    source = FakeLearner()
    before = pickle.dumps(source.snapshot_state())
    default = run_attribution_stream(
        source, training, final, budget=budget(), checkpoint_updates=(0, 2, 4), label_delay=2
    )
    explicit = run_attribution_stream(
        source,
        training,
        final,
        budget=budget(),
        capacity=None,
        checkpoint_updates=(0, 2, 4),
        label_delay=2,
    )
    assert (
        default == explicit
        and "capacity" not in default
        and "applied_consolidations" not in default
    )
    assert pickle.dumps(source.snapshot_state()) == before


@pytest.mark.parametrize(
    "predictions",
    [
        np.array([[0.5]], dtype=np.float32),
        np.array([[float("nan")]]),
        np.array([[1.1]]),
        np.empty((0, 1)),
    ],
)
def test_should_preserve_shared_metric_refusals_after_removing_import_cycle(predictions):
    with pytest.raises(ValueError):
        probability_scores(predictions, np.array([[1.0]]))


def test_should_refuse_capacity_pause_without_renewing_or_losing_wake_work():
    source = FakeLearner()
    wake = budget()
    helper = StreamCapacity(
        source,
        wake,
        total_updates=4,
        measure_state=lambda s: CapacityUsage(9, 72, 0),
        max_parameters=17,
        max_state_array_bytes=200,
    )
    stream = AttributionStream(
        source,
        *fake_data(),
        checkpoint_updates=(0, 2, 4),
        label_delay=2,
        budget=wake,
        capacity=helper,
    )
    stream.run_until(2)
    with pytest.raises(ValueError, match="capacity continuation"):
        stream.pause()
    assert wake.updates_completed == 2 and helper.report["completed_native_calls"] == 2
    assert not any(r[0] == "score" for r in source.trace)
    stream.run_until(4)
    assert stream.finish()["completed_updates"] == 4


@pytest.mark.parametrize("value", [False, True, object()])
def test_should_refuse_noncapacity_helper_before_any_learner_fork(value):
    source = FakeLearner()
    with pytest.raises(ValueError, match="exact StreamCapacity"):
        AttributionStream(
            source,
            *fake_data(),
            checkpoint_updates=(0, 4),
            label_delay=0,
            budget=budget(),
            capacity=value,
        )
    assert source.trace == []


def test_should_refuse_replaced_capacity_identity_before_next_wake():
    source = FakeLearner()
    wake = budget()
    helper = StreamCapacity(
        source,
        wake,
        total_updates=4,
        measure_state=lambda s: CapacityUsage(9, 72, 0),
        max_parameters=17,
        max_state_array_bytes=200,
    )
    stream = AttributionStream(
        source, *fake_data(), checkpoint_updates=(0, 4), label_delay=2, budget=wake, capacity=helper
    )
    stream.run_until(1)
    stream._capacity = None
    with pytest.raises(ValueError, match="original budget, clock and helper"):
        stream.run_until(2)
    assert wake.updates_completed == helper.report["completed_native_calls"] == 1
    assert not any(r[0] == "score" for r in source.trace)


@pytest.mark.parametrize("field,value", [("_calls", 0), ("_rows", 0), ("_max_parameters", 18)])
def test_should_refuse_changed_capacity_work_or_ceilings_before_next_native_wake(field, value):
    source = FakeLearner()
    wake = budget()
    helper = StreamCapacity(
        source,
        wake,
        total_updates=4,
        measure_state=lambda s: CapacityUsage(9, 72, 0),
        max_parameters=17,
        max_state_array_bytes=200,
    )
    stream = AttributionStream(
        source, *fake_data(), checkpoint_updates=(0, 4), label_delay=2, budget=wake, capacity=helper
    )
    stream.run_until(1)
    setattr(helper, field, value)
    with pytest.raises(ValueError, match="capacity"):
        stream.run_until(2)
    assert wake.updates_completed == 1
    assert len([r for r in source.trace if r[0] == "wake"]) == 1
    assert not any(r[0] == "score" for r in source.trace)


def forbidden(*args, **kwargs):
    pytest.fail("unexpected native work")


def test_should_describe_fixed_work_and_caps_without_models_or_sampler(monkeypatch):
    for name in (
        "build_learner",
        "initialize_coupled_scaling",
        "split_coupled_scaling_width",
        "ProcessRssSampler",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    report = pilot.describe_protocol()
    assert report["seed"] == 23 and report["planned_work"]["native_updates"] == 24
    assert report["planned_work"]["training_rows"] == 48
    assert (
        report["planned_work"]["scheduled_splits"]
        == report["planned_work"]["fixed_initial_splits"]
        == 3
    )
    assert report["expected_wake_mean_parameters"] == dict(
        fixed_final_width=17.0, scheduled_split=13.0
    )
    assert report["limits"] == dict(max_parameters=17, max_state_array_bytes=200, max_width=4)
    assert "original R2.6" in report["limitations"]


def test_should_preserve_every_unreached_arm_on_sampler_failure(monkeypatch):
    def fail():
        raise RuntimeError("declared sampler failure")

    monkeypatch.setattr(pilot, "ProcessRssSampler", fail)
    monkeypatch.setattr(pilot, "build_learner", forbidden)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and report["wake_updates"] == 0
    assert len(report["runs"]) == 6 and all(r["status"] == "not_run" for r in report["runs"])
    assert "declared sampler failure" in report["global_errors"][0]


@pytest.mark.parametrize("action", ["describe", "run"])
def test_should_route_capacity_cli_only_to_requested_fixed_scenario(monkeypatch, capsys, action):
    monkeypatch.setattr(cli, "describe_protocol", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    monkeypatch.setattr(
        cli,
        "describe_capacity_protocol" if action == "describe" else "run_capacity_pilot",
        lambda: dict(status="complete", selected=action),
    )
    monkeypatch.setattr(sys, "argv", ["capacity", f"--{action}", "--scenario", "capacity"])
    assert cli.main() == 0 and json.loads(capsys.readouterr().out)["selected"] == action


@pytest.mark.parametrize("flag", ["--recent-only", "--diagnose-learnability"])
def test_should_reject_other_scenario_flags_before_capacity_work(monkeypatch, flag):
    monkeypatch.setattr(cli, "run_capacity_pilot", forbidden)
    monkeypatch.setattr(sys, "argv", ["capacity", "--run", "--scenario", "capacity", flag])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


class FakeSampler:
    start_bytes = 1024

    def sample(self):
        return 1024

    def snapshot(self):
        return ProcessRssSegment(1, 1024, 1024, 1, 0.005)


def test_should_execute_actual_split_delayed_stream_and_scoring_with_capacity_costs(monkeypatch):
    monkeypatch.setattr(budget_module, "ProcessRssSampler", FakeSampler)
    row: dict[str, Any] = dict(
        method="pc", mode="scheduled_split", native_checkpoints=[], checkpoint_reporting_seconds=0.0
    )
    pilot._run_one(row, pilot.monotonic() + 10, cast(Any, FakeSampler()))
    assert row["status"] == "complete", row.get("reason")
    assert row["source_learner_unchanged"] and row["wake_updates"] == 4
    assert [p["width"] for p in row["native_checkpoints"]] == [2, 4, 4]
    assert [p["transition_kinds"] for p in row["native_checkpoints"]] == [[], ["split"], ["split"]]
    assert row["capacity"]["peak_parameters"] == 17
    assert row["capacity"]["mean_parameters"] == 13
    assert row["capacity"]["completed_training_rows"] == 8
    assert row["capacity"]["split_committed"] == 1
    assert row["stream_result"]["score_forward_calls"] == 6
    assert len(row["stream_result"]["applied_consolidations"]) == 1
