"""Bounded inference binding through the real delayed stream, fake native work.

These checks establish composition, isolation and accounting. They do not test
the numerical kernel again or infer a speed or learning benefit from fake work.
"""

from copy import deepcopy
from dataclasses import asdict
import json
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scripts.run_coupled_inference as cli
import src.adapters.coupled_inference_stream as demo
import src.app.toy_execution_budget as budget_module
from src.app.toy_execution_budget import ToyBudgetSession
from src.core.coupled_energy import CoupledParameters
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.learner_ports import TrainingDiagnostic
from src.shared.process_memory import ProcessRssSegment


def forbidden(*args, **kwargs):
    pytest.fail("unselected or native operational work reached")


class FakeSampler:
    start_bytes = 1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def sample(self):
        return 1

    def snapshot(self):
        return ProcessRssSegment(1, 1, 1, 1, 0.005)


class FakeLearner:
    def __init__(self, method, trace, controls, policy, value=0):
        self.method, self.trace, self.controls = method, trace, controls
        self.inference_policy, self.value = deepcopy(policy), value

    def fork(self):
        return FakeLearner(
            self.method, self.trace, self.controls, self.inference_policy, self.value
        )

    def snapshot_state(self):
        return np.array([float(self.value)])

    def restore_state(self, state):
        self.value = int(state[0])

    def train_batch(self, features, targets):
        self.trace.append(("train", self.method, self.value, features.copy(), targets.copy()))
        if self.method == "pc" and self.controls.get("fail_train") and self.value == 5:
            raise RuntimeError("injected paid bounded wake failure")
        self.value += 1
        return TrainingDiagnostic("fake_bounded_stream_v1", 0.0)

    def predict(self, features):
        self.trace.append(("predict", self.method, self.value, features.copy()))
        if self.method == "pc" and self.controls.get("fail_score") and self.value == 6:
            raise RuntimeError("injected paid bounded final failure")
        return np.full((len(features), 1), 0.25 + 0.001 * self.value)


@pytest.fixture
def fake_ports(monkeypatch):
    sources, bindings, sessions = [], [], []
    trace: list[Any] = []
    controls: dict[str, Any] = {}
    parameters = CoupledParameters(
        (np.zeros((2, 8)), np.zeros((8, 8))),
        (np.zeros((1, 8)), np.zeros((1, 8))),
        np.zeros((8, 1)),
        np.zeros((1, 1)),
    )
    original_session = ToyBudgetSession.__init__

    def initialize(self, *args, **kwargs):
        original_session(self, *args, **kwargs)
        sessions.append(self)
        trace.append(("budget", len(sessions)))

    def build(method, initial, *, inference_policy):
        bindings.append((method, initial, inference_policy))
        trace.append(("build", method))
        source = FakeLearner(method, trace, controls, inference_policy)
        sources.append(source)
        return source

    def network(*args, **kwargs):
        trace.append(("network", args, kwargs))
        return SimpleNamespace(snapshot_parameters=lambda: parameters)

    monkeypatch.setattr(ToyBudgetSession, "__init__", initialize)
    monkeypatch.setattr(demo, "build_learner", build)
    monkeypatch.setattr(demo, "CoupledPredictiveCodingNetwork", network)
    monkeypatch.setattr(demo, "ProcessRssSampler", FakeSampler)
    monkeypatch.setattr(budget_module, "ProcessRssSampler", FakeSampler)
    monkeypatch.setattr(demo, "monotonic", lambda: 0.0)
    return SimpleNamespace(
        sources=sources,
        bindings=bindings,
        trace=trace,
        sessions=sessions,
        controls=controls,
        parameters=parameters,
    )


def test_should_describe_frozen_stream_controls_without_constructing_data_or_models(monkeypatch):
    for name in (
        "build_learner",
        "build_stream",
        "CoupledPredictiveCodingNetwork",
        "ProcessRssSampler",
    ):
        monkeypatch.setattr(demo, name, forbidden)
    descriptor = demo.describe_stream_protocol()
    assert descriptor["methods"] == ["pc", "neutral_cpc", "gated_cpc"]
    assert descriptor["resources"]["training_updates"] == 54
    assert descriptor["resources"]["wall_seconds"] == 30
    assert descriptor["resources"]["process_memory_bytes"] == 256 * 1024**2
    assert descriptor["resources"]["cpu_threads"] == 1
    assert descriptor["resources"]["gpu_memory_bytes"] == 0


def test_should_forward_the_prespecified_policy_and_identical_initial_parameters(fake_ports):
    report = demo.run_stream_example()
    assert report["status"] == "complete", report["global_errors"]
    assert [method for method, _, _ in fake_ports.bindings] == ["pc", "neutral_cpc", "gated_cpc"]
    for _, initial, policy in fake_ports.bindings:
        assert type(policy) is CoupledInferencePolicy
        assert asdict(policy) == dict(
            min_steps=2, gradient_tolerance=1.0, relative_energy_tolerance=None
        )
        for left, right in zip(
            initial.hidden_weights, fake_ports.parameters.hidden_weights, strict=True
        ):
            np.testing.assert_array_equal(left, right)
        for left, right in zip(
            initial.hidden_biases, fake_ports.parameters.hidden_biases, strict=True
        ):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(initial.output_weight, fake_ports.parameters.output_weight)
        np.testing.assert_array_equal(initial.output_bias, fake_ports.parameters.output_bias)
    assert all(source.value == 0 for source in fake_ports.sources)
    assert report["scientific_advantage_verified"] is report["full_R15b_complete"] is False


def test_should_run_all_delayed_wakes_before_each_cases_final_scores_and_preserve_sources(
    fake_ports,
):
    report = demo.run_stream_example()
    assert report["status"] == "complete", report["global_errors"]
    assert [row["method"] for row in report["runs"]] == ["pc", "neutral_cpc", "gated_cpc"]
    training, final = demo.build_stream("recurring")
    for row in report["runs"]:
        assert row["status"] == "complete", row["reason"]
        assert row["completed_updates"] == 18 and row["source_learner_unchanged"] is True
        stream = row["stream_result"]
        assert stream["completed_updates"] == len(stream["applied_updates"]) == 18
        assert stream["training_examples"] == 144 and stream["score_forward_calls"] == 8
        assert [point["updates"] for point in stream["checkpoints"]] == [0, 6, 12, 18]
        assert [point["training_context"] for point in stream["checkpoints"]] == [
            None,
            "A",
            "B",
            "A",
        ]
        trace = [
            event
            for event in fake_ports.trace
            if event[0] in {"train", "predict"} and event[1] == row["method"]
        ]
        assert [event[0] for event in trace] == ["train"] * 18 + ["predict"] * 8
        assert [event[2] for event in trace[:18]] == list(range(18))
        assert [event[2] for event in trace[18:]] == [0, 0, 6, 6, 12, 12, 18, 18]
        for event, batch in zip(trace[:18], training, strict=True):
            np.testing.assert_array_equal(event[3], batch.features)
            np.testing.assert_array_equal(event[4], batch.targets)
        for index, event in enumerate(trace[18:]):
            np.testing.assert_array_equal(event[3], final[index % 2].features)
        progress = row["final_score_progress"]
        assert progress["status"] == "complete"
        assert progress["prediction_calls_attempted"] == progress["prediction_calls_completed"] == 8
        assert progress["prediction_rows_attempted"] == progress["prediction_rows_completed"] == 512
    assert report["pc_neutral_exact"] is True
    assert all(source.value == 0 for source in fake_ports.sources)
    assert sum(event[0] == "train" for event in fake_ports.trace) == 54
    assert sum(event[0] == "predict" for event in fake_ports.trace) == 24


def test_should_use_original_setup_sessions_with_one_global_clock_deadline_and_sampler(fake_ports):
    report = demo.run_stream_example()
    assert report["status"] == "complete", report["global_errors"]
    sessions = fake_ports.sessions
    assert len(sessions) == len({id(session) for session in sessions}) == 3
    assert all(s.budget.max_training_updates == s.updates_completed == 18 for s in sessions)
    assert all(s.budget.max_wall_seconds == 30 and s.started_at == 0.0 for s in sessions)
    assert all(s.budget.max_process_rss_bytes == 256 * 1024**2 for s in sessions)
    assert all(s.budget.max_replay_examples is None for s in sessions)
    assert all(s.clock is sessions[0].clock for s in sessions)
    assert all(s.process_rss_sampler is sessions[0].process_rss_sampler for s in sessions)
    first_build = next(index for index, event in enumerate(fake_ports.trace) if event[0] == "build")
    assert sum(event[0] == "budget" for event in fake_ports.trace[:first_build]) == 3


def test_should_preserve_paid_wakes_and_refuse_finals_after_a_native_update_failure(fake_ports):
    fake_ports.controls["fail_train"] = True
    report = demo.run_stream_example()
    assert report["status"] == "incomplete"
    assert len(report["runs"]) == 3
    row = report["runs"][0]
    assert row["method"] == "pc" and row["status"] == "failed"
    assert "paid bounded wake failure" in row["reason"]
    assert row["completed_updates"] == 5
    assert fake_ports.sessions[0].updates_completed == 5
    trace = [
        event for event in fake_ports.trace if event[0] in {"train", "predict"} and event[1] == "pc"
    ]
    assert len(trace) == 6 and all(event[0] == "train" for event in trace)
    assert not row["final_score_progress"]
    assert all(source.value == 0 for source in fake_ports.sources)
    assert report["pc_neutral_exact"] is None


def test_should_preserve_paid_partial_final_predictions_after_training_completed(fake_ports):
    fake_ports.controls["fail_score"] = True
    report = demo.run_stream_example()
    assert report["status"] == "incomplete"
    row = report["runs"][0]
    assert row["status"] == "failed" and "paid bounded final failure" in row["reason"]
    assert row["completed_updates"] == fake_ports.sessions[0].updates_completed == 18
    progress = row["final_score_progress"]
    assert progress["status"] == "failed"
    assert progress["prediction_calls_attempted"] == 3
    assert progress["prediction_calls_completed"] == 2
    assert progress["prediction_rows_attempted"] == 192
    assert progress["prediction_rows_completed"] == 128
    assert [point["updates"] for point in progress["checkpoints"]] == [0, 6]
    assert progress["checkpoints"][0]["status"] == "complete"
    assert set(progress["checkpoints"][0]["scores"]) == {"A", "B"}
    assert progress["checkpoints"][1]["status"] == "incomplete"
    assert progress["checkpoints"][1]["scores"] == {}
    assert all(source.value == 0 for source in fake_ports.sources)


@pytest.mark.parametrize("phase", ["constructor", "enter"])
def test_should_keep_all_three_planned_rows_when_global_setup_fails(monkeypatch, phase):
    class FailedSampler:
        def __init__(self):
            if phase == "constructor":
                raise RuntimeError("declared sampler setup failure")

        def __enter__(self):
            raise RuntimeError("declared sampler setup failure")

        def __exit__(self, *args):
            return False

        def snapshot(self):
            raise RuntimeError("no memory observation")

    monkeypatch.setattr(demo, "ProcessRssSampler", FailedSampler)
    monkeypatch.setattr(demo, "CoupledPredictiveCodingNetwork", forbidden)
    monkeypatch.setattr(demo, "build_learner", forbidden)
    report = demo.run_stream_example()
    assert report["status"] == "incomplete" and report["global_errors"]
    assert len(report["runs"]) == 3
    assert [row["method"] for row in report["runs"]] == ["pc", "neutral_cpc", "gated_cpc"]
    assert all(row["completed_updates"] == 0 for row in report["runs"])
    assert all(row["status"] == "not_run" for row in report["runs"])
    assert all(not row["final_score_progress"] for row in report["runs"])
    assert report["pc_neutral_exact"] is None


@pytest.mark.parametrize("stream", [False, True])
def test_should_route_only_requested_cli_and_preserve_the_original_zero_argument_call(
    stream, monkeypatch, capsys
):
    calls = []

    def selected():
        calls.append("stream" if stream else "legacy")
        return {"status": "complete", "scope": calls[-1]}

    monkeypatch.setattr(cli, "run_example", forbidden if stream else selected)
    monkeypatch.setattr(cli, "run_stream_example", selected if stream else forbidden)
    monkeypatch.setattr(sys, "argv", ["inference", *(["--stream"] if stream else [])])
    cli.main()
    assert calls == ["stream" if stream else "legacy"]
    assert json.loads(capsys.readouterr().out) == {"status": "complete", "scope": calls[0]}


def test_should_refuse_unfrozen_cli_arguments_before_any_work(monkeypatch):
    monkeypatch.setattr(cli, "run_example", forbidden)
    monkeypatch.setattr(cli, "run_stream_example", forbidden)
    monkeypatch.setattr(sys, "argv", ["inference", "--stream", "--seed", "7"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_should_preserve_both_complete_streams_but_fail_qualification_on_neutral_difference(
    fake_ports, monkeypatch
):
    original = FakeLearner.predict

    def changed(self, features):
        value = original(self, features)
        return value + 0.01 if self.method == "neutral_cpc" else value

    monkeypatch.setattr(FakeLearner, "predict", changed)
    report = demo.run_stream_example()
    assert all(r["status"] == "complete" for r in report["runs"])
    assert report["pc_neutral_exact"] is False
    assert report["status"] == "incomplete" and report["global_errors"]


def test_should_print_incomplete_stream_report_and_exit_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_example", forbidden)
    monkeypatch.setattr(cli, "run_stream_example", lambda: {"status": "incomplete"})
    monkeypatch.setattr(sys, "argv", ["inference", "--stream"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out) == {"status": "incomplete"}
