"""Thin fixed protocol/CLI binding without native models or physical RSS."""

from dataclasses import dataclass
import json
import sys

import numpy as np
import pytest

import scripts.run_stream_rehearsal_pilot as cli
import src.adapters.stream_rehearsal_pilot as pilot
import src.app.toy_execution_budget as budget_module
from src.core.learner_ports import TrainingDiagnostic
from src.shared.process_memory import ProcessRssSegment


def forbidden(*args, **kwargs):
    pytest.fail("unexpected native work")


def test_should_describe_frozen_comparison_without_constructing_models(monkeypatch):
    for name in ("build_stream", "build_learner", "CoupledPredictiveCodingNetwork"):
        monkeypatch.setattr(pilot, name, forbidden)
    report = pilot.describe_protocol()
    assert report["methods"] == ["bp", "pc", "neutral_cpc", "gated_cpc"]
    assert report["seeds"] == [23]
    assert report["checkpoint_updates"] == [0, 6, 12, 18]
    assert report["planned_work"] == dict(
        wake_updates=72,
        replay_updates=48,
        optimizer_updates=120,
        wake_row_exposures=576,
        replay_row_exposures=48,
        total_row_exposures=624,
        consolidation_events=24,
        final_score_forwards=32,
        probes=0,
        svd=0,
    )
    assert report["replay"]["every_updates"] == 3
    assert report["replay"]["rows_per_event"] == 2
    assert report["replay"]["seed"] == 53
    assert report["resources"]["replay_bytes"] == 384
    assert "no confirmatory" in report["limitations"]


@pytest.mark.parametrize("action", ("describe", "run"))
def test_should_route_cli_only_to_requested_action(monkeypatch, capsys, action):
    monkeypatch.setattr(cli, "describe_protocol", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    monkeypatch.setattr(
        cli,
        "describe_protocol" if action == "describe" else "run_pilot",
        lambda: dict(status="complete", selected=action),
    )
    monkeypatch.setattr(sys, "argv", ["rehearsal", f"--{action}"])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["selected"] == action


def test_should_return_failure_status_for_incomplete_run(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_pilot", lambda: dict(status="incomplete"))
    monkeypatch.setattr(sys, "argv", ["rehearsal", "--run"])
    assert cli.main() == 2
    assert json.loads(capsys.readouterr().out)["status"] == "incomplete"


def test_should_preserve_all_method_rows_on_global_setup_failure(monkeypatch):
    def fail():
        raise RuntimeError("declared sampler failure")

    monkeypatch.setattr(pilot, "ProcessRssSampler", fail)
    monkeypatch.setattr(pilot, "CoupledPredictiveCodingNetwork", forbidden)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete"
    assert report["wake_updates"] == report["replay_charged_examples"] == 0
    assert len(report["runs"]) == 4
    assert all(row["status"] == "not_run" for row in report["runs"])
    assert "declared sampler failure" in report["global_errors"][0]


class FakeSampler:
    start_bytes = 1024

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def sample(self):
        return 1024

    def snapshot(self):
        return ProcessRssSegment(1, 1024, 1024, 1, 0.005)


@dataclass
class Parameters:
    weights: np.ndarray


class FakeModel:
    def __init__(self, *args):
        pass

    def snapshot_parameters(self):
        return Parameters(np.zeros((2, 1)))


class FakeLearner:
    def __init__(self, trace, state=0):
        self.trace, self.state = trace, state

    def fork(self):
        return FakeLearner(self.trace, self.state)

    def snapshot_state(self):
        return self.state

    def restore_state(self, state):
        self.state = state

    def train_batch(self, features, targets):
        self.trace.append(("train", len(features)))
        self.state += 1
        return TrainingDiagnostic("fake_count", float(self.state))

    def predict(self, features):
        self.trace.append(("score", self.state))
        return np.full((len(features), 1), 0.5 + self.state / 1000)


def test_should_compose_all_four_arms_with_separate_initial_budgets_and_deferred_scores(
    monkeypatch,
):
    traces = []

    def build(method, initial):
        trace: list[tuple[str, int]] = []
        traces.append(trace)
        return FakeLearner(trace)

    monkeypatch.setattr(pilot, "ProcessRssSampler", FakeSampler)
    # Bind both sides of the exact sampler boundary to this test-only double.
    monkeypatch.setattr(budget_module, "ProcessRssSampler", FakeSampler)
    monkeypatch.setattr(pilot, "CoupledPredictiveCodingNetwork", FakeModel)
    monkeypatch.setattr(pilot, "build_learner", build)
    report = pilot.run_pilot()
    assert report["status"] == "complete", [row.get("reason") for row in report["runs"]]
    assert report["wake_updates"] == 72
    assert report["replay_charged_examples"] == 48
    assert report["row_exposures_charged"] == 624
    assert report["pc_neutral_exact"] is True
    for trace, row in zip(traces, report["runs"], strict=True):
        train = [event for event in trace if event[0] == "train"]
        assert len(train) == 30
        assert sum(event[1] for event in train) == 156
        assert [event[0] for event in trace] == ["train"] * 30 + ["score"] * 8
        assert row["source_learner_unchanged"] is True
        assert len(row["stream_result"]["applied_consolidations"]) == 6
        assert len(row["stream_result"]["applied_updates"]) == 18
    assert report["scientific_advantage_verified"] is False
    assert report["full_R23_complete"] is False
