"""Explicit replay removal through the existing runner, using fake learners."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.run_stream_rehearsal_pilot as cli
import src.adapters.stream_rehearsal_pilot as pilot
import src.app.toy_execution_budget as budget_module
from src.app.actor_shadow import ActorShadowRuntime
from src.app.toy_execution_budget import ToyBudgetSession
from test_stream_rehearsal_pilot import FakeLearner, FakeModel, FakeSampler
from test_rehearsal_learnability import FakeLearner as ProbeLearner, fake_environment


def forbidden(*args, **kwargs):
    pytest.fail("replay construction or native work reached in the no-replay control")


def refuse_replay(monkeypatch):
    for name in ("replay_policy", "StreamRehearsal", "ReplayRetentionBudget", "ReplayAllocation"):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(ActorShadowRuntime, "consolidate", forbidden)


def capture_wake_and_probe_sessions(monkeypatch):
    sessions = []
    original = ToyBudgetSession.__init__

    def initialize(self, budget, *args, **kwargs):
        assert budget.max_replay_examples is None, "a removed replay allowance was allocated"
        original(self, budget, *args, **kwargs)
        sessions.append(self)

    monkeypatch.setattr(ToyBudgetSession, "__init__", initialize)
    return sessions


@pytest.fixture
def fake_ports(monkeypatch):
    traces, sources = [], []

    def build(method, initial):
        trace: list[tuple[str, int]] = []
        source = FakeLearner(trace)
        traces.append(trace)
        sources.append(source)
        return source

    monkeypatch.setattr(pilot, "ProcessRssSampler", FakeSampler)
    monkeypatch.setattr(budget_module, "ProcessRssSampler", FakeSampler)
    monkeypatch.setattr(pilot, "CoupledPredictiveCodingNetwork", FakeModel)
    monkeypatch.setattr(pilot, "build_learner", build)
    monkeypatch.setattr(pilot, "monotonic", lambda: 0.0)
    refuse_replay(monkeypatch)
    sessions = capture_wake_and_probe_sessions(monkeypatch)
    return SimpleNamespace(traces=traces, sources=sources, sessions=sessions)


@pytest.mark.parametrize("diagnose", [False, True])
def test_should_describe_actual_removed_replay_work_without_constructing_a_policy(
    diagnose, monkeypatch
):
    refuse_replay(monkeypatch)
    monkeypatch.setattr(pilot, "build_learner", forbidden)
    monkeypatch.setattr(pilot, "CoupledPredictiveCodingNetwork", forbidden)
    descriptor = pilot.describe_protocol(no_replay=True, diagnose_learnability=diagnose)
    assert descriptor["protocol_id"] == (
        "coherent_no_rehearsal_reserved_sixteen_v1"
        if diagnose
        else "coherent_stream_no_rehearsal_v1"
    )
    assert descriptor["replay"] is None
    assert descriptor["methods"] == ["bp", "pc", "neutral_cpc", "gated_cpc"]
    assert descriptor["seeds"] == [23]
    assert descriptor["training_rng"] == 7301 and descriptor["final_rng"] == 9101
    assert descriptor["hidden_dims"] == [8, 8]
    assert descriptor["learning_rate"] == 0.03
    assert descriptor["inference_steps"] == 4 and descriptor["inference_learning_rate"] == 0.2
    assert descriptor["label_delay"] == 2
    assert descriptor["checkpoint_updates"] == [0, 6, 12, 18]
    plan = descriptor["planned_work"]
    assert plan["wake_updates"] == 72 and plan["wake_row_exposures"] == 576
    assert (
        plan["replay_updates"] == plan["replay_row_exposures"] == plan["consolidation_events"] == 0
    )
    assert plan["optimizer_updates"] == (328 if diagnose else 72)
    assert plan["total_row_exposures"] == (2624 if diagnose else 576)
    assert descriptor["resources"]["training_updates"] == (328 if diagnose else 72)
    assert descriptor["resources"]["replay_bytes"] == 0


def test_should_run_only_eighteen_arrived_wakes_then_deferred_scores_in_each_arm(fake_ports):
    report = pilot.run_pilot(no_replay=True)
    assert report["status"] == "complete", [row["reason"] for row in report["runs"]]
    assert [row["method"] for row in report["runs"]] == ["bp", "pc", "neutral_cpc", "gated_cpc"]
    assert report["replay_disabled"] is report["no_replay_verified"] is True
    assert report["wake_updates"] == 72 and report["row_exposures_charged"] == 576
    assert (
        report["replay_charged_examples"]
        == report["replay_returned_updates"]
        == report["replay_committed_updates"]
        == 0
    )
    assert report["shared_replay_plans_exact"] is None
    assert report["pc_neutral_exact"] is True
    assert len(fake_ports.sessions) == 4
    assert all(
        session.budget.max_training_updates == session.updates_completed == 18
        for session in fake_ports.sessions
    )
    for source, trace, row in zip(
        fake_ports.sources, fake_ports.traces, report["runs"], strict=True
    ):
        assert source.state == 0 and row["source_learner_unchanged"] is True
        assert row["rehearsal"] is None and row["planned_event_updates"] == []
        assert row["wake_updates"] == 18 and row["replay_charged_examples"] == 0
        assert trace == [("train", 8)] * 18 + [("score", n) for n in (0, 0, 6, 6, 12, 12, 18, 18)]
        stream = row["stream_result"]
        assert len(stream["applied_updates"]) == stream["completed_updates"] == 18
        assert "applied_consolidations" not in stream
        assert stream["training_examples"] == 144 and stream["score_forward_calls"] == 8
    assert report["scientific_advantage_verified"] is report["full_R23_complete"] is False


def test_should_retain_completed_wakes_and_refuse_final_scores_after_a_paid_failure(
    fake_ports, monkeypatch
):
    original = FakeLearner.train_batch

    def fail(self, features, targets):
        if self.trace is fake_ports.traces[0] and self.state == 5:
            self.trace.append(("failed_attempt", len(features)))
            raise RuntimeError("injected no-replay wake failure")
        return original(self, features, targets)

    monkeypatch.setattr(FakeLearner, "train_batch", fail)
    report = pilot.run_pilot(no_replay=True)
    assert report["status"] == "incomplete" and report["no_replay_verified"] is False
    assert report["replay_disabled"] is True
    first = report["runs"][0]
    assert first["status"] == "failed" and "no-replay wake failure" in first["reason"]
    assert first["wake_updates"] == fake_ports.sessions[0].updates_completed == 5
    assert fake_ports.traces[0] == [("train", 8)] * 5 + [("failed_attempt", 8)]
    assert report["wake_updates"] == 59 and report["row_exposures_charged"] == 472
    assert report["replay_charged_examples"] == 0 and report["shared_replay_plans_exact"] is None
    assert report["pc_neutral_exact"] is None
    assert all(source.state == 0 for source in fake_ports.sources)


def test_should_refuse_verified_completion_if_the_source_learner_changed(fake_ports, monkeypatch):
    original = pilot.run_attribution_stream

    def mutate(source, *args, **kwargs):
        result = original(source, *args, **kwargs)
        source.state = 999
        return result

    monkeypatch.setattr(pilot, "run_attribution_stream", mutate)
    report = pilot.run_pilot(no_replay=True)
    assert report["status"] == "incomplete" and report["no_replay_verified"] is False
    assert all(row["status"] == "failed" for row in report["runs"])
    assert report["wake_updates"] == 72 and report["replay_charged_examples"] == 0
    assert report["shared_replay_plans_exact"] is None


def test_should_compose_real_disposable_probes_without_a_replay_budget_or_consolidation(
    monkeypatch,
):
    control = fake_environment(monkeypatch, mutate_arguments=True)
    refuse_replay(monkeypatch)
    sessions = capture_wake_and_probe_sessions(monkeypatch)
    report = pilot.run_pilot(no_replay=True, diagnose_learnability=True)
    assert report["status"] == "complete", [row["reason"] for row in report["runs"]]
    assert report["protocol"]["protocol_id"] == "coherent_no_rehearsal_reserved_sixteen_v1"
    assert report["protocol"]["replay"] is None
    assert report["no_replay_verified"] is report["replay_disabled"] is True
    assert report["total_completed_updates"] == 328
    assert report["total_training_row_exposures"] == 2624
    assert report["probe_attempted_updates"] == report["probe_completed_updates"] == 256
    assert report["probe_training_row_exposures"] == 2048
    assert report["replay_charged_examples"] == report["replay_committed_updates"] == 0
    assert report["pc_neutral_exact"] is report["pc_neutral_probe_curves_exact"] is True
    assert len(sessions) == 20  # Four wake sessions and sixteen fixed probe allowances.
    assert all(session.budget.max_replay_examples is None for session in sessions)
    for method, row in zip(pilot.METHODS, report["runs"], strict=True):
        assert row["rehearsal"] is None and row["planned_event_updates"] == []
        assert "applied_consolidations" not in row["stream_result"]
        assert row["source_learner_unchanged"] is row["probe_template_unchanged"] is True
        probes = [entry for entry in control["probes"] if entry[0].method == method]
        assert [state for _, state, _ in probes] == [0.0, 6.0, 12.0, 18.0]
        events = [event for event in control["trace"] if event[0] == method]
        benchmark = [event for event in events if event[1:3] == ("benchmark", "train")]
        assert len(benchmark) == 18 and all(event[3] == 8 for event in benchmark)
        first_final = next(
            index for index, event in enumerate(events) if event[1:3] == ("benchmark", "predict")
        )
        assert first_final > max(index for index, event in enumerate(events) if event[2] == "train")


def test_should_preserve_partial_probe_work_and_stop_the_corresponding_benchmark(monkeypatch):
    control = fake_environment(monkeypatch)
    refuse_replay(monkeypatch)
    original = ProbeLearner.train_batch

    def fail(self, *args):
        if self.origin == "template" and self.start_state == 6 and self.trained == 2:
            raise RuntimeError("injected no-replay probe failure")
        return original(self, *args)

    monkeypatch.setattr(ProbeLearner, "train_batch", fail)
    report = pilot.run_pilot(no_replay=True, diagnose_learnability=True)
    assert report["status"] == "incomplete" and report["no_replay_verified"] is False
    assert report["wake_updates"] == 24 and report["replay_charged_examples"] == 0
    assert report["probe_attempted_updates"] == 76 and report["probe_completed_updates"] == 72
    for row in report["runs"]:
        assert row["wake_updates"] == 6 and row["status"] == "failed"
        assert [observation["status"] for observation in row["observations"]] == [
            "complete",
            "failed",
            "not_run",
            "not_run",
        ]
        probe = row["observations"][1]["probe"]
        assert probe["attempted_updates"] == 3 and probe["completed_updates"] == 2
    assert not any(event[1:3] == ("benchmark", "predict") for event in control["trace"])


@pytest.mark.parametrize("diagnose", [False, True])
def test_should_preserve_existing_descriptors_when_no_replay_is_explicitly_false(diagnose):
    assert pilot.describe_protocol(diagnose_learnability=diagnose) == pilot.describe_protocol(
        diagnose_learnability=diagnose, no_replay=False
    )
    assert pilot.describe_protocol(
        diagnose_learnability=diagnose, recent_only=True
    ) == pilot.describe_protocol(diagnose_learnability=diagnose, recent_only=True, no_replay=False)


@pytest.mark.parametrize("diagnose", [False, True])
def test_should_preserve_old_internal_calls_and_omit_new_fields_when_false(diagnose, monkeypatch):
    calls = []

    def case(row, *args, **kwargs):
        calls.append(kwargs)
        row.update(status="failed", reason="stubbed old-mode execution")

    monkeypatch.setattr(pilot, "ProcessRssSampler", FakeSampler)
    monkeypatch.setattr(pilot, "CoupledPredictiveCodingNetwork", FakeModel)
    monkeypatch.setattr(pilot, "_run_one", case)
    report = pilot.run_pilot(diagnose_learnability=diagnose, no_replay=False)
    assert len(calls) == 4
    assert calls == ([{"diagnose_learnability": True}] * 4 if diagnose else [{}] * 4)
    assert "replay_disabled" not in report and "no_replay_verified" not in report


@pytest.mark.parametrize("entry", [pilot.describe_protocol, pilot.run_pilot])
@pytest.mark.parametrize("invalid", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_non_boolean_removal_before_native_work(entry, invalid, monkeypatch):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(pilot, "build_stream", forbidden)
    with pytest.raises(ValueError, match="bool"):
        entry(no_replay=invalid)


@pytest.mark.parametrize("entry", [pilot.describe_protocol, pilot.run_pilot])
def test_should_refuse_conflicting_recent_and_no_replay_modes_before_work(entry, monkeypatch):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError):
        entry(no_replay=True, recent_only=True)


def test_should_refuse_redefining_the_tracked_preset_with_replay_removed(monkeypatch):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError):
        pilot.run_pilot(no_replay=True, track_unit_history=True, diagnose_learnability=True)


def test_should_preserve_all_unreached_rows_on_global_resource_setup_failure(monkeypatch):
    def fail():
        raise RuntimeError("declared no-replay setup failure")

    monkeypatch.setattr(pilot, "ProcessRssSampler", fail)
    monkeypatch.setattr(pilot, "build_learner", forbidden)
    report = pilot.run_pilot(no_replay=True)
    assert report["status"] == "incomplete" and report["no_replay_verified"] is False
    assert report["replay_disabled"] is True
    assert len(report["runs"]) == 4 and all(row["status"] == "not_run" for row in report["runs"])
    assert all(
        row["planned_event_updates"] == [] and row["rehearsal"] is None for row in report["runs"]
    )
    assert report["wake_updates"] == report["replay_charged_examples"] == 0
    assert report["shared_replay_plans_exact"] is None


@pytest.mark.parametrize("action", ["describe", "run"])
@pytest.mark.parametrize("diagnose", [False, True])
def test_should_forward_only_explicit_no_replay_cli_options(action, diagnose, monkeypatch, capsys):
    calls = []

    def selected(**options):
        calls.append(options)
        return {"status": "complete"}

    monkeypatch.setattr(cli, "describe_protocol", selected if action == "describe" else forbidden)
    monkeypatch.setattr(cli, "run_pilot", selected if action == "run" else forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rehearsal",
            f"--{action}",
            "--no-replay",
            *(["--diagnose-learnability"] if diagnose else []),
        ],
    )
    assert cli.main() == 0
    assert calls == [dict(no_replay=True, **({"diagnose_learnability": True} if diagnose else {}))]
    assert json.loads(capsys.readouterr().out) == {"status": "complete"}


@pytest.mark.parametrize(
    "scenario",
    [
        "paired-retention",
        "tracked-learnability",
        "long-stream",
        "slow-teacher",
        "capacity",
        "capacity-bp",
        "capacity-comparison",
    ],
)
def test_should_refuse_no_replay_for_other_frozen_cli_scenarios(scenario, monkeypatch):
    monkeypatch.setattr(cli, "describe_protocol", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    for name in (
        "describe_paired_protocol",
        "describe_tracked_protocol",
        "describe_long_protocol",
        "describe_teacher_protocol",
        "describe_capacity_protocol",
        "describe_bp_protocol",
        "describe_capacity_comparison_protocol",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(
        sys, "argv", ["rehearsal", "--describe", "--scenario", scenario, "--no-replay"]
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_should_refuse_conflicting_cli_replay_controls_before_dispatch(monkeypatch):
    monkeypatch.setattr(cli, "describe_protocol", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    monkeypatch.setattr(sys, "argv", ["rehearsal", "--describe", "--recent-only", "--no-replay"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
