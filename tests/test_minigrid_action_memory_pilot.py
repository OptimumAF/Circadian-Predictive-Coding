"""Real count/stream integration with explicit fake physics, not gameplay evidence."""

from dataclasses import replace
import json
import pickle
import sys
from types import SimpleNamespace
from typing import Any

import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import minigrid_action_memory_pilot as pilot
from src.adapters import minigrid_learning_pilot as shared
from src.adapters import minigrid_policy_evaluation as evaluation
from src.app import toy_execution_budget as budgets
from src.core.action_memory import ACTION_MEMORY_COUNT_DIAGNOSTIC
from src.adapters.action_memory_learner import ActionMemoryLearner
from time import monotonic
from test_minigrid_learning_pilot import Environment, source_batch
from test_torch_feature_rehearsal_pilot import Sampler, forbidden


@pytest.fixture
def case(monkeypatch):
    sources = tuple(
        source_batch(context, cue) for context in ("A", "B", "C_train") for cue in ("blue", "red")
    )
    work: dict[str, Any] = {}
    monkeypatch.setattr(Environment, "work", work)
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", Environment)
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(budgets, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(shared, "require_torch", forbidden)
    monkeypatch.setattr(shared, "_same_state", forbidden)

    def collect(check, reports):
        check()
        reports.extend(dict(status="engineering_fixture", environment_steps=9) for _ in sources)
        return sources

    monkeypatch.setattr(shared, "_collect_sources", collect)
    row = shared._row(pilot.METHOD)
    monkeypatch.setattr(shared, "_row", lambda method: row)
    return sources, row, work


def test_should_describe_and_dispatch_without_any_model_or_physics(monkeypatch, capsys):
    monkeypatch.setattr(pilot, "ActionMemoryLearner", forbidden)
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(pilot, "run_pilot", forbidden)
    monkeypatch.setattr(
        sys, "argv", ["runner", "--scenario", "minigrid-action-memory", "--describe"]
    )
    cli.main()
    report = json.loads(capsys.readouterr().out)
    assert report == pilot.describe_protocol()
    assert report["policy"] == dict(
        method="observation_memory",
        feature_dim=296,
        num_classes=7,
        max_entries=64,
        max_payload_bytes=155136,
    )
    assert report["checkpoint_updates"] == [0, 32, 64, 96, 128]
    assert report["planned_work"]["storage_updates"] == 128
    assert report["planned_work"]["training_rows"] == 1152
    assert report["limits"]["gpu"] == 0 and report["trainable_parameters"] is None


def test_should_run_real_128_update_memory_stream_and_all_train_readouts_without_torch(case):
    sources, row, work = case
    before = pickle.dumps(sources)
    report = pilot.run_pilot()
    assert report["status"] == "complete", report["reason"]
    assert report["scientific_advantage_verified"] is False
    result = report["methods"][0]
    assert result["charged_updates"] == 128 and result["native_training_examples"] == 1152
    assert result["source_unchanged"] is result["actor_unchanged"] is True
    assert result["stream_result"]["evaluation_scope"] == "training_only"
    receipts = result["stream_result"]["applied_updates"]
    assert len(receipts) == 128
    assert all(r["diagnostic"]["definition"] == ACTION_MEMORY_COUNT_DIAGNOSTIC for r in receipts)
    assert result["final_evaluation"]["status"] == "not_applicable"
    assert [r["updates"] for r in result["readouts"]] == [0, 32, 64, 96, 128]
    assert all(r["status"] == "complete" for r in result["readouts"])
    assert result["work"]["raw_score_forwards_completed"] == 30
    assert result["work"]["raw_score_rows_completed"] == 270
    assert result["work"]["live_prediction_calls_returned"] == 60
    assert result["work"]["environment_steps_returned"] == 60
    assert len(work["environments"]) == 30 and all(e.closed for e in work["environments"])
    assert all(e.role == "train" for e in work["environments"])
    assert len(report["observations"]) == 5
    assert all(o["logical_payload_bytes"] == 2424 * o["entries"] for o in report["observations"])
    assert [o["native_training_examples"] for o in report["observations"]] == [
        0,
        288,
        576,
        864,
        1152,
    ]
    assert pickle.dumps(sources) == before
    assert all(type(value) is bytes for value in row["readouts"])
    assert report["methods"][0] is not row


@pytest.mark.parametrize("fault", ["corrupt", "replace", "source", "charge"])
def test_should_refuse_prior_success_or_original_source_budget_mutation_before_more_work(
    case, monkeypatch, fault
):
    sources, row, work = case
    original = pilot.evaluate_checkpoint
    calls = []
    native_budget = []
    train = shared._train_arm

    def capture(*args, **options):
        arm = train(*args, **options)
        native_budget.append(arm["budget"])
        return arm

    monkeypatch.setattr(shared, "_train_arm", capture)

    def evaluate(*args, **options):
        calls.append(options["updates"])
        if len(calls) == 2:
            if fault == "corrupt":
                row["readouts"][0] = b"corrupt"
            elif fault == "replace":
                row["readouts"][0] = bytes(bytearray(row["readouts"][0]))
            elif fault == "source":
                sources[0].features[0, 0] = 0.5
            else:
                native_budget[0].updates_completed = 127
        return original(*args, **options)

    monkeypatch.setattr(pilot, "evaluate_checkpoint", evaluate)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete"
    result = report["methods"][0]
    assert calls == [0, 32]
    assert result["charged_updates"] == 128  # original committed report, never refunded
    assert result["readouts"][0]["status"] == "complete"
    assert all(r["status"] == "not_run" for r in result["readouts"][2:])
    assert len(work["environments"]) == 6  # only preceding successful readout ran
    assert len(report["observations"]) == 5


def test_should_preserve_partial_training_charges_and_refuse_a_different_memory_policy(
    case, monkeypatch
):
    from src.adapters.action_memory_learner import ActionMemoryLearner

    monkeypatch.setattr(
        pilot,
        "_learner",
        lambda method: ActionMemoryLearner(replace(pilot.memory_policy(), max_payload_bytes=10752)),
    )
    report = pilot.run_pilot()
    assert report["status"] == "incomplete"
    assert "policy" in report["reason"]
    assert report["methods"][0]["wake_calls_attempted"] == 0
    assert report["methods"][0]["charged_updates"] is None


def test_should_retain_failed_prediction_work_without_refunding_training(case, monkeypatch):
    original = pilot.evaluate_checkpoint
    calls = []

    def evaluate(*args, **options):
        calls.append(options["updates"])
        if len(calls) == 2:
            case[2]["step_error"] = True
        return original(*args, **options)

    monkeypatch.setattr(pilot, "evaluate_checkpoint", evaluate)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete"
    row = report["methods"][0]
    assert row["charged_updates"] == 128 and row["native_training_examples"] == 1152
    assert row["readouts"][0]["status"] == "complete"
    failed = row["readouts"][1]
    assert failed["status"] == "failed"
    assert failed["work"]["live_prediction_calls_returned"] == 1
    assert failed["work"]["environment_steps_attempted"] == 1
    assert failed["work"]["environment_steps_returned"] == 0
    assert failed["work"]["failure_work_may_be_unobserved"] is True
    assert all(r["status"] == "not_run" for r in row["readouts"][2:])
    assert len(report["observations"]) == 5


def small_arm(case):
    row = shared._row(pilot.METHOD)
    packets = pilot._history_packets(shared._packets(case[0])[:32])
    return row, packets


def test_should_preserve_receipts_before_a_closed_candidate_readback(case, monkeypatch):
    row, packets = small_arm(case)
    original = ActionMemoryLearner.train_batch
    calls = []
    candidates = []
    original_stream = shared.AttributionStream

    def stream(*args, **options):
        owner = original_stream(*args, **options)
        candidates.append(owner._runtime._candidate)
        return owner

    def unavailable():
        raise RuntimeError("engineering closed candidate readback")

    monkeypatch.setattr(shared, "AttributionStream", stream)

    def train(self, features, targets):
        calls.append(1)
        if len(calls) == 4:
            # Native failures do not all close reads. Inject this separate
            # documented port failure explicitly instead of assuming closure.
            monkeypatch.setattr(candidates[0], "snapshot_state", unavailable)
            raise RuntimeError("engineering fourth storage call failure")
        return original(self, features, targets)

    monkeypatch.setattr(ActionMemoryLearner, "train_batch", train)
    with pytest.raises(RuntimeError, match="fourth storage"):
        shared._train_arm(
            row,
            packets,
            Sampler(),
            monotonic() + 180,
            lambda: None,
            learner_factory=pilot._learner,
            checkpoint_updates=(0, 32),
        )
    assert row["charged_updates"] == 3 and row["wake_calls_attempted"] == 4
    assert row["wake_calls_returned"] == 3
    assert row["partial_stream"] is not None
    partial = row["partial_stream"]
    assert len(partial["applied_updates"]) == 3
    assert partial["committed_training_examples"] == 27
    assert partial["native_candidate_training_examples"] is None
    assert partial["current_native_state_unavailable"] is True


def test_should_refuse_source_mutation_during_terminal_candidate_readback(case, monkeypatch):
    row, packets = small_arm(case)
    arm = shared._train_arm(
        row,
        packets,
        Sampler(),
        monotonic() + 180,
        lambda: None,
        learner_factory=pilot._learner,
        checkpoint_updates=(0, 32),
    )
    candidate = arm["stream"]._runtime._candidate
    original = candidate.snapshot_state

    def snapshot():
        observed = original()
        # Keep the source's native state valid: the guard must refuse an
        # unauthorized update, independently of a malformed-policy port error.
        arm["source"].train_batch(packets[0].features, packets[0].targets)
        return observed

    monkeypatch.setattr(candidate, "snapshot_state", snapshot)
    with pytest.raises(ValueError, match="source|actor"):
        arm["guard"]()


def test_should_preserve_dynamic_legacy_readback_comparator_and_original_record_shape(monkeypatch):
    initial = object()
    calls = []

    def compare(left, right):
        calls.append((left, right))
        return left is right

    monkeypatch.setattr(shared, "_same_state", compare)
    source = SimpleNamespace(snapshot_state=lambda: initial)
    candidate = SimpleNamespace(
        snapshot_state=lambda: SimpleNamespace(completed_updates=1, training_examples=9)
    )
    actor = SimpleNamespace(snapshot_state=lambda: SimpleNamespace(state=initial))
    stream = SimpleNamespace(
        completed_updates=1,
        _runtime=SimpleNamespace(_candidate=candidate, actor=actor, applied_updates=()),
    )
    row = shared._row("bp")
    shared._readback(row, source, initial, stream, SimpleNamespace(updates_completed=1))
    assert calls == [(initial, initial)] * 2
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert set(row["partial_stream"]) == {
        "completed_wake_updates",
        "native_candidate_updates",
        "native_candidate_training_examples",
        "applied_updates",
    }
