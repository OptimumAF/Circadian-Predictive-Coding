"""Recurring bound count control using discrete arrays and fake game accounting.

The count backend, delayed stream and evaluator are real. The synthetic teacher
episodes and two-step environments are engineering fixtures, not acquisition or
gameplay evidence. Each full composition spends128 storage updates/1152 rows.
"""

import builtins
from dataclasses import asdict, replace
import pickle
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_action_memory_pilot as pilot
from src.adapters import minigrid_learning_pilot as shared
from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters.action_memory_learner import ActionMemoryLearner
from src.adapters.invariant_action_memory_learner import (
    InvariantActionMemoryLearner,
    InvariantMemoryPolicy,
    copy_invariant_memory_snapshot,
)
from src.app import toy_execution_budget as budgets
from src.core.action_memory import ACTION_MEMORY_COUNT_DIAGNOSTIC
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from src.core.minigrid_invariant_encoding import INVARIANT_HISTORY_ENCODING_ID
from test_minigrid_invariant_evaluation import DiscreteEnvironment, public_frame
from test_minigrid_learning_pilot import Environment, POINTS
from test_torch_feature_rehearsal_pilot import Sampler, forbidden


def source_batch(context, cue):
    color, rotation = (4, 2) if context == "B" else (5, 0)
    frames = tuple(
        public_frame(cue, step, wall_color=color, rotation=rotation) for step in range(10)
    )
    episode = DemonstrationEpisode(
        f"source-{context}-{cue}",
        context,
        "train",
        tuple(
            DemonstrationTransition(
                step, frames[step], 1, 0.5 if step == 8 else 0.0, frames[step + 1], step == 8, 7
            )
            for step in range(9)
        ),
        7,
    )
    return shared._batch(episode)


class RecurringEnvironment(DiscreteEnvironment):
    def _frame(self, step):
        # The recurring TRAIN triple includes blue walls; the A/E fake only
        # covered grey/yellow. These remain discrete accounting frames.
        wall_color = {"grey": 5, "yellow": 4, "blue": 2, "red": 0}[self.layout.wall_color]
        return public_frame(self.cue, step, wall_color=wall_color, rotation=self.layout.rotation)


@pytest.fixture
def case(monkeypatch):
    sources = tuple(
        source_batch(context, cue) for context in ("A", "B", "C_train") for cue in ("blue", "red")
    )
    work: dict[str, Any] = dict(trains=[], predicts=[], readouts=[], arms=[], sources=[])
    monkeypatch.setattr(Environment, "work", work)
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", RecurringEnvironment)
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(budgets, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(shared, "require_torch", forbidden)
    monkeypatch.setattr(shared, "_same_state", forbidden)
    monkeypatch.setattr(evaluation, "_same_state", forbidden)
    old_import = builtins.__import__

    def no_torch(name, *args, **options):
        if name == "torch" or name.startswith("torch."):
            pytest.fail("recurring count control imported Torch")
        return old_import(name, *args, **options)

    monkeypatch.setattr(builtins, "__import__", no_torch)

    def collect(check, reports):
        check()
        reports.extend(dict(status="engineering_fixture", environment_steps=9) for _ in sources)
        return sources

    monkeypatch.setattr(shared, "_collect_sources", collect)
    row = shared._row(pilot.INVARIANT_METHOD)
    monkeypatch.setattr(shared, "_row", lambda method: row)
    native_train, native_predict = (
        InvariantActionMemoryLearner.train_batch,
        InvariantActionMemoryLearner.predict,
    )
    old_train_arm, old_factory = shared._train_arm, pilot._invariant_learner
    old_evaluate = pilot.evaluate_invariant_memory_checkpoint

    def train(owner, features, targets):
        work["trains"].append((owner, features.copy(), targets.copy()))
        if callback := work.get("before_train"):
            callback(owner)
        result = native_train(owner, features, targets)
        return result

    def predict(owner, features):
        result = native_predict(owner, features)
        work["predicts"].append((owner, features.copy(), result.copy()))
        return result

    def factory(method):
        learner = old_factory(method)
        work["sources"].append(learner)
        return learner

    def train_arm(*args, **options):
        arm = old_train_arm(*args, **options)
        work["arms"].append(arm)
        return arm

    def evaluate(*args, **options):
        assert len(work["trains"]) == 128
        assert "method" not in options and "public_history" not in options
        work["readouts"].append((args, options))
        if callback := work.get("before_readout"):
            callback(options["updates"])
        result = old_evaluate(*args, **options)
        if callback := work.get("after_readout"):
            callback(result)
        return result

    monkeypatch.setattr(InvariantActionMemoryLearner, "train_batch", train)
    monkeypatch.setattr(InvariantActionMemoryLearner, "predict", predict)
    monkeypatch.setattr(pilot, "_invariant_learner", factory)
    monkeypatch.setattr(shared, "_train_arm", train_arm)
    monkeypatch.setattr(pilot, "evaluate_invariant_memory_checkpoint", evaluate)
    return SimpleNamespace(sources=sources, row=row, work=work)


def test_should_run_original_recurring_stream_with_bound_native_states_and_no_readout_learning(
    case,
):
    before = pickle.dumps(case.sources)
    report = pilot.run_pilot(invariant_memory=True)
    assert report["status"] == "complete", report["reason"]
    row = report["methods"][0]
    assert row["method"] == pilot.INVARIANT_METHOD
    assert row["charged_updates"] == 128 and row["native_training_examples"] == 1152
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert row["stream_result"]["evaluation_scope"] == "training_only"
    updates = row["stream_result"]["applied_updates"]
    assert len(updates) == 128
    assert all(
        update["diagnostic"]["definition"] == ACTION_MEMORY_COUNT_DIAGNOSTIC for update in updates
    )
    assert [options["updates"] for _, options in case.work["readouts"]] == list(POINTS)
    assert all(options["role"] == "train" for _, options in case.work["readouts"])
    assert [x[1].shape for x in case.work["trains"]] == [(9, 296)] * 128
    packets = shared._packets(case.sources)
    assert [packet.context for packet in packets] == ["A"] * 32 + ["B"] * 32 + ["C_train"] * 32 + [
        "A"
    ] * 32
    for (_, actual_x, actual_y), expected in zip(
        case.work["trains"], pilot._history_packets(packets), strict=True
    ):
        np.testing.assert_array_equal(actual_x, expected.features)
        np.testing.assert_array_equal(actual_y, expected.targets)
    native_budget = case.work["arms"][0]["budget"]
    assert native_budget.budget.max_training_updates == native_budget.updates_completed == 128
    assert all(options["budget"] is native_budget for _, options in case.work["readouts"])
    assert len(case.work["predicts"]) == 90  # Thirty nine-row scores and sixty live predictions.
    assert sum(len(features) for _, features, _ in case.work["predicts"]) == 330
    assert len(case.work["environments"]) == 30
    assert all(env.role == "train" and env.closed for env in case.work["environments"])
    assert all(readout["status"] == "complete" for readout in row["readouts"])
    for readout in row["readouts"]:
        costs = readout["work"]
        assert costs["internal_projection_rows_returned"] == 66
        assert costs["internal_projection_array_bytes_returned"] == 66 * 2368
    assert row["work"]["raw_score_forwards_completed"] == 30
    assert row["work"]["raw_score_rows_completed"] == 270
    assert row["work"]["live_prediction_calls_returned"] == 60
    assert row["final_evaluation"]["status"] == "not_applicable"
    assert report["scientific_advantage_verified"] is False
    assert pickle.dumps(case.sources) == before
    assert case.work["sources"][0].snapshot_state().completed_updates == 0
    assert all(type(value) is bytes for value in case.row["readouts"])
    states = case.work["arms"][0]["states"]
    for observation, (point, state) in zip(report["observations"], states, strict=True):
        assert observation["updates"] == point
        assert observation["native_training_examples"] == 9 * point
        assert observation["snapshot"] == asdict(copy_invariant_memory_snapshot(state))
        assert observation["snapshot"]["policy"]["encoding_id"] == INVARIANT_HISTORY_ENCODING_ID
        assert "entries" not in observation["snapshot"]
        assert observation["entries"] == len(state.native_state.entries)
        assert observation["logical_payload_bytes"] == 2424 * observation["entries"] <= 155136
    report["observations"][-1]["snapshot"]["native_state"]["entries"] = []
    assert asdict(states[-1][1])["native_state"]["entries"]


@pytest.mark.parametrize("fault", ["prior_equal_bytes", "source_array", "budget", "final_slot"])
def test_should_refuse_changed_paid_or_original_evidence_before_next_readout(case, fault):
    def corrupt(point):
        if point != 32:
            return
        if fault == "prior_equal_bytes":
            case.row["readouts"][0] = bytes(bytearray(case.row["readouts"][0]))
        elif fault == "source_array":
            case.sources[0].features[0, 0] = 0.5
        elif fault == "budget":
            case.work["arms"][0]["budget"].updates_completed = 127
        else:
            case.row["final_evaluation"]["status"] = "complete"

    case.work["before_readout"] = corrupt
    report = pilot.run_pilot(invariant_memory=True)
    assert report["status"] == "incomplete" and report["reason"]
    row = report["methods"][0]
    assert row["charged_updates"] == 128 and row["native_training_examples"] == 1152
    assert row["readouts"][0]["status"] == "complete"
    assert [options["updates"] for _, options in case.work["readouts"]] == [0, 32]
    assert all(readout["status"] == "not_run" for readout in row["readouts"][2:])
    assert len(case.work["environments"]) == 6
    assert len(case.work["trains"]) == 128
    assert row["final_evaluation"]["status"] == "not_applicable"


def test_should_keep_paid_prediction_prefix_and_all_training_after_live_step_failure(case):
    def fail(point):
        if point == 32:
            case.work["step_error"] = True

    case.work["before_readout"] = fail
    report = pilot.run_pilot(invariant_memory=True)
    assert report["status"] == "incomplete"
    row = report["methods"][0]
    assert row["charged_updates"] == 128 and row["native_training_examples"] == 1152
    failed = row["readouts"][1]
    assert failed["status"] == "failed"
    assert failed["work"]["live_prediction_calls_returned"] == 1
    assert failed["work"]["environment_steps_attempted"] == 1
    assert failed["work"]["environment_steps_returned"] == 0
    assert failed["work"]["failure_work_may_be_unobserved"] is True
    assert all(readout["status"] == "not_run" for readout in row["readouts"][2:])
    assert len(case.work["trains"]) == 128


def test_should_keep_returned_paid_readout_before_refusing_closing_budget_mutation(case):
    def corrupt(result):
        if len(case.work["readouts"]) == 2:
            assert result["status"] == "complete"
            case.work["arms"][0]["budget"].updates_completed = 127

    case.work["after_readout"] = corrupt
    report = pilot.run_pilot(invariant_memory=True)
    assert report["status"] == "incomplete" and report["reason"]
    row = report["methods"][0]
    assert row["charged_updates"] == 128 and row["native_training_examples"] == 1152
    assert all(readout["status"] == "complete" for readout in row["readouts"][:2])
    assert all(readout["status"] == "not_run" for readout in row["readouts"][2:])
    assert row["readouts"][1]["work"]["live_prediction_calls_returned"] == 12
    assert len(case.work["environments"]) == 12 and len(case.work["trains"]) == 128


@pytest.mark.parametrize("fault", ["getter_error", "source_mutation"])
def test_should_refuse_final_resource_readback_fault_without_losing_all_paid_readouts(
    case, monkeypatch, fault
):
    triggered = []

    class ClosingSampler(Sampler):
        def snapshot(self):
            if len(case.work["readouts"]) == 5 and all(
                type(value) is bytes for value in case.row["readouts"]
            ):
                triggered.append(1)
                if fault == "getter_error":
                    raise RuntimeError("engineering final RSS readback unavailable")
                object.__setattr__(case.work["sources"][0]._policy, "encoding_id", "corrupt")
            return super().snapshot()

    monkeypatch.setattr(pilot, "ProcessRssSampler", ClosingSampler)
    monkeypatch.setattr(budgets, "ProcessRssSampler", ClosingSampler)
    report = pilot.run_pilot(invariant_memory=True)
    assert triggered and report["status"] == "incomplete" and report["reason"]
    row = report["methods"][0]
    assert row["charged_updates"] == 128 and row["native_training_examples"] == 1152
    assert all(readout["status"] == "complete" for readout in row["readouts"])
    assert row["work"]["raw_score_forwards_completed"] == 30
    assert row["work"]["live_prediction_calls_returned"] == 60
    assert len(case.work["trains"]) == 128


def test_should_preserve_committed_storage_prefix_when_fourth_update_fails(case):
    def fail(owner):
        if len(case.work["trains"]) == 4:
            raise RuntimeError("engineering fourth count update refusal")

    case.work["before_train"] = fail
    report = pilot.run_pilot(invariant_memory=True)
    assert report["status"] == "incomplete" and "fourth" in report["reason"]
    row = report["methods"][0]
    assert row["charged_updates"] == 3 and row["wake_calls_attempted"] == 4
    assert row["native_training_examples"] == 27
    assert len(row["partial_stream"]["applied_updates"]) == 3
    assert case.work["readouts"] == [] and case.work.get("environments", []) == []


@pytest.mark.parametrize("fault", ["raw", "capacity", "encoding"])
def test_should_refuse_wrong_factory_policy_before_native_updates(case, monkeypatch, fault):
    def wrong(method):
        if fault == "raw":
            return ActionMemoryLearner(pilot.memory_policy())
        policy = InvariantMemoryPolicy(
            replace(pilot.memory_policy(), max_payload_bytes=10752)
            if fault == "capacity"
            else pilot.memory_policy()
        )
        learner = InvariantActionMemoryLearner(policy)
        if fault == "encoding":
            object.__setattr__(learner._policy, "encoding_id", "not-the-bound-encoding")
        return learner

    monkeypatch.setattr(pilot, "_invariant_learner", wrong)
    report = pilot.run_pilot(invariant_memory=True)
    assert report["status"] == "incomplete" and report["reason"]
    assert case.work["trains"] == case.work["predicts"] == case.work["readouts"] == []
    assert report["methods"][0]["wake_calls_attempted"] == 0


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], np.bool_(True)])
def test_should_reject_nonboolean_invariant_option_before_ports(monkeypatch, value):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(shared, "_collect_sources", forbidden)
    monkeypatch.setattr(pilot, "_invariant_learner", forbidden)
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(invariant_memory=value)


def test_should_mark_lost_admitted_readout_unknown_and_keep_preceding_paid_work(case, monkeypatch):
    original = pilot.evaluate_invariant_memory_checkpoint

    def lost(*args, **options):
        result = original(*args, **options)
        if options["updates"] == 32:
            assert result["work"]["live_prediction_calls_returned"] == 12
            raise RuntimeError("engineering lost admitted readout receipt")
        return result

    monkeypatch.setattr(pilot, "evaluate_invariant_memory_checkpoint", lost)
    report = pilot.run_pilot(invariant_memory=True)
    assert report["status"] == "incomplete" and "lost admitted" in report["reason"]
    row = report["methods"][0]
    assert row["charged_updates"] == 128 and row["native_training_examples"] == 1152
    assert row["readouts"][0]["status"] == "complete"
    missing = row["readouts"][1]
    assert missing["status"] == "failed" and missing["evaluator_admitted"] is True
    assert missing["receipt_unavailable"] is True and missing["work_unknown"] is True
    assert all(r["status"] == "not_run" for r in row["readouts"][2:])
    assert row["work"]["failure_work_may_be_unobserved"] is True
    assert row["work"]["raw_score_forwards_completed"] is None
    assert row["work"]["live_prediction_calls_returned"] is None
    lower = row["work"]["known_readout_work_lower_bounds"]
    assert lower["raw_score_forwards_completed"] == 6
    assert lower["live_prediction_calls_returned"] == 12
    assert len(case.work["trains"]) == 128
