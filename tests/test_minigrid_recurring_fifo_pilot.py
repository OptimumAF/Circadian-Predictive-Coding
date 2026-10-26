"""Finite recurring-memory integration on context-distinct engineering arrays.

Real FIFO/count/stream/evaluator code runs against two-step accounting fakes.
Each task has eighteen distinct canonical keys; these fixtures establish neither
MiniGrid acquisition nor an algorithmic advantage.
"""

from dataclasses import asdict
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from src.adapters import minigrid_action_memory_pilot as pilot
from src.adapters import minigrid_learning_pilot as shared
from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters.invariant_action_memory_learner import (
    InvariantActionMemoryLearner,
    InvariantMemoryPolicy,
    InvariantMemorySnapshot,
    copy_invariant_memory_snapshot,
)
from src.app import toy_execution_budget as budgets
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from src.core.action_memory import ActionMemorySnapshot
from src.core.fifo_action_memory import (
    FIFO_ACTION_MEMORY_COUNT_DIAGNOSTIC,
    FifoActionMemoryPolicy,
    FifoActionMemorySnapshot,
)
from src.core.minigrid_invariant_encoding import canonicalize_public_history
from test_minigrid_invariant_encoding import change_pixel, frame
from test_minigrid_invariant_evaluation import DiscreteEnvironment
from test_minigrid_learning_pilot import POINTS
import test_minigrid_recurring_memory_pilot as recurring_memory
from test_torch_feature_rehearsal_pilot import Sampler, forbidden

base_case = recurring_memory.case


def context_frame(context, cue, step):
    """Only public nonwall pixels distinguish tasks and nine decision steps."""
    color = 2 if cue == "blue" else 0
    value = frame(wall_color=step % 6, direction=step % 4, key_color=color, cue_color=color)
    value = change_pixel(value, 8, (3, {"A": 0, "B": 1, "C_train": 2}[context], 0))
    return change_pixel(value, 9, (3, step % 6, step // 6))


def source_batch(context, cue):
    observations = tuple(context_frame(context, cue, step) for step in range(10))
    action = 1 if cue == "blue" else 2
    episode = DemonstrationEpisode(
        f"source-{context}-{cue}",
        context,
        "train",
        tuple(
            DemonstrationTransition(
                step,
                observations[step],
                action,
                0.5 if step == 8 else 0.0,
                observations[step + 1],
                step == 8,
                7,
            )
            for step in range(9)
        ),
        7,
    )
    return shared._batch(episode)


class ContextEnvironment(DiscreteEnvironment):
    def _frame(self, step):
        return context_frame(self.layout.context, self.cue, step)


@pytest.fixture
def case(base_case, monkeypatch):
    sources = tuple(
        source_batch(context, cue) for context in ("A", "B", "C_train") for cue in ("blue", "red")
    )
    work, row = base_case.work, base_case.row
    row["method"] = pilot.INVARIANT_FIFO_METHOD
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", ContextEnvironment)

    def collect(check, reports):
        check()
        reports.extend(dict(status="engineering_fixture", environment_steps=9) for _ in sources)
        return sources

    old_factory, old_evaluate = (
        pilot._invariant_fifo_learner,
        pilot.evaluate_invariant_fifo_checkpoint,
    )

    def factory(method):
        learner = old_factory(method)
        work["sources"].append(learner)
        return learner

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

    monkeypatch.setattr(shared, "_collect_sources", collect)
    monkeypatch.setattr(pilot, "_invariant_fifo_learner", factory)
    monkeypatch.setattr(pilot, "evaluate_invariant_fifo_checkpoint", evaluate)
    return SimpleNamespace(sources=sources, row=row, work=work)


def test_should_retire_old_task_counts_and_relearn_on_return_without_exceeding_fifo36(case):
    before = pickle.dumps(case.sources)
    packets = pilot._history_packets(shared._packets(case.sources))
    keys = {}
    for context in ("A", "B", "C_train"):
        keys[context] = {
            canonicalize_public_history(tuple(float(value) for value in features))
            for packet in packets
            if packet.context == context
            for features in packet.features
        }
    assert all(len(values) == 18 for values in keys.values())
    assert len(keys["A"] | keys["B"] | keys["C_train"]) == 54
    report = pilot.run_pilot(invariant_fifo_memory=True)
    assert report["status"] == "complete", report["reason"]
    row = report["methods"][0]
    assert row["method"] == pilot.INVARIANT_FIFO_METHOD
    assert row["charged_updates"] == 128 and row["native_training_examples"] == 1152
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert len(case.work["trains"]) == 128 and len(case.work["predicts"]) == 90
    for (_, actual_x, actual_y), expected in zip(case.work["trains"], packets, strict=True):
        np.testing.assert_array_equal(actual_x, expected.features)
        np.testing.assert_array_equal(actual_y, expected.targets)
    assert [options["updates"] for _, options in case.work["readouts"]] == list(POINTS)
    original_budget = case.work["arms"][0]["budget"]
    assert original_budget.updates_completed == original_budget.budget.max_training_updates == 128
    assert all(options["budget"] is original_budget for _, options in case.work["readouts"])
    assert all(
        receipt["diagnostic"]["definition"] == FIFO_ACTION_MEMORY_COUNT_DIAGNOSTIC
        for receipt in row["stream_result"]["applied_updates"]
    )
    observations = report["observations"]
    assert [o["entries"] for o in observations] == [0, 18, 36, 36, 36]
    assert [o["evictions"] for o in observations] == [0, 0, 0, 18, 36]
    assert [o["retired_examples"] for o in observations] == [0, 0, 0, 288, 576]
    for observation, (point, state) in zip(
        observations, case.work["arms"][0]["states"], strict=True
    ):
        native = copy_invariant_memory_snapshot(state).native_state
        assert type(native) is FifoActionMemorySnapshot
        assert observation["snapshot"] == asdict(state)
        assert observation["retirement_metadata_bytes"] == 16
        assert observation["logical_payload_bytes"] == 2424 * len(native.entries) <= 87264
        assert (
            native.retired_examples + sum(sum(entry.counts) for entry in native.entries)
            == 9 * point
        )
    fit = {
        (readout["updates"], score["context"]): score["exact_sequence_accuracy"]
        for readout in row["readouts"]
        for score in readout["scores"]
    }
    assert fit[32, "A"] == fit[64, "B"] == fit[96, "C_train"] == fit[128, "A"] == 1.0
    assert fit[96, "A"] == fit[128, "B"] == 0.0
    assert row["work"]["evictions"] == 36 and row["work"]["retired_examples"] == 576
    assert row["work"]["maximum_retained_entries"] == 36
    assert row["work"]["maximum_retained_logical_payload_bytes"] == 87264
    assert row["work"]["retirement_metadata_bytes"] == 16
    assert row["work"]["raw_score_forwards_completed"] == 30
    assert row["work"]["live_prediction_calls_returned"] == 60
    assert len(case.work["environments"]) == 30
    assert all(env.role == "train" and env.closed for env in case.work["environments"])
    assert case.work["sources"][0].snapshot_state().completed_updates == 0
    assert pickle.dumps(case.sources) == before and report["scientific_advantage_verified"] is False
    assert row["final_evaluation"]["status"] == "not_applicable"


@pytest.mark.parametrize("fault", ["source", "counter"])
def test_should_refuse_changed_preceding_full_state_before_next_game_cohort(case, fault):
    def corrupt(point):
        if point == 32:
            if fault == "counter":
                case.work["arms"][0]["budget"].updates_completed = 127
            else:
                object.__setattr__(case.work["sources"][0]._policy, "encoding_id", "corrupt")

    case.work["before_readout"] = corrupt
    report = pilot.run_pilot(invariant_fifo_memory=True)
    assert report["status"] == "incomplete" and report["reason"]
    row = report["methods"][0]
    assert row["charged_updates"] == 128 and row["native_training_examples"] == 1152
    assert row["readouts"][0]["status"] == "complete"
    assert all(readout["status"] == "not_run" for readout in row["readouts"][2:])
    assert len(case.work["environments"]) == 6 and len(case.work["trains"]) == 128


def test_should_keep_returned_paid_report_when_closing_proof_refuses(case):
    def corrupt(result):
        if len(case.work["readouts"]) == 2:
            case.row["readouts"][0] = bytes(bytearray(case.row["readouts"][0]))

    case.work["after_readout"] = corrupt
    report = pilot.run_pilot(invariant_fifo_memory=True)
    assert report["status"] == "incomplete"
    row = report["methods"][0]
    assert row["charged_updates"] == 128
    assert all(readout["status"] == "complete" for readout in row["readouts"][:2])
    assert row["readouts"][1]["work"]["live_prediction_calls_returned"] == 12
    assert len(case.work["trains"]) == 128 and len(case.work["environments"]) == 12


@pytest.mark.parametrize("readback_available", [True, False])
def test_should_preserve_missing_receipt_native_work_without_inventing_a_commit(
    case, monkeypatch, readback_available
):
    train = InvariantActionMemoryLearner.train_batch

    def lose_return(owner, features, targets):
        result = train(owner, features, targets)
        if len(case.work["trains"]) == 4:
            if not readback_available:

                def unavailable():
                    raise RuntimeError("engineering FIFO native readback unavailable")

                monkeypatch.setattr(owner, "snapshot_state", unavailable)
            raise RuntimeError("engineering lost FIFO update receipt")
        return result

    monkeypatch.setattr(InvariantActionMemoryLearner, "train_batch", lose_return)
    report = pilot.run_pilot(invariant_fifo_memory=True)
    assert report["status"] == "incomplete" and "lost FIFO" in report["reason"]
    row = report["methods"][0]
    assert row["wake_calls_attempted"] == 4 and row["wake_calls_returned"] == 3
    assert row["charged_updates"] == 3
    assert row["native_training_examples"] == (36 if readback_available else None)
    assert len(row["partial_stream"]["applied_updates"]) == 3
    assert row["partial_stream"]["committed_training_examples"] == 27
    assert row["work"]["native_training_work_may_be_unobserved"] is True
    assert row["work"]["count_update_calls"] == (4 if readback_available else None)
    assert row["work"]["charged_count_updates"] == 3
    assert row["work"]["committed_training_projection_rows"] == 27
    assert row["work"]["observed_native_training_projection_rows"] == (
        36 if readback_available else None
    )
    assert case.work["readouts"] == [] and case.work.get("environments", []) == []


def test_should_retain_all_paid_work_after_final_resource_getter_failure(case, monkeypatch):
    class ClosingSampler(Sampler):
        def snapshot(self):
            if len(case.work["readouts"]) == 5 and all(
                type(value) is bytes for value in case.row["readouts"]
            ):
                raise RuntimeError("engineering closing FIFO RSS failure")
            return super().snapshot()

    monkeypatch.setattr(pilot, "ProcessRssSampler", ClosingSampler)
    monkeypatch.setattr(budgets, "ProcessRssSampler", ClosingSampler)
    report = pilot.run_pilot(invariant_fifo_memory=True)
    assert report["status"] == "incomplete" and report["reason"]
    row = report["methods"][0]
    assert row["charged_updates"] == 128 and row["native_training_examples"] == 1152
    assert all(readout["status"] == "complete" for readout in row["readouts"])
    assert row["work"]["raw_score_forwards_completed"] == 30
    assert row["work"]["live_prediction_calls_returned"] == 60


@pytest.mark.parametrize("fault", ["refusal", "fifo64", "fifo_wrong_bytes"])
def test_should_refuse_other_storage_presets_before_updates(case, monkeypatch, fault):
    def wrong(method):
        policy = (
            pilot.memory_policy()
            if fault == "refusal"
            else FifoActionMemoryPolicy(
                feature_dim=296,
                num_classes=7,
                max_entries=64 if fault == "fifo64" else 36,
                max_payload_bytes=155136,
            )
        )
        return InvariantActionMemoryLearner(InvariantMemoryPolicy(policy))

    monkeypatch.setattr(pilot, "_invariant_fifo_learner", wrong)
    report = pilot.run_pilot(invariant_fifo_memory=True)
    assert report["status"] == "incomplete" and report["reason"]
    assert case.work["trains"] == case.work["predicts"] == case.work["readouts"] == []


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], np.bool_(True)])
def test_should_reject_nonboolean_fifo_mode_before_ports(monkeypatch, value):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(shared, "_collect_sources", forbidden)
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(invariant_fifo_memory=value)


def test_should_admit_fifo36_counter_only_through_explicit_narrow_preset():
    policy = pilot.fifo_memory_policy()
    state = InvariantMemorySnapshot(
        InvariantMemoryPolicy(policy), FifoActionMemorySnapshot(policy, (), 0, 0, 0, 0)
    )
    with pytest.raises(ValueError):
        shared._invariant_memory_counter(state, 0, pilot.INVARIANT_FIFO_METHOD)
    shared._invariant_memory_counter(state, 0, pilot.INVARIANT_FIFO_METHOD, fifo36=True)
    refusal = pilot.memory_policy()
    fifo64 = FifoActionMemoryPolicy(feature_dim=296, num_classes=7, max_payload_bytes=155136)
    wrong_states = (
        InvariantMemorySnapshot(
            InvariantMemoryPolicy(refusal), ActionMemorySnapshot(refusal, (), 0, 0)
        ),
        InvariantMemorySnapshot(
            InvariantMemoryPolicy(fifo64), FifoActionMemorySnapshot(fifo64, (), 0, 0, 0, 0)
        ),
    )
    for wrong in wrong_states:
        with pytest.raises(ValueError):
            shared._invariant_memory_counter(wrong, 0, wrong.policy.method, fifo36=True)
