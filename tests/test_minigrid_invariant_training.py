"""Bound invariant count controls through the existing delayed MiniGrid TRAIN arm.

All nine-step episodes are discrete engineering arrays. No environment, teacher,
neural optimizer or scientific acquisition result is involved.
"""

import builtins
from copy import deepcopy
from dataclasses import replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_action_memory_pilot as memory
from src.adapters import minigrid_learning_pilot as shared
from src.adapters.action_memory_learner import ActionMemoryLearner
from src.adapters.invariant_action_memory_learner import (
    InvariantActionMemoryLearner,
    InvariantMemoryPolicy,
)
from src.app import toy_execution_budget as budgets
from src.core.action_memory import (
    ACTION_MEMORY_COUNT_DIAGNOSTIC,
    ActionMemoryEntry,
    ActionMemoryPolicy,
)
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from src.core.fifo_action_memory import FIFO_ACTION_MEMORY_COUNT_DIAGNOSTIC, FifoActionMemoryPolicy
from src.core.minigrid_invariant_encoding import (
    INVARIANT_HISTORY_ENCODING_ID,
    canonicalize_public_history,
)
from test_minigrid_invariant_encoding import frame
from test_torch_feature_rehearsal_pilot import Sampler, forbidden


METHODS = ("public_invariant_observation_memory", "public_invariant_observation_memory_fifo")


def native_policy(method):
    kind = FifoActionMemoryPolicy if method.endswith("_fifo") else ActionMemoryPolicy
    return kind(feature_dim=296, num_classes=7, max_payload_bytes=155136)


def training_packets():
    packets = []
    for index in range(32):
        color = 2 if index % 2 == 0 else 0
        frames = tuple(
            frame(
                wall_color=step % 6,
                direction=step % 4,
                key_color=color,
                cue_color=color,
                door_state=2 if step < 5 else 0,
            )
            for step in range(10)
        )
        episode = DemonstrationEpisode(
            f"minigrid-training-{index}",
            "A",
            "train",
            tuple(
                DemonstrationTransition(
                    step,
                    frames[step],
                    step % 7,
                    0.5 if step == 8 else 0.0,
                    frames[step + 1],
                    step == 8,
                    7,
                )
                for step in range(9)
            ),
            7,
        )
        packets.append(shared._batch(episode))
    return memory._history_packets(tuple(packets))


@pytest.fixture
def case(monkeypatch):
    work: dict[str, Any] = dict(
        factory_calls=[],
        sources=[],
        streams=[],
        budgets=[],
        attempts=[],
        trains=[],
        after_train=None,
        failure=None,
    )
    original_train = InvariantActionMemoryLearner.train_batch
    original_stream = shared.AttributionStream
    original_budget = shared.ToyBudgetSession
    original_import = builtins.__import__
    packets = training_packets()

    def no_torch(name: str, *args: Any, **options: Any) -> Any:
        if name == "torch" or name.startswith("torch."):
            pytest.fail("invariant TRAIN arm imported Torch")
        return original_import(name, *args, **options)

    def factory(method):
        work["factory_calls"].append(method)
        source = InvariantActionMemoryLearner(InvariantMemoryPolicy(native_policy(method)))
        work["sources"].append(source)
        return source

    def stream(*args, **options):
        assert options["training_only"] is True and args[2] == ()
        assert options["label_delay"] == 1 and options["checkpoint_updates"] == (0, 32)
        value = original_stream(*args, **options)
        work["streams"].append(value)
        return value

    def budget(*args, **options):
        value = original_budget(*args, **options)
        work["budgets"].append(value)
        return value

    def train(owner, features, targets):
        work["attempts"].append((owner, features.copy(), targets.copy()))
        callback = work["failure"]
        if callback is not None:
            callback(owner)
        result = original_train(owner, features, targets)
        work["trains"].append((owner, features.copy(), targets.copy(), owner.snapshot_state()))
        callback = work["after_train"]
        if callback is not None:
            callback(owner)
        return result

    monkeypatch.setattr(shared, "monotonic", lambda: 1.0)
    monkeypatch.setattr(budgets, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(shared, "ToyBudgetSession", budget)
    monkeypatch.setattr(shared, "AttributionStream", stream)
    monkeypatch.setattr(shared, "require_torch", forbidden)
    monkeypatch.setattr(shared, "_same_state", forbidden)
    monkeypatch.setattr(shared, "_learner", forbidden)
    monkeypatch.setattr(InvariantActionMemoryLearner, "train_batch", train)
    monkeypatch.setattr(InvariantActionMemoryLearner, "predict", forbidden)
    monkeypatch.setattr(builtins, "__import__", no_torch)
    work.update(packets=packets, factory=factory)
    return work


def train(case, method=METHODS[0], *, row=None, factory=None, boundary=None):
    return shared._train_arm(
        shared._row(method) if row is None else row,
        case["packets"],
        Sampler(),
        100.0,
        (lambda: None) if boundary is None else boundary,
        learner_factory=case["factory"] if factory is None else factory,
        checkpoint_updates=(0, 32),
    )


@pytest.mark.parametrize("method", METHODS)
def test_should_train_both_native_controls_through_the_original_delayed_stream(case, method):
    before = pickle.dumps(case["packets"])
    arm = train(case, method)
    arm["guard"]()
    row, source, stream, budget = arm["row"], arm["source"], arm["stream"], arm["budget"]
    assert row["status"] == "trained" and row["method"] == method
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert row["charged_updates"] == row["wake_calls_attempted"] == row["wake_calls_returned"] == 32
    assert row["native_training_examples"] == 288
    assert budget.budget.max_training_updates == budget.updates_completed == 32
    assert budget.started_at == 1.0 and len(case["budgets"]) == len(case["streams"]) == 1
    assert source.completed_updates == source.training_examples == 0
    assert stream._runtime.actor.snapshot_state().state == source.snapshot_state()
    assert [point for point, _ in arm["states"]] == [0, 32]
    initial, final = [state for _, state in arm["states"]]
    assert initial.native_state.entries == ()
    assert (
        final.policy.method == method and final.policy.encoding_id == INVARIANT_HISTORY_ENCODING_ID
    )
    assert (final.completed_updates, final.training_examples) == (32, 288)
    assert sum(sum(entry.counts) for entry in final.native_state.entries) == 288
    assert all(
        canonicalize_public_history(entry.features) == entry.features
        for entry in final.native_state.entries
    )
    if method.endswith("_fifo"):
        assert final.native_state.evictions == final.native_state.retired_examples == 0
    result = row["stream_result"]
    assert result["evaluation_scope"] == "training_only" and result["checkpoints"] == []
    assert result["score_forward_calls"] == 0 and result["training_examples"] == 288
    receipts = result["applied_updates"]
    assert len(receipts) == 32
    assert [r["observed_at"] for r in receipts] == list(range(32))
    assert [r["arrived_at"] for r in receipts] == [i + 1 for i in range(32)]
    assert [r["applied_at"] for r in receipts] == [i + 1 for i in range(32)]
    diagnostic = (
        FIFO_ACTION_MEMORY_COUNT_DIAGNOSTIC
        if method.endswith("_fifo")
        else ACTION_MEMORY_COUNT_DIAGNOSTIC
    )
    assert all(r["diagnostic"] == {"definition": diagnostic, "value": 9.0} for r in receipts)
    assert len(case["trains"]) == 32 and case["factory_calls"] == [method]
    for observed, packet in zip(case["trains"], case["packets"], strict=True):
        np.testing.assert_array_equal(observed[1], packet.features)
        np.testing.assert_array_equal(observed[2], packet.targets)
        assert observed[1].shape == (9, 296) and packet.episode is not None
        assert packet.role == packet.episode.role == "train" and packet.context == "A"
    assert pickle.dumps(case["packets"]) == before


@pytest.mark.parametrize("fault", ["raw", "wrong_kind", "cap", "encoding"])
def test_should_refuse_wrong_factory_native_state_or_policy_before_training_or_budget(case, fault):
    def factory(method):
        case["factory_calls"].append(method)
        if fault == "raw":
            return ActionMemoryLearner(native_policy(method))
        if fault == "wrong_kind":
            return InvariantActionMemoryLearner(InvariantMemoryPolicy(native_policy(METHODS[1])))
        if fault == "cap":
            return InvariantActionMemoryLearner(
                InvariantMemoryPolicy(
                    ActionMemoryPolicy(
                        feature_dim=296, num_classes=7, max_entries=32, max_payload_bytes=155136
                    )
                )
            )
        value = InvariantActionMemoryLearner()
        object.__setattr__(value._policy, "encoding_id", "not-the-bound-encoding")
        return value

    row = shared._row(METHODS[0])
    with pytest.raises(ValueError):
        train(case, row=row, factory=factory)
    assert row["status"] == "incomplete" and not case["trains"]
    assert not case["streams"] and not case["budgets"]


def test_should_require_an_explicit_factory_and_preserve_the_existing_paired_only_flag(case):
    row = shared._row(METHODS[0])
    with pytest.raises(ValueError, match="factory"):
        shared._train_arm(
            row, case["packets"], Sampler(), 100.0, lambda: None, checkpoint_updates=(0, 32)
        )
    with pytest.raises(ValueError, match="paired"):
        shared._train_arm(
            row,
            case["packets"],
            Sampler(),
            100.0,
            lambda: None,
            learner_factory=case["factory"],
            checkpoint_updates=(0, 32),
            public_history=True,
        )
    assert not case["factory_calls"] and not case["budgets"]


def test_should_stop_after_a_paid_update_when_the_preceding_boundary_is_corrupted(case):
    proof = {"accepted": True}

    def boundary():
        if not proof["accepted"]:
            raise ValueError("preceding accepted evidence changed")

    def corrupt(owner):
        proof["accepted"] = False

    case["after_train"] = corrupt
    row = shared._row(METHODS[0])
    with pytest.raises(ValueError, match="preceding"):
        train(case, row=row, boundary=boundary)
    assert row["status"] == "incomplete" and row["charged_updates"] == 1
    assert row["native_training_examples"] == 9 and len(case["trains"]) == 1
    assert len(row["partial_stream"]["applied_updates"]) == 1


@pytest.mark.parametrize(
    "target", ["receipt", "equal_state_replacement", "partial_report", "budget", "candidate"]
)
def test_should_refuse_changed_or_replaced_completed_training_evidence(case, target):
    arm = train(case)
    if target == "receipt":
        arm["row"]["stream_result"]["applied_updates"][0]["diagnostic"]["value"] = 0.0
    elif target == "equal_state_replacement":
        point, state = arm["states"][0]
        arm["states"][0] = (point, deepcopy(state))
    elif target == "partial_report":
        arm["row"]["partial_stream"]["committed_training_examples"] = 0
    elif target == "budget":
        arm["budget"].updates_completed = 0
    else:
        owner = arm["stream"]._runtime._candidate
        state = owner.snapshot_state()
        entry = state.native_state.entries[0]
        counts = tuple(entry.counts[(i + 1) % 7] for i in range(7))
        changed = replace(
            state.native_state,
            entries=(ActionMemoryEntry(entry.features, counts), *state.native_state.entries[1:]),
        )
        owner.restore_state(replace(state, native_state=changed))
    with pytest.raises(ValueError):
        arm["guard"]()


def test_should_refuse_source_change_during_late_candidate_readback(case, monkeypatch):
    arm = train(case)
    candidate = arm["stream"]._runtime._candidate
    original = candidate.snapshot_state

    def snapshot():
        observed = original()
        arm["source"].restore_state(arm["states"][-1][1])
        return observed

    monkeypatch.setattr(candidate, "snapshot_state", snapshot)
    with pytest.raises(ValueError, match="source|actor"):
        arm["guard"]()
    assert arm["budget"].updates_completed == 32


def test_should_preserve_original_paid_row_when_the_final_readback_corrupts_it(case, monkeypatch):
    original = shared._readback
    calls = []

    def readback(row, *args, **options):
        original(row, *args, **options)
        calls.append(row)
        if len(calls) == 2:
            row["charged_updates"] = 0
            row["stream_result"]["training_examples"] = 0
            row["partial_stream"]["applied_updates"] = []

    monkeypatch.setattr(shared, "_readback", readback)
    row = shared._row(METHODS[0])
    with pytest.raises(ValueError):
        train(case, row=row)
    assert row["status"] == "incomplete" and "readback_error" in row
    assert row["charged_updates"] == 32 and row["native_training_examples"] == 288
    assert row["stream_result"]["training_examples"] == 288
    assert len(row["partial_stream"]["applied_updates"]) == 32
    assert row["partial_stream"]["committed_training_examples"] == 288


def test_should_keep_receipts_when_an_unsuccessful_native_call_closes_candidate_reads(
    case, monkeypatch
):
    def unavailable():
        raise RuntimeError("closed invariant candidate readback")

    def fail(owner):
        if len(case["attempts"]) == 3:
            candidate = case["streams"][0]._runtime._candidate
            monkeypatch.setattr(candidate, "snapshot_state", unavailable)
            raise RuntimeError("third invariant storage call failed")

    case["failure"] = fail
    row = shared._row(METHODS[0])
    with pytest.raises(RuntimeError, match="third invariant"):
        train(case, row=row)
    assert row["status"] == "incomplete" and row["charged_updates"] == 2
    assert row["wake_calls_attempted"] == 3 and row["wake_calls_returned"] == 2
    partial = row["partial_stream"]
    assert len(partial["applied_updates"]) == 2 and partial["committed_training_examples"] == 18
    assert partial["native_candidate_training_examples"] is None
    assert partial["current_native_state_unavailable"] is True
