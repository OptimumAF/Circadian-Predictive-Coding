"""Original public native methods replay fixed old-A TRAIN episodes in one stream.

Four synthetic cases:24 native updates/216rows total, no score/game/probe calls.
Other cases reject input before constructing or updating a native learner.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import pickle

import numpy as np
import pytest

from src.adapters import minigrid_fixed_rehearsal as adapter
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeaturePolicy,
)
from src.adapters.minigrid_learnability import _packet
from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy
from src.app.coupled_attribution import AttributionStream
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.minigrid_invariant_encoding import canonicalize_public_history
from src.core.minigrid_public_history import encode_public_history_episode
from src.shared.torch_runtime import require_torch
from test_minigrid_invariant_learnability import episode, train_sources
from test_torch_feature_learner import assert_tree


@pytest.fixture(scope="module")
def native_work():
    torch = require_torch()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    patch = pytest.MonkeyPatch()
    original = TorchFeatureLearner.train_batch
    work = dict(calls=0, rows=0)

    def train(owner, features, targets):
        work["calls"] += 1
        work["rows"] += len(features)
        assert work["calls"] <= 32 and work["rows"] <= 300
        return original(owner, features, targets)

    def forbidden(*args, **kwargs):
        pytest.fail("training-only rehearsal attempted scoring")

    patch.setattr(TorchFeatureLearner, "train_batch", train)
    patch.setattr(InvariantTorchFeatureLearner, "predict", forbidden)
    yield torch, work
    patch.undo()


def budgets():
    clock = lambda: 1.0
    return (
        ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), clock),
        ToyBudgetSession(ToyExecutionBudget(max_replay_examples=36), clock),
    )


@pytest.mark.parametrize("method", ["bp", "pc", "neutral_cpc", "gated_cpc"])
def test_should_replay_complete_old_a_bank_with_each_original_native_method(native_work, method):
    torch, work = native_work
    before_calls, before_rows = work["calls"], work["rows"]
    rng = torch.get_rng_state().clone()
    numpy_rng = pickle.dumps(np.random.get_state())
    learner = InvariantTorchFeatureLearner(
        InvariantTorchFeaturePolicy(TorchFeaturePolicy(method, 296, 8, 17, num_classes=7))
    )
    original = learner.snapshot_state()
    bank = tuple(packet.episode for packet in train_sources()[:2])
    # These are newly constructed TRAIN fixtures, not reserved probe data relabeled.
    training = tuple(
        _packet(encode_public_history_episode(episode(f"E-{cue}", cue, "train", f"new-{cue}")))
        for cue in ("blue", "red")
    )
    inputs_before = pickle.dumps((bank, training))
    wake, replay = budgets()
    authority = (wake.budget, replay.budget, wake.started_at, replay.started_at)
    plan = adapter.build_minigrid_old_a_rehearsal(
        learner, wake, bank, replay_budget=replay, total_updates=2
    )
    helper = plan.rehearsal
    assert (plan.validation_projection_rows, plan.validation_projected_array_equivalent_bytes) == (
        36,
        85248,
    )
    assert helper.retention.example_count == 18
    assert helper.retention.retained_bytes == adapter.BANK_ARRAY_BYTES == 87280
    assert (
        work["calls"] == before_calls
        and wake.updates_completed == replay.replay_examples_completed == 0
    )
    run = AttributionStream(
        learner,
        training,
        (),
        checkpoint_updates=(0, 1, 2),
        label_delay=1,
        budget=wake,
        rehearsal=helper,
        training_only=True,
    )
    run.run_until(1)
    preceding = deepcopy(helper.report["events"][0])
    pause = run.pause()
    run.resume(pause)
    run.run_until(2)
    report = run.finish()["rehearsal"]
    assert report["events"][0] == preceding
    assert report["control"] == "fixed_train_episodes_v1"
    assert report["fixed_bank"]["episode_ids"] == [item.episode_id for item in bank]
    assert report["fixed_bank"]["construction_updates"] == 0
    for event in report["events"]:
        assert event["status"] == "complete" and event["charged_exposures"] == 18
        assert (
            event["attempted_updates"]
            == event["returned_updates"]
            == event["committed_updates"]
            == 2
        )
        assert event["selected_episodes"] == [asdict(item) for item in bank]
    candidate = run._runtime._candidate.snapshot_state()
    assert candidate.completed_updates == 6 and candidate.training_examples == 54
    assert wake.updates_completed == 2 and wake.replay_examples_completed == 0
    assert replay.updates_completed == 0 and replay.replay_examples_completed == 36
    assert authority == (wake.budget, replay.budget, wake.started_at, replay.started_at)
    assert work["calls"] - before_calls == 6 and work["rows"] - before_rows == 54
    assert len(run._runtime.applied_consolidations) == 2
    assert_tree(torch, learner.snapshot_state(), original)
    assert_tree(torch, run._runtime.actor.snapshot_state().state, original)
    assert pickle.dumps((bank, training)) == inputs_before
    assert (
        torch.equal(torch.get_rng_state(), rng) and pickle.dumps(np.random.get_state()) == numpy_rng
    )
    # The fixed buffer stores original raw histories, rather than preprojected rows.
    assert (
        report["events"][0]["selected_episodes"][0]["transitions"][0]["observation"]
        == bank[0].transitions[0].observation
    )
    assert (
        canonicalize_public_history(bank[0].transitions[1].observation)
        != bank[0].transitions[1].observation
    )


@pytest.mark.parametrize(
    "fault", ["role", "context", "steps", "causal", "next_view", "extra", "duplicate", "list"]
)
def test_should_refuse_invalid_public_bank_before_learner_admission(fault):
    bank = tuple(packet.episode for packet in train_sources()[:2])
    if fault == "role":
        bank = (bank[0], replace(bank[1], role="probe_train"))
    elif fault == "context":
        bank = (bank[0], replace(bank[1], context="F"))
    elif fault == "steps":
        shorter = (*bank[1].transitions[:-2], replace(bank[1].transitions[-2], terminal=True))
        bank = (bank[0], replace(bank[1], transitions=shorter))
    elif fault in {"causal", "next_view"}:
        item = bank[1].transitions[-1]
        field = "observation" if fault == "causal" else "next_observation"
        values = list(getattr(item, field))
        values[0] = 0.123
        object.__setattr__(item, field, tuple(values))
    elif fault == "extra":
        object.__setattr__(bank[1], "unknown", True)
    elif fault == "duplicate":
        bank = (bank[0], bank[0])
    else:
        bank = list(bank)
    wake, replay = budgets()
    with pytest.raises(ValueError) as failure:
        adapter.build_minigrid_old_a_rehearsal(
            object(), wake, bank, replay_budget=replay, total_updates=2
        )
    assert "learner" not in str(failure.value)
    assert wake.updates_completed == replay.replay_examples_completed == 0
