"""Weighted PC/neutral schedules through the existing TRAIN-only delayed stream.

These engineering cases reuse the accepted episode update as the numerical oracle;
they establish no acquisition benefit and run no environment physics.
"""

from copy import deepcopy
from dataclasses import replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters import torch_pc_weighted as native_weighted
from src.adapters import torch_train_weighted_learner as adapter
from src.adapters.invariant_torch_feature_learner import InvariantTorchFeatureLearner
from src.adapters.torch_feature_learner import TorchFeaturePolicy
from src.app import coupled_attribution as stream_app
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.demonstration_class_weights import episode_class_weights
from src.core.demonstration_episode import episode_arrays
from src.core.learner_ports import TrainingDiagnostic
from src.core.resnet50_variants import CircadianPredictiveCodingHead, PredictiveCodingHead
from src.shared.torch_runtime import require_torch
from test_torch_feature_learner import NAMES, assert_tree, tensors
from test_weighted_feature_stream_adapter import episode, policy, schedule


@pytest.fixture(scope="module")
def torch():
    return require_torch()


@pytest.fixture(autouse=True)
def work_accounting(monkeypatch, record_property):
    work = dict(
        weighted_attempts=0,
        weighted_rows=0,
        primitive_attempts=0,
        primitive_rows=0,
        nominal_latent_row_iterations=0,
        commit_attempts=0,
        commit_successes=0,
    )
    original = native_weighted.train_weighted_episode
    proposal = native_weighted.class_weighted_pc_update

    def weighted(owner, packet, weights):
        work["weighted_attempts"] += 1
        work["weighted_rows"] += len(packet.transitions)
        return original(owner, packet, weights)

    def propose(features, targets, parameters, **kwargs):
        work["primitive_attempts"] += 1
        work["primitive_rows"] += len(features)
        work["nominal_latent_row_iterations"] += len(features) * kwargs["inference_steps"]
        return proposal(features, targets, parameters, **kwargs)

    monkeypatch.setattr(native_weighted, "train_weighted_episode", weighted)
    monkeypatch.setattr(adapter, "train_weighted_pc_episode", weighted)
    monkeypatch.setattr(native_weighted, "class_weighted_pc_update", propose)
    for kind in (PredictiveCodingHead, CircadianPredictiveCodingHead):
        original_commit = kind._apply_training_gradients

        def commit(head, *args, _original=original_commit, **kwargs):
            work["commit_attempts"] += 1
            result = _original(head, *args, **kwargs)
            work["commit_successes"] += 1
            return result

        monkeypatch.setattr(kind, "_apply_training_gradients", commit)
    yield work
    for name, value in work.items():
        record_property(name, value)


def pc_policy(method, steps=2):
    original = policy(method)
    return replace(original, native_policy=replace(original.native_policy, inference_steps=steps))


def learner(method, records=None, steps=2):
    return adapter.TorchTrainWeightedLearner(
        pc_policy(method, steps), schedule() if records is None else records
    )


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
@pytest.mark.parametrize("steps", [2, 64])
def test_should_match_direct_weighted_full_state_through_exhaustion(torch, method, steps):
    source = learner(method, steps=steps)
    reference = InvariantTorchFeatureLearner(pc_policy(method, steps))
    initial = source.snapshot_state()
    for index, packet in enumerate(schedule()):
        actual = source.train_batch(*episode_arrays(packet))
        expected = native_weighted.train_weighted_episode(
            reference, packet, episode_class_weights(packet)
        )
        assert actual == expected
        assert actual.definition == native_weighted.TORCH_INVARIANT_WEIGHTED_PC_ID
        assert_tree(torch, source.snapshot_state(), reference.snapshot_state())
        assert source.completed_updates == index + 1
    assert initial.completed_updates == initial.training_examples == 0
    assert source.training_examples == 7
    assert source.snapshot_state().native_state.optimizer_state is None
    with pytest.raises(ValueError, match="exhausted"):
        source.train_batch(*episode_arrays(schedule()[-1]))


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
def test_should_derive_weights_only_from_the_current_owned_train_episode(
    torch, monkeypatch, method
):
    records = schedule()
    source = learner(method, records)
    first = deepcopy(records[0])
    object.__setattr__(records[0], "role", "final_test")
    seen = []
    original = adapter.train_weighted_pc_episode

    def observe(owner, packet, weights):
        seen.append((deepcopy(packet), weights))
        return original(owner, packet, weights)

    monkeypatch.setattr(adapter, "train_weighted_pc_episode", observe)
    for packet in (first, records[1]):
        features, targets = episode_arrays(packet)
        source.train_batch(features, targets)
        features.fill(-1)
        targets.fill(1)
    assert [packet for packet, _ in seen] == [first, records[1]]
    assert seen[0][1] == (0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5)
    assert seen[1][1] == episode_class_weights(records[1])
    assert seen[0][1] != seen[1][1]
    assert source.training_examples == 7


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
def test_should_fork_restore_and_deepcopy_without_sharing_state_or_schedule(torch, method):
    source = learner(method)
    initial = source.snapshot_state()
    source.train_batch(*episode_arrays(schedule()[0]))
    saved = source.snapshot_state()
    fork, copied = source.fork(), deepcopy(source)
    for child in (fork, copied):
        assert child._learner is not source._learner
        assert child._learner._native is not source._learner._native
        assert child._episodes is not source._episodes
        assert child._episodes[0] is not source._episodes[0]
        child.train_batch(*episode_arrays(schedule()[1]))
    assert_tree(torch, fork.snapshot_state(), copied.snapshot_state())
    assert_tree(torch, source.snapshot_state(), saved)
    assert all(
        left.data_ptr() != right.data_ptr()
        for left, right in zip(
            tensors(torch, source.snapshot_state()),
            tensors(torch, fork.snapshot_state()),
            strict=True,
        )
    )
    source.restore_state(initial)
    source.train_batch(*episode_arrays(schedule()[0]))
    assert_tree(torch, source.snapshot_state(), saved)


def test_should_preserve_pc_neutral_parameter_identity_for_matched_weighted_schedule(torch):
    pc, neutral = learner("pc"), learner("neutral_cpc")
    for packet in schedule():
        assert pc.train_batch(*episode_arrays(packet)) == neutral.train_batch(
            *episode_arrays(packet)
        )
        left, right = pc.snapshot_state(), neutral.snapshot_state()
        for name in NAMES:
            assert torch.equal(
                left.native_state.head_state[name], right.native_state.head_state[name]
            )


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
def test_should_execute_three_real_delayed_updates_with_stable_actor_and_train_receipts(
    torch, monkeypatch, method
):
    records = (*schedule(), episode("third", (1, 6), cue=4))
    source = learner(method, records)
    initial = source.snapshot_state()
    packets = tuple(
        stream_app.AttributionBatch(
            "train",
            record.context,
            tuple(f"{record.episode_id}:{index}" for index in range(len(record.transitions))),
            *episode_arrays(record),
            num_classes=7,
        )
        for record in records
    )
    original_packets = pickle.dumps(packets)
    streams = []
    original_stream = stream_app.AttributionStream

    def track(*args, **kwargs):
        result = original_stream(*args, **kwargs)
        streams.append(result)
        return result

    monkeypatch.setattr(stream_app, "AttributionStream", track)
    states = []
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=3), lambda: 0.0)

    def observe(point, state, packet):
        states.append((point, state))
        assert_tree(torch, source.snapshot_state(), initial)

    report = stream_app.run_attribution_stream(
        source,
        packets,
        (),
        checkpoint_updates=(0, 1, 2, 3),
        label_delay=1,
        budget=budget,
        observe_checkpoint=observe,
        training_only=True,
    )
    assert report["completed_updates"] == budget.updates_completed == 3
    assert report["training_examples"] == 9 and report["score_forward_calls"] == 0
    assert report["evaluation_scope"] == "training_only"
    assert [point for point, _ in states] == [0, 1, 2, 3]
    receipts = report["applied_updates"]
    assert [row["arrived_at"] for row in receipts] == [1, 2, 3]
    assert [row["update_number"] for row in receipts] == [1, 2, 3]
    assert all(
        row["diagnostic"]["definition"] == native_weighted.TORCH_INVARIANT_WEIGHTED_PC_ID
        for row in receipts
    )
    assert_tree(torch, source.snapshot_state(), initial)
    assert_tree(torch, streams[0]._runtime.actor.snapshot_state().state, initial)
    assert states[-1][1].completed_updates == 3 and states[-1][1].training_examples == 9
    assert pickle.dumps(packets) == original_packets


@pytest.mark.parametrize("fault", ["reserved", "late_next", "gated", "unbound", "foreign"])
def test_should_refuse_invalid_policy_or_whole_schedule_before_native_construction(
    monkeypatch, fault
):
    records = schedule()
    candidate: Any = pc_policy("neutral_cpc")
    if fault == "reserved":
        object.__setattr__(records[-1], "role", "probe_train")
    elif fault == "late_next":
        transition = records[-1].transitions[-1]
        values = list(transition.next_observation)
        values[-1] = 0.123
        object.__setattr__(transition, "next_observation", tuple(values))
    elif fault == "gated":
        candidate = pc_policy("gated_cpc")
    elif fault == "unbound":
        candidate = TorchFeaturePolicy("pc", 296, 8, 17, num_classes=7)
    else:
        candidate = object()
    monkeypatch.setattr(
        adapter,
        "InvariantTorchFeatureLearner",
        lambda *a, **kw: pytest.fail("invalid admission allocated a bound native owner"),
    )
    with pytest.raises(ValueError):
        adapter.TorchTrainWeightedLearner(candidate, records)


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
@pytest.mark.parametrize("fault", ["order", "labels", "dtype"])
def test_should_refuse_nonmatching_arrays_without_paid_native_work(
    torch, work_accounting, method, fault
):
    source = learner(method)
    before = source.snapshot_state()
    features, targets = episode_arrays(schedule()[0])
    if fault == "order":
        features, targets = episode_arrays(schedule()[1])
    elif fault == "labels":
        targets[0, 0] = 1
    else:
        features = features.astype(np.float32)
    with pytest.raises(ValueError):
        source.train_batch(features, targets)
    assert_tree(torch, source.snapshot_state(), before)
    assert work_accounting["weighted_attempts"] == work_accounting["commit_attempts"] == 0


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
@pytest.mark.parametrize("fault", ["native", "objective", "counter", "input"])
def test_should_not_publish_failed_candidates_and_keep_paid_failure_work(
    torch, monkeypatch, work_accounting, method, fault
):
    source = learner(method)
    before = source.snapshot_state()
    features, targets = episode_arrays(schedule()[0])
    original = adapter.train_weighted_pc_episode
    primary = RuntimeError("injected weighted PC candidate failure")

    def inject(owner, packet, weights):
        if fault == "native":
            raise primary
        result = original(owner, packet, weights)
        if fault == "objective":
            return TrainingDiagnostic("wrong_objective", result.value)
        if fault == "counter":
            head, optimizer, updates, rows = owner._native._payload
            owner._native._payload = (head, optimizer, updates, rows + 1)
        elif fault == "input":
            features[0, 0] += 0.1
        return result

    monkeypatch.setattr(adapter, "train_weighted_pc_episode", inject)
    with pytest.raises((ValueError, RuntimeError)) as error:
        source.train_batch(features, targets)
    if fault == "native":
        assert error.value is primary
    assert_tree(torch, source.snapshot_state(), before)
    assert work_accounting["commit_successes"] == (0 if fault == "native" else 1)


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
@pytest.mark.parametrize("fault", ["weights", "shared_native"])
def test_should_refuse_foreign_candidate_before_weighted_work(
    torch, monkeypatch, work_accounting, method, fault
):
    source = learner(method)
    before = source.snapshot_state()
    candidate = source._learner.fork()
    if fault == "weights":
        with torch.no_grad():
            candidate._native._head.bias_output[0, 0] += 0.125
    else:
        candidate._native = source._learner._native
    monkeypatch.setattr(source._learner, "fork", lambda: candidate)
    with pytest.raises(ValueError):
        source.train_batch(*episode_arrays(schedule()[0]))
    assert_tree(torch, source.snapshot_state(), before)
    assert work_accounting["weighted_attempts"] == 0


@pytest.mark.parametrize("operation", ["train", "restore"])
@pytest.mark.parametrize("fault", ["payload", "policy", "rng"])
def test_should_preserve_foreign_owner_identity_or_rng_change_and_refuse_publication(
    torch, monkeypatch, operation, fault
):
    source = learner("neutral_cpc" if operation == "restore" else "pc")
    before = source.snapshot_state()
    owner, raw = source._learner, source._learner._native
    changed = []

    def mutate():
        if fault == "payload":
            raw._payload = tuple(list(raw._payload))
            changed.append(raw._payload)
        elif fault == "policy":
            owner._policy = replace(owner.policy)
            changed.append(owner._policy)
        else:
            torch.rand(1)
            changed.append(torch.get_rng_state().clone())

    if operation == "train":
        original = adapter.train_weighted_pc_episode

        def inject(candidate, packet, weights):
            result = original(candidate, packet, weights)
            mutate()
            return result

        monkeypatch.setattr(adapter, "train_weighted_pc_episode", inject)
        invoke = lambda: source.train_batch(*episode_arrays(schedule()[0]))
    else:
        restore = adapter.feature_native._restore_components

        def inject_restore(snapshot):
            result = restore(snapshot)
            if not changed:
                mutate()
            return result

        monkeypatch.setattr(adapter.feature_native, "_restore_components", inject_restore)
        invoke = lambda: source.restore_state(before)
    initial_rng = torch.get_rng_state().clone()
    try:
        with pytest.raises(ValueError):
            invoke()
        assert changed
        if fault == "payload":
            assert raw._payload is changed[0]
        elif fault == "policy":
            assert owner._policy is changed[0]
        else:
            assert torch.equal(torch.get_rng_state(), changed[0])
        assert_tree(torch, owner.snapshot_state(), before)
    finally:
        torch.set_rng_state(initial_rng)


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
def test_should_preserve_foreign_donor_parameter_change_without_rollback(
    torch, monkeypatch, method
):
    source = learner(method)
    owner = source._learner
    before = owner.snapshot_state()
    original = adapter.train_weighted_pc_episode

    def inject(candidate, packet, weights):
        result = original(candidate, packet, weights)
        with torch.no_grad():
            owner._native._head.bias_output[0, 0] += 0.125
        return result

    monkeypatch.setattr(adapter, "train_weighted_pc_episode", inject)
    with pytest.raises(ValueError):
        source.train_batch(*episode_arrays(schedule()[0]))
    expected = deepcopy(before)
    expected.native_state.head_state["bias_output"][0, 0] += 0.125
    assert_tree(torch, owner.snapshot_state(), expected)
    assert source._learner is owner
    with pytest.raises(ValueError):
        source.snapshot_state()


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
def test_should_refuse_changed_restore_input_without_publishing(torch, monkeypatch, method):
    source = learner(method)
    before = source.snapshot_state()
    requested = deepcopy(before)
    restore = adapter.feature_native._restore_components

    def inject(snapshot):
        result = restore(snapshot)
        requested.native_state.head_state["bias_output"][0, 0] += 0.125
        return result

    monkeypatch.setattr(adapter.feature_native, "_restore_components", inject)
    with pytest.raises(ValueError):
        source.restore_state(requested)
    assert_tree(torch, source.snapshot_state(), before)


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
def test_should_predict_without_update_and_refuse_recurrent_session(torch, work_accounting, method):
    source = learner(method)
    before = source.snapshot_state()
    probabilities = source.predict(episode_arrays(schedule()[0])[0])
    assert probabilities.shape == (3, 7) and np.all(np.isfinite(probabilities))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    with pytest.raises(ValueError):
        source.start_episode()
    assert_tree(torch, source.snapshot_state(), before)
    assert work_accounting["weighted_attempts"] == 0
