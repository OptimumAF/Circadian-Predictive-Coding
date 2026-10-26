"""Bound weighted BP through the existing delayed stream, without game physics.

These are engineering checks, not a comparative learning experiment. The root
runs them serially and records every weighted/reference attempt and SGD step.
"""

from copy import deepcopy
from dataclasses import replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters import torch_train_weighted_learner as adapter
from src.adapters import torch_feature_weighted as native_weighted
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeaturePolicy,
    InvariantTorchFeatureSnapshot,
)
from src.adapters.torch_feature_learner import TorchFeaturePolicy
from src.app import coupled_attribution as stream_app
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.demonstration_class_weights import episode_class_weights
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    episode_arrays,
)
from src.core.learner_ports import TrainingDiagnostic
from src.shared.torch_runtime import require_torch
from test_minigrid_invariant_encoding import frame
from test_torch_feature_learner import assert_tree, tensors


@pytest.fixture(scope="module")
def torch():
    return require_torch()


@pytest.fixture(autouse=True)
def update_accounting(monkeypatch, record_property, torch):
    work = dict(train_attempts=0, train_rows=0, sgd_steps=0)
    original = native_weighted.train_weighted_episode
    original_step = torch.optim.SGD.step

    def counted(owner, packet, weights):
        work["train_attempts"] += 1
        work["train_rows"] += len(packet.transitions)
        return original(owner, packet, weights)

    def step(*args, **kwargs):
        work["sgd_steps"] += 1
        return original_step(*args, **kwargs)

    monkeypatch.setattr(native_weighted, "train_weighted_episode", counted)
    monkeypatch.setattr(adapter, "train_weighted_feature_episode", counted)
    monkeypatch.setattr(torch.optim.SGD, "step", step)
    yield work
    assert work["train_attempts"] <= 64 and work["train_rows"] <= 1024
    for name, value in work.items():
        record_property(name, value)


def policy(method="bp"):
    return InvariantTorchFeaturePolicy(TorchFeaturePolicy(method, 296, 8, 17, num_classes=7))


def episode(identity="first", actions=(0, 0, 6), *, role="train", cue=0):
    initial = frame(wall_color=1, direction=2, cue_color=cue, key_color=cue)
    observations = tuple(
        initial
        + frame(
            wall_color=index % 6,
            direction=index % 4,
            cue_color=cue,
            key_color=cue,
            door_state=index % 3,
        )
        for index in range(len(actions) + 1)
    )
    return DemonstrationEpisode(
        identity,
        "A",
        role,
        tuple(
            DemonstrationTransition(
                index,
                observations[index],
                action,
                0.0,
                observations[index + 1],
                index == len(actions) - 1,
                7,
            )
            for index, action in enumerate(actions)
        ),
        7,
    )


def schedule():
    return (episode(), episode("second", (2, 2, 2, 6), cue=2))


def learner(records=None):
    return adapter.TorchTrainWeightedLearner(policy(), schedule() if records is None else records)


def test_should_match_native_weighted_bp_full_state_and_objective_across_updates(torch):
    source, reference = learner(), InvariantTorchFeatureLearner(policy())
    originals = schedule()
    saved = source.snapshot_state()
    assert type(saved) is InvariantTorchFeatureSnapshot and source.policy == policy()
    for index, packet in enumerate(originals):
        result = source.train_batch(*episode_arrays(packet))
        expected = native_weighted.train_weighted_episode(
            reference, packet, episode_class_weights(packet)
        )
        assert result == expected
        assert result.definition == native_weighted.TORCH_INVARIANT_WEIGHTED_CE_ID
        assert_tree(torch, source.snapshot_state(), reference.snapshot_state())
        assert source.completed_updates == index + 1
    assert source.training_examples == 7 and saved.completed_updates == 0
    assert saved.native_state.optimizer_state is not None
    assert saved.native_state.optimizer_state["state"] == {}
    assert source.snapshot_state().native_state.optimizer_state["state"]
    with pytest.raises(ValueError, match="exhausted"):
        source.train_batch(*episode_arrays(originals[-1]))


def test_should_own_original_schedule_and_arrays_and_derive_only_current_weights(
    torch, monkeypatch
):
    records = schedule()
    source = learner(records)
    saved_packet = deepcopy(records[0])
    object.__setattr__(records[0], "role", "final_test")
    seen = []
    original = adapter.train_weighted_feature_episode

    def observe(owner, packet, weights):
        seen.append((deepcopy(packet), weights))
        return original(owner, packet, weights)

    monkeypatch.setattr(adapter, "train_weighted_feature_episode", observe)
    features, targets = episode_arrays(saved_packet)
    source.train_batch(features, targets)
    features.fill(-1)
    targets.fill(1)
    packet, weights = seen[0]
    assert packet == saved_packet and packet is not records[0]
    assert weights == episode_class_weights(saved_packet) == (0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5)
    assert source.completed_updates == 1 and source.training_examples == 3


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "empty",
        "oversized",
        "final_test",
        "probe_train",
        "probe_test",
        "extra",
        "late_current",
        "late_next",
        "incomplete",
        "wrong_width",
    ],
)
def test_should_preflight_whole_schedule_before_any_native_owner(monkeypatch, fault):
    records: Any = schedule()
    if fault == "list":
        records = list(records)
    elif fault == "empty":
        records = ()
    elif fault == "oversized":
        records = (records[0],) * 513
    elif fault in ("final_test", "probe_train", "probe_test"):
        records = (records[0], episode("reserved", role=fault))
    elif fault == "extra":
        object.__setattr__(records[-1], "foreign_field", 1)
    elif fault in ("late_current", "late_next"):
        transition = records[-1].transitions[-1]
        field = "observation" if fault == "late_current" else "next_observation"
        values = list(getattr(transition, field))
        values[-1] = 0.123
        object.__setattr__(transition, field, tuple(values))
    elif fault == "incomplete":
        object.__setattr__(records[-1].transitions[-1], "terminal", False)
    else:
        object.__setattr__(records[-1].transitions[-1], "next_observation", (0.0,) * 148)
    monkeypatch.setattr(
        adapter,
        "InvariantTorchFeatureLearner",
        lambda *a, **kw: pytest.fail("invalid schedule created native resources"),
    )
    with pytest.raises(ValueError):
        adapter.TorchTrainWeightedLearner(policy(), records)


@pytest.mark.parametrize("method", ["gated_cpc"])
def test_should_refuse_other_bound_objectives_before_native_resources(monkeypatch, method):
    monkeypatch.setattr(
        adapter,
        "InvariantTorchFeatureLearner",
        lambda *a, **kw: pytest.fail("unsupported feature policy created native resources"),
    )
    with pytest.raises(ValueError):
        adapter.TorchTrainWeightedLearner(policy(method), schedule())


@pytest.mark.parametrize("fault", ["order", "features", "labels", "dtype", "list", "nan"])
def test_should_refuse_arrays_not_matching_next_packet_before_native_update(torch, fault):
    source = learner()
    before = source.snapshot_state()
    features, targets = episode_arrays(schedule()[0])
    if fault == "order":
        features, targets = episode_arrays(schedule()[1])
    elif fault == "features":
        features[0, 1] += 0.1
    elif fault == "labels":
        targets[0, 0] = 1
    elif fault == "dtype":
        features = features.astype(np.float32)
    elif fault == "list":
        features = features.tolist()
    else:
        features[0, 0] = np.nan
    with pytest.raises(ValueError):
        source.train_batch(features, targets)
    assert_tree(torch, source.snapshot_state(), before)


@pytest.mark.parametrize("fault", ["schedule", "role", "objective", "owner", "policy", "state"])
def test_should_fail_closed_on_sealed_metadata_or_source_changes_before_work(torch, fault):
    source = learner()
    if fault == "schedule":
        source._episodes = deepcopy(schedule())
    elif fault == "role":
        object.__setattr__(source._episodes[-1], "role", "probe_train")
    elif fault == "objective":
        source._objective_id = "ordinary_mean_ce"
    elif fault == "owner":
        source._learner = InvariantTorchFeatureLearner(policy())
    elif fault == "policy":
        object.__setattr__(source._policy.native_policy, "momentum", 0.5)
    else:
        with torch.no_grad():
            source._learner._native._head.bias_output[0, 0] += 0.125
    with pytest.raises(ValueError):
        source.train_batch(*episode_arrays(schedule()[0]))


def test_should_fork_restore_and_deepcopy_full_state_without_aliasing(torch):
    source = learner()
    initial = source.snapshot_state()
    source.train_batch(*episode_arrays(schedule()[0]))
    saved = source.snapshot_state()
    children = (source.fork(), deepcopy(source))
    for child in children:
        assert child._learner is not source._learner
        assert child._episodes is not source._episodes
        assert child._episodes[0] is not source._episodes[0]
        assert_tree(torch, child.snapshot_state(), saved)
        child.train_batch(*episode_arrays(schedule()[1]))
    assert_tree(torch, children[0].snapshot_state(), children[1].snapshot_state())
    assert_tree(torch, source.snapshot_state(), saved)
    source.restore_state(initial)
    source.train_batch(*episode_arrays(schedule()[0]))
    assert_tree(torch, source.snapshot_state(), saved)
    pairs = zip(
        tensors(torch, source.snapshot_state()),
        tensors(torch, children[0].snapshot_state()),
        strict=True,
    )
    assert all(left.data_ptr() != right.data_ptr() for left, right in pairs)
    object.__setattr__(children[0]._episodes[-1], "role", "final_test")
    with pytest.raises(ValueError):
        children[0].snapshot_state()
    assert_tree(torch, source.snapshot_state(), saved)


def test_should_predict_without_updates_and_refuse_recurrent_sessions(torch, update_accounting):
    source = learner()
    before = source.snapshot_state()
    probabilities = source.predict(episode_arrays(schedule()[0])[0])
    assert probabilities.shape == (3, 7) and np.all(np.isfinite(probabilities))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    with pytest.raises(ValueError):
        source.start_episode()
    assert_tree(torch, source.snapshot_state(), before)
    assert update_accounting["train_attempts"] == update_accounting["sgd_steps"] == 0


@pytest.mark.parametrize("fault", ["native", "objective", "counter", "input"])
def test_should_not_publish_failed_detached_updates(torch, monkeypatch, fault):
    source = learner()
    before = source.snapshot_state()
    features, targets = episode_arrays(schedule()[0])
    original = adapter.train_weighted_feature_episode
    primary = RuntimeError("injected detached weighted failure")

    def inject(owner, packet, weights):
        if fault == "native":
            raise primary
        result = original(owner, packet, weights)
        if fault == "objective":
            return TrainingDiagnostic("wrong_objective", result.value)
        if fault == "counter":
            head, optimizer, count, rows = owner._native._payload
            owner._native._payload = (head, optimizer, count, rows + 1)
        else:
            features[0, 0] += 0.1
        return result

    monkeypatch.setattr(adapter, "train_weighted_feature_episode", inject)
    with pytest.raises((RuntimeError, ValueError)) as failure:
        source.train_batch(features, targets)
    if fault == "native":
        assert failure.value is primary
    assert_tree(torch, source.snapshot_state(), before)


@pytest.mark.parametrize("fault", ["donor", "schedule"])
def test_should_preserve_external_changes_and_refuse_publication_without_rollback(
    torch, monkeypatch, fault
):
    source = learner()
    owner = source._learner
    before = owner.snapshot_state()
    original = adapter.train_weighted_feature_episode

    def inject(candidate, packet, weights):
        result = original(candidate, packet, weights)
        if fault == "donor":
            with torch.no_grad():
                owner._native._head.bias_output[0, 0] += 0.125
        else:
            object.__setattr__(source._episodes[-1], "role", "final_test")
        return result

    monkeypatch.setattr(adapter, "train_weighted_feature_episode", inject)
    with pytest.raises(ValueError):
        source.train_batch(*episode_arrays(schedule()[0]))
    assert source._learner is owner
    if fault == "donor":
        changed = owner.snapshot_state()
        expected = deepcopy(before)
        expected.native_state.head_state["bias_output"][0, 0] += 0.125
        assert_tree(torch, changed, expected)
    else:
        assert source._episodes[-1].role == "final_test"
        assert_tree(torch, owner.snapshot_state(), before)
    with pytest.raises(ValueError):
        source.snapshot_state()


def test_should_refuse_invalid_restore_prefix_without_changing_source(torch):
    source = learner()
    source.train_batch(*episode_arrays(schedule()[0]))
    saved = source.snapshot_state()
    bad = replace(
        saved, native_state=replace(saved.native_state, completed_updates=1, training_examples=4)
    )
    with pytest.raises(ValueError):
        source.restore_state(bad)
    assert_tree(torch, source.snapshot_state(), saved)


def test_should_preserve_external_donor_change_during_detached_restore(torch, monkeypatch):
    source = learner()
    initial = source.snapshot_state()
    source.train_batch(*episode_arrays(schedule()[0]))
    owner = source._learner
    before = owner.snapshot_state()
    original = adapter.feature_native._restore_components
    changed = False

    def inject(snapshot):
        nonlocal changed
        result = original(snapshot)
        if not changed:
            changed = True
            with torch.no_grad():
                owner._native._head.bias_output[0, 0] += 0.125
        return result

    monkeypatch.setattr(adapter.feature_native, "_restore_components", inject)
    with pytest.raises(ValueError):
        source.restore_state(initial)
    assert source._learner is owner
    expected = deepcopy(before)
    expected.native_state.head_state["bias_output"][0, 0] += 0.125
    assert_tree(torch, owner.snapshot_state(), expected)
    with pytest.raises(ValueError):
        source.snapshot_state()


def test_should_execute_three_delayed_weighted_updates_with_receipts_and_stable_actor(
    torch, monkeypatch
):
    records = (*schedule(), episode("third", (1, 6), cue=4))
    source = learner(records)
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
    actual_streams = []
    native_stream = stream_app.AttributionStream

    def track(*args, **kwargs):
        result = native_stream(*args, **kwargs)
        actual_streams.append(result)
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
        row["diagnostic"]["definition"] == native_weighted.TORCH_INVARIANT_WEIGHTED_CE_ID
        for row in receipts
    )
    assert_tree(torch, source.snapshot_state(), initial)
    assert_tree(torch, actual_streams[0]._runtime.actor.snapshot_state().state, initial)
    assert states[-1][1].completed_updates == 3 and states[-1][1].training_examples == 9
    assert pickle.dumps(packets) == original_packets


@pytest.mark.parametrize("fault", ["weights", "shared_native"])
def test_should_refuse_substituted_fork_before_weighted_work(
    torch, monkeypatch, update_accounting, fault
):
    source = learner()
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
    assert update_accounting["train_attempts"] == 0


@pytest.mark.parametrize("operation", ["train", "restore"])
@pytest.mark.parametrize("fault", ["grad", "payload", "policy", "rng"])
def test_should_refuse_foreign_identity_metadata_and_rng_changes(
    torch, monkeypatch, operation, fault
):
    source = learner()
    saved = source.snapshot_state()
    owner = source._learner
    raw = owner._native
    changed = []

    def mutate():
        if fault == "grad":
            raw._head.weight_feature_hidden.requires_grad_(False)
            changed.append(raw._head.weight_feature_hidden)
        elif fault == "payload":
            raw._payload = tuple(list(raw._payload))
            changed.append(raw._payload)
        elif fault == "policy":
            owner._policy = replace(owner.policy)
            changed.append(owner._policy)
        else:
            torch.rand(1)
            changed.append(torch.get_rng_state().clone())

    if operation == "train":
        original = adapter.train_weighted_feature_episode

        def inject(candidate, packet, weights):
            result = original(candidate, packet, weights)
            mutate()
            return result

        monkeypatch.setattr(adapter, "train_weighted_feature_episode", inject)
        invoke = lambda: source.train_batch(*episode_arrays(schedule()[0]))
    else:
        original_restore = adapter.feature_native._restore_components

        def inject_restore(snapshot):
            result = original_restore(snapshot)
            if not changed:
                mutate()
            return result

        monkeypatch.setattr(adapter.feature_native, "_restore_components", inject_restore)
        invoke = lambda: source.restore_state(saved)
    original_rng = torch.get_rng_state().clone()
    try:
        with pytest.raises(ValueError):
            invoke()
        assert changed
        if fault == "grad":
            assert raw._head.weight_feature_hidden is changed[0]
            assert not raw._head.weight_feature_hidden.requires_grad
        elif fault == "payload":
            assert raw._payload is changed[0]
        elif fault == "policy":
            assert owner._policy is changed[0]
        else:
            assert torch.equal(torch.get_rng_state(), changed[0])
        assert_tree(torch, owner.snapshot_state(), saved)
    finally:
        torch.set_rng_state(original_rng)


def test_should_refuse_valid_but_different_detached_restore(torch, monkeypatch):
    source = learner()
    saved = source.snapshot_state()
    original = adapter.feature_native._restore_components

    def inject(snapshot):
        head, optimizer = original(snapshot)
        with torch.no_grad():
            head.bias_output[0, 0] += 0.125
        return head, optimizer

    monkeypatch.setattr(adapter.feature_native, "_restore_components", inject)
    with pytest.raises(ValueError):
        source.restore_state(saved)
    assert_tree(torch, source.snapshot_state(), saved)


def test_should_refuse_restore_input_mutation_without_publishing(torch, monkeypatch):
    source = learner()
    saved = source.snapshot_state()
    requested = deepcopy(saved)
    original = adapter.feature_native._restore_components

    def inject(snapshot):
        result = original(snapshot)
        requested.native_state.head_state["bias_output"][0, 0] += 0.125
        return result

    monkeypatch.setattr(adapter.feature_native, "_restore_components", inject)
    with pytest.raises(ValueError):
        source.restore_state(requested)
    assert_tree(torch, source.snapshot_state(), saved)


def test_should_refuse_public_fork_sharing_raw_native(torch, monkeypatch):
    source = learner()
    candidate = source._learner.fork()
    candidate._native = source._learner._native
    monkeypatch.setattr(source._learner, "fork", lambda: candidate)
    with pytest.raises(ValueError):
        source.fork()
