"""Atomic complete-TRAIN weighted PC updates, with no environment physics.

The accepted primitive supplies the numerical oracle. These boundary checks
exercise full native state, source admission, continuation, and foreign changes;
they do not establish acquisition or a circadian benefit.
"""

from copy import deepcopy
from dataclasses import replace
import pickle
import random
from typing import Any, cast

import numpy as np
import pytest

from src.adapters import torch_feature_learner as native
from src.adapters import torch_pc_weighted as weighted
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeaturePolicy,
)
from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy
from src.adapters.torch_recurrent_learner import TorchRecurrentLearner, TorchRecurrentPolicy
from src.core.categorical_pc_weighted import class_weighted_pc_update
from src.core.demonstration_episode import episode_arrays
from src.core.resnet50_variants import CircadianPredictiveCodingHead, PredictiveCodingHead
from src.shared.torch_runtime import require_torch
from test_torch_feature_learner import NAMES, assert_tree, tensors
from test_torch_feature_weighted import episode, owner


WEIGHTS = (0.25, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
ONES = (1.0,) * 7


@pytest.fixture(scope="module")
def torch():
    return require_torch()


@pytest.fixture(autouse=True)
def work_accounting(monkeypatch, record_property):
    work = dict(
        commit_attempts=0,
        commit_successes=0,
        commit_rows=0,
        primitive_attempts=0,
        primitive_rows=0,
        primitive_nominal_latent_row_iterations=0,
        native_calls=0,
        native_rows=0,
        native_nominal_latent_row_iterations=0,
    )
    original_proposal = weighted.class_weighted_pc_update

    def propose(features, targets, parameters, **kwargs):
        work["primitive_attempts"] += 1
        work["primitive_rows"] += len(features)
        work["primitive_nominal_latent_row_iterations"] += len(features) * kwargs["inference_steps"]
        return original_proposal(features, targets, parameters, **kwargs)

    monkeypatch.setattr(weighted, "class_weighted_pc_update", propose)
    for kind in (PredictiveCodingHead, CircadianPredictiveCodingHead):
        original = kind._apply_training_gradients
        original_train = kind.train_step

        def train(head, features, targets, learning_rate, steps, rate, _train=original_train):
            work["native_calls"] += 1
            work["native_rows"] += len(features)
            work["native_nominal_latent_row_iterations"] += len(features) * steps
            return _train(head, features, targets, learning_rate, steps, rate)

        def commit(head, hidden_state, *args, _original=original, **kwargs):
            work["commit_attempts"] += 1
            work["commit_rows"] += len(hidden_state)
            result = _original(head, hidden_state, *args, **kwargs)
            work["commit_successes"] += 1
            return result

        monkeypatch.setattr(kind, "_apply_training_gradients", commit)
        monkeypatch.setattr(kind, "train_step", train)
    yield work
    for name, value in work.items():
        record_property(name, value)


def install_after_commit(monkeypatch, callback):
    """Inject after actual candidate work, before the adapter may publish it."""
    for kind in (PredictiveCodingHead, CircadianPredictiveCodingHead):
        original = kind._apply_training_gradients

        def commit(head, *args, _original=original, **kwargs):
            result = _original(head, *args, **kwargs)
            callback(head)
            return result

        monkeypatch.setattr(kind, "_apply_training_gradients", commit)


def pc_owner(method, steps):
    return InvariantTorchFeatureLearner(
        InvariantTorchFeaturePolicy(
            TorchFeaturePolicy(method, 296, 8, 17, inference_steps=steps, num_classes=7)
        )
    )


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
@pytest.mark.parametrize("steps", [2, 64])
def test_should_preserve_complete_native_all_one_state_and_fork_restore(torch, method, steps):
    learner, reference, packet = pc_owner(method, steps), pc_owner(method, steps), episode()
    for _ in range(2):
        actual = weighted.train_weighted_episode(learner, packet, ONES)
        ordinary = reference.train_batch(*episode_arrays(packet))
        assert actual.definition == weighted.TORCH_INVARIANT_WEIGHTED_PC_ID
        assert actual.definition != ordinary.definition
        assert_tree(torch, learner.snapshot_state(), reference.snapshot_state())
    saved = learner.snapshot_state()
    child, restored = learner.fork(), pc_owner(method, steps)
    restored.restore_state(saved)
    for item in (learner, child, restored):
        weighted.train_weighted_episode(item, packet, WEIGHTS)
    expected = learner.snapshot_state()
    assert_tree(torch, child.snapshot_state(), expected)
    assert_tree(torch, restored.snapshot_state(), expected)
    assert expected.completed_updates == 3 and expected.training_examples == 12
    assert all(
        left.data_ptr() != right.data_ptr()
        for left, right in zip(tensors(torch, saved), tensors(torch, expected), strict=True)
    )
    for value in tensors(torch, saved):
        value.zero_()
    assert_tree(torch, learner.snapshot_state(), expected)


def test_should_match_accepted_primitive_and_neutral_parameters_with_explicit_mass(
    torch, record_property
):
    pc, neutral, packet = owner("pc"), owner("neutral_cpc"), episode()
    state = pc.snapshot_state().native_state
    raw, labels = episode_arrays(packet)
    features = native._features(pc._features(raw), 296)
    proposal = class_weighted_pc_update(
        features,
        torch.from_numpy(labels[:, 0].astype(np.int64)),
        cast(tuple[Any, Any, Any, Any], tuple(state.head_state[name] for name in NAMES)),
        class_weights=WEIGHTS,
        learning_rate=0.03,
        inference_steps=2,
        inference_learning_rate=0.04,
        role="train",
    )
    record_property("oracle_primitive_calls", 1)
    record_property("oracle_primitive_rows", 4)
    record_property("oracle_primitive_nominal_latent_row_iterations", 8)
    first = weighted.train_weighted_episode(pc, packet, WEIGHTS)
    second = weighted.train_weighted_episode(neutral, packet, WEIGHTS)
    assert first == second
    assert first.value == proposal.energy
    for name, candidate in zip(NAMES, proposal.parameters, strict=True):
        assert torch.equal(pc.snapshot_state().native_state.head_state[name], candidate)
        assert torch.equal(neutral.snapshot_state().native_state.head_state[name], candidate)
    for item in (pc, neutral):
        saved = item.snapshot_state()
        assert saved.completed_updates == 1 and saved.training_examples == 4
        assert saved.native_state.optimizer_state is None
        assert saved.native_state.head_state["traffic_steps"] == 1
    full = neutral.snapshot_state().native_state.head_state
    assert full["wake_examples"] == 4 and full["steps_since_sleep"] == 1
    assert full["energy_history"] == [proposal.legacy_diagnostic]


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
def test_should_preserve_serving_donor_rng_inputs_and_no_grad_mode(torch, method):
    learner, packet = owner(method), episode()
    before = learner.snapshot_state()
    child = learner.fork()
    inputs = pickle.dumps((packet, WEIGHTS))
    rng = (torch.get_rng_state().clone(), pickle.dumps(np.random.get_state()), random.getstate())
    with torch.no_grad():
        weighted.train_weighted_episode(child, packet, WEIGHTS)
        assert not torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
    assert_tree(torch, learner.snapshot_state(), before)
    assert child.completed_updates == 1 and child.training_examples == 4
    assert pickle.dumps((packet, WEIGHTS)) == inputs
    assert torch.equal(torch.get_rng_state(), rng[0])
    assert pickle.dumps(np.random.get_state()) == rng[1]
    assert random.getstate() == rng[2]


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
def test_should_refuse_inference_mode_before_candidate_work(torch, monkeypatch, method):
    learner = owner(method)
    before = learner.snapshot_state()
    monkeypatch.setattr(
        learner, "snapshot_state", lambda: pytest.fail("inference mode reached snapshot")
    )
    with torch.inference_mode():
        with pytest.raises(ValueError):
            weighted.train_weighted_episode(learner, episode(), WEIGHTS)
        assert torch.is_inference_mode_enabled()
    monkeypatch.undo()
    assert_tree(torch, learner.snapshot_state(), before)


@pytest.mark.parametrize(
    "weights",
    [[1.0] * 7, (1.0,) * 6, (True,) + (1.0,) * 6, (0.0,) * 7, (float("nan"),) * 7],
)
def test_should_refuse_invalid_weights_before_snapshot(monkeypatch, weights):
    learner = owner("pc")
    monkeypatch.setattr(
        learner, "snapshot_state", lambda: pytest.fail("invalid weights reached snapshot")
    )
    with pytest.raises(ValueError):
        weighted.train_weighted_episode(learner, episode(), weights)


@pytest.mark.parametrize(
    "fault",
    [
        "final_test",
        "probe_train",
        "probe_test",
        "width",
        "classes",
        "empty",
        "incomplete",
        "oversized",
        "extra",
        "projection",
        "late_next_projection",
    ],
)
def test_should_refuse_reserved_and_malformed_episodes_before_commit(
    torch, monkeypatch, fault, work_accounting
):
    learner = owner("pc")
    packet: Any = (
        episode(role=fault)
        if fault in ("final_test", "probe_train", "probe_test")
        else episode(width=148)
        if fault == "width"
        else episode(classes=8)
        if fault == "classes"
        else episode()
    )
    if fault == "empty":
        object.__setattr__(packet, "transitions", ())
    elif fault == "incomplete":
        object.__setattr__(packet.transitions[-1], "terminal", False)
    elif fault == "oversized":
        object.__setattr__(packet, "transitions", packet.transitions * 5)
    elif fault == "extra":
        object.__setattr__(packet, "unexpected", 1)
    elif fault in ("projection", "late_next_projection"):
        row = packet.transitions[-1]
        field = "observation" if fault == "projection" else "next_observation"
        values = list(getattr(row, field))
        values[-1] = 0.123
        object.__setattr__(row, field, tuple(values))
    before = learner.snapshot_state()
    monkeypatch.setattr(
        weighted,
        "class_weighted_pc_update",
        lambda *a, **k: pytest.fail("invalid source reached primitive"),
    )
    for kind in (PredictiveCodingHead, CircadianPredictiveCodingHead):
        monkeypatch.setattr(
            kind, "_apply_training_gradients", lambda *a, **k: pytest.fail("commit")
        )
    with pytest.raises(ValueError):
        weighted.train_weighted_episode(learner, packet, WEIGHTS)
    assert work_accounting["commit_attempts"] == 0
    assert_tree(torch, learner.snapshot_state(), before)


@pytest.mark.parametrize(
    "kind", ["bp", "gated_cpc", "recurrent", "raw", "foreign", "policy_mismatch"]
)
def test_should_refuse_unbound_or_unmatched_owners_before_snapshot(monkeypatch, kind):
    learner: Any = (
        owner(kind)
        if kind in ("bp", "gated_cpc")
        else TorchRecurrentLearner(TorchRecurrentPolicy(296, 8, 7, 17))
        if kind == "recurrent"
        else TorchFeatureLearner(TorchFeaturePolicy("pc", 296, 8, 17, num_classes=7))
        if kind == "raw"
        else owner("pc")
        if kind == "policy_mismatch"
        else object()
    )
    if kind == "policy_mismatch":
        learner._native._policy = replace(learner._native._policy, learning_rate=0.2)
    if hasattr(learner, "snapshot_state"):
        monkeypatch.setattr(
            learner, "snapshot_state", lambda: pytest.fail("foreign owner reached snapshot")
        )
    with pytest.raises(ValueError):
        weighted.train_weighted_episode(learner, episode(), WEIGHTS)


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
@pytest.mark.parametrize("fault", ["commit", "candidate"])
def test_should_publish_nothing_after_staged_failure(torch, monkeypatch, method, fault):
    learner, packet = owner(method), episode()
    before = learner.snapshot_state()
    payload = learner._native._payload
    primary = RuntimeError("injected staged weighted PC failure")
    committed = []

    def after(head):
        committed.append(head)
        if fault == "commit":
            raise primary

    install_after_commit(monkeypatch, after)
    original_state = native._state

    def validate(*args, **kwargs):
        if committed:
            raise primary
        return original_state(*args, **kwargs)

    if fault == "candidate":
        monkeypatch.setattr(native, "_state", validate)
    with pytest.raises(RuntimeError) as failure:
        weighted.train_weighted_episode(learner, packet, WEIGHTS)
    assert failure.value is primary and committed
    assert learner._native._payload is payload
    monkeypatch.undo()
    assert_tree(torch, learner.snapshot_state(), before)


@pytest.mark.parametrize(
    "fault",
    [
        "parameter",
        "parameter_identity",
        "gradient",
        "payload",
        "native_owner",
        "invariant_policy",
        "native_policy",
        "episode",
        "torch",
        "numpy",
        "python",
        "mode",
    ],
)
def test_should_refuse_foreign_changes_without_erasing_them(torch, monkeypatch, fault):
    learner, packet = owner("pc"), episode()
    before = learner.snapshot_state()
    rng = (torch.get_rng_state().clone(), deepcopy(np.random.get_state()), random.getstate())
    grad_mode = torch.is_grad_enabled()
    observed = []
    raw_parameters = []

    def mutate(head):
        donor = learner._native
        if fault == "parameter":
            with torch.no_grad():
                donor._head.weight_feature_hidden[0, 0] += 0.125
        elif fault == "parameter_identity":
            donor._head.weight_feature_hidden = donor._head.weight_feature_hidden.clone()
        elif fault == "gradient":
            donor._head.weight_feature_hidden.grad = torch.zeros_like(
                donor._head.weight_feature_hidden
            )
        elif fault == "payload":
            donor._payload = tuple(list(donor._payload))
        elif fault == "native_owner":
            learner._native = owner("pc")._native
        elif fault == "invariant_policy":
            learner._policy = replace(learner._policy)
        elif fault == "native_policy":
            donor._policy = replace(donor._policy)
        elif fault == "episode":
            object.__setattr__(packet, "episode_id", "externally-replaced-TRAIN")
        elif fault == "torch":
            torch.rand(1)
        elif fault == "numpy":
            np.random.random()
        elif fault == "python":
            random.random()
        elif fault == "mode":
            torch.set_grad_enabled(not grad_mode)
        raw_parameters.append(
            tuple(
                (getattr(learner._native._head, name), getattr(learner._native._head, name).grad)
                for name in NAMES
            )
        )
        observed.append(
            (
                learner.snapshot_state(),
                learner._native,
                learner._native._payload,
                learner._policy,
                learner._native._policy,
                pickle.dumps(packet),
                torch.get_rng_state().clone(),
                pickle.dumps(np.random.get_state()),
                random.getstate(),
                not grad_mode if fault == "mode" else grad_mode,
            )
        )

    if fault == "mode":
        original_stage = weighted._stage

        def stage(*args, **kwargs):
            result = original_stage(*args, **kwargs)
            mutate(args[0])
            return result

        monkeypatch.setattr(weighted, "_stage", stage)
    else:
        install_after_commit(monkeypatch, mutate)
    try:
        with pytest.raises(ValueError):
            weighted.train_weighted_episode(learner, packet, WEIGHTS)
        assert len(observed) == 1
        saved, donor, payload, policy, native_policy, source, *states = observed[0]
        assert_tree(torch, learner.snapshot_state(), saved)
        assert learner._native is donor and donor._payload is payload
        assert learner._policy is policy and donor._policy is native_policy
        assert pickle.dumps(packet) == source
        assert torch.equal(torch.get_rng_state(), states[0])
        assert pickle.dumps(np.random.get_state()) == states[1]
        assert random.getstate() == states[2] and torch.is_grad_enabled() == states[3]
        for name, (parameter, gradient) in zip(NAMES, raw_parameters[0], strict=True):
            assert getattr(donor._head, name) is parameter and parameter.grad is gradient
        assert learner.completed_updates == before.completed_updates
    finally:
        torch.set_rng_state(rng[0])
        np.random.set_state(rng[1])
        random.setstate(rng[2])
        torch.set_grad_enabled(grad_mode)
