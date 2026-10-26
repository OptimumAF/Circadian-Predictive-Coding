"""Independent weighted feedforward CE/SGD checks without environment physics.

Every attempted update, including rejected calls and ordinary references, is
recorded per test. This file is an engineering gate, not a comparative run.
"""

from copy import deepcopy
from dataclasses import replace
import pickle
import random
from typing import Any

import numpy as np
import pytest

from src.adapters import torch_feature_weighted as weighted
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeaturePolicy,
)
from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy
from src.core.demonstration_class_weights import episode_class_weights
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    episode_arrays,
)
from src.core.minigrid_invariant_encoding import canonicalize_public_history
from src.shared.torch_runtime import require_torch
from test_minigrid_invariant_encoding import frame
from test_torch_feature_learner import NAMES, assert_tree, tensors
from test_torch_recurrent_weighted import assert_close_state


@pytest.fixture(scope="module")
def torch():
    return require_torch()


@pytest.fixture(autouse=True)
def update_accounting(monkeypatch, record_property, torch):
    work = dict(train_attempts=0, train_rows=0, sgd_steps=0)
    original_weighted = weighted.train_weighted_episode
    original_ordinary = TorchFeatureLearner.train_batch
    original_step = torch.optim.SGD.step

    def invoke(learner, packet, weights):
        work["train_attempts"] += 1
        work["train_rows"] += len(packet.transitions) if hasattr(packet, "transitions") else 0
        return original_weighted(learner, packet, weights)

    def ordinary(learner, features, targets):
        work["train_attempts"] += 1
        work["train_rows"] += len(features)
        return original_ordinary(learner, features, targets)

    def step(*args, **kwargs):
        work["sgd_steps"] += 1
        return original_step(*args, **kwargs)

    monkeypatch.setattr(weighted, "train_weighted_episode", invoke)
    monkeypatch.setattr(TorchFeatureLearner, "train_batch", ordinary)
    monkeypatch.setattr(torch.optim.SGD, "step", step)
    yield work
    for key, value in work.items():
        record_property(key, value)


def owner(method="bp"):
    return InvariantTorchFeatureLearner(
        InvariantTorchFeaturePolicy(TorchFeaturePolicy(method, 296, 8, 17, num_classes=7))
    )


def episode(*, actions=(0, 0, 2, 6), role="train", width=296, classes=7):
    initial = frame(wall_color=1, direction=2, cue_color=2, key_color=2)
    observations = [
        (initial + frame(wall_color=index % 6, direction=index % 4, door_state=index % 3))[:width]
        for index in range(len(actions) + 1)
    ]
    return DemonstrationEpisode(
        "weighted-feature-TRAIN",
        "A",
        role,
        tuple(
            DemonstrationTransition(
                index,
                observations[index],
                action,
                float(index == len(actions) - 1),
                observations[index + 1],
                index == len(actions) - 1,
                classes,
            )
            for index, action in enumerate(actions)
        ),
        classes,
    )


WEIGHTS = (0.25, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


def numpy_objective_and_gradients(parameters, packet, weights):
    """Derive both layers directly; no production forward/autograd is called."""
    values = {name: value.detach().numpy().astype(np.float64) for name, value in parameters.items()}
    raw, target = episode_arrays(packet)
    features = np.array(
        [canonicalize_public_history(tuple(float(value) for value in row)) for row in raw]
    )
    labels = target[:, 0].astype(np.int64)
    hidden = np.tanh(features @ values[NAMES[0]] + values[NAMES[1]])
    logits = hidden @ values[NAMES[2]] + values[NAMES[3]]
    mass = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = mass / mass.sum(axis=1, keepdims=True)
    coefficients = np.array(weights)[labels]
    coefficients = coefficients / coefficients.sum()
    loss = -np.sum(coefficients * np.log(probabilities[np.arange(len(labels)), labels]))
    delta = probabilities.copy()
    delta[np.arange(len(labels)), labels] -= 1
    delta *= coefficients[:, None]
    d_hidden = (delta @ values[NAMES[2]].T) * (1 - hidden**2)
    gradients = {
        NAMES[0]: features.T @ d_hidden,
        NAMES[1]: d_hidden.sum(axis=0, keepdims=True),
        NAMES[2]: hidden.T @ delta,
        NAMES[3]: delta.sum(axis=0, keepdims=True),
    }
    return float(loss), gradients


@pytest.mark.parametrize("warm_momentum", [False, True])
def test_should_match_independent_weighted_ce_all_gradients_and_momentum(torch, warm_momentum):
    learner, packet = owner(), episode()
    if warm_momentum:
        learner.train_batch(*episode_arrays(packet))
    before = learner.snapshot_state()
    state = before.native_state
    parameters = {name: state.head_state[name] for name in NAMES}
    expected_loss, gradients = numpy_objective_and_gradients(parameters, packet, WEIGHTS)
    # Independent central differences in every parameter family validate the
    # analytic oracle before comparing it with the production float32 update.
    for name in NAMES:
        position = tuple(0 for _ in parameters[name].shape)
        epsilon = 1e-3
        plus, minus = deepcopy(parameters), deepcopy(parameters)
        plus[name][position] += epsilon
        minus[name][position] -= epsilon
        numerator = numpy_objective_and_gradients(plus, packet, WEIGHTS)[0]
        numerator -= numpy_objective_and_gradients(minus, packet, WEIGHTS)[0]
        actual_step = float(plus[name][position] - minus[name][position])
        assert numerator / actual_step == pytest.approx(
            gradients[name][position], rel=2e-4, abs=2e-6
        )
    diagnostic = weighted.train_weighted_episode(learner, packet, WEIGHTS)
    after = learner.snapshot_state()
    assert diagnostic.definition == weighted.TORCH_INVARIANT_WEIGHTED_CE_ID
    assert diagnostic.value == pytest.approx(expected_loss, rel=2e-6, abs=2e-7)
    assert after.policy == before.policy
    assert after.completed_updates == before.completed_updates + 1
    assert after.training_examples == before.training_examples + len(packet.transitions)
    for index, name in enumerate(NAMES):
        momentum = gradients[name].copy()
        if warm_momentum:
            momentum += 0.9 * state.optimizer_state["state"][index]["momentum_buffer"].numpy()
        np.testing.assert_allclose(
            after.native_state.gradients[name].numpy(), gradients[name], rtol=3e-5, atol=3e-7
        )
        np.testing.assert_allclose(
            after.native_state.optimizer_state["state"][index]["momentum_buffer"].numpy(),
            momentum,
            rtol=3e-5,
            atol=3e-7,
        )
        np.testing.assert_allclose(
            after.native_state.head_state[name].numpy(),
            state.head_state[name].numpy() - 0.03 * momentum,
            rtol=3e-6,
            atol=1e-7,
        )


def test_should_match_unweighted_bp_through_multiple_updates_fork_and_restore(torch):
    learner, reference, packet = owner(), owner(), episode()
    ones = (1.0,) * 7
    for _ in range(3):
        actual = weighted.train_weighted_episode(learner, packet, ones)
        expected = reference.train_batch(*episode_arrays(packet))
        assert actual.value == pytest.approx(expected.value, abs=3e-7)
        assert actual.definition != expected.definition
        assert_close_state(torch, learner.snapshot_state(), reference.snapshot_state())
    saved = learner.snapshot_state()
    child, restored = learner.fork(), owner()
    restored.restore_state(saved)
    for item in (learner, child, restored):
        weighted.train_weighted_episode(item, packet, WEIGHTS)
    expected = learner.snapshot_state()
    assert_tree(torch, child.snapshot_state(), expected)
    assert_tree(torch, restored.snapshot_state(), expected)
    assert expected.completed_updates == 4 and expected.training_examples == 16
    assert all(
        a.data_ptr() != b.data_ptr()
        for a, b in zip(tensors(torch, saved), tensors(torch, expected), strict=True)
    )
    for value in tensors(torch, saved):
        value.zero_()
    assert_tree(torch, learner.snapshot_state(), expected)


def test_should_ignore_absent_class_weights_and_common_scale(torch):
    left, right, scaled, packet = owner(), owner(), owner(), episode()
    changed = (0.25, 999.0, 2.0, 888.0, 777.0, 666.0, 6.0)
    first = weighted.train_weighted_episode(left, packet, WEIGHTS)
    second = weighted.train_weighted_episode(right, packet, changed)
    third = weighted.train_weighted_episode(scaled, packet, tuple(value * 2 for value in WEIGHTS))
    assert first == second
    assert first.value == pytest.approx(third.value, abs=3e-7)
    assert_tree(torch, left.snapshot_state(), right.snapshot_state())
    assert_close_state(torch, left.snapshot_state(), scaled.snapshot_state())


def test_should_preserve_rng_inputs_policy_saved_state_and_serving_donor(torch):
    learner, packet = owner(), episode()
    saved = learner.snapshot_state()
    child = learner.fork()
    original = pickle.dumps((packet, WEIGHTS))
    rng = (torch.get_rng_state().clone(), pickle.dumps(np.random.get_state()), random.getstate())
    mode = torch.is_grad_enabled()
    weighted.train_weighted_episode(child, packet, WEIGHTS)
    assert_tree(torch, learner.snapshot_state(), saved)
    assert child.policy == learner.policy == saved.policy
    assert pickle.dumps((packet, WEIGHTS)) == original
    assert torch.equal(torch.get_rng_state(), rng[0])
    assert pickle.dumps(np.random.get_state()) == rng[1]
    assert random.getstate() == rng[2] and torch.is_grad_enabled() == mode


@pytest.mark.parametrize(
    "weights",
    [
        [1.0] * 7,
        (1.0,) * 6,
        (True,) + (1.0,) * 6,
        (0.0,) + (1.0,) * 6,
        (float("nan"),) + (1.0,) * 6,
    ],
)
def test_should_refuse_invalid_weights_before_snapshot(monkeypatch, weights):
    learner = owner()
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
        "incomplete",
        "oversized",
        "extra",
        "projection",
        "late_next_projection",
    ],
)
def test_should_refuse_reserved_or_malformed_episodes_before_native_update(
    torch, monkeypatch, fault
):
    learner = owner()
    packet: Any = (
        episode(role=fault)
        if fault in ("final_test", "probe_train", "probe_test")
        else episode(width=148)
        if fault == "width"
        else episode(classes=8)
        if fault == "classes"
        else episode()
    )
    if fault == "incomplete":
        object.__setattr__(packet.transitions[-1], "terminal", False)
    elif fault == "oversized":
        object.__setattr__(packet, "transitions", packet.transitions * 5)
    elif fault == "extra":
        object.__setattr__(packet, "unknown", 1)
    elif fault in ("projection", "late_next_projection"):
        step = packet.transitions[-1]
        field = "observation" if fault == "projection" else "next_observation"
        values = list(getattr(step, field))
        values[-1] = 0.123  # Finite normalized scalar, invalid public direction.
        object.__setattr__(step, field, tuple(values))
    before = learner.snapshot_state()
    monkeypatch.setattr(
        torch.optim.SGD, "step", lambda *a, **kw: pytest.fail("invalid episode reached SGD")
    )
    with pytest.raises(ValueError):
        weighted.train_weighted_episode(learner, packet, (1.0,) * 7)
    assert_tree(torch, learner.snapshot_state(), before)


@pytest.mark.parametrize("kind", ["pc", "neutral_cpc", "gated_cpc", "raw", "foreign"])
def test_should_refuse_non_bp_or_unbound_owners_before_native_work(monkeypatch, kind):
    learner: Any = (
        owner(kind)
        if kind in ("pc", "neutral_cpc", "gated_cpc")
        else TorchFeatureLearner(TorchFeaturePolicy("bp", 296, 8, 17, num_classes=7))
        if kind == "raw"
        else object()
    )
    if hasattr(learner, "snapshot_state"):
        monkeypatch.setattr(
            learner, "snapshot_state", lambda: pytest.fail("foreign owner reached snapshot")
        )
    with pytest.raises(ValueError):
        weighted.train_weighted_episode(learner, episode(), WEIGHTS)


@pytest.mark.parametrize("mode", ["no_grad", "inference_mode"])
def test_should_refuse_disabled_autograd_without_changing_mode_or_state(torch, mode):
    learner, packet = owner(), episode()
    before = learner.snapshot_state()
    with getattr(torch, mode)():
        with pytest.raises(ValueError):
            weighted.train_weighted_episode(learner, packet, WEIGHTS)
        assert not torch.is_grad_enabled()
    assert torch.is_grad_enabled()
    assert_tree(torch, learner.snapshot_state(), before)


@pytest.mark.parametrize("fault", ["loss", "backward", "step", "nonfinite", "candidate"])
def test_should_publish_nothing_after_staged_native_failure(torch, monkeypatch, fault):
    learner, packet = owner(), episode()
    learner.train_batch(*episode_arrays(packet))
    before = learner.snapshot_state()
    original_step = torch.optim.SGD.step
    primary = RuntimeError("injected weighted candidate failure")

    def fail(*args, **kwargs):
        raise primary

    def corrupt(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        parameter = optimizer.param_groups[0]["params"][0]
        with torch.no_grad():
            if fault == "nonfinite":
                parameter[0, 0] = float("nan")
            elif fault == "candidate":
                optimizer.param_groups[0]["lr"] = 0.2
        if fault == "step":
            raise primary
        return result

    if fault == "loss":
        monkeypatch.setattr(torch.nn.functional, "cross_entropy", fail)
    elif fault == "backward":
        monkeypatch.setattr(torch.Tensor, "backward", fail)
    else:
        monkeypatch.setattr(torch.optim.SGD, "step", corrupt)
    with pytest.raises((RuntimeError, ValueError)) as failure:
        weighted.train_weighted_episode(learner, packet, WEIGHTS)
    if fault in ("loss", "backward", "step"):
        assert failure.value is primary
    assert_tree(torch, learner.snapshot_state(), before)


def test_should_refuse_donor_mutation_and_preserve_the_external_change(torch, monkeypatch):
    learner, packet = owner(), episode()
    before = learner.snapshot_state()
    original_step = torch.optim.SGD.step

    def replace_donor(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        parameter = learner._native._head.weight_feature_hidden
        with torch.no_grad():
            parameter[0, 0] += 0.125
        return result

    monkeypatch.setattr(torch.optim.SGD, "step", replace_donor)
    with pytest.raises(ValueError):
        weighted.train_weighted_episode(learner, packet, WEIGHTS)
    changed = learner.snapshot_state()
    expected_head = deepcopy(before.native_state.head_state)
    expected_head[NAMES[0]][0, 0] += 0.125
    expected = replace(before, native_state=replace(before.native_state, head_state=expected_head))
    assert_tree(torch, changed, expected)


@pytest.mark.parametrize("kind", ["torch", "numpy", "python"])
def test_should_refuse_rng_changes_without_erasing_foreign_draws(torch, monkeypatch, kind):
    learner, packet = owner(), episode()
    before = learner.snapshot_state()
    rng = (torch.get_rng_state().clone(), deepcopy(np.random.get_state()), random.getstate())
    original_step = torch.optim.SGD.step
    observed = []

    def draw(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        if kind == "torch":
            torch.rand(1)
            observed.append(torch.get_rng_state().clone())
        elif kind == "numpy":
            np.random.random()
            observed.append(pickle.dumps(np.random.get_state()))
        else:
            random.random()
            observed.append(random.getstate())
        return result

    monkeypatch.setattr(torch.optim.SGD, "step", draw)
    try:
        with pytest.raises(ValueError, match="RNG"):
            weighted.train_weighted_episode(learner, packet, WEIGHTS)
        assert_tree(torch, learner.snapshot_state(), before)
        if kind == "torch":
            assert torch.equal(torch.get_rng_state(), observed[0])
        elif kind == "numpy":
            assert pickle.dumps(np.random.get_state()) == observed[0]
        else:
            assert random.getstate() == observed[0]
    finally:
        torch.set_rng_state(rng[0])
        np.random.set_state(rng[1])
        random.setstate(rng[2])


def test_should_use_existing_train_episode_class_mass_without_weight_tuning(torch):
    learner, packet = owner(), episode()
    weights = episode_class_weights(packet)
    assert weights == (2 / 3, 1.0, 4 / 3, 1.0, 1.0, 1.0, 4 / 3)
    before = learner.snapshot_state()
    parameters = {name: before.native_state.head_state[name] for name in NAMES}
    expected, _ = numpy_objective_and_gradients(parameters, packet, weights)
    diagnostic = weighted.train_weighted_episode(learner, packet, weights)
    assert diagnostic.definition == weighted.TORCH_INVARIANT_WEIGHTED_CE_ID
    assert diagnostic.value == pytest.approx(expected, rel=2e-6, abs=2e-7)
