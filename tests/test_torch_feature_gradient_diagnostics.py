"""Prior-CE math/state checks with four initial heads and zero learner updates.

Finite differences use an independent NumPy tanh/log-softmax objective. No
environment, teacher, baseline training, probe or final evaluation is executed.
"""

from copy import deepcopy
from dataclasses import replace
import json
import pickle
import random
from typing import Any, cast

import numpy as np
import pytest

from src.adapters import torch_feature_gradient_diagnostics as diagnostics
from src.adapters import torch_feature_learner as native
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeaturePolicy,
    InvariantTorchFeatureSnapshot,
)
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from src.core.minigrid_invariant_encoding import canonicalize_public_history
from src.shared.torch_runtime import require_torch
from test_minigrid_invariant_encoding import frame
from test_torch_feature_learner import assert_tree

METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")
NAMES = ("weight_feature_hidden", "bias_hidden", "weight_hidden_output", "bias_output")


def pair():
    result = []
    for identity, cue, actions in (("A-blue", 0, (6, 2)), ("A-red", 2, (2, 6))):
        first = frame(wall_color=1, direction=2, cue_color=cue)
        frames = (
            first + first,
            first + frame(wall_color=4, direction=3, cue_color=cue, door_state=0),
            first + frame(wall_color=2, direction=1, cue_color=cue, key_color=2),
        )
        result.append(
            DemonstrationEpisode(
                identity,
                "A",
                "train",
                tuple(
                    DemonstrationTransition(i, frames[i], actions[i], 0.0, frames[i + 1], i == 1, 7)
                    for i in range(2)
                ),
                7,
            )
        )
    return tuple(result)


@pytest.fixture(scope="module")
def originals():
    return {
        method: InvariantTorchFeatureLearner(
            InvariantTorchFeaturePolicy(
                native.TorchFeaturePolicy(method, 296, 8, 17, num_classes=7)
            )
        ).snapshot_state()
        for method in METHODS
    }


@pytest.fixture(autouse=True)
def zero_updates(monkeypatch, record_property):
    def forbidden(*args, **kwargs):
        pytest.fail("prior diagnostic may not train or step any optimizer")

    torch = require_torch()
    monkeypatch.setattr(native.TorchFeatureLearner, "train_batch", forbidden)
    monkeypatch.setattr(torch.optim.SGD, "step", forbidden)
    work = dict(
        pair_calls_attempted=0,
        pair_calls_completed=0,
        forward_calls_completed=0,
        backward_calls_completed=0,
        backward_rows_completed=0,
    )
    original = diagnostics.diagnose_prior_gradients

    def observe(snapshot, episodes):
        work["pair_calls_attempted"] += 1
        try:
            report = original(snapshot, episodes)
        except diagnostics.GradientDiagnosticFailed as error:
            report = error.report
            for key in (
                "forward_calls_completed",
                "backward_calls_completed",
                "backward_rows_completed",
            ):
                work[key] += report["work"][key]
            raise
        work["pair_calls_completed"] += 1
        for key in (
            "forward_calls_completed",
            "backward_calls_completed",
            "backward_rows_completed",
        ):
            work[key] += report["work"][key]
        return report

    monkeypatch.setattr(diagnostics, "diagnose_prior_gradients", observe)
    yield
    for key, value in work.items():
        record_property(key, value)
    record_property("optimizer_steps", 0)
    record_property("native_train_calls", 0)


def numpy_objective(parameters, episode):
    x = np.array([canonicalize_public_history(t.observation) for t in episode.transitions])
    targets = np.array([t.action for t in episode.transitions])
    hidden = np.tanh(x @ parameters[NAMES[0]] + parameters[NAMES[1]])
    logits = hidden @ parameters[NAMES[2]] + parameters[NAMES[3]]
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    log_probabilities = shifted - np.log(exponentials.sum(axis=1, keepdims=True))
    loss = float(-log_probabilities[np.arange(len(targets)), targets].mean())
    residual = exponentials / exponentials.sum(axis=1, keepdims=True)
    residual[np.arange(len(targets)), targets] -= 1
    residual /= len(targets)
    hidden_error = (residual @ parameters[NAMES[2]].T) * (1 - hidden * hidden)
    gradients = dict(
        zip(
            NAMES,
            (
                x.T @ hidden_error,
                hidden_error.sum(axis=0, keepdims=True),
                hidden.T @ residual,
                residual.sum(axis=0, keepdims=True),
            ),
            strict=True,
        )
    )
    return loss, gradients


def expected_geometry(first, second):
    a = np.concatenate([v.ravel() for v in first])
    b = np.concatenate([v.ravel() for v in second])
    norms = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    dot = float(a @ b)
    return dict(
        first_norm=norms[0], second_norm=norms[1], dot=dot, cosine=dot / (norms[0] * norms[1])
    )


@pytest.mark.parametrize("method", METHODS)
def test_should_match_independent_gradients_and_each_group_finite_difference(
    originals, method, monkeypatch
):
    torch = require_torch()
    state, episodes = deepcopy(originals[method]), pair()
    captured = []
    original_grad = torch.autograd.grad

    def observe(*args, **kwargs):
        gradients = original_grad(*args, **kwargs)
        captured.append(tuple(v.detach().numpy().copy() for v in gradients))
        return gradients

    monkeypatch.setattr(torch.autograd, "grad", observe)
    before, inputs = deepcopy(state), pickle.dumps(episodes)
    rng = torch.get_rng_state().clone(), pickle.dumps(np.random.get_state()), random.getstate()
    with torch.no_grad():
        report = diagnostics.diagnose_prior_gradients(state, episodes)
        assert not torch.is_grad_enabled()
    assert torch.is_grad_enabled()
    assert report["definition"] == diagnostics.DEFINITION
    assert report["objective"] == native.TORCH_CE_ID
    assert report["interpretation"] == "prior_not_native_pc_energy"
    json.dumps(report, allow_nan=False)
    parameters = {
        name: state.native_state.head_state[name].numpy().astype(np.float64) for name in NAMES
    }
    expected = []
    for episode, actual, loss in zip(episodes, captured, report["losses"], strict=True):
        objective, gradients = numpy_objective(parameters, episode)
        assert loss["mean_ce"] == pytest.approx(objective, rel=2e-6, abs=2e-7)
        expected.append(gradients)
        for index, name in enumerate(NAMES):
            np.testing.assert_allclose(actual[index], gradients[name], rtol=2e-5, atol=2e-7)
            direction = np.cos(np.arange(parameters[name].size)).reshape(parameters[name].shape)
            direction /= np.linalg.norm(direction)
            epsilon = 1e-5
            plus = dict(parameters, **{name: parameters[name] + epsilon * direction})
            minus = dict(parameters, **{name: parameters[name] - epsilon * direction})
            finite_difference = (
                numpy_objective(plus, episode)[0] - numpy_objective(minus, episode)[0]
            ) / (2 * epsilon)
            assert finite_difference == pytest.approx(
                float(np.sum(actual[index] * direction)), rel=5e-4, abs=2e-7
            )
    for names, actual in [
        (NAMES, report["joint_geometry"]),
        *(([name], report["per_parameter_geometry"][name]) for name in NAMES),
    ]:
        geometry = expected_geometry(
            [expected[0][n] for n in names], [expected[1][n] for n in names]
        )
        for key, value in geometry.items():
            assert actual[key] == pytest.approx(value, rel=2e-5, abs=2e-7)
    work = report["work"]
    assert work["snapshot_validations_completed"] == 1
    assert (
        work["projection_calls_completed"]
        == work["forward_calls_completed"]
        == work["backward_calls_completed"]
        == 2
    )
    assert (
        work["projection_rows_completed"]
        == work["forward_rows_completed"]
        == work["backward_rows_completed"]
        == 4
    )
    assert work["projection_array_bytes_returned"] == 4 * 296 * 8
    assert (
        work["validation_temporary_optimizers"]
        == work["optimizer_steps"]
        == work["published_updates"]
        == 0
    )
    assert not work["failure_work_may_be_unobserved"]
    assert_tree(torch, state, before)
    assert pickle.dumps(episodes) == inputs
    assert torch.equal(torch.get_rng_state(), rng[0])
    assert pickle.dumps(np.random.get_state()) == rng[1] and random.getstate() == rng[2]


@pytest.mark.parametrize("method", METHODS)
def test_should_preserve_existing_gradient_buffers_and_chemistry(originals, method):
    torch = require_torch()
    state = deepcopy(originals[method])
    head = state.native_state.head_state
    if method == "bp":
        for name in NAMES:
            state.native_state.gradients[name] = torch.full_like(head[name], 0.125)
    if method in ("neutral_cpc", "gated_cpc"):
        head["reward_error_ema"], head["last_reward_scale"] = 0.25, 0.75
    head[NAMES[0]].requires_grad_(True)
    head[NAMES[0]].grad = torch.full_like(head[NAMES[0]], -0.25)
    before, saved_grad = deepcopy(state), head[NAMES[0]].grad.clone()
    diagnostics.diagnose_prior_gradients(state, pair())
    assert_tree(torch, state, before)
    assert head[NAMES[0]].requires_grad
    assert torch.equal(head[NAMES[0]].grad, saved_grad)


@pytest.mark.parametrize(
    "fault", ["second_forward", "second_backward", "second_projection", "nonfinite_gradient"]
)
def test_should_retain_paid_prefix_on_failure_without_mutating_source(
    originals, monkeypatch, fault
):
    torch = require_torch()
    state, before = deepcopy(originals["bp"]), deepcopy(originals["bp"])
    if fault == "second_forward":
        name, owner = "_prior_logits", diagnostics
    elif fault in ("second_backward", "nonfinite_gradient"):
        name, owner = "grad", torch.autograd
    else:
        name, owner = "canonicalize_public_history", diagnostics
    original, calls = getattr(owner, name), 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if fault == "nonfinite_gradient":
            values = list(original(*args, **kwargs))
            values[0] = torch.full_like(values[0], float("nan"))
            return tuple(values)
        threshold = 3 if fault == "second_projection" else 2
        if calls == threshold:
            raise RuntimeError("synthetic diagnostic failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, fail)
    with pytest.raises(diagnostics.GradientDiagnosticFailed) as caught:
        diagnostics.diagnose_prior_gradients(state, pair())
    report, work = caught.value.report, caught.value.report["work"]
    assert report["status"] == "incomplete" and report["joint_geometry"] is None
    assert report["per_parameter_geometry"] is None and work["failure_work_may_be_unobserved"]
    assert work["projection_calls_attempted"] == (1 if fault == "nonfinite_gradient" else 2)
    assert work["forward_calls_attempted"] == (
        1 if fault in ("second_projection", "nonfinite_gradient") else 2
    )
    assert work["backward_calls_attempted"] == (2 if fault == "second_backward" else 1)
    assert work["backward_calls_completed"] == 1
    assert_tree(torch, state, before)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("fault", ["parameter", "gradient", "torch_rng", "numpy_rng", "python_rng"])
def test_should_refuse_observed_late_changes_without_restoring_unobserved_state(
    originals, monkeypatch, fault
):
    torch = require_torch()
    state, original = deepcopy(originals["bp"]), diagnostics._prior_logits
    saved = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
    calls = 0

    def mutate(*args):
        nonlocal calls
        result = original(*args)
        calls += 1
        if calls == 2:
            if fault == "parameter":
                with torch.no_grad():
                    state.native_state.head_state["bias_output"].add_(0.25)
            elif fault == "gradient":
                tensor = state.native_state.head_state[NAMES[0]]
                tensor.grad = torch.ones_like(tensor)
            elif fault == "torch_rng":
                torch.rand(1)
            elif fault == "numpy_rng":
                np.random.random()
            else:
                random.random()
        return result

    monkeypatch.setattr(diagnostics, "_prior_logits", mutate)
    try:
        with pytest.raises(diagnostics.GradientDiagnosticFailed, match="changed") as caught:
            diagnostics.diagnose_prior_gradients(state, pair())
        assert caught.value.report["work"]["backward_calls_completed"] == 2
        if fault == "parameter":
            assert not torch.equal(
                state.native_state.head_state["bias_output"],
                originals["bp"].native_state.head_state["bias_output"],
            )
        if fault == "gradient":
            assert state.native_state.head_state[NAMES[0]].grad is not None
    finally:
        torch.set_rng_state(saved[0])
        np.random.set_state(saved[1])
        random.setstate(saved[2])


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "one",
        "duplicate",
        "context",
        "final_test",
        "probe_train",
        "probe_test",
        "extra",
        "transition_extra",
        "overlong",
        "width",
        "classes",
        "foreign_snapshot",
        "foreign_policy",
    ],
)
def test_should_refuse_metadata_before_optional_native_work(monkeypatch, fault):
    policy = InvariantTorchFeaturePolicy(native.TorchFeaturePolicy("bp", 296, 8, 17, num_classes=7))
    state: Any = InvariantTorchFeatureSnapshot(policy, cast(Any, None))
    episodes: Any = pair()
    if fault == "list":
        episodes = list(episodes)
    elif fault == "one":
        episodes = episodes[:1]
    elif fault == "duplicate":
        episodes = (episodes[0], replace(episodes[1], episode_id=episodes[0].episode_id))
    elif fault == "context":
        episodes = (episodes[0], replace(episodes[1], context="B"))
    elif fault in ("final_test", "probe_train", "probe_test"):
        episodes = (replace(episodes[0], role=fault), episodes[1])
    elif fault == "extra":
        object.__setattr__(episodes[0], "extra", 1)
    elif fault == "transition_extra":
        object.__setattr__(episodes[0].transitions[0], "extra", 1)
    elif fault == "overlong":
        object.__setattr__(episodes[0], "transitions", episodes[0].transitions * 65)
    elif fault == "width":
        object.__setattr__(episodes[0].transitions[0], "observation", (0.0,) * 148)
    elif fault == "classes":
        object.__setattr__(episodes[0], "num_classes", 2)
    elif fault == "foreign_snapshot":
        state = dict(vars(state))
    else:
        object.__setattr__(state, "policy", dict(vars(policy)))

    def forbidden(*args, **kwargs):
        pytest.fail("malformed metadata must fail before Torch/native copy")

    monkeypatch.setattr(diagnostics, "require_torch", forbidden)
    monkeypatch.setattr(diagnostics, "copy_invariant_torch_feature_snapshot", forbidden)
    with pytest.raises(ValueError):
        diagnostics.diagnose_prior_gradients(state, episodes)


def test_should_refuse_inference_mode_before_native_snapshot_work(originals, monkeypatch):
    torch = require_torch()

    def forbidden(*args):
        pytest.fail("inference mode must fail before snapshot work")

    monkeypatch.setattr(diagnostics, "copy_invariant_torch_feature_snapshot", forbidden)
    with torch.inference_mode(), pytest.raises(ValueError, match="inference_mode"):
        diagnostics.diagnose_prior_gradients(originals["bp"], pair())


def test_should_retain_snapshot_validation_refusal_without_copying_foreign_payload(originals):
    state = deepcopy(originals["bp"])

    class Foreign:
        def __deepcopy__(self, memo):
            pytest.fail("foreign nested payload must not be copied")

    state.native_state.head_state[NAMES[0]] = Foreign()
    with pytest.raises(diagnostics.GradientDiagnosticFailed) as caught:
        diagnostics.diagnose_prior_gradients(state, pair())
    work = caught.value.report["work"]
    assert (
        work["snapshot_validations_attempted"] == 1 and work["snapshot_validations_completed"] == 0
    )
    assert work["forward_calls_attempted"] == work["projection_calls_attempted"] == 0
