"""Independent local-PC arithmetic, bounded admission and detached-state checks.

Synthetic complete public frames only: no environment, optimizer, learner update,
probe or final evaluation. NumPy implements the local equations independently.
"""

from copy import deepcopy
from dataclasses import replace
import json
import pickle
import random
from typing import Any, cast

import numpy as np
import pytest

from src.adapters import torch_feature_learner as native
from src.adapters import torch_pc_update_diagnostics as diagnostics
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeaturePolicy,
    InvariantTorchFeatureSnapshot,
)
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from src.core.minigrid_invariant_encoding import canonicalize_public_history
from src.shared.torch_runtime import require_torch
from test_minigrid_invariant_encoding import frame
from test_torch_feature_gradient_diagnostics import pair
from test_torch_feature_learner import assert_tree

NAMES = ("weight_feature_hidden", "bias_hidden", "weight_hidden_output", "bias_output")
METHODS = ("pc", "neutral_cpc")


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
def prohibit_published_updates(monkeypatch, record_property):
    def forbidden(*args, **kwargs):
        pytest.fail("detached readout may not call learner training or an optimizer")

    monkeypatch.setattr(native.TorchFeatureLearner, "train_batch", forbidden)
    monkeypatch.setattr(require_torch().optim.SGD, "step", forbidden)
    yield
    record_property("native_train_calls", 0)
    record_property("optimizer_steps", 0)


def _arrays(episodes):
    rows = [step for episode in episodes for step in episode.transitions]
    x = np.array([canonicalize_public_history(row.observation) for row in rows], np.float32)
    return x, np.array([row.action for row in rows])


def _probabilities(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _ce(logits, targets):
    values = logits.astype(np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    logs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-logs[np.arange(len(targets)), targets].mean())


def _geometry(left, right):
    a = np.concatenate([value.ravel().astype(np.float64) for value in left])
    b = np.concatenate([value.ravel().astype(np.float64) for value in right])
    first, second = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    dot = float(a @ b)
    return dict(first_norm=first, second_norm=second, dot=dot, cosine=dot / (first * second))


def _oracle(state, episodes):
    """Native float32 local dynamics; independent float64 prior derivative."""
    policy = state.policy.native_policy
    x, y = _arrays(episodes)
    p = {name: state.native_state.head_state[name].numpy().copy() for name in NAMES}
    hidden = np.tanh(x @ p[NAMES[0]] + p[NAMES[1]])
    z = hidden.copy()
    one_hot = np.eye(7, dtype=np.float32)[y]
    points = []
    for step in range(policy.inference_steps + 1):
        logits = z @ p[NAMES[2]] + p[NAMES[3]]
        output_error = _probabilities(logits) - one_hot
        residual = z - hidden
        gradient = residual + output_error @ p[NAMES[2]].T
        ce = _ce(logits, y)
        quadratic = float(0.5 * np.square(residual.astype(np.float64)).sum(axis=1).mean())
        points.append(
            dict(
                step=step,
                mean_ce=ce,
                hidden_quadratic=quadratic,
                objective=ce + quadratic,
                latent_gradient_l2=float(np.linalg.norm(gradient.astype(np.float64))),
                latent_displacement_l2=float(np.linalg.norm(residual.astype(np.float64))),
            )
        )
        if step < policy.inference_steps:
            z = z - policy.inference_learning_rate * gradient
    hidden_prior_gradient = -residual * (1 - hidden * hidden)
    manual = (
        (x.T @ hidden_prior_gradient) / len(y),
        hidden_prior_gradient.mean(axis=0, keepdims=True),
        (z.T @ output_error) / len(y),
        output_error.mean(axis=0, keepdims=True),
    )
    candidate = {
        name: p[name] - policy.learning_rate * grad
        for name, grad in zip(NAMES, manual, strict=True)
    }
    delta = tuple(candidate[name] - p[name] for name in NAMES)
    # This derivative deliberately does not use autograd or production helpers.
    double = {name: value.astype(np.float64) for name, value in p.items()}
    xd = x.astype(np.float64)
    prior_hidden = np.tanh(xd @ double[NAMES[0]] + double[NAMES[1]])
    prior_logits = prior_hidden @ double[NAMES[2]] + double[NAMES[3]]
    prior_error = (_probabilities(prior_logits) - one_hot) / len(y)
    first_error = (prior_error @ double[NAMES[2]].T) * (1 - prior_hidden * prior_hidden)
    prior_gradients = (
        xd.T @ first_error,
        first_error.sum(axis=0, keepdims=True),
        prior_hidden.T @ prior_error,
        prior_error.sum(axis=0, keepdims=True),
    )
    counter_hidden = np.tanh(x @ candidate[NAMES[0]] + candidate[NAMES[1]])
    counter_logits = counter_hidden @ candidate[NAMES[2]] + candidate[NAMES[3]]
    legacy = float(0.5 * (np.square(output_error).mean() + np.square(residual).mean()))
    return points, prior_gradients, delta, _ce(prior_logits, y), _ce(counter_logits, y), legacy


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("steps", [1, 2, 4])
def test_should_match_independent_local_equations_and_actual_delta_geometry(method, steps):
    state = InvariantTorchFeatureLearner(
        InvariantTorchFeaturePolicy(
            native.TorchFeaturePolicy(method, 296, 8, 17, inference_steps=steps, num_classes=7)
        )
    ).snapshot_state()
    episodes = pair()
    points, gradients, delta, prior, counter, legacy = _oracle(state, episodes)
    report = diagnostics.diagnose_pc_update(state, episodes)
    assert report["status"] == "complete" and report["reason"] is None
    assert report["definition"] == diagnostics.DEFINITION
    assert report["method"] == method
    assert report["rows"] == 4 and report["inference_steps"] == steps
    assert report["source_episode_ids"] == [episode.episode_id for episode in episodes]
    assert len(report["relaxation"]) == steps + 1
    for actual, expected in zip(report["relaxation"], points, strict=True):
        for key, value in expected.items():
            assert actual[key] == pytest.approx(value, rel=3e-5, abs=3e-7)
    assert report["prior_mean_ce"] == pytest.approx(prior, rel=2e-6, abs=2e-7)
    assert report["counterfactual_prior_mean_ce"] == pytest.approx(counter, rel=2e-6, abs=2e-7)
    assert report["prior_ce_change"] == pytest.approx(counter - prior, abs=4e-7)
    assert report["legacy_diagnostic"]["definition"] == native.TORCH_PC_ENERGY_ID
    assert report["legacy_diagnostic"]["value"] == pytest.approx(legacy, rel=2e-6)
    assert legacy != pytest.approx(points[-1]["objective"], rel=0.1)
    for actual, expected in [
        (report["gradient_delta_geometry"], _geometry(gradients, delta)),
        *(
            (report["per_parameter_geometry"][name], _geometry((gradients[i],), (delta[i],)))
            for i, name in enumerate(NAMES)
        ),
    ]:
        for key, value in expected.items():
            assert actual[key] == pytest.approx(value, rel=3e-4, abs=3e-7)
    # Sign is the derivative along the actual step, not along its negation.
    assert report["gradient_delta_geometry"]["dot"] < 0
    assert report["prior_ce_change"] < 0
    work = report["work"]
    assert work["snapshot_validations_completed"] == 1
    assert work["projection_calls_completed"] == 1
    assert work["projection_rows_completed"] == 4
    assert work["projection_array_bytes_returned"] == 4 * 296 * 8
    assert work["prior_forward_calls_completed"] == 2
    assert work["prior_forward_rows_completed"] == 8
    assert work["prior_backward_calls_completed"] == 1
    assert work["prior_backward_rows_completed"] == 4
    assert work["relaxation_forward_calls_completed"] == steps + 1
    assert work["relaxation_forward_rows_completed"] == 4 * (steps + 1)
    assert work["candidate_parameter_sets_completed"] == 1
    assert work["native_train_calls"] == work["optimizer_steps"] == work["published_updates"] == 0
    assert work["validation_temporary_optimizers"] == 0
    assert not work["failure_work_may_be_unobserved"]
    json.dumps(report, allow_nan=False)


def test_should_match_neutral_control_without_changing_method_laws(originals):
    first = diagnostics.diagnose_pc_update(originals["pc"], pair())
    second = diagnostics.diagnose_pc_update(originals["neutral_cpc"], pair())
    for key in (
        "relaxation",
        "legacy_diagnostic",
        "prior_mean_ce",
        "counterfactual_prior_mean_ce",
        "prior_ce_change",
        "gradient_delta_geometry",
        "per_parameter_geometry",
    ):
        assert first[key] == second[key]


@pytest.mark.parametrize("method", METHODS)
def test_should_preserve_full_state_gradient_marks_rng_and_no_grad_mode(originals, method):
    torch = require_torch()
    state = deepcopy(originals[method])
    head = state.native_state.head_state
    if method == "neutral_cpc":
        head["reward_error_ema"], head["last_reward_scale"] = 0.25, 0.75
    head[NAMES[0]].requires_grad_(True)
    head[NAMES[0]].grad = torch.full_like(head[NAMES[0]], -0.25)
    before, gradient = deepcopy(state), head[NAMES[0]].grad.clone()
    episodes = pair()
    inputs = pickle.dumps(episodes)
    rng = torch.get_rng_state().clone(), pickle.dumps(np.random.get_state()), random.getstate()
    with torch.no_grad():
        report = diagnostics.diagnose_pc_update(state, episodes)
        assert not torch.is_grad_enabled()
    assert torch.is_grad_enabled()
    assert_tree(torch, state, before)
    assert head[NAMES[0]].requires_grad
    assert torch.equal(head[NAMES[0]].grad, gradient)
    assert pickle.dumps(episodes) == inputs
    assert torch.equal(torch.get_rng_state(), rng[0])
    assert pickle.dumps(np.random.get_state()) == rng[1] and random.getstate() == rng[2]
    assert report["work"]["optimizer_steps"] == report["work"]["published_updates"] == 0


def _episode(identity, count):
    first = frame()
    observation = first + first
    return DemonstrationEpisode(
        identity,
        identity,
        "train",
        tuple(
            DemonstrationTransition(i, observation, i % 7, 0.0, observation, i == count - 1, 7)
            for i in range(count)
        ),
        7,
    )


def test_should_accept_six_complete_contexts_at_exact_fifty_four_row_limit(originals):
    episodes = tuple(_episode(f"context-{i}", 9) for i in range(6))
    report = diagnostics.diagnose_pc_update(originals["pc"], episodes)
    assert report["rows"] == 54 and report["status"] == "complete"
    assert report["source_episode_ids"] == [episode.episode_id for episode in episodes]


def test_should_accept_one_complete_train_episode(originals):
    report = diagnostics.diagnose_pc_update(originals["pc"], pair()[:1])
    assert report["rows"] == 2 and report["status"] == "complete"


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "empty",
        "seven",
        "duplicate",
        "fifty_five",
        "final_test",
        "probe_train",
        "probe_test",
        "extra",
        "transition_extra",
        "overlong",
        "width",
        "classes",
        "nan",
        "broken_continuity",
        "foreign_snapshot",
        "foreign_policy",
        "bp",
        "gated_cpc",
    ],
)
def test_should_refuse_bad_admission_before_snapshot_copy_or_projection(monkeypatch, fault):
    policy = InvariantTorchFeaturePolicy(native.TorchFeaturePolicy("pc", 296, 8, 17, num_classes=7))
    state: Any = InvariantTorchFeatureSnapshot(policy, cast(Any, None))
    episodes: Any = pair()
    if fault == "list":
        episodes = list(episodes)
    elif fault == "empty":
        episodes = ()
    elif fault == "seven":
        episodes = tuple(_episode(str(i), 1) for i in range(7))
    elif fault == "duplicate":
        episodes = (episodes[0], replace(episodes[1], episode_id=episodes[0].episode_id))
    elif fault == "fifty_five":
        episodes = tuple(_episode(str(i), count) for i, count in enumerate((16, 16, 16, 7)))
    elif fault in ("final_test", "probe_train", "probe_test"):
        episodes = (replace(episodes[0], role=fault), episodes[1])
    elif fault == "extra":
        object.__setattr__(episodes[0], "extra", 1)
    elif fault == "transition_extra":
        object.__setattr__(episodes[0].transitions[0], "extra", 1)
    elif fault == "overlong":
        object.__setattr__(episodes[0], "transitions", episodes[0].transitions * 9)
    elif fault == "width":
        object.__setattr__(episodes[0].transitions[0], "observation", (0.0,) * 148)
    elif fault == "classes":
        object.__setattr__(episodes[0], "num_classes", 2)
    elif fault == "nan":
        object.__setattr__(episodes[0].transitions[0], "reward", float("nan"))
    elif fault == "broken_continuity":
        object.__setattr__(
            episodes[0].transitions[0], "next_observation", episodes[1].transitions[0].observation
        )
    elif fault == "foreign_snapshot":
        state = dict(vars(state))
    elif fault == "foreign_policy":
        object.__setattr__(state, "policy", dict(vars(policy)))
    elif fault in ("bp", "gated_cpc"):
        object.__setattr__(
            state,
            "policy",
            InvariantTorchFeaturePolicy(
                native.TorchFeaturePolicy(fault, 296, 8, 17, num_classes=7)
            ),
        )

    def forbidden(*args, **kwargs):
        pytest.fail("bad admission must fail before native snapshot copying/projection")

    monkeypatch.setattr(diagnostics, "copy_invariant_torch_feature_snapshot", forbidden)
    monkeypatch.setattr(diagnostics, "canonicalize_public_history", forbidden)
    with pytest.raises(ValueError):
        diagnostics.diagnose_pc_update(state, episodes)


def test_should_refuse_inference_mode_before_snapshot_copy(originals, monkeypatch):
    def forbidden(*args):
        pytest.fail("inference mode must fail before snapshot copying")

    torch = require_torch()
    monkeypatch.setattr(diagnostics, "copy_invariant_torch_feature_snapshot", forbidden)
    with torch.inference_mode(), pytest.raises(ValueError, match="inference_mode"):
        diagnostics.diagnose_pc_update(originals["pc"], pair())


@pytest.mark.parametrize("fault", ["nonfinite", "foreign_payload"])
def test_should_report_failed_snapshot_validation_without_copying_foreign_state(originals, fault):
    state = deepcopy(originals["pc"])

    class Foreign:
        def __deepcopy__(self, memo):
            pytest.fail("unknown nested payload must not be copied")

    state.native_state.head_state[NAMES[0]] = (
        require_torch().full_like(state.native_state.head_state[NAMES[0]], float("nan"))
        if fault == "nonfinite"
        else Foreign()
    )
    with pytest.raises(diagnostics.GradientDiagnosticFailed) as caught:
        diagnostics.diagnose_pc_update(state, pair())
    report = caught.value.report
    assert report["status"] == "incomplete"
    assert report["work"]["snapshot_validations_attempted"] == 1
    assert report["work"]["snapshot_validations_completed"] == 0
    assert report["gradient_delta_geometry"] is None
    json.dumps(report, allow_nan=False)


def test_should_retain_completed_projection_work_when_prior_backward_fails(originals, monkeypatch):
    torch = require_torch()
    state = deepcopy(originals["pc"])
    before = deepcopy(state)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic prior backward failure")

    monkeypatch.setattr(torch.autograd, "grad", fail)
    with pytest.raises(diagnostics.GradientDiagnosticFailed) as caught:
        diagnostics.diagnose_pc_update(state, pair())
    report = caught.value.report
    assert report["status"] == "incomplete" and "synthetic" in report["reason"]
    assert report["work"]["projection_rows_completed"] == 4
    assert report["work"]["failure_work_may_be_unobserved"]
    assert report["gradient_delta_geometry"] is None
    assert_tree(torch, state, before)
    json.dumps(report, allow_nan=False)


def test_should_preserve_observed_relaxation_prefix_on_later_failure(originals, monkeypatch):
    original = diagnostics._relax
    state = deepcopy(originals["pc"])
    before = deepcopy(state)

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("synthetic completed relaxation failure")

    monkeypatch.setattr(diagnostics, "_relax", fail)
    with pytest.raises(diagnostics.GradientDiagnosticFailed) as caught:
        diagnostics.diagnose_pc_update(state, pair())
    report = caught.value.report
    assert len(report["relaxation"]) == 3
    assert report["work"]["relaxation_forward_calls_completed"] == 3
    assert report["work"]["relaxation_forward_rows_completed"] == 12
    assert report["work"]["candidate_parameter_sets_completed"] == 0
    assert report["counterfactual_prior_mean_ce"] is None
    assert_tree(require_torch(), state, before)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("fault", ["parameter", "gradient", "torch_rng", "numpy_rng", "python_rng"])
def test_should_refuse_late_observed_changes_without_reverting_them(originals, monkeypatch, fault):
    torch = require_torch()
    state = deepcopy(originals["pc"])
    original = diagnostics._measure
    saved = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        if fault == "parameter":
            with torch.no_grad():
                state.native_state.head_state["bias_output"].add_(0.25)
        elif fault == "gradient":
            value = state.native_state.head_state[NAMES[0]]
            value.grad = torch.ones_like(value)
        elif fault == "torch_rng":
            torch.rand(1)
        elif fault == "numpy_rng":
            np.random.random()
        else:
            random.random()
        return result

    monkeypatch.setattr(diagnostics, "_measure", mutate)
    try:
        with pytest.raises(diagnostics.GradientDiagnosticFailed, match="changed") as caught:
            diagnostics.diagnose_pc_update(state, pair())
        work = caught.value.report["work"]
        assert work["candidate_parameter_sets_completed"] == 1
        assert work["prior_forward_calls_completed"] == 2
        if fault == "parameter":
            assert not torch.equal(
                state.native_state.head_state["bias_output"],
                originals["pc"].native_state.head_state["bias_output"],
            )
        elif fault == "gradient":
            assert state.native_state.head_state[NAMES[0]].grad is not None
    finally:
        torch.set_rng_state(saved[0])
        np.random.set_state(saved[1])
        random.setstate(saved[2])


def test_should_refuse_finite_parameter_overflow_before_tanh_masks_it(originals):
    torch = require_torch()
    state = deepcopy(originals["pc"])
    state.native_state.head_state[NAMES[0]].fill_(3e38)
    before = deepcopy(state)
    with pytest.raises(diagnostics.GradientDiagnosticFailed, match="linear prior") as caught:
        diagnostics.diagnose_pc_update(state, pair())
    work = caught.value.report["work"]
    assert work["snapshot_validations_completed"] == 1
    assert work["prior_forward_calls_attempted"] == 1
    assert work["prior_forward_calls_completed"] == 0
    assert work["candidate_parameter_sets_completed"] == 0
    assert_tree(torch, state, before)
