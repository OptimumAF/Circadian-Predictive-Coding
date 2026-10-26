"""Weighted realized-PC readout math, bounded admission and detached-state tests.

NumPy independently implements the weighted local equations and prior derivative.
These checks never train a learner or step an optimizer; native-oracle comparison
belongs to the separate integration tests owned by the integration agent.
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
from src.shared.torch_runtime import require_torch
from test_torch_feature_gradient_diagnostics import pair
from test_torch_feature_learner import assert_tree
from test_torch_pc_update_diagnostics import NAMES, _arrays, _episode, _geometry, _probabilities

METHODS = ("pc", "neutral_cpc")
WEIGHTS = (1.0, 1.0, 0.25, 1.0, 1.0, 1.0, 3.0)
DEFINITION = "torch_invariant_weighted_pc_realized_update_v1"
OBJECTIVE = "categorical_target_weighted_ce_plus_half_row_hidden_squared_sum_v1"


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
        pytest.fail("weighted detached readout may not train or step an optimizer")

    monkeypatch.setattr(native.TorchFeatureLearner, "train_batch", forbidden)
    monkeypatch.setattr(require_torch().optim.SGD, "step", forbidden)
    yield
    record_property("native_train_calls", 0)
    record_property("optimizer_steps", 0)


def _row_ce(logits, targets):
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return -log_probabilities[np.arange(len(targets)), targets]


def _oracle(state, episodes, weights):
    """Independent float32 proposal and float64 weighted prior-CE derivative."""
    policy = state.policy.native_policy
    x, targets = _arrays(episodes)
    p = {name: state.native_state.head_state[name].numpy().copy() for name in NAMES}
    one_hot = np.eye(7, dtype=np.float32)[targets]
    row_weights = np.array(weights, dtype=np.float32)[targets]
    fractions = row_weights / row_weights.sum()
    factors = (np.float32(len(targets)) * fractions)[:, None]
    prior = np.tanh(x @ p[NAMES[0]] + p[NAMES[1]])
    z = prior.copy()
    for _ in range(policy.inference_steps):
        error = _probabilities(z @ p[NAMES[2]] + p[NAMES[3]]) - one_hot
        z -= policy.inference_learning_rate * ((z - prior + error @ p[NAMES[2]].T) * factors)
    logits = z @ p[NAMES[2]] + p[NAMES[3]]
    error = _probabilities(logits) - one_hot
    residual = z - prior
    weighted_prior = -residual * (1 - prior * prior) * factors
    weighted_error = error * factors
    manual = (
        (x.T @ weighted_prior) / len(targets),
        weighted_prior.mean(axis=0, keepdims=True),
        (z.T @ weighted_error) / len(targets),
        weighted_error.mean(axis=0, keepdims=True),
    )
    candidate = {
        name: p[name] - policy.learning_rate * gradient
        for name, gradient in zip(NAMES, manual, strict=True)
    }
    delta = tuple(candidate[name] - p[name] for name in NAMES)

    # Prior gradients are independent of the finite relaxation equations above.
    double = {name: value.astype(np.float64) for name, value in p.items()}
    xd = x.astype(np.float64)
    hidden = np.tanh(xd @ double[NAMES[0]] + double[NAMES[1]])
    prior_logits = hidden @ double[NAMES[2]] + double[NAMES[3]]
    normalized = np.array(weights, dtype=np.float64)[targets]
    normalized /= normalized.sum()
    prior_error = (_probabilities(prior_logits) - one_hot) * normalized[:, None]
    hidden_error = (prior_error @ double[NAMES[2]].T) * (1 - hidden * hidden)
    gradients = (
        xd.T @ hidden_error,
        hidden_error.sum(axis=0, keepdims=True),
        hidden.T @ prior_error,
        prior_error.sum(axis=0, keepdims=True),
    )
    counter_hidden = np.tanh(x @ candidate[NAMES[0]] + candidate[NAMES[1]])
    counter_logits = counter_hidden @ candidate[NAMES[2]] + candidate[NAMES[3]]
    row_before, row_after = _row_ce(prior_logits, targets), _row_ce(counter_logits, targets)
    fixed_before = _row_ce(logits, targets) + 0.5 * np.square(residual).sum(axis=1)
    fixed_after = _row_ce(z @ candidate[NAMES[2]] + candidate[NAMES[3]], targets)
    fixed_after += 0.5 * np.square(z - counter_hidden).sum(axis=1)
    return dict(
        gradients=gradients,
        delta=delta,
        prior_mean_ce=float(row_before.mean()),
        counterfactual_prior_mean_ce=float(row_after.mean()),
        prior_weighted_mean_ce=float(row_before @ normalized),
        counterfactual_prior_weighted_mean_ce=float(row_after @ normalized),
        fixed_state_weighted_energy_before=float(fixed_before @ normalized),
        fixed_state_weighted_energy_after=float(fixed_after @ normalized),
        hidden_displacement_l2=float(np.linalg.norm(residual.astype(np.float64))),
        legacy=float(0.5 * (np.square(error).mean() + np.square(residual).mean())),
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("steps", [1, 2, 4])
def test_should_match_independent_weighted_objective_candidate_and_prior_geometry(method, steps):
    state = InvariantTorchFeatureLearner(
        InvariantTorchFeaturePolicy(
            native.TorchFeaturePolicy(method, 296, 8, 17, inference_steps=steps, num_classes=7)
        )
    ).snapshot_state()
    episodes = pair()
    expected = _oracle(state, episodes, WEIGHTS)
    report = diagnostics.diagnose_pc_update(state, episodes, class_weights=WEIGHTS)
    assert report["status"] == "complete" and report["reason"] is None
    assert report["definition"] == DEFINITION and report["objective"] == OBJECTIVE
    assert report["geometry_objective"] == "weighted_prior_mean_ce"
    assert report["method"] == method and report["inference_steps"] == steps
    assert report["class_weights"] == list(WEIGHTS)
    assert report["source_episode_ids"] == [episode.episode_id for episode in episodes]
    assert report["relaxation"] == []  # The exact primitive has no intermediate trace port.
    for key in (
        "prior_mean_ce",
        "counterfactual_prior_mean_ce",
        "prior_weighted_mean_ce",
        "counterfactual_prior_weighted_mean_ce",
        "fixed_state_weighted_energy_before",
        "fixed_state_weighted_energy_after",
    ):
        assert report[key] == pytest.approx(expected[key], rel=3e-6, abs=3e-7)
    for key, before, after in (
        ("prior_ce_change", "prior_mean_ce", "counterfactual_prior_mean_ce"),
        (
            "weighted_prior_ce_change",
            "prior_weighted_mean_ce",
            "counterfactual_prior_weighted_mean_ce",
        ),
        (
            "fixed_state_weighted_energy_change",
            "fixed_state_weighted_energy_before",
            "fixed_state_weighted_energy_after",
        ),
    ):
        assert report[key] == pytest.approx(expected[after] - expected[before], abs=7e-7)
    proposal = report["proposal"]
    assert proposal["weighted_energy"] == pytest.approx(
        expected["fixed_state_weighted_energy_before"], rel=3e-6
    )
    assert proposal["hidden_displacement_l2"] == pytest.approx(
        expected["hidden_displacement_l2"], rel=3e-5
    )
    assert proposal["nominal_latent_row_iterations"] == 4 * steps
    assert report["legacy_diagnostic"]["definition"] == native.TORCH_PC_ENERGY_ID
    assert report["legacy_diagnostic"]["value"] == pytest.approx(expected["legacy"], rel=3e-6)
    assert proposal["weighted_energy"] != pytest.approx(expected["legacy"], rel=0.1)
    for actual, geometry in [
        (report["gradient_delta_geometry"], _geometry(expected["gradients"], expected["delta"])),
        *(
            (
                report["per_parameter_geometry"][name],
                _geometry((expected["gradients"][i],), (expected["delta"][i],)),
            )
            for i, name in enumerate(NAMES)
        ),
    ]:
        for key, value in geometry.items():
            assert actual[key] == pytest.approx(value, rel=5e-4, abs=4e-7)
    assert report["prior_mean_ce"] != pytest.approx(report["prior_weighted_mean_ce"], abs=1e-3)
    work = report["work"]
    for status in ("attempted", "completed"):
        assert work[f"proposal_calls_{status}"] == 1
        assert work[f"proposal_rows_{status}"] == 4
        assert work[f"proposal_nominal_latent_row_iterations_{status}"] == 4 * steps
        # The original energy is returned by the paid proposal. Only the
        # counterfactual fixed-z energy requires a separate evaluation.
        assert work[f"fixed_state_energy_calls_{status}"] == 1
        assert work[f"fixed_state_energy_rows_{status}"] == 4
        assert work[f"prior_forward_calls_{status}"] == 2
        assert work[f"prior_forward_rows_{status}"] == 8
        assert work[f"prior_backward_calls_{status}"] == 1
        assert work[f"prior_backward_rows_{status}"] == 4
    assert work["projection_rows_completed"] == 4
    assert work["candidate_parameter_sets_completed"] == 1
    assert work["native_train_calls"] == work["optimizer_steps"] == work["published_updates"] == 0
    assert work["validation_temporary_optimizers"] == 0
    assert not work["failure_work_may_be_unobserved"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("method", METHODS)
def test_should_match_uniform_observed_weights_without_relabeling_the_legacy_report(
    originals, method
):
    # Other classes are absent: differing absent weights must have no effect.
    weights = (2.0, 8.0, 1.0, 9.0, 5.0, 4.0, 1.0)
    state, episodes = originals[method], pair()
    original = diagnostics.diagnose_pc_update(state, episodes)
    explicit_none = diagnostics.diagnose_pc_update(state, episodes, class_weights=None)
    weighted = diagnostics.diagnose_pc_update(state, episodes, class_weights=weights)
    assert original == explicit_none
    assert original["definition"] == diagnostics.DEFINITION
    assert "class_weights" not in original and "proposal" not in original
    for key in (
        "legacy_diagnostic",
        "prior_mean_ce",
        "counterfactual_prior_mean_ce",
        "prior_ce_change",
        "gradient_delta_geometry",
        "per_parameter_geometry",
    ):
        assert weighted[key] == original[key]
    assert weighted["prior_weighted_mean_ce"] == weighted["prior_mean_ce"]
    assert (
        weighted["counterfactual_prior_weighted_mean_ce"]
        == weighted["counterfactual_prior_mean_ce"]
    )
    assert weighted["definition"] != original["definition"]


@pytest.mark.parametrize("method", METHODS)
def test_should_preserve_complete_snapshot_gradients_inputs_rng_and_no_grad_mode(originals, method):
    torch = require_torch()
    state = deepcopy(originals[method])
    if method == "neutral_cpc":
        state.native_state.head_state.update(reward_error_ema=0.25, last_reward_scale=0.75)
    tensor = state.native_state.head_state[NAMES[0]]
    tensor.requires_grad_(True)
    tensor.grad = torch.full_like(tensor, -0.25)
    before, gradient = deepcopy(state), tensor.grad.clone()
    episodes, weights = pair(), WEIGHTS
    inputs = pickle.dumps((episodes, weights))
    rng = torch.get_rng_state().clone(), pickle.dumps(np.random.get_state()), random.getstate()
    with torch.no_grad():
        report = diagnostics.diagnose_pc_update(state, episodes, class_weights=weights)
        assert not torch.is_grad_enabled()
    assert torch.is_grad_enabled()
    assert_tree(torch, state, before)
    assert tensor.requires_grad and torch.equal(tensor.grad, gradient)
    assert pickle.dumps((episodes, weights)) == inputs
    assert torch.equal(torch.get_rng_state(), rng[0])
    assert pickle.dumps(np.random.get_state()) == rng[1] and random.getstate() == rng[2]
    assert report["status"] == "complete"


def test_should_match_pc_and_neutral_readouts_without_changing_method_laws(originals):
    reports = [
        diagnostics.diagnose_pc_update(originals[m], pair(), class_weights=WEIGHTS) for m in METHODS
    ]
    for key in (
        "proposal",
        "legacy_diagnostic",
        "gradient_delta_geometry",
        "per_parameter_geometry",
        "prior_mean_ce",
        "prior_weighted_mean_ce",
        "fixed_state_weighted_energy_after",
    ):
        assert reports[0][key] == reports[1][key]


@pytest.mark.parametrize("count", [1, 54])
def test_should_accept_exact_bounded_train_sources(originals, count):
    episodes = (
        (_episode("single", 1),) if count == 1 else tuple(_episode(str(i), 9) for i in range(6))
    )
    report = diagnostics.diagnose_pc_update(originals["pc"], episodes, class_weights=WEIGHTS)
    assert report["status"] == "complete" and report["rows"] == count
    assert report["work"]["proposal_rows_completed"] == count


@pytest.mark.parametrize(
    "weights",
    [
        [],
        [1.0] * 7,
        (),
        (1.0,) * 6,
        (1.0,) * 8,
        (True,) + (1.0,) * 6,
        (np.float64(1.0),) + (1.0,) * 6,
        (0.0,) + (1.0,) * 6,
        (-1.0,) + (1.0,) * 6,
        (1e-7,) + (1.0,) * 6,
        (1e7,) + (1.0,) * 6,
        (float("nan"),) + (1.0,) * 6,
        (float("inf"),) + (1.0,) * 6,
        ("1",) + (1.0,) * 6,
    ],
)
def test_should_refuse_invalid_explicit_weights_before_native_copy(originals, monkeypatch, weights):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid weights must fail before copying a native snapshot")

    monkeypatch.setattr(diagnostics, "copy_invariant_torch_feature_snapshot", forbidden)
    with pytest.raises(ValueError):
        diagnostics.diagnose_pc_update(originals["pc"], pair(), class_weights=weights)


@pytest.mark.parametrize(
    "fault", ["final_test", "probe_train", "probe_test", "bp", "gated_cpc", "rows", "contexts"]
)
def test_should_refuse_weighted_roles_methods_and_work_beyond_bounds(monkeypatch, fault):
    method = fault if fault in ("bp", "gated_cpc") else "pc"
    policy = InvariantTorchFeaturePolicy(
        native.TorchFeaturePolicy(method, 296, 8, 17, num_classes=7)
    )
    snapshot = InvariantTorchFeatureSnapshot(policy, cast(Any, None))
    episodes = pair()
    if fault in ("final_test", "probe_train", "probe_test"):
        episodes = (replace(episodes[0], role=fault), episodes[1])
    elif fault == "rows":
        episodes = tuple(_episode(str(i), count) for i, count in enumerate((16, 16, 16, 7)))
    elif fault == "contexts":
        episodes = tuple(_episode(str(i), 1) for i in range(7))

    def forbidden(*args, **kwargs):
        pytest.fail("invalid metadata must fail before native snapshot copying")

    monkeypatch.setattr(diagnostics, "copy_invariant_torch_feature_snapshot", forbidden)
    with pytest.raises(ValueError):
        diagnostics.diagnose_pc_update(snapshot, episodes, class_weights=WEIGHTS)


def test_should_refuse_weighted_readout_in_inference_mode_before_copy(originals, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("inference mode must fail before snapshot copying")

    torch = require_torch()
    monkeypatch.setattr(diagnostics, "copy_invariant_torch_feature_snapshot", forbidden)
    with torch.inference_mode(), pytest.raises(ValueError, match="inference_mode"):
        diagnostics.diagnose_pc_update(originals["pc"], pair(), class_weights=WEIGHTS)


@pytest.mark.parametrize("fault", ["nonfinite", "foreign"])
def test_should_refuse_corrupt_snapshot_payload_without_copying_unknown_objects(originals, fault):
    state = deepcopy(originals["pc"])

    class Foreign:
        def __deepcopy__(self, memo):
            pytest.fail("unknown native payload may not be copied")

    state.native_state.head_state[NAMES[0]] = (
        require_torch().full_like(state.native_state.head_state[NAMES[0]], float("nan"))
        if fault == "nonfinite"
        else Foreign()
    )
    with pytest.raises(diagnostics.GradientDiagnosticFailed) as caught:
        diagnostics.diagnose_pc_update(state, pair(), class_weights=WEIGHTS)
    report = caught.value.report
    assert report["status"] == "incomplete" and report["proposal"] is None
    assert report["work"]["snapshot_validations_attempted"] == 1
    assert report["work"]["snapshot_validations_completed"] == 0
    assert report["work"]["proposal_calls_attempted"] == 0
    json.dumps(report, allow_nan=False)


def test_should_charge_prior_backward_attempt_before_failure_without_proposing(
    originals, monkeypatch
):
    state, before = deepcopy(originals["pc"]), deepcopy(originals["pc"])

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic weighted backward failure")

    monkeypatch.setattr(require_torch().autograd, "grad", fail)
    with pytest.raises(diagnostics.GradientDiagnosticFailed, match="backward") as caught:
        diagnostics.diagnose_pc_update(state, pair(), class_weights=WEIGHTS)
    report, work = caught.value.report, caught.value.report["work"]
    assert report["status"] == "incomplete"
    assert work["projection_rows_completed"] == work["prior_forward_rows_completed"] == 4
    assert work["prior_backward_calls_attempted"] == 1
    assert work["prior_backward_calls_completed"] == work["proposal_calls_attempted"] == 0
    assert work["failure_work_may_be_unobserved"]
    assert_tree(require_torch(), state, before)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("fault", ["proposal", "late_measure"])
def test_should_retain_observed_work_and_source_state_when_weighted_measurement_fails(
    originals, monkeypatch, fault
):
    state = deepcopy(originals["pc"])
    before = deepcopy(state)
    name = "class_weighted_pc_update" if fault == "proposal" else "_measure_weighted"
    original = getattr(diagnostics, name)

    def fail(*args, **kwargs):
        if fault == "late_measure":
            original(*args, **kwargs)
        raise RuntimeError("synthetic weighted measurement failure")

    monkeypatch.setattr(diagnostics, name, fail)
    with pytest.raises(diagnostics.GradientDiagnosticFailed, match="synthetic") as caught:
        diagnostics.diagnose_pc_update(state, pair(), class_weights=WEIGHTS)
    report, work = caught.value.report, caught.value.report["work"]
    assert report["status"] == "incomplete" and work["failure_work_may_be_unobserved"]
    assert work["projection_rows_completed"] == 4
    assert work["prior_backward_calls_completed"] == 1
    assert work["proposal_calls_attempted"] == 1
    assert work["proposal_rows_attempted"] == 4
    assert work["proposal_nominal_latent_row_iterations_attempted"] == 8
    assert work["proposal_calls_completed"] == (0 if fault == "proposal" else 1)
    assert work["candidate_parameter_sets_completed"] == (0 if fault == "proposal" else 1)
    assert_tree(require_torch(), state, before)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "fault", ["parameter", "gradient", "episodes", "torch_rng", "numpy_rng", "python_rng"]
)
def test_should_refuse_observed_late_changes_without_reverting_them(originals, monkeypatch, fault):
    torch = require_torch()
    state, episodes = deepcopy(originals["pc"]), pair()
    original = diagnostics._measure_weighted
    saved = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        if fault == "parameter":
            with torch.no_grad():
                state.native_state.head_state["bias_output"].add_(0.25)
        elif fault == "gradient":
            value = state.native_state.head_state[NAMES[0]]
            value.grad = torch.ones_like(value)
        elif fault == "episodes":
            object.__setattr__(episodes[0].transitions[0], "reward", 0.5)
        elif fault == "torch_rng":
            torch.rand(1)
        elif fault == "numpy_rng":
            np.random.random()
        else:
            random.random()
        return result

    monkeypatch.setattr(diagnostics, "_measure_weighted", mutate)
    try:
        with pytest.raises(diagnostics.GradientDiagnosticFailed, match="changed") as caught:
            diagnostics.diagnose_pc_update(state, episodes, class_weights=WEIGHTS)
        work = caught.value.report["work"]
        assert work["proposal_calls_completed"] == 1
        assert work["prior_forward_calls_completed"] == 2
        if fault == "parameter":
            assert not torch.equal(
                state.native_state.head_state["bias_output"],
                originals["pc"].native_state.head_state["bias_output"],
            )
        elif fault == "gradient":
            assert state.native_state.head_state[NAMES[0]].grad is not None
        elif fault == "episodes":
            assert episodes[0].transitions[0].reward == 0.5
    finally:
        torch.set_rng_state(saved[0])
        np.random.set_state(saved[1])
        random.setstate(saved[2])
