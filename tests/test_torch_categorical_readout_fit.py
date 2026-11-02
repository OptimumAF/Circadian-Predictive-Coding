"""Small software fixtures, not the native source12 feasibility experiment."""

from copy import deepcopy
from dataclasses import replace
from typing import cast

import numpy as np
import pytest

from src.adapters import torch_categorical_readout_fit as diagnostic
from src.adapters import torch_feature_learner as native
from src.adapters.torch_array_weighted import _ownership, _rng
from src.adapters.torch_decision import _same_values
from src.app.decision_reference_evaluation import DecisionCase
from src.core.decision_features import DecisionFeatureSpec
from src.core.typed_decision import DecisionCandidate, DecisionRequest
from src.shared.torch_runtime import require_torch


def inputs():
    catalog = tuple(DecisionCandidate(name, name) for name in ("a", "b", "c"))
    spec = DecisionFeatureSpec(("alpha", "beta"), catalog)
    cases = tuple(
        DecisionCase(DecisionRequest(str(i), query, catalog), target, "fixture-train", "train")
        for i, (query, target) in enumerate((("alpha", "a"), ("beta", "b"), ("alpha beta", "c")))
    )
    return cases, spec


def owner():
    return native.TorchFeatureLearner(native.TorchFeaturePolicy("bp", 2, 3, 17, num_classes=3))


def arrays():
    return np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), np.array([[0.0], [1.0], [2.0]])


def forbidden(*args, **kwargs):
    raise AssertionError("unexpected work")


def test_hidden_matches_native_logits_and_preserves_populated_optimizer(monkeypatch):
    torch = require_torch()
    learner = owner()
    x, y = arrays()
    learner.train_batch(x, y)
    before, binding, rng = learner.snapshot_state(), _ownership(learner), _rng(torch)
    expected = learner.predict_logits(x)
    monkeypatch.setattr(learner, "snapshot_state", forbidden)
    hidden = diagnostic.read_hidden_features(learner, x)
    reconstructed = (
        (
            torch.from_numpy(hidden.astype(np.float32)) @ before.head_state["weight_hidden_output"]
            + before.head_state["bias_output"]
        )
        .detach()
        .numpy()
        .astype(np.float64)
    )
    np.testing.assert_array_equal(reconstructed, expected)
    assert _same_values(before, native.TorchFeatureLearner.snapshot_state(learner))
    assert _ownership(learner) == binding and _same_values(rng, _rng(torch))
    assert hidden.dtype == np.float64 and hidden.flags.owndata
    hidden[:] = 99
    assert _same_values(before, native.TorchFeatureLearner.snapshot_state(learner))


def test_disposable_fit_has_equal_output_capacity_and_fits_known_rows(monkeypatch):
    torch = require_torch()
    learner = owner()
    with torch.no_grad():
        learner._head.weight_feature_hidden.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        learner._head.bias_hidden.zero_()
    cases, spec = inputs()
    before = learner.snapshot_state()
    monkeypatch.setattr(learner, "snapshot_state", forbidden)
    result = diagnostic.diagnose_categorical_readout_fit(learner, cases, spec)
    assert result.fit.feature_dim == 3 and result.fit.num_classes == 3
    assert sum(len(row) for row in result.fit.coefficients) == (3 + 1) * 3
    assert result.fit.rank == 3 and result.fit.squared_error < 1e-24
    np.testing.assert_array_equal(result.training_logits.argmax(axis=1), [0, 1, 2])
    assert (result.feature_calls, result.solve_calls, result.readout_calls) == (1, 1, 1)
    assert (result.feature_rows, result.solve_rows, result.readout_rows) == (3, 3, 3)
    result.hidden_features[:] = 0
    result.training_logits[:] = -1
    assert _same_values(before, native.TorchFeatureLearner.snapshot_state(learner))


@pytest.mark.parametrize("role", ["outer_selection", "probe_train", "probe_test", "final"])
def test_all_roles_checked_before_labels_encoding_or_model_work(monkeypatch, role):
    cases, spec = inputs()
    object.__setattr__(cases[-1], "role", role)
    monkeypatch.setattr(diagnostic, "build_decision_training_batch", forbidden)
    monkeypatch.setattr(diagnostic, "_unchanged_owner", forbidden)
    with pytest.raises(ValueError, match="TRAIN"):
        diagnostic.diagnose_categorical_readout_fit(
            cast(native.TorchFeatureLearner, object()), cases, spec
        )


@pytest.mark.parametrize("corruption", ["parameters", "optimizer_binding"])
def test_actual_corruption_refused_despite_valid_public_snapshot(monkeypatch, corruption):
    torch = require_torch()
    learner = owner()
    snapshot = learner.snapshot_state()
    monkeypatch.setattr(learner, "snapshot_state", lambda: snapshot)
    monkeypatch.setattr(diagnostic, "_hidden", forbidden)
    if corruption == "parameters":
        with torch.no_grad():
            learner._head.bias_hidden[0, 0] = float("nan")
    else:
        params = learner._optimizer.param_groups[0]["params"]
        params[0] = torch.nn.Parameter(params[0].detach().clone())
    with pytest.raises(ValueError):
        diagnostic.read_hidden_features(learner, arrays()[0])
    if corruption == "parameters":
        assert torch.isnan(learner._head.bias_hidden[0, 0])
    else:
        assert (
            learner._optimizer.param_groups[0]["params"][0]
            is not learner._head.weight_feature_hidden
        )


@pytest.mark.parametrize("change", ["source", "payload", "rng", "input"])
def test_hidden_read_refuses_changes_without_restoring(monkeypatch, change):
    torch = require_torch()
    learner = owner()
    x, _ = arrays()
    before, payload, rng = learner.snapshot_state(), learner._payload, _rng(torch)
    original = torch.tanh

    def changed(values):
        result = original(values)
        if change == "source":
            with torch.no_grad():
                learner._head.bias_output.add_(1)
        elif change == "payload":
            learner._payload = tuple(list(payload))
        elif change == "rng":
            np.random.random()
        else:
            x[0, 0] = 0
        return result

    monkeypatch.setattr(torch, "tanh", changed)
    with pytest.raises(ValueError, match="changed"):
        diagnostic.read_hidden_features(learner, x)
    if change == "source":
        assert not _same_values(before, learner.snapshot_state())
    elif change == "payload":
        assert learner._payload is not payload
    elif change == "rng":
        assert not _same_values(rng, _rng(torch))
    else:
        assert x[0, 0] == 0


@pytest.mark.parametrize("boundary", ["solve", "readout", "cases"])
def test_diagnostic_checks_actual_owner_and_inputs_after_each_boundary(monkeypatch, boundary):
    learner = owner()
    cases, spec = inputs()
    before = learner.snapshot_state()
    name = (
        "predict_categorical_linear_logits" if boundary == "readout" else "fit_categorical_linear"
    )
    original = getattr(diagnostic, name)

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        if boundary == "cases":
            object.__setattr__(cases[0], "role", "outer_selection")
        else:
            with require_torch().no_grad():
                learner._head.bias_output.add_(1)
        return result

    monkeypatch.setattr(diagnostic, name, changed)
    if boundary != "readout":
        monkeypatch.setattr(diagnostic, "predict_categorical_linear_logits", forbidden)
    with pytest.raises(ValueError, match="changed"):
        diagnostic.diagnose_categorical_readout_fit(learner, cases, spec)
    if boundary == "cases":
        assert cases[0].role == "outer_selection" and _same_values(before, learner.snapshot_state())
    else:
        assert not _same_values(before, learner.snapshot_state())


def test_failed_solve_preserves_source_and_exception(monkeypatch):
    learner = owner()
    before = learner.snapshot_state()

    def fail(*args):
        raise ArithmeticError("literal failed solve")

    monkeypatch.setattr(diagnostic, "fit_categorical_linear", fail)
    with pytest.raises(ArithmeticError, match="literal failed solve"):
        diagnostic.diagnose_categorical_readout_fit(learner, *inputs())
    assert _same_values(before, learner.snapshot_state())


def test_working_snapshot_and_actual_owner_cannot_change_together(monkeypatch):
    learner = owner()
    before = learner.snapshot_state()

    def changed(state, features):
        with require_torch().no_grad():
            state.head_state["bias_output"].add_(1)
            learner._head.bias_output.add_(1)
        return np.zeros((len(features), state.policy.hidden_dim), dtype=np.float64)

    monkeypatch.setattr(diagnostic, "_hidden", changed)
    monkeypatch.setattr(diagnostic, "fit_categorical_linear", forbidden)
    with pytest.raises(ValueError, match="changed"):
        diagnostic.diagnose_categorical_readout_fit(learner, *inputs())
    assert not _same_values(before, learner.snapshot_state())


def test_diagnostic_does_not_change_next_native_update():
    learner = owner()
    x, y = arrays()
    learner.train_batch(x, y)
    control = learner.fork()
    diagnostic.diagnose_categorical_readout_fit(learner, *inputs())
    learner.train_batch(x, y)
    control.train_batch(x, y)
    assert _same_values(learner.snapshot_state(), control.snapshot_state())


def test_invalid_inputs_and_schema_refused_before_hidden_work(monkeypatch):
    learner = owner()
    x, _ = arrays()
    monkeypatch.setattr(diagnostic, "_hidden", forbidden)
    for invalid in (x.astype(np.float32), np.ones((2, 3)), x[:0], x * np.nan):
        with pytest.raises(ValueError):
            diagnostic.read_hidden_features(learner, invalid)
    cases, spec = inputs()
    with pytest.raises(ValueError, match="schema"):
        diagnostic.diagnose_categorical_readout_fit(
            learner, cases, replace(spec, vocabulary=("alpha",))
        )
    assert _same_values(cases, deepcopy(cases))
