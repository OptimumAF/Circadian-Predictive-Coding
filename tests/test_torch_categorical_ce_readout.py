"""Bounded software learning/typed fixtures, not the genuine four-task study."""

from copy import deepcopy
from dataclasses import replace
from math import exp, fsum
from typing import cast

import numpy as np
import pytest

from src.adapters import torch_categorical_ce_readout as ce
from src.adapters import torch_feature_learner as native
from src.adapters.torch_array_weighted import _ownership, _rng
from src.adapters.torch_decision import TorchDecisionPredictor, _same_values, decision_from_logits
from src.app.decision_reference_evaluation import DecisionCase
from src.core.decision_features import DecisionFeatureSpec, encode_decision
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


def planted():
    learner = owner()
    torch = require_torch()
    with torch.no_grad():
        learner._head.weight_feature_hidden.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        learner._head.bias_hidden.zero_()
    return learner


def forbidden(*args, **kwargs):
    raise AssertionError("unexpected work")


def test_float32_fit_learns_and_preserves_populated_source_and_continuation(monkeypatch):
    learner = planted()
    cases, spec = inputs()
    batch = ce.build_decision_training_batch(cases, spec, "literal", allow_subsets=True)
    learner.train_batch(batch.features, batch.targets)
    control = learner.fork()
    before, binding, rng = learner.snapshot_state(), _ownership(learner), _rng(require_torch())
    expected_logits = learner.predict_logits(batch.features)
    torch = require_torch()
    expected_loss = float(
        torch.nn.functional.cross_entropy(
            torch.from_numpy(expected_logits.astype(np.float32)),
            torch.from_numpy(batch.targets[:, 0].astype(np.int64)),
        ).item()
    )
    monkeypatch.setattr(learner, "snapshot_state", forbidden)
    fit = ce.fit_categorical_ce_readout(learner, cases, spec)
    assert np.array(fit.training_logits).argmax(axis=1).tolist() == [0, 1, 2]
    assert fit.final_loss < fit.closure_losses[0] and fit.final_loss < 0.01
    assert fit.closure_losses[0] == expected_loss
    assert len(fit.coefficients) == 4 and all(len(row) == 3 for row in fit.coefficients)
    assert all(float(np.float32(v)) == v for row in fit.coefficients for v in row)
    assert 1 <= fit.work["backward_returns"] == len(fit.closure_losses) <= 64
    assert fit.work["step_attempts"] == fit.work["step_returns"] == 1
    assert _same_values(before, native.TorchFeatureLearner.snapshot_state(learner))
    assert _ownership(learner) == binding and _same_values(rng, _rng(require_torch()))
    monkeypatch.undo()
    learner.train_batch(batch.features, batch.targets)
    control.train_batch(batch.features, batch.targets)
    assert _same_values(learner.snapshot_state(), control.snapshot_state())


@pytest.fixture(scope="module")
def learned():
    learner = planted()
    fit = ce.fit_categorical_ce_readout(learner, *inputs())
    return learner, fit


def test_owned_typed_reader_learns_all_rows_and_detaches_result(learned):
    learner, fit = learned
    external = deepcopy(fit)
    predictor = ce.CategoricalCEReadoutPredictor(external)
    cases, _ = inputs()
    before = learner.snapshot_state()
    for case in cases:
        result = predictor(case.request)
        assert result.selected_id == case.target_id
        assert abs(fsum(s.probability for s in result.scores) - 1) < 1e-12
    external.work["step_returns"] = 0
    assert predictor(cases[0].request).selected_id == "a"
    assert predictor.work.decision_returns == 4
    assert predictor.work.hidden_returns == predictor.work.affine_returns == 4
    assert _same_values(before, learner.snapshot_state())
    x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    learner.train_batch(x, np.array([[2.0], [2.0], [2.0]]))
    assert predictor(cases[0].request).selected_id == "a"


def test_decoder_preserves_forbidden_mass_and_lexical_tie_without_models():
    cases, spec = inputs()
    request = replace(cases[0].request, allowed_candidates=tuple(reversed(spec.catalog[1:])))
    result = decision_from_logits(request, spec, [10000.0, 0.0, 0.0], 7)
    assert result.selected_id == "b" and result.token_visits == 7
    assert [(s.candidate_id, s.raw_score, s.probability) for s in result.scores] == [
        ("b", 0.0, 0.5),
        ("c", 0.0, 0.5),
    ]


def test_original_torch_reader_retains_literal_native_normalization():
    learner = owner()
    cases, spec = inputs()
    request = replace(cases[0].request, allowed_candidates=tuple(reversed(spec.catalog[1:])))
    features, _ = encode_decision(request, spec)
    logits = learner.predict_logits(features)[0]
    predictor = TorchDecisionPredictor(spec, learner.snapshot_state())
    result = predictor(request)
    raw = [float(logits[i]) for i in (1, 2)]
    weights = [exp(v - max(raw)) for v in raw]
    assert [(s.raw_score, s.probability) for s in result.scores] == [
        (v, w / fsum(weights)) for v, w in zip(raw, weights, strict=True)
    ]
    assert predictor.work.native_forward_returns == predictor.work.decision_returns == 1


@pytest.mark.parametrize("role", ["outer_selection", "probe_train", "probe_test", "final"])
def test_roles_rejected_before_any_feature_label_or_optimizer_work(monkeypatch, role):
    cases, spec = inputs()
    object.__setattr__(cases[-1], "role", role)
    monkeypatch.setattr(ce, "_unchanged_owner", forbidden)
    monkeypatch.setattr(ce, "build_decision_training_batch", forbidden)
    with pytest.raises(ValueError, match="TRAIN"):
        ce.fit_categorical_ce_readout(cast(native.TorchFeatureLearner, None), cases, spec)


def test_hard_closure_ceiling_counts_paid_work_and_never_changes_source(monkeypatch):
    learner = planted()
    before = learner.snapshot_state()

    def repeat(optimizer, closure):
        for _ in range(65):
            closure()

    monkeypatch.setattr(require_torch().optim.LBFGS, "step", repeat)
    with pytest.raises(ce.CategoricalCEFitFailed, match="allowance") as caught:
        ce.fit_categorical_ce_readout(learner, *inputs())
    work = caught.value.work
    assert work["closure_attempts"] == 65
    assert work["forward_attempts"] == work["backward_returns"] == 64
    assert work["step_attempts"] == 1 and work["step_returns"] == 0
    assert work["final_read_attempts"] == 0
    assert _same_values(before, learner.snapshot_state())


def test_foreign_role_cannot_run_comparison_callbacks_before_source_guard(monkeypatch):
    class ForeignRole(str):
        def __eq__(self, other):
            raise AssertionError("role comparison callback ran")

        def __ne__(self, other):
            raise AssertionError("role comparison callback ran")

    cases, spec = inputs()
    object.__setattr__(cases[0], "role", ForeignRole("train"))
    monkeypatch.setattr(ce, "_unchanged_owner", forbidden)
    with pytest.raises(ValueError, match="TRAIN"):
        ce.fit_categorical_ce_readout(cast(native.TorchFeatureLearner, None), cases, spec)


def test_lost_optimizer_return_preserves_gradient_costs_without_source_changes(monkeypatch):
    learner = planted()
    before = learner.snapshot_state()

    def lost(optimizer, closure):
        closure()
        raise RuntimeError("literal lost return")

    monkeypatch.setattr(require_torch().optim.LBFGS, "step", lost)
    with pytest.raises(ce.CategoricalCEFitFailed, match="lost return") as caught:
        ce.fit_categorical_ce_readout(learner, *inputs())
    assert caught.value.work["backward_returns"] == 1
    assert caught.value.work["step_attempts"] == 1 and caught.value.work["step_returns"] == 0
    assert _same_values(before, learner.snapshot_state())


@pytest.mark.parametrize("kind", ["source", "rng", "cases", "batch"])
def test_fit_refuses_mutation_before_backward_and_preserves_it(monkeypatch, kind):
    torch = require_torch()
    learner = planted()
    cases, spec = inputs()
    before, rng = learner.snapshot_state(), _rng(torch)
    monkeypatch.setattr(learner, "snapshot_state", lambda: before)
    captured = []
    build = ce.build_decision_training_batch

    def batch(*a, **kw):
        value = build(*a, **kw)
        captured.append(value)
        return value

    monkeypatch.setattr(ce, "build_decision_training_batch", batch)
    loss = torch.nn.functional.cross_entropy

    def changed(*args, **kw):
        result = loss(*args, **kw)
        if kind == "source":
            with torch.no_grad():
                learner._head.bias_output.add_(1)
        elif kind == "rng":
            np.random.random()
        elif kind == "cases":
            object.__setattr__(cases[0], "role", "outer_selection")
        else:
            captured[0].targets[0, 0] = 2
        return result

    monkeypatch.setattr(torch.nn.functional, "cross_entropy", changed)
    with pytest.raises(ce.CategoricalCEFitFailed, match="changed") as caught:
        ce.fit_categorical_ce_readout(learner, cases, spec)
    assert caught.value.work["backward_attempts"] == 0
    if kind == "source":
        assert not _same_values(before, native.TorchFeatureLearner.snapshot_state(learner))
    elif kind == "rng":
        assert not _same_values(rng, _rng(torch))
    elif kind == "cases":
        assert cases[0].role == "outer_selection"
    else:
        assert captured[0].targets[0, 0] == 2


def test_boundary_mutation_refused_before_hidden_work(monkeypatch):
    learner = owner()

    def changed():
        learner._payload = tuple(list(learner._payload))

    monkeypatch.setattr(ce, "_hidden", forbidden)
    with pytest.raises(ce.CategoricalCEFitFailed, match="changed") as caught:
        ce.fit_categorical_ce_readout(learner, *inputs(), check_boundary=changed)
    assert caught.value.work["hidden_attempts"] == caught.value.work["optimizer_constructions"] == 0


def test_input_copy_mutation_refused_before_hidden_or_optimizer(monkeypatch):
    learner = owner()
    cases, spec = inputs()
    before = learner.snapshot_state()
    original = ce.deepcopy

    def changed(value):
        if type(value) is tuple and len(value) == 2 and value[0] is cases and value[1] is spec:
            with require_torch().no_grad():
                learner._head.bias_output.add_(1)
        return original(value)

    monkeypatch.setattr(ce, "deepcopy", changed)
    monkeypatch.setattr(ce, "_hidden", forbidden)
    with pytest.raises(ce.CategoricalCEFitFailed, match="changed") as caught:
        ce.fit_categorical_ce_readout(learner, cases, spec)
    assert caught.value.work["hidden_attempts"] == caught.value.work["optimizer_constructions"] == 0
    assert not _same_values(before, learner.snapshot_state())


@pytest.mark.parametrize("kind", ["source", "coefficients", "fit", "work"])
def test_reader_actual_state_and_equal_replacements_refused_and_latched(learned, monkeypatch, kind):
    _, fit = learned
    predictor = ce.CategoricalCEReadoutPredictor(fit)
    request = inputs()[0][0].request
    monkeypatch.setattr(predictor._learner, "snapshot_state", lambda: predictor._pin.source)
    if kind == "source":
        with require_torch().no_grad():
            predictor._learner._head.bias_output.add_(1)
    elif kind == "coefficients":
        predictor._coefficients = predictor._coefficients.clone()
    elif kind == "fit":
        predictor._fit = deepcopy(predictor._fit)
    else:
        predictor._work = replace(predictor._work)
    with pytest.raises(ValueError, match="changed"):
        predictor(request)
    assert predictor.work.hidden_attempts == 0
    with pytest.raises(ValueError):
        predictor(request)


def test_reader_failed_hidden_attempt_is_paid_and_never_repeated(learned, monkeypatch):
    predictor = ce.CategoricalCEReadoutPredictor(learned[1])

    def fail(*args):
        raise RuntimeError("literal hidden failure")

    monkeypatch.setattr(ce, "read_hidden_features", fail)
    request = inputs()[0][0].request
    with pytest.raises(RuntimeError, match="hidden failure"):
        predictor(request)
    assert predictor.work.hidden_attempts == 1 and predictor.work.hidden_returns == 0
    with pytest.raises(ValueError):
        predictor(request)
    assert predictor.work.hidden_attempts == 1


def test_invalid_fit_refused_before_reader_models(learned, monkeypatch):
    fit = learned[1]
    monkeypatch.setattr(native, "_components", forbidden)
    for broken in (
        replace(fit, coefficients=((float("nan"),),)),
        replace(fit, work={}),
        replace(fit, final_loss=float("inf")),
    ):
        with pytest.raises(ValueError):
            ce.CategoricalCEReadoutPredictor(broken)
