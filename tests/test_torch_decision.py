"""Literal native parity/isolation and typed-dispatch integration, not a study."""

from copy import deepcopy
from dataclasses import replace
from math import exp, fsum

import numpy as np
import pytest

from scripts.run_ticket_workflow import FEATURES, training_cases
from src.adapters.perceptron_decision import build_decision_training_batch
from src.adapters.torch_decision import TorchDecisionPredictor, _same_values
from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy
from src.app.ticket_workflow import run_workflow_episode
from src.core.decision_features import DecisionFeatureSpec, encode_decision
from src.core.ticket_workflow import ACTION_IDS, CATALOG, WorkflowCapabilities, WorkflowState
from src.core.typed_decision import DecisionCandidate, DecisionRequest
from src.shared.torch_runtime import require_torch

METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")
SPEC = DecisionFeatureSpec(FEATURES, CATALOG)


@pytest.fixture(scope="module", params=METHODS)
def source(request):
    torch = require_torch()
    torch.set_num_threads(2)
    global_rng = torch.get_rng_state().clone()
    numpy_rng = deepcopy(np.random.get_state())
    learner = TorchFeatureLearner(
        TorchFeaturePolicy(request.param, len(FEATURES), 3, 17, num_classes=6)
    )
    batch = build_decision_training_batch(training_cases(), SPEC, "teacher", allow_subsets=True)
    learner.train_batch(batch.features, batch.targets)
    assert torch.equal(global_rng, torch.get_rng_state())
    current = np.random.get_state()
    assert type(numpy_rng) is tuple and type(current) is tuple
    assert numpy_rng[0] == current[0] and np.array_equal(numpy_rng[1], current[1])
    assert numpy_rng[2:] == current[2:]
    return learner, batch


def test_raw_logits_preserve_native_probabilities_and_own_storage(source):
    torch = require_torch()
    learner, batch = source
    before = learner.snapshot_state()
    values = torch.from_numpy(batch.features.astype(np.float32))
    with torch.no_grad():
        native = (
            learner._head.forward_logits(values)
            if learner.policy.method == "bp"
            else learner._head.predict_logits(values)
        )
        expected = torch.softmax(native, dim=1).numpy().astype(np.float64)
    actual = learner.predict_logits(batch.features)
    assert actual.dtype == np.float64 and actual.flags.owndata
    assert actual.tobytes() == native.numpy().astype(np.float64).tobytes()
    assert learner.predict(batch.features).tobytes() == expected.tobytes()
    actual[:] = 0
    assert _same_values(before, learner.snapshot_state())
    assert (
        learner.predict_logits(batch.features).tobytes()
        == native.numpy().astype(np.float64).tobytes()
    )


def test_typed_subset_uses_native_logits_and_detached_complete_state(source):
    torch = require_torch()
    learner, _ = source
    state = learner.snapshot_state()
    original = deepcopy(state)
    predictor = TorchDecisionPredictor(SPEC, state)
    request = training_cases()[0].request
    request = replace(request, allowed_candidates=tuple(reversed(request.allowed_candidates)))
    global_rng = torch.get_rng_state().clone()
    features, visits = encode_decision(request, SPEC)
    raw = learner.predict_logits(features)[0]
    result = predictor(request)
    ids = tuple(c.candidate_id for c in SPEC.catalog)
    scores = [float(raw[ids.index(s.candidate_id)]) for s in result.scores]
    weights = [exp(v - max(scores)) for v in scores]
    assert [s.raw_score for s in result.scores] == scores
    assert [s.probability for s in result.scores] == [v / fsum(weights) for v in weights]
    assert result.token_visits == visits
    assert torch.equal(global_rng, torch.get_rng_state())
    assert _same_values(original, learner.snapshot_state())
    assert _same_values(original, predictor._learner.snapshot_state())
    state.head_state["bias_output"].add_(7)
    assert predictor(request) == result  # Input snapshot no longer aliases either owner.
    assert _same_values(original, predictor._learner.snapshot_state())
    work = predictor.work
    assert (
        work.decision_attempts,
        work.native_forward_attempts,
        work.native_forward_returns,
        work.decision_returns,
    ) == (2, 2, 2, 2)
    assert predictor.work is not work
    assert not hasattr(predictor, "train_batch") and not hasattr(predictor, "restore_state")


def test_extreme_forbidden_logits_do_not_erase_permitted_mass(source):
    learner, _ = source
    state = learner.snapshot_state()
    state.head_state["weight_hidden_output"].zero_()
    state.head_state["bias_output"].zero_()
    state.head_state["bias_output"][0, 0] = 10000
    predictor = TorchDecisionPredictor(SPEC, state)
    request = DecisionRequest("subset", "invoice", tuple(reversed(CATALOG[1:3])))
    result = predictor(request)
    assert tuple(s.candidate_id for s in result.scores) == tuple(
        c.candidate_id for c in CATALOG[1:3]
    )
    assert all(s.probability == 0.5 and s.raw_score == 0 for s in result.scores)
    assert result.selected_id == CATALOG[1].candidate_id


def test_actual_dispatch_ordinary_recovery_and_revocation_with_literal_native_parameters(source):
    learner, _ = source
    state = learner.snapshot_state()
    # A literal analytic phase-router, not fitted performance or a learned result.
    for name in ("weight_feature_hidden", "bias_hidden", "weight_hidden_output", "bias_output"):
        state.head_state[name].zero_()
    invoice_target = training_cases()[1].target_id
    for hidden, (phase, action) in enumerate(
        (("sealed", "open"), ("ready", invoice_target), ("staged", "commit"))
    ):
        state.head_state["weight_feature_hidden"][FEATURES.index(phase), hidden] = 1
        state.head_state["weight_hidden_output"][hidden, ACTION_IDS.index(action)] = 10
    predictor = TorchDecisionPredictor(SPEC, state)
    before = learner.snapshot_state()
    reports = [
        run_workflow_episode(
            WorkflowState("native-typed", "invoice"),
            predictor,
            capabilities=WorkflowCapabilities(ACTION_IDS),
            scenario=scenario,
        )
        for scenario in ("ordinary", "stale_once", "revoke_commit")
    ]
    assert all(r["complete"] for r in reports)
    assert reports[0]["success"] and reports[1]["success"] and not reports[2]["success"]
    assert [len(r["steps"]) for r in reports] == [3, 5, 3]
    assert reports[2]["steps"][-1]["transition"]["status"] == "permission_denied"
    assert all(r["training_updates"] == 0 and r["external_actions"] == 0 for r in reports)
    assert _same_values(before, learner.snapshot_state())
    predictor.assert_unchanged()
    assert predictor.work.native_forward_attempts == 11


def test_complete_state_drift_is_refused_and_latched(source):
    learner, _ = source
    predictor = TorchDecisionPredictor(SPEC, learner.snapshot_state())
    if learner.policy.method == "bp":
        predictor._learner._head.bias_output.grad.add_(0.1)
    elif learner.policy.method == "pc":
        predictor._learner._head._traffic_sum.add_(0.1)
    else:
        generator = predictor._learner._head._split_generator
        generator.manual_seed(22)
    with pytest.raises(ValueError, match="state changed"):
        predictor(training_cases()[0].request)
    with pytest.raises(ValueError, match="stopped"):
        predictor(training_cases()[0].request)
    assert predictor.work.decision_attempts == 2 and predictor.work.native_forward_attempts == 0


@pytest.mark.parametrize("method", METHODS)
def test_binary_predict_bytes_unchanged(method):
    torch = require_torch()
    learner = TorchFeatureLearner(TorchFeaturePolicy(method, 2, 3, 17))
    features = np.array([[0.3, -0.2], [-0.5, 0.4]])
    raw = learner.predict_logits(features)
    expected = (
        torch.softmax(torch.from_numpy(raw.astype(np.float32)), dim=1)[:, 1:2]
        .numpy()
        .astype(np.float64)
    )
    assert learner.predict(features).tobytes() == expected.tobytes()
    assert learner.predict(features).shape == (2, 1)


def test_schema_admission_before_native_construction(source):
    learner, _ = source
    state = learner.snapshot_state()
    with pytest.raises(ValueError, match="dimensions"):
        TorchDecisionPredictor(replace(SPEC, vocabulary=FEATURES[:-1]), state)
    predictor = TorchDecisionPredictor(SPEC, state)
    request = DecisionRequest(
        "bad-description", "invoice", (DecisionCandidate(CATALOG[0].candidate_id, "replacement"),)
    )
    with pytest.raises(ValueError, match="descriptions"):
        predictor(request)
    assert predictor.work.native_forward_attempts == 0


def test_paid_native_fault_latches_after_no_return(source, monkeypatch):
    learner, _ = source
    predictor = TorchDecisionPredictor(SPEC, learner.snapshot_state())

    def fail(_values):
        raise RuntimeError("lost native return")

    name = "forward_logits" if learner.policy.method == "bp" else "predict_logits"
    with monkeypatch.context() as patch:
        patch.setattr(predictor._learner._head, name, fail)
        with pytest.raises(RuntimeError, match="lost native return"):
            predictor(training_cases()[0].request)
    with pytest.raises(ValueError, match="stopped"):
        predictor(training_cases()[0].request)
    assert predictor.work.native_forward_attempts == 1
    assert predictor.work.native_forward_returns == predictor.work.decision_returns == 0


def test_later_source_training_cannot_change_frozen_readout(source):
    learner, batch = source
    predictor = TorchDecisionPredictor(SPEC, learner.snapshot_state())
    request = training_cases()[0].request
    before = predictor(request)
    learner.train_batch(batch.features, batch.targets)
    assert learner.completed_updates > predictor._learner.completed_updates
    assert predictor(request) == before
    predictor.assert_unchanged()


def test_raw_logits_refuse_bad_features_and_native_returns(source, monkeypatch):
    torch = require_torch()
    learner, batch = source
    before = learner.snapshot_state()
    with pytest.raises(ValueError, match="float64"):
        learner.predict_logits(batch.features.astype(np.float32))
    bad = batch.features.copy()
    bad[0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        learner.predict_logits(bad)
    name = "forward_logits" if learner.policy.method == "bp" else "predict_logits"
    with monkeypatch.context() as patch:
        patch.setattr(learner._head, name, lambda values: torch.zeros((1, 1)))
        with pytest.raises(ValueError, match="geometry"):
            learner.predict_logits(batch.features)
    with monkeypatch.context() as patch:
        patch.setattr(learner._head, name, lambda values: torch.full((12, 6), float("nan")))
        with pytest.raises(ValueError, match="finite"):
            learner.predict_logits(batch.features)
    assert _same_values(before, learner.snapshot_state())
