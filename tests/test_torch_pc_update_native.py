"""Actual native assignment oracle on owned synthetic TRAIN episodes only."""

from copy import deepcopy
import json

import numpy as np
import pytest

from src.adapters import torch_feature_learner as native
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeaturePolicy,
)
from src.adapters.torch_pc_update_diagnostics import diagnose_pc_update
from src.core.gradient_geometry import compare_gradient_geometry
from src.core.minigrid_invariant_encoding import canonicalize_public_history
from src.shared.torch_runtime import require_torch
from test_torch_feature_gradient_diagnostics import pair
from test_torch_feature_learner import assert_tree


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
def test_should_match_actual_native_assignment_without_mutating_original(method, record_property):
    torch = require_torch()
    policy = native.TorchFeaturePolicy(method, 296, 8, 17, num_classes=7)
    learner = InvariantTorchFeatureLearner(InvariantTorchFeaturePolicy(policy))
    snapshot = learner.snapshot_state()
    retained = deepcopy(snapshot)
    sources = pair()
    transitions = [step for episode in sources for step in episode.transitions]
    raw = np.array([step.observation for step in transitions], dtype=np.float64)
    labels = np.array([[step.action] for step in transitions], dtype=np.float64)
    report = diagnose_pc_update(snapshot, sources)
    assert_tree(torch, snapshot, retained)
    assert_tree(torch, learner.snapshot_state(), retained)
    json.dumps(report, allow_nan=False)

    # Actual native operation is an oracle on a disposable learner, not performed
    # by the readout. Its physical work is paid separately in engineering receipts.
    diagnostic = learner.train_batch(raw, labels)
    after = learner.snapshot_state()
    near = report["legacy_diagnostic"]
    assert near["definition"] == diagnostic.definition
    assert near["value"] == pytest.approx(diagnostic.value, abs=1e-8)
    parameters = {
        name: snapshot.native_state.head_state[name].clone().requires_grad_(True)
        for name in native._PARAMETERS
    }
    x = torch.from_numpy(
        np.array([canonicalize_public_history(step.observation) for step in transitions])
    ).float()
    y = torch.tensor([step.action for step in transitions])
    hidden = torch.tanh(x @ parameters[native._PARAMETERS[0]] + parameters[native._PARAMETERS[1]])
    logits = hidden @ parameters[native._PARAMETERS[2]] + parameters[native._PARAMETERS[3]]
    loss = torch.nn.functional.cross_entropy(logits, y)
    gradients = tuple(g.numpy() for g in torch.autograd.grad(loss, tuple(parameters.values())))
    deltas = tuple(
        (after.native_state.head_state[name] - snapshot.native_state.head_state[name]).numpy()
        for name in native._PARAMETERS
    )
    expected = compare_gradient_geometry(gradients, deltas)
    for key in ("first_norm", "second_norm", "dot", "cosine"):
        assert report["gradient_delta_geometry"][key] == pytest.approx(expected[key], abs=1e-12)
    for i, name in enumerate(native._PARAMETERS):
        expected = compare_gradient_geometry((gradients[i],), (deltas[i],))
        for key in ("first_norm", "second_norm", "dot", "cosine"):
            assert report["per_parameter_geometry"][name][key] == pytest.approx(
                expected[key], abs=1e-12
            )
    probabilities = learner.predict(raw)
    actual_ce = float(
        -np.log(probabilities[np.arange(len(labels)), labels[:, 0].astype(int)]).mean()
    )
    assert report["counterfactual_prior_mean_ce"] == pytest.approx(actual_ce, abs=1e-6)
    assert report["prior_ce_change"] == pytest.approx(actual_ce - float(loss.item()), abs=1e-6)
    assert report["work"]["native_train_calls"] == report["work"]["optimizer_steps"] == 0
    assert report["work"]["published_updates"] == 0
    assert_tree(torch, snapshot, retained)
    assert after.completed_updates == 1 and after.training_examples == 4
    record_property("oracle_native_updates", 1)
    record_property("oracle_training_rows", 4)
    record_property("oracle_predictions", 1)
    record_property("oracle_prediction_rows", 4)
