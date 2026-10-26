"""Exact weighted native assignment oracle on disposable synthetic TRAIN owners."""

from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from src.adapters import torch_catalogue_weighted as weighted
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

WEIGHTS = (1.8, 1.8, 0.36, 1.8, 1.0, 1.8, 1.0)


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
@pytest.mark.parametrize("steps", [2, 64])
def test_should_match_weighted_native_candidate_without_mutating_source(
    method, steps, record_property
):
    torch = require_torch()
    policy = native.TorchFeaturePolicy(method, 296, 8, 17, num_classes=7)
    learner = InvariantTorchFeatureLearner(
        InvariantTorchFeaturePolicy(replace(policy, inference_steps=steps))
    )
    sources = pair()
    catalogue = weighted.prepare_train_catalogue(sources)
    snapshot, retained = learner.snapshot_state(), deepcopy(learner.snapshot_state())
    report = diagnose_pc_update(snapshot, sources, class_weights=WEIGHTS)
    assert_tree(torch, snapshot, retained)
    assert_tree(torch, learner.snapshot_state(), retained)
    json.dumps(report, allow_nan=False)

    diagnostic = weighted.train_weighted_catalogue(learner, catalogue, WEIGHTS)
    after = learner.snapshot_state()
    assert report["fixed_state_weighted_energy_before"] == diagnostic.value
    parameters = {
        name: snapshot.native_state.head_state[name].clone().requires_grad_(True)
        for name in native._PARAMETERS
    }
    transitions = [step for source in sources for step in source.transitions]
    raw = np.array([step.observation for step in transitions], dtype=np.float64)
    x = torch.from_numpy(
        np.array([canonicalize_public_history(step.observation) for step in transitions])
    ).float()
    y = torch.tensor([step.action for step in transitions])
    w = torch.tensor(WEIGHTS)[y]
    prior = torch.tanh(x @ parameters[native._PARAMETERS[0]] + parameters[native._PARAMETERS[1]])
    logits = prior @ parameters[native._PARAMETERS[2]] + parameters[native._PARAMETERS[3]]
    row_loss = torch.nn.functional.cross_entropy(logits, y, reduction="none")
    loss = (w * row_loss).sum() / w.sum()
    gradients = tuple(g.numpy() for g in torch.autograd.grad(loss, tuple(parameters.values())))
    deltas = tuple(
        (after.native_state.head_state[name] - snapshot.native_state.head_state[name]).numpy()
        for name in native._PARAMETERS
    )
    expected = compare_gradient_geometry(gradients, deltas)
    assert report["gradient_delta_geometry"] == expected
    for i, name in enumerate(native._PARAMETERS):
        assert report["per_parameter_geometry"][name] == compare_gradient_geometry(
            (gradients[i],), (deltas[i],)
        )
    probabilities = learner.predict(raw)
    ordinary_ce = float(-np.log(probabilities[np.arange(len(y)), y.numpy()]).mean())
    losses = -np.log(probabilities[np.arange(len(y)), y.numpy()])
    expected_weighted_ce = float((w.numpy() * losses).sum() / w.numpy().sum())
    assert report["counterfactual_prior_mean_ce"] == pytest.approx(ordinary_ce, abs=1e-6)
    assert report["counterfactual_prior_weighted_mean_ce"] == pytest.approx(
        expected_weighted_ce, abs=1e-6
    )
    assert report["work"]["native_train_calls"] == report["work"]["optimizer_steps"] == 0
    assert report["work"]["published_updates"] == 0
    assert report["work"]["proposal_calls_completed"] == 1
    assert after.completed_updates == 1 and after.training_examples == len(transitions)
    assert_tree(torch, snapshot, retained)
    record_property("oracle_native_updates", 1)
    record_property("oracle_training_rows", len(transitions))
    record_property("oracle_predictions", 1)
    record_property("oracle_prediction_rows", len(transitions))
    record_property("oracle_nominal_latent_row_iterations", len(transitions) * steps)
