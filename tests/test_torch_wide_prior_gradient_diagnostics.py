"""Four H32 initial-state engineering checks; no task training or environment.

Each method pays for exactly one two-episode TRAIN diagnostic. The synthetic
catalogue is shared with the scripted route tests; this supplies no acquisition,
interference, optimizer-direction or comparative scientific evidence.
"""

from copy import deepcopy
import json
import pickle
import random

import numpy as np
import pytest

from src.adapters import torch_feature_gradient_diagnostics as diagnostics
from src.adapters import torch_feature_learner as native
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeaturePolicy,
)
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.shared.torch_runtime import require_torch
from test_minigrid_pc_update_route import sources
from test_torch_feature_learner import assert_tree, tensors


METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")


def _assert_work(report, observed, record_property):
    work = report["work"]
    assert work["snapshot_validations_attempted"] == work["snapshot_validations_completed"] == 1
    for operation in ("projection", "forward", "backward"):
        assert work[f"{operation}_calls_attempted"] == work[f"{operation}_calls_completed"] == 2
        assert work[f"{operation}_rows_attempted"] == work[f"{operation}_rows_completed"] == 18
    assert observed == dict(projection_rows=18, forward_calls=2, backward_calls=2, forward_rows=18)
    assert work["projection_array_bytes_returned"] == 18 * 296 * 8
    assert (
        work["validation_temporary_optimizers"]
        == work["optimizer_steps"]
        == work["published_updates"]
        == 0
    )
    assert work["failure_work_may_be_unobserved"] is False
    record_property("pair_calls_completed", 1)
    for key, value in work.items():
        record_property(key, value)
    record_property("native_train_calls", 0)


def _observe_native_work(monkeypatch, torch):
    observed = dict(projection_rows=0, forward_calls=0, backward_calls=0, forward_rows=0)
    project = diagnostics.canonicalize_public_history
    forward, backward = diagnostics._prior_logits, torch.autograd.grad

    def projection(*args):
        observed["projection_rows"] += 1
        return project(*args)

    def logits(parameters, values, runtime):
        observed["forward_calls"] += 1
        observed["forward_rows"] += len(values)
        return forward(parameters, values, runtime)

    def gradients(*args, **kwargs):
        observed["backward_calls"] += 1
        return backward(*args, **kwargs)

    monkeypatch.setattr(diagnostics, "canonicalize_public_history", projection)
    monkeypatch.setattr(diagnostics, "_prior_logits", logits)
    monkeypatch.setattr(torch.autograd, "grad", gradients)
    return observed


@pytest.mark.parametrize("method", METHODS)
def test_should_read_h32_prior_pair_without_training_or_changing_bound_state(
    method, monkeypatch, record_property
):
    torch = require_torch()
    state = InvariantTorchFeatureLearner(
        InvariantTorchFeaturePolicy(native.TorchFeaturePolicy(method, 296, 32, 17, num_classes=7))
    ).snapshot_state()
    assert state.policy.hidden_dim == 32 and state.policy.native_policy.inference_steps == 2
    assert state.completed_updates == state.training_examples == 0
    original_episodes = tuple(packet.episode for packet in sources()[:2])
    episodes = deepcopy(original_episodes)
    assert all(
        owned is not original
        and owned.role == "train"
        and owned.context == "A"
        and len(owned.transitions) == 9
        and len(owned.transitions[0].observation) == 296
        for owned, original in zip(episodes, original_episodes, strict=True)
    )
    before, input_bytes = deepcopy(state), pickle.dumps((original_episodes, episodes))
    marks = [(value.requires_grad, value.grad) for value in tensors(torch, state)]
    rng = torch.get_rng_state().clone(), pickle.dumps(np.random.get_state()), random.getstate()

    def forbidden(*args, **kwargs):
        pytest.fail("H32 prior diagnostics may not train, construct optimizers or step them")

    monkeypatch.setattr(native.TorchFeatureLearner, "train_batch", forbidden)
    monkeypatch.setattr(InvariantTorchFeatureLearner, "train_batch", forbidden)
    monkeypatch.setattr(torch.optim.SGD, "__init__", forbidden)
    monkeypatch.setattr(torch.optim.SGD, "step", forbidden)
    observed = _observe_native_work(monkeypatch, torch)
    record_property("pair_calls_attempted", 1)
    try:
        with torch.no_grad():
            report = diagnostics.diagnose_prior_gradients(state, episodes)
            assert not torch.is_grad_enabled()
    finally:
        # Keep observed attempted work even if a native call or assertion fails.
        for key, value in observed.items():
            record_property(f"observed_{key}_attempted", value)
    assert torch.is_grad_enabled()
    assert report["status"] == "complete" and report["reason"] is None
    assert report["definition"] == diagnostics.DEFINITION
    assert report["objective"] == native.TORCH_CE_ID
    assert report["interpretation"] == "prior_not_native_pc_energy"
    assert [row["episode_id"] for row in report["losses"]] == [e.episode_id for e in episodes]
    assert all(row["rows"] == 9 and row["context"] == "A" for row in report["losses"])
    assert report["joint_geometry"] is not None
    assert set(report["per_parameter_geometry"]) == set(native._PARAMETERS)
    json.dumps(report, allow_nan=False)
    _assert_work(report, observed, record_property)
    assert_tree(torch, state, before)
    assert _same_state(state, before)
    assert [(value.requires_grad, value.grad) for value in tensors(torch, state)] == marks
    assert pickle.dumps((original_episodes, episodes)) == input_bytes
    assert torch.equal(torch.get_rng_state(), rng[0])
    assert pickle.dumps(np.random.get_state()) == rng[1] and random.getstate() == rng[2]
