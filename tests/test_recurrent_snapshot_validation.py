"""Exact SGD metadata and owned copies without validation-time optimizers."""

from copy import deepcopy
import pickle
import time

import numpy as np
import pytest

import src.adapters.torch_recurrent_learner as port
from src.shared.torch_runtime import require_torch
from test_torch_feature_learner import assert_tree, forbidden, tensors
from test_torch_recurrent_learner import batch, logits, policy


@pytest.fixture(scope="module")
def learned():
    owner = port.TorchRecurrentLearner(policy())
    owner.train_batch(*batch())
    return owner.snapshot_state()


@pytest.mark.parametrize("rate,momentum", [(0.03, 0.9), (1, 0)])
def test_should_match_installed_default_sgd_metadata_and_two_update_state(rate, momentum):
    torch = require_torch()
    rng = torch.get_rng_state().clone(), pickle.dumps(np.random.get_state())
    config = policy(learning_rate=rate, momentum=momentum)
    owner = port.TorchRecurrentLearner(config)
    initial = owner.snapshot_state()
    parameters = {
        key: value.clone().requires_grad_(True) for key, value in initial.parameters.items()
    }
    default = torch.optim.SGD(list(parameters.values()), lr=rate, momentum=momentum)
    assert_tree(torch, initial.optimizer_state, default.state_dict())
    features, targets = batch()
    for _ in range(2):
        default.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(
            logits(torch, parameters, torch.tensor(features, dtype=torch.float32)),
            torch.tensor(targets[:, 0], dtype=torch.long),
        )
        loss.backward()
        default.step()
        assert owner.train_batch(features, targets).value == float(loss.detach().item())
        actual = owner.snapshot_state()
        assert_tree(torch, actual.parameters, {k: v.detach() for k, v in parameters.items()})
        assert_tree(torch, actual.gradients, {k: v.grad for k, v in parameters.items()})
        assert_tree(torch, actual.optimizer_state, default.state_dict())
    assert torch.equal(torch.get_rng_state(), rng[0])
    assert pickle.dumps(np.random.get_state()) == rng[1]


@pytest.mark.parametrize(
    "field,value",
    [
        ("lr", 0.04),
        ("lr", np.float64(0.03)),
        ("lr", float("nan")),
        ("momentum", 0),
        ("momentum", float("inf")),
        ("dampening", 0.0),
        ("dampening", 1),
        ("weight_decay", 0.0),
        ("nesterov", 0),
        ("maximize", True),
        ("foreach", False),
        ("differentiable", True),
        ("fused", False),
        ("params", (0, 1, 2, 3, 4)),
        ("params", [False, 1, 2, 3, 4]),
        ("params", [0, 2, 1, 3, 4]),
        ("params", [0, 1, 2, 3, 3]),
        ("params", [0, 1, 2, 3]),
        ("extra", 0),
    ],
)
def test_should_refuse_changed_group_metadata_without_native_setup(
    monkeypatch, learned, field, value
):
    incoming = deepcopy(learned)
    incoming.optimizer_state["param_groups"][0][field] = value
    monkeypatch.setattr(port, "_optimizer", forbidden)
    with pytest.raises(ValueError):
        port._snapshot(incoming)


@pytest.mark.parametrize(
    "fault", ["missing", "tuple", "extra_group", "momentum_nan", "momentum_extra"]
)
def test_should_keep_exact_group_and_momentum_guards(monkeypatch, learned, fault):
    state = deepcopy(learned)
    if fault == "missing":
        del state.optimizer_state["param_groups"][0]["foreach"]
    elif fault == "tuple":
        state.optimizer_state["param_groups"] = tuple(state.optimizer_state["param_groups"])
    elif fault == "extra_group":
        state.optimizer_state["param_groups"].append(
            deepcopy(state.optimizer_state["param_groups"][0])
        )
    elif fault == "momentum_nan":
        state.optimizer_state["state"][0]["momentum_buffer"][0, 0] = float("nan")
    else:
        state.optimizer_state["state"][0]["extra"] = 0
    monkeypatch.setattr(port, "_optimizer", forbidden)
    with pytest.raises(ValueError):
        port._snapshot(state)


def test_should_copy_full_owned_state_with_no_validation_optimizer_or_parameter_clones(
    monkeypatch,
    learned,
):
    torch = require_torch()
    before = deepcopy(learned)
    monkeypatch.setattr(port, "_optimizer", forbidden)
    monkeypatch.setattr(torch.optim, "SGD", forbidden)
    monkeypatch.setattr(torch.Tensor, "clone", forbidden)
    for _ in range(16):
        owned = port._snapshot(learned)
        assert_tree(torch, owned, before)
        for original, copied in zip(tensors(torch, learned), tensors(torch, owned), strict=True):
            assert original.data_ptr() != copied.data_ptr()
            copied.fill_(123)
        owned.optimizer_state["param_groups"][0]["lr"] = 99
        assert_tree(torch, learned, before)


def test_should_measure_only_finite_validation_component_cost(learned, record_property):
    torch = require_torch()
    # Legacy metadata work only:16old validations, not another learning run.
    started = time.perf_counter()
    for _ in range(16):
        parameters = {
            k: v.detach().clone().requires_grad_(True) for k, v in learned.parameters.items()
        }
        expected = torch.optim.SGD(
            list(parameters.values()),
            lr=learned.policy.learning_rate,
            momentum=learned.policy.momentum,
        ).state_dict()["param_groups"][0]
        assert_tree(torch, expected, learned.optimizer_state["param_groups"][0])
    legacy = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(16):
        port._validate_optimizer(learned)
    current = time.perf_counter() - started
    record_property("legacy_metadata_component_seconds", legacy)
    record_property("current_full_optimizer_validation_seconds", current)
    record_property("iterations", 16)
    assert legacy >= 0 and current >= 0
