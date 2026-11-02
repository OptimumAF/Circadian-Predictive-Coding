"""Complete owned snapshots, without fitting, forward passes or gradients."""

from copy import deepcopy
from dataclasses import fields, is_dataclass

import pytest

from src.adapters import torch_feature_learner as native
from src.adapters.torch_array_weighted import _rng
from src.adapters.torch_categorical_readout_fit import _unchanged_owner
from src.adapters.torch_decision import _same_values

METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")


@pytest.fixture(scope="module")
def owners():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    learners = {
        method: native.TorchFeatureLearner(
            native.TorchFeaturePolicy(method, 9, 3, 17, num_classes=6)
        )
        for method in METHODS
    }
    bp = learners["bp"]
    for name in native._PARAMETERS:
        value = getattr(bp._head, name)
        value.grad = torch.zeros_like(value)
        bp._optimizer.state[value]["momentum_buffer"] = torch.zeros_like(value)
    # Synthetic populated schema, not twelve performed updates or row exposure.
    bp._payload = (bp._head, bp._optimizer, 12, 72)
    return torch, learners


def tensors(torch, value):
    if type(value) is torch.Tensor:
        yield value
    elif type(value) is dict:
        for child in value.values():
            yield from tensors(torch, child)
    elif type(value) in (tuple, list):
        for child in value:
            yield from tensors(torch, child)
    elif is_dataclass(value):
        for field in fields(value):
            yield from tensors(torch, getattr(value, field.name))


@pytest.mark.parametrize("method", METHODS)
def test_complete_snapshots_match_actual_values_and_own_separate_storage(owners, method):
    torch, learners = owners
    learner = learners[method]
    rng = _rng(torch)
    expected_parameters = {
        name: getattr(learner._head, name).detach().clone() for name in native._PARAMETERS
    }
    saved, other = learner.snapshot_state(), learner.snapshot_state()
    assert _same_values(saved, other)
    for name, expected in expected_parameters.items():
        assert torch.equal(saved.head_state[name], expected)
        assert saved.head_state[name].data_ptr() != getattr(learner._head, name).data_ptr()
        assert not saved.head_state[name].requires_grad
    if method == "bp":
        assert (saved.completed_updates, saved.training_examples) == (12, 72)
        assert saved.gradients is not None and saved.optimizer_state is not None
        for index, name in enumerate(native._PARAMETERS):
            live = getattr(learner._head, name)
            assert torch.equal(saved.gradients[name], live.grad)
            assert saved.gradients[name].data_ptr() != live.grad.data_ptr()
            momentum = saved.optimizer_state["state"][index]["momentum_buffer"]
            live_momentum = learner._optimizer.state[live]["momentum_buffer"]
            assert torch.equal(momentum, live_momentum)
            assert momentum.data_ptr() != live_momentum.data_ptr()
    elif method == "pc":
        assert torch.equal(saved.head_state["traffic_sum"], learner._head._traffic_sum)
        assert saved.head_state["traffic_sum"].data_ptr() != learner._head._traffic_sum.data_ptr()
    for left, right in zip(tensors(torch, saved), tensors(torch, other), strict=True):
        assert left.data_ptr() != right.data_ptr()
    # Corrupt every returned tensor/policy; neither actual owner nor older view changes.
    with torch.no_grad():
        for value in tensors(torch, saved):
            value.fill_(7)
    object.__setattr__(saved.policy, "seed", 19)
    assert _same_values(other, learner.snapshot_state())
    assert _same_values(rng, _rng(torch))


@pytest.mark.parametrize("method", METHODS)
def test_mutating_live_parameters_does_not_change_published_snapshots(owners, method):
    torch, learners = owners
    learner = learners[method]
    saved = learner.snapshot_state()
    baseline = deepcopy(saved)
    value = learner._head.weight_feature_hidden
    original = value.detach().clone()
    try:
        with torch.no_grad():
            value[0, 0] += 1
        assert _same_values(saved, baseline)
        assert not torch.equal(saved.head_state["weight_feature_hidden"], value)
    finally:
        with torch.no_grad():
            value.copy_(original)


@pytest.mark.parametrize("field", ["parameter", "gradient", "momentum", "policy", "metadata"])
def test_invalid_live_bp_state_is_refused_without_restoration(owners, field):
    torch, learners = owners
    learner = learners["bp"]
    value = learner._head.weight_feature_hidden
    parameter, gradient = value.detach().clone(), value.grad
    momentum = learner._optimizer.state[value]["momentum_buffer"]
    group = learner._optimizer.param_groups[0]
    learning_rate = group["lr"]
    width = learner._head.feature_dim
    try:
        if field == "parameter":
            with torch.no_grad():
                value[0, 0] = float("nan")
        elif field == "gradient":
            value.grad = None
        elif field == "momentum":
            learner._optimizer.state[value]["momentum_buffer"] = momentum.double()
        elif field == "policy":
            group["lr"] = 9.0
        else:
            learner._head.feature_dim = 99
        with pytest.raises(ValueError):
            learner.snapshot_state()
        # A refused snapshot must not repair the invalid source behind the caller.
        if field == "parameter":
            assert bool(torch.isnan(value[0, 0]).item())
        elif field == "gradient":
            assert value.grad is None
        elif field == "momentum":
            assert learner._optimizer.state[value]["momentum_buffer"].dtype == torch.float64
        elif field == "policy":
            assert group["lr"] == 9.0
        else:
            assert learner._head.feature_dim == 99
    finally:
        # Fixture cleanup only after proving production retained foreign evidence.
        with torch.no_grad():
            value.copy_(parameter)
        value.grad = gradient
        learner._optimizer.state[value]["momentum_buffer"] = momentum
        group["lr"] = learning_rate
        learner._head.feature_dim = width


@pytest.mark.parametrize("field", ["parameter", "gradient", "momentum"])
def test_unchanged_owner_still_refuses_finite_foreign_buffer_mutations(owners, field):
    torch, learners = owners
    learner = learners["bp"]
    parameter = learner._head.weight_feature_hidden
    value = (
        parameter
        if field == "parameter"
        else parameter.grad
        if field == "gradient"
        else learner._optimizer.state[parameter]["momentum_buffer"]
    )
    original = value.detach().clone()
    try:
        with pytest.raises(ValueError, match="owner|RNG"):
            with _unchanged_owner(learner) as (_, require):
                require()
                with torch.no_grad():
                    value[0, 0] += 1
                require()
        assert not torch.equal(value, original)
    finally:
        with torch.no_grad():
            value.copy_(original)
