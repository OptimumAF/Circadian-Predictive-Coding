"""Weighted native BP math and atomicity, without task or evaluation data."""

from copy import deepcopy
import random

import numpy as np
import pytest

from src.adapters import torch_array_weighted as weighted
from src.adapters import torch_feature_learner as native
from src.adapters.torch_decision import _same_values
from src.app.coupled_attribution import AttributionBatch, AttributionStream
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.shared.torch_runtime import require_torch


@pytest.fixture
def torch():
    return require_torch()


@pytest.fixture(autouse=True)
def preserve_test_rng(torch):
    saved = torch.get_rng_state(), np.random.get_state(), random.getstate(), torch.is_grad_enabled()
    yield
    torch.set_rng_state(saved[0])
    np.random.set_state(saved[1])
    random.setstate(saved[2])
    torch.set_grad_enabled(saved[3])


def model(method="bp"):
    return native.TorchFeatureLearner(native.TorchFeaturePolicy(method, 3, 3, 17, num_classes=3))


def batch():
    return np.array(
        [[0.3, -0.2, 0.1], [-0.5, 0.4, 0.2], [0.7, 0.3, -0.6], [0.2, 0.8, 0.1]]
    ), np.array([[0.0], [0.0], [1.0], [2.0]])


def test_should_match_exact_ordinary_bp_for_all_one_weights_and_continuation():
    ordinary, other = model(), model()
    x, y = batch()
    for _ in range(2):
        plain = ordinary.train_batch(x, y)
        result = weighted.train_class_weighted_batch(other, x, y, (1.0, 1.0, 1.0))
        assert result.definition == weighted.TORCH_ARRAY_WEIGHTED_CE_ID
        assert result.value == plain.value
        assert _same_values(ordinary.snapshot_state(), other.snapshot_state())
    assert other.completed_updates == 2 and other.training_examples == 8


def test_should_match_independent_per_row_weighted_ce_gradient_and_momentum(torch):
    learner = model()
    x, y = batch()
    weights = (0.5, 1.0, 2.0)
    original = learner.snapshot_state()
    head, optimizer = native._restore_components(original)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        values = torch.from_numpy(x.astype(np.float32))
        labels = torch.from_numpy(y[:, 0].astype(np.int64))
        per_row = torch.nn.functional.cross_entropy(
            head.forward_logits(values), labels, reduction="none"
        )
        target_weights = torch.tensor(weights, dtype=torch.float32)[labels]
        loss = (per_row * target_weights).sum() / target_weights.sum()
        loss.backward()
        optimizer.step()
        result = weighted.train_class_weighted_batch(learner, x, y, weights)
        assert result.value == pytest.approx(float(loss.item()), abs=1e-6)
    expected = native._state(head, optimizer, original.policy, 2, 8)
    actual = learner.snapshot_state()
    assert actual.gradients is not None and expected.gradients is not None
    assert actual.optimizer_state is not None and expected.optimizer_state is not None
    for key in native._PARAMETERS:
        assert torch.allclose(
            actual.head_state[key], expected.head_state[key], atol=1e-7, rtol=1e-6
        )
        assert torch.allclose(actual.gradients[key], expected.gradients[key], atol=1e-7, rtol=1e-6)
    for key in actual.optimizer_state["state"]:
        assert torch.allclose(
            actual.optimizer_state["state"][key]["momentum_buffer"],
            expected.optimizer_state["state"][key]["momentum_buffer"],
            atol=1e-7,
            rtol=1e-6,
        )
    assert np.array_equal(x, batch()[0]) and np.array_equal(y, batch()[1])


@pytest.mark.parametrize(
    "weights",
    [
        (1.0, 1.0),
        (0.0, 1.0, 1.0),
        (True, 1.0, 1.0),
        (float("nan"), 1.0, 1.0),
        [1.0, 1.0, 1.0],
        (1e7, 1.0, 1.0),
    ],
)
def test_should_refuse_invalid_weights_before_candidate_work(weights, monkeypatch):
    learner = model()
    saved = learner.snapshot_state()
    x, y = batch()
    monkeypatch.setattr(native, "_restore_components", lambda *a: pytest.fail("invalid staged"))
    with pytest.raises(ValueError):
        weighted.train_class_weighted_batch(learner, x, y, weights)
    assert _same_values(saved, learner.snapshot_state())


@pytest.mark.parametrize("method", ["pc", "neutral_cpc", "gated_cpc"])
def test_should_refuse_other_algorithms_before_staging(method, monkeypatch):
    learner = model(method)
    x, y = batch()
    monkeypatch.setattr(native, "_restore_components", lambda *a: pytest.fail("unsupported staged"))
    with pytest.raises(ValueError):
        weighted.train_class_weighted_batch(learner, x, y, (1.0, 1.0, 1.0))


@pytest.mark.parametrize("fault", ["features", "labels", "mode"])
def test_should_refuse_invalid_arrays_or_mode_before_staging(fault, monkeypatch, torch):
    learner = model()
    x, y = batch()
    if fault == "features":
        x = x.astype(np.float32)
    elif fault == "labels":
        y[0, 0] = 3.0
    else:
        torch.set_grad_enabled(False)
    monkeypatch.setattr(native, "_restore_components", lambda *a: pytest.fail("invalid staged"))
    with pytest.raises(ValueError):
        weighted.train_class_weighted_batch(learner, x, y, (1.0, 1.0, 1.0))


def test_should_not_commit_a_completed_step_when_its_return_is_lost(monkeypatch, torch):
    learner = model()
    saved = learner.snapshot_state()
    payload = learner._payload
    x, y = batch()
    original = torch.optim.SGD.step

    def lost(self, *a, **kw):
        original(self, *a, **kw)
        raise RuntimeError("lost weighted step")

    monkeypatch.setattr(torch.optim.SGD, "step", lost)
    with pytest.raises(RuntimeError, match="lost weighted"):
        weighted.train_class_weighted_batch(learner, x, y, (0.5, 1.0, 2.0))
    assert learner._payload is payload and _same_values(saved, learner.snapshot_state())


@pytest.mark.parametrize(
    "fault",
    [
        "source",
        "policy",
        "gradient",
        "optimizer",
        "input",
        "torch_rng",
        "numpy_rng",
        "python_rng",
        "mode",
    ],
)
def test_should_refuse_foreign_changes_without_erasing_them(fault, monkeypatch, torch):
    learner = model()
    x, y = batch()
    before = learner.snapshot_state()
    payload = learner._payload
    original = torch.optim.SGD.step

    def corrupt(self, *a, **kw):
        result = original(self, *a, **kw)
        if fault == "source":
            learner._head.bias_output.data.add_(1)
        elif fault == "policy":
            learner._policy = deepcopy(learner._policy)
        elif fault == "gradient":
            learner._head.bias_output.grad = torch.zeros_like(learner._head.bias_output)
        elif fault == "optimizer":
            learner._optimizer.state = deepcopy(learner._optimizer.state)
        elif fault == "input":
            x[0, 0] += 1
        elif fault == "torch_rng":
            torch.rand(1)
        elif fault == "numpy_rng":
            np.random.random()
        elif fault == "python_rng":
            random.random()
        else:
            torch.set_grad_enabled(False)
        return result

    monkeypatch.setattr(torch.optim.SGD, "step", corrupt)
    with pytest.raises(ValueError, match="changed"):
        weighted.train_class_weighted_batch(learner, x, y, (0.5, 1.0, 2.0))
    assert learner._payload is payload and learner.completed_updates == 0
    if fault == "source":
        assert not _same_values(before, learner.snapshot_state())
    if fault == "input":
        assert x[0, 0] == pytest.approx(1.3)
    if fault == "gradient":
        assert learner._head.bias_output.grad is not None


def test_should_run_two_arrived_train_updates_in_existing_stream_only():
    class Weighted(native.TorchFeatureLearner):
        def fork(self):
            child = Weighted(self.policy)
            child.restore_state(self.snapshot_state())
            return child

        def train_batch(self, features, targets):
            return weighted.train_class_weighted_batch(self, features, targets, (0.5, 1.0, 2.0))

    owner = Weighted(native.TorchFeaturePolicy("bp", 3, 3, 17, num_classes=3))
    initial = owner.snapshot_state()
    x, y = batch()
    packets = tuple(
        AttributionBatch(
            "train",
            f"phase-{i}",
            tuple(f"{i}-{j}" for j in range(4)),
            x.copy(),
            y.copy(),
            num_classes=3,
        )
        for i in range(2)
    )
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=2, max_wall_seconds=15), lambda: 0.0
    )
    stream = AttributionStream(
        owner,
        packets,
        (),
        checkpoint_updates=(0, 2),
        label_delay=1,
        budget=budget,
        training_only=True,
    )
    stream.run_until(2)
    report = stream.finish()
    assert (
        report["completed_updates"] == 2
        and report["training_examples"] == 8
        and report["score_forward_calls"] == 0
    )
    assert len(stream.applied_updates) == 2 and all(
        r.observed_at + 1 == r.arrived_at <= r.applied_at for r in stream.applied_updates
    )
    assert _same_values(owner.snapshot_state(), initial)
    assert _same_values(stream._runtime.actor.snapshot_state().state, initial)


@pytest.mark.parametrize("when", ["before", "during"])
def test_should_refuse_same_valued_optimizer_parameter_rebinding(when, monkeypatch, torch):
    learner = model()
    x, y = batch()
    saved = learner.snapshot_state()
    payload = learner._payload

    def rebind():
        learner._optimizer.param_groups[0]["params"] = [
            torch.nn.Parameter(p.detach().clone(), requires_grad=p.requires_grad)
            for p in learner._optimizer.param_groups[0]["params"]
        ]
        # Serialized parameter IDs normalize; value equality alone misses this.
        assert _same_values(saved, learner.snapshot_state())

    if when == "before":
        rebind()
        monkeypatch.setattr(
            native, "_restore_components", lambda *a: pytest.fail("corrupt donor staged")
        )
    else:
        original = torch.optim.SGD.step

        def corrupt(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            rebind()
            return result

        monkeypatch.setattr(torch.optim.SGD, "step", corrupt)
    with pytest.raises(ValueError, match="changed"):
        weighted.train_class_weighted_batch(learner, x, y, (0.5, 1.0, 2.0))
    assert learner._payload is payload and learner.completed_updates == 0
    assert all(
        p is not getattr(learner._head, name)
        for p, name in zip(
            learner._optimizer.param_groups[0]["params"], native._PARAMETERS, strict=True
        )
    )
