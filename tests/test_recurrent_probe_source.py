"""Small real recurrent bridge checks; exactly three SGD calls in one happy path."""

from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from src.adapters import recurrent_probe_source as bridge
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.adapters.torch_recurrent_learner import TorchRecurrentLearner, TorchRecurrentPolicy
from src.adapters.torch_reward_update_learner import TorchRewardUpdateLearner
from src.adapters.torch_demonstration_reward_learner import TorchDemonstrationRewardLearner
from src.app.coupled_attribution import AttributionBatch
from src.app.fresh_task_probe import run_fresh_task_probe
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from src.shared.torch_runtime import require_torch


def policy():
    return TorchRecurrentPolicy(
        feature_dim=8,
        hidden_dim=4,
        num_classes=5,
        seed=17,
        learning_rate=0.03,
        momentum=0.9,
        max_episode_steps=8,
    )


def arrays(offset=0) -> tuple[np.ndarray, np.ndarray]:
    features = (np.arange(64, dtype=np.float64).reshape(8, 8) + offset) / 128.0
    targets = (np.arange(8, dtype=np.float64) % 5).reshape(8, 1)
    return features, targets


def tensors(value):
    torch = require_torch()
    if type(value) is torch.Tensor:
        yield value
    elif type(value) is dict:
        for item in value.values():
            yield from tensors(item)
    elif type(value) in (tuple, list):
        for item in value:
            yield from tensors(item)


def assert_detached(left, right):
    for name in ("parameters", "gradients", "optimizer_state"):
        for a, b in zip(tensors(getattr(left, name)), tensors(getattr(right, name)), strict=True):
            assert a.data_ptr() != b.data_ptr()


def test_should_restore_full_trained_state_and_run_only_disposable_array_probe_updates(monkeypatch):
    torch = require_torch()
    steps = []
    original_step = torch.optim.SGD.step

    def step(optimizer, *args, **kwargs):
        steps.append("SGD")
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.SGD, "step", step)
    benchmark = TorchRecurrentLearner(policy())
    benchmark.train_batch(*arrays())
    state = benchmark.snapshot_state()
    expected = deepcopy(state)
    boundaries = []
    source = bridge.recurrent_probe_source(state, check_boundary=lambda: boundaries.append("check"))
    restored = source.snapshot_state()
    assert type(source) is TorchRecurrentLearner and len(boundaries) == 4
    assert _same_state(restored, expected)
    assert restored.completed_updates == 1 and restored.training_examples == 8
    assert len(restored.optimizer_state["state"]) == 5
    assert all(value is not None for value in restored.gradients.values())
    assert_detached(state, restored)

    training = tuple(
        AttributionBatch(
            "probe_train",
            "E",
            tuple(f"probe-train-{i}-{j}" for j in range(8)),
            *arrays(i + 1),
            num_classes=5,
        )
        for i in range(2)
    )
    evaluation = (
        AttributionBatch(
            "probe_test", "E", tuple(f"probe-test-{j}" for j in range(8)), *arrays(3), num_classes=5
        ),
    )
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    report = run_fresh_task_probe(
        source,
        state,
        training,
        evaluation,
        budget=budget,
        checkpoint_updates=(0, 1, 2),
        timer=lambda: 0.0,
    )
    assert report["status"] == "complete", report["reason"]
    assert report["completed_updates"] == budget.updates_completed == 2
    assert report["completed_training_examples"] == 16
    assert report["score_forward_calls_completed"] == 3
    assert [point["updates"] for point in report["checkpoints"]] == [0, 1, 2]
    assert steps == ["SGD"] * 3
    assert _same_state(state, expected)
    assert _same_state(benchmark.snapshot_state(), expected)
    assert _same_state(source.snapshot_state(), expected)
    state.parameters["weight_input"].fill_(7)
    state.gradients["weight_input"].fill_(8)
    state.optimizer_state["state"][0]["momentum_buffer"].fill_(9)
    assert _same_state(source.snapshot_state(), expected)
    assert _same_state(benchmark.snapshot_state(), expected)


def test_should_keep_typed_benchmark_record_admission_closed_to_probe_roles(monkeypatch):
    torch = require_torch()
    monkeypatch.setattr(torch.optim.SGD, "step", lambda *a, **k: pytest.fail("unexpected SGD"))
    native = TorchRecurrentLearner(policy())
    before = native.snapshot_state()
    transition = DemonstrationTransition(0, (0.0,) * 8, 1, 0.0, (0.0,) * 8, True, 5)
    packet = DemonstrationEpisode("probe-role", "E", "probe_train", (transition,), 5)
    with pytest.raises(ValueError, match="train-role"):
        native.train_episode(packet)
    benchmark = TorchRewardUpdateLearner(policy())
    with pytest.raises(ValueError, match="RewardUpdate"):
        invalid_arrays: Any = arrays()
        benchmark.train_batch(*invalid_arrays)
    assert _same_state(native.snapshot_state(), before)
    assert benchmark.completed_updates == benchmark.training_examples == 0
    demonstration_benchmark = TorchDemonstrationRewardLearner(policy())
    with pytest.raises(ValueError, match="features must be"):
        demonstration_benchmark.train_batch(*invalid_arrays)
    assert demonstration_benchmark.completed_updates == 0


@pytest.mark.parametrize("value", [None, object(), {}, 1])
def test_should_refuse_wrong_snapshot_type_before_constructor_or_callback(monkeypatch, value):
    def forbidden(*args, **kwargs):
        pytest.fail("external construction or callback reached")

    monkeypatch.setattr(bridge, "TorchRecurrentLearner", forbidden)
    with pytest.raises(ValueError, match="exact complete"):
        bridge.recurrent_probe_source(value, check_boundary=forbidden)


@pytest.mark.parametrize(
    "fault",
    [
        "extra",
        "missing",
        "version",
        "counter_bool",
        "rows",
        "nan",
        "optimizer",
        "gradient",
        "policy",
    ],
)
def test_should_refuse_corrupt_snapshot_without_any_sgd(monkeypatch, fault):
    torch = require_torch()
    native = TorchRecurrentLearner(policy())
    state = native.snapshot_state()
    monkeypatch.setattr(torch.optim.SGD, "step", lambda *a, **k: pytest.fail("unexpected SGD"))
    if fault == "extra":
        object.__setattr__(state, "extra", 1)
    elif fault == "missing":
        object.__delattr__(state, "parameters")
    elif fault == "version":
        state = replace(state, format_version=True)
    elif fault == "counter_bool":
        state = replace(state, completed_updates=False)
    elif fault == "rows":
        state = replace(state, training_examples=8)
    elif fault == "nan":
        state.parameters["weight_input"][0, 0] = float("nan")
    elif fault == "optimizer":
        state.optimizer_state["param_groups"][0]["lr"] = 0.4
    elif fault == "gradient":
        state.gradients["weight_input"] = torch.zeros_like(state.parameters["weight_input"])
    else:
        object.__setattr__(state.policy, "hidden_dim", False)
    with pytest.raises(ValueError):
        bridge.recurrent_probe_source(state, check_boundary=lambda: None)


def test_should_refuse_invalid_callback_before_native_construction(monkeypatch):
    state = TorchRecurrentLearner(policy()).snapshot_state()
    monkeypatch.setattr(
        bridge, "TorchRecurrentLearner", lambda *a: pytest.fail("constructor reached")
    )
    with pytest.raises(ValueError, match="callable"):
        invalid: Any = None
        bridge.recurrent_probe_source(state, check_boundary=invalid)


@pytest.mark.parametrize("boundary_index", [1, 2, 3, 4])
def test_should_refuse_input_snapshot_mutation_at_each_external_boundary(
    monkeypatch, boundary_index
):
    torch = require_torch()
    state = TorchRecurrentLearner(policy()).snapshot_state()
    calls = []
    monkeypatch.setattr(torch.optim.SGD, "step", lambda *a, **k: pytest.fail("unexpected SGD"))

    def boundary():
        calls.append("check")
        if len(calls) == boundary_index:
            state.parameters["weight_input"].add_(1)

    with pytest.raises(ValueError, match="snapshot changed"):
        bridge.recurrent_probe_source(state, check_boundary=boundary)
    assert len(calls) == boundary_index


def test_should_propagate_boundary_failure_without_constructing_probe_learner(monkeypatch):
    state = TorchRecurrentLearner(policy()).snapshot_state()
    error = RuntimeError("source isolation failed")
    monkeypatch.setattr(
        bridge, "TorchRecurrentLearner", lambda *a: pytest.fail("constructor reached")
    )

    def boundary():
        raise error

    with pytest.raises(RuntimeError, match="source isolation failed") as caught:
        bridge.recurrent_probe_source(state, check_boundary=boundary)
    assert caught.value is error


@pytest.mark.parametrize("field", ["policy", "parameters", "optimizer_state", "gradients"])
def test_should_refuse_equal_valued_original_snapshot_field_replacement(field):
    state = TorchRecurrentLearner(policy()).snapshot_state()

    def boundary():
        object.__setattr__(state, field, deepcopy(getattr(state, field)))

    with pytest.raises(ValueError, match="snapshot changed"):
        bridge.recurrent_probe_source(state, check_boundary=boundary)


def test_should_refuse_corrupted_restored_readback_without_returning_a_source(monkeypatch):
    state = TorchRecurrentLearner(policy()).snapshot_state()
    original_snapshot = TorchRecurrentLearner.snapshot_state

    def snapshot(learner):
        result = original_snapshot(learner)
        return replace(result, completed_updates=True)

    monkeypatch.setattr(TorchRecurrentLearner, "snapshot_state", snapshot)
    with pytest.raises(ValueError, match="complete snapshot"):
        bridge.recurrent_probe_source(state, check_boundary=lambda: None)


def test_should_refuse_callback_mutation_of_new_learner_after_first_readback(monkeypatch):
    state = TorchRecurrentLearner(policy()).snapshot_state()
    original_constructor = bridge.TorchRecurrentLearner
    constructed = []
    calls = []

    def constructor(bound_policy):
        result = original_constructor(bound_policy)
        constructed.append(result)
        return result

    def boundary():
        calls.append("check")
        if len(calls) == 4:
            changed = constructed[0].snapshot_state()
            changed.parameters["weight_input"].add_(1)
            constructed[0].restore_state(changed)

    monkeypatch.setattr(bridge, "TorchRecurrentLearner", constructor)
    with pytest.raises(ValueError, match="after its restored-state readback"):
        bridge.recurrent_probe_source(state, check_boundary=boundary)
