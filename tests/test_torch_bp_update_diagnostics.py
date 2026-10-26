"""Bounded original-BP update readback, with an independent NumPy SGD oracle.

Native case ceiling is15 attempted updates/135 TRAIN rows, including one warmup
and paid failures. All admission cases reject before loading Torch. No final,
probe, environment, teacher, promotion or retained diagnostic model is used.
"""

from copy import deepcopy
from dataclasses import replace
import json
import pickle
import random

import numpy as np
import pytest

from src.adapters import torch_bp_update_diagnostics as diagnostics
from src.adapters import torch_feature_learner as native
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeaturePolicy,
    InvariantTorchFeatureSnapshot,
)
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.demonstration_episode import episode_arrays
from src.core.minigrid_invariant_encoding import canonicalize_public_history
from src.shared.torch_runtime import require_torch
from test_minigrid_invariant_learnability import train_sources
from test_torch_feature_learner import assert_tree

NAMES = ("weight_feature_hidden", "bias_hidden", "weight_hidden_output", "bias_output")
PLANNED_NATIVE_CALLS = 15
PLANNED_NATIVE_ROWS = 135


def policy():
    return InvariantTorchFeaturePolicy(native.TorchFeaturePolicy("bp", 296, 8, 17, num_classes=7))


def episodes():
    sources = train_sources()
    return sources[2].episode, (sources[0].episode, sources[1].episode)


def budget():
    return ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=3, max_replay_examples=18), lambda: 1.0
    )


def forbidden(*args, **kwargs):
    pytest.fail("invalid diagnostic admission loaded Torch or constructed a learner")


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test"])
def test_admission_should_refuse_non_train_before_torch(monkeypatch, position, role):
    wake, replay = episodes()
    values = [wake, *replay]
    values[position] = replace(values[position], role=role)
    snapshot = InvariantTorchFeatureSnapshot(policy(), object())
    monkeypatch.setattr(diagnostics, "require_torch", forbidden)
    with pytest.raises(ValueError):
        diagnostics.diagnose_bp_updates(snapshot, values[0], tuple(values[1:]), budget=budget())


@pytest.mark.parametrize(
    "fault", ["wake_id", "wake_context", "replay_order", "duplicate_replay", "tuple", "short"]
)
def test_admission_should_bind_the_original_complete_cohort_before_torch(monkeypatch, fault):
    wake, replay = episodes()
    if fault == "wake_id":
        wake = replace(wake, episode_id="replacement-B-blue")
    elif fault == "wake_context":
        wake = replace(wake, context="F-blue")
    elif fault == "replay_order":
        replay = tuple(reversed(replay))
    elif fault == "duplicate_replay":
        replay = (replay[0], replay[0])
    elif fault == "tuple":
        replay = list(replay)
    else:
        transition = replace(wake.transitions[0], terminal=True)
        wake = replace(wake, transitions=(transition,))
    monkeypatch.setattr(diagnostics, "require_torch", forbidden)
    with pytest.raises(ValueError):
        diagnostics.diagnose_bp_updates(
            InvariantTorchFeatureSnapshot(policy(), object()), wake, replay, budget=budget()
        )


@pytest.mark.parametrize("fault", ["updates", "replay", "used_update", "used_replay"])
def test_admission_should_require_the_original_fresh_exact_budget_before_torch(monkeypatch, fault):
    session = budget()
    if fault == "updates":
        session.budget = replace(session.budget, max_training_updates=4)
    elif fault == "replay":
        session.budget = replace(session.budget, max_replay_examples=19)
    elif fault == "used_update":
        session.updates_completed = 1
    else:
        session.replay_examples_completed = 1
    monkeypatch.setattr(diagnostics, "require_torch", forbidden)
    wake, replay = episodes()
    with pytest.raises(ValueError):
        diagnostics.diagnose_bp_updates(
            InvariantTorchFeatureSnapshot(policy(), object()), wake, replay, budget=session
        )


@pytest.fixture(scope="module")
def native_work():
    torch = require_torch()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    patch = pytest.MonkeyPatch()
    original_train = native.TorchFeatureLearner.train_batch
    original_ce = torch.nn.functional.cross_entropy
    original_grad = torch.autograd.grad
    original_backward = torch.Tensor.backward
    work = dict(train_calls=0, train_rows=0, forward_calls=0, forward_rows=0, backward_calls=0)

    def train(owner, features, targets):
        work["train_calls"] += 1
        work["train_rows"] += len(features)
        assert work["train_calls"] <= PLANNED_NATIVE_CALLS
        assert work["train_rows"] <= PLANNED_NATIVE_ROWS
        return original_train(owner, features, targets)

    def ce(logits, *args, **kwargs):
        work["forward_calls"] += 1
        work["forward_rows"] += len(logits)
        assert work["forward_calls"] <= 350 and work["forward_rows"] <= 3500
        return original_ce(logits, *args, **kwargs)

    def grad(*args, **kwargs):
        work["backward_calls"] += 1
        assert work["backward_calls"] <= 230
        return original_grad(*args, **kwargs)

    def backward(owner, *args, **kwargs):
        work["backward_calls"] += 1
        assert work["backward_calls"] <= 230
        return original_backward(owner, *args, **kwargs)

    patch.setattr(native.TorchFeatureLearner, "train_batch", train)
    patch.setattr(torch.nn.functional, "cross_entropy", ce)
    patch.setattr(torch.autograd, "grad", grad)
    patch.setattr(torch.Tensor, "backward", backward)
    yield torch, work
    patch.undo()


@pytest.fixture
def counted_native(native_work, record_property):
    torch, work = native_work
    before = dict(work)
    yield torch, work
    for name, count in work.items():
        record_property(name, count - before[name])


def numpy_objective(parameters, selected):
    """Analytic tanh/mean-CE derivative independent of Torch/autograd."""
    transitions = [transition for episode in selected for transition in episode.transitions]
    x = np.array([canonicalize_public_history(t.observation) for t in transitions])
    targets = np.array([t.action for t in transitions])
    hidden = np.tanh(x @ parameters[NAMES[0]] + parameters[NAMES[1]])
    logits = hidden @ parameters[NAMES[2]] + parameters[NAMES[3]]
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    log_probabilities = shifted - np.log(exponentials.sum(axis=1, keepdims=True))
    loss = float(-log_probabilities[np.arange(len(targets)), targets].mean())
    residual = exponentials / exponentials.sum(axis=1, keepdims=True)
    residual[np.arange(len(targets)), targets] -= 1
    residual /= len(targets)
    hidden_error = (residual @ parameters[NAMES[2]].T) * (1 - hidden * hidden)
    return loss, dict(
        zip(
            NAMES,
            (
                x.T @ hidden_error,
                hidden_error.sum(0, keepdims=True),
                hidden.T @ residual,
                residual.sum(0, keepdims=True),
            ),
            strict=True,
        )
    )


def arrays(state):
    return {name: state.head_state[name].numpy().astype(np.float64) for name in NAMES}


def geometry(left, right, names=NAMES):
    first = np.concatenate([left[name].ravel() for name in names])
    second = np.concatenate([right[name].ravel() for name in names])
    first_norm, second_norm = np.linalg.norm(first), np.linalg.norm(second)
    dot = float(first @ second)
    return dict(
        first_norm=float(first_norm),
        second_norm=float(second_norm),
        dot=dot,
        cosine=None if first_norm * second_norm == 0 else dot / (first_norm * second_norm),
    )


def assert_geometry(actual, expected):
    for name, value in expected.items():
        if value is None:
            assert actual[name] is None
        else:
            assert actual[name] == pytest.approx(value, rel=8e-4, abs=3e-7)


@pytest.mark.parametrize("warm", [False, True])
def test_should_match_realized_sgd_momentum_and_old_a_ce_without_changing_source(
    counted_native, monkeypatch, warm
):
    torch, _ = counted_native
    wake, replay = episodes()
    source = InvariantTorchFeatureLearner(policy())
    if warm:
        source.train_batch(*episode_arrays(replay[0]))
    snapshot = source.snapshot_state()
    first_parameter = snapshot.native_state.head_state[NAMES[0]]
    first_parameter.requires_grad_(True)
    first_parameter.grad = torch.full_like(first_parameter, 0.125)
    before, source_before = deepcopy(snapshot), source.snapshot_state()
    saved_gradient = first_parameter.grad.clone()
    input_bytes = pickle.dumps((wake, replay))
    rng = torch.get_rng_state().clone(), pickle.dumps(np.random.get_state()), random.getstate()
    observed = []
    original = native.TorchFeatureLearner.train_batch

    def observe(owner, features, targets):
        old = owner.snapshot_state()
        result = original(owner, features, targets)
        observed.append((old, owner.snapshot_state()))
        return result

    monkeypatch.setattr(native.TorchFeatureLearner, "train_batch", observe)
    session = budget()
    with torch.no_grad():
        report = diagnostics.diagnose_bp_updates(snapshot, wake, replay, budget=session)
        assert not torch.is_grad_enabled()
    assert torch.is_grad_enabled()
    assert report["status"] == "complete"
    assert len(report["steps"]) == len(observed) == 3
    assert report["snapshot_unchanged"] and report["inputs_unchanged"] and report["rng_unchanged"]
    assert (session.updates_completed, session.replay_examples_completed) == (3, 18)
    for index, (step, (old, new), selected) in enumerate(
        zip(report["steps"], observed, (wake, *replay), strict=True)
    ):
        old_parameters, new_parameters = arrays(old), arrays(new)
        old_ce, old_gradient = numpy_objective(old_parameters, replay)
        new_ce, _ = numpy_objective(new_parameters, replay)
        _, update_gradient = numpy_objective(old_parameters, (selected,))
        ids = old.optimizer_state["param_groups"][0]["params"]
        momentum = {}
        momentum_component = {}
        for name, parameter_id in zip(NAMES, ids, strict=True):
            old_buffer = old.optimizer_state["state"].get(parameter_id, {}).get("momentum_buffer")
            momentum[name] = update_gradient[name] + (
                0 if old_buffer is None else 0.9 * old_buffer.numpy().astype(np.float64)
            )
            momentum_component[name] = (
                np.zeros_like(old_parameters[name])
                if old_buffer is None
                else -0.03 * 0.9 * old_buffer.numpy().astype(np.float64)
            )
            np.testing.assert_allclose(
                new_parameters[name],
                old_parameters[name] - 0.03 * momentum[name],
                rtol=2e-5,
                atol=3e-7,
            )
            np.testing.assert_allclose(
                new.optimizer_state["state"][parameter_id]["momentum_buffer"].numpy(),
                momentum[name],
                rtol=3e-5,
                atol=3e-7,
            )
        delta = {name: new_parameters[name] - old_parameters[name] for name in NAMES}
        assert step["episode_id"] == selected.episode_id and step["rows"] == 9
        assert step["phase"] == ("wake" if index == 0 else "replay")
        assert step["native_before_updates"] == old.completed_updates
        assert step["native_after_updates"] == new.completed_updates
        assert step["old_A_ce_before"] == pytest.approx(old_ce, abs=3e-7)
        assert step["old_A_ce_after"] == pytest.approx(new_ce, abs=3e-7)
        assert step["old_A_ce_change"] == pytest.approx(new_ce - old_ce, abs=3e-7)
        assert_geometry(step["old_A_gradient_delta_geometry"], geometry(old_gradient, delta))
        assert_geometry(
            step["old_A_gradient_momentum_geometry"], geometry(old_gradient, momentum_component)
        )
        assert_geometry(step["update_gradient_delta_geometry"], geometry(update_gradient, delta))
        assert step["momentum_expected_max_error"] <= 3e-7
        assert step["parameter_step_expected_max_error"] <= 3e-7
        assert set(step["per_parameter_geometry"]) == set(NAMES)
        for name in NAMES:
            per_parameter = step["per_parameter_geometry"][name]
            assert_geometry(
                per_parameter["old_A_gradient_delta_geometry"],
                geometry(old_gradient, delta, (name,)),
            )
            assert_geometry(
                per_parameter["old_A_gradient_momentum_geometry"],
                geometry(old_gradient, momentum_component, (name,)),
            )
            assert_geometry(
                per_parameter["update_gradient_delta_geometry"],
                geometry(update_gradient, delta, (name,)),
            )
        assert new.completed_updates == old.completed_updates + 1 == int(warm) + index + 1
    work = report["work"]
    assert work["native_train_calls_attempted"] == work["native_train_calls_returned"] == 3
    assert work["native_training_rows_attempted"] == work["native_training_rows_returned"] == 27
    assert work["charged_updates"] == work["optimizer_steps_returned"] == 3
    assert work["charged_replay_rows"] == 18 and work["published_updates"] == 0
    assert_tree(torch, snapshot, before)
    assert_tree(torch, source.snapshot_state(), source_before)
    assert first_parameter.requires_grad and torch.equal(first_parameter.grad, saved_gradient)
    assert pickle.dumps((wake, replay)) == input_bytes
    assert torch.equal(torch.get_rng_state(), rng[0])
    assert pickle.dumps(np.random.get_state()) == rng[1] and random.getstate() == rng[2]
    json.dumps(report, allow_nan=False)


def test_should_retain_paid_attempt_when_second_native_call_fails(counted_native, monkeypatch):
    torch, _ = counted_native
    wake, replay = episodes()
    state = InvariantTorchFeatureLearner(policy()).snapshot_state()
    before = deepcopy(state)
    original, calls = native.TorchFeatureLearner.train_batch, 0

    def fail(owner, features, targets):
        nonlocal calls
        calls += 1
        if calls == 2:
            # Reach the counting boundary, then fail inside native optimizer work.
            with monkeypatch.context() as patch:

                def refused(*args, **kwargs):
                    raise RuntimeError("synthetic paid SGD failure")

                patch.setattr(torch.optim.SGD, "step", refused)
                return original(owner, features, targets)
        return original(owner, features, targets)

    monkeypatch.setattr(native.TorchFeatureLearner, "train_batch", fail)
    session = budget()
    with pytest.raises(diagnostics.GradientDiagnosticFailed) as caught:
        diagnostics.diagnose_bp_updates(state, wake, replay, budget=session)
    report, work = caught.value.report, caught.value.report["work"]
    assert report["status"] == "incomplete" and len(report["steps"]) == 1
    assert work["native_train_calls_attempted"] == 2 and work["native_train_calls_returned"] == 1
    assert work["native_training_rows_attempted"] == 18
    assert work["native_training_rows_returned"] == 9
    assert work["charged_updates"] == session.updates_completed == 2
    assert work["charged_replay_rows"] == session.replay_examples_completed == 9
    assert work["published_updates"] == 0 and work["failure_work_may_be_unobserved"]
    assert_tree(torch, state, before)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "fault", ["limits_owner", "limits_value", "clock", "start", "updates", "clock_floor"]
)
def test_should_refuse_budget_tamper_at_the_next_observed_boundary(counted_native, fault):
    torch, _ = counted_native
    state = InvariantTorchFeatureLearner(policy()).snapshot_state()
    before, session = deepcopy(state), budget()
    wake, replay = episodes()
    mutated = False

    def check():
        nonlocal mutated
        if mutated or session.updates_completed != 1:
            return
        mutated = True
        if fault == "limits_owner":
            session.budget = replace(session.budget)
        elif fault == "limits_value":
            object.__setattr__(session.budget, "max_training_updates", 30)
        elif fault == "clock":
            session.clock = lambda: 1.0
        elif fault == "start":
            session.started_at += 1.0
        elif fault == "updates":
            session.updates_completed = 0
        else:
            session.last_clock += 1.0

    with pytest.raises(diagnostics.GradientDiagnosticFailed) as caught:
        diagnostics.diagnose_bp_updates(state, wake, replay, budget=session, check_boundary=check)
    assert mutated and caught.value.report["status"] == "incomplete"
    assert caught.value.report["work"]["native_train_calls_attempted"] <= 1
    assert caught.value.report["work"]["published_updates"] == 0
    assert_tree(torch, state, before)


def test_should_refuse_inference_mode_without_spending_updates(counted_native):
    torch, work = counted_native
    state = InvariantTorchFeatureLearner(policy()).snapshot_state()
    before, session = work["train_calls"], budget()
    wake, replay = episodes()
    with torch.inference_mode(), pytest.raises(ValueError):
        diagnostics.diagnose_bp_updates(state, wake, replay, budget=session)
    assert work["train_calls"] == before and session.updates_completed == 0


@pytest.mark.parametrize("fault", ["callback", "replay_counter"])
def test_should_refuse_an_unobserved_prepaid_boundary_change(counted_native, fault):
    torch, work = counted_native
    state = InvariantTorchFeatureLearner(policy()).snapshot_state()
    original, session = deepcopy(state), budget()
    before_calls = work["train_calls"]
    wake, replay = episodes()

    def check():
        if fault == "callback":
            raise RuntimeError("synthetic boundary refusal")
        session.replay_examples_completed = 1

    with pytest.raises(diagnostics.GradientDiagnosticFailed) as caught:
        diagnostics.diagnose_bp_updates(state, wake, replay, budget=session, check_boundary=check)
    assert caught.value.report["status"] == "incomplete"
    assert caught.value.report["work"]["native_train_calls_attempted"] == 0
    assert work["train_calls"] == before_calls
    assert_tree(torch, state, original)


@pytest.mark.parametrize("fault", ["parameter", "gradient_metadata"])
def test_should_pin_original_snapshot_before_first_callback(counted_native, fault):
    torch, work = counted_native
    state = InvariantTorchFeatureLearner(policy()).snapshot_state()
    before, session = work["train_calls"], budget()
    wake, replay = episodes()
    changed = False

    def check():
        nonlocal changed
        if changed:
            return
        changed = True
        parameter = state.native_state.head_state[NAMES[0]]
        if fault == "parameter":
            parameter[0, 0] += 1
        else:
            parameter.requires_grad_(True)
            parameter.grad = torch.ones_like(parameter)

    with pytest.raises(diagnostics.GradientDiagnosticFailed):
        diagnostics.diagnose_bp_updates(state, wake, replay, budget=session, check_boundary=check)
    assert changed and session.updates_completed == 0
    assert work["train_calls"] == before
