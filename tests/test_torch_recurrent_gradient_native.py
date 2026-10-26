"""Eight prospectively budgeted synthetic gradient diagnostic unit cases.

Native CE training is used only on disposable oracle forks and two tiny warm
steps; no MiniGrid source, teacher, final layout or comparative learner runs.
"""

from copy import deepcopy
import pickle
import random
from typing import Any

import numpy as np
import pytest

from src.adapters import torch_recurrent_gradient_diagnostics as diagnostics
from src.adapters import torch_recurrent_learner as native
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from src.shared.torch_runtime import require_torch
from test_torch_feature_learner import assert_tree

NAMES = ("weight_input", "weight_recurrent", "bias_hidden", "weight_output", "bias_output")


@pytest.fixture
def cost(monkeypatch, record_property):
    torch = require_torch()
    work: dict[str, Any] = dict(
        source_graphs=0,
        oracle_forks=0,
        optimizer_steps=0,
        committed_updates=0,
        committed_rows=0,
        diagnostic_calls_attempted=0,
        diagnostic_calls_completed=0,
        snapshot_validations_attempted=0,
        snapshot_validations_completed=0,
        forward_calls_attempted=0,
        forward_calls_completed=0,
        forward_rows_attempted=0,
        forward_rows_completed=0,
        backward_calls_attempted=0,
        backward_calls_completed=0,
        backward_rows_attempted=0,
        backward_rows_completed=0,
        diagnosing=False,
        last_rows=0,
        fail_second_forward=False,
    )
    step, forward, grad, snapshot = (
        torch.optim.SGD.step,
        native._forward,
        torch.autograd.grad,
        native._snapshot,
    )

    def observed_step(owner, *args, **kwargs):
        work["optimizer_steps"] += 1
        assert not work["diagnosing"], "diagnostic attempted an optimizer update"
        return step(owner, *args, **kwargs)

    def observed_snapshot(value):
        if work["diagnosing"]:
            work["snapshot_validations_attempted"] += 1
        result = snapshot(value)
        if work["diagnosing"]:
            work["snapshot_validations_completed"] += 1
        return result

    def observed_forward(parameters, values, hidden, **kwargs):
        active = work["diagnosing"]
        rows = len(values)
        if active:
            work["last_rows"] = rows
            work["forward_calls_attempted"] += 1
            work["forward_rows_attempted"] += rows
            if work["fail_second_forward"] and work["forward_calls_attempted"] == 2:
                with torch.no_grad():
                    parameters["weight_input"].add_(0.125)
                raise RuntimeError("synthetic second forward refused after temporary mutation")
        result = forward(parameters, values, hidden, **kwargs)
        if active:
            work["forward_calls_completed"] += 1
            work["forward_rows_completed"] += rows
        return result

    def observed_grad(*args, **kwargs):
        active = work["diagnosing"]
        if active:
            work["backward_calls_attempted"] += 1
            work["backward_rows_attempted"] += work["last_rows"]
        result = grad(*args, **kwargs)
        if active:
            work["backward_calls_completed"] += 1
            work["backward_rows_completed"] += work["last_rows"]
        return result

    monkeypatch.setattr(torch.optim.SGD, "step", observed_step)
    monkeypatch.setattr(native, "_forward", observed_forward)
    monkeypatch.setattr(native, "_snapshot", observed_snapshot)
    monkeypatch.setattr(torch.autograd, "grad", observed_grad)
    yield work
    for key, value in work.items():
        if type(value) is int and key != "last_rows":
            record_property(key, value)


def pair(*, opposed=False):
    def episode(identity, label):
        observations = ((0.2, -0.1), (-0.3, 0.4), (0.0, 0.0))
        return DemonstrationEpisode(
            identity,
            "synthetic-unit",
            "train",
            tuple(
                DemonstrationTransition(
                    i, observations[i], label, 0.0, observations[i + 1], i == 1, 2
                )
                for i in range(2)
            ),
            2,
        )

    return episode("synthetic-blue", 0), episode("synthetic-red", int(opposed))


def source(work, *, warm=False, zero=False, saturated=False):
    torch = require_torch()
    owner = native.TorchRecurrentLearner(native.TorchRecurrentPolicy(2, 2, 2, 17))
    work["source_graphs"] += 1
    if warm:
        owner.train_episode(pair(opposed=True)[0])
        work["committed_updates"] += 1
        work["committed_rows"] += 2
    if zero:
        state = owner.snapshot_state()
        with torch.no_grad():
            for parameter in state.parameters.values():
                parameter.zero_()
            if saturated:
                state.parameters["bias_output"][0, 0] = 1000.0
                state.parameters["bias_output"][0, 1] = -1000.0
        owner.restore_state(state)
    return owner


def rng_state(torch):
    return torch.get_rng_state().clone(), deepcopy(np.random.get_state()), random.getstate()


def assert_rng(torch, before):
    assert torch.equal(torch.get_rng_state(), before[0])
    now: Any = np.random.get_state()
    assert now[0] == before[1][0] and now[2:] == before[1][2:]
    np.testing.assert_array_equal(now[1], before[1][1])
    assert random.getstate() == before[2]


def invoke(work, snapshot, episodes):
    work["diagnostic_calls_attempted"] += 1
    work["diagnosing"] = True
    try:
        result = diagnostics.diagnose_episode_gradients(snapshot, episodes)
        work["diagnostic_calls_completed"] += 1
        return result
    finally:
        work["diagnosing"] = False


def assert_scalar_report(value):
    if isinstance(value, dict):
        for item in value.values():
            assert_scalar_report(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_scalar_report(item)
    else:
        assert type(value) in (str, int, float, bool, type(None))
        if type(value) is float:
            assert np.isfinite(value)


@pytest.mark.parametrize("kind", ["aligned", "opposed", "zero"])
def test_should_match_hand_computable_native_geometry_without_any_update(cost, kind):
    torch = require_torch()
    owner = source(cost, zero=True, saturated=kind == "zero")
    state, episodes = owner.snapshot_state(), pair(opposed=kind == "opposed")
    state_before, owner_before = deepcopy(state), owner.snapshot_state()
    original, rng = pickle.dumps(episodes), rng_state(torch)
    report = invoke(cost, state, episodes)
    assert report["status"] == "complete" and report["objective"] == native.TORCH_RECURRENT_CE_ID
    joint = report["joint_geometry"]
    if kind == "zero":
        assert joint["first_norm"] == joint["second_norm"] == joint["dot"] == 0.0
        assert joint["cosine"] is None and joint["cosine_reason"] == "zero_norm"
        assert all(loss["mean_ce"] == 0 for loss in report["losses"])
    else:
        assert joint["first_norm"] == pytest.approx(np.sqrt(0.5))
        assert joint["second_norm"] == pytest.approx(np.sqrt(0.5))
        assert joint["dot"] == pytest.approx(-0.5 if kind == "opposed" else 0.5)
        assert joint["cosine"] == pytest.approx(-1.0 if kind == "opposed" else 1.0)
        assert joint["cosine_reason"] is None
        assert all(loss["mean_ce"] == pytest.approx(np.log(2)) for loss in report["losses"])
    assert set(report["per_parameter_geometry"]) == set(NAMES)
    assert report["per_parameter_geometry"]["weight_input"]["cosine"] is None
    assert report["work"]["optimizer_steps"] == report["work"]["published_updates"] == 0
    assert (
        report["work"]["forward_rows_completed"] == report["work"]["backward_rows_completed"] == 4
    )
    assert report["work"]["validation_temporary_optimizers"] == 1
    assert_scalar_report(report)
    assert_tree(torch, state, state_before)
    assert_tree(torch, owner.snapshot_state(), owner_before)
    assert pickle.dumps(episodes) == original
    assert_rng(torch, rng)
    assert cost["optimizer_steps"] == 0


def independent_geometry(torch, first, second):
    a = torch.cat([value.flatten().to(torch.float64) for value in first])
    b = torch.cat([value.flatten().to(torch.float64) for value in second])
    na, nb, dot = (
        float(torch.linalg.vector_norm(a)),
        float(torch.linalg.vector_norm(b)),
        float(torch.dot(a, b)),
    )
    return dict(
        first_norm=na,
        second_norm=nb,
        dot=dot,
        cosine=None if na == 0 or nb == 0 else dot / (na * nb),
    )


def test_should_match_original_mean_ce_backward_and_restore_no_grad_context(cost):
    torch = require_torch()
    owner = source(cost, warm=True)
    state, episodes = owner.snapshot_state(), pair(opposed=True)
    before, source_before = deepcopy(state), owner.snapshot_state()
    original, rng = pickle.dumps(episodes), rng_state(torch)
    with torch.no_grad():
        report = invoke(cost, state, episodes)
        assert not torch.is_grad_enabled()
    assert torch.is_grad_enabled()
    assert_rng(torch, rng)
    oracles = []
    for episode, loss in zip(episodes, report["losses"], strict=True):
        fork = owner.fork()
        cost["oracle_forks"] += 1
        diagnostic = fork.train_episode(episode)
        cost["committed_updates"] += 1
        cost["committed_rows"] += 2
        assert loss["mean_ce"] == diagnostic.value
        oracles.append(fork.snapshot_state().gradients)
    for names, actual in [
        (NAMES, report["joint_geometry"]),
        *(([name], report["per_parameter_geometry"][name]) for name in NAMES),
    ]:
        expected = independent_geometry(
            torch, [oracles[0][n] for n in names], [oracles[1][n] for n in names]
        )
        for key, value in expected.items():
            if value is None:
                assert actual[key] is None
            else:
                assert actual[key] == pytest.approx(value, rel=1e-10, abs=1e-12)
    assert_tree(torch, owner.snapshot_state(), source_before)
    assert_tree(torch, state, before)
    assert pickle.dumps(episodes) == original
    assert cost["optimizer_steps"] == 3 and cost["committed_rows"] == 6
    assert_scalar_report(report)


def test_should_preserve_every_input_and_paid_prefix_after_second_forward_failure(cost):
    torch = require_torch()
    owner = source(cost, warm=True)
    state, episodes = owner.snapshot_state(), pair(opposed=True)
    before, source_before = deepcopy(state), owner.snapshot_state()
    original, rng = pickle.dumps(episodes), rng_state(torch)
    cost["fail_second_forward"] = True
    with torch.no_grad():
        with pytest.raises(
            diagnostics.GradientDiagnosticFailed, match="synthetic second forward refused"
        ) as caught:
            invoke(cost, state, episodes)
        assert not torch.is_grad_enabled()
    report = caught.value.report
    assert report["status"] == "incomplete" and report["joint_geometry"] is None
    assert report["per_parameter_geometry"] is None and len(report["losses"]) == 1
    assert report["work"]["forward_calls_attempted"] == 2
    assert (
        report["work"]["forward_calls_completed"] == report["work"]["backward_calls_completed"] == 1
    )
    assert report["work"]["forward_rows_attempted"] == 4
    assert report["work"]["failure_work_may_be_unobserved"]
    assert_tree(torch, owner.snapshot_state(), source_before)
    assert_tree(torch, state, before)
    assert pickle.dumps(episodes) == original
    assert_rng(torch, rng)
    assert torch.is_grad_enabled()
    assert cost["optimizer_steps"] == 1 and cost["committed_rows"] == 2
    assert_scalar_report(report)


def malformed_snapshot():
    class ForeignPayload:
        def __deepcopy__(self, memo):
            pytest.fail("foreign nested payload must be refused before deepcopy")

    return native.TorchRecurrentSnapshot(
        1,
        native.TorchRecurrentPolicy(2, 2, 2, 17),
        {name: ForeignPayload() for name in NAMES},
        {},
        {name: None for name in NAMES},
        0,
        0,
    )


def test_should_refuse_foreign_snapshot_payload_before_copy_or_model_work(cost):
    torch = require_torch()
    state, episodes = malformed_snapshot(), pair()
    identities, rng = tuple(state.parameters.values()), rng_state(torch)
    with pytest.raises(diagnostics.GradientDiagnosticFailed) as caught:
        invoke(cost, state, episodes)
    report = caught.value.report
    assert "finite CPU float32 tensor" in report["reason"]
    assert report["work"]["snapshot_validations_attempted"] == 1
    assert report["work"]["snapshot_validations_completed"] == 0
    assert report["work"]["forward_calls_attempted"] == 0
    assert tuple(state.parameters.values()) == identities
    assert_rng(torch, rng)
    assert cost["optimizer_steps"] == cost["source_graphs"] == 0


def test_should_refuse_inference_mode_before_snapshot_validation_or_model_work(cost):
    torch = require_torch()
    with torch.inference_mode():
        with pytest.raises(ValueError, match="inference_mode"):
            invoke(cost, malformed_snapshot(), pair())
        assert torch.is_inference_mode_enabled()
    assert cost["snapshot_validations_attempted"] == cost["forward_calls_attempted"] == 0
    assert cost["optimizer_steps"] == cost["source_graphs"] == 0


def test_should_refuse_late_caller_mutation_without_restoring_unknown_changes(cost, monkeypatch):
    torch = require_torch()
    owner = source(cost)
    state, source_before = owner.snapshot_state(), owner.snapshot_state()
    expected = deepcopy(state)
    with torch.no_grad():
        expected.parameters["bias_output"].add_(0.25)
    forward = native._forward

    def mutate_after_second_forward(*args, **kwargs):
        result = forward(*args, **kwargs)
        if cost["diagnosing"] and cost["forward_calls_completed"] == 2:
            with torch.no_grad():
                state.parameters["bias_output"].add_(0.25)
        return result

    monkeypatch.setattr(native, "_forward", mutate_after_second_forward)
    with pytest.raises(
        diagnostics.GradientDiagnosticFailed, match="snapshot, RNG or mode changed"
    ) as caught:
        invoke(cost, state, pair(opposed=True))
    report = caught.value.report
    assert report["joint_geometry"] is report["per_parameter_geometry"] is None
    assert len(report["losses"]) == 2
    assert (
        report["work"]["forward_rows_completed"] == report["work"]["backward_rows_completed"] == 4
    )
    assert report["work"]["failure_work_may_be_unobserved"]
    assert_tree(torch, state, expected)
    assert_tree(torch, owner.snapshot_state(), source_before)
    assert cost["optimizer_steps"] == 0
