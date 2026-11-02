"""Tiny actual application through the existing runner, plus scripted fit faults."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from src.adapters import torch_ce_probe_update as ce
from src.adapters import torch_feature_learner as native
from src.adapters.torch_array_weighted import _ownership, _rng
from src.adapters.torch_decision import _same_values
from src.app.coupled_attribution import AttributionBatch
from src.app.fresh_task_probe import run_fresh_task_probe
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.probe_update import ProbeUpdateFailed
from src.shared.torch_runtime import require_torch


@pytest.fixture(scope="module")
def actual():
    torch = require_torch()
    source = native.TorchFeatureLearner(native.TorchFeaturePolicy("bp", 2, 3, 17, num_classes=3))
    with torch.no_grad():
        source._head.weight_feature_hidden.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        source._head.bias_hidden.zero_()
    source.train_batch(
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), np.array([[0.0], [1.0], [2.0]])
    )
    before, ownership, rng = source.snapshot_state(), _ownership(source), _rng(torch)
    current = AttributionBatch(
        "probe_train", "new", ("new:0",), np.array([[1.0, 1.0]]), np.array([[2.0]]), num_classes=3
    )
    anchors = (
        AttributionBatch(
            "train", "old", ("old:0", "old:1"), np.eye(2), np.array([[0.0], [1.0]]), num_classes=3
        ),
    )
    evaluation = (
        replace(
            current,
            role="probe_test",
            row_ids=("held-out:0",),
            features=current.features.copy(),
            targets=current.targets.copy(),
        ),
    )
    children, proposals = [], []
    fork, fit = source.fork, ce.fit_ce_probe_update
    with pytest.MonkeyPatch.context() as patch:

        def tracked_fork():
            child = fork()
            children.append(child)
            return child

        def tracked_fit(*args, **kwargs):
            result = fit(*args, **kwargs)
            proposals.append(deepcopy(result))
            return result

        patch.setattr(source, "fork", tracked_fork)
        patch.setattr(ce, "fit_ce_probe_update", tracked_fit)
        budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
        report = run_fresh_task_probe(
            source,
            before,
            (current,),
            evaluation,
            budget=budget,
            checkpoint_updates=(0, 1),
            training_readouts=anchors,
            update_callback=ce.apply_ce_probe_update,
            timer=lambda: 0.0,
        )
        assert report["status"] == "complete", report["reason"]
        child = children[0]
        returned = ce.apply_ce_probe_update(
            child,
            replace(
                anchors[0],
                row_ids=("return:0", "return:1"),
                features=anchors[0].features.copy(),
                targets=anchors[0].targets.copy(),
            ),
            anchors,
            phase="return",
            check_boundary=lambda: None,
        )
    assert _same_values(source.snapshot_state(), before) and _ownership(source) == ownership
    assert _same_values(_rng(torch), rng)
    return source, before, child, current, anchors, proposals[0], report, returned


def test_actual_runner_and_return_application_preserve_donor_and_native_sgd(actual):
    source, before, child, current, anchors, proposal, report, returned = actual
    assert report["completed_updates"] == 1 and report["diagnostics"] == []
    event = report["auxiliary_updates"]["events"][0]
    assert event["applied"] and event["work"]["application_verified"] == 1
    assert event["work"]["probe_gradient_rows"] == event["work"]["backward_attempts"]
    assert returned.applied and returned.work["current_train_gradient_rows"] > 0
    assert returned.work["probe_gradient_rows"] == 0
    state = child.snapshot_state()
    for key in native._PARAMETERS[:2]:
        assert _same_values(state.head_state[key], before.head_state[key])
    assert _same_values(state.gradients, before.gradients)
    assert _same_values(state.optimizer_state, before.optimizer_state)
    assert state.completed_updates == before.completed_updates == 1
    assert state.training_examples == before.training_examples == 3
    features = np.concatenate((anchors[0].features, current.features))
    assert child.predict_logits(features).argmax(axis=1).tolist() == [0, 1, 2]
    assert report["checkpoints"][-1]["scores"]["new"]["accuracy"] == 1.0


@pytest.mark.parametrize(
    "fault", ["before", "owner", "replace", "restore", "late", "late_owner", "fit_failure"]
)
def test_application_refuses_foreign_state_and_keeps_actual_partial_work(
    actual, monkeypatch, fault
):
    source, before, child, current, anchors, proposal, report, returned = actual
    owner = native.TorchFeatureLearner(before.policy)
    owner.restore_state(before)
    result = deepcopy(proposal)
    if fault == "before":
        # Still internally valid, but from a different owner before-state.
        altered = replace(result.before, completed_updates=result.before.completed_updates + 1)
        result = replace(
            result,
            before=altered,
            state=replace(result.state, completed_updates=altered.completed_updates),
        )

    def fit(*args, **kwargs):
        if fault == "fit_failure":
            raise ce.CEProbeUpdateFailed(
                "paid fit failure", dict(native_sgd_updates=0, backward_attempts=3)
            )
        return result

    monkeypatch.setattr(ce, "fit_ce_probe_update", fit)
    calls = 0

    def boundary():
        nonlocal calls
        calls += 1
        if fault == "owner" or fault == "late_owner" and calls == 2:
            h, o, n, r = owner._payload
            owner._payload = (h, o, n + 1, r + 3)
        elif fault == "replace":
            owner.restore_state(owner.snapshot_state())
        elif fault == "late" and calls == 2:
            raise ValueError("late boundary failure")

    if fault == "restore":

        def broken_restore(state):
            raise RuntimeError("lost restore return")

        monkeypatch.setattr(owner, "restore_state", broken_restore)
    with pytest.raises(ProbeUpdateFailed) as caught:
        ce.apply_ce_probe_update(owner, current, anchors, phase="fresh", check_boundary=boundary)
    receipt = caught.value.receipt
    assert receipt.applied == (fault in ("late", "late_owner"))
    assert receipt.work["native_sgd_updates"] == 0
    if fault == "fit_failure":
        assert receipt.work["backward_attempts"] == 3
    else:
        assert receipt.work["backward_attempts"] == proposal.work["backward_attempts"]
    if fault == "restore":
        assert receipt.work["application_attempts"] == 1
        assert receipt.work["application_returns"] == 0
    if fault in ("owner", "late_owner"):
        assert owner.completed_updates == before.completed_updates + 1
    assert _same_values(source.snapshot_state(), before)
