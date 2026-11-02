"""Tiny real staged updates; no ordinary source12 or full adaptation case."""

from copy import deepcopy
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from src.adapters import torch_categorical_ce_readout as ce
from src.adapters import torch_ce_probe_update as probe
from src.adapters import torch_feature_learner as native
from src.adapters.torch_array_weighted import _ownership, _rng
from src.adapters.torch_decision import _same_values
from src.app.coupled_attribution import AttributionBatch
from src.app.decision_reference_evaluation import DecisionCase
from src.core.decision_features import DecisionFeatureSpec
from src.core.typed_decision import DecisionCandidate, DecisionRequest
from src.shared.torch_runtime import require_torch


def forbidden(*a, **kw):
    raise AssertionError("unbudgeted model work")


def batches():
    current = AttributionBatch(
        "probe_train", "new", ("new:0",), np.array([[1.0, 1.0]]), np.array([[2.0]]), num_classes=3
    )
    retained = AttributionBatch(
        "train", "old", ("old:0", "old:1"), np.eye(2), np.array([[0.0], [1.0]]), num_classes=3
    )
    return current, (retained,)


@pytest.fixture(scope="module")
def qualified():
    torch = require_torch()
    rng = _rng(torch)
    source = native.TorchFeatureLearner(native.TorchFeaturePolicy("bp", 2, 3, 17, num_classes=3))
    with torch.no_grad():
        source._head.weight_feature_hidden.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        source._head.bias_hidden.zero_()
    catalog = tuple(DecisionCandidate(name, name) for name in ("a", "b", "c"))
    spec = DecisionFeatureSpec(("alpha", "beta"), catalog)
    cases = tuple(
        DecisionCase(DecisionRequest(str(i), query, catalog), target, "fixture-train", "train")
        for i, (query, target) in enumerate((("alpha", "a"), ("beta", "b"), ("alpha beta", "c")))
    )
    batch = ce.build_decision_training_batch(cases, spec, "fixture", allow_subsets=True)
    source.train_batch(batch.features, batch.targets)
    before, ownership = source.snapshot_state(), _ownership(source)
    current = AttributionBatch(
        "probe_train",
        "new",
        ("new:0",),
        batch.features[2:].copy(),
        batch.targets[2:].copy(),
        num_classes=3,
    )
    retained = AttributionBatch(
        "train",
        "old",
        ("old:0", "old:1"),
        batch.features[:2].copy(),
        batch.targets[:2].copy(),
        num_classes=3,
    )
    update = probe.fit_ce_probe_update(source, current, (retained,))
    reference = ce.fit_categorical_ce_readout(source, cases, spec)
    assert update.coefficients == reference.coefficients
    assert (
        update.closure_losses == reference.closure_losses
        and update.final_loss == reference.final_loss
    )
    assert {k: update.work[k] for k in reference.work} == reference.work
    assert update.final_loss < 0.01 and update.final_loss < update.closure_losses[0]
    assert _same_values(source.snapshot_state(), before) and _ownership(source) == ownership
    assert _same_values(_rng(torch), rng)
    return source, update, batch


def test_actual_private_application_learns_and_preserves_original_native_work(qualified):
    source, update, batch = qualified
    torch = require_torch()
    rng = _rng(torch)
    pin = source.snapshot_state()
    private = native.TorchFeatureLearner(source.policy)
    private.restore_state(probe.copy_ce_probe_update(update).state)
    assert private.predict_logits(batch.features).argmax(axis=1).tolist() == [0, 1, 2]
    for name in native._PARAMETERS[:2]:
        assert _same_values(update.before.head_state[name], update.state.head_state[name])
    assert _same_values(update.before.gradients, update.state.gradients)
    assert _same_values(update.before.optimizer_state, update.state.optimizer_state)
    assert update.before.completed_updates == update.state.completed_updates == 1
    assert update.before.training_examples == update.state.training_examples == 3
    assert update.work["native_sgd_updates"] == update.work["current_train_gradient_rows"] == 0
    calls = update.work["backward_attempts"]
    assert update.work["probe_gradient_rows"] == calls
    assert update.work["retained_train_gradient_rows"] == 2 * calls
    assert _same_values(pin, source.snapshot_state()) and _same_values(_rng(torch), rng)


def test_return_phase_accepts_same_fact_train_duplicates_and_charges_them(qualified):
    source, update, _ = qualified
    current, retained = batches()
    current = replace(
        retained[0],
        row_ids=("old:0",),
        features=retained[0].features[:1].copy(),
        targets=retained[0].targets[:1].copy(),
    )
    private = native.TorchFeatureLearner(source.policy)
    private.restore_state(update.state)
    before = private.snapshot_state()
    returned = probe.fit_ce_probe_update(private, current, retained, phase="return")
    assert returned.work["probe_gradient_rows"] == returned.work["native_sgd_updates"] == 0
    assert returned.work["current_train_gradient_rows"] == returned.work["backward_attempts"]
    assert returned.work["retained_train_gradient_rows"] == 2 * returned.work["backward_attempts"]
    assert _same_values(before, private.snapshot_state())
    assert returned.state.completed_updates == before.completed_updates


@pytest.mark.parametrize("role", ["train", "probe_test", "final", "controller_train"])
def test_fresh_role_refuses_before_source_or_optimizer(role, monkeypatch):
    current, retained = batches()
    monkeypatch.setattr(probe, "_unchanged_owner", forbidden)
    with pytest.raises(ValueError, match="roles"):
        probe.fit_ce_probe_update(None, replace(current, role=role), retained)


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final", "controller_train"])
def test_retained_roles_never_become_a_probe_label_replay_bank(role, monkeypatch):
    current, retained = batches()
    monkeypatch.setattr(probe, "_unchanged_owner", forbidden)
    with pytest.raises(ValueError, match="roles"):
        probe.fit_ce_probe_update(None, current, (replace(retained[0], role=role),))


@pytest.mark.parametrize("phase", [None, False, "other"])
def test_phase_is_exact_and_refuses_before_owner(phase, monkeypatch):
    monkeypatch.setattr(probe, "_unchanged_owner", forbidden)
    with pytest.raises(ValueError, match="phase"):
        probe.fit_ce_probe_update(None, *batches(), phase=phase)


def test_all_roles_precede_any_matrix_access_and_foreign_role_comparison(monkeypatch):
    class ForeignRole(str):
        def __eq__(self, other):
            raise AssertionError("foreign comparison")

    current, retained = batches()
    monkeypatch.setattr(probe, "_unchanged_owner", forbidden)
    for role in ("final", ForeignRole("train")):
        foreign = replace(retained[0], role=role, features=cast(Any, None), targets=cast(Any, None))
        with pytest.raises(ValueError, match="roles"):
            probe.fit_ce_probe_update(None, replace(current, features=cast(Any, None)), (foreign,))


@pytest.mark.parametrize(
    "fault", ["collision", "soft_target", "classes", "rows", "return_facts", "return_probe"]
)
def test_invalid_batch_contracts_refuse_before_owner(fault, monkeypatch):
    current, retained = batches()
    phase = "fresh"
    if fault == "collision":
        current = replace(current, row_ids=("old:0",))
    elif fault == "soft_target":
        current = replace(current, targets=np.array([[0.5]]))
    elif fault == "classes":
        current = replace(current, num_classes=4)
    elif fault == "rows":
        current = replace(
            current,
            row_ids=tuple(f"n{i}" for i in range(128)),
            features=np.ones((128, 2)),
            targets=np.zeros((128, 1)),
        )
    elif fault == "return_facts":
        phase = "return"
        current = replace(current, role="train", context="old", row_ids=("old:0",))
    else:
        phase = "return"
    monkeypatch.setattr(probe, "_unchanged_owner", forbidden)
    with pytest.raises(ValueError):
        probe.fit_ce_probe_update(None, current, retained, phase=phase)


@pytest.mark.parametrize("fault", ["phase", "hidden", "sgd_credit", "cost", "coefficients"])
def test_proposal_copy_refuses_changed_role_state_or_work(qualified, fault):
    update = deepcopy(qualified[1])
    if fault == "phase":
        update = replace(update, phase="return")
    elif fault == "hidden":
        update.state.head_state["bias_hidden"].add_(1)
    elif fault == "sgd_credit":
        update = replace(update, state=replace(update.state, completed_updates=2))
    elif fault == "cost":
        update.work["retained_train_gradient_rows"] += 1
    else:
        update = replace(
            update, coefficients=tuple(tuple(0.0 for _ in row) for row in update.coefficients)
        )
    with pytest.raises(ValueError):
        probe.copy_ce_probe_update(update)


def test_probe_result_is_not_train_fit_authority(qualified):
    with pytest.raises(ValueError, match="exact categorical CE fit"):
        ce._fit_copy(cast(Any, qualified[1]))


def test_lost_step_returns_no_proposal_and_keeps_role_charges(qualified, monkeypatch):
    source = qualified[0]
    before = source.snapshot_state()

    def lost(optimizer, closure):
        closure()
        raise RuntimeError("injected lost return")

    monkeypatch.setattr(require_torch().optim.LBFGS, "step", lost)
    with pytest.raises(probe.CEProbeUpdateFailed, match="lost return") as caught:
        probe.fit_ce_probe_update(source, *batches())
    work = caught.value.work
    assert work["step_attempts"] == work["backward_returns"] == 1 and work["step_returns"] == 0
    assert work["probe_gradient_rows"] == 1 and work["retained_train_gradient_rows"] == 2
    assert work["native_sgd_updates"] == 0 and _same_values(before, source.snapshot_state())


@pytest.mark.parametrize("fault", ["current_values", "anchor_owner", "current_role"])
def test_foreign_input_changes_are_refused_before_backward_and_preserved(
    qualified, monkeypatch, fault
):
    source = qualified[0]
    current, retained = batches()
    old_owner = retained[0].features
    before = source.snapshot_state()
    original = require_torch().nn.functional.cross_entropy

    def changed(*args, **kwargs):
        loss = original(*args, **kwargs)
        if fault == "current_values":
            current.targets[0, 0] = 0
        elif fault == "anchor_owner":
            object.__setattr__(retained[0], "features", old_owner.copy())
        else:
            object.__setattr__(current, "role", "final")
        return loss

    monkeypatch.setattr(require_torch().nn.functional, "cross_entropy", changed)
    with pytest.raises(probe.CEProbeUpdateFailed) as caught:
        probe.fit_ce_probe_update(source, current, retained)
    assert caught.value.work["forward_returns"] == 1 and caught.value.work["backward_attempts"] == 0
    assert _same_values(before, source.snapshot_state())
    assert current.targets[0, 0] == 0 if fault == "current_values" else True
    assert retained[0].features is not old_owner if fault == "anchor_owner" else True
    assert current.role == "final" if fault == "current_role" else True


def test_post_fit_foreign_source_change_is_not_overwritten(qualified, monkeypatch):
    private = native.TorchFeatureLearner(qualified[0].policy)
    private.restore_state(qualified[0].snapshot_state())
    before = private.snapshot_state()
    original = probe._derived

    def changed(*args):
        result = original(*args)
        with require_torch().no_grad():
            private._head.bias_output.add_(1)
        return result

    monkeypatch.setattr(probe, "_derived", changed)
    with pytest.raises(probe.CEProbeUpdateFailed, match="changed") as caught:
        probe.fit_ce_probe_update(private, *batches())
    assert caught.value.work["step_returns"] == 1 and caught.value.work["backward_rows"] > 0
    assert not _same_values(before, private.snapshot_state())
