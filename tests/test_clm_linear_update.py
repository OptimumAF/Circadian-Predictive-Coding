"""Sequential updates preserve exact roles and expose interference and return."""

from dataclasses import asdict, replace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from src.adapters import clm_linear_update as warm
from src.adapters.clm_linear_fit_control import fit_clm_linear_control, predict_clm_linear_control
from src.core.clm_catalog import CLMCatalogTargets, FrozenCatalogTrainingBatch
from src.core.clm_catalog_probe import FrozenProbeCatalogBatch
from src.core.clm_head import FrozenChoiceBatch, FrozenEncoderIdentity
from src.core.clm_probe import FrozenProbeChoiceBatch


def batch(role="train", xs=(-1.0, 1.0), ys=(0, 1)):
    identity = FrozenEncoderIdentity("literal", "fixed", "fixed", "last", "none", 2)
    kind = FrozenCatalogTrainingBatch if role == "train" else FrozenProbeCatalogBatch
    return kind(
        identity, torch.tensor([[x, 0] for x in xs]), torch.zeros((2, 2)), ("a", "b")
    ), CLMCatalogTargets(("a", "b"), ys)


def read(control, data):
    kind = FrozenChoiceBatch if data.role == "train" else FrozenProbeChoiceBatch
    return (
        predict_clm_linear_control(
            control,
            kind(data.identity, data.states, data.actions, role=data.role),
            data.candidate_ids,
            {},
        )
        .argmax(axis=1)
        .tolist()
    )


@pytest.mark.parametrize("role", ["train", "probe_train"])
def test_should_learn_interfere_and_recover_without_changing_source(role):
    a, targets = batch(role)
    source = fit_clm_linear_control(a, targets, {})
    saved, rng = asdict(source), torch.get_rng_state().clone()
    assert read(source, a) == [0, 1]
    # Identical observations with changed task labels force visible interference.
    b, other_targets = batch(role, ys=(1, 0))
    report: dict[str, Any] = {}
    learned_b = warm.update_clm_linear_control(source, b, other_targets, report)
    assert read(learned_b, b) == [1, 0] and read(learned_b, a) != [0, 1]
    returned = warm.update_clm_linear_control(learned_b, a, targets, {})
    assert read(returned, a) == [0, 1]
    assert asdict(source) == saved and torch.equal(torch.get_rng_state(), rng)
    assert learned_b.role == returned.role == role
    work = report["linear_update"]
    assert work["complete"] and work["attempts"] == work["returns"] == 1
    assert work["rows_attempted"] == work["rows_returned"] == 2
    assert work["initialization"] == "minimum_change_from_prior_coefficients"
    assert work["optimizer_steps"] == work["native_head_forwards"] == work["encoder_forwards"] == 0
    assert work["source_coefficient_array_bytes"] == 48
    assert work["residual_target_array_bytes"] == 32
    with pytest.raises(ValueError, match="unused"):
        warm.update_clm_linear_control(source, a, targets, report)


def test_should_preserve_compatible_old_constraint_when_replayed_explicitly():
    a, y_a = batch(xs=(0.0, 0.0), ys=(0, 0))
    source = fit_clm_linear_control(a, y_a, {})
    current, y_b = batch(xs=(0.5, 0.5), ys=(1, 1))
    only = warm.update_clm_linear_control(source, current, y_b, {})
    assert read(only, a) == [1, 1] and read(only, current) == [1, 1]
    replay, replay_labels = batch(xs=(0.0, 0.5), ys=(0, 1))
    report: dict[str, Any] = {}
    combined = warm.update_clm_linear_control(source, replay, replay_labels, report)
    assert read(combined, replay) == [0, 1]
    assert report["linear_update"]["rows_attempted"] == 2  # Includes replay row.


@pytest.mark.parametrize(
    "fault",
    ["role", "reverse_role", "final", "readout", "catalog", "identity", "targets", "features"],
)
def test_should_refuse_cross_boundary_update_before_paid_solve(monkeypatch, fault):
    data, labels = batch()
    source = fit_clm_linear_control(data, labels, {})
    if fault == "role":
        data, labels = batch("probe_train")
    elif fault == "reverse_role":
        probe_data, probe_labels = batch("probe_train")
        source = fit_clm_linear_control(probe_data, probe_labels, {})
    elif fault == "final":
        object.__setattr__(data, "role", "final")
    elif fault == "readout":
        data = FrozenChoiceBatch(data.identity, data.states, data.actions)
    elif fault == "catalog":
        data = replace(data, candidate_ids=("b", "a"))
        labels = replace(labels, candidate_ids=("b", "a"))
    elif fault == "identity":
        data = replace(data, identity=replace(data.identity, revision="different"))
    elif fault == "targets":
        labels = replace(labels, candidate_ids=("b", "a"))
    else:
        data.states[0, 0] = float("nan")
    monkeypatch.setattr(
        warm, "update_wide_categorical_linear", lambda *a: pytest.fail("invalid solve")
    )
    report: dict[str, Any] = {}
    with pytest.raises(ValueError):
        warm.update_clm_linear_control(source, data, labels, report)
    assert report == {}


@pytest.mark.parametrize("fault", ["lost", "source", "source_value", "feature"])
def test_should_preserve_paid_failure_and_refuse_changed_inputs(monkeypatch, fault):
    data, labels = batch()
    source = fit_clm_linear_control(data, labels, {})
    real = warm.update_wide_categorical_linear

    def changed(*args):
        result = real(*args)
        if fault == "lost":
            raise RuntimeError("lost solve return")
        if fault == "source":
            object.__setattr__(source, "model", replace(source.model))
        elif fault == "source_value":
            object.__setattr__(source.model, "squared_error", 1.0)
        else:
            data.states.add_(0.1)
        return result

    monkeypatch.setattr(warm, "update_wide_categorical_linear", changed)
    report: dict[str, Any] = {}
    with pytest.raises((ValueError, RuntimeError)):
        warm.update_clm_linear_control(source, data, labels, report)
    work = report["linear_update"]
    assert not work["complete"] and work["unreturned_work_unknown"]
    assert work["attempts"] == 1 and work["rows_attempted"] == 2
    assert work["returns"] == (0 if fault == "lost" else 1)
    with pytest.raises(ValueError):
        warm.update_clm_linear_control(source, data, labels, report)
