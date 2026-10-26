"""Role-bound detached linear control, original features and paid failures."""

from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.adapters import clm_linear_fit_control as control
from src.core.clm_catalog import FrozenCatalogTrainingBatch, CLMCatalogTargets
from src.core.clm_catalog_probe import FrozenProbeCatalogBatch
from src.core.clm_head import FrozenEncoderIdentity, FrozenChoiceBatch
from src.core.clm_probe import FrozenProbeChoiceBatch


def packet(role="train", width=2, rows=2):
    identity = FrozenEncoderIdentity("literal", "fixed", "fixed", "last", "none", width)
    x = torch.zeros((rows, width), dtype=torch.float32)
    x[:, 0] = torch.tensor([-1.0 if i % 2 == 0 else 1.0 for i in range(rows)])
    actions = torch.zeros((2, width), dtype=torch.float32)
    kind = FrozenCatalogTrainingBatch if role == "train" else FrozenProbeCatalogBatch
    return kind(identity, x, actions, ("a", "b")), CLMCatalogTargets(
        ("a", "b"), tuple(i % 2 for i in range(rows))
    )


def choices(batch):
    kind = FrozenChoiceBatch if batch.role == "train" else FrozenProbeChoiceBatch
    return kind(batch.identity, batch.states, batch.actions, role=batch.role)


@pytest.mark.parametrize("role", ["train", "probe_train"])
def test_should_fit_and_read_existing_roles_with_detached_source_rng_and_costs(role):
    batch, labels = packet(role)
    before = (
        batch.states.clone(),
        batch.actions.clone(),
        torch.get_rng_state().clone(),
        deepcopy(np.random.get_state()),
    )
    report: dict[str, Any] = {}
    model = control.fit_clm_linear_control(batch, labels, report)
    original = asdict(model)
    p = control.predict_clm_linear_control(model, choices(batch), batch.candidate_ids, report)
    assert np.argmax(p, axis=1).tolist() == [0, 1]
    assert report["linear_fit"]["complete"] and report["linear_prediction"]["complete"]
    assert report["linear_fit"]["attempts"] == report["linear_fit"]["returns"] == 1
    assert report["linear_fit"]["design_array_bytes"] == 48
    assert report["linear_fit"]["one_hot_array_bytes"] == 32
    assert report["linear_prediction"]["output_probability_array_bytes"] == 32
    assert torch.equal(batch.states, before[0]) and torch.equal(batch.actions, before[1])
    assert torch.equal(torch.get_rng_state(), before[2])
    after = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(after, before[3], strict=True))
    single = replace(
        choices(batch),
        states=batch.states[:1].clone(),
        role="outer_selection" if role == "train" else "probe_test",
    )
    single_report: dict[str, Any] = {}
    single_p = control.predict_clm_linear_control(model, single, batch.candidate_ids, single_report)
    assert single_p.shape == (1, 2) and int(single_p.argmax()) == 0
    p[:] = 0
    batch.states.add_(0.1)  # Fit owns no live source feature reference.
    assert asdict(model) == original
    assert not hasattr(model, "train_batch") and not hasattr(model, "restore_state")
    fresh_batch, fresh_labels = packet(role)
    with pytest.raises(ValueError, match="unused"):
        control.fit_clm_linear_control(fresh_batch, fresh_labels, report)


def test_should_fit_full_4096_width_and16_rows_with_literal_scalar_solution():
    batch, labels = packet(width=4096, rows=16)
    # Repeated Hadamard columns: XX^T=I, every coordinate informative/bounded.
    x = np.array(
        [[(-1.0 if (i & j).bit_count() % 2 else 1.0) / 64 for j in range(4096)] for i in range(16)],
        dtype=np.float32,
    )
    batch = replace(batch, states=torch.tensor(x))
    report: dict[str, Any] = {}
    real = np.linalg.lstsq
    # This is the sole full4096 literal case; no model initialization or random draw.
    model = control.fit_clm_linear_control(batch, labels, report)
    assert model.model.feature_dim == 4096 and model.model.fit_rows == 16
    assert model.model.rank == 16 and model.model.squared_error < 1e-25
    coefficients = np.array(model.model.coefficients)
    # [X,1][X,1]^T=I+11^T, inverse=I-11^T/17; no solver oracle.
    one_hot = np.array([[1.0, 0.0] if i % 2 == 0 else [0.0, 1.0] for i in range(16)])
    expected_weights = x.astype(np.float64).T @ (one_hot - 8 / 17)
    # The original pointwise1e-14 check failed near zero on the wide SVD.
    # Use a dimension/analytic-condition normwise f64 roundoff scale instead;
    # preserve that failed engineering check, not a new scientific threshold.
    scaled_epsilon = 4097 * np.finfo(np.float64).eps
    gamma = scaled_epsilon / (1 - scaled_epsilon)
    error_bound = gamma * np.sqrt(17) * max(1.0, float(np.linalg.norm(expected_weights)))
    assert float(np.linalg.norm(coefficients[:-1] - expected_weights)) <= error_bound
    np.testing.assert_allclose(coefficients[-1], [8 / 17, 8 / 17], atol=1e-14)
    np.testing.assert_allclose(model.model.singular_values, [np.sqrt(17)] + [1.0] * 15, atol=1e-14)
    p = control.predict_clm_linear_control(model, choices(batch), batch.candidate_ids, report)
    assert np.argmax(p, axis=1).tolist() == list(labels.candidate_indices)
    assert np.linalg.lstsq is real


@pytest.mark.parametrize(
    "fault",
    [
        "final",
        "readout",
        "identity",
        "order",
        "labels",
        "dtype",
        "nan",
        "rows",
        "grad",
        "hidden_storage",
    ],
)
def test_should_refuse_invalid_fit_before_lapack(monkeypatch, fault):
    batch, labels = packet()
    if fault == "final":
        object.__setattr__(batch, "role", "final")
    elif fault == "readout":
        batch = choices(batch)
    elif fault == "identity":
        object.__setattr__(batch.identity, "embedding_dim", 3)
    elif fault == "order":
        labels = replace(labels, candidate_ids=("b", "a"))
    elif fault == "labels":
        labels = replace(labels, candidate_indices=(0, 1, 0))
    elif fault == "dtype":
        batch = replace(batch, states=batch.states.double())
    elif fault == "nan":
        batch.states[0, 0] = float("nan")
    elif fault == "rows":
        batch = replace(batch, states=torch.zeros((17, 2)))
    elif fault == "grad":
        batch.states.requires_grad_(True)
    elif fault == "hidden_storage":
        batch = replace(batch, states=torch.zeros((4, 2))[:2])
    monkeypatch.setattr(np.linalg, "lstsq", lambda *a, **k: pytest.fail("invalid input solved"))
    report: dict[str, Any] = {}
    with pytest.raises(ValueError):
        control.fit_clm_linear_control(batch, labels, report)
    assert report == {}


@pytest.mark.parametrize("fault", ["solver", "values", "owner", "targets"])
def test_should_preserve_attempted_fit_and_refuse_changed_inputs_without_publication(
    monkeypatch, fault
):
    batch, labels = packet()
    real = control.fit_wide_categorical_linear

    def fit(*args):
        if fault == "solver":
            raise np.linalg.LinAlgError("controlled solver failure")
        model = real(*args)
        if fault == "values":
            batch.states.add_(0.1)
        elif fault == "owner":
            object.__setattr__(batch, "states", batch.states.clone())
        elif fault == "targets":
            object.__setattr__(labels, "candidate_indices", (1, 0))
        return model

    monkeypatch.setattr(control, "fit_wide_categorical_linear", fit)
    report: dict[str, Any] = {}
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        control.fit_clm_linear_control(batch, labels, report)
    row = report["linear_fit"]
    assert row["attempts"] == 1 and not row["complete"] and row["unreturned_work_unknown"]
    assert row["returns"] == (0 if fault == "solver" else 1)


@pytest.mark.parametrize(
    "fault", ["final", "wrong_role", "order", "encoder", "model", "predict", "features"]
)
def test_should_refuse_invalid_or_corrupted_readout_and_retain_paid_work(monkeypatch, fault):
    batch, labels = packet("probe_train")
    report: dict[str, Any] = {}
    model = control.fit_clm_linear_control(batch, labels, report)
    values, ids = choices(batch), batch.candidate_ids
    if fault == "final":
        object.__setattr__(values, "role", "final")
    elif fault == "wrong_role":
        values = FrozenChoiceBatch(batch.identity, batch.states, batch.actions, "train")
    elif fault == "order":
        ids = ("b", "a")
    elif fault == "encoder":
        values = replace(values, identity=replace(values.identity, revision="changed"))
    else:
        real = control.predict_wide_categorical_linear

        def predict(*args):
            if fault == "predict":
                raise RuntimeError("controlled readout failure")
            p = real(*args)
            if fault == "features":
                values.states.add_(0.1)
            else:
                object.__setattr__(
                    model.model, "coefficients", tuple(reversed(model.model.coefficients))
                )
            return p

        monkeypatch.setattr(control, "predict_wide_categorical_linear", predict)
    with pytest.raises((ValueError, RuntimeError)):
        control.predict_clm_linear_control(model, values, ids, report)
    if fault in ("model", "predict", "features"):
        assert not report["linear_prediction"]["complete"]
        assert report["linear_prediction"]["attempts"] == 1
    else:
        assert "linear_prediction" not in report


def test_should_refuse_different_fitted_row_count_after_paid_return(monkeypatch):
    batch, labels = packet()
    real = control.fit_wide_categorical_linear

    def fit(*args):
        model = real(*args)
        return replace(model, fit_rows=1, rank=1, singular_values=(1.0,))

    monkeypatch.setattr(control, "fit_wide_categorical_linear", fit)
    report: dict[str, Any] = {}
    with pytest.raises(ValueError, match="fitted row count"):
        control.fit_clm_linear_control(batch, labels, report)
    assert report["linear_fit"]["returns"] == 1
    assert not report["linear_fit"]["complete"]
    assert report["linear_fit"]["unreturned_work_unknown"]
