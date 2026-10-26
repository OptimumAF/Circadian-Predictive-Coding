"""Detached categorical fit of existing role-bound frozen CLM features.

No native learner/head/encoder, deployment, promotion, release, replay or IO.
The caller owns its prospective experiment allowance; this control neither
creates nor renews one. Reports count logical arrays, not all LAPACK workspace.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from time import monotonic

import numpy as np
import torch

from src.core.clm_catalog import CLMCatalogTargets, FrozenCatalogTrainingBatch, _candidate_ids
from src.core.clm_catalog_probe import FrozenProbeCatalogBatch
from src.core.clm_head import FrozenChoiceBatch, FrozenEncoderIdentity, _exact
from src.core.clm_probe import FrozenProbeChoiceBatch
from src.core.categorical_probability_scores import validate_class_probabilities
from src.core.wide_categorical_linear_fit import (
    WideCategoricalLinearFit,
    fit_wide_categorical_linear,
    predict_wide_categorical_linear,
    predict_wide_categorical_active,
)

DEFINITION = "clm_frozen_feature_linear_fit_control_v1"


@dataclass(frozen=True)
class CLMLinearFitControl:
    identity: FrozenEncoderIdentity
    candidate_ids: tuple[str, ...]
    role: str
    model: WideCategoricalLinearFit

    def __post_init__(self):
        _exact(self.identity, FrozenEncoderIdentity)
        _candidate_ids(self.candidate_ids)
        if type(self.role) is not str or self.role not in ("train", "probe_train"):
            raise ValueError("linear control requires its original fitting role")
        if type(self.model) is not WideCategoricalLinearFit:
            raise ValueError("linear control requires an exact wide fit")
        self.model.__post_init__()
        if self.model.feature_dim != self.identity.embedding_dim or self.model.num_classes != len(
            self.candidate_ids
        ):
            raise ValueError("linear control encoder/catalog geometry differs")


def _tensor(value, rows, width):
    if (
        type(value) is not torch.Tensor
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or value.layout != torch.strided
        or value.requires_grad
        or tuple(value.shape) != (rows, width)
        or not 1 <= rows <= 16
        or not 1 <= width <= 4096
        or not value.is_contiguous()
        or value.storage_offset() != 0
        or value.untyped_storage().nbytes() != value.numel() * value.element_size()
        or not torch.isfinite(value).all().item()
        or torch.any(value.abs() > 1).item()
    ):
        raise ValueError(
            "linear control needs bounded owned-layout detached normalized CPUfloat32 features"
        )


def _features(batch, candidate_ids, *, min_rows=2):
    width = batch.identity.embedding_dim
    _candidate_ids(candidate_ids)
    if type(batch.states) is not torch.Tensor or batch.states.ndim != 2:
        raise ValueError("linear control requires a native two-dimensional state tensor")
    if len(candidate_ids) > 16 or not min_rows <= len(batch.states) <= 16:
        raise ValueError("linear control query rows or candidate count exceed bounded geometry")
    _tensor(batch.states, len(batch.states), width)
    _tensor(batch.actions, len(candidate_ids), width)


def _capture(batch, targets=None):
    arrays = tuple(t.detach().numpy().copy() for t in (batch.states, batch.actions))
    traits = tuple(
        (t._version, t.untyped_storage().data_ptr(), t.stride())
        for t in (batch.states, batch.actions)
    )
    return (
        dict(vars(batch)),
        dict(vars(batch.identity)),
        arrays,
        traits,
        None if targets is None else dict(vars(targets)),
    )


def _require_original(batch, targets, pin):
    fields, identity, arrays, traits, labels = pin
    for value, original in ((batch, fields), (batch.identity, identity)):
        if set(vars(value)) != set(original) or any(
            getattr(value, k) is not v for k, v in original.items()
        ):
            raise ValueError("original linear control feature/identity owner changed")
    if targets is not None and (
        set(vars(targets)) != set(labels)
        or any(getattr(targets, k) is not v for k, v in labels.items())
    ):
        raise ValueError("original linear control targets changed")
    for t, array, trait in zip((batch.states, batch.actions), arrays, traits, strict=True):
        _tensor(t, array.shape[0], array.shape[1])
        if (t._version, t.untyped_storage().data_ptr(), t.stride()) != trait or not np.array_equal(
            t.detach().numpy(), array
        ):
            raise ValueError("original linear control features changed")


def _start_report(report, key, rows, width, classes, pin):
    if type(report) is not dict or key in report:
        raise ValueError("linear control requires an unused exact report key")
    item = report[key] = dict(
        definition=DEFINITION,
        complete=False,
        attempts=1,
        returns=0,
        rows_attempted=rows,
        rows_returned=0,
        feature_dim=width,
        classes=classes,
        source_guard_array_bytes=sum(a.nbytes for a in pin[2]),
        feature_input_array_bytes=rows * width * 8,
        design_array_bytes=rows * (width + 1) * 8,
        coefficient_array_bytes=(width + 1) * classes * 8,
        output_probability_array_bytes=0,
        unreturned_work_unknown=True,
        native_head_forwards=0,
        encoder_forwards=0,
        optimizer_steps=0,
    )
    return item


def fit_clm_linear_control(batch, targets, report):
    """One detached solve; TRAIN/probe inputs keep separate fitting roles."""
    if type(batch) not in (FrozenCatalogTrainingBatch, FrozenProbeCatalogBatch):
        raise ValueError(
            "linear fit admits exact TRAIN or disposable probe_train catalog batches only"
        )
    _exact(batch, type(batch))
    _exact(targets, CLMCatalogTargets)
    _features(batch, batch.candidate_ids)
    if targets.candidate_ids != batch.candidate_ids or len(targets.candidate_indices) != len(
        batch.states
    ):
        raise ValueError("linear control requires matching ordered catalog labels for every row")
    pin = _capture(batch, targets)
    row = _start_report(
        report,
        "linear_fit",
        len(batch.states),
        batch.identity.embedding_dim,
        len(batch.candidate_ids),
        pin,
    )
    row["one_hot_array_bytes"] = len(batch.states) * len(batch.candidate_ids) * 8
    started = monotonic()
    try:
        features = pin[2][0].astype(np.float64)
        labels = np.array(targets.candidate_indices, dtype=np.float64).reshape(-1, 1)
        model = fit_wide_categorical_linear(features, labels, len(batch.candidate_ids))
        row.update(returns=1, rows_returned=len(features))
        if type(model) is not WideCategoricalLinearFit or model.fit_rows != len(features):
            raise ValueError("linear fit returned a different fitted row count or model type")
        _require_original(batch, targets, pin)
        result = CLMLinearFitControl(
            deepcopy(batch.identity), batch.candidate_ids, batch.role, model
        )
        row.update(complete=True, unreturned_work_unknown=False)
        return result
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"[:512]
        raise
    finally:
        row["seconds"] = monotonic() - started


def _active_candidate_columns(candidate_ids, active_ids):
    _candidate_ids(active_ids)
    if tuple(key for key in candidate_ids if key in active_ids) != active_ids:
        raise ValueError("active catalog must be an ordered subset of original candidate IDs")
    return tuple(candidate_ids.index(key) for key in active_ids)


def predict_clm_linear_control(control, batch, candidate_ids, report, *, active_candidate_ids=None):
    """Read-only class scores with explicit catalog order; no held-out labels."""
    _exact(control, CLMLinearFitControl)
    kind = FrozenChoiceBatch if control.role == "train" else FrozenProbeChoiceBatch
    _exact(batch, kind)
    allowed = (
        ("train", "outer_selection")
        if control.role == "train"
        else ("train", "probe_train", "probe_test", "outer_selection")
    )
    if (
        batch.role not in allowed
        or candidate_ids != control.candidate_ids
        or batch.identity != control.identity
    ):
        raise ValueError("linear readout role, encoder or original catalog differs")
    read_ids = candidate_ids if active_candidate_ids is None else active_candidate_ids
    columns = (
        None if active_candidate_ids is None else _active_candidate_columns(candidate_ids, read_ids)
    )
    _features(batch, read_ids, min_rows=1)
    pin, original = _capture(batch), asdict(control)
    row = _start_report(
        report,
        "linear_prediction",
        len(batch.states),
        control.model.feature_dim,
        len(read_ids),
        pin,
    )
    if columns is not None:
        row.update(
            definition="clm_frozen_feature_linear_active_catalog_v1",
            source_candidate_ids=list(candidate_ids),
            active_candidate_ids=list(read_ids),
            active_columns=list(columns),
            source_coefficient_array_bytes=(control.model.feature_dim + 1) * len(candidate_ids) * 8,
            projection_is_read_only=True,
        )
    started = monotonic()
    try:
        features = pin[2][0].astype(np.float64)
        probabilities = (
            predict_wide_categorical_linear(control.model, features)
            if columns is None
            else predict_wide_categorical_active(control.model, features, columns)
        )
        row.update(
            returns=1,
            rows_returned=len(features),
            output_probability_array_bytes=probabilities.nbytes,
        )
        _require_original(batch, None, pin)
        _exact(control, CLMLinearFitControl)
        if asdict(control) != original:
            raise ValueError("original fitted linear control changed during readout")
        validate_class_probabilities(probabilities, len(read_ids))
        if probabilities.shape != (len(features), len(read_ids)) or np.shares_memory(
            probabilities, features
        ):
            raise ValueError("linear readout returned invalid or shared probabilities")
        row.update(complete=True, unreturned_work_unknown=False)
        return probabilities
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"[:512]
        raise
    finally:
        row["seconds"] = monotonic() - started
