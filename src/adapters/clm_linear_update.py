"""Role-preserving warm residual update on existing frozen catalog batches.

Returns one detached current control and a paid-operation report; no encoding,
native-head work, deployment, replay selection, retained history or budget grant.
Caller owns lifetime exposure and experiment limits, including any replay rows.
"""

from copy import deepcopy
from dataclasses import asdict
from time import monotonic

import numpy as np

from src.adapters.clm_linear_fit_control import (
    CLMLinearFitControl,
    _capture,
    _features,
    _require_original,
    _start_report,
)
from src.core.clm_catalog import CLMCatalogTargets, FrozenCatalogTrainingBatch
from src.core.clm_catalog_probe import FrozenProbeCatalogBatch
from src.core.clm_head import _exact
from src.core.wide_categorical_linear_fit import (
    WideCategoricalLinearFit,
    update_wide_categorical_linear,
)

DEFINITION = "clm_frozen_feature_minimum_change_linear_update_v1"


def update_clm_linear_control(prior, batch, targets, report):
    """One paid residual solve; prior role, identity and catalog stay fixed."""
    _exact(prior, CLMLinearFitControl)
    kind = FrozenCatalogTrainingBatch if prior.role == "train" else FrozenProbeCatalogBatch
    _exact(batch, kind)
    _exact(targets, CLMCatalogTargets)
    _features(batch, batch.candidate_ids)
    if (
        batch.role != prior.role
        or batch.identity != prior.identity
        or batch.candidate_ids != prior.candidate_ids
        or targets.candidate_ids != batch.candidate_ids
        or len(targets.candidate_indices) != len(batch.states)
    ):
        raise ValueError("warm update requires original role, encoder, ordered catalog and labels")
    pin, prior_values = _capture(batch, targets), asdict(prior)
    owners = dict(vars(prior)), dict(vars(prior.identity))
    row = _start_report(
        report,
        "linear_update",
        len(batch.states),
        prior.model.feature_dim,
        prior.model.num_classes,
        pin,
    )
    row.update(
        definition=DEFINITION,
        initialization="minimum_change_from_prior_coefficients",
        source_coefficient_array_bytes=(prior.model.feature_dim + 1) * prior.model.num_classes * 8,
        one_hot_array_bytes=len(batch.states) * prior.model.num_classes * 8,
        residual_target_array_bytes=len(batch.states) * prior.model.num_classes * 8,
        correction_array_bytes=(prior.model.feature_dim + 1) * prior.model.num_classes * 8,
    )
    started = monotonic()
    try:
        features = pin[2][0].astype(np.float64)
        labels = np.array(targets.candidate_indices, dtype=np.float64).reshape(-1, 1)
        model = update_wide_categorical_linear(prior.model, features, labels)
        row.update(returns=1, rows_returned=len(features))
        if type(model) is not WideCategoricalLinearFit or model.fit_rows != len(features):
            raise ValueError("warm update returned a different model or row count")
        _require_original(batch, targets, pin)
        _exact(prior, CLMLinearFitControl)
        for current, original in zip((prior, prior.identity), owners, strict=True):
            if vars(current).keys() != original.keys() or any(
                getattr(current, key) is not value for key, value in original.items()
            ):
                raise ValueError("warm update prior owner changed")
        if asdict(prior) != prior_values:
            raise ValueError("warm update prior values changed")
        result = CLMLinearFitControl(
            deepcopy(prior.identity), prior.candidate_ids, prior.role, model
        )
        row.update(complete=True, unreturned_work_unknown=False)
        return result
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"[:512]
        raise
    finally:
        row["seconds"] = monotonic() - started
