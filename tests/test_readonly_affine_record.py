"""Literal report round trips; no fit, prediction, RNG or native environment."""

from copy import deepcopy
from dataclasses import asdict
import json
from typing import Any

import numpy as np
import pytest

from src.adapters.readonly_continuous_affine import readonly_affine_from_fit_record
from src.core import continuous_affine_fit as core
from src.core.continuous_readout_fit import _coefficient_norm
from src.core.cube_taskspace_state import cube_taskspace_spec


@pytest.fixture(autouse=True)
def no_model_work(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("record conversion must not fit/predict/draw a model or create native work")

    monkeypatch.setattr(core, "_solve", forbidden)
    monkeypatch.setattr(core, "fit_continuous_affine", forbidden)
    monkeypatch.setattr(core, "predict_continuous_affine", forbidden)
    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    monkeypatch.setattr(np, "matmul", forbidden)


def record():
    # Same production serialization shape, without loading any experiment.
    coefficients = tuple(tuple(float(i + j) / 100 for j in range(4)) for i in range(13))
    model = core.ContinuousAffineFit(
        cube_taskspace_spec(), coefficients, 198, 13, (1.0,) * 13, 1.0, 1.0
    )
    value = dict(asdict(model), coefficient_norm=_coefficient_norm(np.asarray(coefficients)))
    return json.loads(json.dumps(value)), model


def test_should_round_trip_complete_production_record_without_forwarding_diagnostic():
    value, expected = record()
    before = deepcopy(value)
    reader = readonly_affine_from_fit_record(value, role="train")
    assert reader.snapshot_state().model == expected
    assert value == before
    value["coefficients"][0][0] = 999.0
    value["spec"]["lower_bounds"][0] = -0.5
    assert reader.snapshot_state().model == expected


@pytest.mark.parametrize(
    "fault",
    [
        "extra",
        "missing_norm",
        "norm_nan",
        "norm_bool",
        "norm_mismatch",
        "coefficient_nan",
        "coefficient_shape",
        "spec_extra",
        "wrong_spec",
        "rank",
        "condition",
    ],
)
def test_should_refuse_malformed_or_inconsistent_report_without_work(fault):
    value, _ = record()
    if fault == "extra":
        value["unrecognized"] = 0
    elif fault == "missing_norm":
        del value["coefficient_norm"]
    elif fault.startswith("norm_"):
        value["coefficient_norm"] = {
            "norm_nan": float("nan"),
            "norm_bool": True,
            "norm_mismatch": 42.0,
        }[fault]
    elif fault == "coefficient_nan":
        value["coefficients"][0][0] = float("nan")
    elif fault == "coefficient_shape":
        value["coefficients"].append([0.0] * 4)
    elif fault == "spec_extra":
        value["spec"]["unused"] = 1
    elif fault == "wrong_spec":
        value["spec"]["observation_dim"] = 42
    elif fault == "rank":
        value["rank"] = 12
    else:
        value["condition_number"] = 2.0
    with pytest.raises(ValueError):
        readonly_affine_from_fit_record(value, role="train")


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final", True])
def test_should_refuse_nontrain_role_before_converting_record(role: Any):
    with pytest.raises(ValueError, match="TRAIN"):
        readonly_affine_from_fit_record({}, role=role)
