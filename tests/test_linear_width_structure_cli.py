"""Actual isolated engineering CLI: four SGD calls, no scientific experiment."""

import json
from math import ldexp
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from src.core.linear_pc_reference import evaluate_linear_equilibrium
from src.core.linear_pc_scaling import (
    LinearScalingConfig,
    LinearScalingState,
    effective_linear_weights,
)
from src.core.linear_width_structure import merge_linear_width, split_linear_width
from src.core.residual_linear_pc import (
    LinearResidualConfig,
    LinearResidualState,
    effective_residual_weights,
)


@pytest.mark.parametrize("family", ("mlp", "residual"))
def test_should_rebind_extreme_supported_gamma_without_losing_raw_weights(family):
    gamma = ldexp(1.0, -530)
    state: LinearScalingState | LinearResidualState
    effective: Any
    if family == "mlp":
        state = LinearScalingState(
            LinearScalingConfig("mean_field", 1, 1, 1, gamma),
            (np.ones((1, 1)), np.array([[gamma]])),
        )
        effective = effective_linear_weights
    else:
        state = LinearResidualState(
            LinearResidualConfig("mean_field", 1, 1, 1, gamma),
            (np.ones((1, 1)), np.array([[gamma]])),
        )
        effective = effective_residual_weights
    split = split_linear_width(state, 0)
    merged = merge_linear_width(split.state, split.lineage)
    assert np.all(split.state.raw_weights[-1] != 0)
    for raw, expected in zip(merged.raw_weights, state.raw_weights, strict=True):
        np.testing.assert_allclose(raw, expected, rtol=1e-12, atol=0)
    evaluations = [
        evaluate_linear_equilibrium(effective(s), np.ones((1, 1)), np.zeros((1, 1)))
        for s in (state, split.state, merged)
    ]
    for value in evaluations:
        assert value.energy == pytest.approx(0.25, rel=1e-12)
        assert value.predictions[0, 0] == pytest.approx(1.0, rel=1e-12)


def _invoke(*arguments):
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.run_linear_width_structure", *arguments],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    assert completed.stderr == ""
    assert len(completed.stdout.encode("utf-8")) < 1024**2
    return json.loads(completed.stdout)


def test_should_describe_without_states_or_updates():
    report = _invoke("--describe")
    assert report["sgd_calls"] == 4 and "controls" not in report
    assert report["families"] == ["mlp", "residual"]


def test_should_run_both_families_all_controls_and_continue_after_split_merge():
    report = _invoke()
    assert report["sgd_calls"] == 4 and report["equilibrium_evaluations"] == 32
    assert len(report["controls"]) == 6
    assert [(c["family"], c["scheme"]) for c in report["controls"]] == [
        (family, scheme)
        for family in ("mlp", "residual")
        for scheme in ("legacy_gaussian", "fan_in", "mean_field")
    ]
    for control in report["controls"]:
        assert control["source_config"]["width"] == 2 and control["expanded_config"]["width"] == 3
        assert control["lineage"]["source_config"] == control["source_config"]
        assert control["lineage"]["family"] == control["family"]
        assert control["lineage"]["split_index"] == 0 and control["lineage"]["format_version"] == 1
        assert control["incompatible_merge_refusal"]
        for stage in ("after_split", "after_merge"):
            for metric in ("energy", "rescaling", "mse"):
                assert control[stage][metric] == pytest.approx(
                    control["before"][metric], rel=1e-12, abs=1e-12
                )
            np.testing.assert_allclose(
                control[stage]["predictions"],
                control["before"]["predictions"],
                rtol=1e-12,
                atol=1e-12,
            )
        for source, merged in zip(
            control["source_raw_weights"], control["merged_raw_weights"], strict=True
        ):
            np.testing.assert_allclose(source, merged, rtol=1e-12, atol=1e-12)
        continuation = control["continuation"]
        if control["scheme"] == "mean_field":
            assert (
                control["source_effective_learning_rate_factors"]
                != control["expanded_effective_learning_rate_factors"]
            )
            assert continuation is not None
            assert continuation["wide_updated"]["energy"] == pytest.approx(
                continuation["projected_updated"]["energy"], rel=1e-12, abs=1e-12
            )
            assert all(
                np.isfinite(continuation["continued_original_width"][key])
                for key in ("energy", "rescaling", "mse")
            )
        else:
            assert continuation is None
