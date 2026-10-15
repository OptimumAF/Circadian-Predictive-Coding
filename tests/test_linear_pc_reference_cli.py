"""Exercise the actual fixed reference CLI and its report contract."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_should_run_fixed_linear_reference_cli() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_linear_pc_reference"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    report = json.loads(result.stdout)
    assert result.stderr == ""
    assert report["energy_definition"] == "linear_clamped_pc_equilibrium_v1"
    assert report["reference"]["author_commit"] == "18a732e1d495d833cb97cf305f9d6fbca7bdddeb"
    assert report["features"] == [[1.0, -0.5], [-0.25, 0.75]]
    assert report["targets"] == [[1.0], [-1.0]]
    assert report["learning_rate"] == 0.01
    np.testing.assert_allclose(report["rescaling"], 1.52, atol=1e-15, rtol=0)
    np.testing.assert_allclose(
        report["energy"], report["mse"] / report["rescaling"], atol=1e-14, rtol=1e-14
    )
    assert report["energy_identity_absolute_error"] < 1e-14
    assert report["max_stationarity_residual"] < 1e-14
    assert 0 < report["updated_energy"] < report["energy"]
    for weight, gradient, updated in zip(
        report["weights"], report["weight_gradients"], report["updated_weights"]
    ):
        np.testing.assert_allclose(
            updated, np.array(weight) - 0.01 * np.array(gradient), atol=1e-15, rtol=0
        )
