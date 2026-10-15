"""Exercise the user-facing residual example, exactly three tiny update calls."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_should_print_fixed_residual_controls_and_real_steps():
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.run_residual_linear_pc"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    assert completed.stderr == ""
    report = json.loads(completed.stdout)
    assert report["scope"] == "engineering_equation_check"
    assert report["seed"] == 23 and report["learning_rate"] == 0.01
    assert [c["config"]["scheme"] for c in report["controls"]] == [
        "legacy_gaussian",
        "fan_in",
        "mean_field",
    ]
    x, y = np.array(report["features"]), np.array(report["targets"])
    for control in report["controls"]:
        assert control["config"]["hidden_layers"] == 2 and control["depth"] == 3
        assert control["identity_layer_indices_zero_based"] == [1]
        assert len(control["initial_raw_weights"]) == 3
        scales = control["branch_forward_scales"]
        raw_initial, raw_updated = control["initial_raw_weights"], control["updated_raw_weights"]
        for raw_group, effective_group in (
            (raw_initial, control["initial_effective_weights"]),
            (raw_updated, control["updated_effective_weights"]),
        ):
            for index, (raw, scale, effective) in enumerate(
                zip(raw_group, scales, effective_group)
            ):
                expected = np.array(raw) * scale
                if index == 1:
                    expected = expected + np.eye(3)
                np.testing.assert_array_equal(expected, effective)
        weights = tuple(np.array(a) for a in control["initial_effective_weights"])
        prediction = x.copy()
        for weight in weights:
            prediction = prediction @ weight
        denominator, suffix = 1.0, np.ones((1, 1))
        for weight in reversed(weights[1:]):
            suffix = weight @ suffix
            denominator += float(np.sum(suffix**2))
        half_mse = float(np.mean((prediction - y) ** 2) / 2)
        np.testing.assert_allclose(
            control["preupdate_energy"], half_mse / denominator, atol=1e-14, rtol=1e-13
        )
        np.testing.assert_allclose(
            control["preupdate_rescaling"], denominator, atol=1e-14, rtol=1e-13
        )
        assert any(not np.array_equal(a, b) for a, b in zip(raw_initial, raw_updated))
    mean_field = report["controls"][2]
    np.testing.assert_allclose(mean_field["branch_forward_scales"], [1 / np.sqrt(2), 1 / 3, 1 / 3])
    np.testing.assert_allclose(mean_field["effective_learning_rate_factors"], [1.5, 1 / 3, 1 / 3])
    assert mean_field["raw_learning_rate_factor"] == 3.0
