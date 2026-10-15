"""Exercise the fixed user-facing scaling control command, three tiny steps."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_should_print_fixed_linear_scaling_controls_and_real_steps():
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.run_linear_pc_scaling"],
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
    for control in report["controls"]:
        assert control["config"]["hidden_layers"] == 2
        assert len(control["initial_raw_weights"]) == 3
        np.testing.assert_allclose(
            control["preupdate_energy"],
            control["preupdate_mse"] / control["preupdate_rescaling"],
            atol=1e-14,
            rtol=1e-13,
        )
        for raw, scale, effective in zip(
            control["initial_raw_weights"],
            control["forward_scales"],
            control["initial_effective_weights"],
        ):
            np.testing.assert_array_equal(np.array(raw) * scale, effective)
        assert any(
            not np.array_equal(before, after)
            for before, after in zip(control["initial_raw_weights"], control["updated_raw_weights"])
        )
    mean_field = report["controls"][2]
    np.testing.assert_allclose(
        mean_field["forward_scales"], [1 / np.sqrt(2), 1 / np.sqrt(3), 1 / 3]
    )
    assert mean_field["raw_learning_rate_factor"] == 3.0
    np.testing.assert_allclose(mean_field["effective_learning_rate_factors"], [1.5, 1, 1 / 3])
