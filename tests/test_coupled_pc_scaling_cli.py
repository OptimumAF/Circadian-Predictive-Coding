"""Actual bounded nonlinear equation/fork/native-control example."""

import json
from pathlib import Path
import subprocess
import sys


def _run(*arguments: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_coupled_pc_scaling", *arguments],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert not result.stderr
    assert len(result.stdout.encode("utf-8")) <= 2 * 1024**2
    return json.loads(result.stdout)


def test_should_describe_frozen_mapping_without_training() -> None:
    report = _run("--describe")
    assert report["raw_step_calls"] == 27 and report["native_train_calls"] == 6
    assert report["guarded_infer_calls"] == 0
    assert "methods" not in report


def test_should_run_all_raw_schemes_with_exact_forks_native_control_and_neutral_parity() -> None:
    report = _run()
    assert report["scope"] == "local_equation_and_state_check_not_scaling_measurement"
    assert len(report["methods"]) == 9
    assert all(report["pc_neutral_raw_parameter_parity"].values())
    assert report["training_batch"]["role"] == "train"
    assert report["coupled_energy_evaluations"] == 132
    for method in report["methods"]:
        assert method["exact_fork_continuation"]
        assert method["final"] == method["fork_final"]
        assert method["initial"]["raw_state"]["config"]["precisions"] == [0.7, 1.3]
        factor = method["initial"]["effective_bias_learning_rate_factor"]
        assert abs(factor - 1.0) < 1e-14
        assert all(
            d["definition"] == "numpy_coupled_binary_energy_v1"
            for d in method["training_diagnostics"]
        )
        wake = method["final"]["raw_state"]["wake_state"]
        if method["variant"] == "pc":
            assert wake is None
        else:
            assert wake["wake_updates"] == 2 and any(v > 0 for v in wake["chemical_state"])
            if method["variant"] == "neutral_cpc":
                assert wake["plasticity_state"] == [1.0] * 3
            else:
                assert all(0.2 <= v < 1.0 for v in wake["plasticity_state"])
        if method["scheme"] == "fan_in":
            native = method["native_fan_in_control"]
            assert native["numerical_complete_state_agreement"]
            assert native["rtol"] == report["native_agreement_rtol"] == 1e-13
            assert native["atol"] == report["native_agreement_atol"] == 1e-15
            assert native["max_state_absolute_difference"] < 1e-13
            assert native["max_energy_absolute_difference"] < 1e-13
        else:
            assert method["native_fan_in_control"] is None
