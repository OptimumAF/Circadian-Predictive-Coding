"""Actual bounded BP/PC matching example and exact BP fork continuation."""

import json
from pathlib import Path
import subprocess
import sys


def _run(*arguments: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_scaled_matched_backprop", *arguments],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert not result.stderr and len(result.stdout.encode("utf-8")) <= 2 * 1024**2
    return json.loads(result.stdout)


def test_should_describe_without_model_updates() -> None:
    report = _run("--describe")
    assert report["raw_bp_step_calls"] == 9 and report["raw_pc_step_calls"] == 6
    assert report["native_bp_train_calls"] == 2 and "methods" not in report


def test_should_match_initial_models_rates_and_native_bp_and_preserve_exact_forks() -> None:
    report = _run()
    assert len(report["methods"]) == 3 and report["training_batch"]["role"] == "train"
    assert report["bp_and_coupled_evaluations"] == 46
    for method in report["methods"]:
        assert method["exact_shared_initial_state"] and method["exact_shared_initial_predictions"]
        assert method["exact_bp_fork_continuation"]
        assert method["bp_final"] == method["bp_fork_final"]
        assert method["initial"]["raw_state"]["config"]["precisions"] == [0.7, 1.3]
        for key in (
            "weight_forward_scales",
            "bias_forward_scale",
            "raw_learning_rate_factor",
            "effective_weight_learning_rate_factors",
            "effective_bias_learning_rate_factor",
        ):
            assert method["bp_final"][key] == method["pc_final"][key] == method["initial"][key]
        assert [d["definition"] for d in method["bp_diagnostics"]] == [
            "numpy_matched_binary_bce_preupdate_v1",
        ] * 2
        assert [d["definition"] for d in method["pc_diagnostics"]] == [
            "numpy_coupled_binary_energy_v1",
        ] * 2
        assert method["bp_final"]["raw_state"]["wake_state"] is None
        if method["scheme"] == "fan_in":
            control = method["native_fan_in_control"]
            assert control["numerical_agreement"]
            assert control["rtol"] == report["native_agreement_rtol"] == 1e-13
            assert control["atol"] == report["native_agreement_atol"] == 1e-15
            assert control["max_parameter_absolute_difference"] < 1e-13
            assert control["max_loss_absolute_difference"] < 1e-13
        else:
            assert method["native_fan_in_control"] is None
