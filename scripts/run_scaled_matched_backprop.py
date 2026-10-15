"""Show matching nonlinear raw BP/PC initialization and rates, without a study."""

import argparse
from dataclasses import is_dataclass, asdict
import json
import os
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

from src.adapters.coupled_scaling_learner import CoupledScalingLearner
from src.adapters.scaled_matched_backprop_learner import ScaledMatchedBackpropLearner
from src.core.coupled_pc_scaling import (
    CoupledScalingConfig,
    CoupledScalingState,
    effective_coupled_parameters,
    initialize_coupled_scaling,
)
from src.core.matched_backprop import MatchedBackpropNetwork


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"unsupported matched baseline record: {type(value).__name__}")


def _encoded(value: Any) -> str:
    return json.dumps(value, default=_json_default, allow_nan=False, sort_keys=True)


def describe_example() -> dict[str, Any]:
    return {
        "protocol_id": "scaled_matched_bp_equation_control_v1",
        "scope": "engineering_matching_check_not_quality_or_scaling_study",
        "seed": 23,
        "input_dim": 2,
        "width": 3,
        "hidden_layers": 2,
        "schemes": ["legacy_gaussian", "fan_in", "mean_field"],
        "precisions": [0.7, 1.3],
        "mean_field_gamma0": 1.5,
        "learning_rate": 0.01,
        "pc_inference_steps": 2,
        "pc_inference_learning_rate": 0.05,
        "raw_bp_step_calls": 9,
        "raw_pc_step_calls": 6,
        "native_bp_train_calls": 2,
        "bp_and_coupled_evaluations": 46,
        "native_agreement_rtol": 1e-13,
        "native_agreement_atol": 1e-15,
        "limitations": "BP feedforward BCE and PC relaxed total energy differ; work differs. No accuracy, efficiency, width stability or circadian benefit claim, no heldout/final data.",
    }


def _record(state: CoupledScalingState) -> dict[str, Any]:
    config = state.config
    return {
        "raw_state": state,
        "effective_parameters": effective_coupled_parameters(state),
        "weight_forward_scales": config.forward_scales,
        "bias_forward_scale": config.bias_scale,
        "raw_learning_rate_factor": config.raw_learning_rate_factor,
        "effective_weight_learning_rate_factors": config.effective_learning_rate_factors,
        "effective_bias_learning_rate_factor": config.effective_bias_learning_rate_factor,
    }


def _method(scheme: str, features: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    config = CoupledScalingConfig(
        scheme,
        2,
        3,
        2,
        gamma0=1.5 if scheme == "mean_field" else 1.0,
        precisions=(0.7, 1.3),
    )
    state = initialize_coupled_scaling(config, 23)
    bp = ScaledMatchedBackpropLearner(state, learning_rate=0.01)
    pc = CoupledScalingLearner(
        state, learning_rate=0.01, inference_steps=2, inference_learning_rate=0.05
    )
    equal_initial = _encoded(bp.snapshot_state()) == _encoded(pc.snapshot_state())
    equal_predictions = _encoded(bp.predict(features)) == _encoded(pc.predict(features))
    if not equal_initial or not equal_predictions:
        raise AssertionError("BP and PC do not start from the same raw/effective model")
    first_bp, first_pc = bp.train_batch(features, targets), pc.train_batch(features, targets)
    after_first_bp, after_first_pc = bp.snapshot_state(), pc.snapshot_state()
    fork = bp.fork()
    second_bp, second_pc = bp.train_batch(features, targets), pc.train_batch(features, targets)
    second_fork = fork.train_batch(features, targets)
    final_bp, final_pc, fork_final = bp.snapshot_state(), pc.snapshot_state(), fork.snapshot_state()
    exact_fork = _encoded(final_bp) == _encoded(fork_final) and second_bp == second_fork
    if not exact_fork:
        raise AssertionError("raw BP fork lost state or fixed training policy")
    native_check = None
    if scheme == "fan_in":
        native = MatchedBackpropNetwork(effective_coupled_parameters(state))
        losses = [native.train_epoch(features, targets, 0.01).loss for _ in range(2)]
        expected, actual = native.snapshot_parameters(), effective_coupled_parameters(final_bp)
        pairs = list(
            zip(
                (
                    *actual.hidden_weights,
                    *actual.hidden_biases,
                    actual.output_weight,
                    actual.output_bias,
                ),
                (
                    *expected.hidden_weights,
                    *expected.hidden_biases,
                    expected.output_weight,
                    expected.output_bias,
                ),
            )
        )
        for observed, reference in pairs:
            np.testing.assert_allclose(observed, reference, rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(
            [first_bp.value, second_bp.value], losses, rtol=1e-13, atol=1e-15
        )
        native_check = {
            "reference_parameters": expected,
            "reference_preupdate_losses": losses,
            "numerical_agreement": True,
            "bitwise_parameters": _encoded(expected) == _encoded(actual),
            "max_parameter_absolute_difference": max(
                float(np.max(np.abs(a - b))) for a, b in pairs
            ),
            "max_loss_absolute_difference": float(
                np.max(np.abs(np.array(losses) - [first_bp.value, second_bp.value]))
            ),
            "rtol": 1e-13,
            "atol": 1e-15,
        }
    return {
        "scheme": scheme,
        "initial": _record(state),
        "exact_shared_initial_state": equal_initial,
        "exact_shared_initial_predictions": equal_predictions,
        "bp_after_first": _record(after_first_bp),
        "pc_after_first": _record(after_first_pc),
        "bp_final": _record(final_bp),
        "pc_final": _record(final_pc),
        "bp_fork_final": _record(fork_final),
        "exact_bp_fork_continuation": exact_fork,
        "bp_diagnostics": [first_bp, second_bp],
        "pc_diagnostics": [first_pc, second_pc],
        "bp_fork_second_diagnostic": second_fork,
        "native_fan_in_control": native_check,
        "bp_predictions": bp.predict(features),
        "pc_predictions": pc.predict(features),
    }


def run_example() -> dict[str, Any]:
    features, targets = np.array([[0.2, -0.1], [-0.3, 0.4]]), np.array([[1.0], [0.0]])
    return {
        **describe_example(),
        "training_batch": {
            "role": "train",
            "row_ids": ["train-0", "train-1"],
            "features": features,
            "targets": targets,
        },
        "methods": [_method(scheme, features, targets) for scheme in describe_example()["schemes"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", help="Describe without model work")
    args = parser.parse_args()
    encoded = json.dumps(
        describe_example() if args.describe else run_example(),
        default=_json_default,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    if len(encoded.encode("utf-8")) > 2 * 1024**2:
        raise RuntimeError("baseline example exceeds 2MiB output allowance")
    print(encoded)


if __name__ == "__main__":
    main()
