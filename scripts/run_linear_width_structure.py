"""Show owned linear MLP/residual split, compatible merge and real continuation."""

import argparse
from dataclasses import asdict
import json
import os
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

from src.core.linear_pc_reference import LinearEquilibrium, evaluate_linear_equilibrium
from src.core.linear_pc_scaling import (
    LinearScalingConfig,
    LinearScalingState,
    effective_linear_weights,
    initialize_linear_scaling,
    step_linear_pc,
)
from src.core.linear_width_structure import merge_linear_width, split_linear_width
from src.core.residual_linear_pc import (
    LinearResidualConfig,
    LinearResidualState,
    effective_residual_weights,
    initialize_residual_linear,
    step_residual_linear_pc,
)

State = LinearScalingState | LinearResidualState


def describe_example() -> dict[str, Any]:
    return {
        "protocol_id": "linear_isometric_width_equation_example_v1",
        "scope": "engineering_equation_check_not_scientific_comparison",
        "families": ["mlp", "residual"],
        "schemes": ["legacy_gaussian", "fan_in", "mean_field"],
        "seed": 23,
        "input_dim": 2,
        "initial_width": 2,
        "hidden_layers": 2,
        "split_index": 0,
        "learning_rate": 0.01,
        "sgd_calls": 4,
        "equilibrium_evaluations": 32,
        "limitations": "No pruning/nonlinear/CPC trace integration or future-SGD equality. Tolerance-compatible merge has no global function-error bound.",
    }


def _effective(state: State) -> tuple[np.ndarray, ...]:
    if isinstance(state, LinearScalingState):
        return effective_linear_weights(state)
    return effective_residual_weights(state)


def _evaluate(state: State, features: np.ndarray, targets: np.ndarray) -> LinearEquilibrium:
    return evaluate_linear_equilibrium(_effective(state), features, targets)


def _step(state: State, features: np.ndarray, targets: np.ndarray) -> State:
    if isinstance(state, LinearScalingState):
        return step_linear_pc(state, features, targets, 0.01).state
    return step_residual_linear_pc(state, features, targets, 0.01).state


def _numbers(evaluation: LinearEquilibrium) -> dict[str, Any]:
    return {
        "energy": evaluation.energy,
        "mse": evaluation.mse,
        "rescaling": evaluation.rescaling,
        "predictions": evaluation.predictions.tolist(),
    }


def _control(family: str, scheme: str, features: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    state: State = (
        initialize_linear_scaling(LinearScalingConfig(scheme, 2, 2, 2), 23)
        if family == "mlp"
        else initialize_residual_linear(LinearResidualConfig(scheme, 2, 2, 2), 23)
    )
    before = _evaluate(state, features, targets)
    split = split_linear_width(state, 0)
    expanded = split.state
    after_split = _evaluate(expanded, features, targets)
    merged = merge_linear_width(expanded, split.lineage)
    after_merge = _evaluate(merged, features, targets)
    # This deliberately incompatible candidate is detached; refusal must leave
    # its source state and the accepted split available to the caller.
    bad: State = (
        LinearScalingState(expanded.config, expanded.raw_weights)
        if isinstance(expanded, LinearScalingState)
        else LinearResidualState(expanded.config, expanded.raw_weights)
    )
    bad.raw_weights[0][0, -1] += 1.0
    try:
        merge_linear_width(bad, split.lineage)
    except ValueError as error:
        refused = str(error)
    else:
        raise AssertionError("off-image merge unexpectedly accepted")
    continuation = None
    if scheme == "mean_field":
        trained_wide = _step(expanded, features, targets)
        wide_evaluation = _evaluate(trained_wide, features, targets)
        trained_merged = merge_linear_width(trained_wide, split.lineage)
        merged_evaluation = _evaluate(trained_merged, features, targets)
        continued = _step(trained_merged, features, targets)
        continuation = {
            "wide_updated": _numbers(wide_evaluation),
            "projected_updated": _numbers(merged_evaluation),
            "continued_original_width": _numbers(_evaluate(continued, features, targets)),
            "wide_updated_raw_weights": [a.tolist() for a in trained_wide.raw_weights],
            "continued_raw_weights": [a.tolist() for a in continued.raw_weights],
        }
    return {
        "family": family,
        "scheme": scheme,
        "source_config": asdict(state.config),
        "expanded_config": asdict(expanded.config),
        "lineage": asdict(split.lineage),
        "source_forward_scales": state.config.forward_scales,
        "expanded_forward_scales": expanded.config.forward_scales,
        "source_effective_learning_rate_factors": state.config.effective_learning_rate_factors,
        "expanded_effective_learning_rate_factors": expanded.config.effective_learning_rate_factors,
        "source_raw_weights": [a.tolist() for a in state.raw_weights],
        "expanded_raw_weights": [a.tolist() for a in expanded.raw_weights],
        "merged_raw_weights": [a.tolist() for a in merged.raw_weights],
        "source_effective_weights": [a.tolist() for a in _effective(state)],
        "expanded_effective_weights": [a.tolist() for a in _effective(expanded)],
        "before": _numbers(before),
        "after_split": _numbers(after_split),
        "after_merge": _numbers(after_merge),
        "maximum_prediction_change_split": float(
            np.max(np.abs(before.predictions - after_split.predictions))
        ),
        "maximum_prediction_change_merge": float(
            np.max(np.abs(before.predictions - after_merge.predictions))
        ),
        "incompatible_merge_refusal": refused,
        "continuation": continuation,
    }


def run_example() -> dict[str, Any]:
    features = np.array([[0.3, -0.2], [-0.5, 0.4]], dtype=np.float64)
    targets = np.array([[1.0], [-1.0]], dtype=np.float64)
    return {
        **describe_example(),
        "features": features.tolist(),
        "targets": targets.tolist(),
        "controls": [
            _control(family, scheme, features, targets)
            for family in ("mlp", "residual")
            for scheme in ("legacy_gaussian", "fan_in", "mean_field")
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Isometric linear width split/merge example")
    parser.add_argument("--describe", action="store_true", help="Describe the example without SGD")
    args = parser.parse_args()
    encoded = json.dumps(
        describe_example() if args.describe else run_example(), allow_nan=False, indent=2
    )
    if len(encoded.encode("utf-8")) > 1024**2:
        raise RuntimeError("engineering example exceeds 1MiB output allowance")
    print(encoded)


if __name__ == "__main__":
    main()
