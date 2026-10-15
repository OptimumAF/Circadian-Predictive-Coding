"""Show raw nonlinear split/compatible merge and owned learner continuation."""

import argparse
from dataclasses import asdict
import json
import os
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

from src.adapters.coupled_scaling_learner import CoupledScalingLearner
from src.adapters.topology_coupled_scaling_learner import TopologyCoupledScalingLearner
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_energy import evaluate_coupled, feedforward_latents, predict_coupled
from src.core.coupled_pc_scaling import (
    CoupledScalingConfig,
    CoupledScalingState,
    copy_coupled_scaling_state,
    effective_coupled_parameters,
    initialize_coupled_scaling,
)
from src.core.coupled_scaling_structure import (
    merge_coupled_scaling_width,
    split_coupled_scaling_width,
)
from src.core.coupled_scaling_topology import CoupledScalingTopologyState


def describe_example() -> dict[str, Any]:
    return dict(
        example_id="raw_coupled_dyadic_structure_engineering_v1",
        classification="engineering_state_check_not_scientific_comparison",
        variants=["pc", "neutral_cpc", "active_cpc"],
        scheme="mean_field",
        input_dim=2,
        source_width=3,
        hidden_layers=2,
        gamma0=1.5,
        precisions=[0.8, 1.2],
        replication_factor=2,
        seed=23,
        learning_rate=0.01,
        source_inference_rate=0.05,
        split_inference_rate=0.1,
        inference_steps=2,
        training_calls=12,
        guarded_inference_calls=0,
        objective_evaluations=60,
        limitations="Diagonal energy/function preservation in exact arithmetic; encoding rounds. No raw-bitwise, raw-SGD, guarded-stop, runtime-publication or scientific-benefit identity.",
    )


def _default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"unsupported example value: {type(value).__name__}")


def _canonical(value: Any) -> str:
    return json.dumps(value, default=_default, sort_keys=True, allow_nan=False)


def _learner(state: CoupledScalingState, inference_rate: float) -> CoupledScalingLearner:
    return CoupledScalingLearner(
        state, learning_rate=0.01, inference_steps=2, inference_learning_rate=inference_rate
    )


def _source(variant: str, features: np.ndarray, targets: np.ndarray) -> CoupledScalingState:
    policy = (
        None
        if variant == "pc"
        else CoupledCircadianConfig.matched_pc_control()
        if variant == "neutral_cpc"
        else CoupledCircadianConfig()
    )
    state = initialize_coupled_scaling(
        CoupledScalingConfig("mean_field", 2, 3, 2, 1.5, (0.8, 1.2), policy), 23
    )
    for index, bias in enumerate(state.raw_parameters.hidden_biases):
        bias[:] = np.array([[0.01, -0.02, 0.03]]) * (index + 1)
    state.raw_parameters.output_bias[:] = 0.02
    learner = _learner(state, 0.05)
    for _ in range(2):
        learner.train_batch(features.copy(), targets.copy())
    return learner.snapshot_state()


def _record(state: CoupledScalingState) -> dict[str, Any]:
    return dict(
        state=asdict(state),
        effective_parameters=asdict(effective_coupled_parameters(state)),
        forward_scales=state.config.forward_scales,
        bias_scale=state.config.bias_scale,
        raw_rate_factor=state.config.raw_learning_rate_factor,
        effective_rate_factors=state.config.effective_learning_rate_factors,
        effective_bias_rate_factor=state.config.effective_bias_learning_rate_factor,
    )


def _diagonal_check(
    small: CoupledScalingState, wide: CoupledScalingState, features: np.ndarray, targets: np.ndarray
) -> dict[str, Any]:
    left, right = effective_coupled_parameters(small), effective_coupled_parameters(wide)
    # Nonzero residuals exercise every hidden penalty, beyond feedforward BCE.
    latents = tuple(z + 0.03 * (i + 1) for i, z in enumerate(feedforward_latents(left, features)))
    lifted = tuple(np.repeat(z, 2, axis=1) for z in latents)
    assert small.config.precisions is not None and wide.config.precisions is not None
    a = evaluate_coupled(left, features, targets, latents, small.config.precisions)
    b = evaluate_coupled(right, features, targets, lifted, wide.config.precisions)
    gradient_errors = [
        float(np.max(np.abs(np.repeat(g, 2, axis=1) / 2 - h)))
        for g, h in zip(a.latent_gradients, b.latent_gradients)
    ]
    probability_error = float(
        np.max(np.abs(predict_coupled(left, features) - predict_coupled(right, features)))
    )
    if (
        abs(a.energy - b.energy) > 1e-12
        or max(gradient_errors) > 1e-12
        or probability_error > 1e-12
    ):
        raise AssertionError("fixed diagonal fixture exceeds its declared numerical check")
    return dict(
        latents=latents,
        lifted_latents=lifted,
        source_evaluation=asdict(a),
        split_evaluation=asdict(b),
        latent_gradient_max_errors=gradient_errors,
        prediction_max_error=probability_error,
    )


def run_example() -> dict[str, Any]:
    features = np.array([[0.2, -0.1], [-0.3, 0.4]])
    targets = np.array([[1.0], [0.0]])
    rows, parity = [], {}
    for variant in describe_example()["variants"]:
        source = _source(variant, features, targets)
        before = _canonical(asdict(source))
        split = split_coupled_scaling_width(source)
        initial_check = _diagonal_check(source, split.state, features, targets)
        merged = merge_coupled_scaling_width(split.state, split.lineage)
        strict = _learner(source, 0.05)
        try:
            strict.restore_state(split.state)
        except ValueError:
            strict_refused = True
        else:
            raise AssertionError("strict source learner admitted changed width/precision")
        topology = TopologyCoupledScalingLearner(
            source, learning_rate=0.01, inference_steps=2, inference_learning_rate=0.05
        )
        try:
            topology.restore_state(CoupledScalingTopologyState(1, split.state))
        except ValueError:
            topology_refused = True
        else:
            raise AssertionError("topology adapter lost its fixed precision binding")
        learner = _learner(split.state, 0.1)
        fork = learner.fork()
        diagnostic = learner.train_batch(features.copy(), targets.copy())
        fork.train_batch(features.copy(), targets.copy())
        continued = learner.snapshot_state()
        exact_fork = _canonical(asdict(continued)) == _canonical(asdict(fork.snapshot_state()))
        if not exact_fork:
            raise AssertionError("owned split fork differs on matched continued update")
        continued_merge = merge_coupled_scaling_width(continued, split.lineage)
        continued_check = _diagonal_check(continued_merge.state, continued, features, targets)
        incompatible = copy_coupled_scaling_state(continued)
        incompatible.raw_parameters.hidden_weights[-1][0, 1] += 0.01
        bad_before = _canonical(asdict(incompatible))
        try:
            merge_coupled_scaling_width(incompatible, split.lineage)
        except ValueError as error:
            refusal = str(error)
        else:
            raise AssertionError("asymmetric middle pair was admitted")
        if _canonical(asdict(incompatible)) != bad_before or _canonical(asdict(source)) != before:
            raise AssertionError("transform or continued candidate mutated its source")
        parity[variant] = asdict(continued.raw_parameters)
        rows.append(
            dict(
                variant=variant,
                source=_record(source),
                split=_record(split.state),
                lineage=asdict(split.lineage),
                split_encoding_error=split.max_effective_relative_error,
                immediate_merge=_record(merged.state),
                merge_encoding_error=merged.max_effective_relative_error,
                diagonal_check=initial_check,
                continued=_record(continued),
                continued_diagnostic=asdict(diagnostic),
                continued_merge=_record(continued_merge.state),
                continued_diagonal_check=continued_check,
                exact_fork=exact_fork,
                strict_restore_refused=strict_refused,
                topology_precision_restore_refused=topology_refused,
                incompatible_merge_refusal=refusal,
            )
        )
    neutral_exact = _canonical(parity["pc"]) == _canonical(parity["neutral_cpc"])
    if not neutral_exact:
        raise AssertionError("PC/neutral raw continuation differs")
    return dict(
        description=describe_example(),
        training_features=features,
        training_targets=targets,
        records=rows,
        exact_pc_neutral_continuation=neutral_exact,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", help="No native model work")
    args = parser.parse_args()
    encoded = json.dumps(
        describe_example() if args.describe else run_example(),
        default=_default,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    if len((encoded + "\n").replace("\n", os.linesep).encode("utf-8")) > 2 * 1024**2:
        raise RuntimeError("raw structure engineering example exceeds2MiB")
    print(encoded)


if __name__ == "__main__":
    main()
