"""Show raw PC/CPC width rebinding, isolated damage decisions and continuation."""

import argparse
from dataclasses import asdict, replace
import json
import os
from time import monotonic
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

from src.adapters.coupled_scaling_learner import CoupledScalingLearner
from src.app.coupled_pruning_gate import CoupledPruneLimits
from src.app.coupled_scaling_pruning_gate import gate_coupled_scaling_prune
from src.app.linear_scaling_measurement import ScalingBatch
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_pc_scaling import (
    CoupledScalingConfig,
    CoupledScalingState,
    effective_coupled_parameters,
    initialize_coupled_scaling,
)
from src.core.coupled_scaling_pruning import MAX_EFFECTIVE_RELATIVE_ERROR


def describe_example() -> dict[str, Any]:
    return {
        "example_id": "coupled_raw_uniform_prune_engineering_v1",
        "classification": "engineering_state_check_not_scientific_comparison",
        "scheme": "mean_field",
        "gamma0": 1.5,
        "input_dim": 2,
        "source_width": 3,
        "hidden_layers": 2,
        "removed_units": [0, 1],
        "variants": ["pc", "neutral_cpc", "active_cpc"],
        "scenarios": ["disconnected_accepted", "damage_rejected", "budget_failed"],
        "seed": 23,
        "precisions": [1.0, 1.0],
        "learning_rate": 0.01,
        "inference_steps": 2,
        "inference_learning_rate": 0.05,
        "initial_wake_updates": 7,
        "state_fixture": "synthetic valid nonzero chemistry; not claimed as measured history",
        "max_effective_relative_error": MAX_EFFECTIVE_RELATIVE_ERROR,
        "training_calls": 6,
        "guarded_inference_calls": 24,
        "objective_evaluations": 96,
        "limitations": "Encoding tolerance is not a function/energy/trajectory bound. Fixed probe energy is not equilibrium. No final data, benefit claim or runtime publication.",
    }


def _default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"unsupported example value: {type(value).__name__}")


def _canonical(value: Any) -> str:
    return json.dumps(value, default=_default, sort_keys=True, allow_nan=False)


def _record(state: CoupledScalingState) -> dict[str, Any]:
    config = state.config
    return {
        "state": asdict(state),
        "effective_parameters": asdict(effective_coupled_parameters(state)),
        "forward_scales": config.forward_scales,
        "raw_rate_factor": config.raw_learning_rate_factor,
        "bias_scale": config.bias_scale,
        "effective_rate_factors": config.effective_learning_rate_factors,
        "effective_bias_rate_factor": config.effective_bias_learning_rate_factor,
    }


def _probes() -> tuple[ScalingBatch, ...]:
    return (
        ScalingBatch(
            "train",
            ("train-0", "train-1"),
            np.array([[0.2, -0.1], [-0.3, 0.4]]),
            np.array([[1.0], [0.0]]),
        ),
        ScalingBatch(
            "validation",
            ("validation-0", "validation-1"),
            np.array([[0.7, 0.2], [-0.5, 0.1]]),
            np.array([[1.0], [0.0]]),
        ),
    )


def _source(variant: str, *, damaging: bool = False) -> CoupledScalingState:
    chemistry = (
        CoupledCircadianConfig.matched_pc_control()
        if variant == "neutral_cpc"
        else CoupledCircadianConfig()
        if variant == "active_cpc"
        else None
    )
    state = initialize_coupled_scaling(
        CoupledScalingConfig("mean_field", 2, 3, 2, gamma0=1.5, circadian=chemistry), 23
    )
    parameters, wake = state.raw_parameters, state.wake_state
    for layer, unit in enumerate((0, 1)):
        parameters.hidden_weights[layer][:, unit] = 0
        parameters.hidden_biases[layer][:] = np.array([[0.01, 0.02, 0.03]])
        parameters.hidden_biases[layer][:, unit] = 0
        if layer + 1 < 2:
            parameters.hidden_weights[layer + 1][unit, :] = 0
        else:
            parameters.output_weight[unit, :] = 0
    parameters.output_bias[:] = 0.01
    if damaging:
        for value in (
            *parameters.hidden_weights,
            *parameters.hidden_biases,
            parameters.output_weight,
            parameters.output_bias,
        ):
            value.fill(0)
        parameters.hidden_biases[-1][0, 1] = np.arctanh(0.5) / state.config.bias_scale
        parameters.output_weight[1, 0] = 2.0 / state.config.forward_scales[-1]
    if wake is not None:
        chemicals = np.array([0.1, 0.4, 0.7])
        wake = replace(
            wake,
            chemical_state=chemicals,
            plasticity_state=np.clip(
                np.exp(-wake.plasticity_sensitivity * chemicals), wake.min_plasticity, 1.0
            ),
            wake_updates=7,
        )
    return CoupledScalingState(state.config, parameters, wake)


def _learner(state: CoupledScalingState) -> CoupledScalingLearner:
    return CoupledScalingLearner(
        state, learning_rate=0.01, inference_steps=2, inference_learning_rate=0.05
    )


def run_example() -> dict[str, Any]:
    probes = _probes()
    controls, continued = [], {}
    for variant in describe_example()["variants"]:
        for scenario in describe_example()["scenarios"]:
            source = _source(variant, damaging=scenario == "damage_rejected")
            initial = _record(source)
            limits = (
                CoupledPruneLimits(0, 0, 0)
                if scenario == "damage_rejected"
                else CoupledPruneLimits(1e-12, 1e-12, 1e-12)
            )
            decision = gate_coupled_scaling_prune(
                source,
                (0, 1),
                probes,
                limits=limits,
                budget=ToyBudgetSession(
                    ToyExecutionBudget(max_wall_seconds=0 if scenario == "budget_failed" else 5),
                    monotonic,
                ),
                inference_steps=2,
                inference_learning_rate=0.05,
            )
            expected = {
                "disconnected_accepted": "accepted",
                "damage_rejected": "rejected",
                "budget_failed": "failed",
            }[scenario]
            if decision.status != expected:
                raise AssertionError(
                    f"unexpected {variant}/{scenario}: {decision.status}: {decision.reason}"
                )
            continuation = None
            if decision.status == "accepted":
                strict = _learner(source)
                try:
                    strict.restore_state(decision.state)
                except ValueError:
                    strict_refused = True
                else:
                    raise AssertionError("old raw learner accepted incompatible geometry")
                learner = _learner(decision.state)
                fork = learner.fork()
                diagnostic = learner.train_batch(
                    probes[0].features.copy(), probes[0].targets.copy()
                )
                fork.train_batch(probes[0].features.copy(), probes[0].targets.copy())
                equal = _canonical(asdict(learner.snapshot_state())) == _canonical(
                    asdict(fork.snapshot_state())
                )
                if not equal:
                    raise AssertionError("accepted raw fork continuation differs")
                continued[variant] = asdict(learner.snapshot_state().raw_parameters)
                continuation = dict(
                    diagnostic=asdict(diagnostic),
                    state=_record(learner.snapshot_state()),
                    exact_fork=equal,
                    strict_old_restore_refused=strict_refused,
                )
            if _canonical(_record(source)) != _canonical(initial):
                raise AssertionError("source changed during gate or continuation")
            if decision.status != "accepted" and _canonical(_record(decision.state)) != _canonical(
                initial
            ):
                raise AssertionError("fallback lost full raw state")
            controls.append(
                dict(
                    variant=variant,
                    scenario=scenario,
                    source=initial,
                    limits=asdict(limits),
                    decision=asdict(decision),
                    continuation=continuation,
                )
            )
    neutral_exact = _canonical(continued["pc"]) == _canonical(continued["neutral_cpc"])
    if not neutral_exact:
        raise AssertionError("PC/neutral raw continuation differs")
    return dict(
        description=describe_example(),
        controls=controls,
        exact_pc_neutral_continuation=neutral_exact,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", help="No model work")
    args = parser.parse_args()
    report = describe_example() if args.describe else run_example()
    encoded = json.dumps(report, default=_default, allow_nan=False, indent=2)
    if len((encoded + "\n").replace("\n", os.linesep).encode("utf-8")) > 2 * 1024**2:
        raise RuntimeError("raw pruning example exceeds its 2MiB output allowance")
    print(encoded)


if __name__ == "__main__":
    main()
