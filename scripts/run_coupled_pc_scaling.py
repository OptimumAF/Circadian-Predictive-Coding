"""Fixed nonlinear raw/effective equation check, not a scaling/quality study."""

import argparse
from dataclasses import asdict, is_dataclass
import json
import os
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

from src.adapters.coupled_scaling_learner import CoupledScalingLearner
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_pc_scaling import (
    CoupledScalingConfig,
    CoupledScalingState,
    effective_coupled_parameters,
    initialize_coupled_scaling,
)
from src.core.coupled_predictive_coding import CoupledSnapshot
from src.core.coupled_width_pruning import restore_coupled_model


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"unsupported equation record: {type(value).__name__}")


def _encoded(value: Any) -> str:
    return json.dumps(value, default=_json_default, allow_nan=False, sort_keys=True)


def describe_example() -> dict[str, Any]:
    return {
        "protocol_id": "coupled_nonlinear_raw_effective_equation_check_v1",
        "scope": "local_equation_and_state_check_not_scaling_measurement",
        "schemes": ["legacy_gaussian", "fan_in", "mean_field"],
        "variants": ["pc", "neutral_cpc", "active_cpc"],
        "seed": 23,
        "input_dim": 2,
        "width": 3,
        "hidden_layers": 2,
        "mean_field_gamma0": 1.5,
        "precisions": [0.7, 1.3],
        "learning_rate": 0.01,
        "inference_steps": 2,
        "inference_learning_rate": 0.05,
        "raw_step_calls": 27,
        "native_train_calls": 6,
        "guarded_infer_calls": 0,
        "coupled_energy_evaluations": 132,
        "native_agreement_rtol": 1e-13,
        "native_agreement_atol": 1e-15,
        "limitations": "Local nonlinear bias/precision/objective choice, no width invariance, convergence, compute efficiency or comparative learning benefit proved. No heldout/final data.",
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


def _native_agreement(
    actual: CoupledSnapshot,
    reference: CoupledSnapshot,
    energies: list[float],
    reference_energies: list[float],
) -> dict[str, Any]:
    if (
        actual.format_version != reference.format_version
        or actual.precisions != reference.precisions
    ):
        raise AssertionError("native control metadata differs")
    left, right = actual.parameters, reference.parameters
    pairs = list(
        zip(
            (*left.hidden_weights, *left.hidden_biases, left.output_weight, left.output_bias),
            (*right.hidden_weights, *right.hidden_biases, right.output_weight, right.output_bias),
        )
    )
    if (actual.wake_state is None) != (reference.wake_state is None):
        raise AssertionError("native control family differs")
    if actual.wake_state is not None and reference.wake_state is not None:
        a, b = actual.wake_state, reference.wake_state
        policy_fields = (
            "chemical_decay",
            "chemical_buildup_rate",
            "plasticity_sensitivity",
            "min_plasticity",
            "wake_updates",
        )
        if any(getattr(a, name) != getattr(b, name) for name in policy_fields):
            raise AssertionError("native control chemistry policy/count differs")
        pairs.extend(
            ((a.chemical_state, b.chemical_state), (a.plasticity_state, b.plasticity_state))
        )
    # Stable products and native G*gate then rate can round in a different order.
    # Freeze numerical agreement before fixtures; retain exact neutral/fork checks.
    for observed, expected in pairs:
        np.testing.assert_allclose(observed, expected, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(energies, reference_energies, rtol=1e-13, atol=1e-15)
    return {
        "reference_state": reference,
        "reference_preupdate_energies": reference_energies,
        "numerical_complete_state_agreement": True,
        "bitwise_complete_effective_state": _encoded(actual) == _encoded(reference),
        "max_state_absolute_difference": max(float(np.max(np.abs(a - b))) for a, b in pairs),
        "max_energy_absolute_difference": float(
            np.max(np.abs(np.array(energies) - reference_energies))
        ),
        "rtol": 1e-13,
        "atol": 1e-15,
    }


def _method(scheme: str, variant: str, features: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    chemistry = None
    if variant != "pc":
        chemistry = (
            CoupledCircadianConfig.matched_pc_control()
            if variant == "neutral_cpc"
            else CoupledCircadianConfig()
        )
    config = CoupledScalingConfig(
        scheme,
        2,
        3,
        2,
        gamma0=1.5 if scheme == "mean_field" else 1.0,
        precisions=(0.7, 1.3),
        circadian=chemistry,
    )
    initial = initialize_coupled_scaling(config, 23)
    learner = CoupledScalingLearner(
        initial,
        learning_rate=0.01,
        inference_steps=2,
        inference_learning_rate=0.05,
    )
    first = learner.train_batch(features, targets)
    after_first = learner.snapshot_state()
    fork = learner.fork()
    second = learner.train_batch(features, targets)
    fork_second = fork.train_batch(features, targets)
    final, fork_final = learner.snapshot_state(), fork.snapshot_state()
    continued_exact = _encoded(final) == _encoded(fork_final) and second == fork_second
    if not continued_exact:
        raise AssertionError("raw snapshot/fork continuation changed native state or policy")
    native_check = None
    if scheme == "fan_in":
        native = restore_coupled_model(
            "pc" if chemistry is None else "cpc",
            2,
            (3, 3),
            (0.7, 1.3),
            CoupledSnapshot(
                1, effective_coupled_parameters(initial), (0.7, 1.3), initial.wake_state
            ),
        )
        diagnostics = [native.train_epoch(features, targets, 0.01, 2, 0.05) for _ in range(2)]
        reference = native.snapshot_state()
        actual = CoupledSnapshot(
            1, effective_coupled_parameters(final), (0.7, 1.3), final.wake_state
        )
        native_check = _native_agreement(
            actual,
            reference,
            [first.value, second.value],
            [diagnostic.energy for diagnostic in diagnostics],
        )
    return {
        "scheme": scheme,
        "variant": variant,
        "initial": _record(initial),
        "after_first": _record(after_first),
        "final": _record(final),
        "fork_final": _record(fork_final),
        "exact_fork_continuation": continued_exact,
        "training_diagnostics": [first, second],
        "fork_second_diagnostic": fork_second,
        "native_fan_in_control": native_check,
        "predictions": learner.predict(features),
    }


def run_example() -> dict[str, Any]:
    features, targets = np.array([[0.2, -0.1], [-0.3, 0.4]]), np.array([[1.0], [0.0]])
    methods = [
        _method(scheme, variant, features, targets)
        for scheme in describe_example()["schemes"]
        for variant in describe_example()["variants"]
    ]
    parity = {}
    for scheme in describe_example()["schemes"]:
        pc, neutral = [m for m in methods if m["scheme"] == scheme][:2]
        parity[scheme] = all(
            _encoded(pc[stage]["raw_state"].raw_parameters)
            == _encoded(neutral[stage]["raw_state"].raw_parameters)
            for stage in ("initial", "after_first", "final", "fork_final")
        )
    if not all(parity.values()):
        raise AssertionError("PC/neutral raw parameter parity lost")
    return {
        **describe_example(),
        "training_batch": {
            "role": "train",
            "row_ids": ["train-0", "train-1"],
            "features": features,
            "targets": targets,
        },
        "pc_neutral_raw_parameter_parity": parity,
        "methods": methods,
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
        raise RuntimeError("equation example exceeds 2MiB output allowance")
    print(encoded)


if __name__ == "__main__":
    main()
