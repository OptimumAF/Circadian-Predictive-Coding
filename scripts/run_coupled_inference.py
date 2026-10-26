"""Fixed coupled PC/neutral-CPC early-stop and fallback engineering trace."""

from dataclasses import asdict
import argparse
import json
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

from src.adapters.coupled_inference_stream import run_stream_example

from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.coupled_energy import CoupledParameters
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork


def _parameters(value: CoupledParameters) -> dict[str, object]:
    return {
        "hidden_weights": [a.tolist() for a in value.hidden_weights],
        "hidden_biases": [a.tolist() for a in value.hidden_biases],
        "output_weight": value.output_weight.tolist(),
        "output_bias": value.output_bias.tolist(),
    }


def run_example() -> dict[str, object]:
    features = np.array([[0.4, -0.2], [-0.3, 0.1], [0.2, 0.6]], dtype=np.float64)
    targets = np.array([[1.0], [0.0], [1.0]], dtype=np.float64)
    source = CoupledPredictiveCodingNetwork(2, (3, 2), seed=23)
    initial = source.snapshot_parameters()
    early = CoupledInferencePolicy(min_steps=2, gradient_tolerance=1.0)
    guarded = CoupledInferencePolicy(min_steps=2, gradient_tolerance=None)
    cases = []
    settings = [("fixed", "pc", None, 0.1)] + [
        (scenario, method, policy, rate)
        for scenario, policy, rate in (("early_stop", early, 0.1), ("fallback", guarded, 1e6))
        for method in ("pc", "neutral_cpc")
    ]
    for scenario, method, policy, inference_rate in settings:
        model: CoupledPredictiveCodingNetwork
        if method == "pc":
            model = CoupledPredictiveCodingNetwork(2, (3, 2), seed=23)
        else:
            model = CoupledCircadianPredictiveCodingNetwork(
                2, (3, 2), seed=23, config=CoupledCircadianConfig.matched_pc_control()
            )
        model.restore_parameters(initial)
        result = model.train_epoch(
            features, targets, 0.01, 6, inference_rate, inference_policy=policy
        )
        diagnostic = None if result.inference is None else asdict(result.inference)
        cases.append(
            {
                "scenario": scenario,
                "method": method,
                "policy": None if policy is None else asdict(policy),
                "inference_step_cap": 6,
                "inference_learning_rate": inference_rate,
                "energy": result.energy,
                "energy_definition": result.energy_definition,
                "inference": diagnostic,
                "fixed_inference_energy_evaluations": 7 if policy is None else None,
                "parameter_candidate_energy_evaluations": 1,
                "updated_parameters": _parameters(model.snapshot_parameters()),
                "wake_state": None
                if not isinstance(model, CoupledCircadianPredictiveCodingNetwork)
                else {
                    "chemical_state": model.get_chemical_state().tolist(),
                    "plasticity_state": model.get_plasticity_state().tolist(),
                    "wake_updates": model.wake_updates,
                },
            }
        )
    return {
        "protocol_id": "coupled_bounded_inference_example_v1",
        "scope": "engineering_control_no_heldout_data",
        "seed": 23,
        "learning_rate": 0.01,
        "precisions": [1.0, 1.0],
        "features": features.tolist(),
        "targets": targets.tolist(),
        "initial_parameters": _parameters(initial),
        "cases": cases,
        "limitations": "Counts include stopping norms/comparisons; no measured speed, matched-work learning benefit or convergence claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", action="store_true", help="run the fixed recurring stream demo")
    args = parser.parse_args()
    result = run_stream_example() if args.stream else run_example()
    print(json.dumps(result, allow_nan=False, indent=2))
    if args.stream and result.get("status") != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
