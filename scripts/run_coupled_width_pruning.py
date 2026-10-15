"""Show complete native PC/CPC prune transfer, observed gate and continuation."""

import argparse
from dataclasses import asdict
import json
import os
from time import monotonic
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

from src.app.coupled_pruning_gate import (
    CoupledPruneDecision,
    CoupledPruneLimits,
    gate_coupled_prune,
)
from src.app.linear_scaling_measurement import ScalingBatch
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork


def describe_example() -> dict[str, Any]:
    return {
        "protocol_id": "coupled_native_prune_transfer_equation_example_v1",
        "scope": "engineering_state_and_equation_check_not_scientific_comparison",
        "variants": ["pc", "neutral_cpc", "active_cpc"],
        "scenarios": ["disconnected_accepted", "damage_rejected", "budget_failed"],
        "seed": 23,
        "input_dim": 2,
        "hidden_dims": [3, 3],
        "precisions": [1.0, 1.0],
        "learning_rate": 0.03,
        "inference_steps": 2,
        "inference_learning_rate": 0.05,
        "train_calls": 12,
        "guarded_infer_calls": 24,
        "coupled_energy_evaluations": 120,
        "limitations": "Fixed-step relaxed energy, not equilibrium; equal steps not equal compute. No final data, global damage/future-learning/scaling claim or shared-holder publication.",
    }


def _parameters(model: CoupledPredictiveCodingNetwork) -> dict[str, Any]:
    params = model.snapshot_parameters()
    return {
        "hidden_weights": [a.tolist() for a in params.hidden_weights],
        "hidden_biases": [a.tolist() for a in params.hidden_biases],
        "output_weight": params.output_weight.tolist(),
        "output_bias": params.output_bias.tolist(),
    }


def _model_record(model: CoupledPredictiveCodingNetwork) -> dict[str, Any]:
    snapshot = model.snapshot_state()
    wake = snapshot.wake_state
    return {
        "snapshot_format": snapshot.format_version,
        "input_dim": model.input_dim,
        "hidden_dims": model.hidden_dims,
        "precisions": model.precisions,
        "parameters": _parameters(model),
        "wake_state": None
        if wake is None
        else {
            "chemical_decay": wake.chemical_decay,
            "chemical_buildup_rate": wake.chemical_buildup_rate,
            "plasticity_sensitivity": wake.plasticity_sensitivity,
            "min_plasticity": wake.min_plasticity,
            "chemical_state": wake.chemical_state.tolist(),
            "plasticity_state": wake.plasticity_state.tolist(),
            "wake_updates": wake.wake_updates,
        },
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
            ("validation-0", "validation-1", "validation-2"),
            np.array([[0.7, 0.2], [-0.5, 0.1], [0.1, -0.8]]),
            np.array([[1.0], [0.0], [0.0]]),
        ),
    )


def _source(variant: str, training: ScalingBatch) -> CoupledPredictiveCodingNetwork:
    model: CoupledPredictiveCodingNetwork
    if variant == "pc":
        model = CoupledPredictiveCodingNetwork(2, (3, 3), 23)
    else:
        config = (
            CoupledCircadianConfig.matched_pc_control()
            if variant == "neutral_cpc"
            else CoupledCircadianConfig()
        )
        model = CoupledCircadianPredictiveCodingNetwork(2, (3, 3), 23, config=config)
    for _ in range(2):
        model.train_epoch(training.features, training.targets, 0.03, 2, 0.05)
    # A declared engineering fixture: disconnect last unit after real history.
    # Parameter-only restore deliberately retains nonzero CPC chemistry/count.
    parameters = model.snapshot_parameters()
    parameters.hidden_weights[-1][:, 2] = 0
    parameters.hidden_biases[-1][:, 2] = 0
    parameters.output_weight[2, :] = 0
    model.restore_parameters(parameters)
    return model


def _record(decision: CoupledPruneDecision) -> dict[str, Any]:
    return {
        "status": decision.status,
        "reason": decision.reason,
        "returned_model": _model_record(decision.model),
        "accepted_lineage": None if decision.lineage is None else asdict(decision.lineage),
        "diagnostics": [asdict(item) for item in decision.diagnostics],
        "planned_probe_count": decision.planned_probe_count,
    }


def run_example() -> dict[str, Any]:
    probes = _probes()
    controls = []
    for variant in describe_example()["variants"]:
        source = _source(variant, probes[0])
        source_record = _model_record(source)
        for scenario in describe_example()["scenarios"]:
            unit = 0 if scenario == "damage_rejected" else 2
            limits = (
                CoupledPruneLimits(0, 0, 0)
                if scenario == "damage_rejected"
                else CoupledPruneLimits(1e-12, 1e-12, 1e-12)
            )
            budget = ToyBudgetSession(
                ToyExecutionBudget(max_wall_seconds=0.0 if scenario == "budget_failed" else 5.0),
                monotonic,
            )
            decision = gate_coupled_prune(
                source,
                1,
                unit,
                probes,
                limits=limits,
                budget=budget,
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
            recorded = _record(decision)
            continuation = None
            if scenario != "budget_failed":
                result = decision.model.train_epoch(
                    probes[0].features, probes[0].targets, 0.03, 2, 0.05
                )
                continuation = {
                    "training_role": probes[0].role,
                    "preupdate_energy": result.energy,
                    "model": _model_record(decision.model),
                }
            controls.append(
                {
                    "variant": variant,
                    "scenario": scenario,
                    "layer_index": 1,
                    "unit_index": unit,
                    "source": source_record,
                    "limits": asdict(limits),
                    "decision": recorded,
                    "continuation": continuation,
                }
            )
    parity = {}
    for scenario in describe_example()["scenarios"]:
        pc = next(c for c in controls if c["variant"] == "pc" and c["scenario"] == scenario)
        neutral = next(
            c for c in controls if c["variant"] == "neutral_cpc" and c["scenario"] == scenario
        )
        parity[scenario] = (
            pc["decision"]["returned_model"]["parameters"]
            == neutral["decision"]["returned_model"]["parameters"]
        )
        if pc["continuation"] is not None:
            parity[scenario + "_continuation"] = (
                pc["continuation"]["model"]["parameters"]
                == neutral["continuation"]["model"]["parameters"]
            )
    if not all(parity.values()):
        raise AssertionError("PC/neutral parameter parity was lost")
    return {
        **describe_example(),
        "probes": [
            {
                "role": p.role,
                "row_ids": p.row_ids,
                "features": p.features.tolist(),
                "targets": p.targets.tolist(),
            }
            for p in probes
        ],
        "pc_neutral_parameter_parity": parity,
        "controls": controls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Native coupled prune transfer engineering example"
    )
    parser.add_argument(
        "--describe", action="store_true", help="Describe without native model work"
    )
    args = parser.parse_args()
    encoded = json.dumps(
        describe_example() if args.describe else run_example(), allow_nan=False, indent=2
    )
    if len(encoded.encode("utf-8")) > 1024**2:
        raise RuntimeError("engineering example exceeds 1MiB output allowance")
    print(encoded)


if __name__ == "__main__":
    main()
