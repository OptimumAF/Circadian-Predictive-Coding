"""Show accepted/rejected/failed pruning with retained source and real SGD."""

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

from src.app.linear_pruning_gate import PruneDamageLimits, PruneDecision, gate_linear_prune
from src.app.linear_scaling_measurement import ScalingBatch
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.linear_pc_scaling import (
    LinearScalingConfig,
    LinearScalingState,
    LinearScalingStep,
    step_linear_pc,
)
from src.core.residual_linear_pc import (
    LinearResidualConfig,
    LinearResidualState,
    LinearResidualStep,
    step_residual_linear_pc,
)

State = LinearScalingState | LinearResidualState


def describe_example() -> dict[str, Any]:
    return {
        "protocol_id": "linear_width_pruning_gate_equation_example_v1",
        "scope": "engineering_equation_check_not_scientific_comparison",
        "families": ["mlp", "residual"],
        "scenarios": ["disconnected_accepted", "energy_damage_rejected", "budget_failed"],
        "prune_index": 1,
        "initial_width": 2,
        "learning_rate": 0.01,
        "sgd_calls": 4,
        "equilibrium_evaluations": 20,
        "limitations": "Gate limits only declared probes; no global damage/future-SGD/nonlinear/CPC guarantee. No final-test data.",
    }


def _state(family: str, disconnected: bool) -> State:
    hidden = 2 if disconnected else 1
    raw = (
        (np.array([[0.4, 0.0]]), np.array([[0.3, 0.0], [0.0, 0.0]]), np.array([[0.5], [0.0]]))
        if disconnected
        else (np.zeros((1, 2)), np.array([[0.0], [1.0]]))
    )
    scheme = "mean_field" if disconnected else "legacy_gaussian"
    if family == "mlp":
        return LinearScalingState(LinearScalingConfig(scheme, 1, 2, hidden), raw)
    return LinearResidualState(LinearResidualConfig(scheme, 1, 2, hidden), raw)


def _state_record(state: State) -> dict[str, Any]:
    return {"config": asdict(state.config), "raw_weights": [a.tolist() for a in state.raw_weights]}


def _probes(disconnected: bool) -> tuple[ScalingBatch, ...]:
    if not disconnected:
        return (
            ScalingBatch("validation", ("energy-validation-0",), np.zeros((1, 1)), np.ones((1, 1))),
        )
    return (
        ScalingBatch(
            "train",
            ("probe-train-0", "probe-train-1"),
            np.array([[0.5], [-0.5]]),
            np.array([[0.2], [-0.2]]),
        ),
        ScalingBatch(
            "validation",
            ("probe-validation-0", "probe-validation-1"),
            np.array([[0.7], [-0.8]]),
            np.array([[0.3], [-0.4]]),
        ),
    )


def _decision_record(decision: PruneDecision) -> dict[str, Any]:
    return {
        "status": decision.status,
        "reason": decision.reason,
        "returned_state": _state_record(decision.state),
        "accepted_lineage": None if decision.lineage is None else asdict(decision.lineage),
        "diagnostics": [asdict(item) for item in decision.diagnostics],
        "planned_probe_count": decision.planned_probe_count,
    }


def _step(state: State, training: ScalingBatch) -> dict[str, Any]:
    result: LinearScalingStep | LinearResidualStep
    if isinstance(state, LinearScalingState):
        result = step_linear_pc(state, training.features, training.targets, 0.01)
    else:
        result = step_residual_linear_pc(state, training.features, training.targets, 0.01)
    return {"state": _state_record(result.state), "preupdate_energy": result.evaluation.energy}


def run_example() -> dict[str, Any]:
    # These are explicitly training labels; validation-only energy probes below
    # never supply labels to SGD, including continuation after rejected pruning.
    training = ScalingBatch(
        "train",
        ("continue-train-0", "continue-train-1"),
        np.array([[0.3], [-0.4]]),
        np.array([[0.1], [-0.2]]),
    )
    controls = []
    for family in ("mlp", "residual"):
        for scenario in describe_example()["scenarios"]:
            disconnected = scenario != "energy_damage_rejected"
            source = _state(family, disconnected)
            probes = _probes(disconnected)
            limits = (
                PruneDamageLimits(1e-12, 1e-12, 1e-12)
                if disconnected
                else PruneDamageLimits(0.0, 0.0, 0.24)
            )
            budget = ToyBudgetSession(
                ToyExecutionBudget(max_wall_seconds=0.0 if scenario == "budget_failed" else 5.0),
                monotonic,
            )
            decision = gate_linear_prune(source, 1, probes, limits=limits, budget=budget)
            expected = {
                "disconnected_accepted": "accepted",
                "energy_damage_rejected": "rejected",
                "budget_failed": "failed",
            }[scenario]
            if decision.status != expected:
                raise AssertionError(
                    f"unexpected {scenario} outcome: {decision.status}: {decision.reason}"
                )
            controls.append(
                {
                    "family": family,
                    "scenario": scenario,
                    "source": _state_record(source),
                    "limits": asdict(limits),
                    "probes": [
                        {
                            "role": p.role,
                            "row_ids": p.row_ids,
                            "features": p.features.tolist(),
                            "targets": p.targets.tolist(),
                        }
                        for p in probes
                    ],
                    "decision": _decision_record(decision),
                    "continuation": None
                    if scenario == "budget_failed"
                    else _step(decision.state, training),
                }
            )
    return {
        **describe_example(),
        "continuation_training": {
            "role": training.role,
            "row_ids": training.row_ids,
            "features": training.features.tolist(),
            "targets": training.targets.tolist(),
        },
        "controls": controls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear prune damage gate engineering example")
    parser.add_argument(
        "--describe", action="store_true", help="Describe without evaluating or SGD"
    )
    args = parser.parse_args()
    encoded = json.dumps(
        describe_example() if args.describe else run_example(), allow_nan=False, indent=2
    )
    if len(encoded.encode("utf-8")) > 1024**2:
        raise RuntimeError("engineering example exceeds its 1MiB output allowance")
    print(encoded)


if __name__ == "__main__":
    main()
