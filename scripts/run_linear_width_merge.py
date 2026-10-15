"""Show accepted, amplification-rejected and failed tolerant linear merges."""

import argparse
from dataclasses import asdict
import json
import os
from typing import Any, cast

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

from src.app.linear_merge_gate import MergeDamageLimits, MergeDecision, gate_linear_merge
from src.app.linear_scaling_measurement import ScalingBatch
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.linear_pc_scaling import (
    LinearScalingConfig,
    LinearScalingState,
    effective_linear_weights,
    step_linear_pc,
)
from src.core.linear_width_structure import (
    MERGE_RTOL,
    MERGE_ATOL,
    LinearWidthLineage,
    merge_linear_width,
    split_linear_width,
)
from src.core.residual_linear_pc import (
    LinearResidualConfig,
    LinearResidualState,
    effective_residual_weights,
    step_residual_linear_pc,
)

State = LinearScalingState | LinearResidualState


def describe_example() -> dict[str, Any]:
    return dict(
        protocol_id="linear_projected_merge_damage_engineering_v1",
        scope="engineering_equation_check_not_scientific_comparison",
        families=["mlp", "residual"],
        scenarios=["compatible_accepted", "amplification_rejected", "budget_failed"],
        merge_rtol=MERGE_RTOL,
        merge_atol=MERGE_ATOL,
        accepted_scheme="mean_field",
        accepted_gamma0=1.5,
        accepted_hidden_layers=2,
        accepted_source_width=2,
        split_index=1,
        amplification_scheme="legacy_gaussian",
        amplification_hidden_layers={"mlp": 1, "residual": 2},
        amplification_prediction_limit=1e-6,
        learning_rate=0.01,
        sgd_calls=2,
        equilibrium_evaluations=20,
        limitations="Observed train/validation damage only, not a global bound, serving approval or future-SGD identity. Original merge tolerance and amplification remain unchanged; no final-test data.",
    )


def _make(
    family: str, width: int, hidden: int, raw: tuple[np.ndarray, ...], scheme: str, gamma: float
) -> State:
    if family == "mlp":
        return LinearScalingState(LinearScalingConfig(scheme, 1, width, hidden, gamma), raw)
    return LinearResidualState(LinearResidualConfig(scheme, 1, width, hidden, gamma), raw)


def _source(family: str, amplification: bool) -> tuple[State, LinearWidthLineage]:
    if not amplification:
        source = _make(
            family,
            2,
            2,
            (
                np.array([[0.4, -0.2]]),
                np.array([[0.3, 0.1], [-0.1, 0.2]]),
                np.array([[0.5], [-0.3]]),
            ),
            "mean_field",
            1.5,
        )
        change = split_linear_width(source, 1)
        return change.state, change.lineage
    hidden = 1 if family == "mlp" else 2
    raw = (np.zeros((1, 1)),) + (() if hidden == 1 else (np.zeros((1, 1)),)) + (np.zeros((1, 1)),)
    change = split_linear_width(_make(family, 1, hidden, raw, "legacy_gaussian", 1.0), 0)
    # Preserve the existing MLP counterexample; residual H2 has a zero branch,
    # hence its complete middle matrix is the fixed identity.
    wide = (
        (np.array([[1e-13, -1e-13]]),)
        + (() if hidden == 1 else (np.zeros((2, 2)),))
        + (np.array([[1e20 + 1e9], [1e20 - 1e9]]),)
    )
    return _make(family, 2, hidden, wide, "legacy_gaussian", 1.0), change.lineage


def _record(state: State) -> dict[str, Any]:
    effective = (
        effective_linear_weights(state)
        if type(state) is LinearScalingState
        else effective_residual_weights(cast(LinearResidualState, state))
    )
    return dict(
        config=asdict(state.config),
        raw_weights=state.raw_weights,
        effective_weights=effective,
        forward_scales=state.config.forward_scales,
        raw_rate_factor=state.config.raw_learning_rate_factor,
        effective_rate_factors=state.config.effective_learning_rate_factors,
    )


def _probes(amplification: bool) -> tuple[ScalingBatch, ...]:
    return (
        ScalingBatch(
            "train",
            ("probe-train-0", "probe-train-1"),
            np.array([[1.0], [-1.0]]) if amplification else np.array([[0.5], [-0.5]]),
            np.zeros((2, 1)) if amplification else np.array([[0.2], [-0.2]]),
        ),
        ScalingBatch(
            "validation",
            ("probe-validation-0", "probe-validation-1"),
            np.array([[0.5], [-0.75]]) if amplification else np.array([[0.7], [-0.8]]),
            np.zeros((2, 1)) if amplification else np.array([[0.3], [-0.4]]),
        ),
    )


def _decision_record(decision: MergeDecision) -> dict[str, Any]:
    return dict(
        status=decision.status,
        reason=decision.reason,
        returned_state=_record(decision.state),
        accepted_lineage=None if decision.lineage is None else asdict(decision.lineage),
        diagnostics=[asdict(d) for d in decision.diagnostics],
        planned_probe_count=decision.planned_probe_count,
    )


def _default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"unsupported example value: {type(value).__name__}")


def _canonical(value: Any) -> str:
    return json.dumps(value, default=_default, sort_keys=True, allow_nan=False)


def run_example() -> dict[str, Any]:
    training = ScalingBatch(
        "train",
        ("continuation-0", "continuation-1"),
        np.array([[0.3], [-0.4]]),
        np.array([[0.1], [-0.2]]),
    )
    controls = []
    for family in describe_example()["families"]:
        for scenario in describe_example()["scenarios"]:
            amplification = scenario == "amplification_rejected"
            source, lineage = _source(family, amplification)
            # Encode before later work to retain an independent initial record.
            original = _canonical(_record(source))
            probes = _probes(amplification)
            limits = (
                MergeDamageLimits(1e-6, 0, 0)
                if amplification
                else MergeDamageLimits(1e-12, 1e-12, 1e-12)
            )
            pure_merge = None if not amplification else _record(merge_linear_width(source, lineage))
            ticks = iter((0.0, 1.0))
            budget = ToyBudgetSession(
                ToyExecutionBudget(max_wall_seconds=0 if scenario == "budget_failed" else 5),
                (lambda: next(ticks, 1.0)) if scenario == "budget_failed" else (lambda: 0.0),
            )
            decision = gate_linear_merge(source, lineage, probes, limits=limits, budget=budget)
            expected = {
                "compatible_accepted": "accepted",
                "amplification_rejected": "rejected",
                "budget_failed": "failed",
            }[scenario]
            if decision.status != expected:
                raise AssertionError(
                    f"unexpected {family}/{scenario}: {decision.status}: {decision.reason}"
                )
            continuation = None
            if decision.status == "accepted":
                if type(decision.state) is LinearScalingState:
                    step = step_linear_pc(
                        decision.state, training.features.copy(), training.targets.copy(), 0.01
                    )
                    continuation = dict(
                        state=_record(step.state), preupdate_energy=step.evaluation.energy
                    )
                else:
                    residual_step = step_residual_linear_pc(
                        cast(LinearResidualState, decision.state),
                        training.features.copy(),
                        training.targets.copy(),
                        0.01,
                    )
                    continuation = dict(
                        state=_record(residual_step.state),
                        preupdate_energy=residual_step.evaluation.energy,
                    )
            elif _canonical(_record(decision.state)) != original:
                raise AssertionError("refused merge lost complete source state")
            if _canonical(_record(source)) != original:
                raise AssertionError("merge gate or detached continuation mutated its input")
            controls.append(
                dict(
                    family=family,
                    scenario=scenario,
                    source=json.loads(original),
                    declared_lineage=asdict(lineage),
                    probes=[asdict(p) for p in probes],
                    limits=asdict(limits),
                    pure_merge_candidate=pure_merge,
                    decision=_decision_record(decision),
                    continuation=continuation,
                )
            )
    return dict(**describe_example(), continuation_training=asdict(training), controls=controls)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", help="No model/equilibrium/SGD work")
    args = parser.parse_args()
    encoded = json.dumps(
        describe_example() if args.describe else run_example(),
        default=_default,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    )
    if len((encoded + "\n").replace("\n", os.linesep).encode("utf-8")) > 2 * 1024**2:
        raise RuntimeError("linear merge example exceeds2MiB")
    print(encoded)


if __name__ == "__main__":
    main()
