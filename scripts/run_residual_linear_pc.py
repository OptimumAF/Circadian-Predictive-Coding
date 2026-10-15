"""Print exact residual initialization/factors and one real step per control."""

from dataclasses import asdict
import json
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

from src.core.linear_pc_reference import evaluate_linear_equilibrium
from src.core.residual_linear_pc import (
    LinearResidualConfig,
    effective_residual_weights,
    initialize_residual_linear,
    step_residual_linear_pc,
)


def run_example() -> dict[str, object]:
    features = np.array([[0.3, -0.2], [-0.5, 0.4]], dtype=np.float64)
    targets = np.array([[1.0], [-1.0]], dtype=np.float64)
    controls = []
    for scheme in ("legacy_gaussian", "fan_in", "mean_field"):
        config = LinearResidualConfig(scheme, 2, 3, 2)
        state = initialize_residual_linear(config, seed=23)
        effective = effective_residual_weights(state)
        result = step_residual_linear_pc(state, features, targets, learning_rate=0.01)
        updated = effective_residual_weights(result.state)
        after = evaluate_linear_equilibrium(updated, features, targets)
        controls.append(
            {
                "config": asdict(config),
                "depth": config.depth,
                "identity_layer_indices_zero_based": list(range(1, config.hidden_layers)),
                "raw_initial_stddevs": list(config.initial_stddevs),
                "branch_forward_scales": list(config.forward_scales),
                "raw_learning_rate_factor": config.raw_learning_rate_factor,
                "effective_learning_rate_factors": list(config.effective_learning_rate_factors),
                "initial_raw_weights": [a.tolist() for a in state.raw_weights],
                "initial_effective_weights": [a.tolist() for a in effective],
                "updated_raw_weights": [a.tolist() for a in result.state.raw_weights],
                "updated_effective_weights": [a.tolist() for a in updated],
                "preupdate_energy": result.evaluation.energy,
                "preupdate_mse": result.evaluation.mse,
                "preupdate_rescaling": result.evaluation.rescaling,
                "updated_energy": after.energy,
            }
        )
    return {
        "protocol_id": "residual_linear_pc_equation_example_v1",
        "scope": "engineering_equation_check",
        "reference_author_commit": "18a732e1d495d833cb97cf305f9d6fbca7bdddeb",
        "seed": 23,
        "learning_rate": 0.01,
        "features": features.tolist(),
        "targets": targets.tolist(),
        "controls": controls,
        "limitations": "Scalar linear, fixed middle identity skips, unit precision; no residual study, convergence or nonlinear/CPC claim.",
    }


def main() -> None:
    print(json.dumps(run_example(), allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
