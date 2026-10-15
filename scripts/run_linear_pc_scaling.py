"""Print exact initialization/factors and one step for three fixed linear controls."""

from dataclasses import asdict
import json

import numpy as np

from src.core.linear_pc_reference import evaluate_linear_equilibrium
from src.core.linear_pc_scaling import (
    LinearScalingConfig,
    effective_linear_weights,
    initialize_linear_scaling,
    step_linear_pc,
)


def run_example() -> dict[str, object]:
    features = np.array([[0.3, -0.2], [-0.5, 0.4]], dtype=np.float64)
    targets = np.array([[1.0], [-1.0]], dtype=np.float64)
    controls = []
    for scheme in ("legacy_gaussian", "fan_in", "mean_field"):
        config = LinearScalingConfig(scheme, 2, 3, 2)
        state = initialize_linear_scaling(config, seed=23)
        effective = effective_linear_weights(state)
        result = step_linear_pc(state, features, targets, learning_rate=0.01)
        updated = effective_linear_weights(result.state)
        after = evaluate_linear_equilibrium(updated, features, targets)
        controls.append(
            {
                "config": asdict(config),
                "raw_initial_stddevs": list(config.initial_stddevs),
                "forward_scales": list(config.forward_scales),
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
        "protocol_id": "linear_pc_scaling_control_example_v1",
        "scope": "engineering_equation_check",
        "reference_author_commit": "18a732e1d495d833cb97cf305f9d6fbca7bdddeb",
        "seed": 23,
        "learning_rate": 0.01,
        "features": features.tolist(),
        "targets": targets.tolist(),
        "controls": controls,
        "limitations": "Scalar linear/no bias/no skip; no width/depth study or learning advantage claim.",
    }


def main() -> None:
    print(json.dumps(run_example(), allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
