"""Run fixed continuous-PC latent inference without training a policy.

Usage: python -m scripts.continuous_pc_inference_example
Synthetic labels condition free states; deployed predictions remain target-free.
"""

from dataclasses import asdict
import json

import numpy as np

from src.core.continuous_demonstration import ContinuousActionSpec
from src.core.continuous_pc import predict_continuous_pc
from src.core.continuous_pc_inference import infer_continuous_pc
from src.core.coupled_energy import CoupledParameters
from src.core.coupled_inference import CoupledInferencePolicy


def main() -> None:
    spec = ContinuousActionSpec(
        2,
        ("normalized_root_dx", "normalized_root_dy", "normalized_root_dz", "normalized_gripper"),
        (-1.0,) * 4,
        (1.0,) * 4,
    )
    features = np.array([[0.2, -0.3], [0.7, 0.4]], dtype=np.float64)
    targets = np.array([[0.1, -0.2, 0.3, -1.0], [-0.4, 0.5, -0.6, 1.0]], dtype=np.float64)
    parameters = CoupledParameters(
        (np.array([[0.1, -0.2], [0.3, 0.15]], dtype=np.float64),),
        (np.array([[0.02, -0.03]], dtype=np.float64),),
        np.array([[0.2, -0.1, 0.15, 0.05], [-0.3, 0.25, 0.1, -0.2]], dtype=np.float64),
        np.array([[0.05, -0.04, 0.03, 0.01]], dtype=np.float64),
    )
    arrays = (
        *parameters.hidden_weights,
        *parameters.hidden_biases,
        parameters.output_weight,
        parameters.output_bias,
        features,
        targets,
    )
    originals = tuple(value.copy() for value in arrays)
    result = infer_continuous_pc(
        parameters,
        features,
        targets,
        (1.0,),
        spec,
        max_steps=4,
        learning_rate=0.05,
        policy=CoupledInferencePolicy(gradient_tolerance=None),
    )
    unchanged = all(np.array_equal(a, b) for a, b in zip(arrays, originals, strict=True))
    work = result.diagnostics
    if (
        not unchanged
        or work.reason != "max_steps"
        or work.used_fallback
        or work.attempted_steps != 4
        or work.accepted_steps != 4
        or work.energy_evaluations != 5
        or work.gradient_norm_evaluations != 5
        or work.energy_comparisons != 4
        or not work.final_energy < work.initial_energy
    ):
        raise ValueError("fixed continuous latent-inference fixture failed its declared gates")
    # No parameters changed. These actions use only features and must remain
    # separate from the target-conditioned free-state predictions in the report.
    target_free = predict_continuous_pc(parameters, features, spec)
    print(
        json.dumps(
            dict(
                scenario="synthetic_continuous_pc_latent_inference",
                energy_definition=result.evaluation.energy_definition,
                task_definition=result.evaluation.task_definition,
                rows=2,
                action_axes=4,
                hidden_depth=1,
                precisions=result.evaluation.precisions,
                learning_rate=0.05,
                maximum_steps=4,
                work=asdict(work),
                final_energy=result.evaluation.energy,
                final_task_loss=result.evaluation.task_loss,
                final_hidden_penalties=result.evaluation.hidden_penalties,
                target_conditioned_predictions=result.evaluation.predictions.tolist(),
                target_free_predictions=target_free.tolist(),
                unchanged_inputs=unchanged,
                parameter_updates=0,
                rng_draws=0,
                native_steps=0,
            ),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
