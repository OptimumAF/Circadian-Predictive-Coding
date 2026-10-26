"""Print one deterministic continuous-PC math fixture, with no learning or IO.

Usage: python -m scripts.continuous_pc_example
Synthetic rows illustrate an energy calculation, not a simulator or benchmark.
"""

import json

import numpy as np

from src.core import continuous_bc as bc
from src.core.continuous_demonstration import ContinuousActionSpec
from src.core.continuous_pc import (
    evaluate_continuous_pc,
    feedforward_continuous_latents,
    predict_continuous_pc,
)
from src.core.coupled_energy import CoupledParameters


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
    initial = feedforward_continuous_latents(parameters, features, spec)
    at_initial = evaluate_continuous_pc(parameters, features, targets, initial, (1.0,), spec)
    # A declared free-state perturbation demonstrates a nonzero residual; no
    # gradient step, inferred optimum, trained model or policy is constructed.
    free = (initial[0] + np.array([[0.01, -0.02], [0.03, 0.01]], dtype=np.float64),)
    evaluated = evaluate_continuous_pc(parameters, features, targets, free, (1.0,), spec)
    reference = bc.continuous_bc_loss_and_gradients(
        features,
        targets,
        (
            parameters.hidden_weights[0],
            parameters.hidden_biases[0][0],
            parameters.output_weight,
            parameters.output_bias[0],
        ),
        spec,
    )
    head_exact = (
        at_initial.task_loss == reference.loss
        and np.array_equal(at_initial.predictions, reference.predictions)
        and np.array_equal(at_initial.parameter_gradients.output_weight, reference.gradients[2])
        and np.array_equal(at_initial.parameter_gradients.output_bias[0], reference.gradients[3])
        and np.array_equal(predict_continuous_pc(parameters, features, spec), reference.predictions)
    )
    if not head_exact:
        raise ValueError("the fixed continuous PC task head differs from the matched BC reference")
    print(
        json.dumps(
            dict(
                scenario="synthetic_continuous_pc_energy_math_fixture",
                energy_definition=evaluated.energy_definition,
                task_definition=evaluated.task_definition,
                rows=2,
                action_axes=4,
                hidden_depth=1,
                precisions=evaluated.precisions,
                feedforward_task_head_exact=head_exact,
                feedforward_task_loss=at_initial.task_loss,
                free_state_energy=evaluated.energy,
                free_state_task_loss=evaluated.task_loss,
                free_state_hidden_penalties=evaluated.hidden_penalties,
                free_state_predictions=evaluated.predictions.tolist(),
                free_state_latent_gradients=[
                    value.tolist() for value in evaluated.latent_gradients
                ],
                free_state_parameter_gradients=dict(
                    hidden_weights=[
                        value.tolist() for value in evaluated.parameter_gradients.hidden_weights
                    ],
                    hidden_biases=[
                        value.tolist() for value in evaluated.parameter_gradients.hidden_biases
                    ],
                    output_weight=evaluated.parameter_gradients.output_weight.tolist(),
                    output_bias=evaluated.parameter_gradients.output_bias.tolist(),
                ),
                optimizer_updates=0,
                rng_draws=0,
                native_steps=0,
            ),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
