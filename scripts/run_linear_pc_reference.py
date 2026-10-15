"""Print one fixed, local linear PC equilibrium calculation as JSON.

This CLI owns the synthetic fixture and presentation, not learner state or data
authorization. Run with ``python -m scripts.run_linear_pc_reference``.
"""

import json

import numpy as np

from src.core.linear_pc_reference import LINEAR_ENERGY_ID, evaluate_linear_equilibrium


def run_reference() -> dict[str, object]:
    """Evaluate Eq. 2–6 on explicit tensors, then take one small gradient step."""
    weights = (
        np.array([[0.5, -0.25], [0.125, 0.75]], dtype=np.float64),
        np.array([[0.4], [-0.6]], dtype=np.float64),
    )
    features = np.array([[1.0, -0.5], [-0.25, 0.75]], dtype=np.float64)
    targets = np.array([[1.0], [-1.0]], dtype=np.float64)
    before = evaluate_linear_equilibrium(weights, features, targets)
    learning_rate = 0.01
    updated_weights = tuple(
        weight - learning_rate * gradient
        for weight, gradient in zip(weights, before.weight_gradients)
    )
    after = evaluate_linear_equilibrium(updated_weights, features, targets)
    # Adjacent residuals cancel at the optimum of every free hidden latent.
    stationarity = max(
        float(np.max(np.abs(residual - downstream @ weight.T)))
        for residual, downstream, weight in zip(
            before.residuals[:-1], before.residuals[1:], weights[1:]
        )
    )
    return {
        "energy_definition": LINEAR_ENERGY_ID,
        "reference": {
            "paper": "https://proceedings.mlr.press/v306/innocenti26a.html",
            "equations": [1, 2, 3, 4, 5, 6],
            "author_commit": "18a732e1d495d833cb97cf305f9d6fbca7bdddeb",
            "parameterization": "sp, gamma=1, unit precisions, no biases/skips",
        },
        "features": features.tolist(),
        "targets": targets.tolist(),
        "weights": [weight.tolist() for weight in weights],
        "predictions": before.predictions.tolist(),
        "equilibrium_latents": [latent.tolist() for latent in before.latents],
        "residuals": [residual.tolist() for residual in before.residuals],
        "weight_gradients": [gradient.tolist() for gradient in before.weight_gradients],
        "mse": before.mse,
        "rescaling": before.rescaling,
        "energy": before.energy,
        "energy_identity_absolute_error": abs(before.energy - before.mse / before.rescaling),
        "max_stationarity_residual": stationarity,
        "learning_rate": learning_rate,
        "updated_weights": [weight.tolist() for weight in updated_weights],
        "updated_energy": after.energy,
        "scope": "Synthetic correctness fixture; no scaling or comparative learning claim.",
    }


def main() -> None:
    print(json.dumps(run_reference(), allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
