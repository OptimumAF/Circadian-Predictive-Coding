"""One fixed owned PC update and detached restore, with no performance claim.

Usage: python -m scripts.continuous_pc_learner_example
Synthetic released arrays; no simulator, final/probe data, selection or IO.
"""

from dataclasses import asdict
import json

import numpy as np

from src.adapters.numpy_continuous_bc import ContinuousBCPolicy
from src.adapters.numpy_continuous_pc import ContinuousPCPolicy, NumpyContinuousPCLearner
from src.core.continuous_demonstration import ContinuousActionSpec
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
    original_inputs = (features.copy(), targets.copy())
    learner = NumpyContinuousPCLearner(
        ContinuousPCPolicy(
            ContinuousBCPolicy(spec, hidden_dim=2, seed=17),
            inference_policy=CoupledInferencePolicy(gradient_tolerance=None),
        )
    )
    initial = learner.snapshot_state()
    initial_prediction = learner.predict(features)
    update = learner.train_batch_with_work(features, targets)
    updated = learner.snapshot_state()
    updated_prediction = learner.predict(features)
    child = learner.fork()
    child.restore_state(initial)
    restored_prediction = child.predict(features)
    retained = learner.snapshot_state()
    fork_restored = np.array_equal(restored_prediction, initial_prediction)
    source_unchanged = all(
        np.array_equal(a, b)
        for a, b in zip(
            (*updated.parameters, *updated.velocity),
            (*retained.parameters, *retained.velocity),
            strict=True,
        )
    )
    inputs_owned = all(
        np.array_equal(a, b) for a, b in zip((features, targets), original_inputs, strict=True)
    )
    if (
        not (fork_restored and source_unchanged and inputs_owned)
        or retained.completed_updates != 1
        or retained.training_examples != 2
        or child.snapshot_state().completed_updates != 0
        or update.work["published_updates"] != 1
        or update.work["parameter_proposals_completed"] != 1
        or update.work["candidate_predictions_completed"] != 1
    ):
        raise ValueError("fixed owned continuous PC update failed ownership/publication gates")
    print(
        json.dumps(
            dict(
                scenario="synthetic_owned_continuous_pc_update",
                rows=2,
                action_axes=4,
                hidden_dim=2,
                seed=17,
                learning_rate=0.01,
                momentum=0.9,
                precision=1.0,
                inference_rate=0.05,
                inference_cap=4,
                diagnostic=asdict(update.diagnostic),
                preupdate_inferred_task_loss=update.task_loss,
                inference=asdict(update.inference),
                work=update.work,
                initial_target_free_actions=initial_prediction.tolist(),
                updated_target_free_actions=updated_prediction.tolist(),
                fork_restored_to_initial=fork_restored,
                source_unchanged_by_child_restore=source_unchanged,
                original_inputs_unchanged=inputs_owned,
                source_updates=retained.completed_updates,
                source_training_rows=retained.training_examples,
                child_updates=0,
                parameter_initializations=1,
                seeded_normal_values=12,
                native_steps=0,
                final_or_probe_data_rows=0,
            ),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
