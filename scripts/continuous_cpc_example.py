"""Fixed PC/neutral-CPC identity and fork continuation, not a benefit study.

Usage: python -m scripts.continuous_cpc_example
Four tiny parameter updates total; no active scientific arm or simulator.
"""

from dataclasses import asdict, replace
import json

import numpy as np

from src.adapters.numpy_continuous_bc import ContinuousBCPolicy
from src.adapters.numpy_continuous_pc import (
    ContinuousPCPolicy,
    ContinuousPCSnapshot,
    NumpyContinuousPCLearner,
)
from src.core.continuous_demonstration import ContinuousActionSpec
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_inference import CoupledInferencePolicy

CPC_WORK = (
    "chemistry_attempts",
    "chemistry_returns",
    "gate_applications_attempted",
    "gate_applications_returned",
)


def _common_equal(left: ContinuousPCSnapshot, right: ContinuousPCSnapshot) -> bool:
    return (
        replace(left.policy, circadian=None) == replace(right.policy, circadian=None)
        and left.completed_updates == right.completed_updates
        and left.training_examples == right.training_examples
        and all(
            np.array_equal(a, b)
            for a, b in zip(
                (*left.parameters, *left.velocity),
                (*right.parameters, *right.velocity),
                strict=True,
            )
        )
    )


def _wake(snapshot: ContinuousPCSnapshot) -> dict:
    wake = snapshot.wake_state
    if wake is None:
        raise ValueError("neutral CPC lost its complete wake state")
    return dict(
        chemical_state=wake.chemical_state.tolist(),
        plasticity_state=wake.plasticity_state.tolist(),
        wake_updates=wake.wake_updates,
        min_plasticity=wake.min_plasticity,
    )


def _same_update(left, right) -> bool:
    return (
        left.diagnostic == right.diagnostic
        and left.task_loss == right.task_loss
        and left.inference == right.inference
        and left.work == {key: value for key, value in right.work.items() if key not in CPC_WORK}
        and all(right.work[key] == 1 for key in CPC_WORK)
    )


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
    policy = ContinuousPCPolicy(
        ContinuousBCPolicy(spec, hidden_dim=2, seed=17),
        inference_steps=2,
        inference_policy=CoupledInferencePolicy(gradient_tolerance=None),
    )
    pc = NumpyContinuousPCLearner(policy)
    neutral = NumpyContinuousPCLearner(
        replace(policy, circadian=CoupledCircadianConfig.matched_pc_control())
    )
    initial_prediction_equal = np.array_equal(pc.predict(features), neutral.predict(features))
    first_pc = pc.train_batch_with_work(features, targets)
    first_neutral = neutral.train_batch_with_work(features, targets)
    saved_pc, saved_neutral = pc.snapshot_state(), neutral.snapshot_state()
    first_equal = _common_equal(saved_pc, saved_neutral) and _same_update(first_pc, first_neutral)
    pc_child, neutral_child = pc.fork(), neutral.fork()
    pc_child.restore_state(saved_pc)
    neutral_child.restore_state(saved_neutral)
    second_pc = pc_child.train_batch_with_work(features, targets)
    second_neutral = neutral_child.train_batch_with_work(features, targets)
    final_pc, final_neutral = pc_child.snapshot_state(), neutral_child.snapshot_state()
    final_prediction_equal = np.array_equal(
        pc_child.predict(features), neutral_child.predict(features)
    )
    second_equal = _common_equal(final_pc, final_neutral) and _same_update(
        second_pc, second_neutral
    )
    source_pc, source_neutral = pc.snapshot_state(), neutral.snapshot_state()
    source_unchanged = (
        _common_equal(saved_pc, source_pc)
        and _common_equal(saved_neutral, source_neutral)
        and _wake(saved_neutral) == _wake(source_neutral)
    )
    inputs_unchanged = all(
        np.array_equal(a, b) for a, b in zip((features, targets), original_inputs, strict=True)
    )
    if (
        not all(
            (
                initial_prediction_equal,
                first_equal,
                second_equal,
                final_prediction_equal,
                source_unchanged,
                inputs_unchanged,
            )
        )
        or final_pc.completed_updates != 2
        or final_pc.training_examples != 4
        or source_pc.completed_updates != 1
        or neutral_child.wake_updates != 2
        or neutral.wake_updates != 1
        or final_neutral.wake_state is None
        or not np.array_equal(final_neutral.wake_state.plasticity_state, np.ones(2))
    ):
        raise ValueError("fixed PC/neutral-CPC common-state/continuation gates failed")
    print(
        json.dumps(
            dict(
                scenario="matched_continuous_pc_neutral_cpc_engineering_fixture",
                seed=17,
                rows=2,
                hidden_dim=2,
                action_axes=4,
                inference_cap=2,
                inference_rate=0.05,
                total_parameter_updates=4,
                total_training_row_exposures=8,
                parameter_initializations=2,
                seeded_normal_values=24,
                initial_prediction_exact=initial_prediction_equal,
                first_update_common_exact=first_equal,
                continuation_common_exact=second_equal,
                final_prediction_exact=final_prediction_equal,
                sources_unchanged=source_unchanged,
                inputs_unchanged=inputs_unchanged,
                source_updates=[source_pc.completed_updates, source_neutral.completed_updates],
                child_updates=[final_pc.completed_updates, final_neutral.completed_updates],
                pc_reports=[asdict(first_pc), asdict(second_pc)],
                neutral_reports=[asdict(first_neutral), asdict(second_neutral)],
                source_neutral_wake=_wake(source_neutral),
                child_neutral_wake=_wake(final_neutral),
                native_steps=0,
                final_or_probe_data_rows=0,
            ),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
