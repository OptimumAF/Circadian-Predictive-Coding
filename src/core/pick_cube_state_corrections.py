"""Owned canonical state42 feature views of raw readiness correction evidence.

Inputs are an exact readiness batch and the fixed quaternion sign recipe. The
output retains detached raw and derived batches with the same labels, student
actions, episode identity and arrival ticks. No label release, teacher query,
normalization fit, learner, geometry, simulator or IO occurs here.
"""

from dataclasses import dataclass

from src.core.pick_cube_readiness_corrections import (
    ReadinessCorrectionBatch,
    ObservedGraspCorrectionBatch,
    GoalConsistentObservedGraspCorrectionBatch,
    copy_readiness_correction_batch,
    ReadinessCorrectionLabel,
)
from src.core.pick_cube_state_quaternions import (
    PickCubeStateQuaternionProjection,
    state_quaternion_training_episode,
    validate_state_quaternion_projection,
)


@dataclass(frozen=True)
class StateQuaternionCorrectionWork:
    """Logical projection/pairing payload, excluding validation copies and RSS."""

    observation_projections: int
    source_values_validated: int
    projected_values: int
    projected_numeric_bytes: int
    projected_label_observation_values: int
    projected_label_target_values: int
    projected_label_observation_numeric_bytes: int
    projected_label_target_numeric_bytes: int
    quaternion_unit_checks: int
    quaternion_components_checked: int


@dataclass(frozen=True)
class StateQuaternionCorrectionView:
    source: ReadinessCorrectionBatch
    model: ReadinessCorrectionBatch
    projection: PickCubeStateQuaternionProjection
    work: StateQuaternionCorrectionWork


def state_quaternion_correction_view(
    batch: ReadinessCorrectionBatch, projection: PickCubeStateQuaternionProjection
) -> StateQuaternionCorrectionView:
    """Derive a detached feature view; original arrival gates still apply.

    For example, inspect ``view.source`` as raw evidence and release optimizer
    inputs only through ``view.model.training_pairs(now=original_arrival)``.
    The caller must retain the original projection metadata alongside the view.
    """
    if type(batch) not in (
        ReadinessCorrectionBatch,
        ObservedGraspCorrectionBatch,
        GoalConsistentObservedGraspCorrectionBatch,
    ):
        raise ValueError("canonical state42 corrections require an exact raw readiness batch")
    source = copy_readiness_correction_batch(batch)
    config = validate_state_quaternion_projection(projection)
    episode = state_quaternion_training_episode(source.episode, config)
    # Reuse projected transition inputs for paired labels. Projecting each label
    # separately would duplicate work and create another opportunity to diverge.
    labels = tuple(
        ReadinessCorrectionLabel(label.step, row.observation, label.target, label.stage)
        for label, row in zip(source.labels, episode.transitions, strict=True)
    )
    model = copy_readiness_correction_batch(source, episode=episode, labels=labels)
    count = len(source.labels)
    work = StateQuaternionCorrectionWork(
        observation_projections=2 * count,
        source_values_validated=2 * count * config.source_dim,
        projected_values=2 * count * config.feature_dim,
        projected_numeric_bytes=2 * count * config.feature_dim * 8,
        projected_label_observation_values=count * config.feature_dim,
        projected_label_target_values=count * 4,
        projected_label_observation_numeric_bytes=count * config.feature_dim * 8,
        projected_label_target_numeric_bytes=count * 4 * 8,
        quaternion_unit_checks=2 * count * len(config.quaternion_starts),
        quaternion_components_checked=(
            2 * count * len(config.quaternion_starts) * config.quaternion_width
        ),
    )
    return StateQuaternionCorrectionView(source, model, config, work)
