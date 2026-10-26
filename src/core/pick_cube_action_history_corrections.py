"""Causal model views of unchanged raw readiness correction evidence.

History follows successful episode actions, never provisional label targets.
Outputs own raw/model batches and bounded payload counts. No label release,
normalization fit, learner, geometry, simulator, clock advancement or IO occurs.
"""

from dataclasses import asdict, dataclass

from src.core.pick_cube_action_history import (
    ActionHistoryRecipe,
    _native_action,
    action_history_spec,
    action_history_training_episode,
    validate_action_history_recipe,
)
from src.core.pick_cube_readiness_corrections import (
    ReadinessCorrectionBatch,
    ObservedGraspCorrectionBatch,
    GoalConsistentObservedGraspCorrectionBatch,
    copy_readiness_correction_batch,
    ReadinessCorrectionLabel,
)


@dataclass(frozen=True)
class ActionHistoryCorrectionWork:
    observation_projections: int
    source_values_validated: int
    projected_values: int
    projected_numeric_bytes: int
    projected_label_observation_values: int
    projected_label_target_values: int
    projected_label_observation_numeric_bytes: int
    projected_label_target_numeric_bytes: int
    history_advances: int
    quaternion_unit_checks: int
    quaternion_components_checked: int


@dataclass(frozen=True)
class ActionHistoryCorrectionView:
    source: ReadinessCorrectionBatch
    model: ReadinessCorrectionBatch
    recipe: ActionHistoryRecipe
    work: ActionHistoryCorrectionWork


def _source(batch, recipe):
    if type(batch) not in (
        ReadinessCorrectionBatch,
        ObservedGraspCorrectionBatch,
        GoalConsistentObservedGraspCorrectionBatch,
    ):
        raise ValueError("action history corrections require an exact raw readiness batch")
    owned = copy_readiness_correction_batch(batch)
    config = validate_action_history_recipe(recipe)
    action_history_spec(owned.episode.spec, config)
    for row in owned.episode.transitions:
        _native_action(row.action)
    return owned, config


def _work(count: int, config: ActionHistoryRecipe) -> ActionHistoryCorrectionWork:
    return ActionHistoryCorrectionWork(
        observation_projections=2 * count,
        source_values_validated=2 * count * config.source_dim,
        projected_values=2 * count * config.feature_dim,
        projected_numeric_bytes=2 * count * config.feature_dim * 8,
        projected_label_observation_values=count * config.feature_dim,
        projected_label_target_values=count * 4,
        projected_label_observation_numeric_bytes=count * config.feature_dim * 8,
        projected_label_target_numeric_bytes=count * 4 * 8,
        history_advances=count,
        quaternion_unit_checks=4 * count if config.canonical_quaternions else 0,
        quaternion_components_checked=16 * count if config.canonical_quaternions else 0,
    )


def action_history_correction_view(
    batch: ReadinessCorrectionBatch, recipe: ActionHistoryRecipe
) -> ActionHistoryCorrectionView:
    """Own raw labels and derive features from original actually executed actions.

    The original arrival gate still applies to ``view.model.training_pairs(now=...)``.
    In particular, a teacher target replacing a student's command must not become
    the preceding command in the next observation's history.
    """
    source, config = _source(batch, recipe)
    episode = action_history_training_episode(source.episode, config)
    labels = tuple(
        ReadinessCorrectionLabel(label.step, row.observation, label.target, label.stage)
        for label, row in zip(source.labels, episode.transitions, strict=True)
    )
    model = copy_readiness_correction_batch(source, episode=episode, labels=labels)
    return ActionHistoryCorrectionView(source, model, config, _work(len(source.labels), config))


def action_history_correction_metadata(
    rawbatch: ReadinessCorrectionBatch, recipe: ActionHistoryRecipe
) -> dict:
    """Describe the owned raw/model binding without performing a second projection.

    This is metadata, not an alternative admission/release path. The actual view
    validates observation projection (including optional quaternion unit checks).
    """
    source, config = _source(rawbatch, recipe)
    return dict(
        projection=asdict(config),
        source_episode_id=source.episode.episode_id,
        source_spec=asdict(source.episode.spec),
        model_spec=asdict(action_history_spec(source.episode.spec, config)),
        work=asdict(_work(len(source.labels), config)),
        source_encoding="official_state42_raw",
        model_encoding=config.definition,
    )
