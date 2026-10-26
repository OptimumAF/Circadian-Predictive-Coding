"""Owned public21 correction views of complete raw789 TRAIN trajectories.

Input: an exact provisional correction batch, fixed public projection and default
RGB recipe. Output: detached raw/model batches and explicit numeric payload work.
No collection, geometry, fitting, qualification, arrival release or IO occurs;
``PickCubeCorrectionBatch.training_pairs`` remains the label arrival gate.
"""

from dataclasses import dataclass

from src.core.pick_cube_corrections import PickCubeCorrectionBatch, PickCubeCorrectionLabel
from src.core.pick_cube_readiness_corrections import (
    ReadinessCorrectionBatch,
    ReadinessCorrectionLabel,
)
from src.core.pick_cube_public_vectors import (
    PickCubePublicVectorProjection,
    public_vector_training_episode,
    validate_public_vector_projection,
)
from src.core.pick_cube_rgb_features import PickCubeRGBRecipe, validate_pick_cube_rgb_recipe


@dataclass(frozen=True)
class PublicVectorCorrectionWork:
    """Logical projection and copied pairing payload; bytes count float64 values."""

    observation_projections: int
    source_values_validated: int
    projected_values: int
    projected_numeric_bytes: int
    projected_label_observation_values: int
    projected_label_target_values: int
    projected_label_observation_numeric_bytes: int
    projected_label_target_numeric_bytes: int


@dataclass(frozen=True)
class PublicVectorCorrectionView:
    source: PickCubeCorrectionBatch | ReadinessCorrectionBatch
    model: PickCubeCorrectionBatch | ReadinessCorrectionBatch
    projection: PickCubePublicVectorProjection
    source_recipe: PickCubeRGBRecipe
    work: PublicVectorCorrectionWork


def public_vector_correction_view(
    batch: PickCubeCorrectionBatch | ReadinessCorrectionBatch,
    projection: PickCubePublicVectorProjection,
    *,
    source_recipe: PickCubeRGBRecipe,
) -> PublicVectorCorrectionView:
    """Revalidate raw evidence and derive its independent public21 pairing view.

    Example: ``view = public_vector_correction_view(batch, projection,
    source_recipe=recipe)``; release only through ``view.model.training_pairs(now=now)``.
    """
    if type(batch) not in (PickCubeCorrectionBatch, ReadinessCorrectionBatch):
        raise ValueError("public-vector21 corrections require an exact raw correction batch")
    source: PickCubeCorrectionBatch | ReadinessCorrectionBatch
    if isinstance(batch, ReadinessCorrectionBatch):
        source = ReadinessCorrectionBatch(batch.train_seed, batch.episode, batch.labels)
    else:
        source = PickCubeCorrectionBatch(batch.train_seed, batch.episode, batch.labels)
    config = validate_public_vector_projection(projection)
    recipe = validate_pick_cube_rgb_recipe(source_recipe)
    episode = public_vector_training_episode(source.episode, config, source_recipe=recipe)
    # Why: each label uses the matching projected transition, so its prediction
    # pairing cannot drift and its observation needs no additional projection.
    model: PickCubeCorrectionBatch | ReadinessCorrectionBatch
    if isinstance(source, ReadinessCorrectionBatch):
        readiness_labels = tuple(
            ReadinessCorrectionLabel(label.step, row.observation, label.target, label.stage)
            for label, row in zip(source.labels, episode.transitions, strict=True)
        )
        model = ReadinessCorrectionBatch(source.train_seed, episode, readiness_labels)
    else:
        labels = tuple(
            PickCubeCorrectionLabel(label.step, row.observation, label.target, label.stage)
            for label, row in zip(source.labels, episode.transitions, strict=True)
        )
        model = PickCubeCorrectionBatch(source.train_seed, episode, labels)
    count = len(source.labels)
    work = PublicVectorCorrectionWork(
        observation_projections=2 * count,
        source_values_validated=2 * count * config.source_dim,
        projected_values=2 * count * config.feature_dim,
        projected_numeric_bytes=2 * count * config.feature_dim * 8,
        projected_label_observation_values=count * config.feature_dim,
        projected_label_target_values=count * 4,
        projected_label_observation_numeric_bytes=count * config.feature_dim * 8,
        projected_label_target_numeric_bytes=count * 4 * 8,
    )
    return PublicVectorCorrectionView(source, model, config, recipe, work)
