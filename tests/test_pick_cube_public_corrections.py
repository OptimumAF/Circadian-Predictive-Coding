"""Pure correction projection checks: three base observations, no SGD or native IO.

Corruptions reuse one two-transition fixture shape. The suite has at most
19 distinct synthetic observations, including deliberately invalid evidence.
"""

from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.pick_cube_camera_context import camera_context_recipe, pick_cube_camera_context
from src.core.pick_cube_corrections import PickCubeCorrectionBatch, PickCubeCorrectionLabel
from src.core import pick_cube_public_corrections as public
from src.core.pick_cube_public_vectors import public_vector_projection
from src.core.pick_cube_rgb_features import pick_cube_rgb_recipe

ACTION_UNITS = (
    "normalized_root_dx",
    "normalized_root_dy",
    "normalized_root_dz",
    "normalized_gripper",
)
STUDENT_ACTION = (0.125, -0.25, 0.5, -0.5)
TARGET = (0.5, 0.0, 1.0, 1.0)


def raw_batch(seed=17, *, terminated=False):
    observations = tuple(
        tuple(float(index + coordinate) for coordinate in range(21)) + (0.25,) * 768
        for index in range(3)
    )
    episode = ContinuousEpisode(
        f"pick-cube-smoke-{seed}",
        "pick-cube" if seed == 17 else "pick-cube-train-19",
        "train",
        ContinuousActionSpec(789, ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4),
        tuple(
            ContinuousTransition(
                index,
                observations[index],
                STUDENT_ACTION,
                float(index == 1),
                observations[index + 1],
                index == 1 and terminated,
                index == 1 and not terminated,
            )
            for index in range(2)
        ),
    )
    labels = tuple(
        PickCubeCorrectionLabel(row.step, row.observation, TARGET, "align")
        for row in episode.transitions
    )
    return PickCubeCorrectionBatch(seed, episode, labels)


def view(batch, *, projection=None, recipe=None):
    return public.public_vector_correction_view(
        batch,
        public_vector_projection() if projection is None else projection,
        source_recipe=pick_cube_rgb_recipe() if recipe is None else recipe,
    )


@pytest.mark.parametrize("seed", [17, 19])
@pytest.mark.parametrize("terminated", [False, True])
def test_should_keep_raw_controls_and_exact_pairing_in_owned_model_view(seed, terminated):
    batch = raw_batch(seed, terminated=terminated)
    before = asdict(batch)
    result = view(batch)
    assert result.source == batch and result.source is not batch
    assert result.source.episode is not batch.episode
    assert result.model.train_seed == batch.train_seed
    assert result.model.episode.spec.observation_dim == 21
    assert result.source.episode.spec.observation_dim == 789
    assert (
        result.model.episode.episode_id,
        result.model.episode.context,
        result.model.episode.role,
    ) == (batch.episode.episode_id, batch.episode.context, "train")
    assert replace(result.model.episode.spec, observation_dim=789) == batch.episode.spec
    for raw, model, raw_label, model_label in zip(
        batch.episode.transitions,
        result.model.episode.transitions,
        batch.labels,
        result.model.labels,
        strict=True,
    ):
        assert model.observation == raw.observation[:21]
        assert model.next_observation == raw.next_observation[:21]
        assert (model.step, model.action, model.reward, model.terminated, model.truncated) == (
            raw.step,
            raw.action,
            raw.reward,
            raw.terminated,
            raw.truncated,
        )
        assert model_label.observation == model.observation
        assert model_label.target == raw_label.target == TARGET
        assert model.action == STUDENT_ACTION and model.action != model_label.target
        assert (model_label.step, model_label.stage) == (raw_label.step, raw_label.stage)
        assert (model_label.observed_at, model_label.labels_arrived_at) == (
            raw_label.observed_at,
            raw_label.labels_arrived_at,
        )
        assert model_label is not raw_label
        assert model_label is not result.source.labels[raw.step]
    assert asdict(batch) == before
    assert asdict(result.work) == dict(
        observation_projections=4,
        source_values_validated=3156,
        projected_values=84,
        projected_numeric_bytes=672,
        projected_label_observation_values=42,
        projected_label_target_values=8,
        projected_label_observation_numeric_bytes=336,
        projected_label_target_numeric_bytes=64,
    )
    assert asdict(result)["source_recipe"] == asdict(pick_cube_rgb_recipe())


def test_should_preserve_label_arrival_gate_after_projection():
    result = view(raw_batch())
    for batch in (result.source, result.model):
        with pytest.raises(ValueError, match="arrived"):
            batch.training_pairs(now=2)
        observations, targets = batch.training_pairs(now=3)
        assert observations == tuple(row.observation for row in batch.episode.transitions)
        assert targets == (TARGET, TARGET)
        assert [label.observed_at for label in batch.labels] == [0, 2]
        assert [label.labels_arrived_at for label in batch.labels] == [1, 3]


def test_should_reject_already_projected_batch_as_raw_source():
    projected = view(raw_batch()).model
    with pytest.raises(ValueError, match="raw789D spec"):
        view(projected)


def test_should_detach_source_model_and_metadata_from_caller_mutation():
    batch, projection, recipe = raw_batch(), public_vector_projection(), pick_cube_rgb_recipe()
    result = view(batch, projection=projection, recipe=recipe)
    original_source, original_model = asdict(result.source), asdict(result.model)
    object.__setattr__(batch.labels[0], "target", (0.0, 0.0, 0.0, 1.0))
    object.__setattr__(batch.episode.transitions[0], "action", (0.0,) * 4)
    object.__setattr__(batch.episode, "context", "corrupt")
    object.__setattr__(projection, "feature_dim", 20)
    object.__setattr__(recipe, "recipe_id", "corrupt")
    assert asdict(result.source) == original_source
    assert asdict(result.model) == original_model
    assert result.projection == public_vector_projection() and result.projection is not projection
    assert result.source_recipe == pick_cube_rgb_recipe() and result.source_recipe is not recipe
    object.__setattr__(result.source.labels[0], "target", (0.0, 0.0, 0.0, 1.0))
    object.__setattr__(result.source.episode.transitions[0], "action", (0.0,) * 4)
    assert asdict(result.model) == original_model


@pytest.mark.parametrize("field", ["source", "model", "projection", "source_recipe", "work"])
def test_should_freeze_view_fields(field):
    result = view(raw_batch())
    with pytest.raises(FrozenInstanceError):
        setattr(result, field, None)
    with pytest.raises(FrozenInstanceError):
        result.work.observation_projections = 0


@pytest.mark.parametrize("damage", ["none", "mapping", "subclass"])
def test_should_require_exact_correction_batch(damage):
    batch = raw_batch()
    if damage == "none":
        batch = None
    elif damage == "mapping":
        batch = asdict(batch)
    else:

        class Derived(PickCubeCorrectionBatch):
            pass

        batch = Derived(batch.train_seed, batch.episode, batch.labels)
    with pytest.raises(ValueError, match="exact raw correction batch"):
        view(batch)


@pytest.mark.parametrize("role", ["inner_guard", "probe_train", "probe_test", "final_test"])
def test_should_refuse_non_train_before_projecting_any_evidence(monkeypatch, role):
    batch = raw_batch()
    object.__setattr__(batch.episode, "role", role)
    monkeypatch.setattr(
        public, "public_vector_training_episode", lambda *_a, **_k: pytest.fail("role leaked")
    )
    with pytest.raises(ValueError, match="TRAIN"):
        view(batch)


@pytest.mark.parametrize(
    "damage",
    [
        "context",
        "camera_context",
        "identity",
        "seed",
        "seed_type",
        "units",
        "bounds",
        "source_dimension",
        "projection_dimension",
        "projection_type",
        "recipe_type",
        "recipe_camera",
        "recipe_numeric_type",
    ],
)
def test_should_refuse_changed_source_qualification_or_projection_metadata(damage):
    batch, projection, recipe = raw_batch(), public_vector_projection(), pick_cube_rgb_recipe()
    if damage in ("context", "camera_context", "identity"):
        field, value = {
            "context": ("context", "pick-cube-train-19"),
            "camera_context": ("context", pick_cube_camera_context().context),
            "identity": ("episode_id", "pick-cube-task-teacher-17"),
        }[damage]
        object.__setattr__(batch.episode, field, value)
    elif damage in ("seed", "seed_type"):
        object.__setattr__(batch, "train_seed", 23 if damage == "seed" else True)
    elif damage in ("units", "bounds", "source_dimension"):
        field, value = {
            "units": ("action_units", ("other",) * 4),
            "bounds": ("upper_bounds", (2.0,) * 4),
            "source_dimension": ("observation_dim", 21),
        }[damage]
        object.__setattr__(batch.episode.spec, field, value)
    elif damage == "projection_dimension":
        object.__setattr__(projection, "feature_dim", 20)
    elif damage == "projection_type":
        projection = asdict(projection)
    elif damage == "recipe_type":
        recipe = asdict(recipe)
    elif damage == "recipe_camera":
        recipe = camera_context_recipe(pick_cube_camera_context())
    else:
        object.__setattr__(recipe, "pixel_divisor", 255)
    with pytest.raises(ValueError):
        view(batch, projection=projection, recipe=recipe)


@pytest.mark.parametrize("side", ["current", "terminal_next"])
@pytest.mark.parametrize(
    "damage", ["list", "short", "long", "nan_prefix", "inf_rgb", "bool_rgb", "rgb_low", "rgb_high"]
)
def test_should_validate_full_current_and_next_raw_evidence_before_dropping_rgb(side, damage):
    batch = raw_batch()
    row = batch.episode.transitions[0 if side == "current" else -1]
    field = "observation" if side == "current" else "next_observation"
    original = getattr(row, field)
    if damage == "list":
        changed = list(original)
    elif damage == "short":
        changed = original[:-1]
    elif damage == "long":
        changed = original + (0.25,)
    else:
        index, value = {
            "nan_prefix": (0, float("nan")),
            "inf_rgb": (788, float("inf")),
            "bool_rgb": (788, False),
            "rgb_low": (21, -1e-12),
            "rgb_high": (788, 1.0 + 1e-12),
        }[damage]
        changed = original[:index] + (value,) + original[index + 1 :]
    object.__setattr__(row, field, changed)
    if side == "current":
        object.__setattr__(batch.labels[0], "observation", changed)
    with pytest.raises(ValueError):
        view(batch)


@pytest.mark.parametrize(
    "damage", ["missing", "reversed", "step", "observation", "target", "stage", "label_type"]
)
def test_should_revalidate_every_raw_label_before_creating_model_pairing(damage):
    batch = raw_batch()
    if damage == "missing":
        object.__setattr__(batch, "labels", batch.labels[:1])
    elif damage == "reversed":
        object.__setattr__(batch, "labels", tuple(reversed(batch.labels)))
    elif damage == "label_type":
        object.__setattr__(batch, "labels", (asdict(batch.labels[0]), batch.labels[1]))
    else:
        field, value = {
            "step": ("step", 1),
            "observation": ("observation", batch.labels[1].observation),
            "target": ("target", (2.0, 0.0, 0.0, 1.0)),
            "stage": ("stage", "descend"),
        }[damage]
        object.__setattr__(batch.labels[0], field, value)
    with pytest.raises(ValueError):
        view(batch)


@pytest.mark.parametrize(
    "damage", ["incomplete", "early_terminal", "broken_join", "action", "reward", "too_long"]
)
def test_should_refuse_corrupt_executed_trajectory_or_incomplete_horizon(damage):
    batch = raw_batch()
    if damage == "incomplete":
        object.__setattr__(batch.episode.transitions[-1], "truncated", False)
    elif damage == "early_terminal":
        object.__setattr__(batch.episode.transitions[0], "terminated", True)
    elif damage == "broken_join":
        object.__setattr__(
            batch.episode.transitions[0], "next_observation", batch.labels[0].observation
        )
    elif damage == "action":
        object.__setattr__(batch.episode.transitions[0], "action", (2.0,) * 4)
    elif damage == "reward":
        object.__setattr__(batch.episode.transitions[0], "reward", float("nan"))
    else:
        # Reuse existing records: testing admission must not create a long fixture.
        object.__setattr__(batch.episode, "transitions", (batch.episode.transitions[0],) * 51)
        object.__setattr__(batch, "labels", (batch.labels[0],) * 51)
    with pytest.raises(ValueError):
        view(batch)
