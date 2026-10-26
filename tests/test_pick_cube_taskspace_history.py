"""Official-state world displacements plus causal execution history, pure only.

Literal complete episodes have three rows/raw42/projected12. There are no model
constructors, arrays, RNG, fits, predictions, gradients, updates, native calls or
IO. Raw quaternion coordinates are finite validation inputs, never task features.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from src.core import pick_cube_action_history as history
from src.core import pick_cube_taskspace_history as taskspace
from src.core.pick_cube_action_history_corrections import (
    action_history_correction_metadata,
    action_history_correction_view,
)
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.pick_cube_readiness_corrections import (
    ReadinessCorrectionBatch,
    ReadinessCorrectionLabel,
)


UNITS = ("normalized_root_dx", "normalized_root_dy", "normalized_root_dz", "normalized_gripper")
ACTUAL = ((0.5, 0.0, 0.0, 1.0), (0.0, -0.25, 0.0, -1.0), (0.0, 0.0, 0.5, -1.0))


def raw(index=0):
    values = [0.0] * 42
    values[0], values[10], values[18] = 999.0 + index, -123.0, float(index % 2)
    values[36:42] = (0.05, -0.025, 0.0125, -0.1, 0.2, -0.4)
    return tuple(values)


def source_episode(seed=17, actions=ACTUAL):
    return ContinuousEpisode(
        f"pick-cube-smoke-{seed}",
        "pick-cube" if seed == 17 else "pick-cube-train-19",
        "train",
        ContinuousActionSpec(42, UNITS, (-1.0,) * 4, (1.0,) * 4),
        tuple(
            ContinuousTransition(i, raw(i), action, float(i == 2), raw(i + 1), i == 2, False)
            for i, action in enumerate(actions)
        ),
    )


def batch():
    episode = source_episode()
    targets = ((0.0, 0.0, 0.0, -1.0), (0.5, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, -1.0))
    stages = ("grasp", "align", "settle")
    return ReadinessCorrectionBatch(
        17,
        episode,
        tuple(
            ReadinessCorrectionLabel(i, row.observation, target, stage)
            for i, (row, target, stage) in enumerate(
                zip(episode.transitions, targets, stages, strict=True)
            )
        ),
    )


def test_should_own_explicit_fixed_world_frame_recipe_and_keep_legacy_shape_distinct():
    recipe = taskspace.PickCubeTaskSpaceHistoryRecipe()
    owned = history.validate_action_history_recipe(recipe)
    assert (
        type(owned) is taskspace.PickCubeTaskSpaceHistoryRecipe
        and owned == recipe
        and owned is not recipe
    )
    assert history.is_taskspace_recipe(owned) is True
    assert history.is_taskspace_recipe(history.PickCubeActionHistoryRecipe()) is False
    assert recipe.source_dim == 42 and recipe.feature_dim == 12 and recipe.closing_limit == 6
    assert type(recipe.displacement_scale) is float and recipe.displacement_scale == 0.1
    assert recipe.displacement_frame == "world" and recipe.setup == "panda_table_pd_ee_delta_pos"
    assert history.action_history_spec(source_episode().spec, owned).observation_dim == 12
    assert (
        history.action_history_spec(
            source_episode().spec, history.PickCubeActionHistoryRecipe()
        ).observation_dim
        == 47
    )
    with pytest.raises(FrozenInstanceError):
        setattr(recipe, "mode", "zero")
    object.__setattr__(recipe, "feature_dim", 13)
    assert owned.feature_dim == 12


@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", "teacher"),
        ("canonical_quaternions", True),
        ("definition", "other"),
        ("source_dim", 42.0),
        ("feature_dim", 12.0),
        ("closing_limit", True),
        ("displacement_scale", 1),
        ("displacement_scale", 0.2),
        ("displacement_frame", "tool"),
        ("setup", "other"),
    ],
)
def test_should_refuse_changed_or_aliased_taskspace_recipe(field, value):
    recipe = taskspace.PickCubeTaskSpaceHistoryRecipe()
    object.__setattr__(recipe, field, value)
    with pytest.raises(ValueError):
        history.validate_action_history_recipe(recipe)


def test_should_read_flag_and_six_world_displacements_in_metres_without_rotation_or_clipping():
    recipe = taskspace.PickCubeTaskSpaceHistoryRecipe()
    state = history.ActionHistoryState((0.5, 0.0, 0.0, -1.0), 2)
    source = raw(1)
    encoded = history.action_history_observation(source, state, recipe)
    assert encoded == (1.0, 0.5, -0.25, 0.125, -1.0, 2.0, -4.0, 0.5, 0.0, 0.0, -1.0, 2 / 6)
    assert (
        taskspace.taskspace_observation(source, recipe)
        == taskspace.taskspace_observation(source)
        == encoded[:7]
    )
    assert encoded[5:7] == (2.0, -4.0)  # Physical displacements are not action clipping.
    flipped = list(source)
    flipped[22:26], flipped[32:36] = [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]
    assert history.action_history_observation(tuple(flipped), state, recipe) == encoded
    assert source[22:26] == source[32:36] == (0.0,) * 4
    assert source == raw(1)


def test_should_publish_only_owned_fixed_recipe_metadata_without_inferred_phase_or_normalizer():
    recipe = taskspace.PickCubeTaskSpaceHistoryRecipe()
    metadata = taskspace.taskspace_metadata(recipe)
    assert metadata == asdict(recipe)
    assert set(metadata) == {
        "mode",
        "canonical_quaternions",
        "definition",
        "source_dim",
        "feature_dim",
        "closing_limit",
        "displacement_scale",
        "displacement_frame",
        "setup",
    }
    assert metadata["definition"] == "pick_cube_official_state42_taskspace7_executed_history5_v1"
    metadata["displacement_scale"] = 1.0
    assert taskspace.taskspace_metadata(recipe)["displacement_scale"] == 0.1


@pytest.mark.parametrize(
    "fault",
    [
        "short",
        "list",
        "dropped_bool",
        "dropped_nan",
        "flag",
        "flag_bool",
        "selected_inf",
        "division_overflow",
    ],
)
def test_should_validate_all_raw_coordinates_and_finite_physical_scaling(fault):
    source = raw()
    if fault == "short":
        source = source[:-1]
    elif fault == "list":
        source = list(source)
    else:
        index, value = {
            "dropped_bool": (0, True),
            "dropped_nan": (25, float("nan")),
            "flag": (18, 0.5),
            "flag_bool": (18, True),
            "selected_inf": (40, float("inf")),
            "division_overflow": (36, 1e308),
        }[fault]
        source = (*source[:index], value, *source[index + 1 :])
    with pytest.raises(ValueError):
        history.action_history_observation(
            source, history.ActionHistoryState(), taskspace.PickCubeTaskSpaceHistoryRecipe()
        )


@pytest.fixture(scope="module")
def encoded():
    source, recipe = source_episode(), taskspace.PickCubeTaskSpaceHistoryRecipe()
    return source, recipe, history.action_history_training_episode(source, recipe)


def test_should_preserve_causal_before_terminal_next_seams_and_full_original_evidence(encoded):
    source, recipe, model = encoded
    assert model.spec.observation_dim == 12 and model.spec.action_units == UNITS
    before = ((0.0,) * 5, (*ACTUAL[0], 0.0), (*ACTUAL[1], 1 / 6))
    after = ((*ACTUAL[0], 0.0), (*ACTUAL[1], 1 / 6), (*ACTUAL[2], 2 / 6))
    for i, (original, projected) in enumerate(
        zip(source.transitions, model.transitions, strict=True)
    ):
        assert projected.observation[:7] == (float(i % 2), 0.5, -0.25, 0.125, -1.0, 2.0, -4.0)
        assert projected.observation[7:] == before[i] and projected.next_observation[7:] == after[i]
        assert (projected.action, projected.reward, projected.terminated, projected.truncated) == (
            original.action,
            original.reward,
            original.terminated,
            original.truncated,
        )
        if i:
            assert model.transitions[i - 1].next_observation == projected.observation
    assert (model.episode_id, model.context, model.role) == (
        source.episode_id,
        source.context,
        source.role,
    )
    assert history.validate_action_history_training_episode(model, recipe) == model
    assert source == source_episode()


def test_should_not_read_current_or_future_execution_and_reset_per_train_episode(encoded):
    _, recipe, model = encoded
    future = source_episode(19, actions=(*ACTUAL[:2], (0.0, 0.0, 0.0, 1.0)))
    other = history.action_history_training_episode(future, recipe)
    assert tuple(row.observation for row in other.transitions) == tuple(
        row.observation for row in model.transitions
    )
    assert other.transitions[0].observation[7:] == (0.0,) * 5
    assert other.transitions[-1].next_observation[7:] == (0.0, 0.0, 0.0, 1.0, 0.0)


def test_should_keep_zero_history_same_twelve_coordinates_and_same_physical_prefix(encoded):
    source, recipe, model = encoded
    zero_recipe = replace(recipe, mode="zero")
    zero = history.action_history_training_episode(source, zero_recipe)
    assert zero.spec == model.spec
    for control, executed in zip(zero.transitions, model.transitions, strict=True):
        assert control.observation[:7] == executed.observation[:7]
        assert control.next_observation[:7] == executed.next_observation[:7]
        assert control.observation[7:] == control.next_observation[7:] == (0.0,) * 5
    assert history.validate_action_history_training_episode(zero, zero_recipe) == zero


@pytest.mark.parametrize(
    "fault", ["flag", "count", "previous_action", "native32", "nonfinite", "seam"]
)
def test_should_revalidate_encoded_tails_physical_prefix_and_actual_native_actions(encoded, fault):
    _, recipe, model = encoded
    original = deepcopy(model)
    damaged = deepcopy(model)
    if fault == "flag":
        row = damaged.transitions[0]
        object.__setattr__(row, "observation", (0.5, *row.observation[1:]))
    elif fault == "count":
        row = damaged.transitions[-1]
        object.__setattr__(row, "next_observation", (*row.next_observation[:-1], 1.0))
    elif fault == "previous_action":
        row = damaged.transitions[0]
        object.__setattr__(row, "observation", (*row.observation[:7], *ACTUAL[0], 0.0))
    elif fault == "native32":
        object.__setattr__(damaged.transitions[0], "action", (0.1, 0.0, 0.0, 1.0))
    elif fault == "nonfinite":
        row = damaged.transitions[0]
        object.__setattr__(
            row, "observation", (row.observation[0], float("inf"), *row.observation[2:])
        )
    else:
        row = damaged.transitions[0]
        object.__setattr__(
            row, "next_observation", (row.next_observation[0], 99.0, *row.next_observation[2:])
        )
    with pytest.raises(ValueError):
        history.validate_action_history_training_episode(damaged, recipe)
    assert model == original


@pytest.mark.parametrize("role", ["final_test", "probe_train", "probe_test", "inner_guard"])
def test_should_refuse_nontrain_projection_and_encoded_release(encoded, role):
    source, recipe, model = encoded
    with pytest.raises(ValueError):
        history.action_history_training_episode(replace(source, role=role), recipe)
    with pytest.raises(ValueError):
        history.validate_action_history_training_episode(replace(model, role=role), recipe)


def test_should_pair_raw_readiness_targets_with_causal_model_inputs_and_original_arrival():
    source = batch()
    original = deepcopy(source)
    recipe = taskspace.PickCubeTaskSpaceHistoryRecipe()
    view = action_history_correction_view(source, recipe)
    assert view.source == original and view.source is not source
    assert view.model.episode.spec.observation_dim == 12
    assert view.model.episode.transitions[1].observation[7:] == (*ACTUAL[0], 0.0)
    assert view.model.episode.transitions[2].observation[7:] == (*ACTUAL[1], 1 / 6)
    assert tuple(row.action for row in view.model.episode.transitions) == ACTUAL
    assert tuple(label.target for label in view.model.labels) == tuple(
        label.target for label in source.labels
    )
    assert tuple(label.stage for label in view.model.labels) == ("grasp", "align", "settle")
    assert tuple(label.labels_arrived_at for label in view.model.labels) == (1, 3, 5)
    with pytest.raises(ValueError):
        view.model.training_pairs(now=4)
    features, targets = view.model.training_pairs(now=5)
    assert features == tuple(row.observation for row in view.model.episode.transitions)
    assert targets == tuple(label.target for label in source.labels)
    assert asdict(view.work) == dict(
        observation_projections=6,
        source_values_validated=252,
        projected_values=72,
        projected_numeric_bytes=576,
        projected_label_observation_values=36,
        projected_label_target_values=12,
        projected_label_observation_numeric_bytes=288,
        projected_label_target_numeric_bytes=96,
        history_advances=3,
        quaternion_unit_checks=0,
        quaternion_components_checked=0,
    )
    metadata = action_history_correction_metadata(source, recipe)
    assert (
        metadata["source_spec"]["observation_dim"] == 42
        and metadata["model_spec"]["observation_dim"] == 12
    )
    assert metadata["projection"] == asdict(recipe) and metadata["work"] == asdict(view.work)
    object.__setattr__(source.labels[0], "target", (0.5, 0.0, 0.0, 1.0))
    assert view.source == original and view.model.labels[0].target == (0.0, 0.0, 0.0, -1.0)
