"""Raw readiness corrections derive history from actual commands, not targets.

Only pure native records with three42D rows and derived47D views are used.
Model constructors, arrays, RNG, fits, predictions, gradients, optimizer updates,
simulator calls and IO are zero. Arrival checks retain the original0/2/4 and1/3/5
ticks; constructing a view itself grants no label release authority.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
from typing import Any

import pytest

from src.core import pick_cube_action_history_corrections as corrections
from src.core.pick_cube_action_history import (
    PickCubeActionHistoryRecipe,
    action_history_training_episode,
)
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.pick_cube_corrections import PickCubeCorrectionBatch, PickCubeCorrectionLabel
from src.core.pick_cube_readiness_corrections import (
    ReadinessCorrectionBatch,
    ReadinessCorrectionLabel,
)


UNITS = ("normalized_root_dx", "normalized_root_dy", "normalized_root_dz", "normalized_gripper")
ACTUAL = ((-0.5, 0.0, 0.0, 1.0), (0.25, 0.0, 0.0, -1.0), (0.0, -0.5, 0.0, -1.0))
TARGETS = ((0.0, 0.0, 0.0, -1.0), (0.5, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, -1.0))
STAGES = ("grasp", "align", "settle")


def raw(index):
    values = [0.0] * 42
    values[0], values[40] = float(index), float(index + 10)
    values[22], values[33] = (-1.0 if index % 2 else 1.0), (1.0 if index % 2 else -1.0)
    return tuple(values)


def batch(seed=17):
    episode = ContinuousEpisode(
        f"pick-cube-smoke-{seed}",
        "pick-cube" if seed == 17 else "pick-cube-train-19",
        "train",
        ContinuousActionSpec(42, UNITS, (-1.0,) * 4, (1.0,) * 4),
        tuple(
            ContinuousTransition(i, raw(i), action, float(i == 2), raw(i + 1), i == 2, False)
            for i, action in enumerate(ACTUAL)
        ),
    )
    return ReadinessCorrectionBatch(
        seed,
        episode,
        tuple(
            ReadinessCorrectionLabel(i, raw(i), target, stage)
            for i, (target, stage) in enumerate(zip(TARGETS, STAGES, strict=True))
        ),
    )


def expected_work(canonical=False):
    return dict(
        observation_projections=6,
        source_values_validated=252,
        projected_values=282,
        projected_numeric_bytes=2256,
        projected_label_observation_values=141,
        projected_label_target_values=12,
        projected_label_observation_numeric_bytes=1128,
        projected_label_target_numeric_bytes=96,
        history_advances=3,
        quaternion_unit_checks=12 if canonical else 0,
        quaternion_components_checked=48 if canonical else 0,
    )


@pytest.fixture(scope="module")
def selected():
    return corrections.action_history_correction_view(batch(), PickCubeActionHistoryRecipe())


def test_should_derive_history_only_from_original_executed_actions_even_when_teacher_opposes(
    selected,
):
    view = selected
    assert type(view.source) is type(view.model) is ReadinessCorrectionBatch
    assert (
        view.source.episode.spec.observation_dim == 42
        and view.model.episode.spec.observation_dim == 47
    )
    before = ((0.0,) * 5, (*ACTUAL[0], 0.0), (*ACTUAL[1], 1 / 6))
    after = ((*ACTUAL[0], 0.0), (*ACTUAL[1], 1 / 6), (*ACTUAL[2], 2 / 6))
    for i, row in enumerate(view.model.episode.transitions):
        assert row.observation[:42] == raw(i) and row.next_observation[:42] == raw(i + 1)
        assert row.observation[42:] == before[i] and row.next_observation[42:] == after[i]
        assert row.action == ACTUAL[i]
        assert view.model.labels[i].target == TARGETS[i] and view.model.labels[i].stage == STAGES[i]
        assert view.model.labels[i].observation == row.observation
        if i:
            assert view.model.episode.transitions[i - 1].next_observation == row.observation
    # Target0 closes, actual0 opens: the following count must be zero.
    assert view.model.episode.transitions[1].observation[-1] == 0.0
    assert view.model.episode.transitions[-1].next_observation[-1] == 2 / 6
    assert asdict(view.work) == expected_work()


def test_should_preserve_original_arrival_gate_targets_stages_and_all_terminal_metadata(selected):
    view = selected
    for now in (0, 1, 3, 4):
        with pytest.raises(ValueError):
            view.model.training_pairs(now=now)
    features, targets = view.model.training_pairs(now=5)
    assert features == tuple(row.observation for row in view.model.episode.transitions)
    assert targets == TARGETS
    assert tuple(label.observed_at for label in view.model.labels) == (0, 2, 4)
    assert tuple(label.labels_arrived_at for label in view.model.labels) == (1, 3, 5)
    assert (
        view.model.train_seed,
        view.model.episode.episode_id,
        view.model.episode.context,
        view.model.episode.role,
    ) == (17, "pick-cube-smoke-17", "pick-cube", "train")
    for original, model in zip(
        view.source.episode.transitions, view.model.episode.transitions, strict=True
    ):
        assert (
            original.step,
            original.action,
            original.reward,
            original.terminated,
            original.truncated,
        ) == (model.step, model.action, model.reward, model.terminated, model.truncated)


def test_should_not_use_changed_teacher_targets_or_future_execution_for_current_features():
    original = batch()
    changed_labels = (
        replace(original.labels[0], target=(0.25, 0.0, 0.0, 1.0), stage="align"),
        replace(original.labels[1], target=(-0.5, 0.0, 0.0, 1.0)),
        original.labels[2],
    )
    changed = replace(original, labels=changed_labels)
    recipe = PickCubeActionHistoryRecipe()
    first = corrections.action_history_correction_view(original, recipe)
    second = corrections.action_history_correction_view(changed, recipe)
    assert first.model.episode == second.model.episode
    assert first.model.labels[0].target != second.model.labels[0].target
    last = replace(original.episode.transitions[-1], action=(0.0, 0.0, 0.0, 1.0))
    future = replace(
        original,
        episode=replace(original.episode, transitions=(*original.episode.transitions[:-1], last)),
    )
    third = corrections.action_history_correction_view(future, recipe)
    assert tuple(row.observation for row in first.model.episode.transitions) == tuple(
        row.observation for row in third.model.episode.transitions
    )
    assert third.model.episode.transitions[-1].next_observation[42:] == (0.0, 0.0, 0.0, 1.0, 0.0)


@pytest.mark.parametrize("mode", ["executed", "zero"])
@pytest.mark.parametrize("canonical", [False, True])
def test_should_preserve_equal_width_zero_and_canonical_views_with_honest_work(mode, canonical):
    source = batch()
    recipe = PickCubeActionHistoryRecipe(mode, canonical)
    view = corrections.action_history_correction_view(source, recipe)
    assert asdict(view.work) == expected_work(canonical)
    assert view.model.episode.spec.observation_dim == 47
    assert tuple(row.action for row in view.model.episode.transitions) == ACTUAL
    if mode == "zero":
        assert all(
            row.observation[42:] == row.next_observation[42:] == (0.0,) * 5
            for row in view.model.episode.transitions
        )
    else:
        assert view.model.episode.transitions[2].observation[42:] == (*ACTUAL[1], 1 / 6)
    for row in view.model.episode.transitions:
        if canonical:
            assert row.observation[22] == row.observation[33] == 1.0
        else:
            assert row.observation[:42] == source.episode.transitions[row.step].observation
    assert source.episode.transitions[1].observation[22] == -1.0


def test_should_reset_history_for_new_train_episode_and_keep_source_model_recipe_detached():
    source, recipe = batch(19), PickCubeActionHistoryRecipe()
    original = deepcopy(source)
    view = corrections.action_history_correction_view(source, recipe)
    assert source == original and view.source == original and view.source is not source
    assert view.model.episode.transitions[0].observation[42:] == (0.0,) * 5
    assert view.recipe == recipe and view.recipe is not recipe
    assert view.model.labels[0] is not source.labels[0]
    with pytest.raises(FrozenInstanceError):
        setattr(view, "recipe", recipe)
    with pytest.raises(FrozenInstanceError):
        setattr(view.work, "history_advances", 0)
    object.__setattr__(source.labels[0], "target", (0.0, 0.0, 0.0, 1.0))
    object.__setattr__(source.episode.transitions[0], "action", (0.0, 0.0, 0.0, -1.0))
    object.__setattr__(recipe, "mode", "zero")
    assert view.source == original and view.model.labels[0].target == TARGETS[0]
    assert view.model.episode.transitions[0].action == ACTUAL[0] and view.recipe.mode == "executed"


def test_should_return_bounded_detached_metadata_without_replacing_raw_or_model_specs(selected):
    metadata = corrections.action_history_correction_metadata(selected.source, selected.recipe)
    assert metadata == dict(
        projection=asdict(selected.recipe),
        source_episode_id="pick-cube-smoke-17",
        source_spec=asdict(selected.source.episode.spec),
        model_spec=asdict(selected.model.episode.spec),
        work=expected_work(),
        source_encoding="official_state42_raw",
        model_encoding=selected.recipe.definition,
    )
    metadata["projection"]["mode"] = "zero"
    metadata["work"]["history_advances"] = 0
    metadata["model_spec"]["observation_dim"] = 42
    fresh = corrections.action_history_correction_metadata(selected.source, selected.recipe)
    assert (
        fresh["projection"]["mode"] == "executed" and fresh["model_spec"]["observation_dim"] == 47
    )
    assert fresh["work"] == expected_work()


@pytest.mark.parametrize("fault", ["mapping", "fixed_family", "subclass", "wrong47"])
def test_should_refuse_nonreadiness_or_already_projected_sources(fault):
    source: Any = batch()
    recipe = PickCubeActionHistoryRecipe()
    if fault == "mapping":
        source = asdict(source)
    elif fault == "fixed_family":
        labels = tuple(
            PickCubeCorrectionLabel(i, raw(i), (0.5, 0.0, 0.0, 1.0), "align") for i in range(3)
        )
        source = PickCubeCorrectionBatch(17, source.episode, labels)
    elif fault == "subclass":

        class ForeignBatch(ReadinessCorrectionBatch):
            pass

        source = ForeignBatch(source.train_seed, source.episode, source.labels)
    else:
        model_episode = action_history_training_episode(source.episode, recipe)
        source = ReadinessCorrectionBatch(
            17,
            model_episode,
            tuple(
                replace(label, observation=row.observation)
                for label, row in zip(source.labels, model_episode.transitions, strict=True)
            ),
        )
    with pytest.raises(ValueError):
        corrections.action_history_correction_view(source, recipe)


@pytest.mark.parametrize(
    "fault",
    [
        "role",
        "context",
        "seed",
        "label_observation",
        "label_target",
        "label_clock",
        "actual32",
        "seam",
        "terminal",
        "quaternion",
    ],
)
def test_should_revalidate_complete_raw_evidence_before_releasing_any_view(fault):
    source = batch()
    recipe = PickCubeActionHistoryRecipe()
    if fault == "role":
        object.__setattr__(source.episode, "role", "final_test")
    elif fault == "context":
        object.__setattr__(source.episode, "context", "pick-cube-train-19")
    elif fault == "seed":
        object.__setattr__(source, "train_seed", True)
    elif fault == "label_observation":
        object.__setattr__(source.labels[0], "observation", raw(1))
    elif fault == "label_target":
        object.__setattr__(source.labels[0], "target", (0.25, 0.0, 0.0, -1.0))
    elif fault == "label_clock":
        object.__setattr__(source.labels[1], "step", 0)
    elif fault == "actual32":
        object.__setattr__(source.episode.transitions[0], "action", (0.1, 0.0, 0.0, 1.0))
    elif fault == "seam":
        object.__setattr__(source.episode.transitions[0], "next_observation", raw(2))
    elif fault == "terminal":
        object.__setattr__(source.episode.transitions[-1], "terminated", False)
    else:
        bad = (0.0,) * 42
        object.__setattr__(source.episode.transitions[0], "observation", bad)
        object.__setattr__(source.labels[0], "observation", bad)
        recipe = replace(recipe, canonical_quaternions=True)
    with pytest.raises(ValueError):
        corrections.action_history_correction_view(source, recipe)


@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", "teacher"),
        ("canonical_quaternions", 1),
        ("feature_dim", 47.0),
        ("closing_limit", True),
    ],
)
def test_should_refuse_recipe_alias_or_corruption_before_view_and_metadata(field, value):
    recipe = PickCubeActionHistoryRecipe()
    object.__setattr__(recipe, field, value)
    with pytest.raises(ValueError):
        corrections.action_history_correction_view(batch(), recipe)
    with pytest.raises(ValueError):
        corrections.action_history_correction_metadata(batch(), recipe)
