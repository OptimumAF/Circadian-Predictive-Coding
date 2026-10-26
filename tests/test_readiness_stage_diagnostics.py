"""Owned readiness row partitions: pure native records, no learner or IO.

Every test uses literal finite scalar observations/actions. Model initializers,
RNG, fitting, predictions, gradients, optimizer updates and native calls are zero.
The largest records are one1024-coordinate row and fifty2-coordinate rows.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest

from src.core import readiness_stage_diagnostics as stages
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.pick_cube_readiness_teacher import DEFINITION


UNITS = ("normalized_root_dx", "normalized_root_dy", "normalized_root_dz", "normalized_gripper")
ALL_STAGES = ("align", "descend", "grasp", "lift", "settle")


def command(stage):
    if stage in ("grasp", "settle"):
        return (0.0, 0.0, 0.0, -1.0)
    return (0.2, -0.1, 0.3, 1.0 if stage in ("align", "descend") else -1.0)


def episode(seed=17, labels=ALL_STAGES, dimension=2):
    offset = float(seed == 19)

    def observation(index):
        return (float(index),) + (offset,) * (dimension - 1)

    return ContinuousEpisode(
        f"pick-cube-readiness-teacher-{seed}",
        "pick-cube" if seed == 17 else "pick-cube-train-19",
        "train",
        ContinuousActionSpec(dimension, UNITS, (-1.0,) * 4, (1.0,) * 4),
        tuple(
            ContinuousTransition(
                i,
                observation(i),
                command(label),
                float(i == len(labels) - 1),
                observation(i + 1),
                False,
                i == len(labels) - 1,
            )
            for i, label in enumerate(labels)
        ),
    )


def pair():
    return (
        stages.ReadinessStageBinding(episode(), ALL_STAGES),
        stages.ReadinessStageBinding(episode(19, ("descend", "grasp")), ("descend", "grasp")),
    )


def test_should_bind_original_known_stages_and_own_all_complete_episode_fields():
    original = episode()
    bound = stages.ReadinessStageBinding(original, ALL_STAGES)
    assert bound.episode == original and bound.episode is not original
    assert bound.episode.spec is not original.spec
    assert all(
        a is not b for a, b in zip(bound.episode.transitions, original.transitions, strict=True)
    )
    assert bound.stages == ALL_STAGES and type(bound.stages) is tuple
    assert bound.episode.transitions[-1].truncated is True
    assert bound.episode.transitions[-1].terminated is False
    with pytest.raises(FrozenInstanceError):
        setattr(bound, "stages", ("settle",) * 5)
    object.__setattr__(original.transitions[-1], "next_observation", (999.0, 999.0))
    assert bound.episode.transitions[-1].next_observation == (5.0, 0.0)


def test_should_preserve_dynamic_stage_order_without_inventing_an_elapsed_schedule():
    labels = ("settle", "align", "lift", "descend", "grasp")
    source = episode(labels=labels)
    bound = stages.ReadinessStageBinding(source, labels)
    assert bound.stages == labels
    assert tuple(row.action for row in bound.episode.transitions) == tuple(
        command(label) for label in labels
    )
    # Binding is an explicit partition, not external collection qualification.
    assert not hasattr(bound, "qualified") and not hasattr(bound, "teacher_success")


@pytest.mark.parametrize("dimension,rows", [(1, 1), (1024, 1), (2, 50)])
def test_should_accept_bounded_generic_model_dimensions_and_fifty_original_rows(dimension, rows):
    labels = ("align",) * rows
    bound = stages.ReadinessStageBinding(episode(labels=labels, dimension=dimension), labels)
    assert (
        len(bound.episode.transitions) == rows and bound.episode.spec.observation_dim == dimension
    )
    assert bound.stages == labels


class StringAlias(str):
    pass


class TupleAlias(tuple):
    pass


@pytest.mark.parametrize(
    "labels",
    [
        list(ALL_STAGES),
        TupleAlias(ALL_STAGES),
        (),
        ALL_STAGES[:-1],
        ALL_STAGES + ("align",),
        ("ALIGN", *ALL_STAGES[1:]),
        (" align", *ALL_STAGES[1:]),
        (True, *ALL_STAGES[1:]),
        (StringAlias("align"), *ALL_STAGES[1:]),
    ],
)
def test_should_refuse_unknown_aliased_or_unmatched_stage_records(labels):
    with pytest.raises(ValueError):
        stages.ReadinessStageBinding(episode(), labels)


@pytest.mark.parametrize(
    "stage,bad_action",
    [
        ("align", (0.2, -0.1, 0.3, -1.0)),
        ("descend", (0.2, -0.1, 0.3, 0.9)),
        ("lift", (0.2, -0.1, 0.3, 1.0)),
        ("grasp", (0.0, 0.0, 0.0, 1.0)),
        ("settle", (0.0, 0.0, 0.0, 1.0)),
        ("grasp", (0.01, 0.0, 0.0, -1.0)),
        ("settle", (0.0, 0.0, -0.01, -1.0)),
    ],
)
def test_should_refuse_gripper_or_hold_commands_that_disagree_with_original_stage(
    stage, bad_action
):
    original = episode(labels=(stage,))
    changed = replace(original, transitions=(replace(original.transitions[0], action=bad_action),))
    with pytest.raises(ValueError):
        stages.ReadinessStageBinding(changed, (stage,))


@pytest.mark.parametrize("role", ["inner_guard", "probe_train", "probe_test", "final_test"])
def test_should_refuse_heldout_or_probe_episode_roles(role):
    with pytest.raises(ValueError):
        stages.ReadinessStageBinding(replace(episode(), role=role), ALL_STAGES)


@pytest.mark.parametrize(
    "field,value",
    [
        ("episode_id", "pick-cube-task-teacher-17"),
        ("episode_id", "pick-cube-readiness-teacher-19"),
        ("episode_id", "pick-cube-smoke-17"),
        ("episode_id", "pick-cube-readiness-teacher-31"),
        ("context", "pick-cube-train-19"),
        ("context", "pick-cube-camera-y005-train-17"),
        ("context", "pick-cube-mass-x2-train-17"),
    ],
)
def test_should_refuse_relabelled_legacy_correction_or_other_context_identities(field, value):
    with pytest.raises(ValueError):
        stages.ReadinessStageBinding(replace(episode(), **{field: value}), ALL_STAGES)


@pytest.mark.parametrize(
    "fault", ["units", "bounds", "many_rows", "bool_action", "nonfinite", "seam", "last_flag"]
)
def test_should_revalidate_spec_rows_and_frozen_episode_corruption(fault):
    original = episode()
    if fault == "units":
        original = replace(original, spec=replace(original.spec, action_units=("x", *UNITS[1:])))
    elif fault == "bounds":
        original = replace(original, spec=replace(original.spec, lower_bounds=(-2.0,) * 4))
    elif fault == "many_rows":
        original = episode(labels=("align",) * 51)
    elif fault == "bool_action":
        object.__setattr__(original.transitions[0], "action", (True, -0.1, 0.3, 1.0))
    elif fault == "nonfinite":
        object.__setattr__(original.transitions[0], "observation", (float("nan"), 0.0))
    elif fault == "seam":
        object.__setattr__(original.transitions[0], "next_observation", (99.0, 0.0))
    else:
        object.__setattr__(original.transitions[-1], "truncated", False)
    labels = ("align",) * 51 if fault == "many_rows" else ALL_STAGES
    with pytest.raises(ValueError):
        stages.ReadinessStageBinding(original, labels)


def test_should_copy_pair_in_warm_new_order_and_emit_detached_tuple_metadata():
    original = pair()
    episodes = tuple(binding.episode for binding in original)
    owned = stages.copy_readiness_stage_bindings(original, episodes)
    assert owned == original and owned is not original
    assert all(
        a is not b and a.episode is not b.episode for a, b in zip(owned, original, strict=True)
    )
    metadata = stages.readiness_stage_metadata(owned)
    assert metadata == {
        "warm": dict(
            definition=DEFINITION,
            episode_id="pick-cube-readiness-teacher-17",
            context="pick-cube",
            role="train",
            rows=5,
            stages=ALL_STAGES,
        ),
        "new": dict(
            definition=DEFINITION,
            episode_id="pick-cube-readiness-teacher-19",
            context="pick-cube-train-19",
            role="train",
            rows=2,
            stages=("descend", "grasp"),
        ),
    }
    assert type(metadata["warm"]["stages"]) is tuple
    metadata["warm"]["stages"] = ("wrong",)
    metadata["new"]["rows"] = 999
    assert stages.readiness_stage_metadata(owned)["warm"]["stages"] == ALL_STAGES
    object.__setattr__(original[0], "stages", ("settle",) * 5)
    assert owned[0].stages == ALL_STAGES


@pytest.mark.parametrize("kind", ["episode", "binding"])
def test_should_refuse_record_subclasses_instead_of_implicitly_trusting_their_fields(kind):
    bindings = pair()
    if kind == "episode":

        class EpisodeAlias(ContinuousEpisode):
            pass

        original = bindings[0].episode
        alias = EpisodeAlias(
            original.episode_id,
            original.context,
            original.role,
            original.spec,
            original.transitions,
        )
        with pytest.raises(ValueError):
            stages.ReadinessStageBinding(alias, ALL_STAGES)
    else:

        class BindingAlias(stages.ReadinessStageBinding):
            pass

        binding_alias = BindingAlias(bindings[0].episode, ALL_STAGES)
        with pytest.raises(ValueError):
            stages.copy_readiness_stage_bindings(
                (binding_alias, bindings[1]), tuple(binding.episode for binding in bindings)
            )


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "tuple_alias",
        "one",
        "three",
        "duplicate",
        "reversed",
        "foreign_binding",
        "episode_list",
    ],
)
def test_should_refuse_nonexact_or_misordered_pair_before_metadata_release(fault):
    bindings = pair()
    episodes: Any = tuple(binding.episode for binding in bindings)
    if fault == "list":
        bindings = list(bindings)
    elif fault == "tuple_alias":
        bindings = TupleAlias(bindings)
    elif fault == "one":
        bindings = bindings[:1]
    elif fault == "three":
        bindings = (*bindings, bindings[0])
    elif fault == "duplicate":
        bindings, episodes = (bindings[0], bindings[0]), (episodes[0], episodes[0])
    elif fault == "reversed":
        bindings, episodes = bindings[::-1], episodes[::-1]
    elif fault == "foreign_binding":
        bindings = (dict(episode=episodes[0], stages=ALL_STAGES), bindings[1])
    else:
        episodes = list(episodes)
    with pytest.raises(ValueError):
        stages.copy_readiness_stage_bindings(bindings, episodes)


@pytest.mark.parametrize("field", ["observation", "action", "reward", "end_flags", "spec"])
def test_should_bind_all_episode_data_not_just_id_and_row_count(field):
    bindings = pair()
    source = bindings[0].episode
    if field == "observation":
        changed_rows = tuple(
            replace(
                row,
                observation=tuple(x + 0.5 for x in row.observation),
                next_observation=tuple(x + 0.5 for x in row.next_observation),
            )
            for row in source.transitions
        )
        changed = replace(source, transitions=changed_rows)
    elif field == "action":
        changed = replace(
            source,
            transitions=(
                replace(source.transitions[0], action=(0.8, -0.1, 0.3, 1.0)),
                *source.transitions[1:],
            ),
        )
    elif field == "reward":
        changed = replace(
            source,
            transitions=(replace(source.transitions[0], reward=-0.5), *source.transitions[1:]),
        )
    elif field == "end_flags":
        changed = replace(
            source,
            transitions=(
                *source.transitions[:-1],
                replace(source.transitions[-1], terminated=True, truncated=False),
            ),
        )
    else:
        changed = replace(source, spec=replace(source.spec, action_units=("different", *UNITS[1:])))
    with pytest.raises(ValueError):
        stages.copy_readiness_stage_bindings(bindings, (changed, bindings[1].episode))


@pytest.mark.parametrize("fault", ["stages", "action", "role"])
def test_should_refuse_previously_frozen_binding_corruption_on_copy_and_metadata(fault):
    bindings = deepcopy(pair())
    if fault == "stages":
        object.__setattr__(bindings[0], "stages", ("wrong",) * 5)
    elif fault == "action":
        object.__setattr__(bindings[0].episode.transitions[0], "action", (0.2, -0.1, 0.3, -1.0))
    else:
        object.__setattr__(bindings[0].episode, "role", "final_test")
    with pytest.raises(ValueError):
        stages.copy_readiness_stage_bindings(
            bindings, tuple(binding.episode for binding in bindings)
        )
    with pytest.raises(ValueError):
        stages.readiness_stage_metadata(bindings)
