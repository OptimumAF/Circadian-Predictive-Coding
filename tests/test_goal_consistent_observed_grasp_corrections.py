"""Versioned Euclidean readiness labels through one private mixed BP update.

One D12/H2/A4 initializer32 normal values/P38; warm1+private8=9 SGD18rows,
zero fits/replay.9 backwards342 gradient values2736B;13 forwards22rows and
13 normalizations22rows264coords include4 fake frozen predictions. Derived
dense MAC1424. Two scenes4controls2geometry queries, no native physics.
"""

from copy import deepcopy
from dataclasses import asdict, fields, replace
from typing import Any

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from src.adapters import continuous_correction_demo as demo
from src.adapters import numpy_continuous_bc as bp
from src.core import continuous_bc as core
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.pick_cube_action_history import PickCubeActionHistoryRecipe
from src.core.pick_cube_action_history_corrections import action_history_correction_view
from src.core.pick_cube_readiness_corrections import (
    GOAL_CONSISTENT_OBSERVED_GRASP_PROVENANCE,
    OBSERVED_GRASP_PROVENANCE,
    GoalConsistentObservedGraspCorrectionBatch,
    ObservedGraspCorrectionBatch,
    ReadinessCorrectionLabel,
    copy_readiness_correction_batch,
    readiness_correction_definition,
    readiness_correction_provenance,
)
from src.core.pick_cube_readiness_roll_in import ReadinessRollInPlan
from src.core.pick_cube_readiness_teacher import (
    GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION,
    OBSERVED_GRASP_DEFINITION,
    GoalConsistentObservedGraspPickCubeTeacher,
    ObservedGraspPickCubeTeacher,
)
from src.core.pick_cube_state_corrections import state_quaternion_correction_view
from src.core.pick_cube_state_quaternions import state_quaternion_projection
from src.core.pick_cube_teacher import PickCubeGeometry
from src.infra import maniskill_continuous as native
from src.infra import maniskill_corrections as queries
from test_maniskill_joint_positions import StateEnvironment
from test_observed_grasp_corrections import RECIPE, assert_state, raw as observed_raw


def raw(index):
    values = list(observed_raw(index))
    values[18] = 1.0
    return tuple(values)


def batch():
    actions = ((0.5, 0.0, 0.0, 0.5), (0.0, -0.25, 0.0, -0.5))
    episode = ContinuousEpisode(
        "pick-cube-smoke-19",
        "pick-cube-train-19",
        "train",
        ContinuousActionSpec(42, native.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4),
        tuple(
            ContinuousTransition(i, raw(i), action, float(i == 1), raw(i + 1), i == 1, False)
            for i, action in enumerate(actions)
        ),
    )
    return GoalConsistentObservedGraspCorrectionBatch(
        19,
        episode,
        (
            ReadinessCorrectionLabel(0, raw(0), (0.18, 0.18, 0.0, -1.0), "lift"),
            ReadinessCorrectionLabel(1, raw(1), (0.0, 0.0, 0.0, -1.0), "settle"),
        ),
        (True, True),
    )


def forbidden(*args, **kwargs):
    pytest.fail("refused/cached goal-consistent correction admitted extra work")


@pytest.mark.parametrize("view_kind", ["copy", "canonical42", "history47", "taskspace12"])
def test_should_preserve_exact_v2_family_feedback_targets_and_original_arrivals_through_views(
    view_kind,
):
    original = batch()
    before = asdict(original)
    if view_kind == "copy":
        model = copy_readiness_correction_batch(original)
    elif view_kind == "canonical42":
        model = state_quaternion_correction_view(original, state_quaternion_projection()).model
    else:
        model = action_history_correction_view(
            original, PickCubeActionHistoryRecipe() if view_kind == "history47" else RECIPE
        ).model
    assert type(model) is GoalConsistentObservedGraspCorrectionBatch and model is not original
    assert model.observed_grasp == (True, True)
    assert readiness_correction_definition(model) == GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION
    assert readiness_correction_provenance(model) == GOAL_CONSISTENT_OBSERVED_GRASP_PROVENANCE
    assert tuple(row.action for row in model.episode.transitions) == tuple(
        row.action for row in original.episode.transitions
    )
    assert [
        (label.stage, label.observed_at, label.labels_arrived_at) for label in model.labels
    ] == [("lift", 0, 1), ("settle", 2, 3)]
    with pytest.raises(ValueError, match="arrived"):
        model.training_pairs(now=2)
    assert model.training_pairs(now=3)[1] == original.training_pairs(now=3)[1]
    if view_kind in ("history47", "taskspace12"):
        assert model.labels[1].observation[-5:] == (0.5, 0.0, 0.0, 0.5, 0.0)
        assert model.labels[1].observation[-5:-1] != model.labels[0].target
    object.__setattr__(model.labels[0], "target", (0.0, 0.0, 0.0, -1.0))
    assert asdict(original) == before


def test_should_keep_v1_payload_schema_but_distinguish_its_explicit_class_and_metadata():
    selected = batch()
    old = ObservedGraspCorrectionBatch(
        selected.train_seed, selected.episode, selected.labels, selected.observed_grasp
    )
    assert asdict(selected) == asdict(old)
    assert [field.name for field in fields(type(selected))] == [
        field.name for field in fields(type(old))
    ]
    assert type(copy_readiness_correction_batch(old)) is ObservedGraspCorrectionBatch
    assert readiness_correction_provenance(old) == OBSERVED_GRASP_PROVENANCE
    assert readiness_correction_definition(old) == OBSERVED_GRASP_DEFINITION
    assert readiness_correction_provenance(selected) != readiness_correction_provenance(old)


@pytest.mark.parametrize("fault", ["feedback", "current_bool", "next_fraction", "role", "subclass"])
def test_should_reown_and_refuse_corrupted_v2_flags_roles_and_foreign_subclasses(fault):
    value = batch()
    if fault == "feedback":
        object.__setattr__(value, "observed_grasp", (True, False))
    elif fault in ("current_bool", "next_fraction"):
        row = value.episode.transitions[0 if fault == "current_bool" else -1]
        name = "observation" if fault == "current_bool" else "next_observation"
        values = list(getattr(row, name))
        values[18] = True if fault == "current_bool" else 0.5
        object.__setattr__(row, name, tuple(values))
    elif fault == "role":
        object.__setattr__(value.episode, "role", "final_test")
    else:

        class ForeignBatch(GoalConsistentObservedGraspCorrectionBatch):
            pass

        value = ForeignBatch(value.train_seed, value.episode, value.labels, value.observed_grasp)
    with pytest.raises(ValueError):
        copy_readiness_correction_batch(value)


@pytest.mark.parametrize("fault", ["mode", "teacher"])
def test_should_poison_a_query_whose_admitted_goal_mode_or_exact_teacher_is_replaced(
    monkeypatch, fault
):
    query = queries.PickCubeCorrectionQuery(
        19,
        readiness_corrections=True,
        observed_grasp_corrections=True,
        goal_consistent_corrections=True,
    )
    assert type(query._teacher) is GoalConsistentObservedGraspPickCubeTeacher
    if fault == "mode":
        query._goal_consistent_corrections = False
    else:
        query._teacher = ObservedGraspPickCubeTeacher()
    report = query.report()
    assert query._failed is True
    assert (
        report["geometry_attempts"]
        == report["teacher_actions"]
        == report["teacher_decision_steps"]
        == 0
    )
    assert report["provenance"] == GOAL_CONSISTENT_OBSERVED_GRASP_PROVENANCE
    assert report["teacher_definition"] == GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION
    query._goal_consistent_corrections = True
    query._teacher = query._admitted_teacher
    monkeypatch.setattr(queries, "_geometry", forbidden)
    with pytest.raises(ValueError, match="failed"):
        query.query(None, 0, raw(0))


@pytest.fixture(scope="module")
def source():
    view = action_history_correction_view(batch(), RECIPE)
    owner = bp.NumpyContinuousBCLearner(
        bp.ContinuousBCPolicy(view.model.episode.spec, hidden_dim=2, seed=17, binary_gripper_axis=3)
    )
    x, y = view.model.training_pairs(now=3)
    owner.train_batch(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    assert any(np.any(row != 0.0) for row in owner.snapshot_state().velocity)
    return owner


class GoalEnvironment(StateEnvironment):
    def __init__(self):
        super().__init__()
        self.reset_observation = np.asarray([raw(0)], dtype=np.float64)

    def step(self, command):
        returned = super().step(command)
        return (np.asarray([raw(len(self.commands))], dtype=np.float64), *returned[1:])


@pytest.fixture(scope="module")
def pipeline(source, tmp_path_factory):
    owner, initial = source.fork(), source.snapshot_state()
    scenes, outcomes, calls, updates = [], [], [], []
    captured: dict[str, Any] = dict(trains=[])
    original_rollout, original_update = (
        smoke.run_frozen_rollout,
        smoke.run_continuous_correction_update,
    )
    runtime_factory, train = demo.ActorShadowRuntime, bp.NumpyContinuousBCLearner.train_batch

    def environment():
        scene = GoalEnvironment()
        scenes.append(scene)
        return scene

    def geometry(port):
        return PickCubeGeometry(
            (-0.3, 0.0, 0.25),
            (0.0, 0.0, 0.0),
            (0.018, 0.018, 0.0) if not port._records else (0.01, 0.0, 0.0),
        )

    def rollout(*args, **options):
        calls.append(options)
        result = original_rollout(*args, **options)
        outcomes.append(result)
        return result

    def update(*args, **options):
        captured["update_batch"] = args[1]
        result = original_update(*args, **options)
        updates.append(result)
        return result

    def runtime(*args, **options):
        result = runtime_factory(*args, **options)
        captured.update(runtime=result, budget=options["budget"], clock=options["clock"])
        return result

    def training(model, features, targets):
        captured["trains"].append((features.copy(), targets.copy()))
        return train(model, features, targets)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(native, "_make_environment", environment)
        patch.setattr(queries, "_geometry", geometry)
        patch.setattr(smoke, "version", lambda _: "fake-goal-corrections")
        patch.setattr(smoke, "run_frozen_rollout", rollout)
        patch.setattr(smoke, "run_continuous_correction_update", update)
        patch.setattr(demo, "ActorShadowRuntime", runtime)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", training)
        patch.setattr(bp.observations, "fit_observation_standardization", forbidden)
        summary = smoke.run_corrective_adaptation(
            owner,
            output_directory=tmp_path_factory.mktemp("goal-corrections") / "case",
            visual_observation=False,
            readiness_corrections=True,
            observed_grasp_corrections=True,
            goal_consistent_corrections=True,
            acquisition_only=True,
            action_history_recipe=RECIPE,
            binary_gripper_axis=3,
        )
    assert summary["complete"] is True, summary
    return owner, initial, summary, scenes, outcomes, updates[0], calls, captured


def test_should_collect_lift_at_the_goal_corner_then_settle_inside_the_euclidean_ball(pipeline):
    _, _, summary, scenes, outcomes, _, calls, _ = pipeline
    assert [row["train_seed"] for row in calls] == [17, 17]
    assert calls[0]["goal_consistent_corrections"] is True
    assert calls[0]["observed_grasp_corrections"] is calls[0]["readiness_corrections"] is True
    assert "goal_consistent_corrections" not in calls[1] and "corrective_queries" not in calls[1]
    value = outcomes[0].corrections
    assert type(value) is GoalConsistentObservedGraspCorrectionBatch
    assert value.observed_grasp == (True, True)
    assert [label.stage for label in value.labels] == ["lift", "settle"]
    assert value.labels[0].target == pytest.approx((0.18, 0.18, 0.0, -1.0))
    assert value.labels[1].target == (0.0, 0.0, 0.0, -1.0)
    assert summary["goal_consistent_corrections"] is True
    assert (
        outcomes[0].report["corrections"]["teacher_definition"]
        == GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION
    )
    assert (
        outcomes[0].report["corrections"]["provenance"] == GOAL_CONSISTENT_OBSERVED_GRASP_PROVENANCE
    )
    assert all(len(scene.commands) == 2 and scene.close_calls == 1 for scene in scenes)


def test_should_continue_eight_private_mixed_updates_with_v2_receipts_and_original_clocks(pipeline):
    owner, initial, _, _, _, result, _, captured = pipeline
    assert type(captured["update_batch"]) is GoalConsistentObservedGraspCorrectionBatch
    report = result.report
    assert report["provenance"] == GOAL_CONSISTENT_OBSERVED_GRASP_PROVENANCE
    assert report["teacher_definition"] == GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION
    assert report["observed_grasp"] == [True, True]
    assert report["source_updates"] == 1 and report["source_training_rows"] == 2
    assert report["work"]["optimizer_updates"] == 8 and report["work"]["training_rows"] == 16
    assert report["work"]["fit_calls"] == report["work"]["reference_replay_rows"] == 0
    assert report["work"]["binary_gripper_loss_rows"] == 16
    assert len(report["applied_updates"]) == 8
    for index, receipt in enumerate(report["applied_updates"]):
        assert receipt["diagnostic"]["definition"] == core.MIXED_OBJECTIVE
        assert receipt["update_number"] == index + 1
        assert receipt["observed_at"] == 2 and receipt["arrived_at"] == receipt["applied_at"] == 3
        assert receipt["episode_id"] == "pick-cube-smoke-17"
    assert (
        captured["budget"].updates_completed == 8
        and captured["budget"].replay_examples_completed == 0
    )
    assert report["progress"]["next_tick"] == 4
    final = result.candidate.snapshot_state()
    assert (final.completed_updates, final.training_examples) == (9, 18)
    assert final.policy == initial.policy
    assert_state(initial, owner.snapshot_state())
    assert_state(initial, captured["runtime"].actor.snapshot_state().state)
    assert_state(final, captured["runtime"].candidate_snapshot().state)


def test_should_use_actual_soft_command_history_and_return_only_the_unchanged_frozen_student(
    pipeline,
):
    owner, initial, _, _, outcomes, result, _, captured = pipeline
    value = outcomes[0].corrections
    model = action_history_correction_view(value, RECIPE).model
    expected_x, expected_y = model.training_pairs(now=3)
    assert len(captured["trains"]) == 8
    for features, targets in captured["trains"]:
        np.testing.assert_array_equal(features, expected_x)
        np.testing.assert_array_equal(targets, expected_y)
    assert all(abs(row.action[3]) < 1.0 for row in value.episode.transitions)
    assert model.labels[1].observation[-5:-1] == value.episode.transitions[0].action
    assert model.labels[1].observation[-5:-1] != model.labels[0].target
    assert outcomes[1].corrections is None and "corrections" not in outcomes[1].report
    assert all(
        outcome.report["frozen_state_exact"]
        is outcome.report["original_owner_exact"]
        is outcome.report["action_history_exact"]
        is True
        for outcome in outcomes
    )
    assert (
        outcomes[1].report["source_updates"]
        == result.candidate.snapshot_state().completed_updates
        == 9
    )
    assert_state(initial, owner.snapshot_state())


@pytest.mark.parametrize("fault", ["integer", "none", "readiness", "observed", "visual", "roll_in"])
def test_should_refuse_invalid_goal_mode_before_private_fork_scene_or_output(
    source, monkeypatch, tmp_path, fault
):
    owner = source.fork()
    monkeypatch.setattr(owner, "fork", forbidden)
    monkeypatch.setattr(native, "_make_environment", forbidden)
    options: dict[str, Any] = dict(
        readiness_corrections=True,
        observed_grasp_corrections=True,
        goal_consistent_corrections=True,
        visual_observation=False,
    )
    if fault in ("integer", "none"):
        options["goal_consistent_corrections"] = 1 if fault == "integer" else None
    elif fault == "readiness":
        options["readiness_corrections"] = False
    elif fault == "observed":
        options["observed_grasp_corrections"] = False
    elif fault == "visual":
        options["visual_observation"] = True
    else:
        options["readiness_roll_in"] = ReadinessRollInPlan(1)
    with pytest.raises(ValueError):
        smoke.run_corrective_adaptation(
            owner,
            output_directory=tmp_path / "refused",
            action_history_recipe=RECIPE,
            binary_gripper_axis=3,
            **options,
        )
    assert not (tmp_path / "refused").exists()


@pytest.mark.parametrize("fault", ["future", "feedback"])
def test_should_keep_original_arrival_and_feedback_admission_before_runtime_or_fork(
    source, monkeypatch, fault
):
    owner, value = source.fork(), batch()
    if fault == "feedback":
        object.__setattr__(value, "observed_grasp", (False, True))
    monkeypatch.setattr(owner, "fork", forbidden)
    monkeypatch.setattr(demo, "ActorShadowRuntime", forbidden)
    with pytest.raises(ValueError):
        demo.run_continuous_correction_update(
            owner,
            value,
            now=2 if fault == "future" else 3,
            action_history_recipe=RECIPE,
            binary_gripper_axis=3,
        )


@pytest.mark.parametrize("fault", ["collection_family", "collection_metadata", "update_metadata"])
def test_should_refuse_cached_old_version_substitution_without_more_paid_model_work(
    pipeline, monkeypatch, tmp_path, fault
):
    owner, initial, _, _, outcomes, updated, _, _ = pipeline
    collection = outcomes[0]
    if fault == "collection_family":
        value = collection.corrections
        old = ObservedGraspCorrectionBatch(
            value.train_seed, value.episode, value.labels, value.observed_grasp
        )
        collection = smoke.FrozenRolloutResult(deepcopy(collection.report), old)
    elif fault == "collection_metadata":
        report = deepcopy(collection.report)
        report["corrections"].update(
            provenance=OBSERVED_GRASP_PROVENANCE, teacher_definition=OBSERVED_GRASP_DEFINITION
        )
        collection = smoke.FrozenRolloutResult(report, collection.corrections)
    calls = []

    def rollout(*args, **options):
        calls.append(options)
        assert len(calls) == 1
        return collection

    def update(*args, **options):
        if fault != "update_metadata":
            forbidden()
        report = deepcopy(updated.report)
        report.update(
            provenance=OBSERVED_GRASP_PROVENANCE, teacher_definition=OBSERVED_GRASP_DEFINITION
        )
        return replace(updated, report=report, candidate=updated.candidate.fork())

    monkeypatch.setattr(smoke, "run_frozen_rollout", rollout)
    monkeypatch.setattr(smoke, "run_continuous_correction_update", update)
    monkeypatch.setattr(native, "_make_environment", forbidden)
    report = smoke.run_corrective_adaptation(
        owner.fork(),
        output_directory=tmp_path / fault,
        visual_observation=False,
        readiness_corrections=True,
        observed_grasp_corrections=True,
        goal_consistent_corrections=True,
        acquisition_only=True,
        action_history_recipe=RECIPE,
        binary_gripper_axis=3,
    )
    assert report["complete"] is False and len(calls) == 1
    if fault == "update_metadata":
        assert report["update"]["work"]["optimizer_updates"] == 8
    assert_state(initial, owner.snapshot_state())
