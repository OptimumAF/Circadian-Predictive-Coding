"""Synthetic saved-TRAIN curation: new clocks, paid queries, and actual history.

No simulator, learned model, artifact data, or numerical learner is used. The
declared fixture allowance counts teacher attempts, including failed queries.
"""

from dataclasses import FrozenInstanceError, replace

import pytest

import src.adapters.saved_readiness_corrections as saved
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.curated_readiness_corrections import (
    CuratedReadinessBatch,
    CuratedReadinessLabel,
    copy_curated_readiness_batch,
)
from src.core.pick_cube_readiness_teacher import (
    GoalConsistentObservedGraspPickCubeTeacher,
)
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.core.pick_cube_train_context import pick_cube_training_context


def _raw(*, grasped=0.0, cube=(0.025, -0.05, 0.08), goal=(0.0, 0.0, 0.1)):
    values = [0.0] * 42
    values[18] = grasped
    values[36:39] = cube
    values[39:42] = goal
    return tuple(values)


def _episode(*, seed=17, rows=3, observations=None, actions=None):
    observations = tuple(_raw() for _ in range(rows + 1)) if observations is None else observations
    actions = tuple((0.25, 0.0, -0.5, -0.25) for _ in range(rows)) if actions is None else actions
    spec = ContinuousActionSpec(
        42,
        ("normalized_root_dx", "normalized_root_dy", "normalized_root_dz", "normalized_gripper"),
        (-1.0,) * 4,
        (1.0,) * 4,
    )
    transitions = tuple(
        ContinuousTransition(
            index,
            observations[index],
            actions[index],
            0.0,
            observations[index + 1],
            False,
            index == rows - 1,
        )
        for index in range(rows)
    )
    return ContinuousEpisode(
        f"pick-cube-smoke-{seed}", pick_cube_training_context(seed), "train", spec, transitions
    )


def _curate(episode, **kwargs):
    return saved.curate_saved_readiness(
        episode, train_seed=17, source_available_at=80, started_at=100, **kwargs
    )


def _forbid_queries(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid saved input reached a teacher query")

    monkeypatch.setattr(saved.GoalConsistentObservedGraspPickCubeTeacher, "decide", unexpected)


def _direct_batch():
    episode = _episode(rows=1)
    label = CuratedReadinessLabel(
        0, episode.transitions[0].observation, (0.25, -0.5, 1.0, 1.0), "align", None, 100, 101
    )
    return CuratedReadinessBatch(17, episode, (label,), 80)


def test_should_lift_outside_euclidean_goal_radius_after_settle():
    # The second displacement has each component below .025 but norm > .025.
    # This independently distinguishes the admitted v2 policy from old v1 hold.
    observations = (
        _raw(grasped=1.0, goal=(0.0, 0.0, 0.0)),
        _raw(grasped=1.0, goal=(0.019, 0.019, 0.0)),
        _raw(grasped=0.0),
    )
    result = _curate(_episode(rows=2, observations=observations))

    assert result.batch is not None
    first, second = result.batch.labels
    assert first.stage == "settle"
    assert first.target == (0.0, 0.0, 0.0, -1.0)
    assert second.stage == "lift"
    assert second.target == pytest.approx((0.19, 0.19, 0.0, -1.0))


def test_should_query_relative_geometry_and_preceding_actual_gripper(monkeypatch):
    calls = []
    original = GoalConsistentObservedGraspPickCubeTeacher.decide

    def observe(self, geometry, *, grasped, executed_gripper):
        calls.append((geometry, grasped, executed_gripper))
        return original(self, geometry, grasped=grasped, executed_gripper=executed_gripper)

    monkeypatch.setattr(saved.GoalConsistentObservedGraspPickCubeTeacher, "decide", observe)
    actions = ((0.25, 0.0, -0.5, -0.25), (0.5, -0.25, 0.25, 0.5), (0.0,) * 4)
    episode = _episode(actions=actions)
    result = _curate(episode)

    assert result.batch is not None
    assert [call[2] for call in calls] == [0.0, -0.25, 0.5]
    assert all(call[1] is False for call in calls)
    geometry = calls[0][0]
    assert geometry.tcp == (0.0, 0.0, 0.0)
    assert geometry.cube == (0.025, -0.05, 0.08)
    assert geometry.goal == pytest.approx((0.025, -0.05, 0.18))
    assert result.batch.labels[0].target == (0.25, -0.5, 1.0, 1.0)
    assert result.batch.episode == episode
    assert tuple(row.action for row in episode.transitions) == actions


def test_should_assign_new_query_and_arrival_clocks_without_inventing_observation_time():
    result = _curate(_episode(rows=2))

    assert result.batch is not None
    labels = result.batch.labels
    assert [label.source_observed_at for label in labels] == [None, None]
    assert [label.queried_at for label in labels] == [100, 102]
    assert [label.arrived_at for label in labels] == [101, 103]
    assert result.report["source_observed_at"] == [None, None]
    assert result.report["teacher_queries_attempted"] == 2
    assert result.report["teacher_queries_returned"] == 2
    with pytest.raises(ValueError):
        result.batch.training_pairs(now=102)
    observations, targets = result.batch.training_pairs(now=103)
    assert observations == tuple(label.observation for label in labels)
    assert targets == tuple(label.target for label in labels)


def test_should_preserve_known_original_clocks_as_separate_metadata():
    result = _curate(_episode(rows=2), source_observed_at=(4, 9))

    assert result.batch is not None
    assert [label.source_observed_at for label in result.batch.labels] == [4, 9]
    assert [label.queried_at for label in result.batch.labels] == [100, 102]
    assert result.report["source_observed_at"] == [4, 9]


def test_should_copy_owned_batch_and_keep_frozen_labels():
    result = _curate(_episode(rows=2))
    assert result.batch is not None
    copied = copy_curated_readiness_batch(result.batch)

    assert copied == result.batch
    assert copied is not result.batch
    assert copied.episode is not result.batch.episode
    assert copied.labels[0] is not result.batch.labels[0]
    with pytest.raises(FrozenInstanceError):
        copied.labels[0].arrived_at = 0  # type: ignore[misc]
    object.__setattr__(result.batch.labels[0], "queried_at", True)
    with pytest.raises(ValueError):
        copy_curated_readiness_batch(result.batch)
    assert copied.labels[0].queried_at == 100


def test_should_revalidate_tampered_evidence_before_training_release():
    batch = _direct_batch()
    object.__setattr__(batch.labels[0], "observation", (0.0,) * 42)

    with pytest.raises(ValueError):
        batch.training_pairs(now=200)


@pytest.mark.parametrize(
    "change",
    [
        {"source_observed_at": True},
        {"source_observed_at": -1},
        {"source_observed_at": 101},
        {"queried_at": True},
        {"queried_at": -1},
        {"arrived_at": 99},
        {"arrived_at": True},
        {"step": True},
        {"observation": (0.0,) * 12},
        {"target": (0.0, 0.0, 0.0, -1.0)},
    ],
)
def test_should_refuse_invalid_label_metadata(change):
    label = _direct_batch().labels[0]
    with pytest.raises(ValueError):
        replace(label, **change)


def test_should_refuse_batch_clock_order_and_changed_context():
    batch = _direct_batch()
    with pytest.raises(ValueError):
        replace(batch, source_available_at=102)
    with pytest.raises(ValueError):
        replace(batch, episode=replace(batch.episode, context="pick-cube-train-19"))
    with pytest.raises(ValueError):
        replace(batch, labels=(replace(batch.labels[0], source_observed_at=81),))
    changed_row = replace(batch.episode.transitions[0], action=(0.1, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        replace(batch, episode=replace(batch.episode, transitions=(changed_row,)))


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test", "inner_guard"])
def test_should_refuse_evaluation_sources_before_teacher_query(monkeypatch, role):
    _forbid_queries(monkeypatch)
    episode = replace(_episode(rows=1), role=role)
    with pytest.raises(ValueError):
        _curate(episode)


@pytest.mark.parametrize(
    "change",
    [{"episode_id": "replacement"}, {"context": "pick-cube-train-19"}],
)
def test_should_refuse_wrong_source_binding_before_teacher_query(monkeypatch, change):
    _forbid_queries(monkeypatch)
    with pytest.raises(ValueError):
        _curate(replace(_episode(rows=1), **change))


@pytest.mark.parametrize("seed", [23, 24, 29, True])
def test_should_refuse_reserved_or_noninteger_seed_before_teacher_query(monkeypatch, seed):
    _forbid_queries(monkeypatch)
    with pytest.raises(ValueError):
        saved.curate_saved_readiness(
            _episode(rows=1), train_seed=seed, source_available_at=80, started_at=100
        )


@pytest.mark.parametrize("clocks", [(True,), (-1,), (81,), (1, 2), [1]])
def test_should_refuse_bad_original_clocks_before_teacher_query(monkeypatch, clocks):
    _forbid_queries(monkeypatch)
    with pytest.raises(ValueError):
        _curate(_episode(rows=1), source_observed_at=clocks)


@pytest.mark.parametrize("available,start", [(True, 100), (-1, 100), (101, 100), (80, True)])
def test_should_refuse_bad_new_clocks_before_teacher_query(monkeypatch, available, start):
    _forbid_queries(monkeypatch)
    with pytest.raises(ValueError):
        saved.curate_saved_readiness(
            _episode(rows=1), train_seed=17, source_available_at=available, started_at=start
        )


@pytest.mark.parametrize("field", ["observation", "next_observation", "action"])
def test_should_validate_all_source_flags_and_native_actions_before_query(monkeypatch, field):
    _forbid_queries(monkeypatch)
    episode = _episode(rows=2)
    row = episode.transitions[-1]
    if field == "action":
        changed = (0.1, 0.0, 0.0, 0.0)  # Not already exactly native float32.
    else:
        changed_values = list(getattr(row, field))
        changed_values[18] = 1  # Original exact native float flag is mandatory.
        changed = tuple(changed_values)
    object.__setattr__(row, field, changed)
    with pytest.raises(ValueError):
        _curate(episode)


def test_should_preserve_paid_prefix_when_teacher_raises(monkeypatch):
    original = GoalConsistentObservedGraspPickCubeTeacher.decide
    attempts = 0

    def fail_third(self, geometry, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 3:
            raise RuntimeError("synthetic teacher failure")
        return original(self, geometry, **kwargs)

    monkeypatch.setattr(saved.GoalConsistentObservedGraspPickCubeTeacher, "decide", fail_third)
    result = _curate(_episode(rows=4))

    assert result.batch is None
    assert result.report["complete"] is False
    assert "synthetic teacher failure" in result.report["error"]
    assert result.report["teacher_queries_attempted"] == 3
    assert result.report["teacher_queries_returned"] == 2
    assert len(result.report["labels"]) == 2
    assert [label["queried_at"] for label in result.report["labels"]] == [100, 102]
    assert [label["arrived_at"] for label in result.report["labels"]] == [101, 103]


def test_should_refuse_callback_source_replacement_and_retain_paid_queries(monkeypatch):
    episode = _episode(rows=3)
    original = GoalConsistentObservedGraspPickCubeTeacher.decide
    attempts = 0

    def replace_source(self, geometry, **kwargs):
        nonlocal attempts
        attempts += 1
        decision = original(self, geometry, **kwargs)
        if attempts == 2:
            object.__setattr__(episode, "context", "replaced-after-paid-query")
        return decision

    monkeypatch.setattr(saved.GoalConsistentObservedGraspPickCubeTeacher, "decide", replace_source)
    result = _curate(episode)

    assert result.batch is None
    assert result.report["complete"] is False
    assert result.report["error"]
    assert result.report["teacher_queries_attempted"] == 2
    assert result.report["teacher_queries_returned"] == 2
    assert result.report["source_context"] == "pick-cube"
    assert result.report["labels"][0]["queried_at"] == 100


def test_should_project_executed_history_instead_of_new_teacher_targets():
    actions = ((0.25, 0.0, -0.5, -0.25), (0.5, -0.25, 0.25, 0.5), (0.0,) * 4)
    result = _curate(_episode(actions=actions))
    assert result.batch is not None

    features, targets = saved.project_curated_training_pairs(result.batch, now=105)

    assert features[0] == pytest.approx(
        (0.0, 0.25, -0.5, 0.8, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )
    assert features[1][7:11] == actions[0]
    assert features[1][11] == pytest.approx(1 / 6)
    assert features[2][7:11] == actions[1]
    assert features[2][11] == 0.0
    assert targets == tuple(label.target for label in result.batch.labels)
    assert features[1][7:11] != targets[0]


def test_should_refuse_unarrived_labels_before_projection(monkeypatch):
    batch = _direct_batch()

    def unexpected(*args, **kwargs):
        pytest.fail("unarrived labels reached observation projection")

    monkeypatch.setattr(saved, "action_history_training_episode", unexpected)
    with pytest.raises(ValueError):
        saved.project_curated_training_pairs(batch, now=100)
    with pytest.raises(ValueError):
        saved.project_curated_training_pairs(
            batch, now=101, recipe=PickCubeTaskSpaceHistoryRecipe(mode="zero")
        )
