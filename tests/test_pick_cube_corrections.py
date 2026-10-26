"""Provisional TRAIN targets never replace the frozen student's executed actions."""

from dataclasses import FrozenInstanceError, replace
import json
import sys

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from src.adapters.numpy_continuous_bc import NumpyContinuousBCLearner
from src.adapters.numpy_nearest_continuous_bc import (
    NearestContinuousBCPolicy,
    NumpyNearestContinuousBCLearner,
)
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.continuous_episode_clock import EpisodeClock
from src.core.observation_standardization import ObservationStandardization
from src.core.pick_cube_corrections import (
    PROVENANCE,
    PickCubeCorrectionBatch,
    PickCubeCorrectionLabel,
)
from src.core.pick_cube_teacher import PickCubeGeometry
from src.infra import maniskill_continuous as native
from src.infra import maniskill_corrections as queries
from test_maniskill_bc_smoke import FakeEnvironment, assert_same_state, learner


def episode(seed=17, rows=2):
    spec = ContinuousActionSpec(3, native.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
    records = tuple(
        ContinuousTransition(
            index,
            (float(index), 0.0, 0.0),
            (0.125, -0.25, 0.5, -0.5),
            float(index == rows - 1),
            (float(index + 1), 0.0, 0.0),
            index == rows - 1,
            False,
        )
        for index in range(rows)
    )
    return ContinuousEpisode(
        f"pick-cube-smoke-{seed}",
        "pick-cube" if seed == 17 else "pick-cube-train-19",
        "train",
        spec,
        records,
    )


def labels_for(source):
    return tuple(
        PickCubeCorrectionLabel(index, row.observation, (0.5, 0.0, 1.0, 1.0), "align")
        for index, row in enumerate(source.transitions)
    )


@pytest.fixture
def boundary(monkeypatch, tmp_path):
    environment = FakeEnvironment()
    creations = []

    def factory():
        creations.append(True)
        return environment

    monkeypatch.setattr(native, "_make_environment", factory)
    monkeypatch.setattr(smoke, "version", lambda _: "fake-corrective-boundary")
    monkeypatch.setattr(
        queries,
        "_geometry",
        lambda _: PickCubeGeometry((0.0, 0.0, 0.0), (0.05, 0.0, 0.02), (0.1, 0.0, 0.2)),
    )
    output = tmp_path / "trace.json"
    monkeypatch.setattr(sys, "argv", ["smoke", "--output", str(output)])
    return environment, creations, output


def receipt(output):
    return json.loads(output.read_text(encoding="utf8"))


@pytest.mark.parametrize(
    "step,stage,gripper",
    [
        (0, "align", 1.0),
        (7, "align", 1.0),
        (8, "descend", 1.0),
        (15, "descend", 1.0),
        (16, "grasp", -1.0),
        (21, "grasp", -1.0),
        (22, "lift", -1.0),
        (39, "lift", -1.0),
        (40, "settle", -1.0),
        (49, "settle", -1.0),
    ],
)
def test_should_keep_fixed_stage_and_logical_arrival_clock(step, stage, gripper):
    label = PickCubeCorrectionLabel(step, (2, -3), (0, 0, 0, gripper), stage)
    assert label.observation == (2.0, -3.0)
    assert label.target == (0.0, 0.0, 0.0, gripper)
    assert label.observed_at == 2 * step
    assert label.labels_arrived_at == 2 * step + 1
    with pytest.raises(FrozenInstanceError):
        label.step = 1


@pytest.mark.parametrize(
    "changes",
    [
        dict(step=True),
        dict(step=-1),
        dict(step=50),
        dict(observation=[0.0]),
        dict(observation=(True,)),
        dict(observation=(float("nan"),)),
        dict(target=(0.0, 0.0, 1.01, 1.0)),
        dict(target=(0.0,) * 3),
        dict(target=(0.0, 0.0, 0.0, -1.0)),
        dict(stage="descend"),
        dict(step=16, stage="grasp", target=(0.25, 0.0, 0.0, -1.0)),
        dict(step=40, stage="settle", target=(0.0, 0.0, -0.25, -1.0)),
    ],
)
def test_should_refuse_invalid_or_schedule_inconsistent_labels(changes):
    values = dict(step=0, observation=(0.0,), target=(0.0, 0.0, 0.0, 1.0), stage="align")
    values.update(changes)
    with pytest.raises(ValueError):
        PickCubeCorrectionLabel(**values)


@pytest.mark.parametrize("seed", [17, 19])
def test_should_own_arrived_pairs_without_rewriting_executed_episode(seed):
    source = episode(seed)
    original_labels = labels_for(source)
    batch = PickCubeCorrectionBatch(seed, source, original_labels)
    assert batch.episode == source and batch.episode is not source
    assert batch.labels[0] is not original_labels[0]
    with pytest.raises(ValueError, match="arrived"):
        batch.training_pairs(now=2)
    observations, targets = batch.training_pairs(now=3)
    assert observations == tuple(row.observation for row in source.transitions)
    assert targets == ((0.5, 0.0, 1.0, 1.0),) * 2
    assert targets != tuple(row.action for row in source.transitions)
    assert batch.episode.transitions[0].action == (0.125, -0.25, 0.5, -0.5)
    assert batch.episode.transitions[-1].reward == 1.0
    object.__setattr__(source.transitions[0], "action", (0.0,) * 4)
    object.__setattr__(original_labels[0], "target", (0.0, 0.0, 0.0, 1.0))
    assert batch.training_pairs(now=3) == (observations, targets)
    assert batch.episode.transitions[0].action == (0.125, -0.25, 0.5, -0.5)


@pytest.mark.parametrize("now", [True, -1, 3.0, None])
def test_should_refuse_invalid_release_clock(now):
    source = episode()
    batch = PickCubeCorrectionBatch(17, source, labels_for(source))
    with pytest.raises(ValueError):
        batch.training_pairs(now=now)


@pytest.mark.parametrize("role", ["inner_guard", "probe_train", "probe_test", "final_test"])
def test_should_never_release_heldout_or_guard_episodes(role):
    source = replace(episode(), role=role)
    with pytest.raises(ValueError, match="TRAIN"):
        PickCubeCorrectionBatch(17, source, labels_for(source))


@pytest.mark.parametrize("seed", [True, "17", 23, 29, 31, 37])
def test_should_refuse_other_seed_authorities(seed):
    source = episode()
    with pytest.raises(ValueError):
        PickCubeCorrectionBatch(seed, source, labels_for(source))
    with pytest.raises(ValueError):
        queries.PickCubeCorrectionQuery(seed)


@pytest.mark.parametrize(
    "damage",
    ["missing", "reversed", "observation", "context", "identity", "units", "bounds", "too_many"],
)
def test_should_refuse_inexact_pairing_or_robot_qualification(damage):
    source = episode(rows=51 if damage == "too_many" else 2)
    labels = labels_for(source) if damage != "too_many" else ()
    if damage == "missing":
        labels = labels[:1]
    elif damage == "reversed":
        labels = tuple(reversed(labels))
    elif damage == "observation":
        labels = (replace(labels[0], observation=(9.0, 0.0, 0.0)), labels[1])
    elif damage == "context":
        source = replace(source, context="pick-cube-train-19")
    elif damage == "identity":
        source = replace(source, episode_id="pick-cube-task-teacher-17")
    elif damage == "units":
        source = replace(source, spec=replace(source.spec, action_units=("other",) * 4))
    elif damage == "bounds":
        source = replace(source, spec=replace(source.spec, upper_bounds=(2.0,) * 4))
    with pytest.raises(ValueError):
        PickCubeCorrectionBatch(17, source, labels)


def test_should_revalidate_corrupted_batch_at_release():
    source = episode()
    batch = PickCubeCorrectionBatch(17, source, labels_for(source))
    object.__setattr__(batch.labels[1], "step", 0)
    with pytest.raises(ValueError):
        batch.training_pairs(now=100)


def test_should_accept_exactly_fifty_pairs_without_retaining_an_expert_trajectory():
    source = episode(rows=50)
    labels = []
    for index, row in enumerate(source.transitions):
        stage = next(
            name
            for limit, name in (
                (8, "align"),
                (16, "descend"),
                (22, "grasp"),
                (40, "lift"),
                (50, "settle"),
            )
            if index < limit
        )
        gripper = 1.0 if index < 16 else -1.0
        labels.append(
            PickCubeCorrectionLabel(index, row.observation, (0.0, 0.0, 0.0, gripper), stage)
        )
    batch = PickCubeCorrectionBatch(17, source, tuple(labels))
    with pytest.raises(ValueError, match="arrived"):
        batch.training_pairs(now=98)
    observations, targets = batch.training_pairs(now=99)
    assert len(observations) == len(targets) == 50
    assert batch.episode == source
    assert batch.episode.transitions[-1].action[-1] == -0.5
    assert targets[-1][-1] == -1.0


def test_should_query_same_public_state_without_stepping_and_own_reports(boundary):
    environment, _, _ = boundary
    port = native.ManiSkillPickCubeEpisode("pick-cube-smoke-17", "pick-cube", seed=17)
    query = queries.PickCubeCorrectionQuery(17)
    try:
        for step in range(2):
            before = port.observation
            query.query(port, step, before)
            assert len(environment.commands) == step
            assert environment.reset_calls == [17]
            item = query.report()["labels"][-1]
            assert item["observation"] == before
            assert item["target"] == pytest.approx((0.5, 0.0, 1.0, 1.0))
            assert item["observed_at"] == 2 * step
            assert item["labels_arrived_at"] == 2 * step + 1
            port.step((0.0,) * 4)
        batch = query.finish(port.episode())
        assert len(batch.training_pairs(now=3)[0]) == 2
        report = query.report()
        assert (
            report["geometry_attempts"]
            == report["geometry_returns"]
            == report["teacher_actions"]
            == 2
        )
        assert report["provenance"] == PROVENANCE and report["batch"] is None
        report["labels"][0]["target"] = ()
        assert query.report()["labels"][0]["target"] != ()
    finally:
        port.close()


@pytest.mark.parametrize(
    "damage",
    ["step", "observation", "role", "context", "identity", "seed", "non_native_observation"],
)
def test_should_refuse_invalid_query_before_privileged_geometry(boundary, monkeypatch, damage):
    seed = 19 if damage == "seed" else 17
    port = native.ManiSkillPickCubeEpisode("pick-cube-smoke-17", "pick-cube", seed=seed)
    query = queries.PickCubeCorrectionQuery(17)
    step, observation = 0, port.observation
    if damage == "step":
        step = True
    elif damage == "observation":
        observation = (9.0, 0.0, 0.0)
    elif damage == "role":
        port._role = "probe_train"
    elif damage == "context":
        port._context = "pick-cube-train-19"
    elif damage == "identity":
        port._episode_id = "teacher"
    elif damage == "non_native_observation":
        observation = tuple(np.float64(value) for value in observation)
    monkeypatch.setattr(queries, "_geometry", lambda _: pytest.fail("geometry must not run"))
    try:
        with pytest.raises(ValueError):
            query.query(port, step, observation)
        assert query.report()["geometry_attempts"] == 0
        with pytest.raises(ValueError, match="cannot resume"):
            query.query(port, 0, port.observation)
    finally:
        port.close()


@pytest.mark.parametrize(
    "failure",
    [
        "geometry",
        "geometry_mutates_observation",
        "teacher",
        "teacher_after_advance",
        "teacher_mutates_observation",
    ],
)
def test_should_preserve_partial_query_work_and_permanently_stop(boundary, monkeypatch, failure):
    port = native.ManiSkillPickCubeEpisode("pick-cube-smoke-17", "pick-cube", seed=17)
    query = queries.PickCubeCorrectionQuery(17)
    query.query(port, 0, port.observation)
    port.step((0.0,) * 4)
    geometry = queries._geometry

    def broken(current):
        if failure == "geometry":
            raise RuntimeError("geometry unavailable")
        if failure == "geometry_mutates_observation":
            current._observation = (99.0, 0.0, 0.0)
        return geometry(current)

    monkeypatch.setattr(queries, "_geometry", broken)
    if failure == "teacher":
        monkeypatch.setattr(
            query._teacher,
            "action",
            lambda _: (_ for _ in ()).throw(RuntimeError("teacher failed")),
        )
    elif failure in ("teacher_after_advance", "teacher_mutates_observation"):
        original_action = query._teacher.action

        def changed_action(geometry):
            result = original_action(geometry)
            if failure == "teacher_after_advance":
                raise RuntimeError("teacher failed after advancing")
            port._observation = (99.0, 0.0, 0.0)
            return result

        monkeypatch.setattr(query._teacher, "action", changed_action)
    try:
        with pytest.raises((ValueError, RuntimeError)):
            query.query(port, 1, port.observation)
        report = query.report()
        assert report["geometry_attempts"] == 2
        assert report["geometry_returns"] == (1 if failure == "geometry" else 2)
        assert report["teacher_action_attempts"] == (1 if failure.startswith("geometry") else 2)
        assert report["teacher_actions"] == (2 if failure == "teacher_mutates_observation" else 1)
        assert report["teacher_schedule_steps"] == (
            2 if failure in ("teacher_after_advance", "teacher_mutates_observation") else 1
        )
        assert len(report["labels"]) == (2 if failure == "teacher_mutates_observation" else 1)
        assert report["batch"] is None
        with pytest.raises(ValueError, match="cannot resume"):
            query.query(port, 1, port.observation)
        with pytest.raises(ValueError, match="no usable batch"):
            query.finish(episode())
        assert len(boundary[0].commands) == 1
    finally:
        port.close()


@pytest.mark.parametrize("metadata", ["_seed", "_role", "_context"])
def test_should_bind_one_port_and_refuse_changed_executed_trajectory(boundary, metadata):
    first = native.ManiSkillPickCubeEpisode("pick-cube-smoke-17", "pick-cube", seed=17)
    query = queries.PickCubeCorrectionQuery(17)
    try:
        for step in range(2):
            query.query(first, step, first.observation)
            first.step((0.0,) * 4)
        actual = first.episode()
        altered = replace(
            actual, transitions=(replace(actual.transitions[0], reward=0.5), actual.transitions[1])
        )
        with pytest.raises(ValueError, match="differs"):
            query.finish(altered)
        assert query.finish(actual).episode == actual
        setattr(
            first,
            metadata,
            {"_seed": 19, "_role": "probe_train", "_context": "pick-cube-train-19"}[metadata],
        )
        with pytest.raises(ValueError):
            query.finish(actual)
    finally:
        first.close()


@pytest.mark.parametrize("seed", [17, 19])
def test_should_execute_frozen_student_controls_and_keep_teacher_targets_separate(
    boundary, monkeypatch, seed
):
    environment, creations, output = boundary
    model = learner()
    before = model.snapshot_state()
    original_predict = NumpyContinuousBCLearner.predict
    features = []

    def predict(child, values):
        features.append(values.copy())
        return original_predict(child, values)

    monkeypatch.setattr(NumpyContinuousBCLearner, "predict", predict)
    monkeypatch.setattr(
        NumpyContinuousBCLearner, "train_batch", lambda *args: pytest.fail("no SGD")
    )
    assert smoke.main(learner=model, train_seed=seed, corrective_queries=True) == 0
    result = receipt(output)
    assert creations == [True] and environment.reset_calls == [seed]
    assert result["complete"] is True and result["optimizer_updates"] == 0
    assert result["prediction_attempts"] == result["prediction_returns"] == 2
    assert_same_state(before, model.snapshot_state())
    expected = np.tanh(np.array([[0.2, -0.3, 0.4, -0.5]])).astype(np.float32)
    for index, command in enumerate(environment.commands):
        np.testing.assert_array_equal(command, expected)
        assert result["episode"]["transitions"][index]["action"] == command[0].tolist()
        assert features[index].shape == (1, 3)
        assert features[index][0].tolist() == result["episode"]["transitions"][index]["observation"]
    corrections = result["corrections"]
    assert (
        corrections["geometry_attempts"]
        == corrections["geometry_returns"]
        == corrections["teacher_actions"]
        == 2
    )
    assert corrections["batch"] == dict(
        train_seed=seed, episode_id=f"pick-cube-smoke-{seed}", rows=2, labels_arrived_at=3
    )
    for index, label in enumerate(corrections["labels"]):
        assert label["observation"] == result["episode"]["transitions"][index]["observation"]
        assert label["target"] == pytest.approx((0.5, 0.0, 1.0, 1.0))
        assert label["target"] != result["episode"]["transitions"][index]["action"]


@pytest.mark.parametrize("failure", ["geometry_second", "step", "cleanup", "owner_mutation"])
def test_should_keep_failed_smoke_paid_work_without_releasing_batch(boundary, monkeypatch, failure):
    environment, _, output = boundary
    model = learner()
    geometry = queries._geometry
    calls = []

    def checked(port):
        calls.append(True)
        if failure == "geometry_second" and len(calls) == 2:
            raise RuntimeError("second geometry failed")
        if failure == "owner_mutation":
            changed = model.snapshot_state()
            changed.velocity[0].flat[0] += 1.0
            model.restore_state(changed)
        return geometry(port)

    monkeypatch.setattr(queries, "_geometry", checked)
    if failure == "step":
        environment.step_error = RuntimeError("control failed")
    if failure == "cleanup":
        environment.close_error = RuntimeError("close failed")
    assert smoke.main(learner=model, corrective_queries=True) == 1
    result = receipt(output)
    assert result["complete"] is False and result["corrections"]["batch"] is None
    assert result["optimizer_updates"] == 0 and environment.close_calls == 1
    expected_labels = 2 if failure == "cleanup" else 1
    assert len(result["corrections"]["labels"]) == expected_labels
    assert result["corrections"]["geometry_attempts"] == (
        2 if failure in ("geometry_second", "cleanup") else 1
    )
    assert result["corrections"]["geometry_returns"] == (2 if failure == "cleanup" else 1)
    if failure == "owner_mutation":
        assert environment.commands == []
        assert result["work"]["step_attempts"] == 0
        assert result["original_owner_exact"] is False
    elif failure == "step":
        assert result["work"]["step_attempts"] == 1
        assert result["work"]["completed_steps"] == 0
    else:
        assert len(environment.commands) == (2 if failure == "cleanup" else 1)


@pytest.mark.parametrize(
    "options",
    [{}, {"actor": object()}, {"learner": "model", "observation_clock": EpisodeClock(50, True)}],
)
def test_should_refuse_unsupported_correction_routes_before_scene(boundary, options):
    environment, creations, output = boundary
    if options.get("learner") == "model":
        options = dict(options, learner=learner())
    assert smoke.main(corrective_queries=True, **options) == 1
    result = receipt(output)
    assert result["complete"] is False and result["creation_attempts"] == 0
    assert creations == [] and environment.commands == []


@pytest.mark.parametrize("value", [1, "true", None])
def test_should_refuse_non_boolean_opt_in_before_scene(boundary, value):
    environment, creations, output = boundary
    assert smoke.main(learner=learner(), corrective_queries=value) == 1
    assert receipt(output)["complete"] is False
    assert creations == [] and environment.commands == []


def test_should_preserve_default_without_teacher_queries(boundary, monkeypatch):
    environment, _, output = boundary
    monkeypatch.setattr(
        smoke, "PickCubeCorrectionQuery", lambda *args: pytest.fail("default query")
    )
    assert smoke.main(learner=learner()) == 0
    assert "corrections" not in receipt(output)
    assert len(environment.commands) == 2


def test_should_accept_explicit_cli_opt_in_and_preserve_exclusive_receipt(boundary, monkeypatch):
    environment, _, output = boundary
    monkeypatch.setattr(sys, "argv", ["smoke", "--output", str(output), "--corrective-queries"])
    assert smoke.main(learner=learner()) == 0
    first = output.read_bytes()
    with pytest.raises(FileExistsError):
        smoke.main(learner=learner())
    assert output.read_bytes() == first
    assert len(environment.commands) == 2


def test_should_refuse_replacement_port_with_identical_visible_metadata(boundary, monkeypatch):
    first = native.ManiSkillPickCubeEpisode("pick-cube-smoke-17", "pick-cube", seed=17)
    query = queries.PickCubeCorrectionQuery(17)
    query.query(first, 0, first.observation)
    first.step((0.0,) * 4)
    other_environment = FakeEnvironment()
    monkeypatch.setattr(native, "_make_environment", lambda: other_environment)
    second = native.ManiSkillPickCubeEpisode("pick-cube-smoke-17", "pick-cube", seed=17)
    try:
        second.step((0.0,) * 4)
        assert first.observation == second.observation and first.spec == second.spec
        assert first.work == second.work
        with pytest.raises(ValueError, match="current TRAIN"):
            query.query(second, 1, second.observation)
        assert query.report()["geometry_attempts"] == 1
        with pytest.raises(ValueError, match="cannot resume"):
            query.query(first, 1, first.observation)
    finally:
        first.close()
        second.close()


def test_should_collect_nearest_student_failure_without_claiming_expert_success(boundary):
    environment, _, output = boundary
    environment.rewards = (0.0,)
    environment.terminated, environment.truncated = False, True
    spec = ContinuousActionSpec(3, native.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
    model = NumpyNearestContinuousBCLearner(
        NearestContinuousBCPolicy(spec, ObservationStandardization((0.0,) * 3, (1.0,) * 3, 1))
    )
    # One tiny prototype append, no SGD; fitting is separate from collection.
    model.train_batch(np.zeros((1, 3)), np.array([[0.25, -0.5, 0.0, -1.0]]))
    before = model.snapshot_state()
    assert smoke.main(learner=model, corrective_queries=True) == 0
    result = receipt(output)
    assert result["complete"] and not result["final_success"] and not result["any_success"]
    assert result["sparse_return"] == 0.0 and result["corrections"]["batch"]["rows"] == 2
    assert result["episode"]["transitions"][-1]["truncated"] is True
    for row in result["episode"]["transitions"]:
        assert row["action"] == [0.25, -0.5, 0.0, -1.0]
    after = model.snapshot_state()
    assert before.policy == after.policy
    assert before.completed_updates == after.completed_updates == 1
    assert before.training_examples == after.training_examples == 1
    np.testing.assert_array_equal(before.prototype_features, after.prototype_features)
    np.testing.assert_array_equal(before.prototype_actions, after.prototype_actions)
