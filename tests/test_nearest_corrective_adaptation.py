"""Arrived corrective labels extend bounded nearest BC without optimizer work.

Prospective per-platform fixture work: 4 true nearest constructors/0 RNG draws;
8 actual prototype appends/16 added rows, hence 8 append normalizations/48 values.
Two fake pipelines plus one earliest-tie query make 11 one-row predictions/33
normalized values, 36 distance candidates/108 distance feature values,
5 factory/reset/close calls and 10 controls;
no simulator, fitting, SGD or gradients. The
shared two-row source is restored directly (no setup append). Other sources and
per-test owners are RNG-free forks or empty constructors. Failure receipts remain
charged; test corruption never authorizes a retry.
"""

from dataclasses import asdict, replace
import json
from typing import Any

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from src.adapters import continuous_correction_demo as demo
from src.adapters.numpy_nearest_continuous_bc import (
    OBJECTIVE,
    NearestContinuousBCPolicy,
    NearestContinuousBCSnapshot,
    NumpyNearestContinuousBCLearner,
)
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.observation_standardization import ObservationStandardization
from src.core.pick_cube_corrections import (
    PROVENANCE,
    PickCubeCorrectionBatch,
    PickCubeCorrectionLabel,
)
from src.core.pick_cube_teacher import PickCubeGeometry
from src.infra import maniskill_continuous as native
from src.infra import maniskill_corrections as queries
from test_maniskill_bc_smoke import FakeEnvironment


def corrections() -> PickCubeCorrectionBatch:
    spec = ContinuousActionSpec(3, native.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
    rows = tuple(
        ContinuousTransition(
            index,
            (float(index), 0.0, 0.0),
            (0.125, -0.25, 0.5, -0.5),
            float(index == 1),
            (float(index + 1), 0.0, 0.0),
            index == 1,
            False,
        )
        for index in range(2)
    )
    episode = ContinuousEpisode("pick-cube-smoke-19", "pick-cube-train-19", "train", spec, rows)
    return PickCubeCorrectionBatch(
        19,
        episode,
        tuple(
            PickCubeCorrectionLabel(index, row.observation, (0.5, 0.0, 1.0, 1.0), "align")
            for index, row in enumerate(rows)
        ),
    )


def restored_source(*, capacity: int = 256, role: str = "train", empty: bool = False):
    spec = corrections().episode.spec
    policy = NearestContinuousBCPolicy(
        spec,
        ObservationStandardization((0.5, 0.0, 0.0), (0.5, 1.0, 1.0), 2),
        max_prototypes=capacity,
        training_role=role,
    )
    owner = NumpyNearestContinuousBCLearner(policy)
    if not empty:
        owner.restore_state(
            NearestContinuousBCSnapshot(
                policy,
                np.array([[-1.0, 0.0, 0.0], [-19.0, 0.0, 0.0]], dtype=np.float64),
                np.array([[0.125, -0.25, 0.5, -0.5], [-0.5, 0.25, -0.5, 1.0]], dtype=np.float64),
                1,
                2,
            )
        )
    return owner


def assert_state(first, second) -> None:
    assert first.policy == second.policy
    assert first.completed_updates == second.completed_updates
    assert first.training_examples == second.training_examples
    np.testing.assert_array_equal(first.prototype_features, second.prototype_features)
    np.testing.assert_array_equal(first.prototype_actions, second.prototype_actions)


@pytest.fixture(scope="module")
def source():
    return restored_source()


@pytest.fixture(scope="module")
def completed(source):
    owner, batch = source.fork(), corrections()
    initial, original_batch = owner.snapshot_state(), asdict(batch)
    captured: dict[str, Any] = {"trains": [], "runtimes": []}
    runtime_factory, train = demo.ActorShadowRuntime, NumpyNearestContinuousBCLearner.train_batch
    with pytest.MonkeyPatch.context() as patch:

        def runtime(model, **options):
            created = runtime_factory(model, **options)
            captured["runtimes"].append((created, options))
            return created

        def append(model, features, targets):
            captured["trains"].append((model, features.copy(), targets.copy()))
            return train(model, features, targets)

        patch.setattr(demo, "ActorShadowRuntime", runtime)
        patch.setattr(NumpyNearestContinuousBCLearner, "train_batch", append)
        result = demo.run_continuous_correction_update(owner, batch, now=3)
    assert result.report["complete"] is True
    assert asdict(batch) == original_batch
    return owner, batch, initial, result, captured


def test_should_append_arrived_targets_once_and_keep_original_prototype_prefix(completed):
    owner, batch, initial, result, captured = completed
    assert type(result) is demo.NearestContinuousCorrectionResult
    assert type(result.candidate) is NumpyNearestContinuousBCLearner
    runtime, options = captured["runtimes"][0]
    assert len(captured["trains"]) == 1
    trained, features, labels = captured["trains"][0]
    assert trained is runtime._candidate
    assert result.candidate is not owner and result.candidate is not runtime._candidate
    observations, targets = batch.training_pairs(now=3)
    np.testing.assert_array_equal(features, np.asarray(observations))
    np.testing.assert_array_equal(labels, np.asarray(targets))
    assert not np.array_equal(labels, np.asarray([row.action for row in batch.episode.transitions]))
    state = result.candidate.snapshot_state()
    assert state.completed_updates == 2 and state.training_examples == 4
    assert state.policy == initial.policy
    np.testing.assert_array_equal(state.prototype_features[:2], initial.prototype_features)
    np.testing.assert_array_equal(state.prototype_actions[:2], initial.prototype_actions)
    np.testing.assert_array_equal(state.prototype_features[2:], [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_array_equal(state.prototype_actions[2:], targets)
    assert_state(initial, owner.snapshot_state())
    assert_state(initial, runtime.actor.snapshot_state().state)
    assert_state(state, runtime.candidate_snapshot().state)
    assert options["budget"].updates_completed == 1
    assert options["budget"].budget.max_training_updates == 1


def test_should_keep_original_clocks_provenance_and_single_permanent_charge(completed):
    _, batch, initial, result, _ = completed
    report = result.report
    assert report["protocol_id"] == "nearest_continuous_correction_1_v1"
    assert report["provenance"] == PROVENANCE
    assert report["source_updates"] == initial.completed_updates
    assert report["source_training_rows"] == initial.training_examples
    assert report["observed_at"] == 2 and report["labels_arrived_at"] == 3
    assert len(report["events"]) == len(report["applied_updates"]) == 1
    event, receipt = report["events"][0], report["applied_updates"][0]
    assert event["success_validated"] is event["update_attempted"] is True
    assert event["applied"] == [receipt]
    assert receipt["episode_id"] == batch.episode.episode_id
    assert (
        receipt["sample_id"] == "corrective-batch-0" and receipt["event_id"] == "corrective-label-0"
    )
    assert receipt["observed_at"] == 2 and receipt["arrived_at"] == receipt["applied_at"] == 3
    assert receipt["update_number"] == 1
    assert receipt["actor_version"] == "actor-0" and receipt["learner_version"] == "candidate-0"
    assert receipt["diagnostic"] == {"definition": OBJECTIVE, "value": 2}
    assert report["progress"]["updates_completed"] == 1
    assert report["progress"]["replay_examples_completed"] == 0
    assert report["settings"]["max_training_updates"] == 1
    for key, value in {
        "prototype_appends": 1,
        "optimizer_updates": 0,
        "training_rows": 2,
        "unique_correction_rows": 2,
        "repeated_training_rows": 0,
        "fit_calls": 0,
        "forward_passes": 0,
        "backward_passes": 0,
        "normalization_calls": 1,
        "normalization_rows": 2,
        "normalization_values": 6,
    }.items():
        assert report["work"][key] == value, key
    json.dumps(report, allow_nan=False)


def test_should_return_owned_candidate_snapshots(completed):
    owner, _, initial, result, captured = completed
    state = result.candidate.snapshot_state()
    state.prototype_features[0, 0] += 100.0
    state.prototype_actions[0, 0] -= 100.0
    assert_state(
        result.candidate.snapshot_state(), captured["runtimes"][0][0].candidate_snapshot().state
    )
    assert_state(initial, owner.snapshot_state())


def test_should_preserve_earliest_original_label_when_appended_input_conflicts(completed):
    _, batch, initial, result, _ = completed
    # The first correction has the same normalized input as old prototype0.
    # Append preserves both labels and the existing earliest-tie policy.
    state = result.candidate.snapshot_state()
    np.testing.assert_array_equal(state.prototype_features[0], state.prototype_features[2])
    assert tuple(state.prototype_actions[2]) == batch.labels[0].target
    assert tuple(state.prototype_actions[0]) != batch.labels[0].target
    prediction = result.candidate.predict(np.array([[0.0, 0.0, 0.0]], dtype=np.float64))
    np.testing.assert_array_equal(prediction, initial.prototype_actions[:1])
    assert_state(state, result.candidate.snapshot_state())


@pytest.mark.parametrize("now", [True, -1, 2, 3.0, None])
def test_should_refuse_unarrived_or_invalid_time_before_fork(source, monkeypatch, now):
    owner = source.fork()
    monkeypatch.setattr(
        NumpyNearestContinuousBCLearner, "fork", lambda *_: pytest.fail("unarrived labels forked")
    )
    with pytest.raises(ValueError):
        demo.run_continuous_correction_update(owner, corrections(), now=now)


@pytest.mark.parametrize(
    "damage", ["capacity", "role", "empty", "batch_role", "pairing", "foreign"]
)
def test_should_refuse_inadmissible_append_before_runtime(source, monkeypatch, damage):
    if damage == "capacity":
        owner = restored_source(capacity=3)
    elif damage == "role":
        owner = restored_source(role="probe_train")
    elif damage == "empty":
        owner = restored_source(empty=True)
    else:
        owner = source.fork()
    batch = corrections()
    if damage == "batch_role":
        object.__setattr__(batch.episode, "role", "final_test")
    elif damage == "pairing":
        object.__setattr__(batch.labels[0], "observation", (99.0, 0.0, 0.0))
    elif damage == "foreign":
        owner = object()
    monkeypatch.setattr(
        demo, "ActorShadowRuntime", lambda *args, **kwargs: pytest.fail("bad append opened runtime")
    )
    monkeypatch.setattr(
        NumpyNearestContinuousBCLearner, "fork", lambda *_: pytest.fail("bad append forked")
    )
    with pytest.raises(ValueError):
        demo.run_continuous_correction_update(owner, batch, now=3)


@pytest.mark.parametrize(
    "failure",
    ["corrupt_returned_receipt", "postcommit_throw", "empty_poll_mutation", "detach_mutation"],
)
def test_should_preserve_one_paid_append_and_refuse_failed_release(source, monkeypatch, failure):
    owner = source.fork()
    initial = owner.snapshot_state()
    captured: list[Any] = []
    factory, restore, fork = (
        demo.ActorShadowRuntime,
        NumpyNearestContinuousBCLearner.restore_state,
        NumpyNearestContinuousBCLearner.fork,
    )
    forks: list[Any] = []

    def tracked_fork(model):
        child = fork(model)
        forks.append(child)
        return child

    def runtime(model, **options):
        created = factory(model, **options)
        captured.append((created, options))
        ready = created.train_ready

        def changed(*args, **kwargs):
            applied = ready(*args, **kwargs)
            if applied and failure == "postcommit_throw":
                raise RuntimeError("synthetic postcommit failure")
            if applied and failure == "corrupt_returned_receipt":
                return (replace(applied[0], arrived_at=4),)
            if not applied and failure == "empty_poll_mutation":
                state = created._candidate.snapshot_state()
                state.prototype_actions[-1, 0] = -0.5
                created._candidate.restore_state(state)
            return applied

        monkeypatch.setattr(created, "train_ready", changed)
        return created

    def changed_restore(model, state):
        restore(model, state)
        if failure == "detach_mutation" and len(forks) >= 3 and model is forks[2]:
            runtime_owner = captured[0][0]
            assert isinstance(runtime_owner._candidate, NumpyNearestContinuousBCLearner)
            changed = runtime_owner._candidate.snapshot_state()
            changed.prototype_actions[-1, 0] = -0.5
            restore(runtime_owner._candidate, changed)

    monkeypatch.setattr(demo, "ActorShadowRuntime", runtime)
    monkeypatch.setattr(NumpyNearestContinuousBCLearner, "fork", tracked_fork)
    monkeypatch.setattr(NumpyNearestContinuousBCLearner, "restore_state", changed_restore)
    result = demo.run_continuous_correction_update(owner, corrections(), now=3)
    runtime_owner, options = captured[0]
    assert result.report["complete"] is False and result.candidate is None
    assert result.report["progress"]["updates_completed"] == 1
    assert options["budget"].updates_completed == 1
    assert len(result.report["applied_updates"]) == 1
    receipt = result.report["applied_updates"][0]
    assert receipt["arrived_at"] == 3 and receipt["diagnostic"] == {
        "definition": OBJECTIVE,
        "value": 2,
    }
    assert "work" not in result.report
    assert_state(initial, owner.snapshot_state())
    assert_state(initial, runtime_owner.actor.snapshot_state().state)


def test_should_refuse_actor_mutated_by_later_source_snapshot_before_append(source, monkeypatch):
    owner = source.fork()
    initial = owner.snapshot_state()
    factory, snapshot = demo.ActorShadowRuntime, NumpyNearestContinuousBCLearner.snapshot_state
    captured: list[Any] = []
    armed: list[bool] = []
    mutated: list[bool] = []

    def runtime(model, **options):
        created = factory(model, **options)
        captured.append((created, options))
        actor_snapshot = created.actor.snapshot_state

        def observed_actor():
            result = actor_snapshot()
            armed.append(True)
            return result

        monkeypatch.setattr(created.actor, "snapshot_state", observed_actor)
        return created

    def observed_source(model):
        result = snapshot(model)
        if model is owner and armed and not mutated:
            captured[0][0].actor._learner._state.prototype_actions[0, 0] = -0.75
            mutated.append(True)
        return result

    monkeypatch.setattr(demo, "ActorShadowRuntime", runtime)
    monkeypatch.setattr(NumpyNearestContinuousBCLearner, "snapshot_state", observed_source)
    result = demo.run_continuous_correction_update(owner, corrections(), now=3)
    assert mutated == [True]
    assert result.report["complete"] is False and result.candidate is None
    assert result.report["progress"]["updates_completed"] == 0
    assert result.report["applied_updates"] == []
    assert captured[0][1]["budget"].updates_completed == 0
    assert "work" not in result.report
    assert_state(initial, snapshot(owner))


def test_should_refuse_detached_owner_mutated_by_final_clock_getter(source, monkeypatch):
    owner = source.fork()
    initial = owner.snapshot_state()
    factory, fork, clock_now = (
        demo.ActorShadowRuntime,
        NumpyNearestContinuousBCLearner.fork,
        demo.LogicalClock.now,
    )
    captured: list[Any] = []
    forks: list[Any] = []
    mutated: list[bool] = []

    def runtime(model, **options):
        created = factory(model, **options)
        captured.append((created, options))
        return created

    def tracked_fork(model):
        child = fork(model)
        forks.append(child)
        return child

    def observed_clock(clock):
        result = clock_now(clock)
        if len(forks) >= 3 and forks[2]._state.completed_updates == 2 and not mutated:
            forks[2]._state.prototype_actions[-1, 0] = -0.75
            mutated.append(True)
        return result

    monkeypatch.setattr(demo, "ActorShadowRuntime", runtime)
    monkeypatch.setattr(NumpyNearestContinuousBCLearner, "fork", tracked_fork)
    monkeypatch.setattr(demo.LogicalClock, "now", observed_clock)
    result = demo.run_continuous_correction_update(owner, corrections(), now=3)
    assert mutated == [True]
    assert result.report["complete"] is False and result.candidate is None
    assert result.report["progress"]["updates_completed"] == 1
    assert len(result.report["applied_updates"]) == 1
    assert result.report["applied_updates"][0]["diagnostic"] == {
        "definition": OBJECTIVE,
        "value": 2,
    }
    assert captured[0][1]["budget"].updates_completed == 1
    assert "work" not in result.report
    assert_state(initial, owner.snapshot_state())
    assert_state(initial, captured[0][0].actor.snapshot_state().state)


@pytest.fixture
def boundary(monkeypatch, tmp_path):
    environments: list[FakeEnvironment] = []

    def factory():
        environment = FakeEnvironment()
        environments.append(environment)
        return environment

    monkeypatch.setattr(native, "_make_environment", factory)
    monkeypatch.setattr(smoke, "version", lambda _: "fake-nearest-corrective-pipeline")
    monkeypatch.setattr(
        queries,
        "_geometry",
        lambda _: PickCubeGeometry((0.0, 0.0, 0.0), (0.05, 0.0, 0.02), (0.1, 0.0, 0.2)),
    )
    return environments, tmp_path


@pytest.mark.parametrize("fail_first_updated", [False, True])
def test_should_collect_once_append_once_and_freeze_candidate_in_existing_pipeline(
    source, boundary, monkeypatch, fail_first_updated
):
    environments, directory = boundary
    owner = source.fork()
    initial = owner.snapshot_state()
    actual_factory = native._make_environment

    def factory():
        environment = actual_factory()
        if fail_first_updated and len(environments) == 2:
            environment.close_error = RuntimeError("synthetic updated rollout cleanup failure")
        return environment

    monkeypatch.setattr(native, "_make_environment", factory)
    target = directory / "pipeline"
    report = smoke.run_corrective_adaptation(
        owner, output_directory=target, visual_observation=False
    )
    assert report["complete"] is (not fail_first_updated)
    assert report["source_state_exact"] is True
    assert_state(initial, owner.snapshot_state())
    expected_seeds = [[19], [17]] if fail_first_updated else [[19], [17], [19]]
    assert [environment.reset_calls for environment in environments] == expected_seeds
    assert all(
        environment.close_calls == 1 and len(environment.commands) == 2
        for environment in environments
    )
    assert report["update"]["complete"] is True
    assert report["update"]["progress"]["updates_completed"] == 1
    assert report["update"]["work"]["prototype_appends"] == 1
    assert report["update"]["work"]["optimizer_updates"] == 0
    assert len(report["rollouts"]) == (1 if fail_first_updated else 2)
    for environment in environments[1:]:
        for command in environment.commands:
            np.testing.assert_array_equal(
                command, np.array([[0.5, 0.0, 1.0, 1.0]], dtype=np.float32)
            )
    collection = json.loads((target / "collection.json").read_text(encoding="utf8"))
    assert all(
        row["action"] != [0.5, 0.0, 1.0, 1.0] for row in collection["episode"]["transitions"]
    )
    assert collection["corrections"]["batch"]["rows"] == 2
    assert collection["optimizer_updates"] == 0
    if fail_first_updated:
        assert not (target / "updated-19.json").exists()
    else:
        assert (target / "updated-19.json").exists()
    assert json.loads((target / "summary.json").read_text(encoding="utf8")) == json.loads(
        json.dumps(report)
    )
    assert (
        json.loads((target / "update.json").read_text(encoding="utf8"))["progress"][
            "updates_completed"
        ]
        == 1
    )
    with pytest.raises(FileExistsError):
        smoke.run_corrective_adaptation(owner, output_directory=target, visual_observation=False)
    assert len(environments) == len(expected_seeds)
