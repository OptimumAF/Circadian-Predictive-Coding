"""Provisional Push labels on student executions, with no learner or SDK math.

Pure labels use one cached50-row trajectory. Integration uses the accepted native
port and scripted fake environment; zero-valued literal learner arrays never run
a model forward or optimizer. Queried expert commands are never executed here.
Full-module forecast:73 actual teacher decisions,21 fake predictions (20 return),
16 controls (15 return),18 factories (17 return),17 resets,32 grasp queries,
17 closes (16 return). An external wrapper hides one paid teacher return.
"""

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.adapters import numpy_continuous_bc as bp
from src.adapters import push_cube_correction_collection as collection
from src.core import push_cube_corrections as corrections
from src.core import push_cube_teacher as teacher
from src.core import continuous_bc as model_math
from src.core.continuous_demonstration import ContinuousEpisode, ContinuousTransition
from src.core.cube_taskspace_state import (
    CubeTaskSpaceState,
    cube_taskspace_observation,
    cube_taskspace_spec,
)
from src.core.pick_cube_action_history import ActionHistoryState, advance_action_history
from src.infra import maniskill_push_cube as native
from test_push_cube_continuous import FakePushEnvironment


def forbidden(*args, **kwargs):
    pytest.fail("correction fixture admitted actual model, optimizer, fit or SDK work")


@pytest.fixture(scope="module", autouse=True)
def ledger():
    counts = dict(
        teacher_attempts=0,
        teacher_returns=0,
        prediction_attempts=0,
        prediction_returns=0,
        factory_attempts=0,
        factory_returns=0,
        reset_attempts=0,
        reset_returns=0,
        control_attempts=0,
        control_returns=0,
        grasp_attempts=0,
        grasp_returns=0,
        close_attempts=0,
        close_returns=0,
    )
    decide = teacher.PushCubeTeacher.decide

    def decided(owner, *args, **options):
        counts["teacher_attempts"] += 1
        returned = decide(owner, *args, **options)
        counts["teacher_returns"] += 1
        return returned

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(teacher.PushCubeTeacher, "decide", decided)
        patch.setattr(bp.NumpyContinuousBCLearner, "__init__", forbidden)
        patch.setattr(bp.NumpyContinuousBCLearner, "fork", forbidden)
        patch.setattr(bp.NumpyContinuousBCLearner, "predict", forbidden)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", forbidden)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", forbidden)
        patch.setattr(bp, "fit_training_standardization", forbidden)
        patch.setattr(model_math, "_forward", forbidden)
        patch.setattr(model_math, "continuous_bc_loss_and_gradients", forbidden)
        patch.setattr(native, "_make_environment", forbidden)
        yield counts
    if os.environ.get("PUSH_CORRECTION_FIXTURE_LEDGER"):
        with Path(os.environ["PUSH_CORRECTION_FIXTURE_LEDGER"]).open(
            "x", encoding="utf8"
        ) as output:
            json.dump(counts, output, sort_keys=True)
            output.write("\n")
    assert counts["teacher_attempts"] <= 1024
    assert counts["prediction_attempts"] <= 512 and counts["control_attempts"] <= 512
    assert counts["factory_attempts"] <= 64 and counts["close_attempts"] <= 64
    assert counts["teacher_attempts"] == counts["teacher_returns"] == 73
    assert (counts["prediction_attempts"], counts["prediction_returns"]) == (21, 20)
    assert (counts["control_attempts"], counts["control_returns"]) == (16, 15)
    assert (counts["factory_attempts"], counts["factory_returns"]) == (18, 17)
    assert counts["reset_attempts"] == counts["reset_returns"] == 17
    assert counts["grasp_attempts"] == counts["grasp_returns"] == 32
    assert (counts["close_attempts"], counts["close_returns"]) == (17, 16)


@pytest.fixture(scope="module")
def batch(ledger):
    state = CubeTaskSpaceState(False, (-0.06, 0.0, 0.12), (0.0, 0.0, 0.02), (0.4, 0.0, 0.02))
    policy, history = teacher.PushCubeTeacher(), ActionHistoryState()
    command = (0.125, 0.0, 0.0, 1.0)
    labels, rows = [], []
    for index in range(50):
        observation = cube_taskspace_observation(state, history)
        decision = policy.decide(state, history=history)
        labels.append(
            corrections.PushCubeCorrectionLabel(
                index, state, observation, decision.action, decision.stage
            )
        )
        history = advance_action_history(history, command)
        rows.append(
            ContinuousTransition(
                index,
                observation,
                command,
                0.0,
                cube_taskspace_observation(state, history),
                index == 49,
                False,
            )
        )
    trajectory = ContinuousEpisode(
        "push-cube-student-17", "push-cube-train-17", "train", cube_taskspace_spec(), tuple(rows)
    )
    return corrections.PushCubeCorrectionBatch(17, trajectory, tuple(labels))


def test_should_release_all_fifty_provisional_labels_at_original_arrival_99_without_teacher_requery(
    batch, ledger
):
    queries = ledger["teacher_attempts"]
    with pytest.raises(ValueError):
        batch.training_pairs(now=98)
    features, targets = batch.training_pairs(now=99)
    assert len(features) == len(targets) == 50
    assert features == tuple(row.observation for row in batch.episode.transitions)
    assert targets == tuple(label.target for label in batch.labels)
    assert all(
        label.observed_at == 2 * index and label.labels_arrived_at == 2 * index + 1
        for index, label in enumerate(batch.labels)
    )
    assert all(
        label.stage == "close" and label.target == (0.0, 0.0, 0.0, -1.0) for label in batch.labels
    )
    assert all(row.action[3] == 1.0 for row in batch.episode.transitions)
    assert ledger["teacher_attempts"] == queries


def test_should_own_targets_and_states_but_preserve_student_execution_history_in_explicit_target_packet(
    batch,
):
    source = deepcopy(batch)
    owned = corrections.PushCubeCorrectionBatch(source.seed, source.episode, source.labels)
    packet = owned.training_episode(now=99)
    assert packet.episode_id == "push-cube-correction-targets-17"
    assert packet.context == owned.episode.context and packet.role == "train"
    assert packet.spec == owned.episode.spec
    for target, original, label in zip(
        packet.transitions, owned.episode.transitions, owned.labels, strict=True
    ):
        assert target == replace(original, action=label.target)
        assert label.observation == original.observation
        assert label.state is not source.labels[label.step].state
    assert packet.transitions[1].observation[7:11] == (0.125, 0.0, 0.0, 1.0)
    assert packet.transitions[1].observation[-1] == 0.0
    object.__setattr__(source.labels[0].state, "tcp_position", (999.0,) * 3)
    assert owned.labels[0].state.tcp_position == (-0.06, 0.0, 0.12)
    assert owned.episode.transitions[0].action != packet.transitions[0].action


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test", "inner_guard"])
def test_should_refuse_heldout_or_probe_authority_without_relabelling(batch, role):
    source = replace(batch.episode, role=role)
    with pytest.raises(ValueError):
        corrections.PushCubeCorrectionBatch(17, source, batch.labels)


@pytest.mark.parametrize(
    "fault",
    [
        "seed",
        "identity",
        "context",
        "spec",
        "label_step",
        "label_state",
        "label_observation",
        "target",
        "stage",
        "native_action",
        "history",
        "next_history",
    ],
)
def test_should_refuse_corrupted_lineage_labels_or_actual_history_without_extra_queries(
    batch, ledger, fault
):
    value = deepcopy(batch)
    seed, trajectory, labels = value.seed, value.episode, value.labels
    if fault == "seed":
        seed = True
    elif fault == "identity":
        trajectory = replace(trajectory, episode_id="push-cube-teacher-17")
    elif fault == "context":
        trajectory = replace(trajectory, context="pick-cube-train-17")
    elif fault == "spec":
        trajectory = replace(trajectory, spec=replace(trajectory.spec, action_units=("m",) * 4))
    elif fault == "label_step":
        object.__setattr__(labels[0], "step", 1)
    elif fault == "label_state":
        object.__setattr__(labels[0].state, "tcp_position", (0.0, 0.0, 0.0))
    elif fault == "label_observation":
        object.__setattr__(labels[0], "observation", (0.0,) * 12)
    elif fault == "target":
        object.__setattr__(labels[0], "target", (0.0, 0.0, 0.0, 1.0))
    elif fault == "stage":
        object.__setattr__(labels[0], "stage", "unknown")
    else:
        row = trajectory.transitions[1]
        if fault == "native_action":
            row = replace(row, action=(0.1, 0.0, 0.0, 1.0))
        elif fault == "history":
            row = replace(row, observation=row.observation[:7] + (0.0,) * 5)
        else:
            row = replace(row, next_observation=row.next_observation[:7] + (0.0,) * 5)
        # Exercise release revalidation; the episode constructor already refuses
        # broken adjacency, so inject corruption into this owned fixture directly.
        object.__setattr__(
            trajectory, "transitions", (trajectory.transitions[0], row, *trajectory.transitions[2:])
        )
    queries = ledger["teacher_attempts"]
    with pytest.raises(ValueError):
        corrections.PushCubeCorrectionBatch(seed, trajectory, labels)
    assert ledger["teacher_attempts"] == queries


@pytest.mark.parametrize("field", ["observation", "next_observation"])
@pytest.mark.parametrize("flag", [True, 1])
def test_should_refuse_original_non_float_grasp_flags_before_episode_copy(
    batch, monkeypatch, field, flag
):
    value = deepcopy(batch)
    row = value.episode.transitions[0]
    object.__setattr__(row, field, (flag,) + getattr(row, field)[1:])
    monkeypatch.setattr(corrections, "copy_continuous_episode", forbidden)
    with pytest.raises(ValueError, match="float"):
        corrections.PushCubeCorrectionBatch(value.seed, value.episode, value.labels)


@pytest.mark.parametrize("now", [True, 99.0])
def test_should_refuse_non_native_release_ticks_instead_of_coercing_arrival(batch, now):
    with pytest.raises(ValueError):
        batch.training_episode(now=now)


@pytest.mark.parametrize("seed", [19, 20])
def test_should_preserve_other_declared_train_seed_namespaces_without_extra_collection(batch, seed):
    trajectory = replace(
        batch.episode, episode_id=f"push-cube-student-{seed}", context=f"push-cube-train-{seed}"
    )
    result = corrections.PushCubeCorrectionBatch(seed, trajectory, batch.labels)
    assert result.seed == seed and result.training_episode(now=99).context == trajectory.context


@pytest.fixture
def source():
    # Controlled literal arrays exercise native identity/state proofs without an
    # initializer, learned checkpoint claim, RNG or model arithmetic.
    value = object.__new__(bp.NumpyContinuousBCLearner)
    value._policy = bp.ContinuousBCPolicy(
        cube_taskspace_spec(), hidden_dim=2, binary_gripper_axis=3
    )
    parameters = (
        np.zeros((12, 2), np.float64),
        np.zeros(2, np.float64),
        np.zeros((2, 4), np.float64),
        np.zeros(4, np.float64),
    )
    value._state = bp.ContinuousBCSnapshot(
        value._policy,
        parameters,
        (parameters[0].copy(), parameters[1].copy(), parameters[2].copy(), parameters[3].copy()),
        112,
        9832,
    )
    return value


@pytest.fixture
def boundary(monkeypatch, ledger):
    environment = FakePushEnvironment(ledger)
    mode: dict[str, Any] = {}

    def factory():
        ledger["factory_attempts"] += 1
        if mode.get("factory_error"):
            raise RuntimeError("correction factory unavailable")
        ledger["factory_returns"] += 1
        return environment

    def predict(owner, values):
        ledger["prediction_attempts"] += 1
        assert values.shape == (1, 12) and values.dtype == np.float64
        if mode.get("prediction_error"):
            raise RuntimeError("student prediction unavailable")
        result = np.array([[0.123456789, 0.0, 0.0, 1.0]], np.float64)
        if mode.get("shape"):
            result = np.zeros((2, 4), np.float64)
        ledger["prediction_returns"] += 1
        if mode.get("prediction_source"):
            owner._state.parameters[3][0] += 0.125
        if mode.get("source_replacement"):
            owner._state = replace(
                owner._state,
                parameters=tuple(value.copy() for value in owner._state.parameters),
                velocity=tuple(value.copy() for value in owner._state.velocity),
            )
        return result

    monkeypatch.setattr(native, "_make_environment", factory)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "predict", predict)
    return environment, mode


@pytest.mark.parametrize("success", [False, True])
def test_should_export_provisional_labels_on_complete_student_actions_even_when_native_outcome_is_negative(
    source, boundary, ledger, success
):
    environment, _ = boundary
    environment.success = success
    before = bp._snapshot(source._state, source._policy)
    queried_history: list[ActionHistoryState] = []
    decide = teacher.PushCubeTeacher.decide

    def observed(owner, state, *, history):
        queried_history.append(history)
        return decide(owner, state, history=history)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(teacher.PushCubeTeacher, "decide", observed)
        result = collection.collect_push_cube_corrections(source)
    report, batch = result.report, result.corrections
    assert report["complete"] is report["training_release"] is True, report.get("error")
    assert type(batch) is corrections.PushCubeCorrectionBatch and batch is not None
    assert report["definition"] == corrections.PROVENANCE
    assert report["source_unchanged"] is True
    assert report["teacher_query_attempts"] == report["teacher_query_returns"] == 2
    assert report["prediction_attempts"] == report["prediction_returns"] == 2
    assert report["control_attempts"] == report["control_returns"] == 2
    assert (
        report["unreturned_query_possible"]
        is report["unreturned_prediction_possible"]
        is report["unreturned_native_work_unknown"]
        is False
    )
    assert batch.episode.episode_id == "push-cube-student-17"
    assert batch.episode.transitions[-1].reward == float(success)
    assert tuple(row.action for row in batch.episode.transitions) == tuple(
        tuple(float(value) for value in command[0]) for command in environment.commands
    )
    assert batch.episode.transitions[0].action[0] != 0.123456789
    assert all(
        row.action[3] == 1.0 and label.target[3] == -1.0
        for row, label in zip(batch.episode.transitions, batch.labels, strict=True)
    )
    assert queried_history[0] == ActionHistoryState()
    assert queried_history[1].previous_action == batch.episode.transitions[0].action
    assert queried_history[1].closing_count == 0
    assert batch.labels[1].observation[7:11] == batch.episode.transitions[0].action
    assert (
        report["work"]["completed_steps"] == 2
        and report["work"]["completed_control_physics_steps"] == 10
    )
    assert report["work"]["close_attempts"] == report["work"]["close_returns"] == 1
    assert environment.reset_seeds == [17] and environment.close_calls == 1
    assert collection.same_continuous_state(before, bp._snapshot(source._state, source._policy))
    with pytest.raises(ValueError):
        batch.training_pairs(now=2)
    packet = batch.training_episode(now=3)
    assert packet.transitions[1].observation == batch.episode.transitions[1].observation
    assert packet.transitions[0].action == batch.labels[0].target
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "options",
    [
        dict(seed=True),
        dict(seed=23),
        dict(seed=31),
        dict(role="probe_train"),
        dict(role="probe_test"),
        dict(role="final_test"),
        dict(check_boundary=1),
    ],
)
def test_should_refuse_non_train_collection_authority_before_factory_or_teacher(
    source, monkeypatch, options
):
    monkeypatch.setattr(collection, "ManiSkillPushCubeEpisode", forbidden)
    monkeypatch.setattr(collection, "PushCubeTeacher", forbidden)
    with pytest.raises(ValueError):
        collection.collect_push_cube_corrections(source, **options)


@pytest.mark.parametrize(
    "fault",
    [
        "query",
        "prediction",
        "shape",
        "control",
        "cleanup",
        "prediction_source",
        "control_source",
        "source_replacement",
    ],
)
def test_should_preserve_paid_query_prediction_control_or_cleanup_prefix_without_release_or_retry(
    source, boundary, monkeypatch, fault
):
    environment, mode = boundary
    if fault == "query":
        decide = teacher.PushCubeTeacher.decide

        def unavailable(owner, *args, **options):
            decide(owner, *args, **options)
            raise RuntimeError("teacher return unavailable after a paid decision")

        monkeypatch.setattr(teacher.PushCubeTeacher, "decide", unavailable)
    elif fault == "prediction":
        mode["prediction_error"] = True
    elif fault == "shape":
        mode["shape"] = True
    elif fault == "control":
        environment.step_error = RuntimeError("student control unavailable")
    elif fault == "cleanup":
        close = environment.close

        def unavailable_close():
            close()
            environment.counts["close_returns"] -= 1
            raise RuntimeError("correction cleanup unavailable")

        monkeypatch.setattr(environment, "close", unavailable_close)
    elif fault in ("prediction_source", "source_replacement"):
        mode[fault] = True
    else:
        environment.after_step = lambda: source._state.parameters[3].fill(0.125)
    result = collection.collect_push_cube_corrections(source)
    report = result.report
    assert result.corrections is None and report["complete"] is report["training_release"] is False
    assert environment.close_calls == 1
    if fault == "query":
        assert report["teacher_query_attempts"] == 1 and report["teacher_query_returns"] == 0
        assert report["unreturned_query_possible"] is True
        assert report["prediction_attempts"] == report["control_attempts"] == 0
        assert len(report["states"]) == 1 and report["labels"] == []
    else:
        n = 2 if fault == "cleanup" else 1
        assert report["teacher_query_attempts"] == report["teacher_query_returns"] == n
        assert len(report["labels"]) == n and len(report["states"]) == n
        assert report["prediction_attempts"] == n
        assert report["prediction_returns"] == (0 if fault == "prediction" else n)
        if fault in ("prediction_source", "source_replacement"):
            assert report["prediction_outputs"] == []
            assert report["refused_return"]["stage"] == "prediction"
            assert report["refused_return"]["value"]["shape"] == [1, 4]
        else:
            assert len(report["prediction_outputs"]) == (0 if fault == "prediction" else n)
    if fault in ("prediction", "shape", "prediction_source", "source_replacement", "query"):
        assert report["control_attempts"] == report["control_returns"] == 0
    elif fault == "control":
        assert report["control_attempts"] == 1 and report["control_returns"] == 0
        assert report["work"]["step_attempts"] == 1
        assert report["work"]["possible_failed_physics_steps"] == 5
        assert (
            len(report["push_cube_attempted_actions"]) == 1
            and report["push_cube_returned_actions"] == ()
        )
    else:
        n = 2 if fault == "cleanup" else 1
        assert report["control_attempts"] == report["control_returns"] == n
        if fault == "control_source":
            assert report["transitions"] == []
            assert report["refused_return"]["stage"] == "control"
            assert report["refused_return"]["value"]["action"][3] == 1.0
        else:
            assert len(report["transitions"]) == n
    if fault in ("prediction_source", "control_source"):
        assert report["source_unchanged"] is False
        assert source._state.parameters[3][0] == 0.125
    if fault == "source_replacement":
        assert report["source_unchanged"] is False
        assert source._state.parameters[3][0] == 0.0


def test_should_refuse_changed_original_label_after_paid_second_prediction_before_another_control(
    source, boundary, monkeypatch
):
    labels: list[Any] = []
    initialize = corrections.PushCubeCorrectionLabel.__post_init__
    predict = bp.NumpyContinuousBCLearner.predict
    calls: list[bool] = []

    def initialized(label):
        initialize(label)
        labels.append(label)

    def predicted(owner, values):
        returned = predict(owner, values)
        calls.append(True)
        if len(calls) == 2:
            # A valid closed-fist align command still cannot replace paid evidence.
            object.__setattr__(labels[0], "target", (0.125, 0.0, 0.0, -1.0))
        return returned

    monkeypatch.setattr(corrections.PushCubeCorrectionLabel, "__post_init__", initialized)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "predict", predicted)
    result = collection.collect_push_cube_corrections(source)
    report = result.report
    assert result.corrections is None and report["complete"] is False
    assert len(calls) == report["prediction_returns"] == 2
    assert report["control_returns"] == 1
    assert len(report["prediction_outputs"]) == 1
    assert report["refused_return"]["stage"] == "prediction"
    assert report["refused_return"]["value"]["shape"] == [1, 4]
    assert report["labels"][0]["target"][-1] == -1.0
    assert tuple(report["labels"][0]["target"]) != (0.125, 0.0, 0.0, -1.0)
    assert "evidence" in report["error"]


@pytest.mark.parametrize("fault", ["boundary", "clock"])
def test_should_refuse_post_cleanup_source_mutation_before_final_release(
    source, boundary, monkeypatch, fault
):
    environment, _ = boundary
    changed: list[bool] = []

    def mutate():
        if environment.close_calls and not changed:
            source._state.parameters[3][0] = 0.125
            changed.append(True)

    if fault == "clock":
        clock = collection.monotonic

        def observed_clock():
            value = clock()
            mutate()
            return value

        monkeypatch.setattr(collection, "monotonic", observed_clock)
    result = collection.collect_push_cube_corrections(
        source, check_boundary=mutate if fault == "boundary" else None
    )
    assert changed == [True] and result.corrections is None and result.report["complete"] is False
    assert result.report["source_unchanged"] is False
    assert result.report["control_returns"] == 2 and environment.close_calls == 1


def test_should_recheck_source_after_native_observation_read_before_first_query_or_prediction(
    source, boundary, monkeypatch
):
    original = vars(native.ManiSkillPushCubeEpisode)["observation"].fget

    def observed(port):
        value = original(port)
        source._state.parameters[3][0] = 0.125
        return value

    monkeypatch.setattr(native.ManiSkillPushCubeEpisode, "observation", property(observed))
    result = collection.collect_push_cube_corrections(source)
    assert result.corrections is None and result.report["complete"] is False
    assert (
        result.report["teacher_query_attempts"]
        == result.report["prediction_attempts"]
        == result.report["control_attempts"]
        == 0
    )
    assert boundary[0].commands == [] and boundary[0].close_calls == 1


def test_should_preserve_unknown_failed_construction_without_teacher_prediction_or_cleanup_retry(
    source, boundary
):
    environment, mode = boundary
    mode["factory_error"] = True
    result = collection.collect_push_cube_corrections(source)
    assert result.corrections is None and result.report["complete"] is False
    assert result.report["unreturned_native_work_unknown"] is True
    assert (
        result.report["teacher_query_attempts"]
        == result.report["prediction_attempts"]
        == result.report["control_attempts"]
        == 0
    )
    assert environment.reset_seeds == [] and environment.close_calls == 0


def test_should_refuse_valid_prior_label_mutation_from_paid_native_work_getter_before_more_queries(
    source, boundary, monkeypatch
):
    environment, _ = boundary
    labels: list[Any] = []
    changed: list[bool] = []
    initialize = corrections.PushCubeCorrectionLabel.__post_init__
    work = vars(native.ManiSkillPushCubeEpisode)["work"].fget

    def initialized(label):
        initialize(label)
        labels.append(label)

    def observed(port):
        value = work(port)
        if len(environment.commands) == 1 and labels and not changed:
            object.__setattr__(labels[0], "target", (0.125, 0.0, 0.0, -1.0))
            changed.append(True)
        return value

    monkeypatch.setattr(corrections.PushCubeCorrectionLabel, "__post_init__", initialized)
    monkeypatch.setattr(native.ManiSkillPushCubeEpisode, "work", property(observed))
    result = collection.collect_push_cube_corrections(source)
    assert changed == [True] and result.corrections is None and result.report["complete"] is False
    assert (
        result.report["teacher_query_returns"]
        == result.report["prediction_returns"]
        == result.report["control_returns"]
        == 1
    )
    assert environment.close_calls == 1
    assert tuple(result.report["labels"][0]["target"]) != (0.125, 0.0, 0.0, -1.0)


def test_should_refuse_valid_candidate_batch_mutation_during_cleanup_before_export(
    source, boundary, monkeypatch
):
    environment, _ = boundary
    batches: list[Any] = []
    initialize = corrections.PushCubeCorrectionBatch.__post_init__
    close = environment.close

    def initialized(batch):
        initialize(batch)
        batches.append(batch)

    def changed_close():
        close()
        assert batches
        object.__setattr__(batches[0].labels[0], "target", (0.125, 0.0, 0.0, -1.0))

    monkeypatch.setattr(corrections.PushCubeCorrectionBatch, "__post_init__", initialized)
    monkeypatch.setattr(environment, "close", changed_close)
    result = collection.collect_push_cube_corrections(source)
    assert batches and result.corrections is None and result.report["complete"] is False
    assert result.report["training_release"] is False
    assert result.report["control_returns"] == 2 and environment.close_calls == 1
    assert tuple(result.report["labels"][0]["target"]) != (0.125, 0.0, 0.0, -1.0)


def test_should_refuse_late_teacher_phase_and_count_change_after_query_before_prediction(
    source, boundary, monkeypatch
):
    queried: list[Any] = []
    changed: list[bool] = []
    decide = teacher.PushCubeTeacher.decide
    observation = vars(native.ManiSkillPushCubeEpisode)["observation"].fget

    def decided(owner, *args, **options):
        returned = decide(owner, *args, **options)
        queried.append(owner)
        return returned

    def observed(port):
        returned = observation(port)
        if queried and not changed:
            queried[0]._steps = 2
            queried[0]._phase = "hold"
            changed.append(True)
        return returned

    monkeypatch.setattr(teacher.PushCubeTeacher, "decide", decided)
    monkeypatch.setattr(native.ManiSkillPushCubeEpisode, "observation", property(observed))
    result = collection.collect_push_cube_corrections(source)
    report = result.report
    assert changed == [True] and result.corrections is None and report["complete"] is False
    assert report["teacher_query_attempts"] == report["teacher_query_returns"] == 1
    assert report["prediction_attempts"] == report["control_attempts"] == 0
    assert len(report["states"]) == 1 and report["labels"] == []
    assert report["refused_return"]["stage"] == "teacher"
    assert report["refused_return"]["value"]["stage"] == "align"
    assert boundary[0].commands == [] and boundary[0].close_calls == 1
    assert queried[0].steps == 2 and queried[0].phase == "hold"
