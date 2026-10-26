"""Reusable joint TRAIN supervision: arrival, provenance, bounded work and refusal.

Only the module fixture initializes a model: D12/H2/A4 seed17,32normal draws.
All native cases fork it without RNG. Synthetic labels require no teacher/SDK.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from src.adapters import joint_readiness_corrections as runtime
from src.adapters import joint_readiness_data as data
from src.adapters import numpy_continuous_bc as bp
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.curated_readiness_corrections import CuratedReadinessBatch, CuratedReadinessLabel
from src.core.pick_cube_readiness_corrections import (
    GoalConsistentObservedGraspCorrectionBatch,
    ReadinessCorrectionLabel,
)
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.core.readiness_stage_diagnostics import ReadinessStageBinding


UNITS = ("normalized_root_dx", "normalized_root_dy", "normalized_root_dz", "normalized_gripper")


def spec(dimension=42):
    return ContinuousActionSpec(dimension, UNITS, (-1.0,) * 4, (1.0,) * 4)


def episode(kind, rows=2):
    reference = kind == "a"
    seed = 19 if kind == "b" else 17
    identity = (
        f"pick-cube-observed-grasp-teacher-{seed}" if reference else f"pick-cube-smoke-{seed}"
    )
    action = (0.25, 0.0, 0.0, 1.0) if reference else (-0.25, 0.0, 0.0, -1.0)
    frames = []
    for i in range(rows + 1):
        frame = [0.0] * 42
        frame[36] = {"a": 0.01, "b": -0.02, "c": 0.03}[kind] + 0.001 * i
        frame[41] = 0.05
        frames.append(tuple(frame))
    return ContinuousEpisode(
        identity,
        "pick-cube-train-19" if seed == 19 else "pick-cube",
        "train",
        spec(),
        tuple(
            ContinuousTransition(i, frames[i], action, 0.0, frames[i + 1], False, i == rows - 1)
            for i in range(rows)
        ),
    )


def inputs(*, a_rows=2, b_rows=2, c_rows=2, c_start=100):
    a, b, c = episode("a", a_rows), episode("b", b_rows), episode("c", c_rows)
    reference = ReadinessStageBinding(a, ("align",) * a_rows)
    corrections = GoalConsistentObservedGraspCorrectionBatch(
        19,
        b,
        tuple(
            ReadinessCorrectionLabel(i, row.observation, (-0.25, 0.0, 0.0, 1.0), "align")
            for i, row in enumerate(b.transitions)
        ),
        (False,) * b_rows,
    )
    curated = CuratedReadinessBatch(
        17,
        c,
        tuple(
            CuratedReadinessLabel(
                i,
                row.observation,
                (0.125, 0.25, 0.0, 1.0),
                "align",
                None,
                c_start + 2 * i,
                c_start + 2 * i + 1,
            )
            for i, row in enumerate(c.transitions)
        ),
        c_start,
    )
    return reference, corrections, curated


def forbidden(*args, **kwargs):
    pytest.fail("refused input reached projection, fork, or numerical training")


def assert_state(actual, expected):
    assert actual.policy == expected.policy
    assert (actual.completed_updates, actual.training_examples) == (
        expected.completed_updates,
        expected.training_examples,
    )
    for a, b in zip(
        actual.parameters + actual.velocity, expected.parameters + expected.velocity, strict=True
    ):
        np.testing.assert_array_equal(a, b)


@pytest.fixture(scope="module")
def ledger():
    counts = dict(
        initializations=0,
        normal_draws=0,
        GN_attempts=0,
        GN_rows_attempted=0,
        GN_returns=0,
        solve_attempts=0,
        solve_completed=0,
        augmented_rows=0,
        prediction_attempts=0,
        prediction_rows_attempted=0,
        prediction_returns=0,
        prediction_rows_returned=0,
        SGD=0,
        fits=0,
        teacher_queries=0,
        simulator_calls=0,
    )
    original_init = bp.NumpyContinuousBCLearner.__init__
    original_train = bp.NumpyContinuousBCLearner.train_head_gauss_newton_batch
    original_predict = bp.NumpyContinuousBCLearner.predict

    def initialize(owner, policy):
        counts["initializations"] += 1
        counts["normal_draws"] += (
            policy.spec.observation_dim * policy.hidden_dim
            + policy.hidden_dim * len(policy.spec.action_units)
        )
        return original_init(owner, policy)

    def train(owner, features, targets):
        counts["GN_attempts"] += 1
        counts["GN_rows_attempted"] += len(features)
        try:
            returned = original_train(owner, features, targets)
        except Exception as error:
            work = getattr(error, "work", {})
            counts["solve_attempts"] += work.get("solve_attempts", 0)
            counts["solve_completed"] += work.get("solve_completed", 0)
            counts["augmented_rows"] += work.get("solve_rows", 0)
            raise
        counts["GN_returns"] += 1
        counts["solve_attempts"] += returned.work["solve_attempts"]
        counts["solve_completed"] += returned.work["solve_completed"]
        counts["augmented_rows"] += returned.work["solve_rows"]
        return returned

    def predict(owner, features):
        counts["prediction_attempts"] += 1
        counts["prediction_rows_attempted"] += len(features)
        result = original_predict(owner, features)
        counts["prediction_returns"] += 1
        counts["prediction_rows_returned"] += len(result)
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bp.NumpyContinuousBCLearner, "__init__", initialize)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", train)
        patch.setattr(bp.NumpyContinuousBCLearner, "predict", predict)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", forbidden)
        yield counts
    if os.environ.get("JRA_LEDGER"):
        with Path(os.environ["JRA_LEDGER"]).open("x", encoding="utf8") as output:
            output.write(json.dumps(counts, sort_keys=True) + "\n")
    assert counts["initializations"] <= 8 and counts["normal_draws"] <= 256
    assert counts["GN_attempts"] <= 160 and counts["GN_rows_attempted"] <= 960
    assert counts["solve_attempts"] <= 640 and counts["augmented_rows"] <= 5760
    assert counts["prediction_attempts"] <= 200 and counts["prediction_rows_attempted"] <= 1200


@pytest.fixture(scope="module")
def owner(ledger):
    return bp.NumpyContinuousBCLearner(bp.ContinuousBCPolicy(spec(12), hidden_dim=2, seed=17))


def test_should_keep_all_targets_commands_history_groups_and_original_clocks():
    original = inputs()
    seals = tuple(asdict(value) for value in original)
    prepared = data.prepare_joint_readiness_data(*original, now=103)
    assert prepared.spec == spec(12)
    assert prepared.groups == (("a", 0, 2), ("b", 2, 4), ("curated_a", 4, 6))
    assert len(prepared.rows) == prepared.work["rows"] == 6
    assert prepared.work["observation_projections"] == 12
    assert prepared.work["source_values_validated"] == 504
    assert prepared.work["projected_values"] == 144
    assert prepared.work["projected_numeric_bytes"] == 1152
    for offset, family in (
        (0, "qualified_demonstration"),
        (2, "provisional_correction"),
        (4, "curated_provisional_correction"),
    ):
        first, second = prepared.rows[offset : offset + 2]
        assert first["source_kind"] == second["source_kind"] == family
        assert first["features"][7:] == [0.0] * 5
        expected_action = list(original[offset // 2].episode.transitions[0].action)
        assert second["features"][7:11] == first["actual_command"] == expected_action
        assert second["features"][11] == (0.0 if offset == 0 else pytest.approx(1 / 6))
        if offset:
            assert first["target"] != first["actual_command"]
    assert [(row["observed_at"], row["labels_arrived_at"]) for row in prepared.rows] == [
        (None, None),
        (None, None),
        (0, 1),
        (2, 3),
        (None, 101),
        (None, 103),
    ]
    assert [
        (row["source_observed_at"], row["queried_at"], row["arrived_at"])
        for row in prepared.rows[4:]
    ] == [(None, 100, 101), (None, 102, 103)]
    assert all(row["role"] == "train" for row in prepared.rows)
    assert [prepared.rows[i]["provenance"] for i in (0, 2, 4)] == [
        None,
        "provisional_geometry_observed_grasp_goal_consistent_v2",
        "provisional_saved_state42_goal_consistent_v1",
    ]
    assert [prepared.rows[i]["episode_id"] for i in (0, 2, 4)] == [
        "pick-cube-observed-grasp-teacher-17",
        "pick-cube-smoke-19",
        "pick-cube-smoke-17",
    ]
    prepared.rows[4]["target"][0] = 99.0
    assert tuple(asdict(value) for value in original) == seals


@pytest.mark.parametrize("family", ["legacy", "curated"])
def test_should_refuse_either_unarrived_family_before_any_projection(monkeypatch, family):
    values, now = (inputs(c_rows=1, c_start=0), 2) if family == "legacy" else (inputs(), 102)
    monkeypatch.setattr(data, "action_history_training_episode", forbidden)
    monkeypatch.setattr(data, "project_curated_training_pairs", forbidden)
    with pytest.raises(ValueError, match="arriv"):
        data.prepare_joint_readiness_data(*values, now=now)


def test_should_preserve_known_original_ticks_and_exact_raw_evidence():
    reference, corrections, curated = inputs()
    curated = replace(
        curated,
        labels=tuple(
            replace(label, source_observed_at=90 + i) for i, label in enumerate(curated.labels)
        ),
    )
    prepared = data.prepare_joint_readiness_data(reference, corrections, curated, now=103)
    for offset, batch in ((0, reference), (2, corrections), (4, curated)):
        for index, row in enumerate(prepared.rows[offset : offset + 2]):
            assert row["source_observation"] == list(batch.episode.transitions[index].observation)
    assert [row["source_observed_at"] for row in prepared.rows[4:]] == [90, 91]
    assert [row["observed_at"] for row in prepared.rows[4:]] == [90, 91]
    assert prepared.work["source_observation_values"] == 252
    assert prepared.work["source_observation_numeric_bytes"] == 2016


@pytest.mark.parametrize(
    "fault",
    ["reference_role", "correction_role", "curated_role", "flag", "spec", "oversize", "recipe"],
)
def test_should_refuse_invalid_joint_sources_before_projection(monkeypatch, fault):
    reference, corrections, curated = inputs()
    options: dict[str, Any] = {}
    if fault.endswith("role"):
        selected = {
            "reference_role": reference,
            "correction_role": corrections,
            "curated_role": curated,
        }[fault]
        object.__setattr__(selected.episode, "role", "probe_test")
    elif fault == "flag":
        values = list(corrections.episode.transitions[0].observation)
        values[18] = 0
        object.__setattr__(corrections.episode.transitions[0], "observation", tuple(values))
    elif fault == "spec":
        object.__setattr__(curated.episode.spec, "upper_bounds", (2.0,) * 4)
    elif fault == "oversize":
        reference, corrections, curated = inputs(a_rows=50, b_rows=50, c_rows=50)
    else:
        options["recipe"] = PickCubeTaskSpaceHistoryRecipe(mode="zero")
    monkeypatch.setattr(data, "action_history_training_episode", forbidden)
    monkeypatch.setattr(data, "project_curated_training_pairs", forbidden)
    with pytest.raises(ValueError):
        data.prepare_joint_readiness_data(reference, corrections, curated, now=199, **options)


@pytest.fixture(scope="module")
def successful(owner):
    source = owner.fork()
    initial = source.snapshot_state()
    # Exercise preservation of existing optimizer velocity without spending an
    # SGD step or fabricating historical training/exposure counters.
    source.restore_state(
        replace(initial, velocity=tuple(np.full_like(v, 0.03125) for v in initial.velocity))
    )
    initial = source.snapshot_state()
    original_inputs = inputs()
    evidence = tuple(asdict(value) for value in original_inputs)
    captured: dict[str, Any] = {}
    original_factory = runtime.ActorShadowRuntime

    def factory(model, **options):
        made = original_factory(model, **options)
        captured.update(runtime=made, budget=options["budget"])
        return made

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runtime, "ActorShadowRuntime", factory)
        result = runtime.run_joint_readiness_corrections(source, *original_inputs, now=103)
    return result, source, initial, captured, original_inputs, evidence


def test_should_complete_eight_native_updates_with_frozen_hidden_and_unchanged_source(successful):
    result, source, initial, captured, values, evidence = successful
    assert result.report["complete"] is True, result.report.get("error")
    assert type(result.candidate) is bp.NumpyContinuousBCLearner
    final = result.candidate.snapshot_state()
    assert final.policy == initial.policy
    assert (final.completed_updates, final.training_examples) == (8, 48)
    for index in (0, 1):
        np.testing.assert_array_equal(final.parameters[index], initial.parameters[index])
        np.testing.assert_array_equal(final.velocity[index], initial.velocity[index])
    for index in (2, 3):
        np.testing.assert_array_equal(final.velocity[index], np.zeros_like(final.velocity[index]))
    assert any(not np.array_equal(final.parameters[i], initial.parameters[i]) for i in (2, 3))
    assert_state(source.snapshot_state(), initial)
    assert_state(captured["runtime"].actor.snapshot_state().state, initial)
    assert result.candidate is not source
    assert result.candidate is not captured["runtime"]._candidate._learner
    assert tuple(asdict(value) for value in values) == evidence
    assert captured["budget"].updates_completed == 8
    assert captured["budget"].replay_examples_completed == 0


def test_should_account_equal_row_exposure_receipts_residuals_and_solver_work(successful):
    result, _, _, _, _, _ = successful
    report = result.report
    assert report["source_updates"] == report["source_training_rows"] == 0
    assert report["candidate_updates"] == report["progress"]["updates_completed"] == 8
    assert report["candidate_training_rows"] == report["attempted_training_rows"] == 48
    assert report["group_training_exposures"] == dict(a=16, b=16, curated_a=16)
    assert report["derived_batch"]["qualified_demonstration"] is False
    assert report["derived_batch"]["observed_at"] == report["derived_batch"]["arrived_at"] == 103
    assert (
        report["native_updates"]
        == report["sealed_native_updates"]
        == report["validated_native_prefix"]
    )
    assert report["applied_updates"] == report["sealed_applied_updates"]
    assert len(report["native_updates"]) == len(report["applied_updates"]) == 8
    for i, (native, receipt) in enumerate(
        zip(report["native_updates"], report["applied_updates"], strict=True)
    ):
        assert native["training_rows"] == 6
        assert native["loss_after"] < native["diagnostic"]["value"]
        assert receipt["diagnostic"] == native["diagnostic"]
        assert receipt["update_number"] == i + 1
        assert (receipt["observed_at"], receipt["arrived_at"], receipt["applied_at"]) == (
            103,
            103,
            103,
        )
    work = report["work"]
    assert work["optimizer_updates"] == work["head_gauss_newton_updates"] == 8
    assert work["training_rows"] == 48 and work["sgd_updates"] == work["fits"] == 0
    assert work["native_update_counts"]["solve_completed"] == 32
    assert work["native_update_counts"]["solve_rows"] == 288
    assert work["inbox_feature_target_numeric_bytes"] == 6144
    for when in ("before", "after"):
        assert set(report["residuals"][when]) == {"a", "b", "curated_a"}
        for item in report["residuals"][when].values():
            assert item["prediction_rows"] == 2 and len(item["predictions"]) == 2
            assert item["prediction_returned"] is item["residual_returned"] is True
            assert item["result"]["overall"]["rows"] == 2
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "fault", ["foreign", "binary", "normalization", "width", "early", "bad_now"]
)
def test_should_refuse_bad_source_or_unarrived_data_before_any_fork(owner, monkeypatch, fault):
    source: Any = owner.fork()
    now: Any = 103
    if fault == "foreign":
        source = object()
    elif fault in ("binary", "width", "normalization"):
        changes: dict[str, Any] = (
            dict(binary_gripper_axis=3) if fault == "binary" else dict(hidden_dim=65)
        )
        if fault == "normalization":
            from src.core.observation_standardization import ObservationStandardization

            changes = dict(
                observation_standardization=ObservationStandardization((0.0,) * 12, (1.0,) * 12, 1)
            )
        policy = replace(source._policy, **changes)
        source._policy, source._state = policy, replace(source._state, policy=policy)
    elif fault == "early":
        now = 102
    else:
        now = True
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forbidden)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", forbidden)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "predict", forbidden)
    monkeypatch.setattr(runtime, "RecordedContinuousHead", forbidden)
    with pytest.raises(ValueError):
        runtime.run_joint_readiness_corrections(source, *inputs(), now=now)


def test_should_preserve_returned_native_work_when_facade_raises_before_receipt(owner, monkeypatch):
    source = owner.fork()
    initial = source.snapshot_state()
    original = runtime.RecordedContinuousHead.train_batch
    calls = 0

    def fail_second(port, features, targets):
        nonlocal calls
        calls += 1
        result = original(port, features, targets)
        if calls == 2:
            raise RuntimeError("after second native return")
        return result

    monkeypatch.setattr(runtime.RecordedContinuousHead, "train_batch", fail_second)
    result = runtime.run_joint_readiness_corrections(source, *inputs(), now=103)
    assert result.candidate is None and result.report["complete"] is False
    assert "after second native return" in result.report["error"]
    assert calls == result.report["update_attempts"] == 2
    assert len(result.report["native_updates"]) == len(result.report["sealed_native_updates"]) == 2
    assert (
        len(result.report["applied_updates"]) == result.report["progress"]["updates_completed"] == 1
    )
    assert result.report["attempted_training_rows"] == 12
    assert_state(source.snapshot_state(), initial)


@pytest.mark.parametrize(
    "fault",
    [
        "native_record",
        "receipt",
        "budget_start",
        "budget_clock",
        "logical_clock",
        "latest_state",
        "public_snapshot",
        "source_payload",
        "source_state",
        "input_evidence",
        "owned_rows",
        "retained_packet",
    ],
)
def test_should_refuse_changed_paid_evidence_before_another_update(owner, monkeypatch, fault):
    source = owner.fork()
    initial = source.snapshot_state()
    values = inputs()
    original_capture = runtime._JointSession.capture
    mutation: dict[str, Any] = {}

    def capture(session):
        original_capture(session)
        if session.committed != 1 or mutation:
            return
        mutation["session"] = session
        if fault == "native_record":
            session.native[0]["work"]["solve_rows"] += 1
        elif fault == "receipt":
            object.__setattr__(session.receipt_objects[0], "applied_at", 0)
        elif fault == "budget_start":
            session.budget.started_at += 1.0
        elif fault == "budget_clock":
            session.budget.clock = lambda: 0.0
        elif fault == "logical_clock":
            session.clock.advance_to(104)
        elif fault == "latest_state":
            session.raw_candidate._state.parameters[3][0] += 0.125
        elif fault == "public_snapshot":
            original_snapshot = session.raw_candidate.snapshot_state

            def mutate_during_observation():
                returned = original_snapshot()
                session.raw_candidate._state.parameters[3][0] += 0.125
                return returned

            monkeypatch.setattr(session.raw_candidate, "snapshot_state", mutate_during_observation)
        elif fault == "source_payload":
            source._state = deepcopy(source._state)
            mutation["payload"] = source._state
        elif fault == "input_evidence":
            label = values[2].labels[0]
            object.__setattr__(label, "target", (0.25, 0.25, 0.0, 1.0))
        elif fault == "owned_rows":
            session.data.rows[0]["target"][0] += 0.125
        elif fault == "retained_packet":
            event = next(iter(session.inbox._experiences.values()))
            object.__setattr__(event, "features", np.zeros_like(event.features))
        else:
            source._state.parameters[3][0] += 0.125

    monkeypatch.setattr(runtime._JointSession, "capture", capture)
    result = runtime.run_joint_readiness_corrections(source, *values, now=103)
    assert mutation, "fault never reached its paid boundary"
    assert result.candidate is None and result.report["complete"] is False
    assert result.report["update_attempts"] == 1
    assert len(result.report["sealed_native_updates"]) == 1
    assert mutation["session"].budget.updates_completed == 1
    if fault == "source_state":
        assert source._state.parameters[3][0] == initial.parameters[3][0] + 0.125
    elif fault == "source_payload":
        assert source._state is mutation["payload"]
    else:
        assert_state(source.snapshot_state(), initial)
    if fault == "input_evidence":
        assert values[2].labels[0].target == (0.25, 0.25, 0.0, 1.0)


def test_should_keep_eight_paid_receipts_when_final_candidate_state_is_corrupted(
    owner, monkeypatch
):
    source = owner.fork()
    original_capture = runtime._JointSession.capture
    changed: list[Any] = []

    def capture(session):
        original_capture(session)
        if session.committed == 8 and not changed:
            changed.append(session)
            session.raw_candidate._state.parameters[3][0] += 0.125

    monkeypatch.setattr(runtime._JointSession, "capture", capture)
    result = runtime.run_joint_readiness_corrections(source, *inputs(), now=103)
    assert changed and result.candidate is None and result.report["complete"] is False
    assert len(result.report["native_updates"]) == len(result.report["applied_updates"]) == 8
    assert result.report["progress"]["updates_completed"] == 8


def test_should_retain_failed_native_work_without_admitting_a_third_update(owner, monkeypatch):
    source = owner.fork()
    original = runtime.RecordedContinuousHead.train_batch
    calls = 0
    failure_work = dict(
        published_updates=0,
        solve_attempts=0,
        solve_completed=0,
        failure_work_may_be_unobserved=True,
    )

    def fail_second(port, features, targets):
        nonlocal calls
        calls += 1
        if calls == 2:
            error: Any = RuntimeError("synthetic native-work failure")
            error.work = dict(failure_work)
            raise error
        return original(port, features, targets)

    monkeypatch.setattr(runtime.RecordedContinuousHead, "train_batch", fail_second)
    result = runtime.run_joint_readiness_corrections(source, *inputs(), now=103)
    assert result.candidate is None and result.report["complete"] is False
    assert calls == result.report["update_attempts"] == 2
    assert len(result.report["native_updates"]) == len(result.report["applied_updates"]) == 1
    assert result.report["failed_native_work"] == failure_work


def test_should_refuse_detached_candidate_with_a_forged_public_snapshot(owner, monkeypatch):
    source = owner.fork()
    original_finish = runtime._JointSession.finish
    escaped = []

    def finish(session):
        original_fork = session.source.fork

        def fork_with_stale_snapshot():
            child = original_fork()
            stale = child.snapshot_state()
            monkeypatch.setattr(child, "snapshot_state", lambda: stale)
            escaped.append(child)
            return child

        monkeypatch.setattr(session.source, "fork", fork_with_stale_snapshot)
        return original_finish(session)

    monkeypatch.setattr(runtime._JointSession, "finish", finish)
    result = runtime.run_joint_readiness_corrections(source, *inputs(), now=103)
    assert escaped and result.candidate is None and result.report["complete"] is False
    assert "public state" in result.report["error"]
    assert len(result.report["native_updates"]) == len(result.report["applied_updates"]) == 8


@pytest.mark.parametrize("fault", ["source", "native_prefix", "report_rows"])
def test_should_refuse_final_snapshot_side_effects_on_previously_checked_evidence(
    owner, monkeypatch, fault
):
    source = owner.fork()
    initial = source.snapshot_state()
    original_finish = runtime._JointSession.finish
    armed = dict(finished=False, changed=False)

    def finish(session):
        original_fork = session.source.fork

        def fork_with_observer():
            child = original_fork()
            original_snapshot = child.snapshot_state

            def snapshot():
                returned = original_snapshot()
                # Exercise the detached getter after finish's own checks, when
                # source/prefix/report evidence has already been inspected.
                if armed["finished"] and not armed["changed"]:
                    armed["changed"] = True
                    if fault == "source":
                        source._state.parameters[3][0] += 0.125
                    elif fault == "native_prefix":
                        session.native[0]["work"]["solve_rows"] += 1
                    else:
                        session.report["rows"][0]["target"][0] += 0.125
                return returned

            monkeypatch.setattr(child, "snapshot_state", snapshot)
            return child

        monkeypatch.setattr(session.source, "fork", fork_with_observer)
        candidate = original_finish(session)
        armed["finished"] = True
        return candidate

    monkeypatch.setattr(runtime._JointSession, "finish", finish)
    result = runtime.run_joint_readiness_corrections(source, *inputs(), now=103)
    assert armed["changed"] and result.candidate is None and result.report["complete"] is False
    assert len(result.report["native_updates"]) == len(result.report["applied_updates"]) == 8
    if fault == "source":
        assert source._state.parameters[3][0] == initial.parameters[3][0] + 0.125


def test_should_refuse_later_owner_getter_mutating_an_earlier_source_before_model_work(
    owner, monkeypatch
):
    source = owner.fork()
    initial = source.snapshot_state()
    original_factory = runtime.ActorShadowRuntime
    changed: list[Any] = []

    def factory(model, **options):
        made = original_factory(model, **options)
        candidate = cast(runtime.RecordedContinuousHead, made._candidate)._learner
        original_snapshot = candidate.snapshot_state

        def snapshot():
            returned = original_snapshot()
            if not changed:
                changed.append(True)
                source._state.parameters[3][0] += 0.125
            return returned

        monkeypatch.setattr(candidate, "snapshot_state", snapshot)
        return made

    monkeypatch.setattr(runtime, "ActorShadowRuntime", factory)
    result = runtime.run_joint_readiness_corrections(source, *inputs(), now=103)
    assert changed and result.candidate is None and result.report["complete"] is False
    assert result.report["update_attempts"] == 0
    assert result.report["native_updates"] == []
    assert source._state.parameters[3][0] == initial.parameters[3][0] + 0.125


@pytest.mark.parametrize("fault", ["target", "recipe"])
def test_should_refuse_input_changed_by_source_admission_observer_before_projection(
    owner, monkeypatch, fault
):
    source = owner.fork()
    values = inputs()
    recipe = PickCubeTaskSpaceHistoryRecipe()
    original_snapshot = source.snapshot_state

    def snapshot():
        returned = original_snapshot()
        if fault == "target":
            object.__setattr__(values[2].labels[0], "target", (0.25, 0.25, 0.0, 1.0))
        else:
            object.__setattr__(recipe, "mode", "zero")
        return returned

    monkeypatch.setattr(source, "snapshot_state", snapshot)
    monkeypatch.setattr(runtime, "prepare_joint_readiness_data", forbidden)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forbidden)
    with pytest.raises((ValueError, RuntimeError), match="evidence"):
        runtime.run_joint_readiness_corrections(
            source, *values, now=103, action_history_recipe=recipe
        )
    if fault == "target":
        assert values[2].labels[0].target == (0.25, 0.25, 0.0, 1.0)
    else:
        assert recipe.mode == "zero"


def test_should_preserve_actual_paid_prefix_when_a_later_train_changes_copied_report(
    owner, monkeypatch
):
    source = owner.fork()
    original_apply = runtime._JointSession.apply_one
    original_train = runtime.RecordedContinuousHead.train_batch
    active = {}
    calls = 0
    changed: list[Any] = []

    def apply(session, index):
        active["session"] = session
        return original_apply(session, index)

    def train(port, features, targets):
        nonlocal calls
        calls += 1
        result = original_train(port, features, targets)
        if calls == 2:
            copied = active["session"].report["native_updates"][0]["diagnostic"]
            copied["value"] += 0.125
            changed.append(copied["value"])
        return result

    monkeypatch.setattr(runtime._JointSession, "apply_one", apply)
    monkeypatch.setattr(runtime.RecordedContinuousHead, "train_batch", train)
    result = runtime.run_joint_readiness_corrections(source, *inputs(), now=103)
    assert changed and result.candidate is None and result.report["complete"] is False
    assert calls == result.report["update_attempts"] == 2
    assert len(result.report["native_updates"]) == len(result.report["sealed_native_updates"]) == 2
    assert (
        len(result.report["applied_updates"]) == result.report["progress"]["updates_completed"] == 2
    )
    corrupted = result.report["corrupted_copied_evidence"]
    assert corrupted["native_updates"][0]["diagnostic"]["value"] == changed[0]
    assert result.report["native_updates"][0]["diagnostic"]["value"] != changed[0]


def test_should_keep_returned_prediction_cost_when_observer_changes_source(owner, monkeypatch):
    source = owner.fork()
    initial = source.snapshot_state()
    original_predict = runtime.RecordedContinuousHead.predict
    calls = 0

    def predict(port, features):
        nonlocal calls
        calls += 1
        result = original_predict(port, features)
        source._state.parameters[3][0] += 0.125
        return result

    monkeypatch.setattr(runtime.RecordedContinuousHead, "predict", predict)
    result = runtime.run_joint_readiness_corrections(source, *inputs(), now=103)
    assert result.candidate is None and result.report["complete"] is False
    item = result.report["residuals"]["before"]["a"]
    assert calls == 1 and item["prediction_returned"] is True
    assert (
        item["prediction_rows"]
        == item["prediction_rows_attempted"]
        == len(item["predictions"])
        == 2
    )
    assert item["residual_attempted"] is False
    assert result.report["native_updates"] == []
    assert source._state.parameters[3][0] == initial.parameters[3][0] + 0.125


def test_should_prepare_explicit_two_source_data_before_any_future_curation(monkeypatch):
    reference, corrections, curated = inputs()
    with monkeypatch.context() as patch:
        patch.setattr(data, "copy_curated_readiness_batch", forbidden)
        patch.setattr(data, "project_curated_training_pairs", forbidden)
        prepared = data.prepare_joint_readiness_data(reference, corrections, None, now=3)
    later = data.prepare_joint_readiness_data(reference, corrections, curated, now=103)
    assert prepared.rows == later.rows[:4]
    assert prepared.groups == (("a", 0, 2), ("b", 2, 4))
    assert prepared.work["rows"] == 4 and prepared.work["observation_projections"] == 8
    assert prepared.work["projected_values"] == 96
    assert prepared.work["projected_numeric_bytes"] == 768
    assert set(prepared.work["per_group"]) == {"a", "b"}
    for key in (
        "curated_projection_calls",
        "curated_label_observation_values",
        "curated_label_target_values",
        "curated_label_numeric_bytes",
    ):
        assert prepared.work[key] == 0
    assert [row["labels_arrived_at"] for row in prepared.rows] == [None, None, 1, 3]
    assert all(row["source_kind"] != "curated_provisional_correction" for row in prepared.rows)


def test_should_require_the_explicit_curated_argument_in_both_public_calls():
    reference, corrections, _ = inputs()
    with pytest.raises(TypeError):
        data.prepare_joint_readiness_data(reference, corrections, now=3)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        runtime.run_joint_readiness_corrections(object(), reference, corrections, now=3)  # type: ignore[call-arg]


@pytest.mark.parametrize("fault", ["false", "foreign", "unarrived", "role", "corrupt"])
def test_should_not_drop_an_invalid_or_unarrived_nonnull_curated_source(owner, monkeypatch, fault):
    reference, corrections, curated = inputs()
    selected: Any = curated
    now = 102 if fault == "unarrived" else 103
    if fault == "false":
        selected = False
    elif fault == "foreign":
        selected = object()
    elif fault == "role":
        object.__setattr__(curated.episode, "role", "probe_train")
    elif fault == "corrupt":
        frame = list(curated.episode.transitions[0].observation)
        frame[18] = 0
        object.__setattr__(curated.episode.transitions[0], "observation", tuple(frame))
    source = owner.fork()
    monkeypatch.setattr(data, "action_history_training_episode", forbidden)
    monkeypatch.setattr(data, "action_history_correction_view", forbidden)
    monkeypatch.setattr(data, "project_curated_training_pairs", forbidden)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forbidden)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", forbidden)
    with pytest.raises(ValueError):
        data.prepare_joint_readiness_data(reference, corrections, selected, now=now)
    with pytest.raises(ValueError):
        runtime.run_joint_readiness_corrections(source, reference, corrections, selected, now=now)


@pytest.fixture(scope="module")
def successful_two_source(owner):
    source = owner.fork()
    initial = source.snapshot_state()
    reference, corrections, _ = inputs()
    captured: dict[str, Any] = {}
    original_factory = runtime.ActorShadowRuntime

    def factory(model, **options):
        made = original_factory(model, **options)
        captured.update(runtime=made, budget=options["budget"])
        return made

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runtime, "ActorShadowRuntime", factory)
        patch.setattr(data, "copy_curated_readiness_batch", forbidden)
        patch.setattr(data, "project_curated_training_pairs", forbidden)
        result = runtime.run_joint_readiness_corrections(
            source, reference, corrections, None, now=3
        )
    return result, source, initial, captured


def test_should_complete_precuration_joint_updates_without_curated_rows(successful_two_source):
    result, source, initial, captured = successful_two_source
    report = result.report
    assert report["complete"] is True, report.get("error")
    assert type(result.candidate) is bp.NumpyContinuousBCLearner
    assert (
        result.candidate.snapshot_state().completed_updates,
        result.candidate.snapshot_state().training_examples,
    ) == (8, 32)
    assert report["groups"] == (("a", 0, 2), ("b", 2, 4))
    assert len(report["rows"]) == 4
    assert report["group_training_exposures"] == dict(a=16, b=16)
    assert report["work"]["training_rows"] == 32
    assert report["work"]["optimizer_updates"] == report["work"]["head_gauss_newton_updates"] == 8
    assert report["work"]["native_update_counts"]["solve_completed"] == 32
    assert report["work"]["native_update_counts"]["solve_rows"] == 224
    assert report["derived_batch"]["episode_id"] == "joint-TRAIN-A17-B19"
    assert report["preparation_work"]["curated_projection_calls"] == 0
    for native, receipt in zip(report["native_updates"], report["applied_updates"], strict=True):
        assert native["training_rows"] == 4
        assert receipt["episode_id"] == "joint-TRAIN-A17-B19"
        assert (receipt["observed_at"], receipt["arrived_at"], receipt["applied_at"]) == (3, 3, 3)
    for when in ("before", "after"):
        assert set(report["residuals"][when]) == {"a", "b"}
        assert all(item["prediction_rows"] == 2 for item in report["residuals"][when].values())
    assert report["native_updates"] == report["sealed_native_updates"]
    assert captured["budget"].updates_completed == 8
    assert_state(source.snapshot_state(), initial)
    assert_state(captured["runtime"].actor.snapshot_state().state, initial)
    assert result.candidate is not source


def test_should_continue_a_separate_branch_after_curation_without_renewing_prior_scope(
    successful_two_source, monkeypatch
):
    first, _, _, first_capture = successful_two_source
    assert first.candidate is not None
    original_report = json.dumps(first.report, sort_keys=True, allow_nan=False)
    original_state = first.candidate.snapshot_state()
    original_receipts = tuple(first_capture["runtime"]._inbox._applied.values())
    original_receipt_values = tuple(asdict(value) for value in original_receipts)
    original_budget = first_capture["budget"]
    budget_origin = (original_budget.budget, original_budget.clock, original_budget.started_at)
    source = first.candidate.fork()
    captured: dict[str, Any] = {}
    original_factory = runtime.ActorShadowRuntime

    def factory(model, **options):
        made = original_factory(model, **options)
        captured["budget"] = options["budget"]
        return made

    monkeypatch.setattr(runtime, "ActorShadowRuntime", factory)
    second = runtime.run_joint_readiness_corrections(source, *inputs(), now=103)
    assert second.report["complete"] is True, second.report.get("error")
    assert second.candidate is not None
    final = second.candidate.snapshot_state()
    assert (final.completed_updates, final.training_examples) == (16, 80)
    assert second.report["source_updates"] == 8 and second.report["source_training_rows"] == 32
    assert second.report["derived_batch"]["episode_id"] == "joint-TRAIN-A17-B19-curated-A17"
    assert set(second.report["group_training_exposures"]) == {"a", "b", "curated_a"}
    assert captured["budget"] is not original_budget
    assert captured["budget"].updates_completed == original_budget.updates_completed == 8
    assert original_budget.budget is budget_origin[0] and original_budget.clock is budget_origin[1]
    assert original_budget.started_at == budget_origin[2]
    assert json.dumps(first.report, sort_keys=True, allow_nan=False) == original_report
    current_receipts = tuple(first_capture["runtime"]._inbox._applied.values())
    assert all(
        current is previous
        for current, previous in zip(current_receipts, original_receipts, strict=True)
    )
    assert tuple(asdict(value) for value in current_receipts) == original_receipt_values
    assert_state(first.candidate.snapshot_state(), original_state)
    assert_state(source.snapshot_state(), original_state)


@pytest.mark.parametrize(
    "fault",
    ["raw_input", "native_prefix", "latest", "budget_clock", "logical_clock", "report_rows"],
)
def test_should_keep_two_source_authority_and_paid_prefix_before_more_work(
    owner, monkeypatch, fault
):
    source = owner.fork()
    reference, corrections, _ = inputs()
    original_capture = runtime._JointSession.capture
    changed: list[Any] = []

    def capture(session):
        original_capture(session)
        if session.committed != 1 or changed:
            return
        changed.append(session)
        if fault == "raw_input":
            object.__setattr__(corrections.labels[0], "target", (0.25, 0.0, 0.0, 1.0))
        elif fault == "native_prefix":
            session.native[0]["work"]["solve_rows"] += 1
        elif fault == "latest":
            session.raw_candidate._state.parameters[3][0] += 0.125
        elif fault == "budget_clock":
            session.budget.clock = lambda: 0.0
        elif fault == "logical_clock":
            session.clock.advance_to(4)
        else:
            session.report["rows"][0]["source_kind"] = "replaced"

    monkeypatch.setattr(runtime._JointSession, "capture", capture)
    result = runtime.run_joint_readiness_corrections(source, reference, corrections, None, now=3)
    assert changed and result.candidate is None and result.report["complete"] is False
    assert result.report["update_attempts"] == 1
    assert len(result.report["sealed_native_updates"]) == 1
    assert result.report["progress"]["updates_completed"] == 1
    if fault == "raw_input":
        assert corrections.labels[0].target == (0.25, 0.0, 0.0, 1.0)


def test_should_refuse_late_two_source_getter_advancing_the_original_clock(owner, monkeypatch):
    source = owner.fork()
    reference, corrections, _ = inputs()
    original_finish = runtime._JointSession.finish
    armed = dict(finished=False, changed=False)

    def finish(session):
        original_fork = session.source.fork

        def fork_with_observer():
            child = original_fork()
            original_snapshot = child.snapshot_state

            def snapshot():
                result = original_snapshot()
                if armed["finished"] and not armed["changed"]:
                    armed["changed"] = True
                    session.clock.advance_to(4)
                return result

            monkeypatch.setattr(child, "snapshot_state", snapshot)
            return child

        monkeypatch.setattr(session.source, "fork", fork_with_observer)
        result = original_finish(session)
        armed["finished"] = True
        return result

    monkeypatch.setattr(runtime._JointSession, "finish", finish)
    result = runtime.run_joint_readiness_corrections(source, reference, corrections, None, now=3)
    assert armed["changed"] and result.candidate is None and result.report["complete"] is False
    assert len(result.report["native_updates"]) == len(result.report["applied_updates"]) == 8
    assert result.report["progress"]["logical_tick"] == 4
