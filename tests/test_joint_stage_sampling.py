"""Fixed recorded-stage sampling: exact plans, native boundaries and forwarding.

One cached H2 source supplies three eight-update runs and short fault prefixes.
Stream-port tests are explicitly controlled orchestration, not another cohort.
"""

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from src.adapters import joint_readiness_corrections as runtime
from src.adapters import joint_readiness_stream as stream
from src.adapters import numpy_continuous_bc as bp
from src.adapters.continuous_learnability_demo import ContinuousLearnabilityResult
from src.adapters.joint_readiness_data import prepare_joint_readiness_data
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.curated_readiness_corrections import CuratedReadinessBatch, CuratedReadinessLabel
from src.core.joint_stage_sampling import plan_joint_stage_sampling
from src.core.pick_cube_readiness_corrections import (
    GoalConsistentObservedGraspCorrectionBatch,
    ReadinessCorrectionLabel,
)
from src.core.readiness_stage_diagnostics import ReadinessStageBinding
from test_joint_readiness_corrections import assert_state, inputs as unaligned_inputs


STAGES = ("align", "descend", "grasp", "lift", "settle")
AB_STAGES = (
    "align",
    "align",
    "align",
    "descend",
    "grasp",
    "lift",
    "settle",
    "lift",
    "lift",
    "lift",
)


def forbidden(*args, **kwargs):
    pytest.fail("refused sampling request admitted forbidden work")


def staged_inputs(c_stages=STAGES):
    spec = ContinuousActionSpec(
        42,
        ("normalized_root_dx", "normalized_root_dy", "normalized_root_dz", "normalized_gripper"),
        (-1.0,) * 4,
        (1.0,) * 4,
    )

    def episode(kind, stages):
        frames = []
        for index, stage in enumerate((*stages, stages[-1])):
            raw = [0.0] * 42
            raw[18] = float(stage in ("lift", "settle"))
            raw[36] = dict(a=0.01, b=-0.02, c=0.03)[kind] + 0.001 * index
            raw[41] = 0.05
            frames.append(tuple(raw))
        seed = 19 if kind == "b" else 17
        return ContinuousEpisode(
            "pick-cube-observed-grasp-teacher-17" if kind == "a" else f"pick-cube-smoke-{seed}",
            "pick-cube-train-19" if seed == 19 else "pick-cube",
            "train",
            spec,
            tuple(
                ContinuousTransition(
                    i,
                    frames[i],
                    (0.25, 0.0, 0.0, 1.0) if kind == "a" else (-0.25, 0.0, 0.0, -1.0),
                    0.0,
                    frames[i + 1],
                    False,
                    i == len(stages) - 1,
                )
                for i in range(len(stages))
            ),
        )

    def target(stage):
        if stage in ("grasp", "settle"):
            return (0.0, 0.0, 0.0, -1.0)
        return (-0.25, 0.0, 0.125, 1.0 if stage in ("align", "descend") else -1.0)

    b_stages = AB_STAGES[2:]
    a, b, c = episode("a", ("align", "align")), episode("b", b_stages), episode("c", c_stages)
    binding = ReadinessStageBinding(a, ("align", "align"))
    corrections = GoalConsistentObservedGraspCorrectionBatch(
        19,
        b,
        tuple(
            ReadinessCorrectionLabel(i, row.observation, target(stage), stage)
            for i, (row, stage) in enumerate(zip(b.transitions, b_stages, strict=True))
        ),
        tuple(stage in ("lift", "settle") for stage in b_stages),
    )
    curated = CuratedReadinessBatch(
        17,
        c,
        tuple(
            CuratedReadinessLabel(
                i, row.observation, target(stage), stage, None, 100 + 2 * i, 101 + 2 * i
            )
            for i, (row, stage) in enumerate(zip(c.transitions, c_stages, strict=True))
        ),
        100,
    )
    new_binding = ReadinessStageBinding(
        replace(
            b,
            episode_id="pick-cube-readiness-teacher-19",
            transitions=tuple(
                replace(row, action=target(stage))
                for row, stage in zip(b.transitions, b_stages, strict=True)
            ),
        ),
        b_stages,
    )
    return binding, new_binding, corrections, curated


@pytest.fixture(scope="module", autouse=True)
def work_ledger():
    counts = dict(
        initializations=0,
        normal_draws=0,
        GN_attempts=0,
        GN_rows=0,
        GN_returns=0,
        solve_attempts=0,
        solve_completed=0,
        augmented_rows=0,
        prediction_calls=0,
        prediction_rows=0,
    )
    initialize = bp.NumpyContinuousBCLearner.__init__
    train = bp.NumpyContinuousBCLearner.train_head_gauss_newton_batch
    predict = bp.NumpyContinuousBCLearner.predict

    def initialized(owner, policy):
        counts["initializations"] += 1
        counts["normal_draws"] += policy.hidden_dim * (policy.spec.observation_dim + 4)
        return initialize(owner, policy)

    def trained(owner, features, targets):
        counts["GN_attempts"] += 1
        counts["GN_rows"] += len(features)
        try:
            returned = train(owner, features, targets)
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

    def predicted(owner, features):
        counts["prediction_calls"] += 1
        counts["prediction_rows"] += len(features)
        return predict(owner, features)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bp.NumpyContinuousBCLearner, "__init__", initialized)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", trained)
        patch.setattr(bp.NumpyContinuousBCLearner, "predict", predicted)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", forbidden)
        patch.setattr(bp, "fit_training_standardization", forbidden)
        yield counts
    if os.environ.get("SBC_LEDGER"):
        with Path(os.environ["SBC_LEDGER"]).open("x", encoding="utf8") as handle:
            json.dump(counts, handle, sort_keys=True)
            handle.write("\n")
    assert counts["initializations"] == 1 and counts["normal_draws"] == 32
    assert counts["GN_attempts"] <= 32 and counts["GN_rows"] <= 400
    assert counts["solve_attempts"] <= 128 and counts["augmented_rows"] <= 2200
    assert counts["prediction_calls"] <= 48 and counts["prediction_rows"] <= 320


@pytest.fixture(scope="module")
def owned_source(work_ledger):
    values = staged_inputs()
    ab = prepare_joint_readiness_data(values[0], values[2], None, now=15)
    abc = prepare_joint_readiness_data(values[0], values[2], values[3], now=109)
    owner = bp.NumpyContinuousBCLearner(bp.ContinuousBCPolicy(ab.spec, hidden_dim=2, seed=17))
    return owner, values, ab, abc


def test_should_plan_immutable_exact_quotas_with_original_order_and_continuing_cursors():
    plan = plan_joint_stage_sampling(AB_STAGES)
    assert type(plan) is tuple and len(plan) == 8
    assert all(type(batch) is tuple and len(batch) == 10 for batch in plan)
    assert plan[0] == (0, 1, 3, 3, 4, 4, 5, 7, 6, 6)
    assert plan[1] == (2, 0, 3, 3, 4, 4, 8, 9, 6, 6)
    assert plan[2] == (1, 2, 3, 3, 4, 4, 5, 7, 6, 6)
    assert plan == plan_joint_stage_sampling(AB_STAGES)
    assert {i for batch in plan for i in batch} == set(range(10))
    for batch in plan:
        assert tuple(AB_STAGES[i] for i in batch) == tuple(
            stage for stage in STAGES for _ in range(2)
        )
    with pytest.raises(TypeError):
        cast(Any, plan[0])[0] = 9


@pytest.mark.parametrize("n", [5, 65, 115, 125])
def test_should_cover_every_original_row_at_fixed_supported_sizes_without_rng(n):
    stages = ("align",) * (n - 4) + STAGES[1:]
    plan = plan_joint_stage_sampling(stages)
    assert len(plan) == 8 and all(len(batch) == n for batch in plan)
    assert {index for batch in plan for index in batch} == set(range(n))
    for batch in plan:
        assert [sum(stages[index] == stage for index in batch) for stage in STAGES] == [n // 5] * 5


@pytest.mark.parametrize(
    "stages",
    [
        list(STAGES),
        (),
        ("align",) * 5,
        STAGES + ("align",),
        ("unknown",) + STAGES[1:],
        (True,) + STAGES[1:],
        STAGES * 26,
    ],
)
def test_should_refuse_invalid_plan_contract(stages):
    with pytest.raises(ValueError):
        plan_joint_stage_sampling(stages)


@pytest.fixture(scope="module")
def completed(owned_source):
    source, values, ab, abc = owned_source
    before = source.snapshot_state()
    original = tuple(asdict(value) for value in values)
    actual: dict[str, Any] = {}
    gathered: dict[str, Any] = {}
    original_train = bp.NumpyContinuousBCLearner.train_head_gauss_newton_batch
    current = []

    def train(owner, features, targets):
        current.append((features.copy(), targets.copy()))
        return original_train(owner, features, targets)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", train)
        for name, curated, now, sampling in (
            ("ab", None, 15, "stage_balanced"),
            ("abc", values[3], 109, "stage_balanced"),
            ("default", None, 15, None),
        ):
            current = gathered[name] = []
            live_source = source.fork()
            live_before = live_source.snapshot_state()
            actual[name] = runtime.run_joint_readiness_corrections(
                live_source,
                values[0],
                values[2],
                curated,
                now=now,
                **({} if sampling is None else dict(sampling=sampling)),
            )
            assert actual[name].report["complete"] is True, actual[name].report.get("error")
            assert actual[name].candidate is not live_source
            assert_state(live_before, live_source.snapshot_state())
    assert_state(before, source.snapshot_state())
    assert tuple(asdict(value) for value in values) == original
    return actual, gathered


@pytest.mark.parametrize(
    "name,n,expected_groups",
    [("ab", 10, {"a": 11, "b": 69}), ("abc", 15, {"a": 12, "b": 62, "curated_a": 46})],
)
def test_should_gather_original_rows_and_publish_exact_balanced_dose(
    completed, owned_source, name, n, expected_groups
):
    _, _, ab, abc = owned_source
    prepared = ab if name == "ab" else abc
    result = completed[0][name]
    report = result.report
    assert report["rows"] == prepared.rows and report["groups"] == prepared.groups
    sampling = report["sampling"]
    assert sampling["method"] == "stage_balanced_round_robin_v1"
    assert tuple(sampling["stage_order"]) == STAGES and sampling["quota_per_stage"] == n // 5
    plan = plan_joint_stage_sampling(tuple(row["stage"] for row in prepared.rows))
    assert tuple(tuple(batch) for batch in sampling["planned_row_indices"]) == plan
    exposure = [sum(batch.count(index) for batch in plan) for index in range(n)]
    assert report["row_training_exposures"] == exposure
    assert report["attempted_row_training_exposures"] == exposure
    assert sampling["planned_row_training_exposures"] == exposure
    assert sampling["planned_group_training_exposures"] == expected_groups
    assert report["group_training_exposures"] == expected_groups
    assert len(report["native_updates"]) == len(report["applied_updates"]) == 8
    assert report["native_updates"] == report["sealed_native_updates"]
    assert report["applied_updates"] == report["sealed_applied_updates"]
    assert report["candidate_updates"] == 8 and report["candidate_training_rows"] == 8 * n
    assert all(record["training_rows"] == n for record in report["native_updates"])
    assert len(completed[1][name]) == 8
    for (features, targets), batch in zip(completed[1][name], plan, strict=True):
        np.testing.assert_array_equal(
            features, np.asarray([prepared.rows[i]["features"] for i in batch])
        )
        np.testing.assert_array_equal(
            targets, np.asarray([prepared.rows[i]["target"] for i in batch])
        )
    for when in ("before", "after"):
        assert {
            key: value["prediction_rows"] for key, value in report["residuals"][when].items()
        } == {group: stop - start for group, start, stop in prepared.groups}
    initial = owned_source[0].snapshot_state()
    trained = result.candidate.snapshot_state()
    np.testing.assert_array_equal(initial.parameters[0], trained.parameters[0])
    np.testing.assert_array_equal(initial.parameters[1], trained.parameters[1])


def test_should_preserve_default_report_and_original_unresampled_exposure(completed):
    report = completed[0]["default"].report
    assert "sampling" not in report and "row_training_exposures" not in report
    assert "attempted_row_training_exposures" not in report
    assert report["group_training_exposures"] == {"a": 16, "b": 64}
    assert report["candidate_training_rows"] == 80
    for features, targets in completed[1]["default"]:
        np.testing.assert_array_equal(
            features, np.asarray([row["features"] for row in report["rows"]])
        )
        np.testing.assert_array_equal(
            targets, np.asarray([row["target"] for row in report["rows"]])
        )


@pytest.mark.parametrize("selection", [False, True, 1, "", "uniform", "stage-balanced"])
def test_should_refuse_invalid_selection_before_source_or_fork(
    owned_source, monkeypatch, selection
):
    source, values, _, _ = owned_source
    monkeypatch.setattr(runtime, "_admit_source", forbidden)
    monkeypatch.setattr(source, "fork", forbidden)
    with pytest.raises(ValueError):
        runtime.run_joint_readiness_corrections(
            source, values[0], values[2], None, now=15, sampling=selection
        )
    monkeypatch.setattr(stream, "run_continuous_learnability", forbidden)
    with pytest.raises(ValueError):
        stream.run_joint_readiness_stream(
            *values, corrections_now=15, curated_now=109, sampling=selection
        )


def test_should_admit_both_stream_plans_before_initializing_any_model(monkeypatch):
    values = staged_inputs(STAGES + ("align",))
    monkeypatch.setattr(stream, "run_continuous_learnability", forbidden)
    monkeypatch.setattr(stream, "run_joint_readiness_corrections", forbidden)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "__init__", forbidden)
    with pytest.raises(ValueError):
        stream.run_joint_readiness_stream(
            *values, corrections_now=15, curated_now=111, sampling="stage_balanced"
        )


def test_should_refuse_missing_recorded_stage_before_private_fork(owned_source, monkeypatch):
    source = owned_source[0].fork()
    reference, corrections, _ = unaligned_inputs(a_rows=2, b_rows=8, c_rows=5)
    monkeypatch.setattr(source, "fork", forbidden)
    with pytest.raises(ValueError):
        runtime.run_joint_readiness_corrections(
            source, reference, corrections, None, now=15, sampling="stage_balanced"
        )


@pytest.mark.parametrize(
    "fault",
    ["plan_identity", "plan_values", "report_sampling", "payload", "paid_evidence", "paid_failure"],
)
def test_should_refuse_changed_plan_or_payload_and_keep_validated_and_attempted_prefixes(
    owned_source, monkeypatch, fault
):
    source, values, prepared, _ = owned_source
    source = source.fork()
    before = source.snapshot_state()
    original_apply = runtime._JointSession.apply_one
    original_train = runtime.RecordedContinuousHead.train_batch
    calls: list[None] = []

    def apply(session, index):
        if index == 1:
            if fault == "plan_identity":
                session.sampling_plan = tuple(list(session.sampling_plan))
            elif fault == "plan_values":
                original = session.sampling_plan
                session.sampling_plan = ((9,) + original[0][1:],) + original[1:]
            elif fault == "report_sampling":
                session.report["sampling"]["planned_row_indices"][0][0] = 9
            elif fault == "payload":
                session.features[0, 0] += 0.125
            elif fault == "paid_evidence":
                session.report["row_training_exposures"][0] += 1
        return original_apply(session, index)

    def train(facade, features, targets):
        returned = original_train(facade, features, targets)
        calls.append(None)
        if fault == "paid_failure" and len(calls) == 2:
            raise RuntimeError("injected boundary failure after second native return")
        return returned

    monkeypatch.setattr(runtime._JointSession, "apply_one", apply)
    monkeypatch.setattr(runtime.RecordedContinuousHead, "train_batch", train)
    result = runtime.run_joint_readiness_corrections(
        source, values[0], values[2], None, now=15, sampling="stage_balanced"
    )
    report = result.report
    assert report["complete"] is False and result.candidate is None
    assert len(calls) == (2 if fault == "paid_failure" else 1)
    assert len(report["sealed_native_updates"]) == len(calls)
    assert len(report["applied_updates"]) == 1
    plan = plan_joint_stage_sampling(tuple(row["stage"] for row in prepared.rows))
    assert report["row_training_exposures"] == [plan[0].count(i) for i in range(10)]
    attempted = plan[:2] if fault == "paid_failure" else plan[:1]
    assert report["attempted_row_training_exposures"] == [
        sum(batch.count(i) for batch in attempted) for i in range(10)
    ]
    assert report["unvalidated_sampling_work_unknown"] is (fault == "paid_failure")
    if fault == "paid_evidence":
        assert (
            report["corrupted_copied_evidence"]["row_training_exposures"]
            != report["row_training_exposures"]
        )
    assert_state(before, source.snapshot_state())


@pytest.mark.parametrize("sampling", [None, "stage_balanced"])
def test_should_forward_only_selected_sampling_to_both_stream_branches(
    owned_source, monkeypatch, sampling
):
    _, values, ab, _ = owned_source
    policy = SimpleNamespace(spec=ab.spec, hidden_dim=64)
    calls = []

    def owner(updates, examples):
        return SimpleNamespace(
            state=SimpleNamespace(
                policy=policy, completed_updates=updates, training_examples=examples
            )
        )

    def joint(source, reference, corrections, curated, **options):
        calls.append((curated is None, options))
        rows = 10 if curated is None else 15
        return ContinuousLearnabilityResult(
            dict(complete=True),
            owner(source.state.completed_updates + 8, source.state.training_examples + 8 * rows),
        )

    def base(warm, new, **options):
        for i, (phase, updates, examples) in enumerate(
            (("initial", 0, 0), ("warm", 8, 16), ("new", 16, 80), ("warm_return", 24, 96))
        ):
            options["checkpoint_observer"](
                owner(updates, examples),
                dict(checkpoint_index=i, phase=phase, updates=updates, training_examples=examples),
            )
        return ContinuousLearnabilityResult(
            dict(complete=True, fixture="controlled stream ports"), owner(24, 96)
        )

    # These controlled ports test delegation only; actual native state guards are
    # exercised by the correction runs and existing stream regression evidence.
    monkeypatch.setattr(stream, "_pin", lambda value: (value, value, value.state))
    monkeypatch.setattr(stream, "_require_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(stream, "run_continuous_learnability", base)
    monkeypatch.setattr(stream, "run_joint_readiness_corrections", joint)
    outcome = stream.run_joint_readiness_stream(
        *values,
        corrections_now=15,
        curated_now=109,
        **({} if sampling is None else dict(sampling=sampling)),
    )
    assert outcome.report["complete"] is True, outcome.report.get("error")
    assert len(calls) == 2 and [empty for empty, _ in calls] == [True, False]
    for _, options in calls:
        assert ("sampling" in options) is (sampling is not None)
        if sampling is not None:
            assert options["sampling"] == sampling
