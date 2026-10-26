"""Supported two-stage composition and controlled boundary failures.

One actual H64 run owns all numerical work. Scripted runner/adapter cases reuse
fixed owned checkpoints to test orchestration; they are not repeated experiments.
"""

from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.adapters import joint_readiness_stream as stream
from src.adapters import numpy_continuous_bc as bp
from src.adapters.continuous_learnability_demo import ContinuousLearnabilityResult
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
from src.core.readiness_stage_diagnostics import ReadinessStageBinding


UNITS = ("normalized_root_dx", "normalized_root_dy", "normalized_root_dz", "normalized_gripper")
META = [
    dict(checkpoint_index=i, phase=phase, updates=updates, training_examples=examples)
    for i, (phase, updates, examples) in enumerate(
        (("initial", 0, 0), ("warm", 8, 16), ("new", 16, 32), ("warm_return", 24, 48))
    )
]


def training_inputs():
    spec = ContinuousActionSpec(42, UNITS, (-1.0,) * 4, (1.0,) * 4)
    episodes = []
    for kind, identity, seed, action in (
        ("a", "pick-cube-observed-grasp-teacher-17", 17, (0.25, 0.0, 0.0, 1.0)),
        ("b", "pick-cube-readiness-teacher-19", 19, (-0.25, 0.0, 0.0, 1.0)),
        ("b", "pick-cube-smoke-19", 19, (-0.25, 0.0, 0.0, -1.0)),
        ("c", "pick-cube-smoke-17", 17, (-0.25, 0.0, 0.0, -1.0)),
    ):
        frames = []
        for i in range(3):
            frame = [0.0] * 42
            frame[36] = {"a": 0.01, "b": -0.02, "c": 0.03}[kind] + 0.001 * i
            frame[41] = 0.05
            frames.append(tuple(frame))
        rows = tuple(
            ContinuousTransition(i, frames[i], action, 0.0, frames[i + 1], False, i == 1)
            for i in range(2)
        )
        episodes.append(
            ContinuousEpisode(
                identity, "pick-cube" if seed == 17 else "pick-cube-train-19", "train", spec, rows
            )
        )
    a, b, executed_b, executed_c = episodes
    corrections = GoalConsistentObservedGraspCorrectionBatch(
        19,
        executed_b,
        tuple(
            ReadinessCorrectionLabel(i, row.observation, (-0.25, 0.0, 0.0, 1.0), "align")
            for i, row in enumerate(executed_b.transitions)
        ),
        (False, False),
    )
    curated = CuratedReadinessBatch(
        17,
        executed_c,
        tuple(
            CuratedReadinessLabel(
                i, row.observation, (0.125, 0.25, 0.0, 1.0), "align", None, 100 + 2 * i, 101 + 2 * i
            )
            for i, row in enumerate(executed_c.transitions)
        ),
        100,
    )
    return (
        ReadinessStageBinding(a, ("align", "align")),
        ReadinessStageBinding(b, ("align", "align")),
        corrections,
        curated,
    )


def forbidden(*args, **kwargs):
    pytest.fail("rejected or unselected boundary performed model work")


def assert_state(first, second):
    assert first.policy == second.policy
    assert (first.completed_updates, first.training_examples) == (
        second.completed_updates,
        second.training_examples,
    )
    for a, b in zip(
        first.parameters + first.velocity, second.parameters + second.velocity, strict=True
    ):
        np.testing.assert_array_equal(a, b)


@pytest.fixture(scope="module")
def numerical_ledger():
    counts = dict(
        initializations=0,
        normal_draws=0,
        GN_attempts=0,
        GN_rows=0,
        GN_returns=0,
        solve_attempts=0,
        solve_completed=0,
        augmented_rows=0,
        prediction_attempts=0,
        prediction_rows=0,
        prediction_returns=0,
        SGD=0,
        teacher_queries=0,
        simulator_calls=0,
    )
    initialize = bp.NumpyContinuousBCLearner.__init__
    train = bp.NumpyContinuousBCLearner.train_head_gauss_newton_batch
    predict = bp.NumpyContinuousBCLearner.predict

    def observed_initialize(owner, policy):
        counts["initializations"] += 1
        counts["normal_draws"] += policy.hidden_dim * (
            policy.spec.observation_dim + len(policy.spec.action_units)
        )
        return initialize(owner, policy)

    def observed_train(owner, features, targets):
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

    def observed_predict(owner, features):
        counts["prediction_attempts"] += 1
        counts["prediction_rows"] += len(features)
        returned = predict(owner, features)
        counts["prediction_returns"] += 1
        return returned

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bp.NumpyContinuousBCLearner, "__init__", observed_initialize)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", observed_train)
        patch.setattr(bp.NumpyContinuousBCLearner, "predict", observed_predict)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", forbidden)
        yield counts
    if os.environ.get("JRS_LEDGER"):
        with Path(os.environ["JRS_LEDGER"]).open("x", encoding="utf8") as output:
            output.write(json.dumps(counts, sort_keys=True) + "\n")
    assert counts["initializations"] <= 4 and counts["normal_draws"] <= 4096
    assert counts["GN_attempts"] <= 256 and counts["GN_rows"] <= 2400
    assert counts["solve_attempts"] <= 1024 and counts["augmented_rows"] <= 40000
    assert counts["prediction_attempts"] <= 512 and counts["prediction_rows"] <= 5120


@pytest.fixture(scope="module")
def completed(numerical_ledger):
    values = training_inputs()
    original_inputs = tuple(asdict(value) for value in values)
    native_base = stream.run_continuous_learnability
    native_joint = stream.run_joint_readiness_corrections
    captured: dict[str, Any] = dict(points={}, observations=[], branches=[], calls=[])

    def base(warm, new, **options):
        captured["base_options"] = dict(options)
        original_observer = options["checkpoint_observer"]

        def observe(model, metadata):
            before = model.snapshot_state()
            captured["points"][metadata["checkpoint_index"]] = model.fork()
            returned = original_observer(model, metadata)
            after = model.snapshot_state()
            captured["observations"].append((deepcopy(metadata), before, after, deepcopy(returned)))
            return returned

        options["checkpoint_observer"] = observe
        result = native_base(warm, new, **options)
        captured["base"] = result
        return result

    def joint(source, reference, corrections, curated, **options):
        before = source.snapshot_state()
        captured["calls"].append((source, curated is None, options["now"], before))
        returned = native_joint(source, reference, corrections, curated, **options)
        captured["branches"].append(returned)
        assert_state(source.snapshot_state(), before)
        return returned

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(stream, "run_continuous_learnability", base)
        patch.setattr(stream, "run_joint_readiness_corrections", joint)
        result = stream.run_joint_readiness_stream(*values, corrections_now=3, curated_now=103)
    return result, captured, values, original_inputs


def test_should_branch_only_at_b_and_leave_the_actual_benchmark_exact(completed):
    result, captured, values, original_inputs = completed
    assert result.report["complete"] is True, result.report.get("error")
    assert type(result.candidate) is bp.NumpyContinuousBCLearner
    assert result.report["stream"] == captured["base"].report
    assert captured["base"].report["complete"] is True
    assert len(captured["calls"]) == len(captured["branches"]) == 2
    assert [(empty, tick) for _, empty, tick, _ in captured["calls"]] == [(True, 3), (False, 103)]
    assert result.report["checkpoint"] == META[2]
    assert [metadata for metadata, _, _, _ in captured["observations"]] == META
    for metadata, before, after, summary in captured["observations"]:
        assert_state(before, after)
        assert before.completed_updates == metadata["updates"]
        assert len(json.dumps(summary)) < 1024  # callback carries scalars, not full branch reports
    options = captured["base_options"]
    assert options["learner_family"] == "bp" and options["phase_updates"] == 8
    assert options["head_gauss_newton"] is True
    assert "probe_plan" not in options and "replay_plan" not in options
    base_state = captured["base"].candidate.snapshot_state()
    final = result.candidate.snapshot_state()
    assert (base_state.completed_updates, base_state.training_examples) == (24, 48)
    assert (final.completed_updates, final.training_examples) == (32, 112)
    assert final.policy.hidden_dim == 64 and final.policy.seed == 17
    assert result.candidate is not captured["base"].candidate
    assert tuple(asdict(value) for value in values) == original_inputs


def test_should_preserve_both_paid_scopes_and_account_preflight_projection_cost(completed):
    result, captured, _, _ = completed
    first, second = captured["branches"]
    assert result.report["joint_ab"] == first.report
    assert result.report["joint_abc"] == second.report
    for report, source, rows, tick in ((first.report, 16, 32, 3), (second.report, 24, 64, 103)):
        assert report["source_updates"] == source and report["source_training_rows"] == rows
        assert report["progress"]["updates_completed"] == 8
        assert report["budget"]["max_training_updates"] == 8
        assert len(report["applied_updates"]) == len(report["sealed_native_updates"]) == 8
        assert report["applied_updates"] == report["sealed_applied_updates"]
        assert all(
            (row["observed_at"], row["arrived_at"], row["applied_at"]) == (tick,) * 3
            for row in report["applied_updates"]
        )
    assert first.report["candidate_updates"] == 24 and first.report["candidate_training_rows"] == 64
    assert (
        second.report["candidate_updates"] == 32 and second.report["candidate_training_rows"] == 112
    )
    for source, _, _, before in captured["calls"]:
        assert_state(source.snapshot_state(), before)
    assert first.candidate is not second.candidate
    preparation = result.report["preparation_work"]
    assert (
        preparation["preflight_calls"]
        == preparation["runtime_calls_attempted"]
        == preparation["runtime_calls_returned"]
        == 2
    )
    assert preparation["preflight_ab"]["rows"] == 4
    assert preparation["preflight_abc"]["rows"] == 6
    assert preparation["source_projection"]["observation_projections"] == 8
    assert first.report["preparation_work"]["curated_projection_calls"] == 0
    assert second.report["preparation_work"]["curated_projection_calls"] == 1


@pytest.mark.parametrize(
    "fault", ["early_curated", "role", "clock_order", "bool_tick", "missing_curated"]
)
def test_should_refuse_invalid_full_recipe_before_base_initialization(monkeypatch, fault):
    warm, new, corrections, curated = training_inputs()
    correction_tick: Any = 3
    curated_tick = 102 if fault == "early_curated" else 103
    selected: Any = curated
    if fault == "role":
        object.__setattr__(new.episode, "role", "probe_test")
    elif fault == "clock_order":
        correction_tick, curated_tick = 104, 103
    elif fault == "bool_tick":
        correction_tick = True
    elif fault == "missing_curated":
        selected = None
    monkeypatch.setattr(stream, "run_continuous_learnability", forbidden)
    monkeypatch.setattr(stream, "run_joint_readiness_corrections", forbidden)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "__init__", forbidden)
    with pytest.raises(ValueError):
        stream.run_joint_readiness_stream(
            warm,
            new,
            corrections,
            selected,
            corrections_now=correction_tick,
            curated_now=curated_tick,
        )


def scripted_boundary(monkeypatch, completed, values, *, failure=None, mutation=None):
    """Controlled orchestration ports; all numerical states come from the one actual run."""
    _, actual, _, _ = completed
    captured: dict[str, Any] = dict(calls=[], results=[], base_returned=False, changed=False)

    def joint(source, reference, corrections, curated, **options):
        index = len(captured["calls"])
        captured["calls"].append((curated is None, options["now"]))
        if failure == ("first" if index == 0 else "second"):
            returned = ContinuousLearnabilityResult(
                dict(
                    complete=False,
                    error="scripted stage failure",
                    fixture="controlled boundary, no numerical work",
                    paid_prefix=[dict(attempt=1)],
                ),
                None,
            )
        else:
            template = actual["branches"][index]
            returned = ContinuousLearnabilityResult(
                deepcopy(template.report), template.candidate.fork()
            )
        captured["results"].append(returned)
        if index == 1 and mutation in ("prior_candidate", "prior_report"):
            first = captured["results"][0]
            if mutation == "prior_candidate":
                captured["prior_state"] = first.candidate.snapshot_state()
                first.candidate._state.parameters[3][0] += 0.125
            else:
                first.report["native_updates"][0]["diagnostic"]["value"] += 0.125
            captured["changed"] = True
        if index == 1 and mutation == "final_getter":
            assert returned.candidate is not None
            original_snapshot = returned.candidate.snapshot_state

            def snapshot():
                result = original_snapshot()
                if captured["base_returned"] and not captured["changed"]:
                    object.__setattr__(values[2].labels[0], "target", (0.25, 0.0, 0.0, 1.0))
                    captured["changed"] = True
                return result

            monkeypatch.setattr(returned.candidate, "snapshot_state", snapshot)
        return returned

    def base(warm, new, **options):
        if failure == "base":
            return ContinuousLearnabilityResult(
                dict(
                    complete=False,
                    error="scripted base failure",
                    fixture="controlled boundary, no numerical work",
                    paid_prefix=[dict(attempt=1)],
                ),
                None,
            )
        observer = options["checkpoint_observer"]
        for i, metadata in enumerate(META):
            if i == 3 and mutation == "later_raw_input":
                object.__setattr__(values[2].labels[0], "target", (0.25, 0.0, 0.0, 1.0))
                captured["changed"] = True
            model = actual["points"][i].fork()
            try:
                observer(model, deepcopy(metadata))
            except Exception as error:
                report = dict(
                    complete=False,
                    error=f"scripted callback: {error}",
                    fixture="controlled boundary, no numerical work",
                    paid_prefix=[dict(checkpoint=i)],
                )
                return ContinuousLearnabilityResult(report, None)
        captured["base_returned"] = True
        return ContinuousLearnabilityResult(
            deepcopy(actual["base"].report), actual["base"].candidate.fork()
        )

    monkeypatch.setattr(stream, "run_continuous_learnability", base)
    monkeypatch.setattr(stream, "run_joint_readiness_corrections", joint)
    return captured


def test_should_accept_unchanged_controlled_boundaries(monkeypatch, completed):
    values = training_inputs()
    captured = scripted_boundary(monkeypatch, completed, values)
    result = stream.run_joint_readiness_stream(*values, corrections_now=3, curated_now=103)
    assert result.report["complete"] is True, result.report.get("error")
    assert "error" not in result.report
    assert captured["calls"] == [(True, 3), (False, 103)]
    assert captured["base_returned"] is True
    assert result.candidate is not None
    state = result.candidate.snapshot_state()
    assert (state.completed_updates, state.training_examples) == (32, 112)


@pytest.mark.parametrize("failure,expected_calls", [("base", 0), ("first", 1), ("second", 2)])
def test_should_preserve_failed_base_or_branch_and_stop_before_next_work(
    monkeypatch, completed, failure, expected_calls
):
    values = training_inputs()
    captured = scripted_boundary(monkeypatch, completed, values, failure=failure)
    result = stream.run_joint_readiness_stream(*values, corrections_now=3, curated_now=103)
    assert result.report["complete"] is False and result.candidate is None
    assert len(captured["calls"]) == expected_calls
    assert result.report["stream"]["complete"] is False
    if failure == "base":
        assert result.report["joint_ab"] is result.report["joint_abc"] is None
        assert result.report["stream"]["paid_prefix"] == [dict(attempt=1)]
    elif failure == "first":
        assert result.report["joint_ab"]["paid_prefix"] == [dict(attempt=1)]
        assert result.report["joint_abc"] is None
    else:
        assert result.report["joint_ab"]["complete"] is True
        assert result.report["joint_abc"]["paid_prefix"] == [dict(attempt=1)]


@pytest.mark.parametrize(
    "mutation", ["prior_candidate", "prior_report", "later_raw_input", "final_getter"]
)
def test_should_refuse_changed_prior_evidence_after_later_observation(
    monkeypatch, completed, mutation
):
    values = training_inputs()
    captured = scripted_boundary(monkeypatch, completed, values, mutation=mutation)
    result = stream.run_joint_readiness_stream(*values, corrections_now=3, curated_now=103)
    assert captured["changed"] and result.report["complete"] is False and result.candidate is None
    assert len(captured["calls"]) == 2
    assert result.report["joint_ab"]["complete"] is result.report["joint_abc"]["complete"] is True
    assert result.report["error"]
    if mutation == "prior_candidate":
        first = captured["results"][0].candidate
        assert first._state.parameters[3][0] == captured["prior_state"].parameters[3][0] + 0.125
    elif mutation == "prior_report":
        assert captured["results"][0].report != result.report["joint_ab"]
    elif mutation in ("later_raw_input", "final_getter"):
        assert values[2].labels[0].target == (0.25, 0.0, 0.0, 1.0)
