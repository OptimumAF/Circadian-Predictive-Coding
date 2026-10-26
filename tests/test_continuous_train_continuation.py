"""Supported private TRAIN continuation, paid evidence and frozen normalization.

One D2/H2/A2 initializer draws8 seeded values and genuinely warms on two rows.
Forecast:11 native GN attempts/171 rows/22 solves/408 augmented rows; no explicit
prediction, SGD, fit, teacher or SDK work. The130-row case permits a recorded
original-objective rejection after its actual solves; it never requires a win.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.adapters import continuous_train_continuation as continuation
from src.adapters import numpy_continuous_bc as bp
from src.app.actor_shadow import ActorShadowRuntime as NativeRuntime
from src.core.observation_standardization import ObservationStandardization
from test_continuous_probe import assert_state, episode


def forbidden(*args, **kwargs):
    pytest.fail("continuation admitted forbidden fork, SGD, fit or prediction work")


@pytest.fixture(scope="module", autouse=True)
def ledger():
    counts = dict(
        initializations=0,
        normal_draws=0,
        GN_attempts=0,
        GN_returns=0,
        GN_rows_attempted=0,
        GN_rows_returned=0,
        solve_attempts=0,
        solve_completed=0,
        augmented_rows=0,
        SGD=0,
        fits=0,
        predictions=0,
    )
    initialize = bp.NumpyContinuousBCLearner.__init__
    train = bp.NumpyContinuousBCLearner.train_head_gauss_newton_batch

    def initialized(owner, policy):
        counts["initializations"] += 1
        counts["normal_draws"] += policy.hidden_dim * (
            policy.spec.observation_dim + len(policy.spec.action_units)
        )
        return initialize(owner, policy)

    def trained(owner, features, targets):
        counts["GN_attempts"] += 1
        counts["GN_rows_attempted"] += len(features)
        try:
            returned = train(owner, features, targets)
        except Exception as error:
            work = getattr(error, "work", {})
            counts["solve_attempts"] += work.get("solve_attempts", 0)
            counts["solve_completed"] += work.get("solve_completed", 0)
            counts["augmented_rows"] += work.get("solve_rows", 0)
            raise
        counts["GN_returns"] += 1
        counts["GN_rows_returned"] += len(features)
        counts["solve_attempts"] += returned.work["solve_attempts"]
        counts["solve_completed"] += returned.work["solve_completed"]
        counts["augmented_rows"] += returned.work["solve_rows"]
        return returned

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bp.NumpyContinuousBCLearner, "__init__", initialized)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", trained)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", forbidden)
        patch.setattr(bp.NumpyContinuousBCLearner, "predict", forbidden)
        patch.setattr(bp, "fit_training_standardization", forbidden)
        yield counts
    if os.environ.get("CTC_FIXTURE_LEDGER"):
        with Path(os.environ["CTC_FIXTURE_LEDGER"]).open("x", encoding="utf8") as output:
            json.dump(counts, output, sort_keys=True)
            output.write("\n")
    assert counts["initializations"] == 1 and counts["normal_draws"] == 8
    assert counts["GN_attempts"] <= 64 and counts["GN_rows_attempted"] <= 256
    assert counts["solve_attempts"] <= 128 and counts["augmented_rows"] <= 896


@pytest.fixture(scope="module")
def owner(ledger):
    source = bp.NumpyContinuousBCLearner(
        bp.ContinuousBCPolicy(episode().spec, hidden_dim=2, seed=17)
    )
    training = episode()
    source.train_head_gauss_newton_batch(
        np.asarray([row.observation for row in training.transitions], dtype=np.float64),
        np.asarray([row.action for row in training.transitions], dtype=np.float64),
    )
    assert (
        source.snapshot_state().completed_updates,
        source.snapshot_state().training_examples,
    ) == (1, 2)
    return source


def capture_runtime(monkeypatch):
    captured: dict[str, Any] = {}
    factory = continuation.ActorShadowRuntime

    def make(model, **options):
        made = factory(model, **options)
        captured.update(
            runtime=made, facade=model, budget=options["budget"], clock=options["clock"]
        )
        return made

    monkeypatch.setattr(continuation, "ActorShadowRuntime", make)
    return captured


@pytest.fixture(scope="module")
def completed(owner):
    source = owner.fork()
    initial = source.snapshot_state()
    training, reference = episode("B"), episode()
    original = asdict(training), asdict(reference)
    with pytest.MonkeyPatch.context() as patch:
        captured = capture_runtime(patch)
        result = continuation.run_continuous_train_continuation(
            source, training, references=(reference,), updates=2
        )
    return result, source, initial, captured, training, reference, original


def test_should_continue_genuinely_learned_source_and_return_only_a_detached_candidate(completed):
    result, source, initial, captured, training, reference, original = completed
    assert result.report["complete"] is True, result.report.get("error")
    assert type(result.candidate) is bp.NumpyContinuousBCLearner
    final = result.candidate.snapshot_state()
    assert (final.completed_updates, final.training_examples) == (3, 12)
    assert final.policy == initial.policy
    assert_state(source.snapshot_state(), initial)
    assert_state(captured["runtime"].actor.snapshot_state().state, initial)
    assert result.candidate is not source
    assert result.candidate is not captured["runtime"]._candidate._learner
    for index in (0, 1):
        np.testing.assert_array_equal(final.parameters[index], initial.parameters[index])
        np.testing.assert_array_equal(final.velocity[index], initial.velocity[index])
    assert not any(
        np.shares_memory(first, second)
        for first in result.candidate._state.parameters + result.candidate._state.velocity
        for second in source._state.parameters + source._state.velocity
    )
    assert (asdict(training), asdict(reference)) == original


def test_should_preserve_delayed_receipts_original_counts_and_group_exposure(completed):
    report = completed[0].report
    assert report["protocol"] == "continuous_train_continuation_v1"
    assert report["initial_counts"] == dict(updates=1, training_rows=2)
    assert report["final_counts"] == dict(updates=3, training_rows=12)
    assert report["groups"] == (("training", 0, 3), ("reference_0", 3, 5))
    assert report["rows"] == 5
    assert [item["group"] for item in report["original_inputs"]] == ["training", "reference_0"]
    assert [item["step"] for item in report["row_bindings"]] == [0, 1, 2, 0, 1]
    assert (
        report["native_updates"]
        == report["sealed_native_updates"]
        == report["validated_native_updates"]
    )
    assert report["applied_updates"] == report["sealed_applied_updates"]
    assert len(report["native_updates"]) == len(report["applied_updates"]) == 2
    for index, (event, receipt) in enumerate(
        zip(report["events"], report["applied_updates"], strict=True)
    ):
        assert event["deferred_exact"] is event["success_validated"] is True
        assert receipt["sample_id"] == f"continuation-{index}"
        assert receipt["event_id"] == f"continuation-label-{index}"
        assert receipt["episode_id"] == "continuous-TRAIN-continuation"
        assert (receipt["observed_at"], receipt["arrived_at"], receipt["applied_at"]) == (
            1000 + 2 * index,
            1001 + 2 * index,
            1001 + 2 * index,
        )
    assert report["source_exact"] is report["actor_exact"] is True
    assert report["work"]["training_rows"] == 10
    assert report["work"]["new_training_rows"] == 6
    assert report["work"]["reference_training_rows"] == 4
    assert report["work"]["group_training_exposures"] == dict(training=6, reference_0=4)
    assert report["work"]["inbox_feature_target_numeric_bytes"] == 320
    assert report["work"]["parameter_momentum_numeric_bytes_per_owner"] == 192
    json.dumps(report, allow_nan=False)


def test_should_keep_default_eight_budget_and_reference_order_before_math(owner, monkeypatch):
    source = owner.fork()
    training, first = episode("B"), episode()
    last = replace(episode(rows=1), episode_id="synthetic-C", context="context-C")
    captured = capture_runtime(monkeypatch)
    arrays: list[tuple[Any, Any]] = []

    def stop(port, features, targets):
        arrays.append((features.copy(), targets.copy()))
        raise RuntimeError("controlled stop before native GN")

    monkeypatch.setattr(continuation.RecordedContinuousHead, "train_batch", stop)
    result = continuation.run_continuous_train_continuation(
        source, training, references=(first, last)
    )
    assert result.candidate is None and result.report["complete"] is False
    assert len(arrays) == 1 and captured["budget"].budget.max_training_updates == 8
    records = training.transitions + first.transitions + last.transitions
    np.testing.assert_array_equal(arrays[0][0], [row.observation for row in records])
    np.testing.assert_array_equal(arrays[0][1], [row.action for row in records])
    assert result.report["events"][0]["deferred_exact"] is True
    assert result.report["native_updates"] == result.report["applied_updates"] == []


def test_should_retain_existing_frozen_normalization_without_fit(owner):
    source = owner.fork()
    original = source.snapshot_state()
    statistics = ObservationStandardization((1.0, -0.5), (2.0, 0.5), 2)
    # A declared synthetic frozen-policy variant; no statistics are fitted here.
    source._policy = replace(original.policy, observation_standardization=statistics)
    source.restore_state(replace(original, policy=source._policy))
    initial = source.snapshot_state()
    result = continuation.run_continuous_train_continuation(source, episode("B"), updates=1)
    assert result.report["complete"] is True, result.report.get("error")
    assert isinstance(result.candidate, bp.NumpyContinuousBCLearner)
    assert result.candidate.policy.observation_standardization == statistics
    assert result.report["native_updates"][0]["work"]["normalization_rows"] == 3
    assert result.report["native_updates"][0]["work"]["normalization_values"] == 6
    assert result.report["work"]["frozen_standardization_numeric_bytes_per_owner"] == 32
    assert_state(source.snapshot_state(), initial)


def test_should_reach_two_real_axis_solves_for_the_130_row_capacity(owner):
    source = owner.fork()
    initial = source.snapshot_state()
    result = continuation.run_continuous_train_continuation(source, episode(rows=130), updates=1)
    report = result.report
    work = (
        report["native_updates"][0]["work"]
        if report["native_updates"]
        else report["failed_native_work"]
    )
    assert work["normalization_rows"] == 130
    assert work["solve_attempts"] == work["solve_completed"] == 2
    assert work["solve_rows"] == 266
    if report["complete"]:
        assert isinstance(result.candidate, bp.NumpyContinuousBCLearner)
        assert (
            result.candidate.snapshot_state().completed_updates,
            result.candidate.snapshot_state().training_examples,
        ) == (2, 132)
    else:
        assert result.candidate is None and "head update rejected" in report["error"]
    assert_state(source.snapshot_state(), initial)


@pytest.mark.parametrize("updates", [0, 9, True, 1.0, None])
def test_should_refuse_invalid_update_budget_before_fork(owner, monkeypatch, updates):
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forbidden)
    with pytest.raises(ValueError):
        continuation.run_continuous_train_continuation(owner, episode("B"), updates=updates)


@pytest.mark.parametrize(
    "fault",
    [
        "role",
        "reference_role",
        "spec",
        "duplicate_id",
        "list",
        "nine_refs",
        "overflow_rows",
        "foreign",
        "binary",
        "future_updates",
        "future_rows",
    ],
)
def test_should_refuse_bad_source_or_episode_authority_before_fork(owner, monkeypatch, fault):
    source: Any = owner.fork()
    training = episode("B")
    references: Any = (episode(),)
    if fault == "role":
        training = replace(training, role="probe_train")
    elif fault == "reference_role":
        references = (replace(episode(), role="final_test"),)
    elif fault == "spec":
        training = episode("B", declared=replace(episode().spec, action_units=("other", "m")))
    elif fault == "duplicate_id":
        references = (replace(episode(), episode_id=training.episode_id),)
    elif fault == "list":
        references = [episode()]
    elif fault == "nine_refs":
        references = tuple(replace(episode(rows=1), episode_id=f"reference-{i}") for i in range(9))
    elif fault == "overflow_rows":
        training, references = episode("B", rows=130), (episode(rows=1),)
    elif fault == "foreign":
        source = object()
    elif fault == "binary":
        binary_spec = replace(source._policy.spec, action_units=("m", "normalized_gripper"))
        source._policy = replace(source._policy, spec=binary_spec, binary_gripper_axis=1)
        source._state = replace(source._state, policy=source._policy)
        training, references = episode("B", declared=binary_spec), (episode(declared=binary_spec),)
    else:
        source.restore_state(
            replace(
                source.snapshot_state(),
                **(
                    dict(completed_updates=2**53 - 1, training_examples=2**53 - 1)
                    if fault == "future_updates"
                    else dict(completed_updates=2**53 - 2, training_examples=2**53 - 1)
                ),
            )
        )
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forbidden)
    if fault in ("role", "reference_role"):
        monkeypatch.setattr(continuation, "copy_continuous_episode", forbidden)
    with pytest.raises(ValueError):
        continuation.run_continuous_train_continuation(
            source, training, references=references, updates=1
        )


def test_should_keep_a_known_success_when_next_facade_call_fails_before_native_return(
    owner, monkeypatch
):
    calls: list[bool] = []
    original = continuation.RecordedContinuousHead.train_batch

    def fail_second(port, features, targets):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("next native return unavailable")
        return original(port, features, targets)

    monkeypatch.setattr(continuation.RecordedContinuousHead, "train_batch", fail_second)
    result = continuation.run_continuous_train_continuation(
        owner, episode("B"), references=(episode(),), updates=3
    )
    report = result.report
    assert result.candidate is None and report["complete"] is False
    assert len(calls) == 2 and len(report["native_updates"]) == len(report["applied_updates"]) == 1
    assert report["events"][0]["success_validated"] is True
    assert "next native return unavailable" in report["error"]
    assert report["progress"]["updates_completed"] == 1


@pytest.mark.parametrize("fault", ["native_content", "native_identity", "receipt_identity"])
def test_should_refuse_corrupted_preceding_success_before_more_native_work(
    owner, monkeypatch, fault
):
    captured = capture_runtime(monkeypatch)
    ready = NativeRuntime.train_ready
    mutations: list[bool] = []

    def changed(runtime, *args, **options):
        returned = ready(runtime, *args, **options)
        if returned == () and len(runtime.applied_updates) == 1 and not mutations:
            if fault == "native_content":
                captured["facade"]._reports[0]["work"]["published_updates"] = 2
            elif fault == "native_identity":
                captured["facade"]._reports[0] = deepcopy(captured["facade"]._reports[0])
            else:
                key = next(iter(runtime._inbox._applied))
                runtime._inbox._applied[key] = replace(runtime._inbox._applied[key])
            mutations.append(True)
        return returned

    monkeypatch.setattr(NativeRuntime, "train_ready", changed)
    result = continuation.run_continuous_train_continuation(
        owner, episode("B"), references=(episode(),), updates=3
    )
    report = result.report
    assert mutations == [True] and result.candidate is None and report["complete"] is False
    assert report["events"][0]["success_validated"] is True
    assert len(report["sealed_native_updates"]) == len(report["validated_native_updates"]) == 1
    assert report["sealed_native_updates"][0]["work"]["published_updates"] == 1
    assert report["progress"]["updates_completed"] == 1


@pytest.mark.parametrize("fault", ["source", "candidate_state"])
def test_should_refuse_late_unobserved_native_change_without_rollback(owner, monkeypatch, fault):
    source = owner.fork()
    initial = source.snapshot_state()
    captured = capture_runtime(monkeypatch)
    ready = NativeRuntime.train_ready
    mutations: list[bool] = []

    def changed(runtime, *args, **options):
        returned = ready(runtime, *args, **options)
        if returned == () and len(runtime.applied_updates) == 1 and not mutations:
            if fault == "source":
                source._state.parameters[3][0] += 0.125
            else:
                model = captured["runtime"]._candidate._learner
                model._state = deepcopy(model._state)
            mutations.append(True)
        return returned

    monkeypatch.setattr(NativeRuntime, "train_ready", changed)
    result = continuation.run_continuous_train_continuation(source, episode("B"), updates=1)
    assert mutations == [True] and result.candidate is None and result.report["complete"] is False
    assert (
        len(result.report["applied_updates"]) == len(result.report["validated_native_updates"]) == 1
    )
    if fault == "source":
        assert source._state.parameters[3][0] == initial.parameters[3][0] + 0.125
    else:
        assert_state(source.snapshot_state(), initial)
