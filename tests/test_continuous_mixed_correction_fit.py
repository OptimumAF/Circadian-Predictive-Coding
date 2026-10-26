"""All148 released TRAIN rows share one ordinary bounded mixed SGD scope.

One H2/D12/A4 initializer consumes32 normal values. Full-module forecast is13
native SGD/gradient/forward returns on1778 rows: a two-row common-MSE warmup,
8x148 successful continuation rows, and two preserved2x148 failure prefixes.
No prediction, GN, fit, teacher, SDK or scientific payload is used.
"""

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.adapters import continuous_train_continuation as continuation
from src.adapters import numpy_continuous_bc as bp
from src.app.actor_shadow import ActorShadowRuntime
from src.core import continuous_bc as core
from src.core.continuous_demonstration import ContinuousTransition
from src.core.observation_standardization import ObservationStandardization
from test_continuous_mixed_continuation import assert_same, capture, episode, pairs


def forbidden(*args, **kwargs):
    pytest.fail("corrective fit admitted refused-input private work, prediction, GN or fitting")


def rows(name, count, *, gripper=-1.0):
    template = episode(name, gripper=gripper)

    def observation(index):
        return (0.0, 0.2 + (index % 3) * 0.1, -0.1) + (0.0,) * 9

    transitions = tuple(
        ContinuousTransition(
            index,
            observation(index),
            (0.2, -0.1, 0.1, gripper),
            0.0,
            observation(index + 1),
            index == count - 1,
            False,
        )
        for index in range(count)
    )
    return replace(template, transitions=transitions)


def inputs():
    return rows("corrections", 50), (
        rows("original-push", 18),
        rows("A", 15, gripper=1.0),
        rows("B", 50, gripper=1.0),
        rows("train20", 15, gripper=1.0),
    )


@pytest.fixture(scope="module", autouse=True)
def ledger():
    counts = dict(
        initializations=0,
        normal_draws=0,
        SGD_attempts=0,
        SGD_returns=0,
        SGD_rows_attempted=0,
        SGD_rows_returned=0,
        gradient_attempts=0,
        gradient_returns=0,
        gradient_rows_attempted=0,
        gradient_rows_returned=0,
        forward_attempts=0,
        forward_returns=0,
        forward_rows_attempted=0,
        forward_rows_returned=0,
    )
    initialize = bp.NumpyContinuousBCLearner.__init__
    train, gradient, forward = (
        bp.NumpyContinuousBCLearner.train_batch,
        core.continuous_bc_loss_and_gradients,
        core._forward,
    )

    def initialized(owner, policy):
        counts["initializations"] += 1
        counts["normal_draws"] += policy.hidden_dim * (
            policy.spec.observation_dim + len(policy.spec.action_units)
        )
        return initialize(owner, policy)

    def counted(original, kind):
        def call(*args, **options):
            features = args[1] if kind == "SGD" else args[0]
            counts[kind + "_attempts"] += 1
            counts[kind + "_rows_attempted"] += len(features)
            result = original(*args, **options)
            counts[kind + "_returns"] += 1
            counts[kind + "_rows_returned"] += len(features)
            return result

        return call

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bp.NumpyContinuousBCLearner, "__init__", initialized)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", counted(train, "SGD"))
        patch.setattr(core, "continuous_bc_loss_and_gradients", counted(gradient, "gradient"))
        patch.setattr(core, "_forward", counted(forward, "forward"))
        patch.setattr(bp.NumpyContinuousBCLearner, "predict", forbidden)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", forbidden)
        patch.setattr(bp, "fit_training_standardization", forbidden)
        yield counts
    if os.environ.get("CORRECTION_FIT_FIXTURE_LEDGER"):
        with Path(os.environ["CORRECTION_FIT_FIXTURE_LEDGER"]).open("x", encoding="utf8") as output:
            json.dump(counts, output, sort_keys=True)
            output.write("\n")
    assert counts["initializations"] == 1 and counts["normal_draws"] == 32
    for kind in ("SGD", "gradient", "forward"):
        assert counts[kind + "_attempts"] == counts[kind + "_returns"] == 13
        assert counts[kind + "_rows_attempted"] == counts[kind + "_rows_returned"] == 1778


@pytest.fixture(scope="module")
def owner(ledger):
    statistics = ObservationStandardization((0.0,) * 12, (2.0,) * 12, 2)
    source = bp.NumpyContinuousBCLearner(
        bp.ContinuousBCPolicy(
            episode().spec,
            hidden_dim=2,
            seed=17,
            learning_rate=0.05,
            momentum=0.2,
            observation_standardization=statistics,
        )
    )
    source.train_batch(*pairs(episode("warm", gripper=1.0)))
    return source


@pytest.fixture(scope="module")
def completed(owner):
    training, references = inputs()
    initial = owner.snapshot_state()
    input_seal = tuple(asdict(item) for item in (training, *references))
    all_rows = tuple(row for value in (training, *references) for row in value.transitions)
    expected_features = np.asarray([row.observation for row in all_rows], np.float64)
    expected_targets = np.asarray([row.action for row in all_rows], np.float64)
    observed: list[Any] = []
    train = bp.NumpyContinuousBCLearner.train_batch

    def trained(source, features, targets):
        np.testing.assert_array_equal(features, expected_features)
        np.testing.assert_array_equal(targets, expected_targets)
        observed.append((features.shape, targets.shape))
        return train(source, features, targets)

    with pytest.MonkeyPatch.context() as patch:
        saved = capture(patch)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", trained)
        result = continuation.run_continuous_train_continuation(
            owner, training, references=references, update_rule="bp_mixed"
        )
    return result, initial, saved, training, references, input_seal, observed


def test_should_complete_all148_with_inherited_momentum_frozen_statistics_and_detached_native_candidate(
    owner, completed
):
    result, initial, saved, training, references, input_seal, _ = completed
    assert result.report["complete"] is True, result.report.get("error")
    assert result.candidate is not None and type(result.candidate) is bp.NumpyContinuousBCLearner
    assert initial.completed_updates == 1 and initial.training_examples == 2
    assert any(np.any(value != 0.0) for value in initial.velocity)
    assert_same(owner.snapshot_state(), initial)
    assert_same(saved["runtime"].actor.snapshot_state().state, initial)
    assert_same(saved["seed"].snapshot_state(), saved["seed_initial"])
    for actual, original in zip(
        saved["seed_initial"].parameters + saved["seed_initial"].velocity,
        initial.parameters + initial.velocity,
        strict=True,
    ):
        np.testing.assert_array_equal(actual, original)
    candidate = result.candidate.snapshot_state()
    assert (candidate.completed_updates, candidate.training_examples) == (9, 1186)
    assert candidate.policy.binary_gripper_axis == 3 and owner.policy.binary_gripper_axis is None
    assert candidate.policy.learning_rate == initial.policy.learning_rate
    assert candidate.policy.momentum == initial.policy.momentum
    assert (
        candidate.policy.observation_standardization == initial.policy.observation_standardization
    )
    assert all(np.isfinite(value).all() for value in candidate.parameters + candidate.velocity)
    actor = saved["runtime"]._actor._learner._learner
    assert not any(
        np.shares_memory(a, b)
        for a in result.candidate._state.parameters + result.candidate._state.velocity
        for b in owner._state.parameters
        + owner._state.velocity
        + saved["seed"]._state.parameters
        + saved["seed"]._state.velocity
        + actor._state.parameters
        + actor._state.velocity
    )
    assert tuple(asdict(item) for item in (training, *references)) == input_seal


def test_should_train_every_original_row_in_order_at_equal_weight_in_each_of_eight_updates(
    completed,
):
    result, _, _, training, references, _, observed = completed
    report = result.report
    assert observed == [((148, 12), (148, 4))] * 8
    assert report["groups"] == (
        ("training", 0, 50),
        ("reference_0", 50, 68),
        ("reference_1", 68, 83),
        ("reference_2", 83, 133),
        ("reference_3", 133, 148),
    )
    assert report["rows"] == 148
    assert report["work"]["new_training_rows"] == 400
    assert report["work"]["reference_training_rows"] == 784
    assert report["work"]["group_training_exposures"] == dict(
        training=400, reference_0=144, reference_1=120, reference_2=400, reference_3=120
    )
    expected_bindings = [
        dict(
            group="training" if index == 0 else f"reference_{index - 1}",
            episode_id=value.episode_id,
            context=value.context,
            role="train",
            step=row.step,
        )
        for index, value in enumerate((training, *references))
        for row in value.transitions
    ]
    assert report["row_bindings"] == tuple(expected_bindings)


def test_should_preserve_eight_receipts_local_clocks_original_counters_and_native_sgd_costs(
    completed,
):
    report = completed[0].report
    assert report["initial_counts"] == dict(updates=1, training_rows=2)
    assert report["final_counts"] == dict(updates=9, training_rows=1186)
    assert report["budget"]["max_training_updates"] == 8
    assert (
        report["native_updates"]
        == report["sealed_native_updates"]
        == report["validated_native_updates"]
    )
    assert report["applied_updates"] == report["sealed_applied_updates"]
    assert (
        len(report["native_updates"])
        == len(report["applied_updates"])
        == len(report["events"])
        == 8
    )
    for index, (receipt, native, event) in enumerate(
        zip(report["applied_updates"], report["native_updates"], report["events"], strict=True)
    ):
        assert event["deferred_exact"] is event["success_validated"] is True
        assert (receipt["observed_at"], receipt["arrived_at"], receipt["applied_at"]) == (
            1000 + 2 * index,
            1001 + 2 * index,
            1001 + 2 * index,
        )
        assert receipt["update_number"] == index + 1
        assert receipt["sample_id"] == f"continuation-{index}"
        assert receipt["event_id"] == f"continuation-label-{index}"
        assert receipt["diagnostic"] == native["diagnostic"]
        assert native["diagnostic"]["definition"] == core.MIXED_OBJECTIVE
        assert native["training_rows"] == 148
    assert report["progress"] == dict(
        updates_completed=8, replay_examples_completed=0, logical_tick=1015
    )
    work = report["work"]
    assert work["training_rows"] == 1184 and work["sgd_updates"] == 8
    assert work["head_gauss_newton_updates"] == work["prediction_calls"] == work["fits"] == 0
    assert work["inbox_feature_target_numeric_bytes"] == 151552
    counts = work["native_update_counts"]
    assert counts["published_updates"] == counts["forward_passes"] == counts["backward_passes"] == 8
    assert counts["forward_rows"] == counts["backward_rows"] == counts["normalization_rows"] == 1184
    assert counts["gradient_values"] == 304 and counts["gradient_numeric_bytes"] == 2432
    assert counts["known_dense_multiply_accumulates"] == 85248


def test_should_keep_one_latest_model_pin_and_only_bounded_scalar_success_records(completed):
    result, _, saved, *_ = completed
    assert len(saved["facade"]._records) == 8
    assert len(saved["facade"]._latest) == len(saved["runtime"]._candidate._latest) == 1
    assert result.report["work"]["named_model_owners"] == 5
    for row, payload in saved["facade"]._records:
        assert set(row) == {"diagnostic", "training_rows", "work"}
        assert json.loads(payload) == row and row["training_rows"] == 148
    json.dumps(result.report, allow_nan=False)


@pytest.mark.parametrize("rule,count", [("bp_mixed", 149), ("head_gauss_newton", 131)])
def test_should_refuse_rule_specific_overflow_before_any_private_fork_or_native_math(
    owner, monkeypatch, rule, count
):
    training = rows("oversize", count)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forbidden)
    monkeypatch.setattr(continuation, "ActorShadowRuntime", forbidden)
    with pytest.raises(ValueError):
        continuation.run_continuous_train_continuation(owner, training, update_rule=rule)


@pytest.mark.parametrize("updates", [True, False, 0, 65, 8.0])
def test_should_keep_native_integer_update_limits_before_any_private_work(
    owner, monkeypatch, updates
):
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forbidden)
    with pytest.raises(ValueError):
        continuation.run_continuous_train_continuation(
            owner, episode(), updates=updates, update_rule="bp_mixed"
        )


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test"])
def test_should_refuse_heldout_role_before_episode_copy_or_private_fork(owner, monkeypatch, role):
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forbidden)
    monkeypatch.setattr(continuation, "copy_continuous_episode", forbidden)
    with pytest.raises(ValueError):
        continuation.run_continuous_train_continuation(
            owner, replace(episode(), role=role), update_rule="bp_mixed"
        )


@pytest.mark.parametrize("fault", ["unavailable", "prefix"])
def test_should_retain_two_paid_148_row_updates_and_refuse_more_work_after_late_failure_or_corruption(
    owner, monkeypatch, fault
):
    initial = owner.snapshot_state()
    saved = capture(monkeypatch)
    training, references = inputs()
    calls: list[int] = []
    changed: list[bool] = []
    train = continuation.RecordedContinuousMixed.train_batch
    ready = ActorShadowRuntime.train_ready

    def trained(port, features, targets):
        calls.append(len(port._reports))
        if fault == "unavailable" and len(calls) == 3:
            raise RuntimeError("third corrective update return unavailable")
        return train(port, features, targets)

    def polled(runtime, *args, **options):
        returned = ready(runtime, *args, **options)
        if (
            fault == "prefix"
            and returned == ()
            and len(runtime.applied_updates) == 2
            and not changed
        ):
            saved["facade"]._reports[0]["diagnostic"]["value"] += 0.125
            changed.append(True)
        return returned

    monkeypatch.setattr(continuation.RecordedContinuousMixed, "train_batch", trained)
    monkeypatch.setattr(ActorShadowRuntime, "train_ready", polled)
    result = continuation.run_continuous_train_continuation(
        owner, training, references=references, update_rule="bp_mixed"
    )
    report = result.report
    assert result.candidate is None and report["complete"] is False
    assert len(report["validated_native_updates"]) == len(report["applied_updates"]) == 2
    assert all(item["training_rows"] == 148 for item in report["validated_native_updates"])
    assert all(event["success_validated"] is True for event in report["events"][:2])
    assert report["progress"]["updates_completed"] == 2
    paid_candidate = saved["runtime"]._candidate._learner
    assert (paid_candidate._state.completed_updates, paid_candidate._state.training_examples) == (
        3,
        298,
    )
    if fault == "unavailable":
        assert calls == [0, 1, 2] and report["update_attempts"] == 3
        assert report["attempted_training_rows"] == 444
        assert report["unreturned_training_work_unknown"] is True
        assert "third corrective" in report["error"]
    else:
        assert calls == [0, 1] and changed == [True] and report["update_attempts"] == 2
        assert report["native_updates"][0] != report["validated_native_updates"][0]
        assert report["sealed_native_updates"] == report["validated_native_updates"]
    assert_same(owner.snapshot_state(), initial)
