"""A finite 64-update TRAIN scope with preserved earlier state and paid failures.

One H2/D12/A4 initializer consumes32 seeded normal values. Full module forecast:
91 native SGD returns/346 rows, including one two-row common-MSE source warmup;
91 gradient and forward calls/346 rows each. No prediction, GN, fit or SDK.
The helpers imported below are pure builders/assertions, never learner fixtures.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from src.adapters import continuous_train_continuation as continuation
from src.adapters import numpy_continuous_bc as bp
from src.app.actor_shadow import ActorShadowRuntime
from src.core import continuous_bc as core
from src.core.observation_standardization import ObservationStandardization
from test_continuous_mixed_continuation import assert_same, capture, episode, pairs


def forbidden(*args, **kwargs):
    pytest.fail("longer TRAIN scope admitted forbidden model work or refused-input fork")


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
            values = args[1] if kind == "SGD" else args[0]
            counts[kind + "_attempts"] += 1
            counts[kind + "_rows_attempted"] += len(values)
            result = original(*args, **options)
            counts[kind + "_returns"] += 1
            counts[kind + "_rows_returned"] += len(values)
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
    if os.environ.get("LONGER_CTC_LEDGER"):
        with Path(os.environ["LONGER_CTC_LEDGER"]).open("x", encoding="utf8") as output:
            json.dump(counts, output, sort_keys=True)
            output.write("\n")
    assert counts["initializations"] == 1 and counts["normal_draws"] == 32
    assert counts["SGD_attempts"] == counts["SGD_returns"] == 91
    assert counts["SGD_rows_attempted"] == counts["SGD_rows_returned"] == 346
    assert counts["gradient_attempts"] == counts["gradient_returns"] == 91
    assert counts["gradient_rows_attempted"] == counts["gradient_rows_returned"] == 346
    assert counts["forward_attempts"] == counts["forward_returns"] == 91
    assert counts["forward_rows_attempted"] == counts["forward_rows_returned"] == 346


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
    initial = owner.snapshot_state()
    training, reference = episode(), episode("old", gripper=1.0)
    input_seal = asdict(training), asdict(reference)
    with pytest.MonkeyPatch.context() as patch:
        saved = capture(patch)
        result = continuation.run_continuous_train_continuation(
            owner, training, references=(reference,), updates=64, update_rule="bp_mixed"
        )
    return result, owner, initial, saved, training, reference, input_seal


def test_should_complete_64_mixed_updates_with_detached_candidate_and_frozen_original_owners(
    completed,
):
    result, source, initial, saved, training, reference, input_seal = completed
    assert result.report["complete"] is True, result.report.get("error")
    assert type(result.candidate) is bp.NumpyContinuousBCLearner
    candidate = result.candidate.snapshot_state()
    assert initial.completed_updates == 1 and initial.training_examples == 2
    assert any(np.any(value != 0.0) for value in initial.velocity)
    assert_same(source.snapshot_state(), initial)
    assert_same(saved["runtime"].actor.snapshot_state().state, initial)
    assert_same(saved["seed"].snapshot_state(), saved["seed_initial"])
    assert candidate.policy.binary_gripper_axis == 3 and source.policy.binary_gripper_axis is None
    assert (
        candidate.policy.observation_standardization == initial.policy.observation_standardization
    )
    assert (candidate.completed_updates, candidate.training_examples) == (65, 258)
    assert all(np.isfinite(value).all() for value in candidate.parameters + candidate.velocity)
    assert any(
        not np.array_equal(first, second)
        for first, second in zip(candidate.parameters, initial.parameters, strict=True)
    )
    assert not any(
        np.shares_memory(first, second)
        for first in result.candidate._state.parameters + result.candidate._state.velocity
        for second in source._state.parameters
        + source._state.velocity
        + saved["seed"]._state.parameters
        + saved["seed"]._state.velocity
    )
    assert (asdict(training), asdict(reference)) == input_seal
    assert result.report["source_exact"] is result.report["actor_exact"] is True


def test_should_keep_exact_64_receipts_delayed_clocks_and_permanent_exposure_accounting(completed):
    result, _, _, saved, *_ = completed
    report = result.report
    assert report["initial_counts"] == dict(updates=1, training_rows=2)
    assert report["final_counts"] == dict(updates=65, training_rows=258)
    assert (
        report["native_updates"]
        == report["validated_native_updates"]
        == report["sealed_native_updates"]
    )
    assert report["applied_updates"] == report["sealed_applied_updates"]
    assert (
        len(report["native_updates"])
        == len(report["applied_updates"])
        == len(report["events"])
        == 64
    )
    for index, (event, receipt, native) in enumerate(
        zip(report["events"], report["applied_updates"], report["native_updates"], strict=True)
    ):
        assert event["deferred_exact"] is event["success_validated"] is True
        assert (event["observed_at"], event["arrived_at"]) == (1000 + 2 * index, 1001 + 2 * index)
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
        assert math.isfinite(native["diagnostic"]["value"])
    assert report["budget"]["max_training_updates"] == 64
    assert saved["budget"].updates_completed == 64
    assert report["progress"] == dict(
        updates_completed=64, replay_examples_completed=0, logical_tick=1127
    )
    assert report["attempted_training_rows"] == 256
    assert report["unreturned_training_work_unknown"] is False
    assert report["work"]["group_training_exposures"] == dict(training=128, reference_0=128)
    assert report["work"]["sgd_updates"] == 64 and report["work"]["head_gauss_newton_updates"] == 0
    counts = report["work"]["native_update_counts"]
    assert (
        counts["published_updates"] == counts["forward_passes"] == counts["backward_passes"] == 64
    )
    assert counts["normalization_rows"] == counts["forward_rows"] == counts["backward_rows"] == 256
    assert counts["gradient_values"] == 64 * 38 and counts["gradient_numeric_bytes"] == 64 * 38 * 8
    assert "solve_attempts" not in counts


def test_should_retain_one_latest_native_state_pin_and_scalar_records_not_64_model_snapshots(
    completed,
):
    report, saved = completed[0].report, completed[3]
    facade = saved["facade"]
    assert len(facade._records) == 64 and len(facade._latest) == 1
    assert len(saved["runtime"]._candidate._latest) == 1
    assert report["work"]["named_model_owners"] == 5
    for row, payload in facade._records:
        assert set(row) == {"diagnostic", "training_rows", "work"}
        assert json.loads(payload) == row
        assert row["training_rows"] == 4
    json.dumps(report, allow_nan=False)


def test_should_preserve_default_eight_mixed_updates_as_a_separate_scope(
    owner, completed, monkeypatch
):
    before = owner.snapshot_state()
    previous = deepcopy(completed[0].report)
    saved = capture(monkeypatch)
    result = continuation.run_continuous_train_continuation(
        owner, episode(), update_rule="bp_mixed"
    )
    assert result.report["complete"] is True, result.report.get("error")
    assert result.report["budget"]["max_training_updates"] == 8
    assert len(result.report["applied_updates"]) == 8
    assert result.report["final_counts"] == dict(updates=9, training_rows=18)
    assert saved["budget"] is not completed[3]["budget"]
    assert completed[0].report == previous
    assert_same(owner.snapshot_state(), before)


@pytest.mark.parametrize(
    "rule,updates",
    [
        ("bp_mixed", 65),
        ("head_gauss_newton", 9),
        ("bp_mixed", True),
        ("head_gauss_newton", False),
        ("bp_mixed", 0),
        ("bp_mixed", -1),
        ("bp_mixed", 64.0),
    ],
)
def test_should_refuse_invalid_or_non_native_update_limit_before_any_fork(
    owner, monkeypatch, rule, updates
):
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forbidden)
    monkeypatch.setattr(continuation, "ActorShadowRuntime", forbidden)
    with pytest.raises(ValueError):
        continuation.run_continuous_train_continuation(
            owner, episode(), updates=updates, update_rule=rule
        )


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test"])
def test_should_refuse_non_train_roles_before_episode_copy_or_fork(owner, monkeypatch, role):
    training = replace(episode(), role=role)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forbidden)
    monkeypatch.setattr(continuation, "copy_continuous_episode", forbidden)
    with pytest.raises(ValueError):
        continuation.run_continuous_train_continuation(
            owner, training, updates=64, update_rule="bp_mixed"
        )


def test_should_retain_nine_successes_when_following_facade_call_fails_without_retry(
    owner, monkeypatch
):
    before = owner.snapshot_state()
    train = continuation.RecordedContinuousMixed.train_batch
    calls: list[int] = []

    def unavailable(port, features, targets):
        calls.append(len(port._reports))
        if len(calls) == 10:
            raise RuntimeError("longer continuation next native return unavailable")
        return train(port, features, targets)

    monkeypatch.setattr(continuation.RecordedContinuousMixed, "train_batch", unavailable)
    result = continuation.run_continuous_train_continuation(
        owner,
        episode(),
        references=(episode("old", gripper=1.0),),
        updates=64,
        update_rule="bp_mixed",
    )
    report = result.report
    assert result.candidate is None and report["complete"] is False
    assert "next native return unavailable" in report["error"]
    assert calls == list(range(10))
    assert len(report["validated_native_updates"]) == len(report["applied_updates"]) == 9
    assert all(event["success_validated"] is True for event in report["events"][:9])
    assert report["progress"]["updates_completed"] == 9
    assert report["update_attempts"] == 10 and report["attempted_training_rows"] == 40
    assert report["unreturned_training_work_unknown"] is True
    assert_same(owner.snapshot_state(), before)


def test_should_refuse_changed_preceding_native_evidence_after_nine_updates_before_more_sgd(
    owner, monkeypatch
):
    before = owner.snapshot_state()
    saved = capture(monkeypatch)
    ready = ActorShadowRuntime.train_ready
    changed: list[bool] = []

    def corrupt(runtime, *args, **options):
        returned = ready(runtime, *args, **options)
        if returned == () and len(runtime.applied_updates) == 9 and not changed:
            saved["facade"]._reports[0]["diagnostic"]["value"] += 0.125
            changed.append(True)
        return returned

    monkeypatch.setattr(ActorShadowRuntime, "train_ready", corrupt)
    result = continuation.run_continuous_train_continuation(
        owner,
        episode(),
        references=(episode("old", gripper=1.0),),
        updates=64,
        update_rule="bp_mixed",
    )
    report = result.report
    assert changed == [True] and result.candidate is None and report["complete"] is False
    assert (
        len(report["native_updates"])
        == len(report["validated_native_updates"])
        == len(report["applied_updates"])
        == 9
    )
    assert report["events"][8]["success_validated"] is True
    assert report["native_updates"][0] != report["validated_native_updates"][0]
    assert report["validated_native_updates"] == report["sealed_native_updates"]
    assert report["progress"]["updates_completed"] == 9
    assert report["update_attempts"] == 9
    assert_same(owner.snapshot_state(), before)
