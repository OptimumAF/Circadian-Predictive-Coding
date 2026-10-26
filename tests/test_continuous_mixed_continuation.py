"""Opt-in mixed gripper learning with honest paid work and isolated owners.

One D12/H2/A4 initializer (32 normals) genuinely warms under original MSE.
Bias20 variants are controlled wrong-endpoint fixtures, not learned task evidence.
Forecast:11 native SGD attempts/34 rows,13 gradient calls/42 rows and15 forwards/
50 rows. No prediction, GN, normalization fit, teacher or SDK work.
"""

from dataclasses import replace
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
from src.core.continuous_demonstration import ContinuousEpisode, ContinuousTransition
from src.core.cube_taskspace_state import cube_taskspace_spec
from src.core.observation_standardization import ObservationStandardization


def forbidden(*args, **kwargs):
    pytest.fail("mixed continuation admitted GN, fitting, prediction or refused-input fork")


def episode(name="new", *, gripper=-1.0):
    def observation(index):
        return (0.0, 0.2 + index * 0.1, -0.1) + (0.0,) * 9

    return ContinuousEpisode(
        f"synthetic-{name}",
        f"context-{name}",
        "train",
        cube_taskspace_spec(),
        tuple(
            ContinuousTransition(
                i, observation(i), (0.2, -0.1, 0.1, gripper), 0.0, observation(i + 1), i == 1, False
            )
            for i in range(2)
        ),
    )


def pairs(training, reference=None):
    rows = training.transitions + (() if reference is None else reference.transitions)
    return (
        np.asarray([row.observation for row in rows], np.float64),
        np.asarray([row.action for row in rows], np.float64),
    )


def assert_same(actual, expected):
    assert actual.policy == expected.policy
    assert (actual.completed_updates, actual.training_examples) == (
        expected.completed_updates,
        expected.training_examples,
    )
    for first, second in zip(
        actual.parameters + actual.velocity, expected.parameters + expected.velocity, strict=True
    ):
        np.testing.assert_array_equal(first, second)


@pytest.fixture(scope="module", autouse=True)
def ledger():
    counts = dict(
        initializations=0,
        normal_draws=0,
        SGD_attempts=0,
        SGD_returns=0,
        SGD_rows=0,
        gradient_attempts=0,
        gradient_returns=0,
        gradient_rows=0,
        forwards=0,
        forward_rows=0,
        prediction_rows=0,
        GN=0,
        fits=0,
    )
    initialize = bp.NumpyContinuousBCLearner.__init__
    train = bp.NumpyContinuousBCLearner.train_batch
    gradient = core.continuous_bc_loss_and_gradients
    forward = core._forward

    def initialized(owner, policy):
        counts["initializations"] += 1
        counts["normal_draws"] += policy.hidden_dim * (
            policy.spec.observation_dim + len(policy.spec.action_units)
        )
        return initialize(owner, policy)

    def trained(owner, features, targets):
        counts["SGD_attempts"] += 1
        counts["SGD_rows"] += len(features)
        result = train(owner, features, targets)
        counts["SGD_returns"] += 1
        return result

    def gradients(features, *args, **options):
        counts["gradient_attempts"] += 1
        counts["gradient_rows"] += len(features)
        result = gradient(features, *args, **options)
        counts["gradient_returns"] += 1
        return result

    def forwarded(features, *args, **options):
        counts["forwards"] += 1
        counts["forward_rows"] += len(features)
        return forward(features, *args, **options)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bp.NumpyContinuousBCLearner, "__init__", initialized)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", trained)
        patch.setattr(core, "continuous_bc_loss_and_gradients", gradients)
        patch.setattr(core, "_forward", forwarded)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", forbidden)
        patch.setattr(bp.NumpyContinuousBCLearner, "predict", forbidden)
        patch.setattr(bp, "fit_training_standardization", forbidden)
        yield counts
    if os.environ.get("GRC_FIXTURE_LEDGER"):
        with Path(os.environ["GRC_FIXTURE_LEDGER"]).open("x", encoding="utf8") as output:
            json.dump(counts, output, sort_keys=True)
            output.write("\n")
    assert counts["initializations"] == 1 and counts["normal_draws"] == 32
    assert counts["SGD_attempts"] <= 64 and counts["SGD_rows"] <= 1024
    assert counts["gradient_attempts"] <= 96 and counts["forwards"] <= 96


@pytest.fixture(scope="module")
def owner(ledger):
    statistics = ObservationStandardization((0.0,) * 12, (2.0,) * 12, 2)
    source = bp.NumpyContinuousBCLearner(
        bp.ContinuousBCPolicy(
            cube_taskspace_spec(),
            hidden_dim=2,
            seed=17,
            learning_rate=0.05,
            momentum=0.2,
            observation_standardization=statistics,
        )
    )
    source.train_batch(*pairs(episode("warm", gripper=1.0)))
    assert source.snapshot_state().completed_updates == 1
    assert any(np.any(value != 0.0) for value in source.snapshot_state().velocity)
    return source


def capture(monkeypatch):
    saved: dict[str, Any] = {}
    runtime_factory = continuation.ActorShadowRuntime
    prepare = continuation.mixed_training_source

    def prepared(source):
        seed = prepare(source)
        saved.update(seed=seed, seed_initial=seed.snapshot_state())
        return seed

    def runtime(model, **options):
        result = runtime_factory(model, **options)
        saved.update(runtime=result, facade=model, clock=options["clock"], budget=options["budget"])
        return result

    monkeypatch.setattr(continuation, "mixed_training_source", prepared)
    monkeypatch.setattr(continuation, "ActorShadowRuntime", runtime)
    return saved


@pytest.fixture(scope="module")
def completed(owner):
    source = owner.fork()
    source._state.parameters[3][3] = 20.0
    initial = source.snapshot_state()
    training, reference = episode(), episode("old", gripper=1.0)
    features, targets = pairs(training, reference)
    standardized = features / 2.0
    mse = core.continuous_bc_loss_and_gradients(
        standardized, targets, initial.parameters, initial.policy.spec
    )
    mixed = core.continuous_bc_loss_and_gradients(
        standardized, targets, initial.parameters, initial.policy.spec, binary_gripper_axis=3
    )
    before_logits = core._forward(standardized, initial.parameters, initial.policy.spec)[3][
        :, 3
    ].copy()
    with pytest.MonkeyPatch.context() as patch:
        saved = capture(patch)
        result = continuation.run_continuous_train_continuation(
            source, training, references=(reference,), updates=2, update_rule="bp_mixed"
        )
    assert result.candidate is not None, result.report.get("error")
    after_logits = core._forward(
        standardized, result.candidate._state.parameters, initial.policy.spec
    )[3][:, 3].copy()
    return result, source, initial, saved, mse, mixed, before_logits, after_logits


def test_should_apply_finite_mixed_gradient_to_controlled_wrong_gripper_without_relabelling_source(
    completed,
):
    result, source, initial, saved, mse, mixed, before_logits, after_logits = completed
    assert np.isfinite(mixed.loss) and mixed.gradients[3][3] > 0.1
    assert abs(mse.gradients[3][3]) < 1e-12
    assert np.all(np.isfinite(after_logits)) and np.all(after_logits < before_logits)
    assert result.candidate._state.parameters[3][3] < initial.parameters[3][3]
    assert type(result.candidate) is bp.NumpyContinuousBCLearner
    assert result.candidate.policy.binary_gripper_axis == 3
    assert source.policy.binary_gripper_axis is None
    assert result.report["complete"] is True
    assert_same(source.snapshot_state(), initial)
    assert_same(saved["runtime"].actor.snapshot_state().state, initial)
    assert_same(saved["seed"].snapshot_state(), saved["seed_initial"])
    final = result.candidate.snapshot_state()
    assert final.policy.observation_standardization == initial.policy.observation_standardization
    assert any(
        not np.array_equal(first, second)
        for first, second in zip(final.parameters[:2], initial.parameters[:2], strict=True)
    )
    assert not any(
        np.shares_memory(first, second)
        for first in result.candidate._state.parameters + result.candidate._state.velocity
        for second in source._state.parameters
        + source._state.velocity
        + saved["seed"]._state.parameters
        + saved["seed"]._state.velocity
    )


def test_should_preserve_clocks_receipts_group_dose_and_original_initial_momentum(completed):
    result, _, initial, saved, *_ = completed
    report = result.report
    assert report["protocol"] == "continuous_train_mixed_gripper_continuation_v1"
    assert report["head_gauss_newton"] is None and report["training_rule"] == "bp_mixed"
    transition = report["policy_transition"]
    assert transition["source_objective"] == core.OBJECTIVE
    assert transition["candidate_objective"] == core.MIXED_OBJECTIVE
    assert transition["binary_gripper_axis"] == 3
    assert transition["parameters"] == "copied"
    assert (
        transition["momentum"]
        == transition["counters"]
        == transition["normalization"]
        == "preserved"
    )
    assert report["initial_counts"] == dict(updates=1, training_rows=2)
    assert report["final_counts"] == dict(updates=3, training_rows=10)
    for first, second in zip(
        saved["seed_initial"].parameters + saved["seed_initial"].velocity,
        initial.parameters + initial.velocity,
        strict=True,
    ):
        np.testing.assert_array_equal(first, second)
    assert (
        report["native_updates"]
        == report["validated_native_updates"]
        == report["sealed_native_updates"]
    )
    assert report["applied_updates"] == report["sealed_applied_updates"]
    assert len(report["native_updates"]) == 2
    for index, event in enumerate(report["events"]):
        assert event["deferred_exact"] is event["success_validated"] is True
        assert (event["observed_at"], event["arrived_at"]) == (1000 + 2 * index, 1001 + 2 * index)
    assert report["work"]["group_training_exposures"] == dict(training=4, reference_0=4)
    assert report["work"]["sgd_updates"] == 2 and report["work"]["head_gauss_newton_updates"] == 0
    assert report["work"]["named_model_owners"] == 5
    counts = report["work"]["native_update_counts"]
    assert (
        counts["gradient_evaluations_completed"]
        == counts["forward_passes"]
        == counts["backward_passes"]
        == 2
    )
    assert counts["normalization_rows"] == counts["forward_rows"] == counts["backward_rows"] == 8
    assert counts["gradient_values"] == 76 and counts["gradient_numeric_bytes"] == 608
    assert report["work"]["native_work_flags"] == dict(
        source_derived_counts=True, native_work_returned=False
    )
    assert "solve_attempts" not in counts
    assert all(
        row["diagnostic"]["definition"] == core.MIXED_OBJECTIVE for row in report["native_updates"]
    )
    assert all(
        "loss_after" not in row and "step_norm" not in row for row in report["native_updates"]
    )
    json.dumps(report, allow_nan=False)


def test_should_continue_an_already_mixed_candidate_in_a_distinct_bounded_scope(
    completed, monkeypatch
):
    source = completed[0].candidate
    initial = source.snapshot_state()
    saved = capture(monkeypatch)
    result = continuation.run_continuous_train_continuation(
        source,
        episode(),
        references=(episode("old", gripper=1.0),),
        updates=2,
        update_rule="bp_mixed",
    )
    assert result.report["complete"] is True, result.report.get("error")
    assert isinstance(result.candidate, bp.NumpyContinuousBCLearner)
    assert result.candidate.policy.binary_gripper_axis == 3
    assert result.report["initial_counts"] == dict(updates=3, training_rows=10)
    assert result.report["final_counts"] == dict(updates=5, training_rows=18)
    assert saved["budget"] is not completed[3]["budget"]
    assert completed[0].report["final_counts"] == dict(updates=3, training_rows=10)
    assert_same(source.snapshot_state(), initial)
    assert_same(saved["runtime"].actor.snapshot_state().state, initial)


@pytest.mark.parametrize(
    "fault",
    [
        "none_rule",
        "bool_rule",
        "unknown_rule",
        "role",
        "reference_role",
        "target",
        "reference_target",
        "axis_units",
    ],
)
def test_should_refuse_invalid_optin_authority_or_nonendpoint_targets_before_fork(
    owner, monkeypatch, fault
):
    source = owner.fork()
    training, reference = episode(), episode("old", gripper=1.0)
    rule: Any = "bp_mixed"
    if fault == "none_rule":
        rule = None
    elif fault == "bool_rule":
        rule = True
    elif fault == "unknown_rule":
        rule = "mixed"
    elif fault == "role":
        training = replace(training, role="probe_train")
    elif fault == "reference_role":
        reference = replace(reference, role="final_test")
    elif fault in ("target", "reference_target"):
        invalid = training if fault == "target" else reference
        rows = (
            replace(invalid.transitions[0], action=(0.2, -0.1, 0.1, 0.0)),
            invalid.transitions[1],
        )
        invalid = replace(invalid, transitions=rows)
        if fault == "target":
            training = invalid
        else:
            reference = invalid
    else:
        spec = replace(
            training.spec,
            action_units=(
                "normalized_root_dx",
                "normalized_root_dy",
                "normalized_root_dz",
                "other",
            ),
        )
        source._policy = replace(source._policy, spec=spec)
        source._state = replace(source._state, policy=source._policy)
        training, reference = replace(training, spec=spec), replace(reference, spec=spec)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forbidden)
    with pytest.raises(ValueError):
        continuation.run_continuous_train_continuation(
            source, training, references=(reference,), updates=1, update_rule=rule
        )


def test_should_keep_first_success_when_following_facade_return_is_unavailable(owner, monkeypatch):
    calls: list[bool] = []
    original = continuation.RecordedContinuousMixed.train_batch

    def stop_second(port, features, targets):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("mixed return unavailable")
        return original(port, features, targets)

    monkeypatch.setattr(continuation.RecordedContinuousMixed, "train_batch", stop_second)
    result = continuation.run_continuous_train_continuation(
        owner,
        episode(),
        references=(episode("old", gripper=1.0),),
        updates=3,
        update_rule="bp_mixed",
    )
    assert result.candidate is None and result.report["complete"] is False
    assert (
        len(calls) == 2
        and len(result.report["applied_updates"])
        == len(result.report["validated_native_updates"])
        == 1
    )
    assert result.report["events"][0]["success_validated"] is True
    assert result.report["progress"]["updates_completed"] == 1
    assert "mixed return unavailable" in result.report["error"]


@pytest.mark.parametrize("fault", ["native_prefix", "clock", "mixed_seed"])
def test_should_refuse_late_evidence_or_authority_change_without_more_sgd(
    owner, monkeypatch, fault
):
    saved = capture(monkeypatch)
    ready = ActorShadowRuntime.train_ready
    changes: list[bool] = []

    def changed(runtime, *args, **options):
        returned = ready(runtime, *args, **options)
        if returned == () and len(runtime.applied_updates) == 1 and not changes:
            if fault == "native_prefix":
                saved["facade"]._reports[0]["work"]["published_updates"] = 2
            elif fault == "clock":
                saved["clock"].advance_to(saved["clock"].now() + 1)
            else:
                saved["seed"]._state.parameters[3][0] += 0.125
            changes.append(True)
        return returned

    monkeypatch.setattr(ActorShadowRuntime, "train_ready", changed)
    reference = (episode("old", gripper=1.0),) if fault == "native_prefix" else ()
    result = continuation.run_continuous_train_continuation(
        owner,
        episode(),
        references=reference,
        updates=3 if fault == "native_prefix" else 1,
        update_rule="bp_mixed",
    )
    assert changes == [True] and result.candidate is None and result.report["complete"] is False
    assert (
        len(result.report["validated_native_updates"]) == len(result.report["applied_updates"]) == 1
    )
    assert result.report["events"][0]["success_validated"] is True
    assert result.report["validated_native_updates"][0]["work"]["published_updates"] == 1


def test_should_use_private_native_birth_instead_of_a_public_snapshot_callback(owner, monkeypatch):
    source = owner.fork()
    initial = source.snapshot_state()
    monkeypatch.setattr(source, "snapshot_state", forbidden)
    diagnostic = source.train_batch(*pairs(episode("native-only", gripper=1.0)))
    final = bp._snapshot(source._state, source._policy)
    assert diagnostic.definition == core.OBJECTIVE
    assert (final.completed_updates, final.training_examples) == (
        initial.completed_updates + 1,
        initial.training_examples + 2,
    )
    assert_same(owner.snapshot_state(), initial)


def test_should_preserve_observed_mid_gradient_state_replacement_without_algorithm_commit(
    owner, monkeypatch
):
    source = owner.fork()
    initial = source.snapshot_state()
    native = source._state
    gradient = core.continuous_bc_loss_and_gradients
    replacements: list[Any] = []

    def changed(*args, **options):
        returned = gradient(*args, **options)
        old = source._state.velocity
        velocity = (old[0].copy(), old[1].copy(), old[2].copy(), old[3].copy())
        velocity[0][0, 0] += 0.125
        replacement = replace(source._state, velocity=velocity)
        source._state = replacement
        replacements.append(replacement)
        return returned

    monkeypatch.setattr(core, "continuous_bc_loss_and_gradients", changed)
    with pytest.raises(bp.TrainingUpdateFailed) as caught:
        source.train_batch(*pairs(episode("native-only", gripper=1.0)))
    assert (
        len(replacements) == 1 and source._state is replacements[0] and source._state is not native
    )
    assert source._state.completed_updates == initial.completed_updates
    assert source._state.training_examples == initial.training_examples
    assert source._state.velocity[0][0, 0] == initial.velocity[0][0, 0] + 0.125
    assert (
        caught.value.work["gradient_evaluations_attempted"]
        == caught.value.work["gradient_evaluations_completed"]
        == 1
    )
    assert caught.value.work["published_updates"] == 0
    assert_same(owner.snapshot_state(), initial)
