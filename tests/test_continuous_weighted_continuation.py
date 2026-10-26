"""Fixed positive TRAIN weights change mass, preserving rows and paid evidence.

Forecast:24 cases, one H2/D12/A4 initializer/32 normal values,21 native SGD/
gradient attempts226rows with20returns222rows;21forward returns226rows. An
actual CE overflow retains failed work. One unavailable wrapper return occurs
after genuine native work; its second update is deliberately not a trusted receipt.
No prediction, head solve, fitting, teacher, SDK or scientific payload is used.
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
from src.core.observation_standardization import ObservationStandardization
from test_continuous_mixed_continuation import assert_same, capture, episode, pairs
from test_continuous_mixed_correction_fit import inputs as full_inputs


def forbidden(*args, **options):
    pytest.fail("weighted continuation admitted refused-input work, prediction, GN or fitting")


@pytest.fixture(scope="module", autouse=True)
def ledger():
    counts = dict(initializations=0, normal_draws=0)
    for kind in ("SGD", "gradient", "forward"):
        for name in ("attempts", "returns", "rows_attempted", "rows_returned"):
            counts[f"{kind}_{name}"] = 0
    initialize = bp.NumpyContinuousBCLearner.__init__

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
            returned = original(*args, **options)
            counts[kind + "_returns"] += 1
            counts[kind + "_rows_returned"] += len(features)
            return returned

        return call

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bp.NumpyContinuousBCLearner, "__init__", initialized)
        patch.setattr(
            bp.NumpyContinuousBCLearner,
            "train_batch",
            counted(bp.NumpyContinuousBCLearner.train_batch, "SGD"),
        )
        patch.setattr(
            core,
            "continuous_bc_loss_and_gradients",
            counted(core.continuous_bc_loss_and_gradients, "gradient"),
        )
        patch.setattr(core, "_forward", counted(core._forward, "forward"))
        patch.setattr(bp.NumpyContinuousBCLearner, "predict", forbidden)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", forbidden)
        patch.setattr(bp, "fit_training_standardization", forbidden)
        yield counts
    if os.environ.get("WMC_FIXTURE_LEDGER"):
        with Path(os.environ["WMC_FIXTURE_LEDGER"]).open("x", encoding="utf8") as output:
            json.dump(counts, output, sort_keys=True)
            output.write("\n")
    assert counts["initializations"] == 1 and counts["normal_draws"] == 32
    for kind in ("SGD", "gradient", "forward"):
        assert counts[kind + "_attempts"] == 21
        assert counts[kind + "_rows_attempted"] == 226
        assert counts[kind + "_returns"] == (21 if kind == "forward" else 20)
        assert counts[kind + "_rows_returned"] == (226 if kind == "forward" else 222)


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
    training, reference = episode(), episode("old", gripper=1.0)
    initial = owner.snapshot_state()
    weights = np.asarray([1.0, 3.0, 2.0, 4.0], np.float64)
    original = tuple(asdict(value) for value in (training, reference))
    with pytest.MonkeyPatch.context() as patch:
        saved = capture(patch)
        result = continuation.run_continuous_train_continuation(
            owner,
            training,
            references=(reference,),
            updates=2,
            update_rule="bp_mixed",
            row_weights=weights,
        )
    manual = continuation.mixed_training_source(owner)
    features, targets = pairs(training, reference)
    for _ in range(2):
        manual.train_batch(features, targets, row_weights=weights)
    return result, initial, saved, weights, training, reference, original, manual


def test_should_match_native_weighted_sgd_with_inherited_momentum_and_frozen_normalization(
    owner,
    completed,
):
    result, initial, saved, weights, training, reference, original, manual = completed
    assert result.report["complete"] is True, result.report.get("error")
    assert result.candidate is not None
    assert_same(result.candidate.snapshot_state(), manual.snapshot_state())
    assert_same(owner.snapshot_state(), initial)
    assert_same(saved["runtime"].actor.snapshot_state().state, initial)
    assert_same(saved["seed"].snapshot_state(), saved["seed_initial"])
    assert initial.completed_updates == 1 and initial.training_examples == 2
    assert any(np.any(value != 0.0) for value in initial.velocity)
    final = result.candidate.snapshot_state()
    assert (final.completed_updates, final.training_examples) == (3, 10)
    assert final.policy.binary_gripper_axis == 3 and initial.policy.binary_gripper_axis is None
    assert final.policy.observation_standardization == initial.policy.observation_standardization
    actor = saved["runtime"]._actor._learner._learner
    for other in (owner, saved["seed"], actor, saved["runtime"]._candidate._learner):
        assert result.candidate is not other
        assert not any(
            np.shares_memory(first, second)
            for first in result.candidate._state.parameters + result.candidate._state.velocity
            for second in other._state.parameters + other._state.velocity
        )
    np.testing.assert_array_equal(weights, [1.0, 3.0, 2.0, 4.0])
    assert not np.shares_memory(weights, saved["runtime"]._candidate._row_weights)
    assert tuple(asdict(value) for value in (training, reference)) == original


def test_should_retain_raw_weights_row_order_receipts_clocks_and_actual_exposure_costs(completed):
    result, _, saved, weights, *_ = completed
    report = result.report
    assert report["protocol"] == "continuous_train_row_weighted_mixed_gripper_continuation_v1"
    assert report["row_weighting"] == dict(
        definition=core.WEIGHTED_MIXED_OBJECTIVE,
        values=weights.tolist(),
        normalized_masses=[0.1, 0.3, 0.2, 0.4],
        row_order="training then references in declared order; see row_bindings",
        scope="positive TRAIN supervision; exposure counters count actual rows",
    )
    assert report["groups"] == (("training", 0, 2), ("reference_0", 2, 4))
    assert report["initial_counts"] == dict(updates=1, training_rows=2)
    assert report["final_counts"] == dict(updates=3, training_rows=10)
    assert (
        report["native_updates"]
        == report["sealed_native_updates"]
        == report["validated_native_updates"]
    )
    assert report["applied_updates"] == report["sealed_applied_updates"]
    assert len(report["native_updates"]) == len(report["events"]) == 2
    for index, (native, receipt, event) in enumerate(
        zip(
            report["native_updates"],
            report["applied_updates"],
            report["events"],
            strict=True,
        )
    ):
        assert set(native) == {"diagnostic", "training_rows", "work", "objective_row_weights"}
        assert native["objective_row_weights"] == weights.tolist()
        assert native["diagnostic"]["definition"] == core.WEIGHTED_MIXED_OBJECTIVE
        assert receipt["diagnostic"] == native["diagnostic"]
        assert native["training_rows"] == 4
        assert (receipt["observed_at"], receipt["arrived_at"], receipt["applied_at"]) == (
            1000 + 2 * index,
            1001 + 2 * index,
            1001 + 2 * index,
        )
        assert event["deferred_exact"] is event["success_validated"] is True
        assert native["work"]["objective_weight_values"] == 4
        assert native["work"]["objective_weight_numeric_bytes"] == 32
        assert native["work"]["native_row_weight_mass_admissions"] == 2
    work = report["work"]
    assert work["training_rows"] == 8
    assert work["group_training_exposures"] == dict(training=4, reference_0=4)
    assert work["objective_weight_values_per_update"] == 4
    assert work["retained_receipt_objective_weight_values"] == 8
    assert work["retained_receipt_objective_weight_numeric_bytes"] == 64
    assert work["native_update_counts"]["objective_weight_values"] == 8
    assert work["native_update_counts"]["objective_weight_numeric_bytes"] == 64
    assert work["native_update_counts"]["native_row_weight_mass_admissions"] == 4
    assert work["head_gauss_newton_updates"] == work["prediction_calls"] == work["fits"] == 0
    assert len(saved["facade"]._records) == 2 and len(saved["facade"]._latest) == 1
    assert report["progress"] == dict(
        updates_completed=2, replay_examples_completed=0, logical_tick=1003
    )


def test_should_keep_uniform_weights_bitwise_equal_and_default_native_call_shape(
    owner, monkeypatch
):
    training, reference = episode(), episode("old", gripper=1.0)
    native = bp.NumpyContinuousBCLearner.train_batch
    options_seen: list[set[str]] = []

    def trained(source, features, targets, **options):
        options_seen.append(set(options))
        return native(source, features, targets, **options)

    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "train_batch", trained)
    plain = continuation.run_continuous_train_continuation(
        owner,
        training,
        references=(reference,),
        updates=2,
        update_rule="bp_mixed",
    )
    weighted = continuation.run_continuous_train_continuation(
        owner,
        training,
        references=(reference,),
        updates=2,
        update_rule="bp_mixed",
        row_weights=np.ones(4, np.float64),
    )
    assert plain.report["complete"] is weighted.report["complete"] is True
    assert plain.candidate is not None and weighted.candidate is not None
    assert_same(plain.candidate.snapshot_state(), weighted.candidate.snapshot_state())
    assert options_seen == [set(), set(), {"row_weights"}, {"row_weights"}]
    assert "row_weighting" not in plain.report
    assert plain.report["protocol"] == "continuous_train_mixed_gripper_continuation_v1"
    for native_row in plain.report["native_updates"]:
        assert "objective_row_weights" not in native_row
        assert "objective_weight_values" not in native_row["work"]
        assert native_row["diagnostic"]["definition"] == core.MIXED_OBJECTIVE
    assert weighted.report["row_weighting"]["normalized_masses"] == [0.25] * 4


def test_should_keep_all148_original_rows_and_groups_in_one_weighted_update(owner):
    training, references = full_inputs()
    initial = owner.snapshot_state()
    weights = np.asarray([float(index % 3 + 1) for index in range(148)], np.float64)
    result = continuation.run_continuous_train_continuation(
        owner,
        training,
        references=references,
        updates=1,
        update_rule="bp_mixed",
        row_weights=weights,
    )
    assert result.report["complete"] is True, result.report.get("error")
    assert result.candidate is not None
    assert result.report["rows"] == 148 and len(result.report["row_bindings"]) == 148
    assert result.report["native_updates"][0]["objective_row_weights"] == weights.tolist()
    assert result.report["native_updates"][0]["training_rows"] == 148
    assert result.report["work"]["group_training_exposures"] == dict(
        training=50,
        reference_0=18,
        reference_1=15,
        reference_2=50,
        reference_3=15,
    )
    assert result.report["final_counts"] == dict(updates=2, training_rows=150)
    assert_same(owner.snapshot_state(), initial)


@pytest.mark.parametrize(
    "fault",
    [
        "head",
        "default_head",
        "list",
        "float32",
        "zero",
        "shape",
        "underflow",
        "probe_train",
        "probe_test",
        "final_test",
    ],
)
def test_should_refuse_inadmissible_weights_or_roles_before_private_fork(owner, monkeypatch, fault):
    training = episode()
    weights: Any = np.asarray([1.0, 2.0], np.float64)
    options: dict[str, Any] = dict(update_rule="bp_mixed")
    if fault == "head":
        options["update_rule"] = "head_gauss_newton"
    elif fault == "default_head":
        options.clear()
    elif fault == "list":
        weights = [1.0, 2.0]
    elif fault == "float32":
        weights = weights.astype(np.float32)
    elif fault == "zero":
        weights[0] = 0.0
    elif fault == "shape":
        weights = weights.reshape(2, 1)
    elif fault == "underflow":
        weights = np.asarray([np.nextafter(0.0, 1.0), np.finfo(np.float64).max], np.float64)
    else:
        training = replace(training, role=fault)
        monkeypatch.setattr(continuation, "copy_continuous_episode", forbidden)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forbidden)
    monkeypatch.setattr(continuation, "ActorShadowRuntime", forbidden)
    with pytest.raises(ValueError):
        continuation.run_continuous_train_continuation(
            owner, training, row_weights=weights, **options
        )


@pytest.mark.parametrize(
    "fault",
    [
        "caller",
        "private_content",
        "private_owner",
        "metadata",
        "receipt",
        "latest",
        "source",
    ],
)
def test_should_refuse_changed_weights_or_paid_evidence_before_another_update(
    owner, monkeypatch, fault
):
    source = owner.fork()
    initial = source.snapshot_state()
    saved = capture(monkeypatch)
    weights = np.asarray([1.0, 3.0, 2.0, 4.0], np.float64)
    metadata: list[dict[str, Any]] = []
    serialize, ready = continuation._json, ActorShadowRuntime.train_ready
    changed: list[bool] = []

    def serialized(value):
        if type(value) is dict and "row_weighting" in value:
            if not metadata:
                metadata.append(value["row_weighting"])
        return serialize(value)

    def polled(runtime, *args, **options):
        returned = ready(runtime, *args, **options)
        if returned == () and len(runtime.applied_updates) == 1 and not changed:
            port = runtime._candidate
            if fault == "caller":
                weights[0] += 0.125
            elif fault == "private_content":
                port._row_weights[0] += 0.125
            elif fault == "private_owner":
                port._row_weights = port._row_weights.copy()
            elif fault == "metadata":
                assert metadata
                metadata[0]["values"][0] += 0.125
            elif fault == "receipt":
                saved["facade"]._reports[0]["objective_row_weights"][0] += 0.125
            elif fault == "latest":
                port._learner._state.parameters[3][0] += 0.125
            else:
                source._state.parameters[3][0] += 0.125
            changed.append(True)
        return returned

    monkeypatch.setattr(continuation, "_json", serialized)
    monkeypatch.setattr(ActorShadowRuntime, "train_ready", polled)
    result = continuation.run_continuous_train_continuation(
        source,
        episode(),
        references=(episode("old", gripper=1.0),),
        updates=3,
        update_rule="bp_mixed",
        row_weights=weights,
    )
    report = result.report
    assert changed == [True] and result.candidate is None and report["complete"] is False
    assert report["update_attempts"] == 1
    assert len(report["validated_native_updates"]) == len(report["applied_updates"]) == 1
    assert report["validated_native_updates"][0]["objective_row_weights"] == [1.0, 3.0, 2.0, 4.0]
    assert report["events"][0]["success_validated"] is True
    assert report["progress"]["updates_completed"] == 1
    if fault == "source":
        assert source._state.parameters[3][0] == initial.parameters[3][0] + 0.125
    else:
        assert_same(source.snapshot_state(), initial)


def test_should_retain_only_trusted_prefix_when_a_second_actual_native_return_is_unavailable(
    owner,
    monkeypatch,
):
    source = owner.fork()
    initial = source.snapshot_state()
    saved = capture(monkeypatch)
    train = bp.NumpyContinuousBCLearner.train_batch
    calls: list[int] = []

    def unavailable(native, features, targets, **options):
        diagnostic = train(native, features, targets, **options)
        calls.append(len(features))
        if len(calls) == 2:
            raise RuntimeError("weighted native return unavailable after actual SGD")
        return diagnostic

    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "train_batch", unavailable)
    result = continuation.run_continuous_train_continuation(
        source,
        episode(),
        references=(episode("old", gripper=1.0),),
        updates=3,
        update_rule="bp_mixed",
        row_weights=np.asarray([1.0, 3.0, 2.0, 4.0], np.float64),
    )
    report = result.report
    assert calls == [4, 4] and result.candidate is None and report["complete"] is False
    assert "weighted native return unavailable" in report["error"]
    assert report["update_attempts"] == 2 and report["attempted_training_rows"] == 8
    assert report["unreturned_training_work_unknown"] is True
    assert len(report["validated_native_updates"]) == len(report["applied_updates"]) == 1
    assert report["progress"]["updates_completed"] == 1
    actual = saved["runtime"]._candidate._learner._state
    assert (actual.completed_updates, actual.training_examples) == (3, 10)
    assert_same(source.snapshot_state(), initial)


def test_should_retain_birth_weights_and_paid_native_work_when_configuration_changes_during_return(
    owner,
    monkeypatch,
):
    source = owner.fork()
    initial = source.snapshot_state()
    saved = capture(monkeypatch)
    train = bp.NumpyContinuousBCLearner.train_batch
    intended = [1.0, 3.0, 2.0, 4.0]

    def changed(native, features, targets, **options):
        result = train(native, features, targets, **options)
        options["row_weights"][0] += 0.125
        return result

    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "train_batch", changed)
    result = continuation.run_continuous_train_continuation(
        source,
        episode(),
        references=(episode("old", gripper=1.0),),
        updates=3,
        update_rule="bp_mixed",
        row_weights=np.asarray(intended, np.float64),
    )
    report = result.report
    assert result.candidate is None and report["complete"] is False
    assert report["update_attempts"] == 1 and report["attempted_training_rows"] == 4
    assert report["validated_native_updates"] == report["applied_updates"] == []
    assert len(report["native_updates"]) == 1
    paid = report["native_updates"][0]
    assert paid["objective_row_weights"] == report["row_weighting"]["values"] == intended
    assert paid["work"]["published_updates"] == 1
    assert report["unreturned_training_work_unknown"] is False
    actual = saved["runtime"]._candidate._learner._state
    assert (actual.completed_updates, actual.training_examples) == (2, 6)
    assert_same(source.snapshot_state(), initial)


def test_should_preserve_actual_native_weighted_overflow_definition_and_failed_work(owner):
    source = owner.fork()
    source._state.parameters[3][3] = 1e308
    initial = source.snapshot_state()
    result = continuation.run_continuous_train_continuation(
        source,
        episode(),
        references=(episode("old", gripper=1.0),),
        updates=2,
        update_rule="bp_mixed",
        row_weights=np.asarray([1.0, 3.0, 2.0, 4.0], np.float64),
    )
    report = result.report
    assert result.candidate is None and report["complete"] is False
    assert report["failed_native_definition"] == core.WEIGHTED_MIXED_OBJECTIVE
    assert report["failed_native_work"] == dict(
        gradient_evaluations_attempted=1,
        gradient_evaluations_completed=0,
        parameter_proposals_attempted=0,
        parameter_proposals_completed=0,
        published_updates=0,
    )
    assert report["update_attempts"] == 1 and report["attempted_training_rows"] == 4
    assert report["native_updates"] == report["applied_updates"] == []
    assert_same(source.snapshot_state(), initial)
