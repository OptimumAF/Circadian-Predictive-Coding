"""Explicit198-row TRAIN continuation and source-bound current-packet intake.

Forecast:38 cases, one H2/D12/A4 initializer/32 normal values,13 native SGD,
gradient and forward returns on826 rows,37 native forks and one restore. All
input receipts are synthetic; no prediction, GN, fitting, teacher or SDK work.
"""

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import push_cube_correction_inputs as intake
from src.adapters import continuous_mixed_training as mixed
from src.adapters import continuous_train_continuation as continuation
from src.adapters import numpy_continuous_bc as bp
from src.app.actor_shadow import ActorShadowRuntime
from src.core import continuous_bc as core
from src.core.observation_standardization import ObservationStandardization
from test_continuous_mixed_continuation import assert_same, capture, episode, pairs
from test_continuous_mixed_correction_fit import rows
from test_push_cube_correction_inputs import synthetic_release


def forbidden(*args, **options):
    pytest.fail("expanded continuation admitted refused-input work, prediction, GN or fitting")


@pytest.fixture(scope="module", autouse=True)
def ledger():
    counts = dict(initializations=0, normal_draws=0, forks=0, restores=0)
    initialize = bp.NumpyContinuousBCLearner.__init__
    fork, restore = bp.NumpyContinuousBCLearner.fork, bp.NumpyContinuousBCLearner.restore_state
    for kind in ("SGD", "gradient", "forward"):
        for name in ("attempts", "returns", "rows_attempted", "rows_returned"):
            counts[kind + "_" + name] = 0

    def initialized(owner, policy):
        counts["initializations"] += 1
        counts["normal_draws"] += policy.hidden_dim * (
            policy.spec.observation_dim + len(policy.spec.action_units)
        )
        return initialize(owner, policy)

    def forked(owner):
        result = fork(owner)
        counts["forks"] += 1
        return result

    def restored(owner, state):
        result = restore(owner, state)
        counts["restores"] += 1
        return result

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
        patch.setattr(bp.NumpyContinuousBCLearner, "fork", forked)
        patch.setattr(bp.NumpyContinuousBCLearner, "restore_state", restored)
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
    if os.environ.get("E198_FIXTURE_LEDGER"):
        with Path(os.environ["E198_FIXTURE_LEDGER"]).open("x", encoding="utf8") as output:
            json.dump(counts, output, sort_keys=True)
            output.write("\n")
    assert counts["initializations"] == 1 and counts["normal_draws"] == 32
    assert counts["forks"] == 37 and counts["restores"] == 1
    for kind in ("SGD", "gradient", "forward"):
        assert counts[kind + "_attempts"] == counts[kind + "_returns"] == 13
        assert counts[kind + "_rows_attempted"] == counts[kind + "_rows_returned"] == 826


@pytest.fixture(scope="module")
def owner(ledger):
    source = bp.NumpyContinuousBCLearner(
        bp.ContinuousBCPolicy(
            episode().spec,
            hidden_dim=2,
            seed=17,
            learning_rate=0.05,
            momentum=0.2,
            observation_standardization=ObservationStandardization((0.0,) * 12, (2.0,) * 12, 2),
        )
    )
    source.train_batch(*pairs(episode("warm", gripper=1.0)))
    return source


def full_inputs():
    return replace(rows("current", 50), episode_id="push-cube-correction-targets-17-source120"), (
        replace(rows("original", 50), episode_id="push-cube-correction-targets-17"),
        rows("expert", 18),
        rows("a", 15, gripper=1.0),
        rows("b", 50, gripper=1.0),
        rows("train20", 15, gripper=1.0),
    )


@pytest.fixture(scope="module")
def completed(owner):
    training, references = full_inputs()
    before = owner.snapshot_state()
    input_seal = tuple(asdict(value) for value in (training, *references))
    all_rows = tuple(row for value in (training, *references) for row in value.transitions)
    features = np.asarray([row.observation for row in all_rows], np.float64)
    targets = np.asarray([row.action for row in all_rows], np.float64)
    weights = np.asarray([float(index % 3 + 1) for index in range(198)], np.float64)
    train = bp.NumpyContinuousBCLearner.train_batch
    seen: list[int] = []

    def trained(source, actual_features, actual_targets, **options):
        np.testing.assert_array_equal(actual_features, features)
        np.testing.assert_array_equal(actual_targets, targets)
        np.testing.assert_array_equal(options["row_weights"], weights)
        seen.append(len(actual_features))
        return train(source, actual_features, actual_targets, **options)

    with pytest.MonkeyPatch.context() as patch:
        saved = capture(patch)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", trained)
        result = continuation.run_continuous_train_continuation(
            owner,
            training,
            references=references,
            updates=2,
            update_rule="bp_mixed",
            maximum_rows=198,
            row_weights=weights,
        )
    manual = mixed.mixed_training_source(owner)
    for _ in range(2):
        manual.train_batch(features, targets, row_weights=weights)
    return result, before, saved, training, references, input_seal, weights, manual, seen


def test_should_match_two_native_198_row_updates_without_changing_source_actor_or_normalizer(
    owner, completed
):
    result, before, saved, training, references, seal, weights, manual, seen = completed
    assert result.report["complete"] is True, result.report.get("error")
    assert result.candidate is not None and seen == [198, 198]
    assert_same(result.candidate.snapshot_state(), manual.snapshot_state())
    assert_same(owner.snapshot_state(), before)
    assert_same(saved["runtime"].actor.snapshot_state().state, before)
    assert_same(saved["seed"].snapshot_state(), saved["seed_initial"])
    assert any(np.any(value != 0.0) for value in before.velocity)
    assert (
        result.candidate.policy.observation_standardization
        == before.policy.observation_standardization
    )
    assert result.candidate.policy.momentum == before.policy.momentum
    assert result.candidate.policy.binary_gripper_axis == 3
    assert result.report["final_counts"] == dict(updates=3, training_rows=398)
    actor = saved["runtime"]._actor._learner._learner
    for other in (owner, actor, saved["seed"], saved["runtime"]._candidate._learner):
        assert result.candidate is not other
        assert not any(
            np.shares_memory(a, b)
            for a in result.candidate._state.parameters + result.candidate._state.velocity
            for b in other._state.parameters + other._state.velocity
        )
    assert tuple(asdict(value) for value in (training, *references)) == seal
    np.testing.assert_array_equal(weights, [float(index % 3 + 1) for index in range(198)])


def test_should_keep_all_six_declared_groups_raw_weights_receipts_and_actual_row_costs(completed):
    result, _, saved, training, references, _, weights, *_ = completed
    report = result.report
    assert report["row_scope"] == dict(
        maximum_rows=198,
        default_maximum_rows=148,
        scope="explicit expanded released TRAIN inputs; no authority or update-budget renewal",
    )
    assert report["groups"] == (
        ("training", 0, 50),
        ("reference_0", 50, 100),
        ("reference_1", 100, 118),
        ("reference_2", 118, 133),
        ("reference_3", 133, 183),
        ("reference_4", 183, 198),
    )
    assert report["row_bindings"] == tuple(
        dict(
            group="training" if index == 0 else f"reference_{index - 1}",
            episode_id=value.episode_id,
            context=value.context,
            role="train",
            step=row.step,
        )
        for index, value in enumerate((training, *references))
        for row in value.transitions
    )
    assert (
        report["native_updates"]
        == report["sealed_native_updates"]
        == report["validated_native_updates"]
    )
    assert report["applied_updates"] == report["sealed_applied_updates"]
    for index, (native, receipt, event) in enumerate(
        zip(report["native_updates"], report["applied_updates"], report["events"], strict=True)
    ):
        assert (
            native["training_rows"] == 198 and native["objective_row_weights"] == weights.tolist()
        )
        assert receipt["diagnostic"] == native["diagnostic"]
        assert native["diagnostic"]["definition"] == core.WEIGHTED_MIXED_OBJECTIVE
        assert event["deferred_exact"] is event["success_validated"] is True
        assert (receipt["observed_at"], receipt["arrived_at"], receipt["applied_at"]) == (
            1000 + 2 * index,
            1001 + 2 * index,
            1001 + 2 * index,
        )
    work = report["work"]
    assert work["training_rows"] == 396
    assert work["group_training_exposures"] == dict(
        training=100,
        reference_0=100,
        reference_1=36,
        reference_2=30,
        reference_3=100,
        reference_4=30,
    )
    assert work["inbox_feature_target_numeric_bytes"] == 50688
    assert work["native_update_counts"]["gradient_values"] == 76
    assert work["native_update_counts"]["known_dense_multiply_accumulates"] == 28512
    assert work["retained_receipt_objective_weight_values"] == 396
    assert work["retained_receipt_objective_weight_numeric_bytes"] == 3168
    assert report["budget"]["max_training_updates"] == 2
    assert report["progress"] == dict(
        updates_completed=2, replay_examples_completed=0, logical_tick=1003
    )
    assert len(saved["facade"]._records) == 2 and len(saved["facade"]._latest) == 1
    assert len(saved["facade"]._original) == 5


def test_should_require_explicit_198_scope_to_aggregate_original_198_row_receipts(completed):
    native = completed[0].report["native_updates"]
    with pytest.raises(ValueError):
        mixed.mixed_native_work(native)
    counts, flags, _ = mixed.mixed_native_work(native, maximum_rows=198)
    assert counts["published_updates"] == 2
    assert counts["objective_weight_values"] == 396
    assert counts["objective_weight_numeric_bytes"] == 3168
    assert counts["native_row_weight_mass_admissions"] == 4
    assert flags == dict(source_derived_counts=True, native_work_returned=False)


@pytest.mark.parametrize("weighted", [False, True])
def test_should_propagate_only_selected_cap_and_owned_weights_through_rng_free_facade_forks(
    owner, weighted
):
    seed = mixed.mixed_training_source(owner)
    weights = np.ones(198, np.float64) if weighted else None
    port = mixed.RecordedContinuousMixed(seed, [], 2, maximum_rows=198, row_weights=weights)
    child = port.fork()
    assert child._maximum_rows == child._maximum_rows_original == 198
    assert child._reports is port._reports and len(child._original) == 5
    assert_same(child.snapshot_state(), port.snapshot_state())
    if weighted:
        np.testing.assert_array_equal(child._row_weights, weights)
        assert not np.shares_memory(child._row_weights, port._row_weights)
        assert not np.shares_memory(port._row_weights, weights)
    assert mixed.RecordedContinuousMixed(seed, [], 1)._maximum_rows == 148


@pytest.mark.parametrize(
    "fault",
    [
        "bool",
        "float",
        "149scope",
        "199scope",
        "head",
        "default149",
        "expanded199",
        "probe_train",
        "probe_test",
        "final_test",
        "weights",
        "references",
    ],
)
def test_should_refuse_wrong_scope_overflow_roles_or_weights_before_private_work(
    owner, monkeypatch, fault
):
    training = episode()
    references: Any = ()
    options: dict[str, Any] = dict(update_rule="bp_mixed", maximum_rows=198)
    if fault in ("bool", "float", "149scope", "199scope"):
        options["maximum_rows"] = {"bool": True, "float": 198.0, "149scope": 149, "199scope": 199}[
            fault
        ]
    elif fault == "head":
        options["update_rule"] = "head_gauss_newton"
    elif fault == "default149":
        options.pop("maximum_rows")
        training = rows("overflow", 149)
    elif fault == "expanded199":
        training = rows("overflow", 199)
    elif fault == "weights":
        options["row_weights"] = np.ones(3, np.float64)
    elif fault == "references":
        references = [episode("old")]
    else:
        training = replace(training, role=fault)
        monkeypatch.setattr(continuation, "copy_continuous_episode", forbidden)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forbidden)
    monkeypatch.setattr(continuation, "ActorShadowRuntime", forbidden)
    with pytest.raises(ValueError):
        continuation.run_continuous_train_continuation(
            owner, training, references=references, **options
        )


@pytest.mark.parametrize("fault", ["cap", "weights", "prefix", "source", "row_scope", "counter"])
def test_should_preserve_first_success_and_refuse_changed_scope_weights_or_prior_native_state(
    owner, monkeypatch, fault
):
    source = owner.fork()
    before = source.snapshot_state()
    saved = capture(monkeypatch)
    weights = np.asarray([1.0, 2.0, 3.0, 4.0], np.float64)
    scope: list[dict[str, Any]] = []
    changed: list[bool] = []
    serialize, ready = continuation._json, ActorShadowRuntime.train_ready

    def serialized(value):
        if type(value) is dict and "row_scope" in value and not scope:
            scope.append(value["row_scope"])
        return serialize(value)

    def polled(runtime, *args, **options):
        result = ready(runtime, *args, **options)
        if result == () and len(runtime.applied_updates) == 1 and not changed:
            if fault == "cap":
                runtime._candidate._maximum_rows = 148
            elif fault == "weights":
                weights[0] += 0.125
            elif fault == "prefix":
                saved["facade"]._reports[0]["diagnostic"]["value"] += 0.125
            elif fault == "source":
                source._state.parameters[3][0] += 0.125
            elif fault == "row_scope":
                assert scope
                scope[0]["maximum_rows"] = 148
            else:
                native = runtime._candidate._learner
                native._state = replace(
                    native._state, completed_updates=native._state.completed_updates + 1
                )
            changed.append(True)
        return result

    monkeypatch.setattr(continuation, "_json", serialized)
    monkeypatch.setattr(ActorShadowRuntime, "train_ready", polled)
    result = continuation.run_continuous_train_continuation(
        source,
        episode(),
        references=(episode("old", gripper=1.0),),
        updates=3,
        update_rule="bp_mixed",
        maximum_rows=198,
        row_weights=weights,
    )
    assert changed == [True] and result.candidate is None and result.report["complete"] is False
    assert result.report["update_attempts"] == 1
    assert (
        len(result.report["validated_native_updates"]) == len(result.report["applied_updates"]) == 1
    )
    assert result.report["events"][0]["success_validated"] is True
    if fault == "source":
        assert source._state.parameters[3][0] == before.parameters[3][0] + 0.125
    else:
        assert_same(source.snapshot_state(), before)


def test_should_retain_unknown_second_actual_update_without_fabricating_a_returned_receipt(
    owner, monkeypatch
):
    source = owner.fork()
    before = source.snapshot_state()
    saved = capture(monkeypatch)
    train = bp.NumpyContinuousBCLearner.train_batch
    calls: list[int] = []

    def unavailable(native, features, targets, **options):
        result = train(native, features, targets, **options)
        calls.append(len(features))
        if len(calls) == 2:
            raise RuntimeError("expanded native return unavailable after real work")
        return result

    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "train_batch", unavailable)
    result = continuation.run_continuous_train_continuation(
        source,
        episode(),
        references=(episode("old", gripper=1.0),),
        updates=3,
        update_rule="bp_mixed",
        maximum_rows=198,
        row_weights=np.ones(4, np.float64),
    )
    report = result.report
    assert calls == [4, 4] and result.candidate is None and report["complete"] is False
    assert report["unreturned_training_work_unknown"] is True
    assert report["update_attempts"] == 2 and report["attempted_training_rows"] == 8
    assert len(report["validated_native_updates"]) == len(report["applied_updates"]) == 1
    native = saved["runtime"]._candidate._learner._state
    assert (native.completed_updates, native.training_examples) == (3, 10)
    assert_same(source.snapshot_state(), before)


def current_release():
    raw = synthetic_release()
    raw["protocol"] = "push_cube_balanced_student_corrections_train_v1"
    raw["push_correction_protocol"] = dict(
        checkpoint=120, source_training_rows=11016, seed=17, role="train"
    )
    raw["push_corrections"].update(source_updates=120, source_training_rows=11016)
    return raw


def write_bound(tmp_path, monkeypatch, raw, *, current):
    path = tmp_path / ("current120.json" if current else "original112.json")
    payload = json.dumps(raw, allow_nan=False).encode("utf8")
    path.write_bytes(payload)
    monkeypatch.setattr(
        intake,
        "BALANCED_CORRECTION_SHA256" if current else "CORRECTION_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    return path, payload


def test_should_rename_only_the_validated120_packet_and_preserve_original112_identity_and_receipts(
    tmp_path, monkeypatch
):
    raw = current_release()
    path, payload = write_bound(tmp_path, monkeypatch, raw, current=True)
    old_path, old_payload = write_bound(tmp_path, monkeypatch, synthetic_release(), current=False)
    current, receipt = intake.load_balanced_push_correction_inputs(path)
    old, old_receipt = intake.load_push_correction_inputs(old_path)
    assert old.episode_id == receipt["original_episode_id"] == "push-cube-correction-targets-17"
    assert (
        current.episode_id
        == receipt["derived_episode_id"]
        == "push-cube-correction-targets-17-source120"
    )
    assert current.episode_id != old.episode_id and current.context == old.context
    assert asdict(replace(current, episode_id=old.episode_id)) == asdict(old)
    assert set(receipt) == set(old_receipt) | {"original_episode_id", "derived_episode_id"}
    assert receipt["source_updates"] == 120 and receipt["source_training_rows"] == 11016
    assert (old_receipt["source_updates"], old_receipt["source_training_rows"]) == (112, 9832)
    assert receipt["original_label_receipts"] == old_receipt["original_label_receipts"]
    assert receipt["original_label_receipts"][-1] == dict(
        step=49, observed_at=98, labels_arrived_at=99
    )
    assert current.transitions[1].observation[7:11] == (0.0, 0.0, 0.0, 1.0)
    assert current.transitions[1].action == (0.0, 0.0, 0.0, -1.0)
    assert path.read_bytes() == payload and old_path.read_bytes() == old_payload


@pytest.mark.parametrize(
    "fault",
    [
        "hash",
        "oversize",
        "protocol",
        "header",
        "counter",
        "release",
        "clock",
        "role",
        "work",
        "packet",
        "history",
        "replaced",
        "already_renamed",
    ],
)
def test_should_refuse_unbound_partial_changed_or_unarrived_current_corrections_before_renaming(
    tmp_path, monkeypatch, fault
):
    raw = current_release()
    report = raw["push_corrections"]
    if fault == "protocol":
        raw["protocol"] = "push_cube_provisional_student_corrections_train_v1"
    elif fault == "header":
        raw["push_correction_protocol"]["checkpoint"] = 112
    elif fault == "counter":
        report["source_updates"] = 112
    elif fault == "release":
        raw["push_correction_release_at"] = 98
    elif fault == "clock":
        report["label_receipts"][-1]["labels_arrived_at"] = 1101
    elif fault == "role":
        report["episode"]["role"] = "final_test"
    elif fault == "work":
        report["work"]["completed_steps"] = 49
    elif fault == "packet":
        raw["push_correction_target_packet"]["transitions"][0]["action"] = (0.125, 0.0, 0.0, -1.0)
    elif fault == "history":
        report["labels"][1]["observation"] = (0.0,) * 12
    elif fault == "already_renamed":
        raw["push_correction_target_packet"]["episode_id"] += "-source120"
    path, payload = write_bound(tmp_path, monkeypatch, raw, current=True)
    if fault == "hash":
        monkeypatch.setattr(intake, "BALANCED_CORRECTION_SHA256", "0" * 64)
    elif fault == "oversize":
        payload = b" " * (4 * 1024**2 + 1)
        path.write_bytes(payload)
        monkeypatch.setattr(
            intake, "BALANCED_CORRECTION_SHA256", hashlib.sha256(payload).hexdigest()
        )
    elif fault == "replaced":
        original = Path.open
        reads: list[bool] = []

        def opened(selected, *args, **options):
            mode = args[0] if args else options.get("mode", "r")
            if selected == path and mode == "rb":
                reads.append(True)
                if len(reads) == 2:
                    path.write_bytes(payload + b" ")
            return original(selected, *args, **options)

        monkeypatch.setattr(Path, "open", opened)
    with pytest.raises(ValueError):
        intake.load_balanced_push_correction_inputs(path)
    if fault != "replaced":
        assert path.read_bytes() == payload
    else:
        assert path.read_bytes() == payload + b" "
