"""Fixed original-A15 rehearsal control: lineage, admission and paid isolation.

One cached H2/D12/A4 initializer consumes32 seeded normal values. One real
eight-update130-row scope and two one-update34-row refusal prefixes total10GN,
1108rows,40 solves/4552 augmented rows and16 predictions/328rows. Restored
40/1856 source counters are synthetic tags, not a reconstructed scientific model.
All other cases prepare typed synthetic rows or stop before private work.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.adapters import joint_readiness_corrections as runtime
from src.adapters import joint_readiness_data as data
from src.adapters import numpy_continuous_bc as bp
from test_joint_readiness_corrections import assert_state, inputs, spec
from test_joint_train20_rehearsal import retained


def forbidden(*args, **kwargs):
    pytest.fail("invalid control admitted projection, fork, SGD or fitting")


def forbid_projection(monkeypatch):
    for name in (
        "action_history_training_episode",
        "action_history_correction_view",
        "project_curated_training_pairs",
    ):
        monkeypatch.setattr(data, name, forbidden)


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
        prediction_attempts=0,
        prediction_returns=0,
        prediction_rows_attempted=0,
        prediction_rows_returned=0,
        SGD=0,
        fits=0,
    )
    initialize = bp.NumpyContinuousBCLearner.__init__
    train = bp.NumpyContinuousBCLearner.train_head_gauss_newton_batch
    predict = bp.NumpyContinuousBCLearner.predict

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

    def predicted(owner, features):
        counts["prediction_attempts"] += 1
        counts["prediction_rows_attempted"] += len(features)
        returned = predict(owner, features)
        counts["prediction_returns"] += 1
        counts["prediction_rows_returned"] += len(returned)
        return returned

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bp.NumpyContinuousBCLearner, "__init__", initialized)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", trained)
        patch.setattr(bp.NumpyContinuousBCLearner, "predict", predicted)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", forbidden)
        patch.setattr(bp, "fit_training_standardization", forbidden)
        yield counts
    if os.environ.get("ROC_FIXTURE_LEDGER"):
        with Path(os.environ["ROC_FIXTURE_LEDGER"]).open("x", encoding="utf8") as output:
            json.dump(counts, output, sort_keys=True)
            output.write("\n")
    assert counts["initializations"] == 1 and counts["normal_draws"] == 32
    assert counts["GN_attempts"] <= 24 and counts["GN_rows_attempted"] <= 3120
    assert counts["solve_attempts"] <= 96 and counts["augmented_rows"] <= 13000
    assert counts["prediction_attempts"] <= 100 and counts["prediction_rows_attempted"] <= 5000


@pytest.fixture(scope="module")
def owner(ledger):
    source = bp.NumpyContinuousBCLearner(bp.ContinuousBCPolicy(spec(12), hidden_dim=2, seed=17))
    state: Any = source.snapshot_state()
    source.restore_state(
        replace(
            state,
            completed_updates=40,
            training_examples=1856,
            velocity=tuple(np.full_like(value, 0.03125) for value in state.velocity),
        )
    )
    return source


@pytest.mark.parametrize("curated_present", [False, True])
def test_should_leave_omitted_and_false_preparation_schema_identical(curated_present):
    a, b, c = inputs()
    c = c if curated_present else None
    original = data.prepare_joint_readiness_data(a, b, c, now=103)
    explicit = data.prepare_joint_readiness_data(a, b, c, now=103, old_information_control=False)
    assert asdict(original) == asdict(explicit)
    assert all(name != "repeated_a" for name, _, _ in original.groups)
    assert not any(key.startswith("repeated_reference") for key in original.work)


def test_should_preserve_default_retained_train20_path_when_control_is_false():
    values, extra = inputs(), retained()
    original = data.prepare_joint_readiness_data(*values, now=103, retained_reference=extra)
    explicit = data.prepare_joint_readiness_data(
        *values, now=103, retained_reference=extra, old_information_control=False
    )
    assert asdict(original) == asdict(explicit)
    assert explicit.groups[-1] == ("retained_train20", 6, 8)


@pytest.fixture(scope="module")
def prepared():
    values = inputs(a_rows=15, b_rows=50, c_rows=50)
    original = tuple(asdict(value) for value in values)
    base = data.prepare_joint_readiness_data(*values, now=199)
    control = data.prepare_joint_readiness_data(*values, now=199, old_information_control=True)
    return values, original, base, control


def test_should_append_all_fifteen_original_rows_in_order_without_new_authority(prepared):
    values, original, base, control = prepared
    assert len(control.rows) == control.work["rows"] == 130
    assert control.rows[:115] == base.rows
    assert control.groups == base.groups + (("repeated_a", 115, 130),)
    for index, repeated in enumerate(control.rows[115:]):
        expected = dict(
            base.rows[index],
            row_index=115 + index,
            source_kind="old_reference_repetition",
            source_row_index=index,
        )
        assert repeated == expected
        assert (
            repeated["observed_at"]
            is repeated["labels_arrived_at"]
            is repeated["provenance"]
            is None
        )
        assert repeated["role"] == "train" and repeated["context"] == "pick-cube"
        assert repeated["target"] == repeated["actual_command"]
    assert tuple(asdict(value) for value in values) == original


def test_should_restart_repeated_episode_history_and_return_detached_rows(prepared):
    _, _, base, control = prepared
    repeated = deepcopy(control)
    assert repeated.rows[115]["features"][7:] == [0.0] * 5
    assert repeated.rows[116]["features"][7:11] == base.rows[0]["actual_command"]
    assert repeated.rows[116]["features"][11] == 0.0
    repeated.rows[115]["features"][0] = 99.0
    repeated.rows[115]["source_observation"][0] = 99.0
    assert repeated.rows[0] == base.rows[0]
    assert control.rows[115]["features"] == base.rows[0]["features"]


def test_should_charge_fresh_repeated_projection_without_inflating_curated_label_work(prepared):
    _, _, base, control = prepared
    work = control.work
    assert work["observation_projections"] == 260
    assert work["source_values_validated"] == 10920
    assert work["projected_values"] == 3120 and work["projected_numeric_bytes"] == 24960
    assert work["history_advances"] == 130
    assert work["repeated_reference_rows"] == 15
    assert work["repeated_reference_projection_calls"] == 1
    assert work["repeated_reference_command_validation_advances"] == 15
    assert work["repeated_reference_observation_values"] == 180
    assert work["repeated_reference_target_values"] == 60
    assert work["repeated_reference_numeric_bytes"] == 1920
    assert work["per_group"]["repeated_a"] == work["per_group"]["a"]
    for key in (
        "curated_projection_calls",
        "curated_label_observation_values",
        "curated_label_target_values",
        "curated_label_numeric_bytes",
    ):
        assert work[key] == base.work[key]


@pytest.mark.parametrize("value", [None, 0, 1, 1.0, "true", [True]])
def test_should_refuse_non_boolean_control_before_any_projection(monkeypatch, value):
    values = inputs(a_rows=15)
    forbid_projection(monkeypatch)
    with pytest.raises(ValueError):
        data.prepare_joint_readiness_data(*values, now=103, old_information_control=value)


@pytest.mark.parametrize(
    "fault",
    [
        "missing_curated",
        "a14",
        "a16",
        "retained",
        "role",
        "identity",
        "current_flag",
        "next_flag",
        "command",
        "oversize",
    ],
)
def test_should_refuse_invalid_control_authority_or_capacity_before_projection(monkeypatch, fault):
    values = inputs(a_rows=14 if fault == "a14" else 16 if fault == "a16" else 15)
    a, b, c = values
    options: dict[str, Any] = {}
    if fault == "missing_curated":
        c = None
    elif fault == "retained":
        options["retained_reference"] = retained()
    elif fault == "role":
        object.__setattr__(a.episode, "role", "probe_train")
    elif fault == "identity":
        object.__setattr__(a.episode, "episode_id", "pick-cube-readiness-teacher-17")
    elif fault in ("current_flag", "next_flag"):
        row = a.episode.transitions[0]
        field = "observation" if fault == "current_flag" else "next_observation"
        raw = list(getattr(row, field))
        raw[18] = False if fault == "current_flag" else 0
        object.__setattr__(row, field, tuple(raw))
    elif fault == "command":
        object.__setattr__(a.episode.transitions[-1], "action", (0.1, 0.0, 0.0, 1.0))
    elif fault == "oversize":
        a, b, c = inputs(a_rows=15, b_rows=50, c_rows=50)
        object.__setattr__(c, "labels", c.labels + (c.labels[-1],))
        object.__setattr__(
            c.episode, "transitions", c.episode.transitions + (c.episode.transitions[-1],)
        )
    forbid_projection(monkeypatch)
    with pytest.raises(ValueError):
        data.prepare_joint_readiness_data(a, b, c, now=199, old_information_control=True, **options)


@pytest.mark.parametrize("family", ["b", "curated_a"])
def test_should_preserve_original_arrival_gates_before_repeated_projection(monkeypatch, family):
    values, now = (
        (inputs(a_rows=15, c_rows=1, c_start=0), 2) if family == "b" else (inputs(a_rows=15), 102)
    )
    forbid_projection(monkeypatch)
    with pytest.raises(ValueError, match="arriv"):
        data.prepare_joint_readiness_data(*values, now=now, old_information_control=True)


@pytest.mark.parametrize("fault", ["sampling", "retained", "non_bool"])
def test_should_refuse_runtime_option_mixes_before_projection_or_fork(owner, monkeypatch, fault):
    options: dict[str, Any] = dict(old_information_control=True)
    if fault == "sampling":
        options["sampling"] = "stage_balanced"
    elif fault == "retained":
        options["retained_reference"] = retained()
    else:
        options["old_information_control"] = 1
    forbid_projection(monkeypatch)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forbidden)
    with pytest.raises(ValueError):
        runtime.run_joint_readiness_corrections(owner, *inputs(a_rows=15), now=103, **options)


def test_should_keep_default_runtime_delegation_and_report_unchanged(owner, monkeypatch):
    prepare = runtime.prepare_joint_readiness_data
    calls = []

    def prepared(*args, **options):
        calls.append(dict(options))
        return prepare(*args, **options)

    def stop(*args, **options):
        raise RuntimeError("controlled pre-private stop")

    monkeypatch.setattr(runtime, "prepare_joint_readiness_data", prepared)
    monkeypatch.setattr(runtime, "_JointSession", stop)
    choices: tuple[dict[str, Any], ...] = ({}, dict(old_information_control=False))
    results = [
        runtime.run_joint_readiness_corrections(owner, *inputs(), now=103, **options)
        for options in choices
    ]
    assert calls[0] == calls[1] and "old_information_control" not in calls[0]
    assert results[0].report == results[1].report
    assert "old_information_control" not in results[0].report


@pytest.fixture(scope="module")
def successful(owner):
    source = owner.fork()
    initial = source.snapshot_state()
    values = inputs(a_rows=15, b_rows=50, c_rows=50)
    original = tuple(asdict(value) for value in values)
    captured: dict[str, Any] = {}
    factory = runtime.ActorShadowRuntime

    def capture(model, **options):
        made = factory(model, **options)
        captured.update(runtime=made, budget=options["budget"])
        return made

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runtime, "ActorShadowRuntime", capture)
        result = runtime.run_joint_readiness_corrections(
            source, *values, now=199, old_information_control=True
        )
    return result, source, initial, captured, values, original


def test_should_complete_eight_130_row_updates_with_unchanged_source_actor_and_hidden_parameters(
    successful,
):
    result, source, initial, captured, values, original = successful
    assert result.report["complete"] is True, result.report.get("error")
    assert type(result.candidate) is bp.NumpyContinuousBCLearner
    final = result.candidate.snapshot_state()
    assert (final.completed_updates, final.training_examples) == (48, 2896)
    assert final.policy == initial.policy
    for index in (0, 1):
        np.testing.assert_array_equal(final.parameters[index], initial.parameters[index])
        np.testing.assert_array_equal(final.velocity[index], initial.velocity[index])
    assert_state(source.snapshot_state(), initial)
    assert_state(captured["runtime"].actor.snapshot_state().state, initial)
    assert result.candidate is not source
    assert result.candidate is not captured["runtime"]._candidate._learner
    assert tuple(asdict(value) for value in values) == original


def test_should_publish_original_receipts_and_actual_context_exposures_with_full_work(successful):
    report = successful[0].report
    assert report["old_information_control"] == dict(
        definition="original_A17_15_repetition_control_v1",
        group="repeated_a",
        repeated_rows=15,
        source_row_indices=list(range(15)),
        source_episode_id="pick-cube-observed-grasp-teacher-17",
        source_context="pick-cube",
        source_role="train",
    )
    assert report["derived_batch"]["episode_id"].endswith("-old-reference-control")
    assert report["budget"]["max_training_updates"] == report["progress"]["updates_completed"] == 8
    assert report["source_updates"] == 40 and report["source_training_rows"] == 1856
    assert report["group_training_exposures"] == dict(a=120, b=400, curated_a=400, repeated_a=120)
    assert (
        report["native_updates"]
        == report["sealed_native_updates"]
        == report["validated_native_prefix"]
    )
    assert report["applied_updates"] == report["sealed_applied_updates"]
    assert len(report["native_updates"]) == len(report["applied_updates"]) == 8
    for index, (native, receipt) in enumerate(
        zip(report["native_updates"], report["applied_updates"], strict=True)
    ):
        assert native["training_rows"] == 130
        assert (
            receipt["sample_id"] == f"joint-{index}"
            and receipt["event_id"] == f"joint-label-{index}"
        )
        assert receipt["update_number"] == index + 1
        assert receipt["diagnostic"] == native["diagnostic"]
        assert (receipt["observed_at"], receipt["arrived_at"], receipt["applied_at"]) == (199,) * 3
    assert report["work"]["training_rows"] == report["attempted_training_rows"] == 1040
    assert report["work"]["native_update_counts"]["solve_completed"] == 32
    assert report["work"]["native_update_counts"]["solve_rows"] == 4256
    for when in ("before", "after"):
        assert {
            name: item["prediction_rows"] for name, item in report["residuals"][when].items()
        } == dict(a=15, b=50, curated_a=50, repeated_a=15)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("fault", ["content", "equal_replacement"])
def test_should_stop_after_validated_paid_prefix_when_control_metadata_changes(
    owner, monkeypatch, fault
):
    source = owner.fork()
    initial = source.snapshot_state()
    original_apply = runtime._JointSession.apply_one
    completions = []

    def change_after_success(session, index):
        returned = original_apply(session, index)
        completions.append(index)
        if fault == "equal_replacement":
            session.report["old_information_control"] = deepcopy(
                session.report["old_information_control"]
            )
        else:
            session.report["old_information_control"]["group"] = "relabelled"
        return returned

    monkeypatch.setattr(runtime._JointSession, "apply_one", change_after_success)
    result = runtime.run_joint_readiness_corrections(
        source, *inputs(a_rows=15), now=103, old_information_control=True
    )
    report = result.report
    assert result.candidate is None and report["complete"] is False
    assert completions == [0] and report["update_attempts"] == 1
    assert len(report["native_updates"]) == len(report["validated_native_prefix"]) == 1
    assert len(report["applied_updates"]) == len(report["sealed_applied_updates"]) == 1
    assert report["group_training_exposures"] == dict(a=15, b=2, curated_a=2, repeated_a=15)
    assert report["attempted_training_rows"] == 34
    assert report["progress"]["updates_completed"] == 1
    assert "error" in report
    assert_state(source.snapshot_state(), initial)
