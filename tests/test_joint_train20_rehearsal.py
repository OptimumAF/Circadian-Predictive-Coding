"""Optional qualified TRAIN20 rehearsal without renewing the joint update scope.

One H2/D12 source (32 normal draws), two eight-update scopes, one paid
mutation prefix and a capacity update: 18 GN returns/250 rows/72 solves/1216 augmented rows;
18 diagnostic predictions/36 rows. Pure 130-row preparation never fits a model.
Synthetic qualified bindings exercise the trusted Python boundary, not teacher
qualification or a claim about adaptation on a native environment.
"""

from dataclasses import asdict
from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.adapters import joint_readiness_corrections as runtime
from src.adapters import joint_readiness_data as data
from src.adapters import numpy_continuous_bc as bp
from src.adapters.continuous_head_learnability import RecordedContinuousHead, _native_work
from src.core import continuous_head_gauss_newton as head
from src.core.continuous_demonstration import ContinuousEpisode, ContinuousTransition
from src.core.pick_cube_readiness_teacher import OBSERVED_GRASP_DEFINITION
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.core.readiness_stage_diagnostics import ReadinessStageBinding, readiness_stage_metadata
from test_joint_readiness_corrections import assert_state, inputs, spec


def retained(rows=2):
    """Independent observed-grasp TRAIN20 history: lift then loss of grasp."""
    frames = []
    for index in range(rows + 1):
        frame = [0.0] * 42
        frame[18] = 1.0 if index == 0 else 0.0
        frame[36] = 0.04 + 0.001 * index
        frame[41] = 0.05
        frames.append(tuple(frame))
    episode = ContinuousEpisode(
        "pick-cube-observed-grasp-teacher-20",
        "pick-cube-train-20",
        "train",
        spec(),
        tuple(
            ContinuousTransition(
                index,
                frames[index],
                (0.25, 0.0, 0.25, -1.0) if index == 0 else (0.25, 0.0, 0.0, 1.0),
                0.0,
                frames[index + 1],
                False,
                index == rows - 1,
            )
            for index in range(rows)
        ),
    )
    return ReadinessStageBinding(episode, ("lift",) + ("align",) * (rows - 1))


def forbidden(*args, **kwargs):
    pytest.fail("invalid rehearsal input reached projection or private numerical work")


def forbid_projections(monkeypatch):
    for name in (
        "action_history_training_episode",
        "action_history_correction_view",
        "project_curated_training_pairs",
    ):
        monkeypatch.setattr(data, name, forbidden)


@pytest.fixture(scope="module")
def ledger():
    counts = dict(
        initializations=0,
        normal_draws=0,
        GN_attempts=0,
        GN_returns=0,
        GN_rows_attempted=0,
        solve_attempts=0,
        solve_completed=0,
        augmented_rows=0,
        prediction_attempts=0,
        prediction_rows_attempted=0,
        prediction_returns=0,
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
        counts["normal_draws"] += policy.hidden_dim * (policy.spec.observation_dim + 4)
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
        returned = original_predict(owner, features)
        counts["prediction_returns"] += 1
        return returned

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bp.NumpyContinuousBCLearner, "__init__", initialize)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", train)
        patch.setattr(bp.NumpyContinuousBCLearner, "predict", predict)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", forbidden)
        yield counts
    if os.environ.get("TR20_LEDGER"):
        with Path(os.environ["TR20_LEDGER"]).open("x", encoding="utf8") as output:
            output.write(json.dumps(counts, sort_keys=True) + "\n")
    assert counts["initializations"] == 1 and counts["normal_draws"] == 32
    assert counts["GN_attempts"] <= 18 and counts["GN_rows_attempted"] <= 250
    assert counts["solve_attempts"] <= 72 and counts["augmented_rows"] <= 1500
    assert counts["prediction_attempts"] <= 40 and counts["prediction_rows_attempted"] <= 400


@pytest.fixture(scope="module")
def owner(ledger):
    return bp.NumpyContinuousBCLearner(bp.ContinuousBCPolicy(spec(12), hidden_dim=2, seed=17))


@pytest.mark.parametrize("with_curated", [False, True])
def test_should_preserve_original_output_when_optional_reference_is_none(with_curated):
    a, b, c = inputs()
    c = c if with_curated else None
    original = data.prepare_joint_readiness_data(a, b, c, now=103)
    explicit = data.prepare_joint_readiness_data(a, b, c, now=103, retained_reference=None)
    assert asdict(original) == asdict(explicit)
    assert not any(key.startswith("retained_reference") for key in original.work)
    assert all(group[0] != "retained_train20" for group in original.groups)


@pytest.mark.parametrize("with_curated", [False, True])
def test_should_append_independent_original_train20_history_without_rewriting_base(with_curated):
    a, b, c = inputs()
    c = c if with_curated else None
    extra = retained()
    original_seal = asdict(extra)
    baseline = data.prepare_joint_readiness_data(a, b, c, now=103)
    prepared = data.prepare_joint_readiness_data(a, b, c, now=103, retained_reference=extra)
    start = len(baseline.rows)
    assert prepared.rows[:start] == baseline.rows
    assert prepared.groups == baseline.groups + (("retained_train20", start, start + 2),)
    first, second = prepared.rows[start:]
    assert first["features"][7:] == [0.0] * 5
    assert first["features"][0] == 1.0 and second["features"][0] == 0.0
    assert second["features"][7:11] == first["actual_command"] == [0.25, 0.0, 0.25, -1.0]
    assert second["features"][11] == pytest.approx(1 / 6)
    for index, row in enumerate(prepared.rows[start:]):
        assert row["row_index"] == start + index and row["step"] == index
        assert row["source_observation"] == list(extra.episode.transitions[index].observation)
        assert (
            row["target"] == row["actual_command"] == list(extra.episode.transitions[index].action)
        )
        assert row["episode_id"] == "pick-cube-observed-grasp-teacher-20"
        assert row["context"] == "pick-cube-train-20" and row["role"] == "train"
        assert row["stage"] == extra.stages[index]
        assert row["source_kind"] == "retained_qualified_demonstration"
        assert row["teacher_definition"] == OBSERVED_GRASP_DEFINITION
        assert row["provenance"] is row["observed_at"] is row["labels_arrived_at"] is None
    for key in (
        "curated_projection_calls",
        "curated_label_observation_values",
        "curated_label_target_values",
        "curated_label_numeric_bytes",
    ):
        assert prepared.work[key] == baseline.work[key]
    assert prepared.work["retained_reference_rows"] == 2
    assert prepared.work["retained_reference_projection_calls"] == 1
    assert prepared.work["retained_reference_observation_values"] == 24
    assert prepared.work["retained_reference_target_values"] == 8
    assert prepared.work["retained_reference_numeric_bytes"] == 256
    assert prepared.work["retained_reference_command_validation_advances"] == 2
    first["source_observation"][0] = 99.0
    assert asdict(extra) == original_seal


def test_should_admit_130_rows_without_expanding_base_128_row_limit():
    prepared = data.prepare_joint_readiness_data(
        *inputs(a_rows=15, b_rows=50, c_rows=50), now=199, retained_reference=retained(15)
    )
    assert len(prepared.rows) == prepared.work["rows"] == 130
    assert prepared.groups[-1] == ("retained_train20", 115, 130)
    assert prepared.work["per_group"]["retained_train20"]["observation_projections"] == 30
    assert prepared.work["curated_label_observation_values"] == 600


@pytest.mark.parametrize("base_rows,extra_rows", [((15, 50, 50), 16), ((29, 50, 50), 1)])
def test_should_refuse_combined_131_or_base_129_before_projection(
    monkeypatch, base_rows, extra_rows
):
    values = inputs(a_rows=base_rows[0], b_rows=base_rows[1], c_rows=base_rows[2])
    extra = retained(extra_rows)
    forbid_projections(monkeypatch)
    with pytest.raises(ValueError):
        data.prepare_joint_readiness_data(*values, now=199, retained_reference=extra)


@pytest.mark.parametrize("family", ["b", "curated_a"])
def test_should_keep_original_release_gates_before_any_projection_with_retained(
    monkeypatch, family
):
    values, now = (inputs(c_rows=1, c_start=0), 2) if family == "b" else (inputs(), 102)
    extra = retained()
    forbid_projections(monkeypatch)
    with pytest.raises(ValueError, match="arriv"):
        data.prepare_joint_readiness_data(*values, now=now, retained_reference=extra)


@pytest.mark.parametrize(
    "fault",
    [
        "type",
        "role",
        "context",
        "reserved_context",
        "ordinary_teacher",
        "wrong_seed",
        "spec",
        "stage",
        "flag_bool",
        "flag_int",
        "next_flag_bool",
        "command",
        "recipe",
    ],
)
def test_should_refuse_forged_retained_authority_before_projection(monkeypatch, fault):
    extra: Any = retained()
    options: dict[str, Any] = {}
    if fault == "type":
        extra = asdict(extra)
    elif fault in ("role", "context", "reserved_context", "ordinary_teacher", "wrong_seed"):
        field, value = {
            "role": ("role", "probe_test"),
            "context": ("context", "pick-cube-train-19"),
            "reserved_context": ("context", "pick-cube-reserved-probe-23-24"),
            "ordinary_teacher": ("episode_id", "pick-cube-readiness-teacher-20"),
            "wrong_seed": ("episode_id", "pick-cube-observed-grasp-teacher-19"),
        }[fault]
        object.__setattr__(extra.episode, field, value)
    elif fault == "spec":
        object.__setattr__(extra.episode.spec, "upper_bounds", (2.0,) * 4)
    elif fault == "stage":
        object.__setattr__(extra, "stages", ("align", "align"))
    elif fault in ("flag_bool", "flag_int", "next_flag_bool"):
        field = "next_observation" if fault == "next_flag_bool" else "observation"
        row = extra.episode.transitions[0]
        vector = list(getattr(row, field))
        vector[18] = False if fault == "next_flag_bool" else (True if fault == "flag_bool" else 1)
        object.__setattr__(row, field, tuple(vector))
    elif fault == "command":
        row = extra.episode.transitions[-1]
        object.__setattr__(row, "action", (0.1, 0.0, 0.0, 1.0))
    else:
        options["recipe"] = PickCubeTaskSpaceHistoryRecipe(mode="zero")
    forbid_projections(monkeypatch)
    with pytest.raises(ValueError):
        data.prepare_joint_readiness_data(*inputs(), now=103, retained_reference=extra, **options)


def test_should_keep_ordered_17_19_stage_pair_strict():
    with pytest.raises(ValueError):
        readiness_stage_metadata((inputs()[0], retained()))


def test_should_record_one_genuine_130_row_head_update_without_changing_source(owner):
    prepared = data.prepare_joint_readiness_data(
        *inputs(a_rows=15, b_rows=50, c_rows=50), now=199, retained_reference=retained(15)
    )
    initial = owner.snapshot_state()
    reports: list[dict[str, Any]] = []
    private = owner.fork()
    port = RecordedContinuousHead(private, reports, 1)
    features = np.asarray([row["features"] for row in prepared.rows], dtype=np.float64)
    targets = np.asarray([row["target"] for row in prepared.rows], dtype=np.float64)
    returned = port.train_batch(features, targets)
    port.require_reports(reports)
    port.require_latest(private.snapshot_state())
    assert len(reports) == 1 and reports[0]["training_rows"] == 130
    assert reports[0]["diagnostic"] == asdict(returned)
    assert reports[0]["loss_after"] < reports[0]["diagnostic"]["value"]
    counts, _, _ = _native_work(reports)
    assert counts["solve_completed"] == 4 and counts["solve_rows"] == 532
    corrupted = deepcopy(reports)
    corrupted[0]["training_rows"] = 131
    with pytest.raises(ValueError):
        _native_work(corrupted)
    assert (
        private.snapshot_state().completed_updates,
        private.snapshot_state().training_examples,
    ) == (1, 130)
    assert_state(owner.snapshot_state(), initial)


@pytest.mark.parametrize("operation", ["linearize_continuous_head", "propose_head_gauss_newton"])
def test_should_refuse_131_rows_before_head_forward_or_solve(owner, monkeypatch, operation):
    initial = owner.snapshot_state()
    monkeypatch.setattr(head, "_hidden", forbidden)
    monkeypatch.setattr(head.np.linalg, "lstsq", forbidden)
    with pytest.raises(ValueError, match="130"):
        getattr(head, operation)(
            np.zeros((131, 12), dtype=np.float64),
            np.zeros((131, 4), dtype=np.float64),
            initial.parameters,
            initial.policy.spec,
        )


@pytest.fixture(scope="module")
def completed(owner):
    outcomes = {}
    for selected in (False, True):
        source = owner.fork()
        initial = source.snapshot_state()
        values, extra = inputs(), retained()
        seals = tuple(asdict(value) for value in (*values, extra))
        captured: dict[str, Any] = {}
        factory = runtime.ActorShadowRuntime

        def capture(model, **options):
            made = factory(model, **options)
            captured.update(runtime=made, budget=options["budget"])
            return made

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(runtime, "ActorShadowRuntime", capture)
            result = runtime.run_joint_readiness_corrections(
                source, *values, now=103, **({"retained_reference": extra} if selected else {})
            )
        outcomes[selected] = result, source, initial, captured, values, extra, seals
    return outcomes


def test_should_complete_retained_updates_with_frozen_source_actor_and_hidden_weights(completed):
    result, source, initial, captured, values, extra, seals = completed[True]
    assert result.report["complete"] is True, result.report.get("error")
    assert type(result.candidate) is bp.NumpyContinuousBCLearner
    final = result.candidate.snapshot_state()
    assert (final.completed_updates, final.training_examples) == (8, 64)
    assert_state(source.snapshot_state(), initial)
    assert_state(captured["runtime"].actor.snapshot_state().state, initial)
    assert result.candidate is not source
    assert result.candidate is not captured["runtime"]._candidate._learner
    for index in (0, 1):
        np.testing.assert_array_equal(final.parameters[index], initial.parameters[index])
        np.testing.assert_array_equal(final.velocity[index], initial.velocity[index])
    assert tuple(asdict(value) for value in (*values, extra)) == seals


def test_should_account_retained_exposures_receipts_and_original_scope_without_renewal(completed):
    report = completed[True][0].report
    default = completed[False][0].report
    assert default["complete"] is True, default.get("error")
    assert report["rows"][:6] == default["rows"]
    assert "retained_reference" not in default
    assert default["candidate_training_rows"] == 48
    assert report["budget"] == default["budget"]
    assert report["budget"]["max_training_updates"] == 8
    assert (
        report["derived_batch"]["episode_id"]
        == default["derived_batch"]["episode_id"] + "-retained-TRAIN20"
    )
    assert report["retained_reference"]["rows"] == 2
    assert report["retained_reference"]["group"] == "retained_train20"
    assert report["group_training_exposures"] == dict(a=16, b=16, curated_a=16, retained_train20=16)
    assert (
        report["native_updates"]
        == report["sealed_native_updates"]
        == report["validated_native_prefix"]
    )
    assert len(report["native_updates"]) == len(report["applied_updates"]) == 8
    assert report["applied_updates"] == report["sealed_applied_updates"]
    assert [receipt["sample_id"] for receipt in report["applied_updates"]] == [
        f"joint-{i}" for i in range(8)
    ]
    for receipt in report["applied_updates"]:
        assert (receipt["observed_at"], receipt["arrived_at"], receipt["applied_at"]) == (103,) * 3
    assert report["work"]["training_rows"] == report["attempted_training_rows"] == 64
    assert report["work"]["native_update_counts"]["solve_completed"] == 32
    assert report["work"]["native_update_counts"]["solve_rows"] == 352
    assert report["progress"]["updates_completed"] == 8
    for when in ("before", "after"):
        assert set(report["residuals"][when]) == {"a", "b", "curated_a", "retained_train20"}
        assert report["residuals"][when]["retained_train20"]["prediction_rows"] == 2


def test_should_preserve_first_paid_prefix_when_original_retained_input_changes(owner, monkeypatch):
    source = owner.fork()
    initial = source.snapshot_state()
    extra = retained()
    original = runtime._JointSession.apply_one
    completed_calls = 0

    def change_after_paid_return(session, index):
        nonlocal completed_calls
        returned = original(session, index)
        completed_calls += 1
        row = extra.episode.transitions[0]
        object.__setattr__(row, "action", (0.5, 0.0, 0.25, -1.0))
        return returned

    monkeypatch.setattr(runtime._JointSession, "apply_one", change_after_paid_return)
    result = runtime.run_joint_readiness_corrections(
        source, *inputs(), now=103, retained_reference=extra
    )
    report = result.report
    assert result.candidate is None and report["complete"] is False
    assert "evidence changed" in report["error"]
    assert completed_calls == report["update_attempts"] == 1
    assert len(report["native_updates"]) == len(report["validated_native_prefix"]) == 1
    assert len(report["applied_updates"]) == len(report["sealed_applied_updates"]) == 1
    assert report["group_training_exposures"]["retained_train20"] == 2
    assert report["attempted_training_rows"] == 8
    assert report["progress"]["updates_completed"] == 1
    assert extra.episode.transitions[0].action[0] == 0.5
    assert_state(source.snapshot_state(), initial)


def test_should_reject_retained_mutation_by_final_admission_observer_before_fork(
    owner, monkeypatch
):
    source = owner.fork()
    extra = retained()
    original_snapshot = bp.NumpyContinuousBCLearner.snapshot_state
    proofs = 0

    def mutate(owner):
        nonlocal proofs
        proofs += 1
        returned = original_snapshot(owner)
        if proofs == 2:
            object.__setattr__(extra.episode, "context", "pick-cube-train-19")
        return returned

    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "snapshot_state", mutate)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forbidden)
    with pytest.raises(RuntimeError, match="evidence changed"):
        runtime.run_joint_readiness_corrections(
            source, *inputs(), now=103, retained_reference=extra
        )
    assert proofs == 2


def test_should_omit_retained_keyword_and_metadata_for_both_default_call_shapes(owner, monkeypatch):
    prepare = runtime.prepare_joint_readiness_data
    calls = []

    def capture(*args, **options):
        calls.append(dict(options))
        return prepare(*args, **options)

    def stop_before_private_work(*args, **options):
        raise RuntimeError("controlled stop before private work")

    monkeypatch.setattr(runtime, "prepare_joint_readiness_data", capture)
    monkeypatch.setattr(runtime, "_JointSession", stop_before_private_work)
    results = [
        runtime.run_joint_readiness_corrections(owner, *inputs(), now=103, **options)
        for options in ({}, {"retained_reference": None})
    ]
    assert len(calls) == 2 and calls[0] == calls[1]
    assert all("retained_reference" not in call for call in calls)
    assert results[0].report == results[1].report
    assert "retained_reference" not in results[0].report
    assert all(result.candidate is None for result in results)
