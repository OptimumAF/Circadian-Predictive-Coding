"""Manual public21 objective-control reports; no fit, SGD or model forward.

One archive model uses one constructor+restore. The accepted initial-parity
helper uses two zero-update templates once; other composition tests stub it.
"""

from copy import deepcopy
from dataclasses import replace
import importlib
import json
import sys

import numpy as np
import pytest

from src.adapters.continuous_learnability_demo import ContinuousLearnabilityResult
from src.adapters.numpy_continuous_bc import ContinuousBCPolicy, NumpyContinuousBCLearner
from src.core.continuous_bc import MIXED_OBJECTIVE, OBJECTIVE
from src.core.observation_standardization import ObservationStandardization
from test_continuous_learnability_demo import same_state
from test_public_vector_mlp_depth import FIELDS, metrics, sequence, training_episode


case = importlib.import_module("artifacts.runs.public-vector-mixed-control-20261019.run_case")


def forbidden(*_args, **_kwargs):
    pytest.fail("invalid comparative evidence admitted new work")


@pytest.fixture(scope="module")
def data():
    warm, new = training_episode(17, 2), training_episode(19, 3)
    stats = ObservationStandardization((0.5,) + (0.0,) * 20, (0.5,) + (1.0,) * 20, 2)
    policy = ContinuousBCPolicy(warm.spec, observation_standardization=stats)
    mse, p8 = sequence(warm, new, policy, 32), sequence(warm, new, policy, 8)
    mixed = deepcopy(mse)
    mixed["policy"]["binary_gripper_axis"] = 3
    mixed["work"]["binary_gripper_loss_rows"] = 244
    for receipt, event in zip(mixed["applied_updates"], mixed["events"], strict=True):
        receipt["diagnostic"].update(definition=MIXED_OBJECTIVE, value=0.75)
        event["applied"] = [deepcopy(receipt)]
    for point in mixed["checkpoints"]:
        for result in point["contexts"].values():
            result["conditioning"]["conditioning"].update(
                objective_definition=MIXED_OBJECTIVE, objective_loss=2.0
            )
    return dict(warm=warm, new=new, statistics=stats, policy=policy, mse=mse, p8=p8, mixed=mixed)


@pytest.fixture(scope="module")
def candidate(data):
    """ONE constructor/restore, manually tagged96/224; zero training history."""
    owner = NumpyContinuousBCLearner(replace(data["policy"], binary_gripper_axis=3))
    initial = owner.snapshot_state()
    velocity = tuple(
        np.full_like(value, (index + 1) * 0.125) for index, value in enumerate(initial.velocity)
    )
    owner.restore_state(
        replace(initial, velocity=velocity, completed_updates=96, training_examples=224)
    )
    return owner


@pytest.mark.parametrize("name,objective", [("mse", OBJECTIVE), ("mixed", MIXED_OBJECTIVE)])
def test_should_validate_fixed96_receipts_and_original_label_clocks_without_training(
    data, name, objective
):
    report = deepcopy(data[name])
    original = deepcopy(report)
    case.check_progress(report, data["warm"], data["new"], objective)
    assert report == original
    assert len(report["applied_updates"]) == 96 and report["progress"]["next_tick"] == 192
    assert [row["training_examples"] for row in report["checkpoints"]] == [0, 64, 160, 224]


@pytest.mark.parametrize(
    "damage",
    [
        "count",
        "counter_type",
        "source",
        "clock",
        "objective",
        "negative",
        "nonfinite",
        "loss_bool",
        "receipt_id",
        "event",
        "endpoint_type",
        "owners",
    ],
)
def test_should_refuse_corrupted_phase_budget_receipts_objective_or_owner_evidence(data, damage):
    report = deepcopy(data["mixed"])
    if damage == "count":
        report["settings"]["phase_updates"] = 8
    elif damage == "counter_type":
        report["progress"]["updates_completed"] = 96.0
    elif damage == "source":
        report["source_episodes"]["new"]["role"] = "probe_train"
    elif damage == "clock":
        report["applied_updates"][0]["arrived_at"] = 0
    elif damage == "objective":
        report["applied_updates"][0]["diagnostic"]["definition"] = OBJECTIVE
    elif damage == "negative":
        report["applied_updates"][0]["diagnostic"]["value"] = -0.01
    elif damage == "nonfinite":
        report["applied_updates"][0]["diagnostic"]["value"] = float("nan")
    elif damage == "loss_bool":
        report["applied_updates"][0]["diagnostic"]["value"] = True
    elif damage == "receipt_id":
        report["applied_updates"][0]["event_id"] = "reused-label"
    elif damage == "event":
        report["events"][0]["candidate_training_examples"] = 3
    elif damage == "endpoint_type":
        report["checkpoints"][0]["updates"] = False
    else:
        report["checkpoints"][2]["owners_exact"] = 1
    if damage in ("clock", "objective", "negative", "nonfinite", "loss_bool", "receipt_id"):
        report["events"][0]["applied"] = [deepcopy(report["applied_updates"][0])]
    with pytest.raises((AssertionError, ValueError)):
        case.check_progress(report, data["warm"], data["new"], MIXED_OBJECTIVE)


def test_should_allow_only_axis3_policy_difference_and_account_additional_loss_rows(data):
    original = deepcopy(data["mixed"]), deepcopy(data["mse"])
    case.verify_pair(data["mixed"], data["mse"], data["warm"], data["new"])
    assert (data["mixed"], data["mse"]) == original
    mixed_policy = case.policy_from(data["mixed"], data["warm"].spec, 3)
    mse_policy = case.policy_from(data["mse"], data["warm"].spec, None)
    assert mixed_policy == replace(mse_policy, binary_gripper_axis=3)
    assert mixed_policy.spec.observation_dim == 21 and mixed_policy.hidden_dim == 64
    assert (
        mixed_policy.seed == 17
        and mixed_policy.learning_rate == 0.01
        and mixed_policy.momentum == 0.9
    )
    assert mixed_policy.observation_standardization == data["statistics"]
    assert data["mixed"]["work"] == dict(data["mse"]["work"], binary_gripper_loss_rows=244)
    assert data["mixed"]["work"]["optimizer_updates"] == 96
    assert data["mixed"]["work"]["training_rows"] == 224
    assert data["mixed"]["work"]["dense_multiply_accumulates"] == 939264


@pytest.mark.parametrize(
    "damage",
    [
        "axis",
        "unknown_policy",
        "statistics",
        "expected_fit",
        "loss_work",
        "common_work",
        "work_type",
        "conditioning_objective",
        "initial_overall",
        "initial_axis",
        "initial_actor",
        "later_actor",
        "metric",
    ],
)
def test_should_refuse_unmatched_policy_exposure_objective_or_initial_physical_parity(data, damage):
    mixed, mse = deepcopy(data["mixed"]), deepcopy(data["mse"])
    if damage == "axis":
        mixed["policy"]["binary_gripper_axis"] = None
    elif damage == "unknown_policy":
        mixed["policy"]["selected_checkpoint"] = 96
    elif damage == "statistics":
        mixed["policy"]["observation_standardization"]["means"] = (0.6,) + (0.0,) * 20
    elif damage == "expected_fit":
        mixed["expected_standardization_checked"] = False
    elif damage == "loss_work":
        mixed["work"]["binary_gripper_loss_rows"] = 243
    elif damage == "common_work":
        mixed["work"]["training_rows"] = 223
    elif damage == "work_type":
        mixed["work"]["optimizer_updates"] = 96.0
    elif damage == "conditioning_objective":
        mixed["checkpoints"][0]["contexts"]["warm"]["conditioning"]["conditioning"][
            "objective_definition"
        ] = OBJECTIVE
    elif damage == "initial_overall":
        mixed["checkpoints"][0]["contexts"]["warm"]["residual"]["residuals"]["overall"][
            "normalized_half_mse"
        ] += 1e-6
        mixed["metrics"] = metrics(mixed["checkpoints"])
    elif damage == "initial_axis":
        mixed["checkpoints"][0]["contexts"]["warm"]["residual"]["residuals"]["overall"]["axes"][3][
            "maximum_absolute_normalized_error"
        ] += 1e-6
    elif damage == "initial_actor":
        mixed["checkpoints"][0]["contexts"]["warm"]["actor_predictions"][0][0] += 1e-6
    elif damage == "later_actor":
        mixed["checkpoints"][3]["contexts"]["new"]["actor_predictions"][0][0] += 1e-6
    else:
        mixed["metrics"]["new_retention_loss"] += 0.01
    with pytest.raises((AssertionError, ValueError)):
        case.verify_pair(mixed, mse, data["warm"], data["new"])


def test_should_accept_only_frozen_absolute_tolerance_for_initial_measured_axis_parity(data):
    mixed = deepcopy(data["mixed"])
    overall = mixed["checkpoints"][0]["contexts"]["warm"]["residual"]["residuals"]["overall"]
    overall["axes"][3]["normalized_bias"] += 5e-13
    mixed["checkpoints"][0]["contexts"]["warm"]["residual"]["residuals"]["stages"][0]["summary"] = (
        deepcopy(overall)
    )
    case.verify_pair(mixed, data["mse"], data["warm"], data["new"])


def test_should_preserve_all_physical_endpoints_axis_fields_and_signed_metrics_without_ranking_train_losses(
    data,
):
    mixed, mse = deepcopy(data["mixed"]), deepcopy(data["mse"])
    shifts = [(0.0, 0.0), (-0.1, 0.05), (0.2, -0.1), (-0.3, 0.15)]
    for point, deltas in zip(mixed["checkpoints"], shifts, strict=True):
        for name, difference in zip(("warm", "new"), deltas, strict=True):
            residual = point["contexts"][name]["residual"]
            overall = residual["residuals"]["overall"]
            overall["normalized_half_mse"] += difference
            for axis in overall["axes"]:
                for field in FIELDS:
                    axis[field] += difference
            residual["residuals"]["stages"][0]["summary"] = deepcopy(overall)
    mixed["metrics"] = metrics(mixed["checkpoints"])
    case.verify_pair(mixed, mse, data["warm"], data["new"])
    before = deepcopy(mixed), deepcopy(mse)
    compared = case.comparison(mixed, mse)
    assert (mixed, mse) == before
    assert compared["primary_endpoint"] == 96
    assert compared["primary"] == pytest.approx(dict(warm=-0.3, new=0.15))
    assert "negative better" in compared["primary_direction"]
    assert [row["updates"] for row in compared["endpoints"]] == [0, 32, 64, 96]
    for point, deltas in zip(compared["endpoints"], shifts, strict=True):
        for name, difference in zip(("warm", "new"), deltas, strict=True):
            result = point["contexts"][name]
            assert result["mixed_minus_mse"] == pytest.approx(difference)
            assert len(result["axes"]) == 4
            for index, axis in enumerate(result["axes"]):
                assert axis["index"] == index
                assert set(axis["mixed_minus_mse"]) == set(FIELDS)
                assert axis["mixed_minus_mse"] == {
                    field: pytest.approx(difference) for field in FIELDS
                }
    assert compared["signed_metric_differences"] == {
        name: mixed["metrics"][name] - mse["metrics"][name] for name in mse["metrics"]
    }
    assert compared["metric_directions"] == dict(
        warm_acquisition_gain="positive better",
        new_acquisition_gain="positive better",
        warm_retention_loss="negative better",
        warm_return_recovery="positive better",
        warm_return_deficit="negative better",
        new_retention_loss="negative better",
    )
    assert compared["metric_directions"] is not case.METRIC_DIRECTIONS
    assert (
        compared["work_differences"]["optimizer_updates"]
        == compared["work_differences"]["training_rows"]
        == compared["work_differences"]["dense_multiply_accumulates"]
        == 0
    )
    assert compared["work_differences"]["binary_gripper_loss_rows"] == 244
    assert "CE elementwise work additional" in compared["exposure_comparison"]
    assert "not ranked" in compared["training_loss_comparison"]
    assert not any("train_loss_delta" in key for key in compared)
    assert mixed["applied_updates"][0]["diagnostic"]["value"] == 0.75
    assert mse["applied_updates"][0]["diagnostic"]["value"] == 0.5


def test_should_reuse_two_zero_update_templates_for_all_eight_array_parity(data, monkeypatch):
    """ExactlyTWO model constructors, no restore/fit/forward/SGD."""
    monkeypatch.setattr(NumpyContinuousBCLearner, "train_batch", forbidden)
    monkeypatch.setattr(NumpyContinuousBCLearner, "predict", forbidden)
    parity = case.initial_parity(data["warm"].spec, data["statistics"])
    assert parity["template_initializations"] == 2
    assert parity["template_optimizer_updates"] == 0
    assert parity["all_eight_arrays_exact"] == [True] * 8
    assert parity["array_values_per_template"] == 3336
    assert parity["numeric_bytes_per_template"] == 26688


@pytest.fixture
def source_admission(data, tmp_path, monkeypatch):
    projection = dict(fixture_projection="same_raw_train_packets_no_new_fit")
    accepted = dict(
        complete=True,
        python=sys.version,
        numpy=np.__version__,
        input_projection=projection,
        sequence=deepcopy(data["mse"]),
    )
    path = tmp_path / "mse-case.json"
    path.write_text(json.dumps(accepted), encoding="utf8")

    def qualified(report):
        report["input_projection"] = deepcopy(projection)
        report["source_paths"] = dict(baseline="unchanged-public-p8.json")
        return data["warm"], data["new"], data["p8"], data["statistics"]

    monkeypatch.setattr(case.depth, "load_inputs", qualified)
    monkeypatch.setattr(case, "BASELINE_PATH", path)
    monkeypatch.setattr(case, "initial_parity", forbidden)
    monkeypatch.setattr(case, "run_continuous_learnability", forbidden)
    return path, accepted


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "complete",
        "runtime",
        "projection",
        "nonfinite",
        "policy",
        "stats",
        "receipt",
        "initial",
        "work",
    ],
)
def test_should_reuse_qualified_sources_and_refuse_invalid_p32_mse_before_new_templates_or_fit(
    data, source_admission, damage
):
    path, accepted = source_admission
    sequence = accepted["sequence"]
    if damage == "complete":
        accepted["complete"] = 1
    elif damage == "runtime":
        accepted["numpy"] = "different-frozen-runtime"
    elif damage == "projection":
        accepted["input_projection"] = dict(selected_projection=True)
    elif damage == "nonfinite":
        sequence["metrics"]["warm_retention_loss"] = float("inf")
    elif damage == "policy":
        sequence["policy"]["binary_gripper_axis"] = 3
    elif damage == "stats":
        sequence["policy"]["observation_standardization"]["fitted_rows"] = 3
    elif damage == "receipt":
        sequence["applied_updates"][0]["arrived_at"] = 0
        sequence["events"][0]["applied"] = [deepcopy(sequence["applied_updates"][0])]
    elif damage == "initial":
        sequence["checkpoints"][0]["contexts"]["warm"]["actor_predictions"][0][0] = 0.125
    elif damage == "work":
        sequence["work"]["gradient_values"] += 1
    path.write_text(json.dumps(accepted), encoding="utf8")
    report = {}
    if damage is None:
        warm, new, mse, statistics = case.load_inputs(report)
        assert warm is data["warm"] and new is data["new"]
        assert case.exact_json(mse, data["mse"])
        assert statistics == data["statistics"]
        assert report["mse_source"] == report["source_paths"]["mse_p32"] == str(path)
    else:
        with pytest.raises((AssertionError, ValueError)):
            case.load_inputs(report)


def test_should_save_all_eight_mixed_arrays_and_full_optimizer_counts_exclusively(
    data, candidate, tmp_path, monkeypatch
):
    monkeypatch.setattr(case, "stage", tmp_path)
    before = candidate.snapshot_state()
    report = dict(sequence=deepcopy(data["mixed"]))
    case.save_state(candidate, report)
    saved = report["state_archive"]
    names = [f"{group}_{index}" for group in ("parameters", "velocity") for index in range(4)]
    assert saved["updates"] == 96 and saved["training_examples"] == 224
    assert saved["array_names"] == names and saved["numeric_bytes"] == 26688
    assert saved["save_attempts"] == saved["save_returns"] == 1
    assert saved["file_bytes"] <= 128 * 1024
    with np.load(tmp_path / "final-state.npz", allow_pickle=False) as archive:
        assert set(archive.files) == set(names)
        for group in ("parameters", "velocity"):
            for index, value in enumerate(getattr(before, group)):
                assert archive[f"{group}_{index}"].dtype == np.float64
                np.testing.assert_array_equal(archive[f"{group}_{index}"], value)
    same_state(before, candidate.snapshot_state())
    original = (tmp_path / "final-state.npz").read_bytes()
    with pytest.raises(FileExistsError):
        case.save_state(candidate, dict(sequence=data["mixed"]))
    assert (tmp_path / "final-state.npz").read_bytes() == original


@pytest.mark.parametrize(
    "damage", ["mse_policy", "updates", "update_type", "rows", "dtype", "nonfinite"]
)
def test_should_refuse_mismatched_or_corrupt_mixed_snapshot_before_archive_open(
    data, candidate, tmp_path, monkeypatch, damage
):
    monkeypatch.setattr(case, "stage", tmp_path)
    state = candidate.snapshot_state()
    report = dict(sequence=deepcopy(data["mixed"]))
    if damage == "mse_policy":
        report["sequence"]["policy"]["binary_gripper_axis"] = None
    elif damage == "updates":
        state = replace(state, completed_updates=24)
    elif damage == "update_type":
        state = replace(state, completed_updates=96.0)
    elif damage == "rows":
        state = replace(state, training_examples=223)
    elif damage == "dtype":
        state = replace(
            state, velocity=(state.velocity[0].astype(np.float32),) + state.velocity[1:]
        )
    else:
        state.velocity[3][0] = np.nan

    class SnapshotOnly:
        policy = state.policy

        def snapshot_state(self):
            return state

    with pytest.raises((AssertionError, ValueError, TypeError)):
        case.save_state(SnapshotOnly(), report)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", ["write", "size"])
def test_should_retain_attempted_archive_work_and_paid_sequence_on_storage_failure(
    data, candidate, tmp_path, monkeypatch, failure
):
    monkeypatch.setattr(case, "stage", tmp_path)

    def save(handle, **_arrays):
        if failure == "write":
            raise OSError("mixed archive write failed")
        handle.write(b"x" * (128 * 1024 + 1))

    monkeypatch.setattr(case.np, "savez", save)
    report = dict(sequence=deepcopy(data["mixed"]))
    with pytest.raises((OSError, AssertionError)):
        case.save_state(candidate, report)
    assert report["state_archive"]["save_attempts"] == 1
    assert report["state_archive"]["save_returns"] == (0 if failure == "write" else 1)
    assert report["sequence"]["progress"]["updates_completed"] == 96
    assert len(report["sequence"]["applied_updates"]) == 96


def test_should_compose_one_fresh_mixed32_run_after_source_and_initialization_gates(
    data, candidate, tmp_path, monkeypatch
):
    monkeypatch.setattr(case, "stage", tmp_path)
    calls = []

    def load(report):
        calls.append("source")
        return data["warm"], data["new"], data["mse"], data["statistics"]

    parity = dict(
        template_initializations=2, template_optimizer_updates=0, all_eight_arrays_exact=[True] * 8
    )

    def initialize(spec, statistics):
        calls.append(("parity", spec, statistics))
        return deepcopy(parity)

    result = ContinuousLearnabilityResult(deepcopy(data["mixed"]), candidate)

    def run(*args, **options):
        calls.append(("run", args, options))
        return result

    monkeypatch.setattr(case, "load_inputs", load)
    monkeypatch.setattr(case, "initial_parity", initialize)
    monkeypatch.setattr(case, "run_continuous_learnability", run)
    report = dict(complete=False)
    case.run_case(report)
    assert calls == [
        "source",
        ("parity", data["warm"].spec, data["statistics"]),
        (
            "run",
            (data["warm"], data["new"]),
            dict(
                expected_standardization=data["statistics"], binary_gripper_axis=3, phase_updates=32
            ),
        ),
    ]
    assert (
        report["complete"]
        is report["initialization_parity_attempted"]
        is report["sequence_attempted"]
        is True
    )
    assert report["sequence"] is result.report and report["mse"] is data["mse"]
    assert report["initialization_parity"] == parity
    assert report["comparison"]["primary"] == dict(warm=0.0, new=0.0)
    assert report["comparison"]["work_differences"]["optimizer_updates"] == 0
    assert report["comparison"]["work_differences"]["dense_multiply_accumulates"] == 0
    assert report["comparison"]["work_differences"]["binary_gripper_loss_rows"] == 244
    assert report["state_archive"]["updates"] == 96 and report["state_archive"]["save_returns"] == 1


def test_should_stop_before_new_model_run_when_template_parity_fails(data, tmp_path, monkeypatch):
    monkeypatch.setattr(case, "stage", tmp_path)
    monkeypatch.setattr(
        case,
        "load_inputs",
        lambda _report: (data["warm"], data["new"], data["mse"], data["statistics"]),
    )

    def fail(*_args):
        raise AssertionError("initial arrays differ")

    monkeypatch.setattr(case, "initial_parity", fail)
    monkeypatch.setattr(case, "run_continuous_learnability", forbidden)
    assert case.main() == 1
    report = json.loads((tmp_path / "case.json").read_text())
    assert report["complete"] is False and report["initialization_parity_attempted"] is True
    assert "sequence" not in report and "state_archive" not in report
    assert "initial arrays differ" in report["error"]
    assert report["mse_new_optimizer_updates"] == 0


@pytest.mark.parametrize("failure", ["incomplete", "no_candidate", "initial_physical", "archive"])
def test_should_preserve_paid_mixed_report_before_later_verification_or_archive_failure(
    data, candidate, tmp_path, monkeypatch, failure
):
    monkeypatch.setattr(case, "stage", tmp_path)
    monkeypatch.setattr(
        case,
        "load_inputs",
        lambda _report: (data["warm"], data["new"], data["mse"], data["statistics"]),
    )
    monkeypatch.setattr(
        case, "initial_parity", lambda *_args: dict(all_eight_arrays_exact=[True] * 8)
    )
    payload = deepcopy(data["mixed"])
    if failure == "incomplete":
        payload.update(complete=False, error="paid endpoint validation failed")
        payload.pop("work")
        payload.pop("metrics")
    elif failure == "initial_physical":
        payload["checkpoints"][0]["contexts"]["warm"]["actor_predictions"][0][0] = 0.1
    result = ContinuousLearnabilityResult(
        payload, None if failure in ("incomplete", "no_candidate") else candidate
    )
    monkeypatch.setattr(case, "run_continuous_learnability", lambda *_a, **_k: result)
    if failure == "archive":

        def fail(*_args):
            raise OSError("paid mixed archive failed")

        monkeypatch.setattr(case, "save_state", fail)
    assert case.main() == 1
    report = json.loads((tmp_path / "case.json").read_text())
    assert report["complete"] is False and "error" in report
    assert report["sequence"]["progress"]["updates_completed"] == 96
    assert len(report["sequence"]["applied_updates"]) == len(report["sequence"]["events"]) == 96
    assert report["sequence"]["policy"]["binary_gripper_axis"] == 3
    assert report["sequence"]["applied_updates"][0]["diagnostic"]["definition"] == MIXED_OBJECTIVE
    assert not (tmp_path / "final-state.npz").exists()
    assert (
        report["mse_new_optimizer_updates"]
        == report["simulator_calls"]
        == report["guard_reads"]
        == report["probe_reads"]
        == report["final_reads"]
        == report["promotion_calls"]
        == 0
    )


@pytest.mark.parametrize("name", ["case.json", "final-state.npz"])
def test_should_preserve_existing_outputs_before_loading_or_initializing_models(
    tmp_path, monkeypatch, name
):
    monkeypatch.setattr(case, "stage", tmp_path)
    path = tmp_path / name
    path.write_bytes(b"accepted or failed prior evidence")
    monkeypatch.setattr(case, "load_inputs", forbidden)
    monkeypatch.setattr(case, "initial_parity", forbidden)
    monkeypatch.setattr(case, "run_continuous_learnability", forbidden)
    with pytest.raises(FileExistsError):
        case.main()
    assert path.read_bytes() == b"accepted or failed prior evidence"
    assert len(list(tmp_path.iterdir())) == 1
