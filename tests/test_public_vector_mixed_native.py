"""Saved mixed-policy binding and native composition with synthetic evidence."""

from copy import deepcopy
from dataclasses import replace
import importlib
import json
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.adapters.numpy_continuous_bc import NumpyContinuousBCLearner
from src.core import continuous_bc
from src.core.pick_cube_public_vectors import public_vector_projection
from test_maniskill_bc_smoke import assert_same_state
from test_public_vector_mlp_depth_native import data as source_data  # noqa: F401 - pytest fixture
from test_public_vector_mlp_native import BIAS, model_rollout
from test_visual_native_context_case import write_new


case = importlib.import_module("artifacts.runs.public-vector-mixed-native-20261019.native_case")


def forbidden(*_args, **_kwargs):
    pytest.fail("invalid evidence admitted more work")


@pytest.fixture(scope="module")
def data(source_data):  # noqa: F811 - pytest injects the imported fixture by name
    """Derive metadata/state only; inherited fixture owns ONE constructor+restore."""
    record = deepcopy(source_data["record"])
    record["mse"] = deepcopy(record["sequence"])
    mixed = record["sequence"]
    mixed["policy"]["binary_gripper_axis"] = 3
    mixed["work"]["binary_gripper_loss_rows"] = 244
    for event, receipt in zip(mixed["events"], mixed["applied_updates"], strict=True):
        receipt["diagnostic"]["definition"] = continuous_bc.MIXED_OBJECTIVE
        event["applied"] = [deepcopy(receipt)]
    for point in mixed["checkpoints"]:
        for result in point["contexts"].values():
            result["conditioning"]["conditioning"].update(
                objective_definition=continuous_bc.MIXED_OBJECTIVE, objective_loss=2.0
            )
    record["mse_new_optimizer_updates"] = 0
    policy = replace(source_data["policy"], binary_gripper_axis=3)
    state = replace(source_data["state"], policy=policy)
    return dict(
        source_data, record=record, policy=policy, mse_policy=source_data["policy"], state=state
    )


@pytest.fixture
def archive(data, tmp_path):
    path = tmp_path / "final-state.npz"
    np.savez(path, **data["arrays"])
    record = deepcopy(data["record"])
    record["state_archive"]["file_bytes"] = path.stat().st_size
    return path, record


def test_should_restore_full_mixed_policy_parameters_momentum_and_counters(
    data, archive, monkeypatch
):
    path, record = archive
    original = deepcopy(record)
    monkeypatch.setattr(NumpyContinuousBCLearner, "train_batch", forbidden)
    audit = {}
    owner = case.restore_candidate(record, path, data["warm"].spec, audit)
    assert_same_state(data["state"], owner.snapshot_state())
    assert owner.policy.binary_gripper_axis == 3
    assert owner.policy == replace(data["mse_policy"], binary_gripper_axis=3) == data["policy"]
    assert record == original
    assert audit["source_updates"] == 96 and audit["source_training_rows"] == 224
    assert audit["source_training_objective"] == continuous_bc.MIXED_OBJECTIVE
    assert audit["constructor_attempts"] == audit["constructor_returns"] == 1
    assert audit["restore_attempts"] == audit["restore_returns"] == 1
    assert audit["archive_numeric_bytes"] == 26688 and audit["archive_array_values"] == 3336
    exported = owner.snapshot_state()
    exported.parameters[3][0] += 1.0
    exported.velocity[0][0, 0] -= 1.0
    assert_same_state(data["state"], owner.snapshot_state())


@pytest.mark.parametrize(
    "damage",
    [
        "mse_axis",
        "wrong_axis",
        "axis_type",
        "extra_policy",
        "updates",
        "rows",
        "save_type",
        "names",
        "complete",
    ],
)
def test_should_refuse_wrong_mixed_policy_or_archive_metadata_before_arrays(
    data, archive, monkeypatch, damage
):
    path, record = archive
    policy, saved = record["sequence"]["policy"], record["state_archive"]
    if damage == "mse_axis":
        policy["binary_gripper_axis"] = None
    elif damage == "wrong_axis":
        policy["binary_gripper_axis"] = 2
    elif damage == "axis_type":
        policy["binary_gripper_axis"] = 3.0
    elif damage == "extra_policy":
        policy["selected_endpoint"] = 96
    elif damage == "updates":
        saved["updates"] = 24
    elif damage == "rows":
        saved["training_examples"] = 223
    elif damage == "save_type":
        saved["save_returns"] = True
    elif damage == "names":
        saved["array_names"] = list(reversed(saved["array_names"]))
    else:
        record["complete"] = 1
    monkeypatch.setattr(case.prior, "read_arrays", forbidden)
    with pytest.raises((AssertionError, ValueError, TypeError)):
        case.restore_candidate(record, path, data["warm"].spec, {})


def test_should_run_only_physical_readout_for_all42_scalars_without_ce_or_gradient(
    data, archive, monkeypatch
):
    path, record = archive
    owner = case.restore_candidate(record, path, data["warm"].spec, {})
    monkeypatch.setattr(NumpyContinuousBCLearner, "train_batch", forbidden)
    monkeypatch.setattr(NumpyContinuousBCLearner, "diagnose_training_conditioning", forbidden)
    monkeypatch.setattr(continuous_bc, "continuous_bc_loss_and_gradients", forbidden)
    audit = dict(readouts=[])
    case.prior.verify_readouts(owner, data["warm"], data["new"], record["sequence"], audit)
    assert audit["readout_numeric_comparisons"] == 42
    assert [row["diagnostic"]["prediction_rows"] for row in audit["readouts"]] == [2, 3]
    assert [row["passed"] for row in audit["readouts"]] == [True, True]
    assert_same_state(data["state"], owner.snapshot_state())


@pytest.mark.parametrize("context", ["warm", "new"])
def test_should_preserve_returned_physical_diagnostics_when_saved_mixed_readout_is_wrong(
    data, archive, context
):
    path, record = archive
    owner = case.restore_candidate(record, path, data["warm"].spec, {})
    residual = record["sequence"]["checkpoints"][-1]["contexts"][context]["residual"]
    residual["residuals"]["overall"]["normalized_half_mse"] += 0.001
    residual["residuals"]["stages"][0]["summary"] = deepcopy(residual["residuals"]["overall"])
    audit = dict(readouts=[])
    with pytest.raises(AssertionError, match="saved residual differs"):
        case.prior.verify_readouts(owner, data["warm"], data["new"], record["sequence"], audit)
    assert audit["readouts"][-1]["returned"] is True and audit["readouts"][-1]["passed"] is False
    assert len(audit["readouts"]) == (1 if context == "warm" else 2)


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "zero_type",
        "work",
        "ce_type",
        "fitted_rows",
        "raw_rows",
        "projection",
        "mse_flag",
        "pair",
        "readout",
    ],
)
def test_should_bind_mixed_source_and_distinct_mse_policy_before_restore_with_reused_qualification(
    data, tmp_path, monkeypatch, damage
):
    """Stubs isolate fixed29/30 admission wiring; accepted qualification is reused."""
    record = deepcopy(data["record"])
    for name in ("sequence", "mse"):
        record[name]["policy"]["observation_standardization"]["fitted_rows"] = 29
    record["sequence"]["work"].update(
        optimizer_updates=96,
        training_rows=2816,
        unique_source_rows=59,
        repeated_training_rows=2757,
        replay_budget_rows=0,
        fit_calls=1,
        fit_rows=29,
        fit_feature_values=609,
        binary_gripper_loss_rows=3052,
    )
    warm = SimpleNamespace(transitions=(data["episodes"]["warm"].transitions[0],) * 29)
    new = SimpleNamespace(transitions=(data["episodes"]["new"].transitions[0],) * 30)
    if damage == "zero_type":
        record["mse_new_optimizer_updates"] = False
    elif damage == "work":
        record["sequence"]["work"]["training_rows"] = 224
    elif damage == "ce_type":
        record["sequence"]["work"]["binary_gripper_loss_rows"] = 3052.0
    elif damage == "fitted_rows":
        record["sequence"]["policy"]["observation_standardization"]["fitted_rows"] = 2
    elif damage == "raw_rows":
        new = SimpleNamespace(transitions=new.transitions[:-1])
    elif damage == "projection":
        record["input_projection"] = dict(changed=True)
    elif damage == "mse_flag":
        record["mse"]["policy"]["binary_gripper_axis"] = 3
    write_new(tmp_path / "case.json", record)
    monkeypatch.setattr(case, "ACCEPTED_STAGE", tmp_path)
    calls, sentinel = [], object()

    def read(_path, seed, **options):
        calls.append(("read", seed, options))
        return warm if seed == 17 else new

    def project(first, second):
        assert first is warm and second is new
        calls.append(("project",))
        return data["warm"], data["new"], data["record"]["input_projection"]

    def pair(current, mse, projected_warm, projected_new):
        calls.append(("pair",))
        assert projected_warm is data["warm"] and projected_new is data["new"]
        assert case.exact_json(current, record["sequence"]) and case.exact_json(mse, record["mse"])
        if damage == "pair":
            raise AssertionError("saved objective comparison refused")

    def restore(current, path, spec, audit):
        if damage not in (None, "readout"):
            pytest.fail("invalid mixed source reached restore")
        calls.append(("restore", path.name, spec.observation_dim))
        assert path == tmp_path / "final-state.npz"
        return sentinel

    def readout(owner, projected_warm, projected_new, current, audit):
        assert owner is sentinel and projected_warm is data["warm"] and projected_new is data["new"]
        calls.append(("readout",))
        if damage == "readout":
            raise AssertionError("saved physical readout refused")

    monkeypatch.setattr(case, "_read_teacher", read)
    monkeypatch.setattr(case, "_project_public_inputs", project)
    monkeypatch.setattr(case.mixed, "verify_pair", pair)
    monkeypatch.setattr(case, "restore_candidate", restore)
    monkeypatch.setattr(case.prior, "verify_readouts", readout)
    audit = dict(readouts=[])
    if damage is None:
        owner, observed_warm, observed_new, mse_policy = case.load_candidate(audit)
        assert owner is sentinel and observed_warm is warm and observed_new is new
        assert mse_policy.binary_gripper_axis is None
        assert mse_policy.observation_standardization.fitted_rows == 29
        assert audit["baseline_policy"]["binary_gripper_axis"] is None
        assert (
            audit["source_python"] == "frozen-source-python"
            and audit["source_numpy"] == "frozen-source-numpy"
        )
        assert calls == [
            ("read", 17, {"visual_observation": True}),
            ("read", 19, {"visual_observation": True}),
            ("project",),
            ("pair",),
            ("restore", "final-state.npz", 21),
            ("readout",),
        ]
    else:
        with pytest.raises(AssertionError):
            case.load_candidate(audit)
        assert not any(item[0] == "restore" for item in calls) or damage == "readout"


@pytest.fixture
def scenario(data, archive, tmp_path, monkeypatch):
    path, record = archive
    stage, baseline = tmp_path / "current", tmp_path / "mse-native"
    stage.mkdir()
    baseline.mkdir()
    monkeypatch.setattr(case, "stage", stage)
    monkeypatch.setattr(case, "BASELINE_STAGE", baseline)
    reference_state = replace(data["state"], policy=data["mse_policy"], training_examples=2816)
    for name, context in (("a-rollout.json", "warm"), ("b-rollout.json", "new")):
        raw = model_rollout(data["raw"][context], reference_state, success=context == "new")
        raw.update(source_updates=96, source_training_rows=2816)
        write_new(baseline / name, raw)
    observed = dict(calls=[], failure=None, successes={17: True, 19: False})

    def load(audit):
        owner = case.restore_candidate(record, path, data["warm"].spec, audit)
        case.prior.verify_readouts(owner, data["warm"], data["new"], record["sequence"], audit)
        return owner, data["episodes"]["warm"], data["episodes"]["new"], data["mse_policy"]

    def rollout(output, *, learner, train_seed, visual_observation, observation_projection):
        assert visual_observation is True and observation_projection == public_vector_projection()
        observed["calls"].append((output.name, train_seed, learner))
        raw = model_rollout(
            data["raw"]["warm" if train_seed == 17 else "new"],
            data["state"],
            success=observed["successes"][train_seed],
        )
        raw.update(source_updates=96, source_training_rows=224)
        for row in raw["episode"]["transitions"]:
            prediction = learner.predict(np.asarray([row["observation"][:21]], dtype=np.float64))
            row["action"] = prediction[0].astype(np.float32).tolist()
            assert (
                row["action"]
                == np.asarray([math.tanh(bias) for bias in BIAS], dtype=np.float32).tolist()
            )
        if observed["failure"] == "current_mse_flag":
            raw["learner_policy"]["binary_gripper_axis"] = None
        elif observed["failure"] == "current_sparse_type":
            raw["sparse_return"] = True
        elif observed["failure"] == "momentum":
            state = learner.snapshot_state()
            state.velocity[3][0] += 0.5
            learner.restore_state(state)
        write_new(output, raw)
        if observed["failure"] == "unreturned":
            raise RuntimeError("paid mixed controls did not return")
        returned = deepcopy(raw)
        if observed["failure"] == "returned":
            returned["elapsed_seconds"] += 0.1
        return case.smoke.FrozenRolloutResult(returned, None)

    monkeypatch.setattr(case, "load_candidate", load)
    monkeypatch.setattr(case.smoke, "run_frozen_rollout", rollout)
    monkeypatch.setattr(NumpyContinuousBCLearner, "train_batch", forbidden)
    monkeypatch.setattr(NumpyContinuousBCLearner, "diagnose_training_conditioning", forbidden)
    monkeypatch.setattr(continuous_bc, "continuous_bc_loss_and_gradients", forbidden)
    return dict(stage=stage, baseline=baseline, observed=observed)


@pytest.mark.parametrize("warm_success", [True, False])
def test_should_keep_mixed_and_mse_policies_distinct_and_accept_negative_task_outcomes(
    data, scenario, warm_success
):
    scenario["observed"]["successes"][17] = warm_success
    original_references = {path.name: path.read_bytes() for path in scenario["baseline"].iterdir()}
    audit = dict(complete=False, readouts=[], rollouts=[])
    case.run_case(audit)
    assert audit["complete"] is audit["final_full_state_exact"] is True
    assert [(name, seed) for name, seed, _ in scenario["observed"]["calls"]] == [
        ("a-rollout.json", 17),
        ("b-rollout.json", 19),
    ]
    owner = scenario["observed"]["calls"][0][2]
    assert scenario["observed"]["calls"][1][2] is owner
    assert owner.policy.binary_gripper_axis == 3
    assert_same_state(data["state"], owner.snapshot_state())
    assert audit["reference_rows"] == 5 and audit["reference_observation_values"] == 7890
    assert audit["reference_action_values"] == 20
    assert audit["primary"] == [
        dict(
            seed=17,
            sparse_return_delta=float(warm_success),
            any_success_delta=float(warm_success),
            final_success_delta=float(warm_success),
        ),
        dict(seed=19, sparse_return_delta=-1.0, any_success_delta=-1.0, final_success_delta=-1.0),
    ]
    for name, original in original_references.items():
        assert (scenario["baseline"] / name).read_bytes() == original
        reference = json.loads(original)
        current = json.loads((scenario["stage"] / name).read_text())
        assert reference["learner_policy"]["binary_gripper_axis"] is None
        assert current["learner_policy"]["binary_gripper_axis"] == 3
        assert current["episode"]["spec"]["observation_dim"] == 789
    for row in audit["rollouts"]:
        assert row["returned"] is row["qualified"] is row["state_exact"] is True
        assert row["prediction_bytes"] == row["steps"] * 32
        assert row["forward_dense_macs"] == row["steps"] * 1600
        assert row["normalization_values"] == row["steps"] * 21
        assert row["projection_work"]["projected_values"] == row["steps"] * 21
        assert row["visual_work"]["returned_feature_values"] == (row["steps"] + 1) * 789


@pytest.mark.parametrize("damage", ["mixed_flag", "source", "sparse_type"])
def test_should_validate_reference_with_its_own_mse_policy_before_new_controls(scenario, damage):
    path = scenario["baseline"] / "a-rollout.json"
    raw = json.loads(path.read_text())
    if damage == "mixed_flag":
        raw["learner_policy"]["binary_gripper_axis"] = 3
    elif damage == "source":
        raw["source_training_rows"] = 224
    else:
        raw["sparse_return"] = False
    path.write_text(json.dumps(raw), encoding="utf8")
    audit = dict(complete=False, readouts=[], rollouts=[])
    with pytest.raises((AssertionError, ValueError)):
        case.run_case(audit)
    assert scenario["observed"]["calls"] == [] and audit["rollouts"] == []
    assert not (scenario["stage"] / "a-rollout.json").exists()


@pytest.mark.parametrize(
    "failure", ["current_mse_flag", "current_sparse_type", "momentum", "returned", "unreturned"]
)
def test_should_keep_first_paid_mixed_trace_and_refuse_further_controls_after_failure(
    data, scenario, failure
):
    scenario["observed"]["failure"] = failure
    audit = dict(complete=False, readouts=[], rollouts=[])
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        case.run_case(audit)
    assert len(scenario["observed"]["calls"]) == 1
    assert (scenario["stage"] / "a-rollout.json").exists()
    assert not (scenario["stage"] / "b-rollout.json").exists()
    row = audit["rollouts"][0]
    assert row["attempted"] is True and row["qualified"] is False
    assert row["returned"] is (failure != "unreturned")
    raw = json.loads((scenario["stage"] / "a-rollout.json").read_text())
    assert raw["work"]["completed_steps"] == raw["prediction_returns"] == 2
    if failure == "momentum":
        assert row["state_exact"] is False
    else:
        assert_same_state(data["state"], scenario["observed"]["calls"][0][2].snapshot_state())


def test_should_reread_first_trace_after_owner_getter_before_second_native_call(
    scenario, monkeypatch
):
    audit = dict(complete=False, readouts=[], rollouts=[])
    guard, changed = case.prior.same_state, []

    def observe(expected, owner):
        guard(expected, owner)
        if audit["rollouts"] and audit["rollouts"][0]["qualified"] and not changed:
            path = scenario["stage"] / "a-rollout.json"
            raw = json.loads(path.read_text())
            raw["elapsed_seconds"] += 0.25
            path.write_text(json.dumps(raw), encoding="utf8")
            changed.append(path)

    monkeypatch.setattr(case.prior, "same_state", observe)
    with pytest.raises(AssertionError, match="preceding mixed trace changed"):
        case.run_case(audit)
    assert len(changed) == 1 and len(scenario["observed"]["calls"]) == 1
    assert audit["rollouts"][0]["qualified"] is True
    assert not (scenario["stage"] / "b-rollout.json").exists()


@pytest.mark.parametrize(
    "name", ["policy-rollout.json", "rollout-audit.json", "a-rollout.json", "b-rollout.json"]
)
def test_should_preserve_existing_outputs_before_loading_or_native_work(
    tmp_path, monkeypatch, name
):
    monkeypatch.setattr(case, "stage", tmp_path)
    monkeypatch.setattr(sys, "argv", ["case", "--output", str(tmp_path / "policy-rollout.json")])
    path = tmp_path / name
    path.write_bytes(b"preserved mixed evidence")
    monkeypatch.setattr(case, "run_case", forbidden)
    with pytest.raises(FileExistsError):
        case.main()
    assert path.read_bytes() == b"preserved mixed evidence"


def test_should_refuse_output_alias_before_source_work(tmp_path, monkeypatch):
    monkeypatch.setattr(case, "stage", tmp_path)
    monkeypatch.setattr(sys, "argv", ["case", "--output", str(tmp_path / "a-rollout.json")])
    monkeypatch.setattr(case, "run_case", forbidden)
    with pytest.raises(AssertionError):
        case.main()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", ["runtime", "summary"])
def test_should_persist_paid_readouts_and_controls_when_runtime_or_summary_fails(
    tmp_path, monkeypatch, failure
):
    monkeypatch.setattr(case, "stage", tmp_path)
    monkeypatch.setattr(sys, "argv", ["case", "--output", str(tmp_path / "policy-rollout.json")])

    def run(audit):
        audit.update(
            complete=True,
            source_training_objective=continuous_bc.MIXED_OBJECTIVE,
            source_updates=96,
            source_training_rows=2816,
            readouts=[dict(returned=True, prediction_rows=29)],
            rollouts=[dict(seed=17, returned=True, steps=29)],
        )
        if failure == "runtime":
            raise RuntimeError("paid mixed controls failed")

    monkeypatch.setattr(case, "run_case", run)
    original = case.prior.write_json

    def write(path, value):
        if failure == "summary" and path.name == "policy-rollout.json":
            raise OSError("mixed summary write failed")
        original(path, value)

    monkeypatch.setattr(case.prior, "write_json", write)
    assert case.main() == 1
    audit = json.loads((tmp_path / "rollout-audit.json").read_text())
    assert audit["complete"] is False
    assert audit["source_training_objective"] == continuous_bc.MIXED_OBJECTIVE
    assert audit["source_updates"] == 96 and audit["source_training_rows"] == 2816
    assert audit["readouts"] == [dict(returned=True, prediction_rows=29)]
    assert audit["rollouts"] == [dict(seed=17, returned=True, steps=29)]
    assert audit["optimizer_updates"] == audit["fit_calls"] == 0
    assert ("summary_write_error" if failure == "summary" else "error") in audit
