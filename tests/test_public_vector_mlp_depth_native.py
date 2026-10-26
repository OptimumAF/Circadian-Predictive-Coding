"""Saved p32 native composition with synthetic receipts; zero fit or SGD."""

from copy import deepcopy
from dataclasses import replace
import importlib
import json
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.run_native_stream_demo import _project_public_inputs
from src.adapters.numpy_continuous_bc import ContinuousBCPolicy, NumpyContinuousBCLearner
from src.core.observation_standardization import ObservationStandardization
from src.core.pick_cube_public_vectors import public_vector_projection
from test_maniskill_bc_smoke import assert_same_state
from test_public_vector_mlp_depth import metrics, sequence
from test_public_vector_mlp_native import BIAS, model_rollout, residual_reference
from test_public_vector_native_case import episode_from_teacher
from test_visual_native_context_case import teacher_receipt, write_new


case = importlib.import_module("artifacts.runs.public-vector-mlp-depth-native-20261019.native_case")


def forbidden(*_args, **_kwargs):
    pytest.fail("failed prerequisite admitted further work")


@pytest.fixture(scope="module")
def data():
    """Five unique TRAIN packets; ONE constructor and restore, no forward."""
    raw = dict(warm=teacher_receipt(17, 2), new=teacher_receipt(19, 3))
    episodes = {name: episode_from_teacher(value) for name, value in raw.items()}
    warm, new, projection = _project_public_inputs(episodes["warm"], episodes["new"])
    statistics = ObservationStandardization((0.5,) + (0.0,) * 20, (0.5,) + (1.0,) * 20, 2)
    policy = ContinuousBCPolicy(warm.spec, observation_standardization=statistics)
    owner = NumpyContinuousBCLearner(policy)
    initial = owner.snapshot_state()
    parameters = tuple(np.zeros_like(value) for value in initial.parameters)
    parameters[3][:] = BIAS
    velocity = tuple(
        np.full_like(value, (index + 1) * 0.125) for index, value in enumerate(initial.velocity)
    )
    owner.restore_state(
        replace(
            initial,
            parameters=parameters,
            velocity=velocity,
            completed_updates=96,
            training_examples=224,
        )
    )
    current, baseline = sequence(warm, new, policy, 32), sequence(warm, new, policy, 8)
    for report in (current, baseline):
        for point in report["checkpoints"]:
            for name, episode in (("warm", warm), ("new", new)):
                diagnostic = residual_reference(episode)
                point["contexts"][name]["residual"] = diagnostic
                point["contexts"][name]["conditioning"]["conditioning"]["objective_loss"] = (
                    diagnostic["residuals"]["overall"]["normalized_half_mse"]
                )
        report["metrics"] = metrics(report["checkpoints"])
    names = [f"{group}_{index}" for group in ("parameters", "velocity") for index in range(4)]
    record = dict(
        complete=True,
        python="frozen-source-python",
        numpy="frozen-source-numpy",
        input_projection=projection,
        sequence=current,
        baseline=baseline,
        state_archive=dict(
            updates=96,
            training_examples=224,
            array_names=names,
            numeric_bytes=26688,
            save_attempts=1,
            save_returns=1,
        ),
    )
    record.update(
        baseline_new_optimizer_updates=0,
        simulator_calls=0,
        guard_reads=0,
        probe_reads=0,
        final_reads=0,
        promotion_calls=0,
    )
    state = owner.snapshot_state()
    arrays = {
        f"{group}_{index}": value
        for group in ("parameters", "velocity")
        for index, value in enumerate(getattr(state, group))
    }
    return dict(
        raw=raw,
        episodes=episodes,
        warm=warm,
        new=new,
        policy=policy,
        state=state,
        arrays=arrays,
        record=record,
    )


@pytest.fixture
def archive(data, tmp_path):
    path = tmp_path / "final-state.npz"
    np.savez(path, **data["arrays"])
    record = deepcopy(data["record"])
    record["state_archive"]["file_bytes"] = path.stat().st_size
    return path, record


def test_should_restore_saved96_model_with_all_eight_arrays_and_frozen_state_ownership(
    data, archive, monkeypatch
):
    path, record = archive
    original = deepcopy(record)
    monkeypatch.setattr(NumpyContinuousBCLearner, "train_batch", forbidden)
    audit = {}
    owner = case.restore_candidate(record, path, data["warm"].spec, audit)
    assert_same_state(data["state"], owner.snapshot_state())
    assert record == original
    assert audit["source_updates"] == 96 and audit["source_training_rows"] == 224
    assert audit["constructor_attempts"] == audit["constructor_returns"] == 1
    assert audit["restore_attempts"] == audit["restore_returns"] == 1
    assert audit["archive_array_values"] == 3336 and audit["archive_numeric_bytes"] == 26688
    exported = owner.snapshot_state()
    exported.parameters[3][0] += 1.0
    exported.velocity[0][0, 0] += 1.0
    assert_same_state(data["state"], owner.snapshot_state())


@pytest.mark.parametrize(
    "damage",
    [
        "complete",
        "sequence",
        "updates",
        "update_type",
        "rows",
        "row_type",
        "policy",
        "axis",
        "statistics",
        "save",
        "names",
    ],
)
def test_should_refuse_wrong_source_contract_before_reading_saved_arrays(
    data, archive, monkeypatch, damage
):
    path, record = archive
    policy, saved = record["sequence"]["policy"], record["state_archive"]
    if damage == "complete":
        record["complete"] = 1
    elif damage == "sequence":
        record["sequence"]["complete"] = 1
    elif damage == "updates":
        saved["updates"] = 24
    elif damage == "update_type":
        saved["updates"] = 96.0
    elif damage == "rows":
        saved["training_examples"] = 223
    elif damage == "row_type":
        saved["training_examples"] = True
    elif damage == "policy":
        policy["training_role"] = "probe_train"
    elif damage == "axis":
        policy["binary_gripper_axis"] = 3
    elif damage == "statistics":
        policy["observation_standardization"]["means"] = (0.5,) * 20
    elif damage == "save":
        saved["save_returns"] = True
    else:
        saved["array_names"] = list(reversed(saved["array_names"]))
    # Patch the already accepted archive reader, not its internals.
    monkeypatch.setattr(case.prior, "read_arrays", forbidden)
    with pytest.raises((AssertionError, ValueError, TypeError)):
        case.restore_candidate(record, path, data["warm"].spec, {})


def test_should_reproduce_all42_saved_scalars_read_only_before_native_controls(data, archive):
    path, record = archive
    owner = case.restore_candidate(record, path, data["warm"].spec, {})
    audit = dict(readouts=[])
    case.prior.verify_readouts(owner, data["warm"], data["new"], record["sequence"], audit)
    assert audit["readout_numeric_comparisons"] == 42
    assert [row["passed"] for row in audit["readouts"]] == [True, True]
    assert [row["diagnostic"]["prediction_rows"] for row in audit["readouts"]] == [2, 3]
    assert_same_state(data["state"], owner.snapshot_state())


@pytest.mark.parametrize("context", ["warm", "new"])
def test_should_stop_on_changed_saved_final_readout_and_keep_completed_diagnostic(
    data, archive, context
):
    path, record = archive
    owner = case.restore_candidate(record, path, data["warm"].spec, {})
    diagnostic = record["sequence"]["checkpoints"][-1]["contexts"][context]["residual"]
    overall = diagnostic["residuals"]["overall"]
    overall["axes"][3]["maximum_absolute_normalized_error"] += 0.01
    diagnostic["residuals"]["stages"][0]["summary"] = deepcopy(overall)
    audit = dict(readouts=[])
    with pytest.raises(AssertionError, match="saved residual differs"):
        case.prior.verify_readouts(owner, data["warm"], data["new"], record["sequence"], audit)
    assert len(audit["readouts"]) == (1 if context == "warm" else 2)
    assert audit["readouts"][-1]["returned"] is True and audit["readouts"][-1]["passed"] is False
    assert_same_state(data["state"], owner.snapshot_state())


def raw_rollout(data, seed, *, success=False, updates=96, training_rows=224):
    raw = model_rollout(
        data["raw"]["warm" if seed == 17 else "new"], data["state"], success=success
    )
    raw.update(source_updates=updates, source_training_rows=training_rows)
    return raw


@pytest.mark.parametrize("success", [False, True])
def test_should_validate_generic_saved_counters_and_preserve_complete_negative_rollouts(
    data, success
):
    raw = raw_rollout(data, 17, success=success)
    audit = {}
    case.validate_rollout(raw, data["episodes"]["warm"], 17, data["policy"], 96, 224, audit)
    assert audit["qualified"] is True and audit["steps"] == 2
    assert audit["sparse_return"] == int(success)
    assert audit["any_success"] is audit["final_success"] is success
    assert audit["prediction_bytes"] == 64 and audit["forward_dense_macs"] == 3200
    assert audit["normalization_values"] == 42
    assert audit["projection_work"]["source_values_validated"] == 1578
    assert audit["action_bound_violations"] == 0


@pytest.mark.parametrize(
    "damage",
    [
        "updates",
        "rows",
        "counter_type",
        "policy",
        "projection",
        "role",
        "actions",
        "visual",
        "frozen",
    ],
)
def test_should_refuse_corrupted_native_policy_counters_projection_and_work(data, damage):
    raw = raw_rollout(data, 17)
    if damage == "updates":
        raw["source_updates"] = 24
    elif damage == "rows":
        raw["source_training_rows"] = 704
    elif damage == "counter_type":
        raw["validated_prediction_bytes"] = 64.0
    elif damage == "policy":
        raw["learner_policy"]["momentum"] = 0.8
    elif damage == "projection":
        raw["projection_work"]["projected_values"] = 41
    elif damage == "role":
        raw["episode"]["role"] = "probe_train"
    elif damage == "actions":
        raw["episode"]["transitions"][0]["action"][0] = 1.01
    elif damage == "visual":
        raw["visual_work"]["returned_feature_values"] = 0
    else:
        raw["frozen_state_exact"] = False
    with pytest.raises((AssertionError, ValueError)):
        case.validate_rollout(raw, data["episodes"]["warm"], 17, data["policy"], 96, 224, {})


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "zero_type",
        "work",
        "work_type",
        "fitted_rows",
        "raw_rows",
        "projection",
        "depth",
        "readout",
    ],
)
def test_should_wire_strict_actual_source_gates_before_restore_and_readouts_with_reused_qualification(
    data, tmp_path, monkeypatch, damage
):
    """Qualification/depth math are reused; stubs isolate loader order, not proof."""
    record = deepcopy(data["record"])
    record["sequence"]["policy"]["observation_standardization"]["fitted_rows"] = 29
    record["sequence"]["work"].update(
        optimizer_updates=96,
        training_rows=2816,
        unique_source_rows=59,
        repeated_training_rows=2757,
        replay_budget_rows=0,
        fit_calls=1,
        fit_rows=29,
        fit_feature_values=609,
    )
    raw_a = SimpleNamespace(transitions=(data["episodes"]["warm"].transitions[0],) * 29)
    raw_b = SimpleNamespace(transitions=(data["episodes"]["new"].transitions[0],) * 30)
    if damage == "zero_type":
        record["guard_reads"] = False
    elif damage == "work":
        record["sequence"]["work"]["training_rows"] = 224
    elif damage == "work_type":
        record["sequence"]["work"]["optimizer_updates"] = 96.0
    elif damage == "fitted_rows":
        record["sequence"]["policy"]["observation_standardization"]["fitted_rows"] = 2
    elif damage == "raw_rows":
        raw_b = SimpleNamespace(transitions=raw_b.transitions[:-1])
    elif damage == "projection":
        record["input_projection"] = dict(changed=True)
    write_new(tmp_path / "case.json", record)
    monkeypatch.setattr(case, "ACCEPTED_STAGE", tmp_path)
    calls, sentinel = [], object()

    def read(path, seed, **options):
        calls.append(("read", seed, options))
        return raw_a if seed == 17 else raw_b

    def project(first, second):
        assert first is raw_a and second is raw_b
        calls.append(("project",))
        return data["warm"], data["new"], data["record"]["input_projection"]

    def verify(current, baseline, warm, new):
        calls.append(("depth",))
        assert warm is data["warm"] and new is data["new"]
        assert case.depth.exact_json(current, record["sequence"])
        assert case.depth.exact_json(baseline, record["baseline"])
        if damage == "depth":
            raise AssertionError("prior sequence refused")

    def restore(current, path, spec, audit):
        if damage not in (None, "readout"):
            pytest.fail("invalid source reached model restoration")
        calls.append(("restore", path.name, spec.observation_dim))
        assert path == tmp_path / "final-state.npz"
        return sentinel

    def readout(owner, warm, new, current, audit):
        assert owner is sentinel and warm is data["warm"] and new is data["new"]
        calls.append(("readout",))
        if damage == "readout":
            raise AssertionError("saved final residual refused")

    monkeypatch.setattr(case, "_read_teacher", read)
    monkeypatch.setattr(case, "_project_public_inputs", project)
    monkeypatch.setattr(case.depth, "verify_pair", verify)
    monkeypatch.setattr(case, "restore_candidate", restore)
    monkeypatch.setattr(case.prior, "verify_readouts", readout)
    audit = dict(readouts=[])
    if damage is None:
        owner, warm, new = case.load_candidate(audit)
        assert owner is sentinel and warm is raw_a and new is raw_b
        assert calls == [
            ("read", 17, {"visual_observation": True}),
            ("read", 19, {"visual_observation": True}),
            ("project",),
            ("depth",),
            ("restore", "final-state.npz", 21),
            ("readout",),
        ]
        assert (
            audit["source_python"] == "frozen-source-python"
            and audit["source_numpy"] == "frozen-source-numpy"
        )
    else:
        with pytest.raises(AssertionError):
            case.load_candidate(audit)
        assert not any(row[0] == "restore" for row in calls) or damage == "readout"
        assert not any(row[0] == "readout" for row in calls) or damage == "readout"


@pytest.fixture
def scenario(data, archive, tmp_path, monkeypatch):
    archive_path, record = archive
    stage, baseline = tmp_path / "current", tmp_path / "p8-native"
    stage.mkdir()
    baseline.mkdir()
    monkeypatch.setattr(case, "stage", stage)
    monkeypatch.setattr(case, "BASELINE_STAGE", baseline)
    for name, seed in (("a-rollout.json", 17), ("b-rollout.json", 19)):
        write_new(
            baseline / name, raw_rollout(data, seed, success=True, updates=24, training_rows=704)
        )
    observed = dict(
        calls=[], success=False, callback=None, failed_return=False, different_return=False
    )

    def load(audit):
        owner = case.restore_candidate(record, archive_path, data["warm"].spec, audit)
        case.prior.verify_readouts(owner, data["warm"], data["new"], record["sequence"], audit)
        return owner, data["episodes"]["warm"], data["episodes"]["new"]

    def frozen(path, *, learner, train_seed, visual_observation, observation_projection):
        assert visual_observation is True and observation_projection == public_vector_projection()
        observed["calls"].append((path.name, train_seed, learner))
        raw = raw_rollout(data, train_seed, success=observed["success"])
        for row in raw["episode"]["transitions"]:
            prediction = learner.predict(np.asarray([row["observation"][:21]], dtype=np.float64))
            row["action"] = prediction[0].astype(np.float32).tolist()
            assert (
                row["action"]
                == np.asarray([math.tanh(bias) for bias in BIAS], dtype=np.float32).tolist()
            )
        if observed["callback"] is not None:
            observed["callback"](raw, learner)
        write_new(path, raw)
        if observed["failed_return"]:
            raise RuntimeError("paid frozen rollout failed after controls")
        returned = deepcopy(raw)
        if observed["different_return"]:
            returned["elapsed_seconds"] += 0.01
        return case.smoke.FrozenRolloutResult(returned, None)

    monkeypatch.setattr(case, "load_candidate", load)
    monkeypatch.setattr(case.smoke, "run_frozen_rollout", frozen)
    monkeypatch.setattr(NumpyContinuousBCLearner, "train_batch", forbidden)
    return dict(stage=stage, baseline=baseline, observed=observed)


@pytest.mark.parametrize("success,old_success", [(False, True), (True, False)])
def test_should_use_restored_actual_predictor_in17_then19_controls_and_keep_both_delta_directions(
    data, scenario, success, old_success
):
    scenario["observed"]["success"] = success
    for name, seed in (("a-rollout.json", 17), ("b-rollout.json", 19)):
        path = scenario["baseline"] / name
        path.write_text(
            json.dumps(raw_rollout(data, seed, success=old_success, updates=24, training_rows=704)),
            encoding="utf8",
        )
    audit = dict(complete=False, readouts=[], rollouts=[])
    case.run_case(audit)
    assert audit["complete"] is audit["final_full_state_exact"] is True
    assert [(name, seed) for name, seed, _ in scenario["observed"]["calls"]] == [
        ("a-rollout.json", 17),
        ("b-rollout.json", 19),
    ]
    assert scenario["observed"]["calls"][0][2] is scenario["observed"]["calls"][1][2]
    assert_same_state(data["state"], scenario["observed"]["calls"][0][2].snapshot_state())
    assert audit["reference_rows"] == 5 and audit["reference_observation_values"] == 7890
    assert audit["reference_action_values"] == 20
    for row in audit["reference_work"]:
        assert row["qualified"] is True and row["prediction_bytes"] == row["steps"] * 32
    for row in audit["rollouts"]:
        assert row["qualified"] is row["returned"] is row["state_exact"] is True
        assert row["prediction_bytes"] == row["steps"] * 32
        assert row["forward_dense_macs"] == row["steps"] * 1600
        assert row["visual_work"]["returned_frame_bytes"] == (row["steps"] + 1) * 49152
    assert [row["seed"] for row in audit["primary"]] == [17, 19]
    for row in audit["primary"]:
        assert (
            row["sparse_return_delta"]
            == row["any_success_delta"]
            == row["final_success_delta"]
            == float(success) - float(old_success)
        )
    retained = json.loads((scenario["stage"] / "a-rollout.json").read_text())
    assert retained["episode"]["spec"]["observation_dim"] == 789
    assert retained["source_updates"] == 96 and retained["source_training_rows"] == 224


@pytest.mark.parametrize("damage", ["updates", "rows", "sparse_bool", "sparse_nan"])
def test_should_refuse_invalid_old_native_reference_before_any_new_controls(data, scenario, damage):
    path = scenario["baseline"] / "a-rollout.json"
    raw = json.loads(path.read_text())
    if damage == "updates":
        raw["source_updates"] = 96
    elif damage == "rows":
        raw["source_training_rows"] = 224
    elif damage == "sparse_bool":
        raw["sparse_return"] = True
    else:
        raw["sparse_return"] = float("nan")
    path.write_text(json.dumps(raw), encoding="utf8")
    audit = dict(complete=False, readouts=[], rollouts=[])
    with pytest.raises((AssertionError, ValueError)):
        case.run_case(audit)
    assert scenario["observed"]["calls"] == [] and audit["rollouts"] == []
    assert not (scenario["stage"] / "a-rollout.json").exists()


@pytest.mark.parametrize(
    "damage",
    ["policy", "projection", "sparse_bool", "momentum", "different_return", "failed_return"],
)
def test_should_keep_first_paid_trace_and_refuse_more_controls_after_failure(
    data, scenario, damage
):
    def callback(raw, owner):
        if damage == "policy":
            raw["learner_policy"]["momentum"] = 0.8
        elif damage == "projection":
            raw["projection_work"]["projected_values"] -= 1
        elif damage == "sparse_bool":
            raw["sparse_return"] = False
        elif damage == "momentum":
            state = owner.snapshot_state()
            state.velocity[3][0] += 0.5
            owner.restore_state(state)

    scenario["observed"]["callback"] = callback
    scenario["observed"]["different_return"] = damage == "different_return"
    scenario["observed"]["failed_return"] = damage == "failed_return"
    audit = dict(complete=False, readouts=[], rollouts=[])
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        case.run_case(audit)
    assert len(scenario["observed"]["calls"]) == 1
    assert (scenario["stage"] / "a-rollout.json").exists()
    assert not (scenario["stage"] / "b-rollout.json").exists()
    row = audit["rollouts"][0]
    assert row["attempted"] is True and row["qualified"] is False
    assert row["returned"] is (damage != "failed_return")
    retained = json.loads((scenario["stage"] / "a-rollout.json").read_text())
    assert retained["work"]["completed_steps"] == retained["prediction_returns"] == 2
    if damage == "momentum":
        assert row["state_exact"] is False
    else:
        assert_same_state(data["state"], scenario["observed"]["calls"][0][2].snapshot_state())


def test_should_reread_preceding_trace_after_owner_getter_before_admitting_second_controls(
    scenario, monkeypatch
):
    audit = dict(complete=False, readouts=[], rollouts=[])
    original = case.prior.same_state
    replaced = []

    def guard(expected, owner):
        original(expected, owner)
        if audit["rollouts"] and audit["rollouts"][0]["qualified"] and not replaced:
            path = scenario["stage"] / "a-rollout.json"
            raw = json.loads(path.read_text())
            raw["elapsed_seconds"] += 0.5
            path.write_text(json.dumps(raw), encoding="utf8")
            replaced.append(path)

    monkeypatch.setattr(case.prior, "same_state", guard)
    with pytest.raises(AssertionError, match="preceding trace changed"):
        case.run_case(audit)
    assert len(replaced) == 1 and len(scenario["observed"]["calls"]) == 1
    assert audit["rollouts"][0]["qualified"] is True
    assert not (scenario["stage"] / "b-rollout.json").exists()


@pytest.mark.parametrize(
    "name", ["policy-rollout.json", "rollout-audit.json", "a-rollout.json", "b-rollout.json"]
)
def test_should_keep_existing_evidence_and_refuse_source_or_native_work(
    tmp_path, monkeypatch, name
):
    monkeypatch.setattr(case, "stage", tmp_path)
    monkeypatch.setattr(sys, "argv", ["case", "--output", str(tmp_path / "policy-rollout.json")])
    path = tmp_path / name
    path.write_bytes(b"original evidence")
    monkeypatch.setattr(case, "run_case", forbidden)
    with pytest.raises(FileExistsError):
        case.main()
    assert path.read_bytes() == b"original evidence"


def test_should_refuse_output_alias_before_restoring_or_creating_native_environment(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(case, "stage", tmp_path)
    monkeypatch.setattr(sys, "argv", ["case", "--output", str(tmp_path / "a-rollout.json")])
    monkeypatch.setattr(case, "run_case", forbidden)
    with pytest.raises(AssertionError):
        case.main()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", ["runtime", "summary"])
def test_should_preserve_paid_source_and_control_evidence_after_runtime_or_summary_write_failure(
    tmp_path, monkeypatch, failure
):
    monkeypatch.setattr(case, "stage", tmp_path)
    monkeypatch.setattr(sys, "argv", ["case", "--output", str(tmp_path / "policy-rollout.json")])

    def run(audit):
        audit.update(
            complete=True,
            source_updates=96,
            source_training_rows=2816,
            readouts=[dict(returned=True, prediction_rows=29)],
            rollouts=[dict(seed=17, returned=True, steps=29)],
        )
        if failure == "runtime":
            raise RuntimeError("late failure after paid controls")

    monkeypatch.setattr(case, "run_case", run)
    original = case.prior.write_json

    def write(path, value):
        if failure == "summary" and path.name == "policy-rollout.json":
            raise OSError("summary disk failure")
        original(path, value)

    monkeypatch.setattr(case.prior, "write_json", write)
    assert case.main() == 1
    audit = json.loads((tmp_path / "rollout-audit.json").read_text())
    assert audit["complete"] is False
    assert audit["source_updates"] == 96 and audit["source_training_rows"] == 2816
    assert audit["readouts"] == [dict(returned=True, prediction_rows=29)]
    assert audit["rollouts"] == [dict(seed=17, returned=True, steps=29)]
    assert audit["optimizer_updates"] == audit["fit_calls"] == 0
    assert ("summary_write_error" if failure == "summary" else "error") in audit
