"""Synthetic saved MLP restoration/readout/native gates, without fit or SGD."""

from copy import deepcopy
from dataclasses import asdict, replace
import importlib
import json
import math
import sys
import zipfile

import numpy as np
import pytest

from src.adapters.numpy_continuous_bc import ContinuousBCPolicy, NumpyContinuousBCLearner
from src.core.continuous_bc import OBJECTIVE
from src.core.observation_standardization import ObservationStandardization
from src.core.pick_cube_public_vectors import public_vector_projection
from test_maniskill_bc_smoke import assert_same_state
from test_public_vector_native_case import episode_from_teacher
from test_visual_native_context_case import teacher_receipt, rollout_receipt, write_new


case = importlib.import_module("artifacts.runs.public-vector-mlp-native-20261019.native_case")
BIAS = (0.2, -0.3, 0.4, -0.5)


def residual_reference(episode):
    """Independent scalar arithmetic for the manually constant model's42 quantities."""
    axes, squares = [], []
    for index, (unit, bias) in enumerate(zip(episode.spec.action_units, BIAS, strict=True)):
        errors = [math.tanh(bias) - row.action[index] for row in episode.transitions]
        mse = math.fsum(value * value for value in errors) / len(errors)
        squares.extend(value * value for value in errors)
        axes.append(
            dict(
                index=index,
                unit=unit,
                active=True,
                normalized_bias=math.fsum(errors) / len(errors),
                normalized_mae=math.fsum(abs(value) for value in errors) / len(errors),
                normalized_rmse=math.sqrt(mse),
                normalized_half_mse=0.5 * mse,
                maximum_absolute_normalized_error=max(abs(value) for value in errors),
            )
        )
    rows = len(episode.transitions)
    overall = dict(
        rows=rows,
        active_axes=4,
        normalized_half_mse=0.5 * math.fsum(squares) / len(squares),
        axes=axes,
    )
    return dict(
        residuals=dict(
            overall=overall,
            stages=[dict(label="all", row_indices=list(range(rows)), summary=deepcopy(overall))],
        ),
        prediction_calls=1,
        prediction_rows=rows,
        prediction_values=4 * rows,
        prediction_bytes=32 * rows,
    )


@pytest.fixture(scope="module")
def saved_data():
    """59unique rows, ONE constructor+restore; no diagnostic forward during setup."""
    raw = dict(warm=teacher_receipt(17, 29), new=teacher_receipt(19, 30))
    episodes = {name: episode_from_teacher(value) for name, value in raw.items()}
    a, b, projection = case._project_public_inputs(episodes["warm"], episodes["new"])
    stats = ObservationStandardization((14.0,) + (0.0,) * 20, (math.sqrt(70.0),) + (1.0,) * 20, 29)
    owner = NumpyContinuousBCLearner(ContinuousBCPolicy(a.spec, observation_standardization=stats))
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
            completed_updates=24,
            training_examples=704,
        )
    )
    state = owner.snapshot_state()
    contexts = {
        name: dict(episode_id=episode.episode_id, residual=residual_reference(episode))
        for name, episode in (("warm", a), ("new", b))
    }
    checkpoints = [
        dict(
            updates=updates,
            training_examples=rows,
            state_exact=True,
            owners_exact=True,
            contexts=deepcopy(contexts),
        )
        for updates, rows in ((0, 0), (8, 232), (16, 472), (24, 704))
    ]
    applied, events = [], []
    for index in range(24):
        episode = b if 8 <= index < 16 else a
        receipt = dict(
            sample_id=f"mlp-batch-{index}",
            episode_id=episode.episode_id,
            event_id=f"mlp-label-{index}",
            actor_version="actor-0",
            learner_version="candidate-0",
            observed_at=2 * index,
            arrived_at=2 * index + 1,
            applied_at=2 * index + 1,
            update_number=index + 1,
            diagnostic=dict(definition=OBJECTIVE, value=0.25),
        )
        applied.append(receipt)
        events.append(
            dict(deferred_exact=True, success_validated=True, applied=[deepcopy(receipt)])
        )
    work = dict(
        fit_calls=1,
        fit_rows=29,
        fit_feature_values=609,
        optimizer_updates=24,
        training_rows=704,
        unique_source_rows=59,
        repeated_training_rows=645,
        replay_budget_rows=0,
        conditioning_calls=8,
        conditioning_rows=236,
        residual_calls=8,
        residual_rows=236,
        actor_prediction_calls=8,
        actor_prediction_rows=236,
        normalization_calls=48,
        normalization_rows=1412,
        normalization_values=29652,
        forward_passes=56,
        forward_rows=1648,
        backward_passes=32,
        backward_rows=940,
        gradient_values=53376,
        gradient_numeric_bytes=427008,
        dense_multiply_accumulates=4381440,
        parameter_values_per_owner=1668,
        model_numeric_bytes_per_owner=26688,
        statistics_numeric_bytes_per_owner=336,
        named_model_owners=4,
        inbox_feature_target_numeric_bytes=140800,
        residual_prediction_bytes=7552,
        actor_prediction_bytes=7552,
    )
    sequence = dict(
        complete=True,
        policy=asdict(state.policy),
        settings=dict(
            phase_updates=8,
            phases=["warm", "new", "warm_return"],
            max_training_updates=24,
            promotion_calls=0,
            simulator_calls=0,
            probe_reads=0,
            final_reads=0,
            guard_reads=0,
        ),
        progress=dict(
            update_attempts=24, updates_completed=24, replay_examples_completed=0, next_tick=48
        ),
        source_episodes={
            name: dict(episode_id=value.episode_id, context=value.context, role="train")
            for name, value in (("warm", a), ("new", b))
        },
        expected_standardization_checked=False,
        checkpoints=checkpoints,
        applied_updates=applied,
        events=events,
        work=work,
    )
    accepted = dict(
        complete=True,
        python=sys.version,
        numpy=np.__version__,
        input_projection=projection,
        sequence=sequence,
        state_archive=dict(
            updates=24,
            training_examples=704,
            array_names=[
                f"{group}_{index}" for group in ("parameters", "velocity") for index in range(4)
            ],
            numeric_bytes=26688,
            save_attempts=1,
            save_returns=1,
        ),
    )
    arrays = {
        f"{group}_{index}": value
        for group in ("parameters", "velocity")
        for index, value in enumerate(getattr(state, group))
    }
    return dict(
        raw=raw,
        episodes=episodes,
        projected=dict(warm=a, new=b),
        accepted=accepted,
        state=state,
        arrays=arrays,
    )


@pytest.fixture
def archived(saved_data, tmp_path):
    path = tmp_path / "final-state.npz"
    np.savez(path, **saved_data["arrays"])
    record = deepcopy(saved_data["accepted"])
    record["state_archive"]["file_bytes"] = path.stat().st_size
    return path, record


def forbidden(*_args, **_kwargs):
    pytest.fail("work was admitted beyond a failed prerequisite")


def test_should_restore_all_parameters_momentum_counters_and_owned_policy_without_learning(
    saved_data, archived, monkeypatch
):
    path, record = archived
    original = deepcopy(record)
    monkeypatch.setattr(NumpyContinuousBCLearner, "train_batch", forbidden)
    audit = {}
    model = case.restore_candidate(record, path, saved_data["projected"]["warm"].spec, audit)
    assert_same_state(saved_data["state"], model.snapshot_state())
    assert record == original
    assert audit["constructor_attempts"] == audit["constructor_returns"] == 1
    assert audit["restore_attempts"] == audit["restore_returns"] == 1
    assert audit["restore_roundtrip_exact"] is True
    assert audit["archive_array_values"] == 3336 and audit["archive_numeric_bytes"] == 26688
    exported = model.snapshot_state()
    exported.parameters[3][0] = 100.0
    exported.velocity[0][0, 0] = 100.0
    assert_same_state(saved_data["state"], model.snapshot_state())
    assert model.snapshot_state().policy.observation_standardization.means == (14.0,) + (0.0,) * 20


@pytest.mark.parametrize(
    "damage",
    [
        "complete",
        "sequence",
        "hidden",
        "role",
        "axis",
        "unknown",
        "statistics",
        "updates",
        "rows",
        "numeric",
        "save",
        "names",
    ],
)
def test_should_refuse_invalid_saved_contract_before_archive_io(
    saved_data, archived, monkeypatch, damage
):
    path, record = archived
    policy, archive = record["sequence"]["policy"], record["state_archive"]
    if damage == "complete":
        record["complete"] = 1
    elif damage == "sequence":
        record["sequence"]["complete"] = 1
    elif damage == "hidden":
        policy["hidden_dim"] = 32
    elif damage == "role":
        policy["training_role"] = "probe_train"
    elif damage == "axis":
        policy["binary_gripper_axis"] = 3
    elif damage == "unknown":
        policy["extra"] = None
    elif damage == "statistics":
        policy["observation_standardization"]["fitted_rows"] = 30
    elif damage == "updates":
        archive["updates"] = 24.0
    elif damage == "rows":
        archive["training_examples"] = True
    elif damage == "numeric":
        archive["numeric_bytes"] = 26689
    elif damage == "save":
        archive["save_returns"] = True
    else:
        archive["array_names"] = list(reversed(archive["array_names"]))
    monkeypatch.setattr(case, "read_arrays", forbidden)
    with pytest.raises((AssertionError, ValueError, TypeError)):
        case.restore_candidate(record, path, saved_data["projected"]["warm"].spec, {})


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "extra",
        "dtype",
        "shape",
        "nan",
        "fortran",
        "body",
        "version",
        "duplicate",
        "large_member",
        "large_file",
    ],
)
def test_should_bound_and_validate_the_complete_archive_before_allocation(
    saved_data, tmp_path, monkeypatch, damage
):
    path = tmp_path / "bad.npz"
    arrays = {key: value.copy() for key, value in saved_data["arrays"].items()}
    if damage == "missing":
        arrays.pop("velocity_3")
    elif damage == "extra":
        arrays["extra"] = np.zeros(1)
    elif damage == "dtype":
        arrays["velocity_2"] = arrays["velocity_2"].astype(np.float32)
    elif damage == "shape":
        arrays["parameters_0"] = np.zeros((20, 64))
    elif damage == "nan":
        arrays["parameters_3"][0] = np.nan
    elif damage == "fortran":
        arrays["parameters_0"] = np.asfortranarray(arrays["parameters_0"])
    elif damage == "large_member":
        arrays["parameters_0"] = np.zeros((10000, 64))
    np.savez_compressed(path, **arrays)
    if damage in ("body", "version", "duplicate"):
        with zipfile.ZipFile(path) as archive:
            members = {name: archive.read(name) for name in archive.namelist()}
        if damage == "body":
            members["velocity_3.npy"] += b"unexpected body"
        elif damage == "version":
            members["velocity_3.npy"] = b"\x93NUMPY\x02\x00" + members["velocity_3.npy"][8:]
        with zipfile.ZipFile(path, "w") as archive:
            for name, payload in members.items():
                archive.writestr(name, payload)
            if damage == "duplicate":
                with pytest.warns(UserWarning, match="Duplicate name"):
                    archive.writestr("velocity_3.npy", members["velocity_3.npy"])
    elif damage == "large_file":
        path.write_bytes(b"x" * (128 * 1024 + 1))
    if damage != "nan":
        monkeypatch.setattr(case.np, "load", forbidden)
    audit = {}
    with pytest.raises((AssertionError, ValueError)):
        case.read_arrays(path, audit)
    assert audit.get("archive_read_returns", 0) == 0


@pytest.mark.parametrize("file_bytes", [True, 0, 123.0])
def test_should_refuse_wrong_file_accounting_before_constructing_a_model(
    saved_data, archived, monkeypatch, file_bytes
):
    path, record = archived
    record["state_archive"]["file_bytes"] = file_bytes
    monkeypatch.setattr(case, "NumpyContinuousBCLearner", forbidden)
    with pytest.raises(AssertionError):
        case.restore_candidate(record, path, saved_data["projected"]["warm"].spec, {})


def test_should_compare_all42_independent_scalar_readouts_without_changing_state(
    saved_data, archived
):
    path, record = archived
    owner = case.restore_candidate(record, path, saved_data["projected"]["warm"].spec, {})
    before, audit = owner.snapshot_state(), dict(readouts=[])
    case.verify_readouts(
        owner,
        saved_data["projected"]["warm"],
        saved_data["projected"]["new"],
        record["sequence"],
        audit,
    )
    assert_same_state(before, owner.snapshot_state())
    assert audit["readout_numeric_comparisons"] == 42
    assert audit["residual_calls"] == 2 and audit["residual_rows"] == 59
    assert audit["residual_prediction_values"] == 236 and audit["residual_prediction_bytes"] == 1888
    assert (
        audit["residual_normalization_values"] == 1239
        and audit["residual_forward_dense_macs"] == 94400
    )
    assert [row["passed"] for row in audit["readouts"]] == [True, True]


@pytest.mark.parametrize(
    "damage", ["loss", "axis", "counter", "rows", "active", "stage", "indices", "unit", "second"]
)
def test_should_retain_returned_readout_and_stop_on_changed_numeric_or_metadata(
    saved_data, archived, damage
):
    path, record = archived
    owner = case.restore_candidate(record, path, saved_data["projected"]["warm"].spec, {})
    name = "new" if damage == "second" else "warm"
    diagnostic = record["sequence"]["checkpoints"][-1]["contexts"][name]["residual"]
    overall = diagnostic["residuals"]["overall"]
    if damage in ("loss", "second"):
        overall["normalized_half_mse"] += 1e-5
        diagnostic["residuals"]["stages"][0]["summary"] = deepcopy(overall)
    elif damage == "axis":
        overall["axes"][2]["normalized_mae"] += 1e-5
        diagnostic["residuals"]["stages"][0]["summary"] = deepcopy(overall)
    elif damage == "counter":
        diagnostic["prediction_calls"] = True
    elif damage == "rows":
        overall["rows"] = float(overall["rows"])
    elif damage == "active":
        overall["active_axes"] = 3
    elif damage == "stage":
        diagnostic["residuals"]["stages"][0]["label"] = "selected"
    elif damage == "indices":
        diagnostic["residuals"]["stages"][0]["row_indices"][0] = 1
    else:
        overall["axes"][0]["unit"] = "wrong"
        diagnostic["residuals"]["stages"][0]["summary"] = deepcopy(overall)
    audit = dict(readouts=[])
    with pytest.raises(AssertionError):
        case.verify_readouts(
            owner,
            saved_data["projected"]["warm"],
            saved_data["projected"]["new"],
            record["sequence"],
            audit,
        )
    assert len(audit["readouts"]) == (2 if damage == "second" else 1)
    assert audit["readouts"][-1]["returned"] is True and audit["readouts"][-1]["passed"] is False
    assert audit["readouts"][-1]["diagnostic"]["prediction_rows"] == (
        30 if damage == "second" else 29
    )
    assert_same_state(saved_data["state"], owner.snapshot_state())


def test_should_retain_readout_work_when_a_callback_changes_only_momentum(
    saved_data, archived, monkeypatch
):
    path, record = archived
    owner = case.restore_candidate(record, path, saved_data["projected"]["warm"].spec, {})
    original = owner.diagnose_training_episode

    def diagnose(*args, **kwargs):
        result = original(*args, **kwargs)
        changed = owner.snapshot_state()
        changed.velocity[2][0, 0] += 1.0
        owner.restore_state(changed)
        return result

    monkeypatch.setattr(owner, "diagnose_training_episode", diagnose)
    audit = dict(readouts=[])
    with pytest.raises(AssertionError, match="state changed"):
        case.verify_readouts(
            owner,
            saved_data["projected"]["warm"],
            saved_data["projected"]["new"],
            record["sequence"],
            audit,
        )
    assert len(audit["readouts"]) == 1
    assert audit["readouts"][0]["returned"] is True and audit["readouts"][0]["passed"] is False
    assert audit["readouts"][0]["diagnostic"]["prediction_bytes"] == 29 * 32


@pytest.mark.parametrize("observed,expected", [(0.0, 1e-10), (1.0, 1.0 + 5e-11), (0.0, -1e-10)])
def test_should_use_frozen_absolute_tolerance(observed, expected):
    case.near(observed, expected)


@pytest.mark.parametrize(
    "observed,expected",
    [
        (0.0, 1.00001e-10),
        (True, 1.0),
        (1.0, True),
        (float("nan"), 0.0),
        (0.0, float("inf")),
        (np.float64(1.0), 1.0),
    ],
)
def test_should_refuse_non_native_nonfinite_or_outside_tolerance(observed, expected):
    with pytest.raises(AssertionError):
        case.near(observed, expected)


def model_rollout(raw_teacher, state, *, success=False):
    """A synthetic boundary receipt; public native observation packets stay789D."""
    raw = rollout_receipt(raw_teacher, success=success)
    rows = len(raw["episode"]["transitions"])
    raw.update(
        policy="frozen_continuous_bc",
        state_guard_scope="private_fork_and_original_learner",
        learner_policy=asdict(state.policy),
        observation_projection=asdict(public_vector_projection()),
        projection_exact=True,
        frozen_state_exact=True,
        original_owner_exact=True,
        source_updates=24,
        source_training_rows=704,
        optimizer_updates=0,
        prediction_attempts=rows,
        prediction_returns=rows,
        validated_prediction_rows=rows,
        validated_prediction_bytes=32 * rows,
        projection_work=dict(
            observation_attempts=rows,
            observation_returns=rows,
            source_values_upper_bound=789 * rows,
            source_values_validated=789 * rows,
            projected_values=21 * rows,
            projected_numeric_bytes=168 * rows,
        ),
    )
    return raw


@pytest.fixture
def scenario(saved_data, archived, tmp_path, monkeypatch):
    archive, record = archived
    output, accepted, baseline = (tmp_path / name for name in ("current", "accepted", "baseline"))
    for directory in (output, accepted, baseline):
        directory.mkdir()
    archive.rename(accepted / "final-state.npz")
    write_new(accepted / "case.json", record)
    for name, context in (("a-rollout.json", "warm"), ("b-rollout.json", "new")):
        write_new(baseline / name, rollout_receipt(saved_data["raw"][context], success=True))
    monkeypatch.setattr(case, "stage", output)
    monkeypatch.setattr(case, "ACCEPTED_STAGE", accepted)
    monkeypatch.setattr(case, "BASELINE_STAGE", baseline)
    warm_path, new_path = tmp_path / "warm.json", tmp_path / "new.json"
    monkeypatch.setattr(case, "WARM_PATH", warm_path)
    monkeypatch.setattr(case, "NEW_PATH", new_path)

    def read_teacher(path, seed, *, visual_observation=False):
        assert visual_observation is True
        assert path == (warm_path if seed == 17 else new_path)
        return saved_data["episodes"]["warm" if seed == 17 else "new"]

    monkeypatch.setattr(case, "_read_teacher", read_teacher)
    monkeypatch.setattr(NumpyContinuousBCLearner, "train_batch", forbidden)
    observed = dict(calls=[], success=False, callback=None, error=False)

    def rollout(path, *, learner, train_seed, visual_observation, observation_projection):
        assert visual_observation is True
        assert observation_projection == public_vector_projection()
        observed["calls"].append((path.name, train_seed, learner))
        context = "warm" if train_seed == 17 else "new"
        raw = model_rollout(
            saved_data["raw"][context], learner.snapshot_state(), success=observed["success"]
        )
        # Invoke the actual restored public predictor for every controlled row.
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
        if observed["error"]:
            raise RuntimeError("boundary failed after paid controls")
        return case.smoke.FrozenRolloutResult(raw, None)

    monkeypatch.setattr(case.smoke, "run_frozen_rollout", rollout)
    return dict(
        output=output, accepted=accepted, baseline=baseline, observed=observed, record=record
    )


@pytest.mark.parametrize("success", [False, True])
def test_should_run_saved_controls17_then19_and_accept_negative_results(
    saved_data, scenario, success
):
    scenario["observed"]["success"] = success
    audit = dict(complete=False, readouts=[], rollouts=[])
    case.run_case(audit)
    assert audit["complete"] is audit["final_full_state_exact"] is True
    assert [(name, seed) for name, seed, _ in scenario["observed"]["calls"]] == [
        ("a-rollout.json", 17),
        ("b-rollout.json", 19),
    ]
    assert scenario["observed"]["calls"][0][2] is scenario["observed"]["calls"][1][2]
    assert_same_state(saved_data["state"], scenario["observed"]["calls"][0][2].snapshot_state())
    assert audit["reference_rows"] == 59 and audit["reference_action_values"] == 236
    assert audit["reference_observation_values"] == 2 * 59 * 789
    for row, count in zip(audit["rollouts"], (29, 30), strict=True):
        assert row["attempted"] is row["returned"] is row["state_exact"] is row["qualified"] is True
        assert row["prediction_bytes"] == count * 32
        assert row["forward_dense_macs"] == count * 1600
        assert row["normalization_values"] == count * 21
        assert row["projection_work"]["projected_numeric_bytes"] == count * 168
        assert row["visual_work"]["returned_frame_bytes"] == (count + 1) * 49152
        assert row["action_bound_violations"] == 0
    for row in audit["primary"]:
        assert (
            row["sparse_return_delta"]
            == row["any_success_delta"]
            == row["final_success_delta"]
            == (0.0 if success else -1.0)
        )
    retained = json.loads((scenario["output"] / "a-rollout.json").read_text())
    assert retained["episode"]["spec"]["observation_dim"] == 789
    assert len(retained["episode"]["transitions"][0]["observation"]) == 789


@pytest.mark.parametrize("damage", [None, "source", "clock", "work", "projection", "receipt"])
def test_should_admit_valid_source_and_refuse_changed_previous_metadata_before_constructor(
    saved_data, scenario, monkeypatch, damage
):
    record = deepcopy(scenario["record"])
    if damage == "source":
        record["sequence"]["source_episodes"]["new"]["role"] = "probe_train"
    elif damage == "clock":
        record["sequence"]["progress"]["next_tick"] = 47
    elif damage == "work":
        record["sequence"]["work"]["training_rows"] = 704.0
    elif damage == "projection":
        record["input_projection"]["projected_values"] += 1
    elif damage == "receipt":
        record["sequence"]["applied_updates"][0]["arrived_at"] = 0
    path = scenario["accepted"] / "case.json"
    path.write_text(json.dumps(record), encoding="utf8")
    if damage is not None:
        monkeypatch.setattr(case, "NumpyContinuousBCLearner", forbidden)
        with pytest.raises(AssertionError):
            case.load_candidate(dict(readouts=[]))
    else:
        audit = dict(readouts=[])
        owner, warm, new = case.load_candidate(audit)
        assert_same_state(saved_data["state"], owner.snapshot_state())
        assert (len(warm.transitions), len(new.transitions)) == (29, 30)
        assert audit["readout_numeric_comparisons"] == 42
    assert scenario["observed"]["calls"] == []


@pytest.mark.parametrize(
    "damage", ["guard", "role", "policy", "counter", "projection", "visual", "actions"]
)
def test_should_refuse_completed_but_invalid_first_receipt_before_next_controls(
    saved_data, scenario, damage
):
    def corrupt(raw, _owner):
        if damage == "guard":
            raw["original_owner_exact"] = False
        elif damage == "role":
            raw["episode"]["role"] = "probe_train"
        elif damage == "policy":
            raw["learner_policy"]["observation_standardization"]["scales"] = [1.0] * 21
        elif damage == "counter":
            raw["validated_prediction_bytes"] = float(raw["validated_prediction_bytes"])
        elif damage == "projection":
            raw["projection_work"]["projected_values"] -= 1
        elif damage == "visual":
            raw["visual_work"]["returned_feature_values"] -= 1
        else:
            raw["episode"]["transitions"][0]["action"][0] = 2.0

    scenario["observed"]["callback"] = corrupt
    audit = dict(complete=False, readouts=[], rollouts=[])
    with pytest.raises((AssertionError, ValueError)):
        case.run_case(audit)
    assert len(scenario["observed"]["calls"]) == 1
    assert audit["rollouts"][0]["returned"] is True
    assert audit["rollouts"][0]["qualified"] is False
    assert (scenario["output"] / "a-rollout.json").exists()
    assert not (scenario["output"] / "b-rollout.json").exists()
    assert_same_state(saved_data["state"], scenario["observed"]["calls"][0][2].snapshot_state())


def test_should_retain_paid_trace_when_public_rollout_does_not_return(scenario):
    scenario["observed"]["error"] = True
    audit = dict(complete=False, readouts=[], rollouts=[])
    with pytest.raises(RuntimeError, match="paid controls"):
        case.run_case(audit)
    assert len(scenario["observed"]["calls"]) == 1
    row = audit["rollouts"][0]
    assert row["attempted"] is True and row["returned"] is row["qualified"] is False
    raw = json.loads((scenario["output"] / "a-rollout.json").read_text())
    assert raw["work"]["completed_steps"] == 29 and raw["prediction_returns"] == 29
    assert not (scenario["output"] / "b-rollout.json").exists()


def test_should_refuse_momentum_mutation_after_controls_before_next_rollout(scenario):
    def corrupt(_raw, owner):
        state = owner.snapshot_state()
        state.velocity[3][0] += 1.0
        owner.restore_state(state)

    scenario["observed"]["callback"] = corrupt
    audit = dict(complete=False, readouts=[], rollouts=[])
    with pytest.raises(AssertionError, match="state changed"):
        case.run_case(audit)
    assert len(scenario["observed"]["calls"]) == 1
    assert audit["rollouts"][0]["returned"] is True
    assert audit["rollouts"][0]["state_exact"] is False
    assert (scenario["output"] / "a-rollout.json").exists()


@pytest.mark.parametrize("damage", ["returned", "preceding"])
def test_should_refuse_replaced_or_different_retained_evidence_before_second_rollout(
    scenario, monkeypatch, damage
):
    original_read = case.read_json
    a_reads = []

    def read(path, cap):
        raw = original_read(path, cap)
        if path == scenario["output"] / "a-rollout.json":
            a_reads.append(path)
            if damage == "returned" or len(a_reads) == 2:
                raw["elapsed_seconds"] += 1.0
        return raw

    monkeypatch.setattr(case, "read_json", read)
    audit = dict(complete=False, readouts=[], rollouts=[])
    with pytest.raises(AssertionError, match="evidence differs|preceding trace changed"):
        case.run_case(audit)
    assert len(scenario["observed"]["calls"]) == 1
    assert audit["rollouts"][0]["returned"] is True
    assert audit["rollouts"][0]["qualified"] is (damage == "preceding")
    assert not (scenario["output"] / "b-rollout.json").exists()


@pytest.mark.parametrize("damage", ["duplicate", "nonfinite", "oversize"])
def test_should_bound_json_and_refuse_ambiguous_or_nonfinite_receipts(tmp_path, damage):
    path = tmp_path / "record.json"
    payload = (
        '{"v":1,"v":2}'
        if damage == "duplicate"
        else '{"value":NaN}'
        if damage == "nonfinite"
        else " " * 33
    )
    path.write_text(payload, encoding="utf8")
    with pytest.raises((AssertionError, ValueError)):
        case.read_json(path, 32)


@pytest.mark.parametrize(
    "name", ["policy-rollout.json", "rollout-audit.json", "a-rollout.json", "b-rollout.json"]
)
def test_should_preserve_existing_output_and_refuse_all_work(tmp_path, monkeypatch, name):
    monkeypatch.setattr(case, "stage", tmp_path)
    monkeypatch.setattr(sys, "argv", ["case", "--output", str(tmp_path / "policy-rollout.json")])
    path = tmp_path / name
    path.write_bytes(b"original evidence")
    monkeypatch.setattr(case, "run_case", forbidden)
    with pytest.raises(FileExistsError):
        case.main()
    assert path.read_bytes() == b"original evidence"


def test_should_refuse_output_alias_before_work(tmp_path, monkeypatch):
    monkeypatch.setattr(case, "stage", tmp_path)
    monkeypatch.setattr(sys, "argv", ["case", "--output", str(tmp_path / "a-rollout.json")])
    monkeypatch.setattr(case, "run_case", forbidden)
    with pytest.raises(AssertionError):
        case.main()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", ["runtime", "summary"])
def test_should_persist_paid_work_and_refuse_complete_after_runtime_or_summary_failure(
    tmp_path, monkeypatch, failure
):
    monkeypatch.setattr(case, "stage", tmp_path)
    monkeypatch.setattr(sys, "argv", ["case", "--output", str(tmp_path / "policy-rollout.json")])

    def run(audit):
        audit.update(
            complete=True,
            readouts=[dict(returned=True, prediction_rows=29)],
            rollouts=[dict(seed=17, returned=True, steps=29)],
        )
        if failure == "runtime":
            raise RuntimeError("late failure after controls")

    monkeypatch.setattr(case, "run_case", run)
    original_write = case.write_json

    def write(path, value):
        if failure == "summary" and path.name == "policy-rollout.json":
            raise OSError("summary disk failure")
        original_write(path, value)

    monkeypatch.setattr(case, "write_json", write)
    assert case.main() == 1
    audit = json.loads((tmp_path / "rollout-audit.json").read_text())
    assert audit["complete"] is False
    assert audit["readouts"] == [dict(returned=True, prediction_rows=29)]
    assert audit["rollouts"] == [dict(seed=17, returned=True, steps=29)]
    assert audit["optimizer_updates"] == audit["fit_calls"] == 0
    assert ("summary_write_error" if failure == "summary" else "error") in audit
