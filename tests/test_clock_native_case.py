"""Synthetic final-state loading and native-caller checks; no simulator or training."""

from dataclasses import asdict
import importlib
import json
import math
import sys

import numpy as np
import pytest

from src.adapters.numpy_continuous_bc import ContinuousBCPolicy, NumpyContinuousBCLearner
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.observation_standardization import ObservationStandardization
from src.infra.maniskill_continuous import ACTION_UNITS

# The saved native harness uses Linux resource accounting. Refuse its import
# during Windows collection; WSL must still execute every case in this module.
if sys.platform == "win32":
    pytest.skip("native harness requires Linux resource accounting", allow_module_level=True)

case = importlib.import_module("artifacts.runs.continuous-bc-clock-20261019.native_case")


def training_episode(role="train"):
    spec = ContinuousActionSpec(42, ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
    return ContinuousEpisode(
        "synthetic-teacher",
        "pick-cube",
        role,
        spec,
        tuple(
            ContinuousTransition(
                step,
                (float(step),) + (0.0,) * 41,
                (0.0,) * 4,
                float(step == 28),
                (float(step + 1),) + (0.0,) * 41,
                step == 28,
                False,
            )
            for step in range(29)
        ),
    )


@pytest.fixture
def synthetic_source(monkeypatch, tmp_path):
    teacher = training_episode()
    report = dict(complete=True, initial_arrays_and_counters_exact=True, methods=[])
    stored = {}
    for method in ("zero", "elapsed"):
        elapsed = method == "elapsed"
        transform = ObservationStandardization(
            (0.0,) * 42 + (0.28 if elapsed else 0.0,),
            (1.0,) * 42 + (math.sqrt(70) / 50 if elapsed else 1.0,),
            29,
        )
        spec = ContinuousActionSpec(43, ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
        policy = ContinuousBCPolicy(spec, observation_standardization=transform)
        shapes = ((43, 64), (64,), (64, 4), (4,))
        arrays = {
            f"{prefix}{index}": np.full(
                shape, (index + 1) * (0.01 if prefix == "p" else -0.01), dtype=np.float64
            )
            for prefix in ("p", "v")
            for index, shape in enumerate(shapes)
        }
        path = tmp_path / f"{method}-256-state.npz"
        np.savez(path, **arrays)
        stored[method] = arrays
        report["methods"].append(
            dict(
                method=method,
                complete=True,
                policy=asdict(policy),
                clock_recipe=dict(horizon=50, elapsed=elapsed),
                checkpoints=[
                    dict(total_updates=updates, full_state_exact=True)
                    for updates in (0, 64, 128, 256)
                ],
                final_updates=256,
                final_training_rows=7424,
                state_file=path.name,
            )
        )
    metadata = tmp_path / "train-comparison.json"
    metadata.write_text(json.dumps(report), encoding="utf8")
    monkeypatch.setattr(case, "source", tmp_path)
    return teacher, report, metadata, stored


def save_metadata(report, path):
    path.write_text(json.dumps(report), encoding="utf8")


@pytest.mark.parametrize("method", ["zero", "elapsed"])
def test_should_restore_explicit_pair_with_full_frozen_policy_arrays_and_counters(
    synthetic_source, monkeypatch, method
):
    teacher, report, metadata, stored = synthetic_source
    before_metadata = metadata.read_bytes()
    state_path = metadata.parent / f"{method}-256-state.npz"
    before_state = state_path.read_bytes()

    def forbidden(*_):
        pytest.fail("restoration attempted fitting or training")

    monkeypatch.setattr(NumpyContinuousBCLearner, "train_batch", forbidden)
    monkeypatch.setattr(NumpyContinuousBCLearner, "train_episode", forbidden)
    model, recipe = case.restore_final(teacher, method)
    snapshot = model.snapshot_state()
    assert (recipe.horizon, recipe.elapsed) == (50, method == "elapsed")
    assert snapshot.completed_updates == 256 and snapshot.training_examples == 7424
    assert snapshot.policy.spec.observation_dim == 43
    assert snapshot.policy.spec.action_units == teacher.spec.action_units
    assert json.loads(json.dumps(asdict(snapshot.policy))) == next(
        item["policy"]
        for item in json.loads(before_metadata)["methods"]
        if item["method"] == method
    )
    for prefix, values in (("p", snapshot.parameters), ("v", snapshot.velocity)):
        for index, actual in enumerate(values):
            np.testing.assert_array_equal(actual, stored[method][f"{prefix}{index}"])
    snapshot.parameters[0].fill(999.0)
    assert model.snapshot_state().parameters[0][0, 0] == 0.01
    assert metadata.read_bytes() == before_metadata and state_path.read_bytes() == before_state
    assert report["methods"][0]["clock_recipe"] == dict(horizon=50, elapsed=False)


@pytest.mark.parametrize("method", [None, True, 0, "", "standardized", "ZERO", "zero "])
def test_should_refuse_invalid_method_before_any_file_access(monkeypatch, tmp_path, method):
    monkeypatch.setattr(case, "source", tmp_path / "does-not-exist")
    with pytest.raises(ValueError, match="method"):
        case.restore_final(training_episode(), method)


def test_should_refuse_string_subclass_method_before_file_access(monkeypatch, tmp_path):
    class ForeignString(str):
        pass

    monkeypatch.setattr(case, "source", tmp_path / "does-not-exist")
    with pytest.raises(ValueError, match="method"):
        case.restore_final(training_episode(), ForeignString("zero"))


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test"])
def test_should_refuse_non_train_source_before_file_access(monkeypatch, tmp_path, role):
    monkeypatch.setattr(case, "source", tmp_path / "does-not-exist")
    with pytest.raises(ValueError, match="TRAIN"):
        case.restore_final(training_episode(role), "zero")


def test_should_revalidate_corrupted_source_before_file_access(monkeypatch, tmp_path):
    teacher = training_episode()
    object.__setattr__(teacher.transitions[0], "next_observation", (999.0,) * 42)
    monkeypatch.setattr(case, "source", tmp_path / "does-not-exist")
    with pytest.raises(ValueError):
        case.restore_final(teacher, "zero")


@pytest.mark.parametrize(
    "bad",
    [
        "experiment_incomplete",
        "initial_mismatch",
        "method_missing",
        "method_duplicate",
        "method_incomplete",
        "recipe_missing",
        "recipe_mode",
        "recipe_horizon",
        "recipe_type",
        "checkpoint_missing",
        "checkpoint_duplicate",
        "checkpoint_order",
        "checkpoint_false",
        "updates",
        "rows",
        "foreign_path",
        "other_method_path",
        "policy_seed",
        "policy_hidden",
        "policy_lr",
        "policy_momentum",
        "policy_role",
        "policy_units",
        "policy_bounds",
        "policy_dimension",
        "fit_rows",
        "fit_dimension",
        "fit_scale",
        "fit_missing",
    ],
)
def test_should_refuse_mismatched_metadata_before_opening_state_archive(
    synthetic_source, monkeypatch, bad
):
    teacher, report, metadata, _ = synthetic_source
    item = report["methods"][0]
    if bad == "experiment_incomplete":
        report["complete"] = False
    elif bad == "initial_mismatch":
        report["initial_arrays_and_counters_exact"] = False
    elif bad == "method_missing":
        report["methods"] = report["methods"][1:]
    elif bad == "method_duplicate":
        report["methods"].append(dict(item))
    elif bad == "method_incomplete":
        item["complete"] = False
    elif bad == "recipe_missing":
        del item["clock_recipe"]
    elif bad == "recipe_mode":
        item["clock_recipe"]["elapsed"] = True
    elif bad == "recipe_horizon":
        item["clock_recipe"]["horizon"] = 49
    elif bad == "recipe_type":
        item["clock_recipe"]["elapsed"] = 0
    elif bad == "checkpoint_missing":
        item["checkpoints"].pop()
    elif bad == "checkpoint_duplicate":
        item["checkpoints"][1]["total_updates"] = 0
    elif bad == "checkpoint_order":
        item["checkpoints"].reverse()
    elif bad == "checkpoint_false":
        item["checkpoints"][2]["full_state_exact"] = False
    elif bad == "updates":
        item["final_updates"] = 255
    elif bad == "rows":
        item["final_training_rows"] = 7423
    elif bad == "foreign_path":
        item["state_file"] = "../zero-256-state.npz"
    elif bad == "other_method_path":
        item["state_file"] = "elapsed-256-state.npz"
    elif bad.startswith("policy_"):
        field = bad.removeprefix("policy_")
        if field in ("units", "bounds", "dimension"):
            target, replacement = {
                "units": ("action_units", ["m"] * 4),
                "bounds": ("lower_bounds", [-2.0] * 4),
                "dimension": ("observation_dim", 42),
            }[field]
            item["policy"]["spec"][target] = replacement
        else:
            target, replacement = {
                "seed": ("seed", 18),
                "hidden": ("hidden_dim", 32),
                "lr": ("learning_rate", 0.02),
                "momentum": ("momentum", 0.8),
                "role": ("training_role", "probe_train"),
            }[field]
            item["policy"][target] = replacement
    else:
        transform = item["policy"]["observation_standardization"]
        if bad == "fit_rows":
            transform["fitted_rows"] = 28
        elif bad == "fit_dimension":
            transform["means"] = [0.0] * 42
        elif bad == "fit_scale":
            transform["scales"] = [0.0] * 43
        elif bad == "fit_missing":
            item["policy"]["observation_standardization"] = None
    save_metadata(report, metadata)
    monkeypatch.setattr(
        case.np,
        "load",
        lambda *_args, **_kwargs: pytest.fail("invalid metadata reached NPZ loading"),
    )
    with pytest.raises((AssertionError, ValueError, KeyError, TypeError)):
        case.restore_final(teacher, "zero")


@pytest.mark.parametrize(
    "bad", ["missing", "extra", "shape", "float32", "nan", "inf", "object", "velocity_shape"]
)
def test_should_refuse_corrupt_or_ambiguous_state_arrays(synthetic_source, bad):
    teacher, _, metadata, stored = synthetic_source
    arrays = {name: value.copy() for name, value in stored["zero"].items()}
    if bad == "missing":
        del arrays["v3"]
    elif bad == "extra":
        arrays["unexpected"] = np.ones(1)
    elif bad == "shape":
        arrays["p0"] = np.zeros((42, 64), dtype=np.float64)
    elif bad == "velocity_shape":
        arrays["v2"] = np.zeros((64, 3), dtype=np.float64)
    elif bad == "float32":
        arrays["p1"] = arrays["p1"].astype(np.float32)
    elif bad == "object":
        arrays["p1"] = arrays["p1"].astype(object)
    else:
        arrays["p2"][0, 0] = np.nan if bad == "nan" else np.inf
    np.savez(metadata.parent / "zero-256-state.npz", **arrays)
    with pytest.raises((AssertionError, ValueError)):
        case.restore_final(teacher, "zero")


@pytest.mark.parametrize("method", ["zero", "elapsed"])
@pytest.mark.parametrize("successful", [False, True])
def test_should_pass_explicit_clock_pair_to_smoke_and_audit_work(
    synthetic_source, monkeypatch, method, successful
):
    teacher, _, metadata, _ = synthetic_source
    output = metadata.parent / "native-trace.json"
    original_arguments = ["native_case", "--method", method, "--output", str(output)]
    monkeypatch.setattr(sys, "argv", original_arguments)
    monkeypatch.setattr(case.faulthandler, "enable", lambda: None)
    monkeypatch.setattr(case.saved, "load_training", lambda: (teacher, ("all",) * 29))
    calls = []

    def fake_smoke(*, learner, observation_clock):
        calls.append((learner, observation_clock))
        rows = len(teacher.transitions)
        trace = asdict(teacher)
        for transition in trace["transitions"]:
            transition["reward"] = float(successful and transition["terminated"])
        result = dict(
            complete=True,
            role="train",
            seed=17,
            source_updates=256,
            source_training_rows=7424,
            frozen_state_exact=True,
            original_owner_exact=True,
            episode=trace,
            observation_clock=asdict(observation_clock),
            clock_configuration_exact=True,
            prediction_attempts=rows,
            prediction_returns=rows,
            validated_prediction_rows=rows,
            validated_prediction_bytes=32 * rows,
            clock_encoding_attempts=rows,
            clock_encoding_returns=rows,
            clock_encoded_values=43 * rows,
            clock_values=[step / 50 if observation_clock.elapsed else 0.0 for step in range(rows)],
            optimizer_updates=0,
            sparse_return=float(successful),
            any_success=successful,
            final_success=successful,
            work=dict(
                completed_steps=rows,
                step_attempts=rows,
                completed_control_physics_steps=5 * rows,
                possible_failed_physics_steps=0,
            ),
        )
        output.write_text(json.dumps(result), encoding="utf8")
        return 0

    monkeypatch.setattr(case.smoke, "main", fake_smoke)
    assert case.main() == 0
    audit = json.loads(output.with_name("rollout-audit.json").read_text(encoding="utf8"))
    assert len(calls) == 1 and calls[0][1].elapsed is (method == "elapsed")
    assert audit["complete"] is True and audit["method"] == method
    assert audit["clock_recipe"] == {"horizon": 50, "elapsed": method == "elapsed"}
    assert audit["runtime_attempts"] == audit["runtime_returns"] == 1
    assert (
        audit["original_full_state_exact"] is True
        and audit["initial_train_observation_exact"] is True
    )
    assert audit["frozen_transform_calls"] == audit["frozen_transform_rows"] == 29
    assert audit["frozen_transform_feature_values"] == 43 * 29
    assert audit["final_terminated"] is True and audit["final_truncated"] is False
    assert audit["sparse_return"] == float(successful)
    assert audit["any_success"] is successful and audit["final_success"] is successful
    assert sys.argv is original_arguments


@pytest.mark.parametrize("oversized", ["metadata", "state"])
def test_should_refuse_oversized_inputs_before_numpy_loading(
    synthetic_source, monkeypatch, oversized
):
    teacher, _, metadata, _ = synthetic_source
    if oversized == "metadata":
        metadata.write_bytes(b" " * (512 * 1024 + 1))
    else:
        (metadata.parent / "zero-256-state.npz").write_bytes(b" " * (1024**2 + 1))
    monkeypatch.setattr(
        case.np,
        "load",
        lambda *_args, **_kwargs: pytest.fail("oversized input reached NPZ loading"),
    )
    with pytest.raises(AssertionError):
        case.restore_final(teacher, "zero")


def test_should_preserve_runtime_exception_attempt_and_restore_arguments(
    synthetic_source, monkeypatch
):
    teacher, _, metadata, _ = synthetic_source
    output = metadata.parent / "failed-trace.json"
    arguments = ["native_case", "--method", "zero", "--output", str(output)]
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(case.faulthandler, "enable", lambda: None)
    monkeypatch.setattr(case.saved, "load_training", lambda: (teacher, ("all",) * 29))

    def broken(**_kwargs):
        raise RuntimeError("runtime failed before return")

    monkeypatch.setattr(case.smoke, "main", broken)
    assert case.main() == 1
    audit = json.loads(output.with_name("rollout-audit.json").read_text(encoding="utf8"))
    assert audit["runtime_attempts"] == 1 and audit["runtime_returns"] == 0
    assert audit["complete"] is False and "runtime failed before return" in audit["error"]
    assert sys.argv is arguments and not output.exists()


def test_should_reserve_audit_exclusively_before_source_or_runtime_work(
    synthetic_source, monkeypatch
):
    _, _, metadata, _ = synthetic_source
    output = metadata.parent / "new-trace.json"
    audit = output.with_name("rollout-audit.json")
    audit.write_bytes(b"preserved earlier failed attempt\n")
    monkeypatch.setattr(sys, "argv", ["native_case", "--method", "zero", "--output", str(output)])
    monkeypatch.setattr(
        case.saved, "load_training", lambda: pytest.fail("existing audit reached source loading")
    )
    monkeypatch.setattr(
        case.smoke, "main", lambda **_: pytest.fail("existing audit reached runtime")
    )
    with pytest.raises(FileExistsError):
        case.main()
    assert audit.read_bytes() == b"preserved earlier failed attempt\n"
    assert not output.exists()
