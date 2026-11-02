"""New literal commands and TRAIN readouts; no RNG, init, fit, data or SDK."""

from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
import json
import struct
from typing import Any, cast

import numpy as np
import pytest

from scripts import push_cube_training_readouts as readouts
from src.adapters import numpy_continuous_bc as bp
from src.adapters import continuous_learnability_modes as modes
from src.core import continuous_bc as bc
from src.core import continuous_command_fidelity as commands
from src.core.cube_taskspace_state import cube_taskspace_spec
from test_push_cube_training_readouts import groups
from test_complete_b22_neural import groups as complete_groups

WORK: defaultdict[str, int] = defaultdict(int)
CAPS = dict(
    sources=32,
    forwards=256,
    forward_rows=4096,
    gradients=96,
    gradient_rows=1024,
    predictions=128,
    prediction_rows=1024,
    svds=128,
    commands=256,
    command_rows=8192,
    conversions=65536,
    snapshots=65536,
    snapshot_bytes=256 * 1024**2,
    forbidden=0,
)


def charge(key, amount=1):
    WORK[key] += amount
    assert WORK[key] <= CAPS[key], "permanent #407 fixture cap:" + key


def forbidden(*args, **options):
    charge("forbidden")


@pytest.fixture(autouse=True)
def budget(monkeypatch):
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "__init__", forbidden)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "train_batch", forbidden)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    forward, gradient, predict, snapshot, svd, fidelity = (
        bc._forward,
        bc.continuous_bc_loss_and_gradients,
        bp.NumpyContinuousBCLearner.predict,
        bp.NumpyContinuousBCLearner.snapshot_state,
        np.linalg.svd,
        commands.continuous_command_fidelity,
    )

    def forwarded(features, *args, **options):
        charge("forwards")
        charge("forward_rows", len(features))
        return forward(features, *args, **options)

    def gradients(features, *args, **options):
        charge("gradients")
        charge("gradient_rows", len(features))
        return gradient(features, *args, **options)

    def predicted(owner, features):
        charge("predictions")
        charge("prediction_rows", len(features))
        return predict(owner, features)

    def snap(owner):
        charge("snapshots")
        value = snapshot(owner)
        charge("snapshot_bytes", sum(a.nbytes for a in value.parameters + value.velocity))
        return value

    def decomposed(*args, **options):
        charge("svds")
        return svd(*args, **options)

    def command(predictions, *args, **options):
        charge("commands")
        if type(predictions) is np.ndarray:
            charge("command_rows", len(predictions))
        work = options.get("work")
        if work is None:
            work = commands.CommandFidelityWork()
            options["work"] = work
        before = work.conversion_attempts
        try:
            return fidelity(predictions, *args, **options)
        finally:
            charge("conversions", work.conversion_attempts - before)

    monkeypatch.setattr(bc, "_forward", forwarded)
    monkeypatch.setattr(bc, "continuous_bc_loss_and_gradients", gradients)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "predict", predicted)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "snapshot_state", snap)
    monkeypatch.setattr(np.linalg, "svd", decomposed)
    monkeypatch.setattr(commands, "continuous_command_fidelity", command)


def source():
    charge("sources")
    owner = object.__new__(bp.NumpyContinuousBCLearner)
    owner._policy = bp.ContinuousBCPolicy(
        cube_taskspace_spec(), hidden_dim=2, binary_gripper_axis=3
    )
    parameters = cast(
        bc.Parameters,
        (
            np.zeros((12, 2)),
            np.zeros(2),
            np.zeros((2, 4)),
            np.arctanh(np.array([0.01, -0.01, 0.0, 0.2])),
        ),
    )
    velocity = cast(bc.Parameters, tuple(np.zeros_like(a) for a in parameters))
    owner._state = bp.ContinuousBCSnapshot(owner._policy, parameters, velocity, 0, 0)
    return owner


def diagnose(predictions, targets=None, *, stages=None, binary=None, work=None):
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = predictions.copy() if targets is None else np.asarray(targets, dtype=np.float64)
    stages = tuple("one" for _ in predictions) if stages is None else stages
    return commands.continuous_command_fidelity(
        predictions,
        targets,
        cube_taskspace_spec(),
        stages,
        binary_gripper_axis=binary,
        work=work,
    )


def test_exact_native_rounding_and_numeric_signed_zero_keep_inputs():
    values = np.array([[0.1, -0.0, 0.0, -1.0], [1.0, -1.0, 0.25, 1.0]])
    prediction = values.copy()
    prediction[0, 0] = np.nextafter(0.1, 1.0)
    prediction[0, 1] = 0.0
    before = prediction.copy()
    result = diagnose(prediction, values, binary=3, stages=("first", "last"))
    assert result["native_mismatch_rows"] == []
    assert result["translation_direction_mismatch_rows"] == [[], [], []]
    assert result["gripper"]["signed_native_margins"] == [1.0, 1.0]
    assert result["gripper"]["positive_rows"] == 2
    assert result["work"]["conversion_attempts"] == result["work"]["conversion_returns"] == 16
    assert result["work"]["equality_comparisons"] == 8
    assert result["work"]["translation_direction_comparisons"] == 6
    assert np.array_equal(before, prediction) and result["gripper"]["meaning"].startswith(
        "target-sign"
    )


def test_small_mse_can_hide_opposite_translation_and_zero_margin_is_distinct():
    result = diagnose(
        [[0.001, 0.0, -1e-50, 0.0], [0.0, 0.1, 0.0, -0.1]],
        [[-0.001, 0.0, 0.0, 1.0], [0.01, -0.1, 0.0, 1.0]],
        binary=3,
    )
    assert result["native_mismatch_rows"] == [0, 1]
    assert result["translation_direction_mismatch_rows"] == [[0, 1], [1], []]
    assert result["gripper"]["zero_rows"] == [0]
    assert result["gripper"]["negative_rows"] == [1]
    assert result["native_axis_errors"][0]["maximum_error_row"] == 1
    assert result["native_axis_errors"][0]["mean_absolute_error"] < 0.01


def test_stage_indices_retain_noncontiguous_original_order_and_omit_gripper_by_default():
    result = diagnose(
        [[0.0, 0.0, 0.0, 0.3], [0.0, 0.0, 0.0, 0.1], [0.0, 0.0, 0.0, -0.1]], stages=("A", "B", "A")
    )
    assert result["gripper"] is None
    assert result["stages"][0]["row_indices"] == [0, 2]
    assert result["stages"][1]["row_indices"] == [1]
    assert result["work"]["gripper_sign_classifications"] == 0


@pytest.mark.parametrize(
    "fault",
    [
        "dtype",
        "nan",
        "bound",
        "target_shape",
        "rows",
        "labels",
        "axis_bool",
        "axis_other",
        "units",
        "bounds",
        "endpoint",
    ],
)
def test_bad_inputs_refuse_before_native_conversion(fault):
    prediction = np.array([[0.0, 0.0, 0.0, 1.0]])
    targets = prediction.copy()
    spec, stages = cube_taskspace_spec(), ("all",)
    axis: Any = None
    if fault == "dtype":
        prediction = prediction.astype(np.float32)
    elif fault == "nan":
        prediction[0, 0] = float("nan")
    elif fault == "bound":
        prediction[0, 0] = 1.000001
    elif fault == "target_shape":
        targets = np.zeros((1, 3))
    elif fault == "rows":
        prediction = np.zeros((257, 4))
    elif fault == "labels":
        stages = (" edge ",)
    elif fault == "axis_bool":
        axis = True
    elif fault == "axis_other":
        axis = 2
    elif fault == "units":
        spec = replace(spec, action_units=("wrong",) * 4)
    elif fault == "bounds":
        spec = replace(spec, upper_bounds=(2.0,) * 4)
    else:
        axis = 3
        targets[0, 3] = 0.5
    work = commands.CommandFidelityWork()
    with pytest.raises(ValueError):
        commands.continuous_command_fidelity(
            prediction, targets, spec, stages, binary_gripper_axis=axis, work=work
        )
    assert work.conversion_attempts == 0


def test_partial_native_conversion_charge_is_never_reset(monkeypatch):
    work = commands.CommandFidelityWork()
    pack = struct.pack

    def failing(*args):
        if work.conversion_attempts == 3:
            raise RuntimeError("native conversion failed")
        return pack(*args)

    monkeypatch.setattr(commands.struct, "pack", failing)
    with pytest.raises(RuntimeError):
        diagnose([[0.0, 0.0, 0.0, 1.0]], work=work)
    assert work.conversion_attempts == 3 and work.conversion_returns == 2


def test_default_and_optin_reuse_same_predictions_and_primary_diagnostics():
    owner = source()
    before = owner.snapshot_state()
    default = readouts.run_training_readout(owner, groups(), label="same", guard=lambda: None)
    enabled = readouts.run_training_readout(
        owner, groups(), label="same", guard=lambda: None, command_fidelity=True
    )
    assert default["complete"] and enabled["complete"], (default, enabled)
    assert "command_fidelity_enabled" not in default
    for name in readouts.GROUPS:
        assert default["groups"][name]["conditioning"] == enabled["groups"][name]["conditioning"]
        assert default["groups"][name]["residual"] == enabled["groups"][name]["residual"]
        assert "command_fidelity" in enabled["groups"][name]
    for key, value in default["work"].items():
        assert enabled["work"][key] == value, key
    assert default["work"]["prediction_calls"] == enabled["work"]["prediction_calls"] == 4
    assert enabled["work"]["command_returns"] == 4 and enabled["work"]["command_rows"] == 5
    assert enabled["work"]["prediction_capture_numeric_bytes"] == 5 * 8 * 8
    assert all(entry["returned"] and entry["validated"] for entry in enabled["operations"])
    assert (
        enabled["groups"]["push"]["command_fidelity"]["input"]["episode_id"]
        == groups()[0][1].episode_id
    )
    assert modes.same_continuous_state(before, owner.snapshot_state())


@pytest.mark.parametrize(
    "fault",
    [
        "throw",
        "change_capture",
        "replace_payload",
        "erase_work",
        "erase_groups",
        "replace_operations",
    ],
)
def test_late_command_failure_preserves_paid_residuals_and_refuses_more_model_work(
    monkeypatch, fault
):
    owner = source()
    original = commands.continuous_command_fidelity
    seen: list[Any] = []

    def command(prediction, target, *args, **options):
        if fault == "throw":
            raise RuntimeError("late diagnostic failure")
        result = original(prediction, target, *args, **options)
        if fault == "change_capture":
            prediction[0, 0] = 0.9
        else:
            # Simulate an opaque callback changing caller-visible paid views.
            import sys

            parent = sys._getframe(1).f_locals["report"]
            if fault == "replace_payload":
                parent["groups"]["push"]["residual"] = deepcopy(
                    parent["groups"]["push"]["residual"]
                )
            elif fault == "erase_work":
                parent["work"] = {}
            elif fault == "erase_groups":
                parent["groups"] = []
            else:
                parent["operations"] = deepcopy(parent["operations"])
        seen.append(result)
        return result

    monkeypatch.setattr(commands, "continuous_command_fidelity", command)
    result = readouts.run_training_readout(
        owner, groups(), label="failure", guard=lambda: None, command_fidelity=True
    )
    assert result["complete"] is False and result["error"]
    assert result["work"]["conditioning_returns"] == result["work"]["residual_returns"] == 1
    assert result["work"]["prediction_calls"] == 1
    assert "residual" in result["groups"]["push"] and "a" not in result["groups"]
    assert result["work"]["command_attempts"] == 1
    assert result["work"]["command_returns"] == (0 if fault == "throw" else 1)
    if seen:
        assert result["groups"]["push"]["command_fidelity"] == seen[0]


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test"])
def test_no_reserved_role_is_admitted_by_training_readout(role):
    owner = source()
    data = groups()
    data = ((data[0][0], replace(data[0][1], role=role)), *data[1:])
    result = readouts.run_training_readout(
        owner, data, label="role", guard=lambda: None, command_fidelity=True
    )
    assert result["complete"] is False and result["phase_readouts"] is None
    assert result["work"]["prediction_calls"] == 0


@pytest.mark.parametrize("flag", [1, "true", None])
def test_nonboolean_optin_refuses_before_model_access(flag):
    result = readouts.run_training_readout(
        object(), (), label="flag", guard=lambda: None, command_fidelity=flag
    )
    assert result["complete"] is False and result["phase_readouts"] is None


def test_real_generated_220row_optin_stays_inside_original_phase_payload_cap():
    owner = source()
    result = readouts.run_training_readout(
        owner,
        complete_groups(),
        label="literal220",
        guard=lambda: None,
        corrective=True,
        aggregated=True,
        complete_b22=True,
        command_fidelity=True,
    )
    assert result["complete"], result["error"]
    assert result["work"]["command_rows"] == 220 and result["work"]["command_returns"] == 7
    assert result["phase_readouts"]["work"]["callback_result_bytes"] <= 65536
    WORK["max_phase_payload_bytes"] = result["phase_readouts"]["work"]["callback_result_bytes"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("field", ["conversion_attempts", "known_owned_numeric_bytes"])
def test_mutated_accumulator_cannot_forget_preceding_paid_group(monkeypatch, field):
    original = commands.continuous_command_fidelity
    calls = 0

    def altered(*args, **options):
        nonlocal calls
        result = original(*args, **options)
        calls += 1
        if calls == 2:
            setattr(options["work"], field, 0)
        return result

    monkeypatch.setattr(commands, "continuous_command_fidelity", altered)
    result = readouts.run_training_readout(
        source(), groups(), label="counter", guard=lambda: None, command_fidelity=True
    )
    assert result["complete"] is False and "accumulator" in result["error"]
    assert calls == 2 and "b" not in result["groups"]
    assert result["work"]["prediction_calls"] == result["work"]["command_returns"] == 2
    assert result["work"]["command_fidelity"]["conversion_attempts"] == 24
    assert result["work"]["command_fidelity"]["known_owned_numeric_bytes"] == 480
    assert result["groups"]["push"]["command_fidelity"]["work"]["conversion_attempts"] == 16


def test_late_accumulator_reset_or_boolean_counter_refuses_before_more_model_work(monkeypatch):
    owner = source()
    original = commands.continuous_command_fidelity
    retained: list[Any] = []
    corrupted = False
    field, value = "conversion_attempts", 0

    def command(*args, **options):
        result = original(*args, **options)
        retained.append(options["work"])
        return result

    def guard():
        nonlocal corrupted
        if retained and not corrupted:
            setattr(retained[0], field, value)
            corrupted = True

    monkeypatch.setattr(commands, "continuous_command_fidelity", command)
    for axis in (3, None):
        retained.clear()
        corrupted = False
        if axis is None:
            field, value = "gripper_margin_products", False
            owner._policy = replace(owner._policy, binary_gripper_axis=None)
            owner._state = replace(owner._state, policy=owner._policy)
        result = readouts.run_training_readout(
            owner, groups(), label="late-counter", guard=guard, command_fidelity=True
        )
        assert result["complete"] is False and "accumulator" in result["error"]
        assert result["work"]["prediction_calls"] == result["work"]["conditioning_returns"] == 1
        assert "a" not in result["groups"]
        assert result["work"]["command_fidelity"]["conversion_attempts"] == 16
