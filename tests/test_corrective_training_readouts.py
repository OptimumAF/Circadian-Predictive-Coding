"""Five-group corrective TRAIN diagnosis, without adapting any learner.

One genuine H2/D12/A4 initializer draws32 values. Zero head weights and declared
biases supply a physical oracle, not evidence of a learned task. Full module:
14 conditioning/gradient returns on496 rows,11 residual/prediction returns on346
rows,39 forwards/1338 rows. No training, fit, teacher, SDK or scientific payload.
"""

from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import push_cube_training_readouts as readouts
from src.adapters import numpy_continuous_bc as bp
from src.core import continuous_bc as core
from src.core.continuous_demonstration import ContinuousEpisode, ContinuousTransition
from src.core.cube_taskspace_state import cube_taskspace_spec


NAMES = ("corrections", "push", "a", "b", "train20")
COUNTS = (50, 18, 15, 50, 15)
BIASES = (0.25, -0.25, 0.0, 0.5)


def forbidden(*args, **kwargs):
    pytest.fail("corrective readout admitted training, fit or refused-input fork")


def episode(name, count):
    first = {
        "corrections": (0.0, 0.0, 0.0, -1.0),
        "push": (0.0, 0.0, 0.0, -1.0),
        "a": (0.2, -0.2, 0.0, 1.0),
        "b": (-0.1, 0.1, 0.2, -1.0),
        "train20": (0.1, 0.2, -0.2, 1.0),
    }[name]

    def observation(index):
        return (0.0, 0.25 + (index % 3) * 0.125, -0.25) + (0.0,) * 9

    rows = tuple(
        ContinuousTransition(
            index,
            observation(index),
            first if index == 0 or name == "corrections" else (0.0, 0.0, 0.0, 1.0),
            0.0,
            observation(index + 1),
            index == count - 1,
            False,
        )
        for index in range(count)
    )
    return ContinuousEpisode(
        f"synthetic-{name}", f"TRAIN-{name}", "train", cube_taskspace_spec(), rows
    )


def groups():
    return tuple((name, episode(name, count)) for name, count in zip(NAMES, COUNTS, strict=True))


def assert_same(actual, expected):
    assert actual.policy == expected.policy
    assert (actual.completed_updates, actual.training_examples) == (
        expected.completed_updates,
        expected.training_examples,
    )
    for first, second in zip(
        actual.parameters + actual.velocity, expected.parameters + expected.velocity, strict=True
    ):
        np.testing.assert_array_equal(first, second)


@pytest.fixture(scope="module", autouse=True)
def ledger():
    counts = dict(
        initializations=0,
        normal_draws=0,
        conditioning_attempts=0,
        conditioning_returns=0,
        conditioning_rows_attempted=0,
        conditioning_rows_returned=0,
        residual_attempts=0,
        residual_returns=0,
        residual_rows_attempted=0,
        residual_rows_returned=0,
        gradient_attempts=0,
        gradient_returns=0,
        gradient_rows_attempted=0,
        gradient_rows_returned=0,
        forward_attempts=0,
        forward_returns=0,
        forward_rows_attempted=0,
        forward_rows_returned=0,
        prediction_attempts=0,
        prediction_returns=0,
        prediction_rows_attempted=0,
        prediction_rows_returned=0,
    )
    initialize = bp.NumpyContinuousBCLearner.__init__
    conditioning, residual = (
        bp.NumpyContinuousBCLearner.diagnose_training_conditioning,
        bp.NumpyContinuousBCLearner.diagnose_training_episode,
    )
    gradient, forward, predict = (
        core.continuous_bc_loss_and_gradients,
        core._forward,
        bp.NumpyContinuousBCLearner.predict,
    )

    def initialized(owner, policy):
        counts["initializations"] += 1
        counts["normal_draws"] += policy.hidden_dim * (
            policy.spec.observation_dim + len(policy.spec.action_units)
        )
        return initialize(owner, policy)

    def counted(original, kind):
        def call(*args, **options):
            value = args[1] if kind in ("conditioning", "residual", "prediction") else args[0]
            rows = len(value.transitions) if kind in ("conditioning", "residual") else len(value)
            counts[kind + "_attempts"] += 1
            counts[kind + "_rows_attempted"] += rows
            returned = original(*args, **options)
            counts[kind + "_returns"] += 1
            counts[kind + "_rows_returned"] += rows
            return returned

        return call

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bp.NumpyContinuousBCLearner, "__init__", initialized)
        patch.setattr(
            bp.NumpyContinuousBCLearner,
            "diagnose_training_conditioning",
            counted(conditioning, "conditioning"),
        )
        patch.setattr(
            bp.NumpyContinuousBCLearner, "diagnose_training_episode", counted(residual, "residual")
        )
        patch.setattr(core, "continuous_bc_loss_and_gradients", counted(gradient, "gradient"))
        patch.setattr(core, "_forward", counted(forward, "forward"))
        patch.setattr(bp.NumpyContinuousBCLearner, "predict", counted(predict, "prediction"))
        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", forbidden)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", forbidden)
        patch.setattr(bp, "fit_training_standardization", forbidden)
        yield counts
    if os.environ.get("CORRECTIVE_READOUT_FIXTURE_LEDGER"):
        with Path(os.environ["CORRECTIVE_READOUT_FIXTURE_LEDGER"]).open(
            "x", encoding="utf8"
        ) as output:
            json.dump(counts, output, sort_keys=True)
            output.write("\n")
    assert counts["initializations"] == 1 and counts["normal_draws"] == 32
    for kind, calls, rows in (
        ("conditioning", 14, 496),
        ("gradient", 14, 496),
        ("residual", 11, 346),
        ("prediction", 11, 346),
        ("forward", 39, 1338),
    ):
        assert counts[kind + "_attempts"] == counts[kind + "_returns"] == calls
        assert counts[kind + "_rows_attempted"] == counts[kind + "_rows_returned"] == rows


@pytest.fixture(scope="module")
def owner(ledger):
    return bp.NumpyContinuousBCLearner(
        bp.ContinuousBCPolicy(cube_taskspace_spec(), hidden_dim=2, seed=17)
    )


@pytest.fixture
def source(owner):
    source = owner.fork()
    source._state.parameters[2].fill(0.0)
    source._state.parameters[3][:] = BIASES
    source._state.velocity[3].fill(0.125)
    return source


@pytest.mark.parametrize("mixed", [False, True])
def test_should_report_all148_rows_with_native_objective_and_independent_physical_axis_oracles(
    source, monkeypatch, mixed
):
    if mixed:
        source._policy = replace(source._policy, binary_gripper_axis=3)
        source._state = replace(source._state, policy=source._policy)
    initial = source.snapshot_state()
    data = groups()
    input_seal = tuple(asdict(value) for _, value in data)
    detached: list[Any] = []
    fork = bp.NumpyContinuousBCLearner.fork

    def forked(owner):
        child = fork(owner)
        assert child is not source
        assert not any(
            np.shares_memory(a, b)
            for a in source._state.parameters + source._state.velocity
            for b in child._state.parameters + child._state.velocity
        )
        detached.append(child)
        return child

    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forked)
    report = readouts.run_training_readout(
        source, data, label="controlled148", guard=lambda: None, corrective=True
    )
    assert report["complete"] is True, report["error"]
    assert report["group_order"] == list(NAMES) and len(detached) == 1
    assert list(report["groups"]) == list(NAMES)
    assert "learner" not in report and "candidate" not in report
    assert report["native_objective"] == (core.MIXED_OBJECTIVE if mixed else core.OBJECTIVE)
    assert report["physical_objective"] == "normalized_active_axis_half_mse"
    assert_same(source.snapshot_state(), initial)
    assert_same(detached[0].snapshot_state(), initial)
    assert tuple(asdict(value) for _, value in data) == input_seal
    predicted = tuple(math.tanh(value) for value in BIASES)
    for name, item in data:
        group = report["groups"][name]
        n = len(item.transitions)
        assert group["rows"] == n
        native = group["conditioning"]["conditioning"]
        assert native["objective_definition"] == report["native_objective"]
        terms = [
            0.5 * (predicted[axis] - row.action[axis]) ** 2
            for row in item.transitions
            for axis in range(3 if mixed else 4)
        ]
        if mixed:
            terms += [
                math.log1p(math.exp(-2 * row.action[3] * BIASES[3])) for row in item.transitions
            ]
        assert native["objective_loss"] == pytest.approx(
            math.fsum(terms) / (4 * n), rel=1e-12, abs=1e-15
        )
        residual = group["residual"]["residuals"]
        assert residual["overall"]["rows"] == n
        for axis in residual["overall"]["axes"]:
            index = axis["index"]
            errors = [predicted[index] - row.action[index] for row in item.transitions]
            assert axis["normalized_bias"] == pytest.approx(
                math.fsum(errors) / n, rel=1e-12, abs=1e-15
            )
            assert axis["normalized_mae"] == pytest.approx(
                math.fsum(abs(value) for value in errors) / n, rel=1e-12, abs=1e-15
            )
            assert axis["normalized_half_mse"] == pytest.approx(
                math.fsum(value * value for value in errors) / (2 * n), rel=1e-12, abs=1e-15
            )
        assert [(entry["label"], entry["summary"]["rows"]) for entry in residual["stages"]] == [
            ("initial", 1),
            ("remainder", n - 1),
        ]
    work = report["work"]
    assert work["conditioning_returns"] == work["residual_returns"] == 5
    assert work["conditioning_returned_rows"] == work["residual_returned_rows"] == 148
    assert work["forward_passes"] == 15 and work["forward_rows"] == 444
    assert work["gradient_evaluations"] == 5 and work["gradient_rows"] == 148
    assert work["gradient_values"] == 190 and work["gradient_numeric_bytes"] == 1520
    assert work["prediction_calls"] == 5 and work["prediction_rows"] == 148
    assert work["prediction_values"] == 592 and work["prediction_bytes"] == 4736
    assert work["normalization_calls"] == 10 and work["normalization_rows"] == 296
    assert work["known_dense_multiply_accumulates"] == 20128
    assert work["unknown_unreturned_work"] is False
    assert len(report["operations"]) == 10
    assert all(value["returned"] is value["validated"] is True for value in report["operations"])
    assert report["phase_readouts"]["forks"] == report["phase_readouts"]["validated"] == 1
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "fault",
    [
        "flag_int",
        "flag_none",
        "default",
        "list",
        "order",
        "missing",
        "duplicate_name",
        "duplicate_episode",
        "probe_train",
        "probe_test",
        "final_test",
        "spec",
        "rows",
    ],
)
def test_should_refuse_nontrain_malformed_or_149_row_diagnosis_before_fork(
    source, monkeypatch, fault
):
    data: Any = groups()
    corrective: Any = True
    if fault == "flag_int":
        corrective = 1
    elif fault == "flag_none":
        corrective = None
    elif fault == "default":
        corrective = False
    elif fault == "list":
        data = list(data)
    elif fault == "order":
        data = (data[1], data[0], *data[2:])
    elif fault == "missing":
        data = data[:-1]
    elif fault == "duplicate_name":
        data = (data[0], (data[0][0], data[1][1]), *data[2:])
    elif fault == "duplicate_episode":
        data = (*data[:-1], ("train20", replace(data[-1][1], episode_id=data[0][1].episode_id)))
    elif fault in ("probe_train", "probe_test", "final_test"):
        data = (("corrections", replace(data[0][1], role=fault)), *data[1:])
        monkeypatch.setattr(readouts, "copy_continuous_episode", forbidden)
    elif fault == "spec":
        data = (
            (
                "corrections",
                replace(data[0][1], spec=replace(data[0][1].spec, action_units=("m",) * 4)),
            ),
            *data[1:],
        )
    else:
        data = (("corrections", episode("corrections", 51)), *data[1:])
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forbidden)
    report = readouts.run_training_readout(
        source, data, label="invalid148", guard=lambda: None, corrective=corrective
    )
    assert report["complete"] is False and report["error"]
    assert report["operations"] == [] and report["phase_readouts"] is None
    assert report["work"]["conditioning_attempts"] == report["work"]["residual_attempts"] == 0


@pytest.mark.parametrize("fault", ["source", "detached"])
def test_should_retain_actual_paid_50_row_conditioning_before_refusing_model_mutation(
    source, monkeypatch, fault
):
    initial = source.snapshot_state()
    diagnose = readouts.diagnose_conditioning

    def changed(detached, item):
        returned = diagnose(detached, item)
        owner = source if fault == "source" else detached
        owner._state.parameters[3][0] += 0.125
        return returned

    monkeypatch.setattr(readouts, "diagnose_conditioning", changed)
    report = readouts.run_training_readout(
        source, groups(), label="paid148", guard=lambda: None, corrective=True
    )
    assert report["complete"] is False and report["error"]
    assert len(report["operations"]) == 1 and report["operations"][0]["returned"] is True
    assert report["operations"][0]["validated"] is False
    assert "conditioning" in report["groups"]["corrections"]
    assert report["work"]["conditioning_returns"] == 1
    assert (
        report["work"]["conditioning_returned_rows"] == 50
        and report["work"]["residual_attempts"] == 0
    )
    if fault == "source":
        assert source._state.parameters[3][0] == initial.parameters[3][0] + 0.125
    else:
        assert_same(source.snapshot_state(), initial)


def test_should_preserve_two_real_returns_when_preceding_serialized_conditioning_is_corrupted(
    source, monkeypatch
):
    encoded: list[dict[str, Any]] = []
    serialize, residual = readouts.asdict, readouts.diagnose_residual

    def captured(value):
        returned = serialize(value)
        if type(value).__name__ == "TrainingConditioningDiagnostic" and not encoded:
            encoded.append(returned)
        return returned

    def changed(*args, **options):
        returned = residual(*args, **options)
        encoded[0]["conditioning"]["objective_loss"] = 999.0
        return returned

    monkeypatch.setattr(readouts, "asdict", captured)
    monkeypatch.setattr(readouts, "diagnose_residual", changed)
    report = readouts.run_training_readout(
        source, groups(), label="corrupt148", guard=lambda: None, corrective=True
    )
    assert report["complete"] is False and "preceding diagnostic payload changed" in report["error"]
    assert len(report["operations"]) == 2
    assert report["operations"][0]["validated"] is True
    assert (
        report["operations"][1]["returned"] is True
        and report["operations"][1]["validated"] is False
    )
    assert (
        report["groups"]["corrections"]["conditioning"]["conditioning"]["objective_loss"] != 999.0
    )
    assert "residual" in report["groups"]["corrections"]
    assert report["work"]["conditioning_returns"] == report["work"]["residual_returns"] == 1


def test_should_mark_real_unobserved_conditioning_cost_unknown_without_inventing_a_return(
    source, monkeypatch
):
    diagnose = readouts.diagnose_conditioning

    def unavailable(*args, **options):
        diagnose(*args, **options)
        raise RuntimeError("paid corrective conditioning return unavailable")

    monkeypatch.setattr(readouts, "diagnose_conditioning", unavailable)
    report = readouts.run_training_readout(
        source, groups(), label="unknown148", guard=lambda: None, corrective=True
    )
    assert report["complete"] is False and "return unavailable" in report["error"]
    assert len(report["operations"]) == 1 and report["operations"][0]["returned"] is False
    assert (
        report["work"]["conditioning_attempts"] == 1 and report["work"]["conditioning_returns"] == 0
    )
    assert report["work"]["unknown_unreturned_work"] is True
    assert report["work"]["forward_passes"] == report["work"]["gradient_evaluations"] == 0
