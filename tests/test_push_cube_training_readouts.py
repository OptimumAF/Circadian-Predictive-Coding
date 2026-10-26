"""Paid TRAIN-only diagnostics on detached copies, never model adaptation.

One genuine D12/H2/A4 initializer draws32 values. Controlled head biases provide
an explicit per-axis residual oracle; no training, fit, teacher or SDK is used.
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


def forbidden(*args, **kwargs):
    pytest.fail("TRAIN readout admitted training, fit, or refused-input fork")


def episode(name, rows=1):
    targets = {
        "push": (0.0, 0.0, 0.0, -1.0),
        "a": (0.2, -0.2, 0.0, 1.0),
        "b": (-0.1, 0.1, 0.2, -1.0),
        "train20": (0.1, 0.2, -0.2, 1.0),
    }

    def observation(index):
        return (0.0, 0.25 + index * 0.125, -0.25) + (0.0,) * 9

    transitions = tuple(
        ContinuousTransition(
            i,
            observation(i),
            (targets[name] if i == 0 else (0.0, 0.0, 0.0, 1.0)),
            0.0,
            observation(i + 1),
            i == rows - 1,
            False,
        )
        for i in range(rows)
    )
    return ContinuousEpisode(
        f"synthetic-{name}", f"TRAIN-{name}", "train", cube_taskspace_spec(), transitions
    )


def groups():
    return tuple(
        (name, episode(name, 2 if name == "push" else 1)) for name in ("push", "a", "b", "train20")
    )


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
        conditioning_rows=0,
        residual_attempts=0,
        residual_returns=0,
        residual_rows=0,
        gradient_attempts=0,
        gradient_returns=0,
        gradient_rows=0,
        forwards=0,
        forward_rows=0,
        prediction_calls=0,
        prediction_rows=0,
    )
    initialize = bp.NumpyContinuousBCLearner.__init__
    conditioning = bp.NumpyContinuousBCLearner.diagnose_training_conditioning
    residual = bp.NumpyContinuousBCLearner.diagnose_training_episode
    gradient = core.continuous_bc_loss_and_gradients
    forward = core._forward
    predict = bp.NumpyContinuousBCLearner.predict

    def initialized(owner, policy):
        counts["initializations"] += 1
        counts["normal_draws"] += policy.hidden_dim * (
            policy.spec.observation_dim + len(policy.spec.action_units)
        )
        return initialize(owner, policy)

    def conditioned(owner, episode):
        counts["conditioning_attempts"] += 1
        counts["conditioning_rows"] += len(episode.transitions)
        result = conditioning(owner, episode)
        counts["conditioning_returns"] += 1
        return result

    def residuals(owner, episode, stages):
        counts["residual_attempts"] += 1
        counts["residual_rows"] += len(episode.transitions)
        result = residual(owner, episode, stages)
        counts["residual_returns"] += 1
        return result

    def gradients(features, *args, **options):
        counts["gradient_attempts"] += 1
        counts["gradient_rows"] += len(features)
        result = gradient(features, *args, **options)
        counts["gradient_returns"] += 1
        return result

    def forwarded(features, *args, **options):
        counts["forwards"] += 1
        counts["forward_rows"] += len(features)
        return forward(features, *args, **options)

    def predicted(owner, features):
        counts["prediction_calls"] += 1
        counts["prediction_rows"] += len(features)
        return predict(owner, features)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bp.NumpyContinuousBCLearner, "__init__", initialized)
        patch.setattr(bp.NumpyContinuousBCLearner, "diagnose_training_conditioning", conditioned)
        patch.setattr(bp.NumpyContinuousBCLearner, "diagnose_training_episode", residuals)
        patch.setattr(core, "continuous_bc_loss_and_gradients", gradients)
        patch.setattr(core, "_forward", forwarded)
        patch.setattr(bp.NumpyContinuousBCLearner, "predict", predicted)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", forbidden)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", forbidden)
        patch.setattr(bp, "fit_training_standardization", forbidden)
        yield counts
    if os.environ.get("TRAIN_FIT_FIXTURE_LEDGER"):
        with Path(os.environ["TRAIN_FIT_FIXTURE_LEDGER"]).open("x", encoding="utf8") as output:
            json.dump(counts, output, sort_keys=True)
            output.write("\n")
    assert counts["initializations"] == 1 and counts["normal_draws"] == 32
    assert counts["conditioning_attempts"] <= 32 and counts["conditioning_rows"] <= 128
    assert counts["residual_attempts"] <= 32 and counts["residual_rows"] <= 128
    assert counts["gradient_attempts"] <= 32 and counts["gradient_rows"] <= 128
    assert counts["forwards"] <= 96 and counts["forward_rows"] <= 384
    assert counts["prediction_rows"] <= 128


@pytest.fixture(scope="module")
def owner(ledger):
    return bp.NumpyContinuousBCLearner(
        bp.ContinuousBCPolicy(cube_taskspace_spec(), hidden_dim=2, seed=17)
    )


@pytest.fixture
def source(owner):
    source = owner.fork()
    source._state.parameters[2].fill(0.0)
    source._state.parameters[3][:] = (0.25, -0.25, 0.0, 0.5)
    return source


@pytest.mark.parametrize("mixed", [False, True])
def test_should_report_each_group_axis_and_first_remainder_without_changing_any_model(
    source, monkeypatch, mixed
):
    if mixed:
        source._policy = replace(source._policy, binary_gripper_axis=3)
        source._state = replace(source._state, policy=source._policy)
    initial = source.snapshot_state()
    data = groups()
    original_data = tuple(asdict(value) for _, value in data)
    detached: list[Any] = []
    fork = bp.NumpyContinuousBCLearner.fork

    def forked(owner):
        result = fork(owner)
        detached.append(result)
        assert result is not source
        assert not any(
            np.shares_memory(a, b)
            for a in source._state.parameters + source._state.velocity
            for b in result._state.parameters + result._state.velocity
        )
        return result

    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forked)
    report = readouts.run_training_readout(source, data, label="synthetic", guard=lambda: None)
    assert report["complete"] is True, report["error"]
    assert report["native_objective"] == (core.MIXED_OBJECTIVE if mixed else core.OBJECTIVE)
    assert report["physical_objective"] == "normalized_active_axis_half_mse"
    assert report["group_order"] == ["push", "a", "b", "train20"]
    assert len(detached) == 1
    assert_same(source.snapshot_state(), initial)
    assert_same(detached[0].snapshot_state(), initial)
    assert tuple(asdict(value) for _, value in data) == original_data
    predicted = tuple(math.tanh(value) for value in (0.25, -0.25, 0.0, 0.5))
    for name, episode in data:
        group = report["groups"][name]
        conditioning = group["conditioning"]["conditioning"]
        assert conditioning["objective_definition"] == report["native_objective"]
        assert math.isfinite(conditioning["objective_loss"])
        assert len(conditioning["output_saturation"]) == 4 and len(conditioning["gradients"]) == 4
        residual = group["residual"]["residuals"]
        assert residual["overall"]["rows"] == len(episode.transitions)
        for index, axis in enumerate(residual["overall"]["axes"]):
            errors = [predicted[index] - row.action[index] for row in episode.transitions]
            assert axis["index"] == index and axis["unit"] == episode.spec.action_units[index]
            assert axis["normalized_bias"] == pytest.approx(sum(errors) / len(errors))
            assert axis["normalized_mae"] == pytest.approx(
                sum(abs(value) for value in errors) / len(errors)
            )
            assert axis["normalized_half_mse"] == pytest.approx(
                sum(value * value for value in errors) / (2 * len(errors))
            )
        assert [stage["label"] for stage in residual["stages"]] == (
            ["initial", "remainder"] if name == "push" else ["initial"]
        )
    push_stages = report["groups"]["push"]["residual"]["residuals"]["stages"]
    assert push_stages[0]["summary"]["axes"][3]["normalized_bias"] == pytest.approx(
        predicted[3] + 1.0
    )
    assert push_stages[1]["summary"]["axes"][3]["normalized_bias"] == pytest.approx(
        predicted[3] - 1.0
    )
    work = report["work"]
    assert work["conditioning_returns"] == work["residual_returns"] == 4
    assert work["conditioning_returned_rows"] == work["residual_returned_rows"] == 5
    assert work["forward_passes"] == 12 and work["forward_rows"] == 15
    assert work["gradient_evaluations"] == 4 and work["gradient_values"] == 152
    assert work["prediction_rows"] == 5 and work["prediction_bytes"] == 160
    assert work["unknown_unreturned_work"] is False
    assert report["phase_readouts"]["validated"] == report["phase_readouts"]["forks"] == 1
    assert all(entry["returned"] is entry["validated"] is True for entry in report["operations"])
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("fault", ["list", "order", "missing", "role", "spec", "rows", "foreign"])
def test_should_refuse_nontrain_or_invalid_group_admission_before_fork(source, monkeypatch, fault):
    data: Any = groups()
    if fault == "list":
        data = list(data)
    elif fault == "order":
        data = (data[1], data[0], *data[2:])
    elif fault == "missing":
        data = data[:3]
    elif fault == "role":
        data = (("push", replace(data[0][1], role="probe_test")), *data[1:])
    elif fault == "spec":
        invalid = replace(data[0][1], spec=replace(data[0][1].spec, action_units=("other",) * 4))
        data = (("push", invalid), *data[1:])
    elif fault == "rows":
        data = (("push", episode("push", 130)), *data[1:])
    else:
        source = object()
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forbidden)
    report = readouts.run_training_readout(source, data, label="invalid", guard=lambda: None)
    assert report["complete"] is False and report["error"]
    assert report["operations"] == [] and report["phase_readouts"] is None
    assert report["work"]["conditioning_attempts"] == report["work"]["residual_attempts"] == 0


@pytest.mark.parametrize("fault", ["source", "detached"])
def test_should_keep_returned_conditioning_before_refusing_paid_model_mutation(
    source, monkeypatch, fault
):
    initial = source.snapshot_state()
    original = readouts.diagnose_conditioning

    def changed(detached, episode):
        result = original(detached, episode)
        owner = source if fault == "source" else detached
        owner._state.parameters[3][0] += 0.125
        return result

    monkeypatch.setattr(readouts, "diagnose_conditioning", changed)
    report = readouts.run_training_readout(
        source, groups(), label="paid-mutation", guard=lambda: None
    )
    assert report["complete"] is False and report["error"]
    assert len(report["operations"]) == 1
    assert (
        report["operations"][0]["returned"] is True
        and report["operations"][0]["validated"] is False
    )
    assert "conditioning" in report["groups"]["push"]
    assert report["work"]["conditioning_returns"] == 1 and report["work"]["residual_attempts"] == 0
    if fault == "source":
        assert source._state.parameters[3][0] == initial.parameters[3][0] + 0.125
    else:
        assert_same(source.snapshot_state(), initial)


def test_should_preserve_both_returns_but_refuse_changed_preceding_serialized_diagnostic(
    source, monkeypatch
):
    encoded: list[dict[str, Any]] = []
    serialize = readouts.asdict
    residual = readouts.diagnose_residual

    def captured(value):
        result = serialize(value)
        if type(value).__name__ == "TrainingConditioningDiagnostic" and not encoded:
            encoded.append(result)
        return result

    def changed(*args, **options):
        result = residual(*args, **options)
        encoded[0]["conditioning"]["objective_loss"] = 999.0
        return result

    monkeypatch.setattr(readouts, "asdict", captured)
    monkeypatch.setattr(readouts, "diagnose_residual", changed)
    report = readouts.run_training_readout(
        source, groups(), label="preceding-payload", guard=lambda: None
    )
    assert report["complete"] is False and "preceding diagnostic payload changed" in report["error"]
    assert len(report["operations"]) == 2 and report["operations"][0]["validated"] is True
    assert (
        report["operations"][1]["returned"] is True
        and report["operations"][1]["validated"] is False
    )
    assert report["groups"]["push"]["conditioning"]["conditioning"]["objective_loss"] != 999.0
    assert "residual" in report["groups"]["push"]
    assert report["work"]["conditioning_returns"] == report["work"]["residual_returns"] == 1


def test_should_mark_unreturned_diagnostic_work_unknown_without_inventing_a_return(
    source, monkeypatch
):
    original = readouts.diagnose_conditioning

    def unavailable(*args, **options):
        original(*args, **options)
        raise RuntimeError("conditioning return unavailable")

    monkeypatch.setattr(readouts, "diagnose_conditioning", unavailable)
    report = readouts.run_training_readout(
        source, groups(), label="unknown-return", guard=lambda: None
    )
    assert report["complete"] is False and "conditioning return unavailable" in report["error"]
    assert (
        report["operations"][0]["attempted"] is True
        and report["operations"][0]["returned"] is False
    )
    assert (
        report["work"]["conditioning_attempts"] == 1 and report["work"]["conditioning_returns"] == 0
    )
    assert report["work"]["unknown_unreturned_work"] is True
    assert report["work"]["forward_passes"] == report["work"]["gradient_evaluations"] == 0
