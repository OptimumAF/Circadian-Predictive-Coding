"""Six-group TRAIN diagnostics and fixed weights, without adaptation.

One H2/D12/A4 initializer/32 normal values. Controlled zero head weights and
biases are physical-error oracles, not task learning. Forecast16 conditioning/
gradient returns596rows,13 residual/prediction returns446rows,45 forwards1638rows.
"""

from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import push_cube_correction_weights as weighting
from scripts import push_cube_training_readouts as readouts
from src.adapters import numpy_continuous_bc as bp
from src.core import continuous_bc as core
from src.core.cube_taskspace_state import cube_taskspace_spec
from test_corrective_training_readouts import episode, assert_same


NAMES = ("current_corrections", "corrections", "push", "a", "b", "train20")
COUNTS = (50, 50, 18, 15, 50, 15)
BIASES = (0.25, -0.25, 0.0, 0.5)


def groups(*, current_rows=50):
    original = episode("corrections", current_rows)
    current = replace(
        original,
        episode_id="push-cube-correction-targets-17-source120",
        transitions=tuple(
            replace(row, action=(0.125, 0.0, 0.0, -1.0)) for row in original.transitions
        ),
    )
    prior = replace(episode("corrections", 50), episode_id="push-cube-correction-targets-17")
    return (("current_corrections", current), ("corrections", prior)) + tuple(
        (name, episode(name, count)) for name, count in zip(NAMES[2:], COUNTS[2:], strict=True)
    )


def forbidden(*args, **options):
    pytest.fail("aggregated readout admitted optimization, fitting or refused-input fork")


@pytest.fixture(scope="module", autouse=True)
def ledger():
    counts = dict(initializations=0, normal_draws=0)
    for kind in ("conditioning", "residual", "gradient", "forward", "prediction"):
        for name in ("attempts", "returns", "rows_attempted", "rows_returned"):
            counts[kind + "_" + name] = 0
    initialize = bp.NumpyContinuousBCLearner.__init__

    def initialized(owner, policy):
        counts["initializations"] += 1
        counts["normal_draws"] += policy.hidden_dim * (
            policy.spec.observation_dim + len(policy.spec.action_units)
        )
        return initialize(owner, policy)

    def counted(original, kind):
        def call(*args, **options):
            if kind in ("conditioning", "residual"):
                rows = len(args[1].transitions)
            else:
                rows = len(args[1] if kind == "prediction" else args[0])
            counts[kind + "_attempts"] += 1
            counts[kind + "_rows_attempted"] += rows
            result = original(*args, **options)
            counts[kind + "_returns"] += 1
            counts[kind + "_rows_returned"] += rows
            return result

        return call

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bp.NumpyContinuousBCLearner, "__init__", initialized)
        patch.setattr(
            bp.NumpyContinuousBCLearner,
            "diagnose_training_conditioning",
            counted(bp.NumpyContinuousBCLearner.diagnose_training_conditioning, "conditioning"),
        )
        patch.setattr(
            bp.NumpyContinuousBCLearner,
            "diagnose_training_episode",
            counted(bp.NumpyContinuousBCLearner.diagnose_training_episode, "residual"),
        )
        patch.setattr(
            bp.NumpyContinuousBCLearner,
            "predict",
            counted(bp.NumpyContinuousBCLearner.predict, "prediction"),
        )
        patch.setattr(
            core,
            "continuous_bc_loss_and_gradients",
            counted(core.continuous_bc_loss_and_gradients, "gradient"),
        )
        patch.setattr(core, "_forward", counted(core._forward, "forward"))
        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", forbidden)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", forbidden)
        patch.setattr(bp, "fit_training_standardization", forbidden)
        yield counts
    if os.environ.get("ACS_READOUT_FIXTURE_LEDGER"):
        with Path(os.environ["ACS_READOUT_FIXTURE_LEDGER"]).open("x", encoding="utf8") as output:
            json.dump(counts, output, sort_keys=True)
            output.write("\n")
    assert counts["initializations"] == 1 and counts["normal_draws"] == 32
    for kind, calls, rows in (
        ("conditioning", 16, 596),
        ("gradient", 16, 596),
        ("residual", 13, 446),
        ("prediction", 13, 446),
        ("forward", 45, 1638),
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
def test_should_keep_all198_rows_and_each_axis_initial_remainder_and_native_objective(
    source, monkeypatch, mixed
):
    if mixed:
        source._policy = replace(source._policy, binary_gripper_axis=3)
        source._state = replace(source._state, policy=source._policy)
    initial = source.snapshot_state()
    data = groups()
    seal = tuple(asdict(value) for _, value in data)
    fork = bp.NumpyContinuousBCLearner.fork
    private: list[Any] = []

    def forked(owner):
        child = fork(owner)
        private.append(child)
        assert not any(
            np.shares_memory(a, b)
            for a in source._state.parameters + source._state.velocity
            for b in child._state.parameters + child._state.velocity
        )
        return child

    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forked)
    result = readouts.run_training_readout(
        source,
        data,
        label="aggregated-controlled",
        guard=lambda: None,
        corrective=True,
        aggregated=True,
    )
    assert result["complete"] is True, result["error"]
    assert len(private) == 1 and result["group_order"] == list(NAMES)
    assert tuple(result["groups"]) == NAMES and len(result["operations"]) == 12
    assert "candidate" not in result and "learner" not in result
    assert_same(source.snapshot_state(), initial)
    assert_same(private[0].snapshot_state(), initial)
    assert tuple(asdict(value) for _, value in data) == seal
    predicted = tuple(math.tanh(value) for value in BIASES)
    for name, item in data:
        count = len(item.transitions)
        group = result["groups"][name]
        assert group["rows"] == count
        condition = group["conditioning"]["conditioning"]
        assert condition["objective_definition"] == (
            core.MIXED_OBJECTIVE if mixed else core.OBJECTIVE
        )
        terms = [
            0.5 * (predicted[axis] - row.action[axis]) ** 2
            for row in item.transitions
            for axis in range(3 if mixed else 4)
        ]
        if mixed:
            terms += [
                math.log1p(math.exp(-2 * row.action[3] * BIASES[3])) for row in item.transitions
            ]
        assert condition["objective_loss"] == pytest.approx(
            math.fsum(terms) / (4 * count), rel=1e-12, abs=1e-15
        )
        residual = group["residual"]["residuals"]
        assert [(value["label"], value["summary"]["rows"]) for value in residual["stages"]] == [
            ("initial", 1),
            ("remainder", count - 1),
        ]
        for axis in residual["overall"]["axes"]:
            errors = [
                predicted[axis["index"]] - row.action[axis["index"]] for row in item.transitions
            ]
            square = math.fsum(value * value for value in errors) / count
            assert axis["normalized_bias"] == pytest.approx(math.fsum(errors) / count)
            assert axis["normalized_mae"] == pytest.approx(
                math.fsum(abs(value) for value in errors) / count
            )
            assert axis["normalized_half_mse"] == pytest.approx(square / 2)
            assert axis["normalized_rmse"] == pytest.approx(math.sqrt(square))
            assert axis["maximum_absolute_normalized_error"] == pytest.approx(max(map(abs, errors)))
    work = result["work"]
    assert work["conditioning_returns"] == work["residual_returns"] == 6
    assert work["conditioning_returned_rows"] == work["residual_returned_rows"] == 198
    assert work["forward_passes"] == 18 and work["forward_rows"] == 594
    assert work["gradient_evaluations"] == 6 and work["gradient_rows"] == 198
    assert work["gradient_values"] == 228 and work["gradient_numeric_bytes"] == 1824
    assert work["prediction_calls"] == 6 and work["prediction_rows"] == 198
    assert work["prediction_values"] == 792 and work["prediction_bytes"] == 6336
    assert work["normalization_calls"] == 12 and work["normalization_rows"] == 396
    assert work["known_dense_multiply_accumulates"] == 26928
    assert work["unknown_unreturned_work"] is False
    assert result["phase_readouts"]["forks"] == result["phase_readouts"]["validated"] == 1


def test_should_prepend_one_fixed_positive_prefix_without_changing_original148_weights():
    data = groups()
    before = tuple(asdict(value) for _, value in data)
    values, metadata = weighting.build_aggregated_correction_weights(data)
    old_values, _ = weighting.build_balanced_correction_weights(data[1:])
    assert values[:50] == (49.0,) + (1.0,) * 49 and values[50:] == old_values
    assert len(values) == 198 and all(
        type(value) is float and value > 0 and math.isfinite(value) for value in values
    )
    assert metadata == dict(
        definition="positive_train_aggregated_prefix_group_weights_v1",
        group_order=list(NAMES),
        groups=[
            dict(group=name, episode_id=value.episode_id, rows=len(value.transitions))
            for name, value in data
        ],
        values=list(values),
        rule="equal six-group mass; both correction initial/remainder split equally",
        ideal_group_mass=1 / 6,
        ideal_correction_initial_mass=1 / 12,
    )
    total, offset = math.fsum(values), 0
    for count in COUNTS:
        assert math.fsum(values[offset : offset + count]) / total == pytest.approx(1 / 6)
        offset += count
    assert values[0] / total == pytest.approx(1 / 12)
    assert values[50] / total == pytest.approx(1 / 12)
    assert (
        data[0][1].context == data[1][1].context and data[0][1].episode_id != data[1][1].episode_id
    )
    assert tuple(asdict(value) for _, value in data) == before


def test_should_return_detached_weight_metadata_without_mutating_immutable_values():
    data = groups()
    values, first = weighting.build_aggregated_correction_weights(data)
    _, second = weighting.build_aggregated_correction_weights(data)
    first["values"][0] = 999.0
    first["groups"][0]["episode_id"] = "replaced"
    assert values[0] == second["values"][0] == 49.0
    assert second["groups"][0]["episode_id"] == data[0][1].episode_id


@pytest.mark.parametrize(
    "fault",
    [
        "flag",
        "requires_corrective",
        "default",
        "order",
        "name",
        "duplicate",
        "role",
        "spec",
        "overflow",
        "list",
    ],
)
def test_should_refuse_inadmissible_six_group_readouts_before_fork(source, monkeypatch, fault):
    data: Any = groups()
    options: dict[str, Any] = dict(corrective=True, aggregated=True)
    if fault == "flag":
        options["aggregated"] = 1
    elif fault == "requires_corrective":
        options["corrective"] = False
    elif fault == "default":
        options.pop("aggregated")
    elif fault == "order":
        data = (data[1], data[0]) + data[2:]
    elif fault == "name":
        data = (("unknown", data[0][1]),) + data[1:]
    elif fault == "duplicate":
        data = ((data[0][0], replace(data[0][1], episode_id=data[1][1].episode_id)),) + data[1:]
    elif fault == "role":
        data = ((data[0][0], replace(data[0][1], role="probe_test")),) + data[1:]
    elif fault == "spec":
        spec = replace(data[0][1].spec, action_units=("other",) + data[0][1].spec.action_units[1:])
        data = ((data[0][0], replace(data[0][1], spec=spec)),) + data[1:]
    elif fault == "overflow":
        data = groups(current_rows=51)
    else:
        data = list(data)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forbidden)
    result = readouts.run_training_readout(
        source, data, label="refused", guard=lambda: None, **options
    )
    assert result["complete"] is False and result["error"]
    assert result["operations"] == [] and result["phase_readouts"] is None


@pytest.mark.parametrize("fault", ["list", "pair_list", "order", "count", "duplicate", "role"])
def test_should_refuse_wrong_weight_group_structure_count_identity_or_role(fault):
    data: Any = groups()
    if fault == "list":
        data = list(data)
    elif fault == "pair_list":
        data = (list(data[0]),) + data[1:]
    elif fault == "order":
        data = (data[1], data[0]) + data[2:]
    elif fault == "count":
        data = groups(current_rows=51)
    elif fault == "duplicate":
        data = ((data[0][0], replace(data[0][1], episode_id=data[1][1].episode_id)),) + data[1:]
    else:
        data = ((data[0][0], replace(data[0][1], role="final_test")),) + data[1:]
    with pytest.raises(ValueError):
        weighting.build_aggregated_correction_weights(data)


@pytest.mark.parametrize("fault", ["source", "detached"])
def test_should_retain_actual_paid_current50_conditioning_before_refusing_owner_mutation(
    source, monkeypatch, fault
):
    before = source.snapshot_state()
    diagnose = readouts.diagnose_conditioning

    def changed(detached, item):
        result = diagnose(detached, item)
        selected = source if fault == "source" else detached
        selected._state.parameters[3][0] += 0.125
        return result

    monkeypatch.setattr(readouts, "diagnose_conditioning", changed)
    result = readouts.run_training_readout(
        source, groups(), label="paid", guard=lambda: None, corrective=True, aggregated=True
    )
    assert result["complete"] is False
    assert len(result["operations"]) == 1 and result["operations"][0]["returned"] is True
    assert result["operations"][0]["validated"] is False
    assert (
        result["work"]["conditioning_returned_rows"] == 50
        and result["work"]["residual_attempts"] == 0
    )
    assert "conditioning" in result["groups"]["current_corrections"]
    if fault == "source":
        assert source._state.parameters[3][0] == before.parameters[3][0] + 0.125
    else:
        assert_same(source.snapshot_state(), before)


def test_should_keep_both_paid_returns_when_preceding_conditioning_payload_changes(
    source, monkeypatch
):
    captured: list[dict[str, Any]] = []
    serialize, residual = readouts.asdict, readouts.diagnose_residual

    def serialized(value):
        result = serialize(value)
        if type(value).__name__ == "TrainingConditioningDiagnostic" and not captured:
            captured.append(result)
        return result

    def changed(*args, **options):
        result = residual(*args, **options)
        captured[0]["conditioning"]["objective_loss"] = 999.0
        return result

    monkeypatch.setattr(readouts, "asdict", serialized)
    monkeypatch.setattr(readouts, "diagnose_residual", changed)
    result = readouts.run_training_readout(
        source, groups(), label="prior-paid", guard=lambda: None, corrective=True, aggregated=True
    )
    assert result["complete"] is False and "preceding diagnostic payload changed" in result["error"]
    assert len(result["operations"]) == 2 and result["operations"][1]["returned"] is True
    assert result["operations"][1]["validated"] is False
    assert (
        result["groups"]["current_corrections"]["conditioning"]["conditioning"]["objective_loss"]
        != 999.0
    )
    assert result["work"]["conditioning_returns"] == result["work"]["residual_returns"] == 1


def test_should_keep_unobserved_conditioning_work_unknown_without_inventing_a_return(
    source, monkeypatch
):
    diagnose = readouts.diagnose_conditioning

    def unavailable(*args, **options):
        diagnose(*args, **options)
        raise RuntimeError("current50 conditioning return unavailable after real diagnostic")

    monkeypatch.setattr(readouts, "diagnose_conditioning", unavailable)
    result = readouts.run_training_readout(
        source, groups(), label="unknown", guard=lambda: None, corrective=True, aggregated=True
    )
    assert result["complete"] is False and "return unavailable" in result["error"]
    assert result["work"]["unknown_unreturned_work"] is True
    assert (
        result["work"]["conditioning_attempts"] == 1 and result["work"]["conditioning_returns"] == 0
    )
    assert result["operations"][0]["returned"] is False
