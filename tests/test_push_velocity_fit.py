"""Private TRAIN velocity preparation and fixed study routing, without fitting.

One genuine tiny initializer consumes32 normal values. Counter112/9832 and the
nonzero velocity are controlled snapshot tags, not evidence of learned behavior.
Every model operation here is a fork/snapshot/restore; SGD and evaluations fail.
"""

from dataclasses import replace
from collections import defaultdict
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.adapters import continuous_probe_control as control
from src.adapters import numpy_continuous_bc as bp
from src.core.cube_taskspace_state import cube_taskspace_spec
from src.core.observation_standardization import ObservationStandardization
from test_continuous_mixed_continuation import assert_same
from test_push_cube_continuation_study import run_routing_case, study


def forbidden(*args, **options):
    pytest.fail("velocity preparation admitted optimization, prediction, fitting or refused work")


@pytest.fixture(scope="module", autouse=True)
def ledger():
    counts: dict[str, int] = defaultdict(
        int,
        initializations=0,
        normal_draws=0,
        native_fork_attempts=0,
        native_fork_returns=0,
        native_restore_attempts=0,
        native_restore_returns=0,
        zero_array_attempts=0,
        zero_array_returns=0,
        zero_values_returned=0,
        zero_bytes_returned=0,
    )
    initialize = bp.NumpyContinuousBCLearner.__init__
    fork, restore = bp.NumpyContinuousBCLearner.fork, bp.NumpyContinuousBCLearner.restore_state
    zero_snapshot = control._zero_velocity_snapshot

    def initialized(owner, policy):
        counts["initializations"] += 1
        counts["normal_draws"] += policy.hidden_dim * (
            policy.spec.observation_dim + len(policy.spec.action_units)
        )
        return initialize(owner, policy)

    def counted(original, name):
        def call(*args, **options):
            counts["native_" + name + "_attempts"] += 1
            result = original(*args, **options)
            counts["native_" + name + "_returns"] += 1
            return result

        return call

    def zeroed(state, work):
        keys = (
            "velocity_zero_array_attempts",
            "velocity_zero_array_returns",
            "velocity_values_zeroed",
            "velocity_numeric_bytes_zeroed",
        )
        previous = [work[key] for key in keys]
        try:
            return zero_snapshot(state, work)
        finally:
            for output, key, before in zip(
                (
                    "zero_array_attempts",
                    "zero_array_returns",
                    "zero_values_returned",
                    "zero_bytes_returned",
                ),
                keys,
                previous,
                strict=True,
            ):
                counts[output] += work[key] - before

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bp.NumpyContinuousBCLearner, "__init__", initialized)
        patch.setattr(bp.NumpyContinuousBCLearner, "fork", counted(fork, "fork"))
        patch.setattr(bp.NumpyContinuousBCLearner, "restore_state", counted(restore, "restore"))
        patch.setattr(control, "_zero_velocity_snapshot", zeroed)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", forbidden)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", forbidden)
        patch.setattr(bp.NumpyContinuousBCLearner, "predict", forbidden)
        patch.setattr(bp, "fit_training_standardization", forbidden)
        yield counts
    if os.environ.get("PVF_FIXTURE_LEDGER"):
        with Path(os.environ["PVF_FIXTURE_LEDGER"]).open("x", encoding="utf8") as output:
            json.dump(counts, output, sort_keys=True)
            output.write("\n")
    assert counts["initializations"] == 1 and counts["normal_draws"] == 32
    assert counts["native_fork_attempts"] == counts["native_fork_returns"] == 12
    assert counts["native_restore_attempts"] == counts["native_restore_returns"] == 4
    assert counts["zero_array_attempts"] == 19 and counts["zero_array_returns"] == 18
    assert counts["zero_values_returned"] == 178 and counts["zero_bytes_returned"] == 1424
    assert counts["routing_source_calls"] == 8 and counts["routing_acquisition_calls"] == 7
    assert counts["routing_continuation_calls"] == 19 and counts["routing_fit_calls"] == 17
    assert counts["routing_zero_preparation_calls"] == 7
    assert counts["routing_push_calls"] == counts["routing_pick_calls"] == 0


@pytest.fixture(scope="module")
def owner(ledger):
    source = bp.NumpyContinuousBCLearner(
        bp.ContinuousBCPolicy(
            cube_taskspace_spec(),
            hidden_dim=2,
            seed=17,
            momentum=0.2,
            observation_standardization=ObservationStandardization((0.0,) * 12, (2.0,) * 12, 2),
        )
    )
    source._state = replace(
        source._state,
        velocity=(
            np.full_like(source._state.velocity[0], 0.125),
            np.full_like(source._state.velocity[1], 0.125),
            np.full_like(source._state.velocity[2], 0.125),
            np.full_like(source._state.velocity[3], 0.125),
        ),
        completed_updates=112,
        training_examples=9832,
    )
    return source


@pytest.fixture(scope="module")
def prepared(owner):
    before = owner.snapshot_state()
    result = control.prepare_zero_velocity_bc(owner)
    return result, before


def test_should_clear_only_four_private_velocity_arrays_without_changing_source_or_policy(
    owner, prepared
):
    result, before = prepared
    assert result.report["complete"] is result.report["source_unchanged"] is True
    candidate = result.candidate
    assert candidate is not None and candidate is not owner
    actual = candidate.snapshot_state()
    assert actual.policy == before.policy
    assert (actual.completed_updates, actual.training_examples) == (112, 9832)
    for original, copied in zip(before.parameters, actual.parameters, strict=True):
        np.testing.assert_array_equal(original, copied)
    assert all(np.all(value == 0.0) for value in actual.velocity)
    assert all(np.all(value == 0.125) for value in before.velocity)
    assert not any(
        np.shares_memory(a, b)
        for a in owner._state.parameters + owner._state.velocity
        for b in candidate._state.parameters + candidate._state.velocity
    )
    assert_same(owner.snapshot_state(), before)


def test_should_record_selected_fork_zeroing_restore_costs_and_preserved_global_counts(prepared):
    report = prepared[0].report
    assert report["definition"] == "continuous_private_zero_velocity_bc_v1"
    assert report["initial_counts"] == dict(updates=112, training_rows=9832)
    work = report["work"]
    assert (
        work["source_fork_attempts"]
        == work["source_fork_returns"]
        == work["additional_source_owners"]
        == 1
    )
    assert work["velocity_zero_array_attempts"] == work["velocity_zero_array_returns"] == 4
    assert work["velocity_values_zeroed"] == 38 and work["velocity_numeric_bytes_zeroed"] == 304
    assert work["restore_attempts"] == work["restore_returns"] == 1
    assert work["parameter_momentum_numeric_bytes_per_source"] == 608
    assert work["fits"] == work["parameter_initializations"] == work["seeded_normal_values"] == 0


def test_should_preserve_an_existing_mixed_objective_and_frozen_normalizer(owner):
    source = owner.fork()
    source._policy = replace(source._policy, binary_gripper_axis=3)
    source._state = replace(source._state, policy=source._policy)
    before = source.snapshot_state()
    result = control.prepare_zero_velocity_bc(source)
    assert result.report["complete"] is True and result.candidate is not None
    assert result.candidate.policy.binary_gripper_axis == 3
    assert (
        result.candidate.policy.observation_standardization
        == before.policy.observation_standardization
    )
    assert_same(source.snapshot_state(), before)


@pytest.mark.parametrize("fault", ["type", "role", "callback"])
def test_should_refuse_nontrain_or_invalid_boundary_before_any_preparation_fork(
    owner, monkeypatch, fault
):
    source: Any = owner
    options = {}
    if fault == "type":
        source = object()
    elif fault == "role":
        source = owner.fork()
        source._policy = replace(source._policy, training_role="probe_train")
        source._state = replace(source._state, policy=source._policy)
    else:
        options["check_boundary"] = 1
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forbidden)
    with pytest.raises(ValueError):
        control.prepare_zero_velocity_bc(source, **options)


@pytest.mark.parametrize("fault", ["self", "foreign"])
def test_should_retain_paid_fork_return_before_refusing_an_invalid_private_owner(
    owner, monkeypatch, fault
):
    before = owner.snapshot_state()
    monkeypatch.setattr(owner, "fork", lambda: owner if fault == "self" else object())
    result = control.prepare_zero_velocity_bc(owner)
    assert result.candidate is None and result.report["complete"] is False
    work = result.report["work"]
    assert work["source_fork_attempts"] == work["source_fork_returns"] == 1
    assert (
        work["additional_source_owners"]
        == work["velocity_zero_array_attempts"]
        == work["restore_attempts"]
        == 0
    )
    assert_same(owner.snapshot_state(), before)


def test_should_preserve_partial_zeroing_cost_and_stop_before_restore(owner, monkeypatch):
    original = control.np.zeros_like
    calls: list[int] = []

    def zeroed(value):
        calls.append(value.size)
        if len(calls) == 3:
            raise RuntimeError("third zeroed array unavailable")
        return original(value)

    before = owner.snapshot_state()
    monkeypatch.setattr(control.np, "zeros_like", zeroed)
    result = control.prepare_zero_velocity_bc(owner)
    assert result.candidate is None and result.report["complete"] is False
    work = result.report["work"]
    assert calls == [24, 2, 8]
    assert work["velocity_zero_array_attempts"] == 3 and work["velocity_zero_array_returns"] == 2
    assert work["velocity_values_zeroed"] == 26 and work["velocity_numeric_bytes_zeroed"] == 208
    assert work["restore_attempts"] == 0
    assert_same(owner.snapshot_state(), before)


def test_should_not_allow_restore_argument_mutation_to_redefine_the_zero_intervention(
    owner, monkeypatch
):
    restore = bp.NumpyContinuousBCLearner.restore_state
    captured: list[Any] = []

    def restored(candidate, argument):
        argument.velocity[0][0, 0] = 0.125
        captured.append(candidate)
        return restore(candidate, argument)

    before = owner.snapshot_state()
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "restore_state", restored)
    result = control.prepare_zero_velocity_bc(owner)
    assert result.candidate is None and result.report["complete"] is False
    assert len(captured) == 1 and captured[0]._state.velocity[0][0, 0] == 0.125
    assert (
        result.report["work"]["restore_attempts"] == result.report["work"]["restore_returns"] == 1
    )
    assert result.report["work"]["velocity_zero_array_returns"] == 4
    assert_same(owner.snapshot_state(), before)


@pytest.mark.parametrize("replacement", ["arrays", "policy"])
def test_should_refuse_an_equal_value_original_owner_replacement_after_fork(
    owner, monkeypatch, replacement
):
    source = owner.fork()
    native = source._state
    original_parameters = native.parameters
    original = source.fork
    children: list[Any] = []
    replaced: list[Any] = []

    def forked():
        child = original()
        children.append(child)
        return child

    def boundary():
        if children and not replaced:
            if replacement == "policy":
                source._policy = replace(source._policy)
                source._state = replace(source._state, policy=source._policy)
            else:
                object.__setattr__(
                    source._state,
                    "parameters",
                    tuple(value.copy() for value in source._state.parameters),
                )
                object.__setattr__(
                    source._state,
                    "velocity",
                    tuple(value.copy() for value in source._state.velocity),
                )
            replaced.append(source._state)

    monkeypatch.setattr(source, "fork", forked)
    result = control.prepare_zero_velocity_bc(source, check_boundary=boundary)
    assert replaced and source._state is replaced[0]
    if replacement == "arrays":
        assert source._state is native and source._state.parameters is not original_parameters
    else:
        assert source._state is not native
    assert result.candidate is None and result.report["complete"] is False
    assert result.report["work"]["source_fork_returns"] == 1
    assert result.report["work"]["velocity_zero_array_attempts"] == 0


def test_should_keep_observed_source_mutation_after_a_returned_restore_without_releasing_candidate(
    owner, monkeypatch
):
    source = owner.fork()
    before = source.snapshot_state()
    restore = bp.NumpyContinuousBCLearner.restore_state

    def restored(candidate, argument):
        returned = restore(candidate, argument)
        source._state.parameters[3][0] += 0.125
        return returned

    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "restore_state", restored)
    result = control.prepare_zero_velocity_bc(source)
    assert result.candidate is None and result.report["complete"] is False
    assert source._state.parameters[3][0] == before.parameters[3][0] + 0.125
    assert (
        result.report["work"]["restore_attempts"] == result.report["work"]["restore_returns"] == 1
    )


@pytest.mark.parametrize(
    "options",
    [
        dict(zero_velocity_fit=1),
        dict(zero_velocity_fit=True),
        dict(
            zero_velocity_fit=True,
            profile="bp_mixed",
            longer_fit=True,
            diagnostics_only=True,
            correction_fit=True,
            correction_readouts=True,
        ),
        dict(
            zero_velocity_fit=True,
            profile="bp_mixed_retention",
            diagnostics_only=True,
            correction_fit=True,
            correction_readouts=True,
        ),
        dict(
            zero_velocity_fit=True,
            profile="bp_mixed_retention",
            longer_fit=True,
            correction_fit=True,
            correction_readouts=True,
        ),
        dict(
            zero_velocity_fit=True,
            profile="bp_mixed_retention",
            longer_fit=True,
            diagnostics_only=True,
            correction_fit=True,
        ),
        dict(
            zero_velocity_fit=True,
            profile="bp_mixed_retention",
            longer_fit=True,
            diagnostics_only=True,
            correction_fit=True,
            correction_readouts=True,
            native_endpoint=True,
        ),
        dict(
            zero_velocity_fit=True,
            profile="bp_mixed_retention",
            longer_fit=True,
            diagnostics_only=True,
            correction_fit=True,
            correction_readouts=True,
            training_diagnostics=True,
        ),
    ],
)
def test_should_refuse_invalid_velocity_study_modes_before_any_saved_input(
    tmp_path, monkeypatch, options
):
    for name in (
        "read_reused_control",
        "read_reused_mixed",
        "read_reused_longer",
        "read_reused_retention",
        "read_reused_native",
        "read_reused_corrective",
        "_read_push_study",
        "load_push_correction_inputs",
        "read_push_training",
        "read_joint_acquisition_inputs",
    ):
        monkeypatch.setattr(study, name, forbidden)
    output = tmp_path / "invalid-zero-fit.json"
    with pytest.raises(ValueError):
        study.run_study(output, **options)
    assert not output.exists()


@pytest.mark.parametrize(
    "zero_fault",
    [
        None,
        "preparation",
        "candidate",
        "private",
        "prior",
        "baseline",
        "readout",
        "raw_preparation",
    ],
)
def test_should_route_one_private_weighted_fit_and_preserve_prior_preparation_or_readout_failure(
    tmp_path,
    monkeypatch,
    owner,
    ledger,
    zero_fault,
):
    # Why: reuse the established boundary fixture with literal source tags. No
    # model initialization, native optimizer or actual H64 preparation is invoked.
    source = object.__new__(bp.NumpyContinuousBCLearner)
    source._policy = owner._policy
    source._state = replace(
        owner._state,
        parameters=tuple(value.copy() for value in owner._state.parameters),
        velocity=tuple(value.copy() for value in owner._state.velocity),
        completed_updates=40,
        training_examples=2776,
    )
    run_routing_case(
        tmp_path,
        monkeypatch,
        source,
        ledger,
        None,
        "bp_mixed_retention",
        longer_fit=True,
        diagnostics_only=True,
        correction_fit=True,
        correction_readouts=True,
        zero_velocity_fit=True,
        zero_fault=zero_fault,
    )
