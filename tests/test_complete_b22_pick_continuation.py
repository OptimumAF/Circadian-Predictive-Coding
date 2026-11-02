"""Literal restoration and paid-prefix contracts; no fit, prediction or SDK."""

from dataclasses import asdict
import runpy
from types import SimpleNamespace

import numpy as np
import pytest

from src.adapters import numpy_continuous_bc as bc
from src.core.cube_taskspace_state import cube_taskspace_spec


@pytest.fixture
def continuation():
    return runpy.run_path(
        "artifacts/runs/complete-b22-neural-study-20261101/pick-continuation/run.py"
    )


@pytest.fixture
def record():
    shapes = ((12, 64), (64,), (64, 4), (4,))
    return dict(
        policy=asdict(bc.ContinuousBCPolicy(cube_taskspace_spec(), binary_gripper_axis=3)),
        parameters=[np.zeros(shape).tolist() for shape in shapes],
        velocity=[np.ones(shape).tolist() for shape in shapes],
        completed_updates=64,
        training_examples=14080,
    )


def test_should_restore_full_owned_state_without_initialization_or_rng(
    continuation, record, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("restoration attempted initialization/training/prediction/RNG")

    monkeypatch.setattr(np.random, "default_rng", forbidden)
    for name in ("__init__", "train_batch", "predict", "restore_state"):
        monkeypatch.setattr(bc.NumpyContinuousBCLearner, name, forbidden)
    owner = continuation["restore"](record, bc, np, cube_taskspace_spec())
    assert type(owner) is bc.NumpyContinuousBCLearner
    assert continuation["state_record"](owner.snapshot_state()) == record
    record["parameters"][0][0][0] = 7.0
    record["velocity"][0][0][0] = 9.0
    assert owner._state.parameters[0][0, 0] == 0.0
    assert owner._state.velocity[0][0, 0] == 1.0


def test_should_refuse_changed_objective(continuation, record):
    record["policy"]["binary_gripper_axis"] = None
    with pytest.raises(ValueError, match="policy differs"):
        continuation["restore"](record, bc, np, cube_taskspace_spec())


def test_should_refuse_counter_boolean(continuation, record):
    record["completed_updates"] = True
    with pytest.raises(ValueError, match="counters differ"):
        continuation["restore"](record, bc, np, cube_taskspace_spec())


def test_should_refuse_wrong_momentum_shape(continuation, record):
    record["velocity"][0] = [[0.0]]
    with pytest.raises(ValueError):
        continuation["restore"](record, bc, np, cube_taskspace_spec())


def test_should_refuse_nonfinite_parameters(continuation, record):
    record["parameters"][0][0][0] = float("nan")
    with pytest.raises(ValueError):
        continuation["restore"](record, bc, np, cube_taskspace_spec())


def test_should_refuse_equal_replacement_and_changed_prior_evidence(continuation):
    result = dict(complete=True, optimizer_updates=0)
    case = dict(seed=17, result=result)
    cases = [case]
    seals = [(case, result, continuation["encoded"](case))]
    continuation["require_paid"](cases, cases, seals)
    with pytest.raises(RuntimeError, match="holder replaced"):
        continuation["require_paid"](list(cases), cases, seals)
    case["result"] = dict(result)
    with pytest.raises(RuntimeError, match="evidence changed"):
        continuation["require_paid"](cases, cases, seals)
    case["result"] = result
    result["optimizer_updates"] = 1
    with pytest.raises(RuntimeError, match="evidence changed"):
        continuation["require_paid"](cases, cases, seals)


def test_should_accept_complete_zero_but_refuse_missing_update_proof(continuation):
    result = dict(
        complete=True,
        optimizer_updates=0,
        seed=17,
        role="train",
        frozen_state_exact=True,
        original_owner_exact=True,
        action_history_exact=True,
        prediction_attempts=50,
        prediction_returns=50,
        any_success=False,
        sparse_return=0.0,
    )
    continuation["require_pick"](result, 17)
    result.pop("optimizer_updates")
    with pytest.raises(RuntimeError, match="optimizer contract"):
        continuation["require_pick"](result, 17)
    result["optimizer_updates"] = False
    with pytest.raises(RuntimeError, match="optimizer contract"):
        continuation["require_pick"](result, 17)


def test_should_preserve_lost_return_and_allow_cleanup_after_stop(continuation, monkeypatch):
    clock = SimpleNamespace(value=0.0)
    monkeypatch.setattr(continuation["time"], "monotonic", lambda: clock.value)
    meter = continuation["Meter"](0, SimpleNamespace(peak_bytes=1024))

    def failed():
        raise ValueError("lost")

    owner = SimpleNamespace(step=failed, close=lambda: None)
    continuation["wrap"](meter, owner, "step", "controls", returned="control_returns")
    continuation["wrap"](meter, owner, "close", "closes", returned="close_returns", cleanup=True)
    with pytest.raises(ValueError):
        owner.step()
    assert meter.counts["controls"] == 1 and meter.counts["control_returns"] == 0
    clock.value = 45.0
    with pytest.raises(RuntimeError):
        meter.check()
    owner.close()
    assert meter.counts["closes"] == meter.counts["close_returns"] == 1
    clock.value = 0.0
    with pytest.raises(RuntimeError):
        meter.check()
