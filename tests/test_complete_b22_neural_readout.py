"""One full H64 seven-group readout, literal state and capped diagnostics only."""

from typing import cast

import numpy as np

from scripts import push_cube_training_readouts as readouts
from src.adapters import numpy_continuous_bc as bp
from src.adapters.continuous_learnability_modes import same_continuous_state
from src.core import continuous_bc as core
from src.core.continuous_bc import Parameters
from src.core.cube_taskspace_state import cube_taskspace_spec
from test_complete_b22_neural import groups

WORK = dict(forwards=0, forward_rows=0, forward_returns=0, svds=0, svd_returns=0)


def test_should_complete_actual_h64_seven_group_readout_within_phase_payload_bound(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("H64 readout must not initialize/train/solve")

    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "__init__", forbidden)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "train_batch", forbidden)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", forbidden)
    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    forward, svd = core._forward, np.linalg.svd

    def forwarded(x, *args, **kwargs):
        WORK["forwards"] += 1
        WORK["forward_rows"] += len(x)
        assert WORK["forwards"] <= 32 and WORK["forward_rows"] <= 1024
        result = forward(x, *args, **kwargs)
        WORK["forward_returns"] += 1
        return result

    def decomposed(*args, **kwargs):
        WORK["svds"] += 1
        assert WORK["svds"] <= 16
        result = svd(*args, **kwargs)
        WORK["svd_returns"] += 1
        return result

    monkeypatch.setattr(core, "_forward", forwarded)
    monkeypatch.setattr(np.linalg, "svd", decomposed)
    owner = object.__new__(bp.NumpyContinuousBCLearner)
    owner._policy = bp.ContinuousBCPolicy(
        cube_taskspace_spec(), hidden_dim=64, binary_gripper_axis=3
    )
    parameters = (np.zeros((12, 64)), np.zeros(64), np.zeros((64, 4)), np.zeros(4))
    owner._state = bp.ContinuousBCSnapshot(
        owner._policy,
        parameters,
        cast(Parameters, tuple(np.zeros_like(value) for value in parameters)),
        0,
        0,
    )
    before = owner.snapshot_state()
    report = readouts.run_training_readout(
        owner,
        groups(),
        label="complete220-H64",
        guard=lambda: None,
        corrective=True,
        aggregated=True,
        complete_b22=True,
    )
    assert report["complete"] is True, report
    assert len(report["groups"]) == 7
    assert [entry["rows"] for entry in report["groups"].values()] == [50, 50, 18, 15, 50, 15, 22]
    assert report["work"]["conditioning_returns"] == report["work"]["residual_returns"] == 7
    assert same_continuous_state(before, owner.snapshot_state())
    assert WORK["forwards"] == WORK["forward_returns"]
    assert WORK["svds"] == WORK["svd_returns"]
