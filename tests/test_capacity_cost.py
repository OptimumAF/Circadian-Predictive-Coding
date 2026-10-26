"""Dense parameter counts and pure complete-state array measurements.

Envelope fixtures contain schema-valid values without constructing native models;
measurement must inspect data without running or restoring a learner.
"""

from copy import deepcopy
import pickle

import numpy as np
import pytest

from src.adapters.numpy_capacity import measure_numpy_capacity
from src.core.capacity_cost import CapacityUsage, count_dense_parameters
from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.coupled_energy import CoupledParameters
from src.core.coupled_pc_scaling import CoupledScalingConfig, CoupledScalingState
from src.core.coupled_predictive_coding import (
    CoupledPredictiveCodingNetwork,
    CoupledSnapshot,
    CoupledWakeState,
)
from src.core.coupled_scaling_topology import CoupledScalingTopologyState
from src.core.coupled_structural_state import CoupledStructuralState
from src.core.coupled_topology import CoupledTopologyState
from src.core.coupled_unit_history import initial_unit_history
from src.core.coupled_unit_renewal import RenewalPolicy, initial_renewal_history
from src.core.matched_backprop import MatchedBackpropNetwork, MatchedBackpropSnapshot
from src.core.renewable_coupled_predictive_coding import RenewableCoupledSnapshot
from src.core.tracked_coupled_predictive_coding import TrackedCoupledSnapshot


def parameters(width=3):
    return CoupledParameters(
        (np.zeros((2, width)),),
        (np.zeros((1, width)),),
        np.zeros((width, 1)),
        np.zeros((1, 1)),
    )


def record(kind, **values):
    # These immutable data constructors normally reconstruct validation models.
    # Borrowed-state measurement must validate their complete values itself.
    value = object.__new__(kind)
    for name, item in values.items():
        object.__setattr__(value, name, item)
    return value


def snapshot(kind):
    p = parameters()
    wake = CoupledWakeState(0.9, 0.2, 0.5, 0.2, np.zeros(3), np.ones(3), 0)
    if kind == "bp":
        return MatchedBackpropSnapshot(1, p), 0
    if kind == "pc":
        return CoupledSnapshot(1, p, (1.0,)), 0
    if kind == "cpc":
        return CoupledSnapshot(1, p, (1.0,), wake), 48
    if kind == "tracked":
        return TrackedCoupledSnapshot(2, p, (1.0,), None, initial_unit_history(3)), 72
    if kind == "renewable":
        return RenewableCoupledSnapshot(
            3, p, (1.0,), None, initial_renewal_history(3), RenewalPolicy(1, 1.0, 1)
        ), 72
    config = CoupledScalingConfig(
        "mean_field", 2, 3, 1, 1.5, (1.0,), CoupledCircadianConfig(0.9, 0.2, 0.5, 0.2)
    )
    raw = record(CoupledScalingState, config=config, raw_parameters=p, wake_state=wake)
    if kind == "raw":
        return raw, 48
    if kind == "structural":
        return record(
            CoupledStructuralState,
            format_version=1,
            origin_config=config,
            raw_state=raw,
            transitions=(),
        ), 48
    if kind == "raw_topology":
        return record(
            CoupledScalingTopologyState, format_version=1, raw_state=raw, prune_lineage=()
        ), 48
    assert kind == "topology"
    return record(
        CoupledTopologyState,
        format_version=1,
        family="cpc",
        input_dim=2,
        hidden_dims=(3,),
        snapshot=CoupledSnapshot(1, p, (1.0,), wake),
        prune_lineage=(),
    ), 48


def forbidden(*args, **kwargs):
    raise AssertionError("capacity measurement must not execute a native model")


@pytest.mark.parametrize("widths", [(2,), (4,), (3, 2)])
def test_should_count_dense_binary_weights_and_biases_independently(widths):
    previous = 2
    expected = 0
    for width in widths:
        expected += previous * width + width
        previous = width
    expected += previous + 1
    assert count_dense_parameters(2, widths) == expected


@pytest.mark.parametrize(
    "dimension,widths", [(True, (2,)), (0, (2,)), (2, [2]), (2, (True,)), (2, ())]
)
def test_should_refuse_invalid_dense_geometry(dimension, widths):
    with pytest.raises(ValueError):
        count_dense_parameters(dimension, widths)


@pytest.mark.parametrize(
    "values", [(True, 8, 0), (0, 0, 0), (13, 103, 0), (13, 104, -1), (13, 104, 1.5)]
)
def test_should_validate_exact_usage_units(values):
    with pytest.raises(ValueError):
        CapacityUsage(*values)


@pytest.mark.parametrize(
    "kind",
    ["bp", "pc", "cpc", "tracked", "renewable", "raw", "structural", "raw_topology", "topology"],
)
def test_should_measure_complete_native_arrays_without_models_or_mutation(monkeypatch, kind):
    state, auxiliary_bytes = snapshot(kind)
    before = pickle.dumps(state)
    for native in (
        CoupledPredictiveCodingNetwork,
        CoupledCircadianPredictiveCodingNetwork,
        MatchedBackpropNetwork,
    ):
        for method in (
            "__init__",
            "train_epoch",
            "predict_proba",
            "snapshot_state",
            "restore_state",
        ):
            monkeypatch.setattr(native, method, forbidden)
    result = measure_numpy_capacity(state)
    assert result == CapacityUsage(13, 104, auxiliary_bytes)
    assert result.total_array_bytes == 104 + auxiliary_bytes
    assert pickle.dumps(state) == before
    detached = deepcopy(result)
    assert detached == result


@pytest.mark.parametrize("fault", ["dtype", "shape", "nonfinite", "clock", "extra", "history"])
def test_should_refuse_incomplete_or_corrupt_state_without_silent_omission(fault):
    state, _ = snapshot("tracked" if fault == "history" else "cpc")
    if fault == "dtype":
        object.__setattr__(state.parameters, "output_weight", np.zeros((3, 1), dtype=np.float32))
    elif fault == "shape":
        object.__setattr__(state.parameters, "output_weight", np.zeros((2, 1)))
    elif fault == "nonfinite":
        state.parameters.output_weight[0, 0] = np.nan
    elif fault == "clock":
        object.__setattr__(state.wake_state, "wake_updates", True)
    elif fault == "extra":
        object.__setattr__(state, "unreported_array", np.zeros(1))
    else:
        state.unit_history.update_age[0] = 1
    with pytest.raises(ValueError):
        measure_numpy_capacity(state)
    if fault == "extra":
        with pytest.raises(ValueError):
            measure_numpy_capacity(object())
