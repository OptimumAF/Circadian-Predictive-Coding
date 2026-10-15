"""Full CPC snapshots preserve continuation and refuse incompatible state atomically."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.coupled_energy import CoupledParameters
from src.core.coupled_predictive_coding import (
    CoupledPredictiveCodingNetwork,
    CoupledSnapshot,
)


def _model(seed: int = 3) -> CoupledCircadianPredictiveCodingNetwork:
    return CoupledCircadianPredictiveCodingNetwork(
        2,
        (3, 2),
        seed,
        (0.8, 1.2),
        config=CoupledCircadianConfig(0.8, 0.7, 0.9, 0.1),
    )


def _step(model: CoupledCircadianPredictiveCodingNetwork) -> float:
    features = np.array([[0.1, -0.2], [0.5, 0.3], [-0.4, 0.8]])
    targets = np.array([[0.0], [1.0], [1.0]])
    return model.train_epoch(features, targets, 0.04, 3, 0.03).energy


def _arrays(parameters: CoupledParameters) -> tuple[np.ndarray, ...]:
    return (
        *parameters.hidden_weights,
        *parameters.hidden_biases,
        parameters.output_weight,
        parameters.output_bias,
    )


def _assert_state(actual: CoupledSnapshot, expected: CoupledSnapshot) -> None:
    assert actual.format_version == expected.format_version
    assert actual.precisions == expected.precisions
    for left, right in zip(_arrays(actual.parameters), _arrays(expected.parameters)):
        np.testing.assert_array_equal(left, right)
    left_wake, right_wake = actual.wake_state, expected.wake_state
    assert left_wake is not None and right_wake is not None
    assert left_wake.chemical_decay == right_wake.chemical_decay
    assert left_wake.chemical_buildup_rate == right_wake.chemical_buildup_rate
    assert left_wake.plasticity_sensitivity == right_wake.plasticity_sensitivity
    assert left_wake.min_plasticity == right_wake.min_plasticity
    assert left_wake.wake_updates == right_wake.wake_updates
    np.testing.assert_array_equal(left_wake.chemical_state, right_wake.chemical_state)
    np.testing.assert_array_equal(left_wake.plasticity_state, right_wake.plasticity_state)


def test_full_snapshot_continues_the_active_gated_trajectory() -> None:
    source, restored = _model(), _model(seed=7)
    _step(source)
    _step(source)
    snapshot = source.snapshot_state()
    assert snapshot.wake_state is not None
    assert np.any(snapshot.wake_state.plasticity_state < 1)
    restored.restore_state(snapshot)
    _assert_state(restored.snapshot_state(), snapshot)
    for _ in range(4):
        assert _step(source) == _step(restored)
        _assert_state(restored.snapshot_state(), source.snapshot_state())


def test_snapshot_and_restored_state_own_all_mutable_arrays() -> None:
    source, restored = _model(), _model(seed=8)
    _step(source)
    saved = source.snapshot_state()
    expected = source.snapshot_state()
    restored.restore_state(saved)
    assert saved.wake_state is not None
    for array in _arrays(saved.parameters):
        array.fill(42)
    saved.wake_state.chemical_state.fill(42)
    saved.wake_state.plasticity_state.fill(42)
    # Frozen outer records do not make their nested arrays immutable. Both APIs
    # must detach their arrays rather than relying on the dataclass's freeze.
    object.__setattr__(saved.wake_state, "wake_updates", 999)
    _assert_state(source.snapshot_state(), expected)
    _assert_state(restored.snapshot_state(), expected)
    fresh = restored.snapshot_state()
    assert fresh.wake_state is not None and expected.wake_state is not None
    for left, right in zip(_arrays(fresh.parameters), _arrays(expected.parameters)):
        assert not np.shares_memory(left, right)
    assert not np.shares_memory(fresh.wake_state.chemical_state, expected.wake_state.chemical_state)
    assert not np.shares_memory(
        fresh.wake_state.plasticity_state, expected.wake_state.plasticity_state
    )


def test_full_restore_accepts_pristine_zero_wake_and_finite_saturated_chemistry() -> None:
    model = _model()
    pristine = model.snapshot_state()
    _step(model)
    model.restore_state(pristine)
    _assert_state(model.snapshot_state(), pristine)
    assert pristine.wake_state is not None
    saturated = replace(
        pristine,
        wake_state=replace(
            pristine.wake_state,
            chemical_state=np.full(2, 1e308),
            plasticity_state=np.full(2, model.config.min_plasticity),
            wake_updates=5,
        ),
    )
    with np.errstate(over="raise", invalid="raise", under="raise"):
        model.restore_state(saturated)
    _assert_state(model.snapshot_state(), saturated)


def test_parameter_only_restore_keeps_wake_state_after_full_restore() -> None:
    source, restored = _model(), _model(seed=7)
    initial_parameters = source.snapshot_parameters()
    _step(source)
    restored.restore_state(source.snapshot_state())
    before = restored.snapshot_state()
    restored.restore_parameters(initial_parameters)
    _assert_state(restored.snapshot_state(), replace(before, parameters=initial_parameters))


@pytest.mark.parametrize(
    "field,value",
    [
        ("wake_updates", True),
        ("wake_updates", -1),
        ("wake_updates", 1.5),
        ("wake_updates", np.int64(1)),
        ("chemical_decay", True),
        ("chemical_decay", float("nan")),
        ("chemical_buildup_rate", 0.5),
        ("plasticity_sensitivity", 0.5),
        ("min_plasticity", 0.2),
        ("chemical_state", np.array([np.nan, 0.0])),
        ("chemical_state", np.array([np.inf, 0.0])),
        ("chemical_state", np.array([-0.1, 0.0])),
        ("chemical_state", np.zeros((1, 2))),
        ("chemical_state", np.zeros(2, dtype=np.float32)),
        ("chemical_state", [0.0, 0.0]),
        ("plasticity_state", np.array([np.nan, 1.0])),
        ("plasticity_state", np.array([np.inf, 1.0])),
        ("plasticity_state", np.array([-1.0, 1.0])),
        ("plasticity_state", np.array([1.1, 1.0])),
        ("plasticity_state", np.ones(3)),
        ("plasticity_state", np.ones(2, dtype=np.float32)),
        ("plasticity_state", np.ones(2)),
    ],
)
def test_malformed_wake_state_refuses_without_changing_any_state(field: str, value: Any) -> None:
    model = _model()
    _step(model)
    before = model.snapshot_state()
    assert before.wake_state is not None
    malformed = replace(before, wake_state=replace(before.wake_state, **{field: value}))
    with pytest.raises(ValueError):
        model.restore_state(malformed)
    _assert_state(model.snapshot_state(), before)


@pytest.mark.parametrize("kind", ["chemistry", "gates", "both"])
def test_zero_wake_invariants_are_checked_even_when_the_gate_formula_agrees(kind: str) -> None:
    model = _model()
    _step(model)
    before = model.snapshot_state()
    assert before.wake_state is not None
    chemicals = np.zeros(2) if kind == "gates" else np.ones(2)
    gates = (
        np.ones(2)
        if kind == "chemistry"
        else np.exp(-model.config.plasticity_sensitivity * chemicals)
    )
    if kind == "gates":
        gates[:] = 0.5
    malformed = replace(
        before,
        wake_state=replace(
            before.wake_state,
            chemical_state=chemicals,
            plasticity_state=gates,
            wake_updates=0,
        ),
    )
    with pytest.raises(ValueError):
        model.restore_state(malformed)
    _assert_state(model.snapshot_state(), before)


@pytest.mark.parametrize(
    "kind",
    [
        "unknown",
        "version",
        "bool_version",
        "precision",
        "bool_precision",
        "parameters",
        "missing_wake",
        "wrong_wake",
    ],
)
def test_incompatible_snapshot_metadata_or_family_is_failure_atomic(kind: str) -> None:
    model = _model()
    _step(model)
    before = model.snapshot_state()
    wrong_wake: Any = object()
    variants: dict[str, Any] = {
        "unknown": object(),
        "version": replace(before, format_version=2),
        "bool_version": replace(before, format_version=True),
        "precision": replace(before, precisions=(0.9, 1.2)),
        "bool_precision": replace(before, precisions=(True, 1.2)),
        "parameters": replace(
            before, parameters=CoupledPredictiveCodingNetwork(2, (2,), 0).snapshot_parameters()
        ),
        "missing_wake": replace(before, wake_state=None),
        "wrong_wake": replace(before, wake_state=wrong_wake),
    }
    with pytest.raises(ValueError):
        model.restore_state(variants[kind])
    _assert_state(model.snapshot_state(), before)


def test_pc_refuses_full_cpc_state_and_cpc_refuses_parameter_only_pc_state() -> None:
    cpc = _model()
    pc = CoupledPredictiveCodingNetwork(2, (3, 2), 4, (0.8, 1.2))
    _step(cpc)
    before = cpc.snapshot_state()
    with pytest.raises(ValueError):
        cpc.restore_state(pc.snapshot_state())
    with pytest.raises(ValueError):
        pc.restore_state(before)
    _assert_state(cpc.snapshot_state(), before)


def test_same_policy_is_required_even_for_a_compatible_topology() -> None:
    model = _model()
    other = CoupledCircadianPredictiveCodingNetwork(2, (3, 2), 3, (0.8, 1.2))
    _step(model)
    before = model.snapshot_state()
    with pytest.raises(ValueError):
        model.restore_state(other.snapshot_state())
    _assert_state(model.snapshot_state(), before)
