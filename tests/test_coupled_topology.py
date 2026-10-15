"""Independent complete topology/history ownership and reconstruction controls."""

from copy import deepcopy
from dataclasses import asdict, FrozenInstanceError, replace
from typing import Any

import numpy as np
import pytest

from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.coupled_width_pruning import CoupledPruneLineage, prune_coupled_width
from src.core.coupled_topology import (
    CoupledTopologyState,
    capture_coupled_topology,
    restore_coupled_topology,
)


def model(family="pc", hidden=(4, 4)):
    result: CoupledPredictiveCodingNetwork
    if family == "pc":
        result = CoupledPredictiveCodingNetwork(2, hidden, seed=23, precisions=(0.7,) * len(hidden))
    else:
        configuration = CoupledCircadianConfig(0.9, 0.2, 0.7, 0.2)
        result = CoupledCircadianPredictiveCodingNetwork(
            2, hidden, seed=23, precisions=(0.7,) * len(hidden), config=configuration
        )
        snapshot = result.snapshot_state()
        assert snapshot.wake_state is not None
        chemicals = 0.3 + 0.2 * np.arange(hidden[-1], dtype=np.float64)
        gates = np.clip(np.exp(-0.7 * chemicals), 0.2, 1.0)
        result.restore_state(
            replace(
                snapshot,
                wake_state=replace(
                    snapshot.wake_state,
                    chemical_state=chemicals,
                    plasticity_state=gates,
                    wake_updates=4,
                ),
            )
        )
    return result


def history(family="pc"):
    source = model(family)
    first = prune_coupled_width(source, 0, 1)
    second = prune_coupled_width(first.model, 1, 2)
    return source, second.model, (first.lineage, second.lineage)


@pytest.mark.parametrize("family", ("pc", "cpc"))
def test_should_own_and_restore_full_native_geometry_wake_policy_and_ordered_history(family):
    source, current, records = history(family)
    original_source = source.snapshot_state()
    original_current = current.snapshot_state()
    state = capture_coupled_topology(current, prune_lineage=records)
    assert type(state) is CoupledTopologyState
    assert type(state.format_version) is int and state.format_version == 1
    assert state.family == family and state.input_dim == 2
    assert state.hidden_dims == (3, 3) and state.prune_lineage == records
    np.testing.assert_equal(asdict(state.snapshot), asdict(original_current))
    restored = restore_coupled_topology(state)
    assert type(restored) is type(current) and restored is not current
    assert restored.hidden_dims == (3, 3) and restored.precisions == (0.7, 0.7)
    np.testing.assert_equal(asdict(restored.snapshot_state()), asdict(original_current))
    if type(restored) is CoupledCircadianPredictiveCodingNetwork:
        assert type(current) is CoupledCircadianPredictiveCodingNetwork
        assert restored.config == current.config and restored.config is not current.config
        assert restored.wake_updates == 4
        object.__setattr__(current.config, "chemical_buildup_rate", 99.0)
        assert restored.config.chemical_buildup_rate == 0.2
    state.snapshot.parameters.output_bias.fill(99)
    np.testing.assert_equal(asdict(restored.snapshot_state()), asdict(original_current))
    np.testing.assert_equal(asdict(source.snapshot_state()), asdict(original_source))
    with pytest.raises(FrozenInstanceError):
        setattr(state, "hidden_dims", (1,))


def chain(length):
    return tuple(
        CoupledPruneLineage(1, "pc", 2, (3 + length - index,), (1.0,), 0, 0)
        for index in range(length)
    )


def test_should_accept_64_consecutive_records_and_distinguish_same_geometry_histories():
    native = CoupledPredictiveCodingNetwork(2, (3,), seed=23)
    with_history = capture_coupled_topology(native, prune_lineage=chain(64))
    plain = capture_coupled_topology(native)
    assert len(with_history.prune_lineage) == 64 and plain.prune_lineage == ()
    np.testing.assert_equal(asdict(with_history.snapshot), asdict(plain.snapshot))
    restored = restore_coupled_topology(with_history)
    assert restored.hidden_dims == (3,)
    assert capture_coupled_topology(
        restored, prune_lineage=with_history.prune_lineage
    ).prune_lineage == chain(64)


@pytest.mark.parametrize(
    "kind",
    (
        "list",
        "too_many",
        "nonconsecutive",
        "wrong_family",
        "wrong_input",
        "wrong_precision",
        "wrong_final",
        "reversed",
    ),
)
def test_should_reject_invalid_history_without_mutating_source(kind):
    _, native, records = history()
    before = native.snapshot_state()
    invalid: Any = records
    if kind == "list":
        invalid = list(records)
    elif kind == "too_many":
        invalid = chain(65)
    elif kind == "nonconsecutive":
        invalid = (records[0], replace(records[1], source_hidden_dims=(4, 4)))
    elif kind == "wrong_family":
        invalid = (replace(records[0], family="cpc"), records[1])
    elif kind == "wrong_input":
        invalid = (replace(records[0], input_dim=3), records[1])
    elif kind == "wrong_precision":
        invalid = (replace(records[0], precisions=(1.0, 1.0)), records[1])
    elif kind == "wrong_final":
        invalid = records[:1]
    else:
        invalid = tuple(reversed(records))
    with pytest.raises(ValueError):
        capture_coupled_topology(native, prune_lineage=invalid)
    np.testing.assert_equal(asdict(native.snapshot_state()), asdict(before))


@pytest.mark.parametrize(
    "changes",
    (
        {"format_version": True},
        {"format_version": 2},
        {"family": "cpc"},
        {"family": "other"},
        {"input_dim": True},
        {"input_dim": 3},
        {"hidden_dims": [3, 3]},
        {"hidden_dims": (3,)},
        {"hidden_dims": (4, 3)},
        {"prune_lineage": []},
    ),
)
def test_should_refuse_malformed_wrapper_metadata(changes: Any):
    _, native, records = history()
    state = capture_coupled_topology(native, prune_lineage=records)
    with pytest.raises(ValueError):
        restore_coupled_topology(replace(state, **changes))


@pytest.mark.parametrize(
    "kind", ("snapshot_format", "removed_nan", "chemistry", "gate", "wake_count")
)
def test_should_validate_complete_native_payload_on_restore(kind):
    native = model("cpc")
    state = capture_coupled_topology(native)
    saved = deepcopy(state)
    snapshot = state.snapshot
    wake = snapshot.wake_state
    assert wake is not None
    if kind == "snapshot_format":
        snapshot = replace(snapshot, format_version=True)
    elif kind == "removed_nan":
        snapshot.parameters.hidden_weights[0][0, 0] = np.nan
    elif kind == "chemistry":
        wake.chemical_state[0] = -1
    elif kind == "gate":
        wake.plasticity_state[0] = 0
    else:
        snapshot = replace(snapshot, wake_state=replace(wake, wake_updates=True))
    with pytest.raises(ValueError):
        restore_coupled_topology(replace(state, snapshot=snapshot))
    np.testing.assert_equal(asdict(native.snapshot_state()), asdict(saved.snapshot))


@pytest.mark.parametrize("dimensions", ((2,) * 17, (1001, 1001)))
def test_should_validate_geometry_before_reconstruction_allocation(monkeypatch, dimensions):
    state = capture_coupled_topology(model())

    def forbidden_constructor(*args, **kwargs):
        raise AssertionError("invalid geometry allocated native model")

    monkeypatch.setattr(CoupledPredictiveCodingNetwork, "__init__", forbidden_constructor)
    with pytest.raises(ValueError):
        restore_coupled_topology(replace(state, hidden_dims=dimensions))


def test_should_refuse_foreign_state_and_model_types():
    invalid: Any
    for invalid in (None, {}, "native"):
        with pytest.raises(ValueError):
            restore_coupled_topology(invalid)
        with pytest.raises(ValueError):
            capture_coupled_topology(invalid)
