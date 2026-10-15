"""Owned raw topology envelopes and bounded ordered configuration histories."""

from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_pc_scaling import CoupledScalingConfig, initialize_coupled_scaling
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.coupled_scaling_pruning import CoupledScalingPruneLineage, prune_coupled_scaling_width
from src.core.coupled_scaling_topology import (
    CoupledScalingTopologyState,
    copy_coupled_scaling_topology,
)


def raw(family="pc", width=4):
    policy = None if family == "pc" else CoupledCircadianConfig(0.9, 0.2, 0.7, 0.2)
    return initialize_coupled_scaling(
        CoupledScalingConfig(
            "mean_field", 2, width, 2, 2.0, precisions=(0.7, 1.3), circadian=policy
        ),
        41,
    )


def history(family="pc"):
    source = raw(family)
    a = prune_coupled_scaling_width(source, (1, 2))
    b = prune_coupled_scaling_width(a.state, (0, 1))
    return source, b.state, (a.lineage, b.lineage)


def chain(config, length):
    return tuple(
        CoupledScalingPruneLineage(1, replace(config, width=config.width + length - i), (0, 0))
        for i in range(length)
    )


@pytest.mark.parametrize("family", ("pc", "active"))
def test_should_own_complete_raw_arrays_wake_policy_and_each_history_configuration(family):
    source, current, records = history(family)
    expected = asdict(current)
    envelope = CoupledScalingTopologyState(1, current, records)
    cloned = copy_coupled_scaling_topology(envelope)
    assert cloned is not envelope and cloned.prune_lineage == records
    np.testing.assert_equal(asdict(cloned.raw_state), expected)
    current.raw_parameters.output_bias[:] = 99
    envelope.raw_state.raw_parameters.hidden_weights[0][:] = 99
    object.__setattr__(records[0].source_config, "gamma0", 3.0)
    if family == "active":
        assert current.wake_state is not None
        current.wake_state.chemical_state[:] = 99
        assert current.config.circadian is not None
        object.__setattr__(current.config.circadian, "min_plasticity", 1.0)
    np.testing.assert_equal(asdict(cloned.raw_state), expected)
    assert cloned.prune_lineage[0].source_config.gamma0 == 2.0
    np.testing.assert_array_equal(source.raw_parameters.output_bias, 0)


def test_should_keep_empty_shorter_and64_record_histories_as_distinct_valid_metadata():
    current = raw(width=3)
    full = CoupledScalingTopologyState(1, current, chain(current.config, 64))
    plain = CoupledScalingTopologyState(1, current)
    short = CoupledScalingTopologyState(1, current, full.prune_lineage[-1:])
    assert (
        len(full.prune_lineage) == 64
        and len(short.prune_lineage) == 1
        and plain.prune_lineage == ()
    )
    np.testing.assert_equal(asdict(full.raw_state), asdict(plain.raw_state))
    np.testing.assert_equal(asdict(copy_coupled_scaling_topology(short)), asdict(short))


@pytest.mark.parametrize(
    "kind",
    (
        "list",
        "too_many",
        "reversed",
        "gap",
        "input",
        "gamma",
        "scheme",
        "precision",
        "policy",
        "wrong_final",
    ),
)
def test_should_refuse_nonconsecutive_or_incompatible_complete_configuration_history(kind):
    _, current, records = history()
    invalid: Any = records
    if kind == "list":
        invalid = list(records)
    elif kind == "too_many":
        invalid = chain(current.config, 65)
    elif kind == "reversed":
        invalid = records[::-1]
    elif kind == "gap":
        invalid = (
            records[0],
            replace(records[1], source_config=replace(records[1].source_config, width=4)),
        )
    elif kind == "wrong_final":
        invalid = records[:1]
    else:
        changes: dict[str, dict[str, Any]] = {
            "input": {"input_dim": 3},
            "gamma": {"gamma0": 3.0},
            "scheme": {"scheme": "fan_in", "gamma0": 1.0},
            "precision": {"precisions": (1.0, 1.0)},
            "policy": {"circadian": CoupledCircadianConfig()},
        }
        invalid = (
            replace(records[0], source_config=replace(records[0].source_config, **changes[kind])),
            records[1],
        )
    before = asdict(current)
    with pytest.raises(ValueError):
        CoupledScalingTopologyState(1, current, invalid)
    np.testing.assert_equal(asdict(current), before)


@pytest.mark.parametrize(
    "invalid",
    ("format_bool", "format_version", "raw_family", "nan", "dtype", "wake", "config", "history"),
)
def test_should_revalidate_mutated_full_envelope_before_copy(invalid):
    envelope = CoupledScalingTopologyState(1, raw("active"))
    source = deepcopy(envelope)
    if invalid == "format_bool":
        object.__setattr__(envelope, "format_version", True)
    elif invalid == "format_version":
        object.__setattr__(envelope, "format_version", 2)
    elif invalid == "raw_family":
        object.__setattr__(envelope, "raw_state", {})
    elif invalid == "nan":
        envelope.raw_state.raw_parameters.output_bias[:] = np.nan
    elif invalid == "dtype":
        object.__setattr__(
            envelope.raw_state.raw_parameters, "output_bias", np.zeros((1, 1), dtype=np.float32)
        )
    elif invalid == "wake":
        assert envelope.raw_state.wake_state is not None
        envelope.raw_state.wake_state.plasticity_state[:] = 0
    elif invalid == "config":
        object.__setattr__(envelope.raw_state.config, "width", True)
    else:
        object.__setattr__(envelope, "prune_lineage", ("invalid",))
    with pytest.raises(ValueError):
        copy_coupled_scaling_topology(envelope)
    np.testing.assert_array_equal(source.raw_state.raw_parameters.output_bias, 0)


def test_should_refuse_overlong_history_before_reconstruction_or_payload_copy(monkeypatch):
    current = raw()

    def forbidden(*args, **kwargs):
        raise AssertionError("allocated native model before history cap")

    monkeypatch.setattr(CoupledPredictiveCodingNetwork, "__init__", forbidden)
    with pytest.raises(ValueError):
        invalid: Any = (None,) * 65
        CoupledScalingTopologyState(1, current, invalid)


@pytest.mark.parametrize("changes", ({"width": 1001}, {"hidden_layers": 17}))
def test_should_bound_raw_geometry_before_reconstruction_allocation(monkeypatch, changes):
    current = raw()
    for name, value in changes.items():
        object.__setattr__(current.config, name, value)

    def forbidden(*args, **kwargs):
        raise AssertionError("allocated invalid geometry")

    monkeypatch.setattr(CoupledPredictiveCodingNetwork, "__init__", forbidden)
    with pytest.raises(ValueError):
        CoupledScalingTopologyState(1, current)
