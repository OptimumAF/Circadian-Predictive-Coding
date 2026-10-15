"""Configuration reachability and owned full raw tagged structural metadata."""

from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_pc_scaling import CoupledScalingConfig, initialize_coupled_scaling
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.coupled_scaling_pruning import prune_coupled_scaling_width
from src.core.coupled_scaling_structure import (
    split_coupled_scaling_width,
    merge_coupled_scaling_width,
)
from src.core.coupled_structural_state import (
    MAX_STRUCTURAL_TRANSITIONS,
    CoupledStructuralState,
    CoupledStructuralTransition,
    copy_coupled_structural_state,
)


def raw(family="pc", **changes):
    config: dict[str, Any] = dict(
        scheme="mean_field",
        input_dim=2,
        width=3,
        hidden_layers=2,
        gamma0=2.0,
        precisions=(0.7, 1.3),
        circadian=None if family == "pc" else CoupledCircadianConfig(),
    )
    config.update(changes)
    return initialize_coupled_scaling(CoupledScalingConfig(**config), 41)


def sequence(family="pc"):
    source = raw(family)
    a = split_coupled_scaling_width(source)
    b = merge_coupled_scaling_width(a.state, a.lineage)
    c = prune_coupled_scaling_width(b.state, (1, 1))
    history = (
        CoupledStructuralTransition("split", a.lineage),
        CoupledStructuralTransition("merge", a.lineage),
        CoupledStructuralTransition("prune", c.lineage),
    )
    return source, c.state, history


@pytest.mark.parametrize("family", ("pc", "active"))
def test_should_own_full_origin_raw_wake_and_ordered_mixed_lineage_configs(family):
    origin, current, history = sequence(family)
    wrapper = CoupledStructuralState(1, origin.config, current, history)
    clone = copy_coupled_structural_state(wrapper)
    expected = asdict(clone)
    assert [t.kind for t in clone.transitions] == ["split", "merge", "prune"]
    assert clone.transitions[0].source_config == origin.config
    assert clone.transitions[0].target_config == clone.transitions[1].source_config
    assert clone.transitions[1].target_config == clone.transitions[2].source_config
    assert clone.transitions[2].target_config == current.config
    current.raw_parameters.output_bias[:] = 99
    object.__setattr__(origin.config, "gamma0", 3.0)
    object.__setattr__(history[0].lineage.source_config, "gamma0", 4.0)
    wrapper.raw_state.raw_parameters.hidden_weights[0][:] = 99
    if family != "pc":
        assert current.wake_state is not None
        current.wake_state.chemical_state[:] = 99
    np.testing.assert_equal(asdict(clone), expected)


def test_should_admit64_config_transitions_and_earlier_valid_rollback_states_without_provenance_claim():
    origin = raw()
    split = split_coupled_scaling_width(origin)
    pair = (
        CoupledStructuralTransition("split", split.lineage),
        CoupledStructuralTransition("merge", split.lineage),
    )
    full = CoupledStructuralState(1, origin.config, origin, pair * 32)
    short = CoupledStructuralState(1, origin.config, split.state, pair[:1])
    empty = CoupledStructuralState(1, origin.config, origin)
    assert MAX_STRUCTURAL_TRANSITIONS == 64
    assert len(full.transitions) == 64 and len(short.transitions) == 1 and empty.transitions == ()
    np.testing.assert_equal(asdict(full.raw_state), asdict(empty.raw_state))


@pytest.mark.parametrize(
    "kind,lineage_kind",
    (
        ("split", "prune"),
        ("merge", "prune"),
        ("prune", "split"),
        ("unknown", "split"),
        (True, "split"),
    ),
)
def test_should_require_exact_native_tag_and_corresponding_lineage_type(kind, lineage_kind):
    source = raw()
    lineage = (
        split_coupled_scaling_width(source).lineage
        if lineage_kind == "split"
        else prune_coupled_scaling_width(source, (1, 1)).lineage
    )
    with pytest.raises(ValueError):
        CoupledStructuralTransition(kind, lineage)


@pytest.mark.parametrize(
    "bad",
    (
        "empty_changed",
        "first_source",
        "gap",
        "reversed",
        "last_target",
        "origin_input",
        "origin_gamma",
        "origin_precision",
        "origin_policy",
        "history_list",
    ),
)
def test_should_require_the_complete_origin_to_current_configuration_chain(bad):
    origin, current, history = sequence()
    config = origin.config
    records: Any = history
    if bad == "empty_changed":
        records = ()
    elif bad == "first_source":
        records = history[1:]
    elif bad == "gap":
        records = (history[0], history[2])
    elif bad == "reversed":
        records = history[::-1]
    elif bad == "last_target":
        records = history[:2]
    elif bad == "history_list":
        records = list(history)
    else:
        updates: dict[str, dict[str, Any]] = {
            "origin_input": {"input_dim": 3},
            "origin_gamma": {"gamma0": 3.0},
            "origin_precision": {"precisions": (1.0, 1.0)},
            "origin_policy": {"circadian": CoupledCircadianConfig()},
        }
        config = replace(config, **updates[bad])
    before = asdict(current)
    with pytest.raises(ValueError):
        CoupledStructuralState(1, config, current, records)
    np.testing.assert_equal(asdict(current), before)


@pytest.mark.parametrize(
    "bad", ("format", "origin", "raw", "nan", "dtype", "wake", "tag", "lineage", "cap")
)
def test_should_revalidate_every_mutable_envelope_field_on_copy(bad):
    origin, current, history = sequence("active")
    state = CoupledStructuralState(1, origin.config, current, history)
    if bad == "format":
        object.__setattr__(state, "format_version", True)
    elif bad == "origin":
        object.__setattr__(state, "origin_config", {})
    elif bad == "raw":
        object.__setattr__(state, "raw_state", {})
    elif bad == "nan":
        state.raw_state.raw_parameters.output_bias[:] = np.nan
    elif bad == "dtype":
        object.__setattr__(
            state.raw_state.raw_parameters, "output_bias", np.zeros((1, 1), dtype=np.float32)
        )
    elif bad == "wake":
        assert state.raw_state.wake_state is not None
        state.raw_state.wake_state.plasticity_state[:] = 0
    elif bad == "tag":
        object.__setattr__(state.transitions[0], "kind", "prune")
    elif bad == "lineage":
        object.__setattr__(state.transitions[0], "lineage", {})
    else:
        object.__setattr__(state, "transitions", state.transitions * 22)
    with pytest.raises(ValueError):
        copy_coupled_structural_state(state)


def test_should_bound_history_length_and_type_before_any_native_allocation(monkeypatch):
    source = raw()

    def forbidden(*args, **kwargs):
        raise AssertionError("history admission reconstructed native payload")

    monkeypatch.setattr(CoupledPredictiveCodingNetwork, "__init__", forbidden)
    invalid: Any = (None,) * 65
    with pytest.raises(ValueError):
        CoupledStructuralState(1, source.config, source, invalid)
    with pytest.raises(ValueError):
        invalid = [None]
        CoupledStructuralState(1, source.config, source, invalid)


@pytest.mark.parametrize("changes", ({"width": 1001}, {"hidden_layers": 17}))
def test_should_bound_origin_and_current_configuration_before_reconstruction(monkeypatch, changes):
    source = raw()
    origin = deepcopy(source.config)
    for name, value in changes.items():
        object.__setattr__(origin, name, value)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid origin allocated native model")

    monkeypatch.setattr(CoupledPredictiveCodingNetwork, "__init__", forbidden)
    with pytest.raises(ValueError):
        CoupledStructuralState(1, origin, source)
