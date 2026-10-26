"""Native unit histories are observations, with detached bounded summaries."""

from collections import deque
from dataclasses import FrozenInstanceError, asdict, fields, is_dataclass
from math import isfinite
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters.native_unit_diagnostics import LegacyUnitDiagnostics, read_legacy_unit_diagnostics
from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.learnability_diagnostics import (
    UnitHistory,
    UnitHistoryDiagnostics,
    VectorSummary,
    summarize_unit_history,
)


def vectors():
    return [
        np.array([7, 2, 9], dtype=np.int64),
        np.array([3, -1, 7], dtype=np.int64),
        np.array([0.1, 0.3, 0.2]),
        np.array([1.5, 0.0, 2.75]),
    ]


def test_should_preserve_unit_order_pruned_parents_and_fractional_native_age():
    values = vectors()
    before = tuple(item.tobytes() for item in values)
    result = summarize_unit_history(*values, next_neuron_id=10)
    assert type(result) is UnitHistoryDiagnostics and type(result.units) is tuple
    assert all(type(item) is UnitHistory for item in result.units)
    assert [asdict(item) for item in result.units] == [
        {"neuron_id": 7, "parent_id": 3, "importance_ema": 0.1, "age_updates": 1.5},
        {"neuron_id": 2, "parent_id": None, "importance_ema": 0.3, "age_updates": 0.0},
        {"neuron_id": 9, "parent_id": 7, "importance_ema": 0.2, "age_updates": 2.75},
    ]
    assert type(result.importance) is type(result.age) is VectorSummary
    assert asdict(result.importance) == {
        "count": 3,
        "minimum": 0.1,
        "mean": pytest.approx(0.2),
        "maximum": 0.3,
    }
    assert asdict(result.age) == {
        "count": 3,
        "minimum": 0.0,
        "mean": pytest.approx(4.25 / 3),
        "maximum": 2.75,
    }
    assert tuple(item.tobytes() for item in values) == before


@pytest.mark.parametrize("equal", (True, False))
def test_should_compute_finite_means_without_overflowing_valid_native_values(equal):
    maximum = np.finfo(np.float64).max
    values = np.array([maximum, maximum] if equal else [maximum, maximum / 2, 0.0])
    width = len(values)
    result = summarize_unit_history(
        np.arange(width, dtype=np.int64),
        np.full(width, -1, dtype=np.int64),
        values,
        values,
        next_neuron_id=width,
    )
    expected = maximum if equal else maximum / 2
    assert isfinite(result.importance.mean) and isfinite(result.age.mean)
    assert result.importance.mean == result.age.mean == pytest.approx(expected)
    assert result.importance.maximum == maximum


def test_should_accept_the_inclusive_width_cap_without_an_arbitrary_next_id_ceiling():
    width = 4096
    result = summarize_unit_history(
        np.arange(width, dtype=np.int64),
        np.full(width, -1, dtype=np.int64),
        np.zeros(width),
        np.zeros(width),
        next_neuron_id=2**100,
    )
    assert len(result.units) == width and result.units[-1].neuron_id == 4095
    assert asdict(result.importance) == {
        "count": width,
        "minimum": 0.0,
        "mean": 0.0,
        "maximum": 0.0,
    }


def test_should_return_frozen_python_values_without_borrowing_any_caller_array():
    values = vectors()
    for item in values:
        item.flags.writeable = False
    result = summarize_unit_history(*values, next_neuron_id=10)
    for item in values:
        item.flags.writeable = True
        item.fill(0)
    assert result.units[0] == UnitHistory(7, 3, 0.1, 1.5)
    assert type(result.units[0].neuron_id) is int and type(result.units[0].age_updates) is float
    for record, attribute in (
        (result, "units"),
        (result.units[0], "age_updates"),
        (result.importance, "mean"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(record, attribute, None)


class ForeignArray(np.ndarray):
    pass


@pytest.mark.parametrize(
    "bad",
    (
        "ids_list",
        "ids_dtype",
        "parents_dtype",
        "importance_dtype",
        "age_dtype",
        "subclass",
        "ids_rank",
        "parents_rank",
        "importance_rank",
        "age_rank",
        "empty",
        "overcap",
        "parents_length",
        "importance_length",
        "age_length",
        "negative_id",
        "duplicate_id",
        "future_id",
        "negative_parent",
        "self_parent",
        "future_parent",
        "negative_importance",
        "nan_importance",
        "inf_importance",
        "negative_age",
        "nan_age",
        "inf_age",
        "next_bool",
        "next_zero",
        "next_negative",
        "next_float",
        "next_numpy",
    ),
)
def test_should_refuse_invalid_vector_metadata_or_lineage_before_summary_statistics(
    monkeypatch, bad
):
    values: Any = vectors()
    next_id: Any = 10
    if bad == "ids_list":
        values[0] = [7, 2, 9]
    elif bad in ("ids_dtype", "parents_dtype", "importance_dtype", "age_dtype"):
        index = {"ids_dtype": 0, "parents_dtype": 1, "importance_dtype": 2, "age_dtype": 3}[bad]
        values[index] = values[index].astype(
            np.float64
            if index == 0
            else np.int32
            if index == 1
            else np.float32
            if index == 2
            else np.int64
        )
    elif bad == "subclass":
        values[0] = values[0].view(ForeignArray)
    elif bad.endswith("_rank"):
        index = {"ids_rank": 0, "parents_rank": 1, "importance_rank": 2, "age_rank": 3}[bad]
        values[index] = values[index].reshape(1, 3)
    elif bad in ("empty", "overcap"):
        width = 0 if bad == "empty" else 4097
        values = [
            np.arange(width, dtype=np.int64),
            np.full(width, -1, dtype=np.int64),
            np.zeros(width),
            np.zeros(width),
        ]
        next_id = max(width, 1)
    elif bad.endswith("_length"):
        index = {"parents_length": 1, "importance_length": 2, "age_length": 3}[bad]
        values[index] = values[index][:-1]
    elif bad == "negative_id":
        values[0][0] = -1
    elif bad == "duplicate_id":
        values[0][0] = 2
    elif bad == "future_id":
        next_id = 9
    elif bad in ("negative_parent", "self_parent", "future_parent"):
        values[1][0] = {"negative_parent": -2, "self_parent": 7, "future_parent": 8}[bad]
    elif bad.endswith("_importance") or bad.endswith("_age"):
        index = 2 if bad.endswith("_importance") else 3
        values[index][0] = (
            -1.0 if bad.startswith("negative") else np.nan if bad.startswith("nan") else np.inf
        )
    else:
        next_id = {
            "next_bool": True,
            "next_zero": 0,
            "next_negative": -1,
            "next_float": 10.0,
            "next_numpy": np.int64(10),
        }[bad]

    def forbidden(*args, **kwargs):
        pytest.fail("invalid native metadata reached summary statistics or SVD")

    monkeypatch.setattr(np, "mean", forbidden)
    monkeypatch.setattr(np.linalg, "svd", forbidden)
    with pytest.raises(ValueError):
        summarize_unit_history(*values, next_neuron_id=next_id)


@pytest.fixture(scope="module")
def bare_model():
    model = object.__new__(CircadianPredictiveCodingNetwork)
    model.__dict__.update(
        weight_input_hidden=np.full((1, 3), np.nan),
        _neuron_ids=np.array([7, 2, 9], dtype=np.int64),
        _parent_ids=np.array([3, -1, 7], dtype=np.int64),
        _importance_ema=np.array([0.1, 0.3, 0.2]),
        _neuron_age=np.array([1.5, 0.0, 2.75]),
        _next_neuron_id=10,
    )
    return model


def test_should_read_only_current_adaptive_metadata_without_certifying_full_weights(bare_model):
    before = pickle.dumps(vars(bare_model))
    result = read_legacy_unit_diagnostics(bare_model)
    assert type(result) is LegacyUnitDiagnostics
    assert result.family == "legacy_cpc" and result.layer == "adaptive_hidden"
    assert [item.neuron_id for item in result.history.units] == [7, 2, 9]
    assert result.history.age.maximum == 2.75
    assert pickle.dumps(vars(bare_model)) == before


class Poison:
    def __getattribute__(self, name):
        pytest.fail("foreign model metadata or getter was accessed")


class ForeignCPC(CircadianPredictiveCodingNetwork):
    def __getattribute__(self, name):
        pytest.fail("nonexact native model hooks were accessed")


@pytest.mark.parametrize("bad", ("none", "poison", "coupled", "subclass"))
def test_should_refuse_other_types_and_model_families_before_accessing_their_handles(bad):
    model: Any = None
    if bad == "poison":
        model = Poison()
    elif bad == "coupled":
        model = object.__new__(CoupledPredictiveCodingNetwork)
    elif bad == "subclass":
        model = object.__new__(ForeignCPC)
    with pytest.raises(ValueError):
        read_legacy_unit_diagnostics(model)


@pytest.mark.parametrize(
    "bad",
    (
        "missing_weight",
        "none",
        "list",
        "dtype",
        "subclass",
        "rank",
        "rows_zero",
        "width_zero",
        "width_cap",
        "vector_width",
        "missing_ids",
        "importance_negative",
        "lineage",
    ),
)
def test_should_refuse_malformed_native_fields_without_model_callbacks(
    bare_model, monkeypatch, bad
):
    original = dict(vars(bare_model))
    try:
        if bad == "missing_weight":
            del bare_model.__dict__["weight_input_hidden"]
        elif bad == "missing_ids":
            del bare_model.__dict__["_neuron_ids"]
        elif bad in (
            "none",
            "list",
            "dtype",
            "subclass",
            "rank",
            "rows_zero",
            "width_zero",
            "width_cap",
        ):
            incoming: Any = None
            if bad == "list":
                incoming = [[0.0] * 3]
            elif bad == "dtype":
                incoming = np.zeros((1, 3), dtype=np.float32)
            elif bad == "subclass":
                incoming = np.zeros((1, 3)).view(ForeignArray)
            elif bad != "none":
                shape = {
                    "rank": (3,),
                    "rows_zero": (0, 3),
                    "width_zero": (1, 0),
                    "width_cap": (1, 4097),
                }[bad]
                incoming = np.zeros(shape)
            bare_model.weight_input_hidden = incoming
        elif bad == "vector_width":
            bare_model._neuron_age = np.zeros(2)
        elif bad == "importance_negative":
            bare_model._importance_ema = np.array([-1.0, 0.0, 0.0])
        else:
            bare_model._parent_ids = np.array([7, -1, 7], dtype=np.int64)

        def forbidden(*args, **kwargs):
            pytest.fail("malformed diagnostic metadata invoked a native getter or operation")

        monkeypatch.setattr(CircadianPredictiveCodingNetwork, "hidden_dim", property(forbidden))
        monkeypatch.setattr(CircadianPredictiveCodingNetwork, "get_neuron_lineage", forbidden)
        with pytest.raises(ValueError):
            read_legacy_unit_diagnostics(bare_model)
    finally:
        bare_model.__dict__.clear()
        bare_model.__dict__.update(original)


def array_inventory(root):
    arrays: dict[int, tuple[Any, ...]] = {}
    seen: set[int] = set()

    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if type(value) is np.ndarray:
            arrays[id(value)] = (
                value.dtype.str,
                value.shape,
                value.strides,
                (
                    value.flags.c_contiguous,
                    value.flags.f_contiguous,
                    value.flags.owndata,
                    value.flags.writeable,
                    value.flags.aligned,
                    value.flags.writebackifcopy,
                ),
                value.tobytes(order="A"),
            )
        elif type(value) is dict:
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple, deque)):
            for item in value:
                visit(item)
        elif is_dataclass(value):
            for field in fields(value):
                visit(getattr(value, field.name))

    visit(root)
    return arrays


def protected_read(model):
    before = pickle.dumps(vars(model))
    references = {name: id(value) for name, value in vars(model).items()}
    arrays = array_inventory(vars(model))

    def forbidden(*args, **kwargs):
        pytest.fail("direct unit reader invoked a native model method")

    with pytest.MonkeyPatch.context() as patch:
        for name, method in vars(CircadianPredictiveCodingNetwork).items():
            if name == "__init__" or not name.startswith("__"):
                if isinstance(method, property):
                    patch.setattr(CircadianPredictiveCodingNetwork, name, property(forbidden))
                elif callable(method) or isinstance(method, (staticmethod, classmethod)):
                    patch.setattr(CircadianPredictiveCodingNetwork, name, forbidden)
        result = read_legacy_unit_diagnostics(model)
    assert pickle.dumps(vars(model)) == before
    assert {name: id(value) for name, value in vars(model).items()} == references
    assert array_inventory(vars(model)) == arrays
    return result


@pytest.fixture(scope="module")
def observed_native_history():
    model = CircadianPredictiveCodingNetwork(
        2, 2, 23, CircadianConfig(sleep_mode="disabled"), min_hidden_dim=1, max_hidden_dim=4
    )
    features, targets = np.array([[0.2, -0.1], [-0.3, 0.4]]), np.array([[1.0], [0.0]])
    initial = protected_read(model)
    for _ in range(2):
        model.train_epoch(features, targets, 0.01, 1, 0.05)
    trained = protected_read(model)
    model._split_neurons((0,))
    split = protected_read(model)
    model.train_epoch(features, targets, 0.01, 1, 0.05)
    continued = protected_read(model)
    model._remove_neurons((0,))
    pruned = protected_read(model)
    return initial, trained, split, continued, pruned


def test_should_observe_actual_native_importance_and_age_without_mutating_any_native_state(
    observed_native_history,
):
    initial, trained, _, _, _ = observed_native_history
    assert initial.history.age == VectorSummary(2, 0.0, 0.0, 0.0)
    assert initial.history.importance == VectorSummary(2, 0.0, 0.0, 0.0)
    assert [item.age_updates for item in trained.history.units] == [2.0, 2.0]
    assert trained.history.importance.maximum > 0.0
    assert all(
        word in trained.importance_definition.lower()
        for word in ("ema", "gradient", "reward", "replay", "split", "prune")
    )
    assert all(
        word in trained.age_definition.lower()
        for word in ("float", "epoch", "replay", "zero", "wall")
    )


def test_should_follow_actual_split_prune_lineage_and_current_width_with_pruned_parent_allowed(
    observed_native_history,
):
    _, _, split, continued, pruned = observed_native_history
    assert [(item.neuron_id, item.parent_id, item.age_updates) for item in split.history.units] == [
        (0, None, 2.0),
        (1, None, 2.0),
        (2, 0, 0.0),
    ]
    assert split.history.units[2].importance_ema == split.history.units[0].importance_ema
    assert [item.age_updates for item in continued.history.units] == [3.0, 3.0, 1.0]
    assert [
        (item.neuron_id, item.parent_id, item.age_updates) for item in pruned.history.units
    ] == [(1, None, 3.0), (2, 0, 1.0)]
    assert pruned.history.units == continued.history.units[1:]


def test_should_keep_earlier_native_observations_detached_and_all_returned_records_frozen(
    observed_native_history,
):
    initial, _, _, _, pruned = observed_native_history
    assert [item.age_updates for item in initial.history.units] == [0.0, 0.0]
    assert len(initial.history.units) == 2 and len(pruned.history.units) == 2
    for record, attribute in (
        (pruned, "family"),
        (pruned.history, "units"),
        (pruned.history.units[0], "parent_id"),
        (pruned.history.age, "mean"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(record, attribute, None)
