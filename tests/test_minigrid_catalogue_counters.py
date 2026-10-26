"""Explicit catalogue clocks without constructing or executing native learners.

Copy sentinels isolate admission ordering. Native snapshot copying, scores and
paid readout isolation are exercised separately by the integration owner.
"""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_learnability as pilot
from src.adapters import minigrid_learning_pilot as shared
from src.adapters.torch_feature_learner import TorchFeaturePolicy


METHODS = (
    "public_invariant_bp",
    "public_invariant_pc",
    "public_invariant_neutral_cpc",
    "public_invariant_gated_cpc",
)
INVALID_ROWS: tuple[Any, ...] = (True, False, 54.0, np.int64(54), 0, 9, 53, 55, "54")
INVALID_POINTS: tuple[Any, ...] = (True, False, 2.0, np.int64(2), -1, 129, "2")


def forbidden(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("admission must refuse before touching native state or doing work")


@pytest.mark.parametrize("point", (0, 2, 128))
def test_should_require_both_actual_native_catalogue_clocks(point):
    state = SimpleNamespace(completed_updates=point, training_examples=54 * point)
    shared._counter(state, point, catalogue_rows=54)
    for field in ("completed_updates", "training_examples"):
        original = getattr(state, field)
        setattr(state, field, original + 1)
        with pytest.raises(ValueError):
            shared._counter(state, point, catalogue_rows=54)
        setattr(state, field, original)


@pytest.mark.parametrize("point", (0, 2, 32, 64, 96, 128))
def test_should_keep_original_nine_row_clock_as_default(point):
    state = SimpleNamespace(completed_updates=point, training_examples=9 * point)
    shared._counter(state, point)
    shared._counter(state, point, catalogue_rows=None)
    if point:
        with pytest.raises(ValueError):
            shared._counter(state, point, catalogue_rows=54)
        state.training_examples = 54 * point
        with pytest.raises(ValueError):
            shared._counter(state, point)


@pytest.mark.parametrize("rows", INVALID_ROWS)
def test_should_refuse_undeclared_catalogue_size_even_at_zero(rows):
    with pytest.raises(ValueError):
        shared._counter(
            SimpleNamespace(completed_updates=0, training_examples=0), 0, catalogue_rows=rows
        )


@pytest.mark.parametrize("point", INVALID_POINTS)
def test_should_refuse_non_native_or_out_of_range_explicit_points(point):
    actual = int(point)
    with pytest.raises(ValueError):
        shared._counter(
            SimpleNamespace(completed_updates=actual, training_examples=54 * actual),
            point,
            catalogue_rows=54,
        )


@pytest.mark.parametrize("field", ("completed_updates", "training_examples"))
@pytest.mark.parametrize("value", (False, 0.0, np.int64(0)))
@pytest.mark.parametrize("rows", (None, 54))
def test_should_refuse_equal_valued_non_native_actual_clocks(field, value, rows):
    state = SimpleNamespace(completed_updates=0, training_examples=0)
    setattr(state, field, value)
    with pytest.raises(ValueError):
        shared._counter(state, 0, catalogue_rows=rows)


@pytest.mark.parametrize("method", METHODS)
def test_should_validate_original_native_recipe_and_explicit_catalogue_rows(monkeypatch, method):
    native = TorchFeaturePolicy(method.removeprefix("public_invariant_"), 296, 8, 17, num_classes=7)
    state = SimpleNamespace(
        policy=SimpleNamespace(method=method, native_policy=native),
        completed_updates=2,
        training_examples=108,
    )
    copies = []

    def copy(owner):
        copies.append(owner)
        return state

    monkeypatch.setattr(shared, "copy_invariant_torch_feature_snapshot", copy)
    shared._invariant_feature_counter(state, 2, method, catalogue_rows=54)
    assert copies == [state]
    state.training_examples = 18
    with pytest.raises(ValueError):
        shared._invariant_feature_counter(state, 2, method, catalogue_rows=54)


@pytest.mark.parametrize("fault", ("method", "seed", "hidden", "features", "classes"))
def test_should_preserve_policy_recipe_checks_with_explicit_catalogue(monkeypatch, fault):
    native_options: dict[str, Any] = dict(
        method="bp", feature_dim=296, hidden_dim=8, seed=17, num_classes=7
    )
    if fault == "seed":
        native_options["seed"] = 18
    elif fault == "hidden":
        native_options["hidden_dim"] = 9
    elif fault == "features":
        native_options["feature_dim"] = 148
    elif fault == "classes":
        native_options["num_classes"] = 6
    native = TorchFeaturePolicy(**native_options)
    state = SimpleNamespace(
        policy=SimpleNamespace(
            method="public_invariant_pc" if fault == "method" else METHODS[0], native_policy=native
        ),
        completed_updates=2,
        training_examples=108,
    )
    monkeypatch.setattr(shared, "copy_invariant_torch_feature_snapshot", lambda owner: state)
    with pytest.raises(ValueError):
        shared._invariant_feature_counter(state, 2, METHODS[0], catalogue_rows=54)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("point", (0, 2, 128))
def test_should_pass_explicit_row_binding_through_bound_snapshot(monkeypatch, method, point):
    state, copied = object(), object()
    calls = []

    def snapshot(owner, updates, selected, *, catalogue_rows=None):
        calls.append((owner, updates, selected, catalogue_rows))
        return copied

    monkeypatch.setattr(shared, "_invariant_feature_snapshot", snapshot)
    assert pilot._invariant_snapshot(state, point, method, catalogue_rows=54) is copied
    assert calls == [(state, point, method, 54)]


@pytest.mark.parametrize("method", METHODS)
def test_should_preserve_declared_checkpoint_points_for_default_snapshot(monkeypatch, method):
    monkeypatch.setattr(shared, "_invariant_feature_snapshot", forbidden)
    monkeypatch.setattr(shared, "copy_invariant_torch_feature_snapshot", forbidden)
    with pytest.raises(ValueError):
        pilot._invariant_snapshot(object(), 2, method)


@pytest.mark.parametrize(
    "method", (pilot.memory.METHOD, "bp", "recurrent_bc", "recurrent_paired_bc", False)
)
def test_should_refuse_catalogue_mode_for_unbound_and_memory_snapshots_before_native_access(
    monkeypatch, method
):
    monkeypatch.setattr(shared, "_invariant_feature_snapshot", forbidden)
    monkeypatch.setattr(shared, "_invariant_memory_counter", forbidden)
    monkeypatch.setattr(shared, "copy_invariant_torch_feature_snapshot", forbidden)
    monkeypatch.setattr(pilot, "copy_invariant_memory_snapshot", forbidden)
    with pytest.raises(ValueError):
        pilot._invariant_snapshot(object(), 0, method, catalogue_rows=54)


@pytest.mark.parametrize("rows", INVALID_ROWS)
def test_should_refuse_invalid_catalogue_snapshot_before_state_copy(monkeypatch, rows):
    monkeypatch.setattr(shared, "_invariant_feature_snapshot", forbidden)
    monkeypatch.setattr(shared, "copy_invariant_torch_feature_snapshot", forbidden)
    with pytest.raises(ValueError):
        pilot._invariant_snapshot(object(), 0, METHODS[0], catalogue_rows=rows)


@pytest.mark.parametrize(
    "method", (pilot.memory.METHOD, "bp", "recurrent_bc", "recurrent_paired_bc")
)
def test_should_refuse_unsupported_catalogue_readout_before_source_or_budget_work(
    monkeypatch, method
):
    monkeypatch.setattr(pilot, "_invariant_fit_sources", forbidden)
    monkeypatch.setattr(pilot, "_invariant_snapshot", forbidden)
    monkeypatch.setattr(pilot, "deepcopy", forbidden)
    pilot_options: dict[str, Any] = dict(
        method=method,
        updates=0,
        sources=(),
        budget=SimpleNamespace(),
        check_boundary=forbidden,
        catalogue_rows=54,
    )
    with pytest.raises(ValueError):
        pilot._score_checkpoint(object(), object(), **pilot_options)


class SnapshotAdmissionObserved(Exception):
    """Stop an admission check before any source validation or native copying."""


def _probe_inputs(method, point):
    return (
        {"method": method, "checkpoint_updates": point},
        {"row": {"method": method}, "source": object()},
        object(),
        ((), ()),
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("point", (0, 2, 128))
def test_should_propagate_explicit_catalogue_binding_to_probe_source_admission(
    monkeypatch, method, point
):
    calls = []
    slot, arm, state, data = _probe_inputs(method, point)

    def admit(owner, updates, selected, *, catalogue_rows=None):
        calls.append((owner, updates, selected, catalogue_rows))
        raise SnapshotAdmissionObserved

    monkeypatch.setattr(pilot, "_invariant_snapshot", admit)
    monkeypatch.setattr(pilot, "_invariant_probe_inputs", forbidden)
    monkeypatch.setattr(pilot, "deepcopy", forbidden)
    with pytest.raises(SnapshotAdmissionObserved):
        pilot._run_probe(
            slot,
            arm,
            state,
            data,
            sampler=None,
            deadline=1.0,
            check_boundary=forbidden,
            catalogue_rows=54,
        )
    assert calls == [(state, point, method, 54)]


@pytest.mark.parametrize("point", (0, 128))
def test_should_keep_original_probe_source_checkpoints_without_catalogue_binding(
    monkeypatch, point
):
    calls = []
    slot, arm, state, data = _probe_inputs(METHODS[0], point)

    def admit(owner, updates, selected, *, catalogue_rows=None):
        calls.append((owner, updates, selected, catalogue_rows))
        raise SnapshotAdmissionObserved

    monkeypatch.setattr(pilot, "_invariant_snapshot", admit)
    with pytest.raises(SnapshotAdmissionObserved):
        pilot._run_probe(
            slot,
            arm,
            state,
            data,
            sampler=None,
            deadline=1.0,
            check_boundary=forbidden,
        )
    assert calls == [(state, point, METHODS[0], None)]


def test_should_refuse_undeclared_default_probe_point_before_source_snapshot(monkeypatch):
    monkeypatch.setattr(pilot, "_invariant_snapshot", forbidden)
    monkeypatch.setattr(pilot, "_invariant_probe_inputs", forbidden)
    monkeypatch.setattr(pilot, "deepcopy", forbidden)
    with pytest.raises(ValueError):
        pilot._run_probe(
            *_probe_inputs(METHODS[0], 2),
            sampler=None,
            deadline=1.0,
            check_boundary=forbidden,
            catalogue_rows=None,
        )


@pytest.mark.parametrize("rows", INVALID_ROWS)
def test_should_refuse_undeclared_probe_catalogue_size_before_source_or_budget_work(
    monkeypatch, rows
):
    monkeypatch.setattr(pilot, "_invariant_snapshot", forbidden)
    monkeypatch.setattr(pilot, "_invariant_probe_inputs", forbidden)
    monkeypatch.setattr(pilot, "deepcopy", forbidden)
    with pytest.raises(ValueError):
        pilot._run_probe(
            *_probe_inputs(METHODS[0], 0),
            sampler=None,
            deadline=1.0,
            check_boundary=forbidden,
            catalogue_rows=rows,
        )


@pytest.mark.parametrize("point", INVALID_POINTS)
def test_should_refuse_invalid_catalogue_probe_point_before_source_snapshot(monkeypatch, point):
    monkeypatch.setattr(pilot, "_invariant_snapshot", forbidden)
    monkeypatch.setattr(pilot, "_invariant_probe_inputs", forbidden)
    monkeypatch.setattr(pilot, "deepcopy", forbidden)
    with pytest.raises(ValueError):
        pilot._run_probe(
            *_probe_inputs(METHODS[0], point),
            sampler=None,
            deadline=1.0,
            check_boundary=forbidden,
            catalogue_rows=54,
        )


@pytest.mark.parametrize(
    "method", (pilot.memory.METHOD, "bp", "recurrent_bc", "recurrent_paired_bc")
)
def test_should_refuse_non_neural_catalogue_probe_before_state_or_model_work(monkeypatch, method):
    monkeypatch.setattr(pilot, "_invariant_snapshot", forbidden)
    monkeypatch.setattr(pilot, "_invariant_probe_inputs", forbidden)
    monkeypatch.setattr(pilot, "deepcopy", forbidden)
    with pytest.raises(ValueError):
        pilot._run_probe(
            *_probe_inputs(method, 0),
            sampler=None,
            deadline=1.0,
            check_boundary=forbidden,
            catalogue_rows=54,
        )


def test_should_refuse_paired_catalogue_transport_before_source_or_budget_work(monkeypatch):
    monkeypatch.setattr(pilot, "_invariant_snapshot", forbidden)
    monkeypatch.setattr(pilot, "_invariant_probe_inputs", forbidden)
    monkeypatch.setattr(pilot, "deepcopy", forbidden)
    with pytest.raises(ValueError):
        pilot._run_probe(
            *_probe_inputs(METHODS[0], 0),
            sampler=None,
            deadline=1.0,
            check_boundary=forbidden,
            paired=True,
            catalogue_rows=54,
        )
