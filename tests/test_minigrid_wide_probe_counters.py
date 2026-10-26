"""Explicit H32 stream admission without native training or prediction.

Copier sentinels test routing and policy/clock refusal. Full native snapshots and
paid stream/probe isolation are covered by the separately owned public chain.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from src.adapters import minigrid_learning_pilot as shared
from src.adapters.torch_feature_learner import TorchFeaturePolicy


METHODS = tuple(f"public_invariant_{method}" for method in ("bp", "pc", "neutral_cpc", "gated_cpc"))


def forbidden(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("admission must refuse before copying state or doing work")


def state_at(method, point, *, hidden=32, **options):
    native = TorchFeaturePolicy(
        method.removeprefix("public_invariant_"), 296, hidden, 17, num_classes=7, **options
    )
    return SimpleNamespace(
        policy=SimpleNamespace(method=method, native_policy=native),
        completed_updates=point,
        training_examples=54 * point,
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("point", (0, 2, 32, 128))
def test_should_admit_explicit_wide_recipe_only_with_actual_catalogue_clocks(
    monkeypatch, method, point
):
    state = state_at(method, point)
    copied = []

    def copy(owner):
        copied.append(owner)
        return state

    monkeypatch.setattr(shared, "copy_invariant_torch_feature_snapshot", copy)
    shared._invariant_feature_counter(state, point, method, catalogue_rows=54, wide_neural=True)
    assert copied == [state]
    for field in ("completed_updates", "training_examples"):
        original = getattr(state, field)
        setattr(state, field, original + 1)
        with pytest.raises(ValueError):
            shared._invariant_feature_counter(
                state, point, method, catalogue_rows=54, wide_neural=True
            )
        setattr(state, field, original)


@pytest.mark.parametrize("wide,hidden", ((False, 32), (True, 8)))
def test_should_refuse_cross_width_recipe_instead_of_inferring_from_state(
    monkeypatch, wide, hidden
):
    state = state_at(METHODS[0], 0, hidden=hidden)
    monkeypatch.setattr(shared, "copy_invariant_torch_feature_snapshot", lambda owner: state)
    with pytest.raises(ValueError):
        shared._invariant_feature_counter(state, 0, METHODS[0], catalogue_rows=54, wide_neural=wide)


@pytest.mark.parametrize("wide", (None, 0, 1, "true", (), []))
def test_should_refuse_non_bool_width_before_native_state_access(monkeypatch, wide):
    monkeypatch.setattr(shared, "copy_invariant_torch_feature_snapshot", forbidden)
    with pytest.raises(ValueError):
        shared._invariant_feature_counter(
            object(), 0, METHODS[0], catalogue_rows=54, wide_neural=wide
        )


@pytest.mark.parametrize("rows", (None, True, 54.0, 9, 53, 55, "54"))
def test_should_refuse_wide_without_exact_original_catalogue_before_copy(monkeypatch, rows):
    monkeypatch.setattr(shared, "copy_invariant_torch_feature_snapshot", forbidden)
    with pytest.raises(ValueError):
        shared._invariant_feature_counter(
            object(), 0, METHODS[0], catalogue_rows=rows, wide_neural=True
        )


@pytest.mark.parametrize(
    "method",
    (
        "bp",
        "recurrent_bc",
        "public_invariant_observation_memory",
        *shared.cue_relative.METHODS,
        False,
    ),
)
def test_should_refuse_foreign_wide_recipes_before_state_access(monkeypatch, method):
    monkeypatch.setattr(shared, "copy_invariant_torch_feature_snapshot", forbidden)
    monkeypatch.setattr(shared.cue_relative, "snapshot", forbidden)
    with pytest.raises(ValueError):
        shared._invariant_feature_counter(object(), 0, method, catalogue_rows=54, wide_neural=True)


@pytest.mark.parametrize(
    "options",
    (
        {"learning_rate": 0.04},
        {"momentum": 0.0},
        {"inference_steps": 64},
        {"inference_learning_rate": 0.03},
    ),
)
def test_should_refuse_changed_native_algorithm_recipe(monkeypatch, options):
    state = state_at(METHODS[0], 0, **options)
    monkeypatch.setattr(shared, "copy_invariant_torch_feature_snapshot", lambda owner: state)
    with pytest.raises(ValueError):
        shared._invariant_feature_counter(state, 0, METHODS[0], catalogue_rows=54, wide_neural=True)


def test_should_preserve_full_snapshot_validation_in_wide_mode():
    # An object with only counters is not an owned full native state.
    with pytest.raises(ValueError, match="exact complete"):
        shared._invariant_feature_counter(
            state_at(METHODS[0], 0), 0, METHODS[0], catalogue_rows=54, wide_neural=True
        )


@pytest.mark.parametrize("point", (True, 0.0, -1, 129))
def test_should_refuse_non_native_or_out_of_range_wide_points(monkeypatch, point):
    state = state_at(METHODS[0], int(point))
    monkeypatch.setattr(shared, "copy_invariant_torch_feature_snapshot", lambda owner: state)
    with pytest.raises(ValueError):
        shared._invariant_feature_counter(
            state, point, METHODS[0], catalogue_rows=54, wide_neural=True
        )


@pytest.mark.parametrize("field", ("completed_updates", "training_examples"))
@pytest.mark.parametrize("value", (False, 0.0))
def test_should_refuse_equal_valued_non_native_wide_clocks(monkeypatch, field, value):
    state = state_at(METHODS[0], 0)
    setattr(state, field, value)
    monkeypatch.setattr(shared, "copy_invariant_torch_feature_snapshot", lambda owner: state)
    with pytest.raises(ValueError):
        shared._invariant_feature_counter(state, 0, METHODS[0], catalogue_rows=54, wide_neural=True)


@pytest.mark.parametrize(
    "overrides",
    (
        {"wide_neural": 1},
        {"wide_neural": None},
        {"catalogue_rows": None},
        {"catalogue_rows": 54.0},
        {"learner_factory": None},
        {"checkpoint_updates": (0, 32)},
        {"checkpoint_updates": [0, 32, 64, 96, 128]},
        {"checkpoint_updates": (False, 32, 64, 96, 128)},
        {"packets": (object(),) * 32},
        {"public_history": True},
        {"invariant_fifo36": True},
        {"method": "public_invariant_observation_memory"},
    ),
)
def test_should_refuse_unbound_wide_training_before_factory_or_batch_work(monkeypatch, overrides):
    options = dict(
        packets=(object(),) * 128,
        sampler=None,
        deadline=1.0,
        check_boundary=forbidden,
        learner_factory=forbidden,
        checkpoint_updates=shared.POINTS,
        catalogue_rows=54,
        wide_neural=True,
    )
    method = overrides.get("method", METHODS[0])
    options.update({key: value for key, value in overrides.items() if key != "method"})
    monkeypatch.setattr(shared, "_validate_batch", forbidden)
    monkeypatch.setattr(shared, "_learner", forbidden)
    with pytest.raises(ValueError):
        shared._train_arm(shared._row(method), **options)


class CounterObserved(Exception):
    """Stop after factory and explicit counter routing, before budget/model work."""


@pytest.mark.parametrize("wide", (False, True))
@pytest.mark.parametrize("method", METHODS)
def test_should_forward_width_only_when_explicitly_enabled(monkeypatch, wide, method):
    state = object()
    packet = SimpleNamespace(
        episode=None,
        context="train-catalogue",
        num_classes=7,
        features=SimpleNamespace(shape=(54, 296)),
        targets=SimpleNamespace(shape=(54, 1)),
    )
    source = SimpleNamespace(snapshot_state=lambda: state)
    calls = []

    def counter(owner, point, selected, **kwargs):
        calls.append((owner, point, selected, kwargs))
        raise CounterObserved

    monkeypatch.setattr(shared, "deepcopy", lambda value: value)
    monkeypatch.setattr(shared, "_validate_batch", lambda *args: None)
    monkeypatch.setattr(shared, "_invariant_feature_counter", counter)
    monkeypatch.setattr(shared, "_readback", lambda *args, **kwargs: None)
    monkeypatch.setattr(shared, "ToyBudgetSession", forbidden)
    with pytest.raises(CounterObserved):
        shared._train_arm(
            shared._row(method),
            (packet,) * 128,
            None,
            1.0,
            lambda: None,
            learner_factory=lambda selected: source,
            catalogue_rows=54,
            wide_neural=wide,
        )
    expected = {"catalogue_rows": 54, **({"wide_neural": True} if wide else {})}
    assert calls == [(state, 0, method, expected)]


@pytest.mark.parametrize(
    "changes",
    (
        {"method": "pc"},
        {"feature_dim": 148},
        {"hidden_dim": 31},
        {"seed": 18},
        {"num_classes": 6},
    ),
)
def test_should_bind_wide_method_geometry_and_seed(monkeypatch, changes):
    state = state_at(METHODS[0], 0)
    options = dict(method="bp", feature_dim=296, hidden_dim=32, seed=17, num_classes=7)
    options.update(changes)
    state.policy.native_policy = TorchFeaturePolicy(**options)
    monkeypatch.setattr(shared, "copy_invariant_torch_feature_snapshot", lambda owner: state)
    with pytest.raises(ValueError):
        shared._invariant_feature_counter(state, 0, METHODS[0], catalogue_rows=54, wide_neural=True)


def test_should_refuse_replaced_wrapper_method_in_wide_snapshot(monkeypatch):
    state = state_at(METHODS[0], 0)
    state.policy.method = METHODS[1]
    monkeypatch.setattr(shared, "copy_invariant_torch_feature_snapshot", lambda owner: state)
    with pytest.raises(ValueError):
        shared._invariant_feature_counter(state, 0, METHODS[0], catalogue_rows=54, wide_neural=True)
