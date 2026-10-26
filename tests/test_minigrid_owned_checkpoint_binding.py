"""Owned, validated checkpoint binding without training or prediction.

Fixtures declare structurally valid128-update clocks for the original H8 and
explicit H32 recipes. Their history was not executed: these checks establish
copy ownership and admission behavior, never acquisition or game performance.
"""

from copy import deepcopy
from dataclasses import replace

import pytest

from src.adapters import minigrid_learnability as pilot
from src.adapters import minigrid_learning_pilot as shared
from src.adapters.invariant_torch_feature_learner import InvariantTorchFeatureLearner
from src.shared.torch_runtime import require_torch
from test_minigrid_wide_evaluation import METHODS, declared_endpoint, fixed_policy
from test_torch_feature_learner import assert_tree, tensors


@pytest.fixture(autouse=True)
def single_thread():
    torch = require_torch()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def checkpoint(method, *, width=32, rows=54):
    """Declare valid full native counters; execute no updates or predictions."""
    state = declared_endpoint(InvariantTorchFeatureLearner(fixed_policy(method, hidden_dim=width)))
    object.__setattr__(state.native_state, "training_examples", rows * 128)
    if method in ("neutral_cpc", "gated_cpc"):
        state.native_state.head_state["wake_examples"] = rows * 128
    return state


def bind(route, state, method, *, width=32, rows=54):
    operation = (
        shared._invariant_feature_snapshot if route == "shared" else pilot._invariant_snapshot
    )
    options = {} if rows == 9 else {"catalogue_rows": 54}
    if width == 32:
        options["wide_neural"] = True
    return operation(state, 128, f"public_invariant_{method}", **options)


@pytest.mark.parametrize("route", ("shared", "learnability"))
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("width,rows", ((8, 9), (8, 54), (32, 54)))
def test_should_return_the_complete_owned_checkpoint_for_each_existing_recipe(
    route, method, width, rows
):
    torch = require_torch()
    state = checkpoint(method, width=width, rows=rows)
    before = deepcopy(state)

    owned = bind(route, state, method, width=width, rows=rows)

    assert owned is not state and owned.native_state is not state.native_state
    assert owned.native_state.head_state is not state.native_state.head_state
    assert_tree(torch, owned, before)
    for actual, original in zip(tensors(torch, owned), tensors(torch, state), strict=True):
        assert actual.data_ptr() != original.data_ptr()
    if method == "bp":
        assert owned.native_state.optimizer_state is not state.native_state.optimizer_state
        assert owned.native_state.gradients is not state.native_state.gradients
    elif method in ("neutral_cpc", "gated_cpc"):
        assert (
            owned.native_state.head_state["energy_history"]
            is not state.native_state.head_state["energy_history"]
        )

    original_weight = next(t for t in tensors(torch, state) if t.dtype == torch.float32)
    original_weight.add_(0.125)
    assert_tree(torch, owned, before)
    changed = deepcopy(state)
    owned_weight = next(t for t in tensors(torch, owned) if t.dtype == torch.float32)
    owned_weight.sub_(0.25)
    assert_tree(torch, state, changed)


@pytest.mark.parametrize("route", ("shared", "learnability"))
@pytest.mark.parametrize("method", METHODS)
def test_should_revalidate_current_native_weights_instead_of_reusing_a_prior_copy(route, method):
    torch = require_torch()
    state = checkpoint(method)
    first = bind(route, state, method)
    before = deepcopy(first)
    current_weight = next(t for t in tensors(torch, state) if t.dtype == torch.float32)
    current_weight.add_(0.125)

    second = bind(route, state, method)

    assert_tree(torch, first, before)
    assert_tree(torch, second, state)
    second_weight = next(t for t in tensors(torch, second) if t.dtype == torch.float32)
    first_weight = next(t for t in tensors(torch, first) if t.dtype == torch.float32)
    assert not torch.equal(second_weight, first_weight)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("width,rows", ((8, 9), (8, 54), (32, 54)))
def test_should_preserve_counter_only_none_return_for_existing_callers(method, width, rows):
    state = checkpoint(method, width=width, rows=rows)
    options = {} if rows == 9 else {"catalogue_rows": 54}
    if width == 32:
        options["wide_neural"] = True
    assert (
        shared._invariant_feature_counter(state, 128, f"public_invariant_{method}", **options)
        is None
    )


@pytest.mark.parametrize("method", METHODS)
def test_should_return_the_validated_copy_when_original_changes_after_copying(monkeypatch, method):
    """A later original read must not replace the checkpoint already validated."""
    torch = require_torch()
    state = checkpoint(method)
    before = deepcopy(state)
    copy = shared.copy_invariant_torch_feature_snapshot
    observed = []

    def copy_then_change_original(value):
        owned = copy(value)
        observed.append(owned)
        object.__setattr__(value.native_state, "training_examples", True)
        return owned

    monkeypatch.setattr(shared, "copy_invariant_torch_feature_snapshot", copy_then_change_original)

    result = bind("learnability", state, method)

    assert len(observed) == 1
    assert_tree(torch, result, before)
    assert state.native_state.training_examples is True
    with pytest.raises(ValueError):
        bind("learnability", state, method)


FAULTS = (
    ("bp", "record_type"),
    ("bp", "extra_field"),
    ("bp", "encoding"),
    ("bp", "native_policy"),
    ("bp", "updates"),
    ("bp", "rows"),
    ("bp", "row_type"),
    ("bp", "weight_nan"),
    ("bp", "weight_shape"),
    ("bp", "optimizer"),
    ("bp", "optimizer_buffer"),
    ("bp", "gradients"),
    ("pc", "traffic"),
    ("neutral_cpc", "age"),
    ("gated_cpc", "history"),
    ("neutral_cpc", "rng"),
    ("gated_cpc", "optimizer"),
)


def corrupt(state, fault):
    native = state.native_state
    if fault == "record_type":
        return native
    if fault == "extra_field":
        object.__setattr__(state, "foreign", True)
    elif fault == "encoding":
        object.__setattr__(state.policy, "encoding_id", "foreign")
    elif fault == "native_policy":
        changed = replace(native.policy, learning_rate=0.04)
        object.__setattr__(native, "policy", changed)
        object.__setattr__(state.policy, "native_policy", changed)
    elif fault in ("updates", "rows", "row_type"):
        name = "completed_updates" if fault == "updates" else "training_examples"
        value = True if fault == "row_type" else getattr(native, name) - 1
        object.__setattr__(native, name, value)
    elif fault == "weight_nan":
        native.head_state["weight_feature_hidden"].fill_(float("nan"))
    elif fault == "weight_shape":
        native.head_state["weight_feature_hidden"] = native.head_state["weight_feature_hidden"][:1]
    elif fault == "optimizer":
        object.__setattr__(native, "optimizer_state", {})
    elif fault == "optimizer_buffer":
        native.optimizer_state["state"][0]["momentum_buffer"].fill_(float("nan"))
    elif fault == "gradients":
        native.gradients.clear()
    elif fault == "traffic":
        native.head_state["traffic_steps"] -= 1
    elif fault == "age":
        native.head_state["neuron_age"].fill_(-1)
    elif fault == "history":
        native.head_state["energy_history"].append(0.0)
    elif fault == "rng":
        native.head_state["split_generator_state"] = require_torch().zeros(1)
    else:
        raise AssertionError(f"unknown corruption fixture: {fault}")
    return state


@pytest.mark.parametrize("route", ("shared", "learnability"))
@pytest.mark.parametrize("method,fault", FAULTS)
def test_should_refuse_corrupted_current_full_state_even_after_successful_binding(
    route, method, fault
):
    state = checkpoint(method)
    bind(route, state, method)
    malformed = corrupt(state, fault)

    with pytest.raises(ValueError):
        bind(route, malformed, method)
