"""Complete fingerprints catch changes beyond predictions and parameters."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_confirmation_state as state
from src.app import continual_gating_pilot as gating
from src.app import continual_parent_factor_preflight as parent
from src.app import continual_replay_factor_pilot as replay
from src.app import continual_sleep_factor_preflight as sleep
from src.app.continual_combined_factor_preflight import _state_hash
from src.app.continual_parent_factor_manifest import fixed_parent_manifest
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork, ReplaySnapshot
from src.core.controlled_parent_selection import ParentControlledCircadianNetwork
from src.core.predictive_coding import PredictiveCodingNetwork


@pytest.mark.parametrize(
    "model_type", [BackpropMLP, PredictiveCodingNetwork, CircadianPredictiveCodingNetwork]
)
def test_should_capture_each_original_hash_contract_from_live_parameters(model_type: type) -> None:
    model = model_type(2, 8, 1042)
    before = state.capture_model(model)
    expected = {
        "p63-shallow-parameter-tensors-v1": gating._parameter_hash(model),
        "p63-replay-shallow-parameter-tensors-v1": replay._parameter_hash(model),
        "p63-sleep-factor-shallow-parameters-v1": sleep._parameter_hash(model),
    }
    assert before.parameter_sha256_by_contract == expected
    assert len(set(expected.values())) == 3
    assert before.parameter_sha256 == expected["p63-replay-shallow-parameter-tensors-v1"]
    model.bias_output[0, 0] += 0.25
    after = state.capture_model(model)
    assert all(after.parameter_sha256_by_contract[key] != value for key, value in expected.items())
    assert after.parameter_sha256_by_contract == {
        "p63-shallow-parameter-tensors-v1": gating._parameter_hash(model),
        "p63-replay-shallow-parameter-tensors-v1": replay._parameter_hash(model),
        "p63-sleep-factor-shallow-parameters-v1": sleep._parameter_hash(model),
    }


@pytest.mark.parametrize("model_type", [BackpropMLP, PredictiveCodingNetwork])
def test_should_bind_baseline_traffic_and_preserve_independent_copies(model_type: type) -> None:
    model = model_type(2, 8, 1042)
    held = deepcopy(model)
    before = state.capture_model(model)
    assert before == state.capture_model(held)
    model._traffic_steps += 1
    after = state.capture_model(model)
    assert before.parameter_sha256 == after.parameter_sha256
    assert before.state_sha256 != after.state_sha256
    assert state.capture_model(held) == before


@pytest.mark.parametrize("model_type", [BackpropMLP, PredictiveCodingNetwork])
def test_should_refuse_detached_baseline_alias_with_identical_parameter_bytes(
    model_type: type,
) -> None:
    model = model_type(2, 8, 1042)
    before = state.capture_model(model)
    model.weight_input_hidden = model.weight_input_hidden.copy()
    assert parent._parameter_hash(model) == before.parameter_sha256
    with pytest.raises(ValueError, match="parameter aliases differ"):
        state.capture_model(model)


def test_should_refuse_wrong_tensor_shape_with_unchanged_parameter_count() -> None:
    model = parent._new_progress(fixed_parent_manifest(), 347).models["neutral_off"]
    before = state.capture_model(model)
    model.weight_input_hidden = model.weight_input_hidden.T
    assert parent._parameter_count(model) == before.parameter_count
    with pytest.raises(ValueError, match="frozen shallow geometry"):
        state.capture_model(model)


@pytest.mark.parametrize("mode", ["usage", "scheduled", "random"])
def test_should_reuse_complete_circadian_fingerprint_with_selector(mode: str) -> None:
    model = parent._new_progress(fixed_parent_manifest(), 347).models[f"{mode}_growth"]
    assert isinstance(model, ParentControlledCircadianNetwork)
    captured = state.capture_model(model)
    assert captured.state_sha256 == _state_hash(model)
    assert captured.clocks == {
        "wake_batches": 0,
        "wake_examples": 0,
        "wake_batches_since_sleep": 0,
        "replay_updates": 0,
        "sleep_events": 0,
    }
    assert captured.selector is not None
    assert captured.retention is not None
    assert captured.lineage is not None
    model._parent_selection_rng.random()
    with pytest.raises(ValueError, match="complete checkpoint changed"):
        state.require_checkpoints({"model": model}, {"model": captured}, "selector_rng")


@pytest.mark.parametrize("change", ["chemistry", "noise_rng", "memory", "lineage"])
def test_should_reject_nonparameter_rollback_corruption(change: str) -> None:
    model = parent._new_progress(fixed_parent_manifest(), 347).models["neutral_off"]
    assert isinstance(model, CircadianPredictiveCodingNetwork)
    before = state.capture_model(model)
    if change == "chemistry":
        model._hidden_chemical[0] += 1
    elif change == "noise_rng":
        model._rng.random()
    elif change == "memory":
        model._replay_memory.append(ReplaySnapshot(np.zeros((1, 2)), np.zeros((1, 1)), 0.5, 0.0))
    else:
        model._next_neuron_id += 1
    assert state.capture_model(model).parameter_sha256 == before.parameter_sha256
    with pytest.raises(ValueError, match="changed complete state"):
        state.guard_witness("neutral_off", "a", 12, "rolled_back", before, model)


def test_should_refuse_nonfinite_array_before_digesting() -> None:
    model = BackpropMLP(2, 8, 1042)
    model._traffic_sums[0][0] = np.nan
    with pytest.raises(ValueError, match="nonfinite array"):
        state.capture_model(model)


@pytest.mark.parametrize("change", ["final", "count", "identity", "content", "overlap"])
def test_should_refuse_changed_arrived_roles(change: str) -> None:
    roles = arrived._build_phase_a_roles(fixed_parent_manifest().source, 347)
    if change == "final":
        roles = replace(roles, final_released=True)
    elif change == "count":
        roles = replace(roles, expected_final_count=39)
    elif change == "identity":
        roles = replace(roles, seed=348)
    elif change == "content":
        roles.train.input[0, 0] += 1
    else:
        changed = dict(roles.sample_ids)
        changed["inner_guard"] = (roles.sample_ids["train"][0], *changed["inner_guard"][1:])
        roles = replace(roles, sample_ids=changed)
    with pytest.raises(ValueError, match="confirmation arrived"):
        state.capture_roles(roles, "a", 347)
