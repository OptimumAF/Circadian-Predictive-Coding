"""Bound native hidden readouts: canonical float32 priors with zero training.

Only two discrete public-history rows are used. They are synthetic engineering
inputs, and their hidden activations establish no task or learning advantage.
"""

from copy import deepcopy
from dataclasses import replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters import torch_activation_diagnostics as diagnostics
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeaturePolicy,
    InvariantTorchFeatureSnapshot,
)
from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy
from src.core.minigrid_invariant_encoding import INVARIANT_HISTORY_ENCODING_ID
from src.shared.torch_runtime import require_torch
from test_minigrid_invariant_encoding import frame
from test_torch_feature_learner import assert_tree
from test_torch_feature_rehearsal_pilot import forbidden


METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")


@pytest.fixture(scope="module")
def states():
    torch = require_torch()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    sources = {
        method: InvariantTorchFeatureLearner(
            InvariantTorchFeaturePolicy(TorchFeaturePolicy(method, 296, 8, 17, num_classes=7))
        )
        for method in METHODS
    }
    return torch, sources, {method: source.snapshot_state() for method, source in sources.items()}


@pytest.fixture(autouse=True)
def no_training_or_prediction(monkeypatch):
    for kind in (TorchFeatureLearner, InvariantTorchFeatureLearner):
        monkeypatch.setattr(kind, "__init__", forbidden)
        monkeypatch.setattr(kind, "train_batch", forbidden)
        monkeypatch.setattr(kind, "predict", forbidden)


def data(*, variant=False, canonical=False):
    rows = []
    for cue in (0, 2):
        first = frame(
            wall_color=5 if canonical else 2 if variant else 1,
            direction=0 if canonical else 1 if variant else 2,
            cue_color=cue,
            key_color=cue,
            door_state=2,
        )
        current = frame(
            wall_color=5 if canonical else 3 if variant else 4,
            direction=0 if canonical else 2 if variant else 3,
            cue_color=cue,
            key_color=cue,
            door_state=0,
        )
        rows.append(first + current)
    return np.array(rows, dtype=np.float64)


def manual_prior(torch, state, values):
    native = state.native_state.head_state
    inputs = torch.from_numpy(values.astype(np.float32))
    with torch.no_grad():
        return (
            torch.tanh(inputs @ native["weight_feature_hidden"] + native["bias_hidden"])
            .numpy()
            .astype(np.float64, copy=True)
        )


@pytest.mark.parametrize("method", METHODS)
def test_should_read_exact_canonical_float32_priors_without_training_or_changing_owned_state(
    states, method
):
    torch, sources, snapshots = states
    source, snapshot = sources[method], snapshots[method]
    before_source, before_snapshot = source.snapshot_state(), deepcopy(snapshot)
    original, variant, canonical = data(), data(variant=True), data(canonical=True)
    inputs_before = pickle.dumps((original, variant, canonical))
    torch_rng, numpy_rng = torch.get_rng_state().clone(), pickle.dumps(np.random.get_state())
    expected = manual_prior(torch, snapshot, canonical)
    raw_prior = manual_prior(torch, snapshot, original)
    assert not np.array_equal(expected, raw_prior), (
        "fixture must distinguish raw from canonical projection"
    )
    for values in (original, variant):
        actual = diagnostics.read_native_activations(snapshot, values)
        assert actual.dtype == np.float64 and actual.shape == (2, 8)
        assert actual.flags.owndata and not np.shares_memory(actual, values)
        assert np.isfinite(actual).all() and np.all(np.abs(actual) <= 1)
        np.testing.assert_array_equal(actual, expected)
        actual[:] = 123.0
        np.testing.assert_array_equal(
            diagnostics.read_native_activations(snapshot, values), expected
        )
    assert snapshot.policy.encoding_id == INVARIANT_HISTORY_ENCODING_ID
    assert snapshot.policy.native_policy.seed == 17
    assert snapshot.completed_updates == snapshot.training_examples == source.completed_updates == 0
    assert_tree(torch, snapshot, before_snapshot)
    assert_tree(torch, source.snapshot_state(), before_source)
    assert pickle.dumps((original, variant, canonical)) == inputs_before
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert pickle.dumps(np.random.get_state()) == numpy_rng


class ArraySubclass(np.ndarray):
    pass


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "subclass",
        "float32",
        "one_dimensional",
        "empty",
        "width",
        "late_nan",
        "late_inf",
        "late_range",
        "late_grid",
        "late_direction",
    ],
)
def test_should_refuse_the_whole_malformed_batch_before_any_diagnostic_forward(
    states, monkeypatch, fault
):
    torch, _, snapshots = states
    snapshot = snapshots["pc"]
    before = deepcopy(snapshot)
    values: Any = data()
    if fault == "list":
        values = values.tolist()
    elif fault == "subclass":
        values = values.view(ArraySubclass)
    elif fault == "float32":
        values = values.astype(np.float32)
    elif fault == "one_dimensional":
        values = values.reshape(-1)
    elif fault == "empty":
        values = values[:0]
    elif fault == "width":
        values = values[:, :148].copy()
    elif fault == "late_nan":
        values[1, 151] = float("nan")
    elif fault == "late_inf":
        values[1, 151] = float("inf")
    elif fault == "late_range":
        values[1, 151] = 1.1
    elif fault == "late_grid":
        values[1, 151] += 0.01
    else:
        values[1, -1] -= 0.01
    monkeypatch.setattr(diagnostics, "require_torch", forbidden)
    monkeypatch.setattr(torch, "tanh", forbidden)
    with pytest.raises(ValueError):
        diagnostics.read_native_activations(snapshot, values)
    assert_tree(torch, snapshot, before)


@pytest.mark.parametrize(
    "fault",
    [
        "encoding",
        "policy_extra",
        "native_policy_extra",
        "bool_seed",
        "unknown_method",
        "geometry",
        "native_policy_mismatch",
        "wrapper_extra",
        "subclass",
        "missing",
        "native_extra",
        "bool_clock",
        "native_dtype",
        "native_nonfinite",
        "gradient",
        "optimizer",
        "native_kind",
    ],
)
def test_should_validate_the_complete_bound_contract_before_a_forward(states, monkeypatch, fault):
    torch, _, snapshots = states
    incoming: Any = deepcopy(snapshots["bp"])
    if fault == "encoding":
        object.__setattr__(incoming.policy, "encoding_id", "original_raw_history")
    elif fault == "policy_extra":
        object.__setattr__(incoming.policy, "extra", True)
    elif fault == "native_policy_extra":
        object.__setattr__(incoming.policy.native_policy, "extra", True)
    elif fault == "bool_seed":
        object.__setattr__(incoming.policy.native_policy, "seed", True)
    elif fault == "unknown_method":
        object.__setattr__(incoming.policy.native_policy, "method", "unknown")
    elif fault == "geometry":
        object.__setattr__(
            incoming.policy,
            "native_policy",
            replace(incoming.policy.native_policy, feature_dim=148),
        )
    elif fault == "native_policy_mismatch":
        object.__setattr__(
            incoming.native_state, "policy", replace(incoming.native_state.policy, seed=18)
        )
    elif fault == "wrapper_extra":
        object.__setattr__(incoming, "extra", 0)
    elif fault == "subclass":

        class DerivedSnapshot(InvariantTorchFeatureSnapshot):
            pass

        incoming = DerivedSnapshot(incoming.policy, incoming.native_state)
    elif fault == "missing":
        object.__delattr__(incoming, "native_state")
    elif fault == "native_extra":
        object.__setattr__(incoming.native_state, "extra", True)
    elif fault == "bool_clock":
        object.__setattr__(incoming.native_state, "completed_updates", False)
    elif fault == "native_dtype":
        incoming.native_state.head_state["weight_feature_hidden"] = (
            incoming.native_state.head_state["weight_feature_hidden"].to(dtype=torch.float64)
        )
    elif fault == "native_nonfinite":
        incoming.native_state.head_state["weight_feature_hidden"][0, 0] = float("inf")
    elif fault == "gradient":
        incoming.native_state.gradients["weight_feature_hidden"] = torch.zeros(
            (1, 1), dtype=torch.float32
        )
    elif fault == "optimizer":
        incoming.native_state.optimizer_state["param_groups"][0]["lr"] = 0.04
    else:
        object.__setattr__(incoming, "native_state", {})
    before = deepcopy(vars(incoming))
    monkeypatch.setattr(diagnostics, "require_torch", forbidden)
    monkeypatch.setattr(torch, "tanh", forbidden)
    with pytest.raises(ValueError):
        diagnostics.read_native_activations(incoming, data())
    assert_tree(torch, vars(incoming), before)
