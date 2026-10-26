"""Native hidden-state readouts preserve learned state and exact float32 math.

Qualification is deliberately bounded: nine one-update, nine-row cases; the
remaining cases use zero-update snapshots and never train or call predict.
"""

from copy import deepcopy
from typing import Any

import numpy as np
import pytest

from src.adapters.torch_activation_diagnostics import read_native_activations
from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.adapters.torch_recurrent_learner import TorchRecurrentLearner, TorchRecurrentPolicy
from src.shared.torch_runtime import require_torch


@pytest.fixture(scope="module")
def torch_runtime():
    torch = require_torch()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    return torch


def learner(method, width):
    if method == "recurrent_bc":
        return TorchRecurrentLearner(TorchRecurrentPolicy(width, 8, 7, 17))
    return TorchFeatureLearner(TorchFeaturePolicy(method, width, 8, 17, num_classes=7))


def features(width, rows=9):
    return np.linspace(-0.7, 0.6, rows * width, dtype=np.float64).reshape(rows, width)


def manual_hidden(torch, snapshot, values):
    """Independent native-precision oracle, including each recurrent state."""
    inputs = torch.from_numpy(values.astype(np.float32))
    with torch.no_grad():
        if hasattr(snapshot, "head_state"):
            state = snapshot.head_state
            hidden = torch.tanh(inputs @ state["weight_feature_hidden"] + state["bias_hidden"])
            logits = hidden @ state["weight_hidden_output"] + state["bias_output"]
        else:
            state = snapshot.parameters
            hidden_row = torch.zeros((1, snapshot.policy.hidden_dim), dtype=torch.float32)
            rows = []
            for row in inputs:
                hidden_row = torch.tanh(
                    row.reshape(1, -1) @ state["weight_input"]
                    + hidden_row @ state["weight_recurrent"]
                    + state["bias_hidden"]
                )
                rows.append(hidden_row)
            hidden = torch.cat(rows)
            logits = hidden @ state["weight_output"] + state["bias_output"]
        return (
            hidden.numpy().astype(np.float64, copy=True),
            torch.softmax(logits, dim=1).numpy().astype(np.float64, copy=True),
        )


def read_preserving(torch, source, snapshot, values):
    source_before, snapshot_before = source.snapshot_state(), deepcopy(snapshot)
    input_before = values.copy()
    torch_rng = torch.get_rng_state().clone()
    numpy_rng: Any = deepcopy(np.random.get_state())
    result = read_native_activations(snapshot, values)
    assert _same_state(snapshot, snapshot_before)
    assert _same_state(source.snapshot_state(), source_before)
    assert torch.equal(torch.get_rng_state(), torch_rng)
    actual_rng: Any = np.random.get_state()
    assert actual_rng[0] == numpy_rng[0] and actual_rng[2:] == numpy_rng[2:]
    np.testing.assert_array_equal(actual_rng[1], numpy_rng[1])
    np.testing.assert_array_equal(values, input_before)
    assert result.dtype == np.float64 and result.flags.owndata
    assert result.shape == (len(values), 8) and not np.shares_memory(result, values)
    return result


@pytest.mark.parametrize(
    "method,width",
    [(method, width) for method in ("bp", "pc", "neutral_cpc", "gated_cpc") for width in (148, 296)]
    + [("recurrent_bc", 148)],
)
def test_should_match_native_hidden_and_prediction_before_and_after_one_update(
    torch_runtime, method, width
):
    torch = torch_runtime
    source, values = learner(method, width), features(width)
    targets = (np.arange(9, dtype=np.float64) % 7).reshape(-1, 1)
    for update in (0, 1):
        snapshot = source.snapshot_state()
        assert snapshot.completed_updates == update and snapshot.training_examples == 9 * update
        expected, probabilities = manual_hidden(torch, snapshot, values)
        actual = read_preserving(torch, source, snapshot, values)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(source.predict(values), probabilities, rtol=1e-6, atol=1e-7)
        if not update:
            source.train_batch(values, targets)
    actual[:] = 123
    owned = read_preserving(torch, source, snapshot, values)
    np.testing.assert_array_equal(owned, expected)


@pytest.fixture(scope="module")
def zero_states(torch_runtime):
    return {
        "feature": learner("bp", 296).snapshot_state(),
        "recurrent": learner("recurrent_bc", 148).snapshot_state(),
    }


def test_should_carry_recurrence_only_within_the_ordered_episode_and_reset_on_each_call(
    zero_states,
):
    state = zero_states["recurrent"]
    values = features(148)
    original = read_native_activations(state, values)
    prefix = read_native_activations(state, values[:4].copy())
    changed = values.copy()
    changed[4:] *= -1
    suffix_changed = read_native_activations(state, changed)
    np.testing.assert_array_equal(prefix, original[:4])
    np.testing.assert_array_equal(suffix_changed[:4], original[:4])
    assert not np.array_equal(suffix_changed[4:], original[4:])
    np.testing.assert_array_equal(read_native_activations(state, values), original)


@pytest.mark.parametrize("method", ["bp", "recurrent_bc"])
def test_should_preserve_small_toy_widths_without_a_minigrid_special_case(torch_runtime, method):
    source = learner(method, 2)
    values = features(2, rows=3)
    expected, _ = manual_hidden(torch_runtime, source.snapshot_state(), values)
    np.testing.assert_array_equal(
        read_native_activations(source.snapshot_state(), values), expected
    )


@pytest.mark.parametrize("kind", ["feature", "recurrent"])
@pytest.mark.parametrize(
    "fault", ["dtype", "list", "empty", "width", "nan", "inf", "float32_overflow"]
)
def test_should_refuse_invalid_features_without_changing_snapshot(zero_states, kind, fault):
    snapshot = zero_states[kind]
    before = deepcopy(snapshot)
    values: Any = features(snapshot.policy.feature_dim)
    if fault == "dtype":
        values = values.astype(np.float32)
    elif fault == "list":
        values = values.tolist()
    elif fault == "empty":
        values = values[:0]
    elif fault == "width":
        values = values[:, :-1].copy()
    else:
        values[0, 0] = {"nan": np.nan, "inf": np.inf, "float32_overflow": 1e300}[fault]
    with pytest.raises((ValueError, FloatingPointError)):
        read_native_activations(snapshot, values)
    assert _same_state(snapshot, before)


@pytest.mark.parametrize("kind", ["feature", "recurrent"])
@pytest.mark.parametrize(
    "fault",
    [
        "extra",
        "missing",
        "subclass",
        "counter",
        "policy",
        "dtype",
        "nonfinite",
        "optimizer",
        "gradient",
    ],
)
def test_should_revalidate_complete_snapshot_before_reading_activations(
    torch_runtime, zero_states, kind, fault
):
    snapshot = deepcopy(zero_states[kind])
    parameters = snapshot.head_state if kind == "feature" else snapshot.parameters
    key = "weight_feature_hidden" if kind == "feature" else "weight_input"
    if fault == "subclass":
        derived = type("DerivedSnapshot", (type(snapshot),), {})
        replacement: Any = object.__new__(derived)
        replacement.__dict__.update(vars(snapshot))
        snapshot = replacement
    elif fault == "extra":
        object.__setattr__(snapshot, "untrusted", True)
    elif fault == "missing":
        object.__delattr__(snapshot, "format_version")
    elif fault == "counter":
        object.__setattr__(snapshot, "completed_updates", True)
    elif fault == "policy":
        object.__setattr__(snapshot.policy, "seed", True)
    elif fault == "dtype":
        parameters[key] = parameters[key].to(dtype=torch_runtime.float64)
    elif fault == "nonfinite":
        parameters[key][0, 0] = float("inf")
    elif fault == "optimizer":
        snapshot.optimizer_state["param_groups"][0]["lr"] = 0.04
    else:
        snapshot.gradients[key] = torch_runtime.zeros_like(
            parameters[key], dtype=torch_runtime.float64
        )
    before = deepcopy(vars(snapshot))
    with pytest.raises(ValueError):
        read_native_activations(snapshot, features(zero_states[kind].policy.feature_dim))
    assert _same_state(vars(snapshot), before)


@pytest.mark.parametrize("invalid", [None, {}, object()])
def test_should_refuse_unrecognized_snapshot_objects(invalid):
    with pytest.raises(ValueError, match="exact"):
        read_native_activations(invalid, features(148))


@pytest.mark.parametrize("kind", ["feature", "recurrent"])
def test_should_refuse_finite_affine_overflow_before_tanh_can_hide_it(zero_states, kind):
    snapshot = deepcopy(zero_states[kind])
    parameters = snapshot.head_state if kind == "feature" else snapshot.parameters
    key = "weight_feature_hidden" if kind == "feature" else "weight_input"
    parameters[key].fill_(1e38)
    values = np.full((1, snapshot.policy.feature_dim), 10.0, dtype=np.float64)
    before = deepcopy(snapshot)
    with pytest.raises(FloatingPointError, match="preactivation"):
        read_native_activations(snapshot, values)
    assert _same_state(snapshot, before)
