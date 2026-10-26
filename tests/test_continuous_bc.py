"""Bounded engineering checks for continuous BC, with independent numeric oracles.

These synthetic arrays are not a robot experiment or evidence of adaptation.
The module meters every attempted native derivative proposal, including refusal.
"""

from dataclasses import replace
import ctypes
import sys

import numpy as np
import pytest

from src.adapters import numpy_continuous_bc as adapter
from src.core import continuous_bc as core
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.learner_ports import TrainingUpdateFailed
from src.shared.process_memory import _WindowsMemoryCounters, _windows_memory_apis


WORK = {"attempts": 0, "rows": 0, "oracle_forwards": 0, "oracle_rows": 0}


@pytest.fixture(autouse=True)
def bounded_native_work(monkeypatch, record_property):
    """Charge attempts before entering the real proposal, even when it raises."""
    original = core.continuous_bc_loss_and_gradients
    before = dict(WORK)

    def charged(features, *args, **kwargs):
        rows = len(features) if hasattr(features, "__len__") else 0
        WORK["attempts"] += 1
        WORK["rows"] += rows
        assert WORK["attempts"] <= 128
        assert WORK["rows"] <= 4096
        return original(features, *args, **kwargs)

    monkeypatch.setattr(core, "continuous_bc_loss_and_gradients", charged)
    yield
    for name, value in WORK.items():
        record_property("continuous_bc_" + name, value - before[name])
    if sys.platform == "win32":
        get_process, read_memory = _windows_memory_apis()
        counters = _WindowsMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        assert read_memory(get_process(), ctypes.byref(counters), counters.cb)
        peak = int(counters.PeakWorkingSetSize)
        assert peak <= 512 * 1024**2
        record_property("peak_rss_bytes", peak)


def spec(*, constant=False):
    return ContinuousActionSpec(
        observation_dim=2,
        action_units=("rad/s", "N"),
        lower_bounds=(-2.0, 4.0),
        upper_bounds=(6.0, 4.0 if constant else 8.0),
    )


def arrays():
    return (
        np.array([[0.2, -0.3], [0.7, 0.4]], dtype=np.float64),
        np.array([[1.0, 5.0], [4.0, 7.0]], dtype=np.float64),
    )


def parameters():
    return (
        np.array([[0.1, -0.2], [0.3, 0.15]], dtype=np.float64),
        np.array([0.02, -0.03], dtype=np.float64),
        np.array([[0.2, -0.1], [-0.3, 0.25]], dtype=np.float64),
        np.array([0.05, -0.04], dtype=np.float64),
    )


def learner(**changes):
    options = dict(spec=spec(), hidden_dim=2, seed=17, learning_rate=0.03, momentum=0.8)
    options.update(changes)
    return adapter.NumpyContinuousBCLearner(adapter.ContinuousBCPolicy(**options))


def assert_same_state(left, right):
    assert left.policy == right.policy
    assert left.completed_updates == right.completed_updates
    assert left.training_examples == right.training_examples
    for group in ("parameters", "velocity"):
        for first, second in zip(getattr(left, group), getattr(right, group), strict=True):
            np.testing.assert_array_equal(first, second)


def oracle_loss(features, targets, weights, geometry):
    """Independent forward equations only; no call to the derivative under test."""
    WORK["oracle_forwards"] += 1
    WORK["oracle_rows"] += len(features)
    assert WORK["oracle_forwards"] <= 512
    assert WORK["oracle_rows"] <= 4096
    first, bias_first, second, bias_second = weights
    normalized = np.tanh(np.tanh(features @ first + bias_first) @ second + bias_second)
    lower, upper = np.array(geometry.lower_bounds), np.array(geometry.upper_bounds)
    active = upper != lower
    if not np.any(active):
        return 0.0
    center, halfwidth = lower / 2 + upper / 2, upper / 2 - lower / 2
    target_normalized = (targets[:, active] - center[active]) / halfwidth[active]
    return float(0.5 * np.mean((normalized[:, active] - target_normalized) ** 2))


def test_should_match_independent_central_difference_for_every_weight_and_bias():
    features, targets = arrays()
    weights = parameters()
    result = core.continuous_bc_loss_and_gradients(features, targets, weights, spec())
    assert result.loss == pytest.approx(oracle_loss(features, targets, weights, spec()))
    for block, gradient in enumerate(result.gradients):
        finite_difference = np.zeros_like(gradient)
        for index in np.ndindex(gradient.shape):
            plus = tuple(value.copy() for value in weights)
            minus = tuple(value.copy() for value in weights)
            plus[block][index] += 1e-6
            minus[block][index] -= 1e-6
            finite_difference[index] = (
                oracle_loss(features, targets, plus, spec())
                - oracle_loss(features, targets, minus, spec())
            ) / 2e-6
        np.testing.assert_allclose(gradient, finite_difference, rtol=2e-6, atol=2e-9)


def test_should_exclude_constant_axis_from_hand_computed_loss_and_gradient():
    geometry = spec(constant=True)
    features = np.zeros((2, 2), dtype=np.float64)
    targets = np.array([[-2.0, 4.0], [6.0, 4.0]], dtype=np.float64)
    weights = tuple(np.zeros_like(value) for value in parameters())
    result = core.continuous_bc_loss_and_gradients(features, targets, weights, geometry)
    assert result.loss == 0.5
    np.testing.assert_array_equal(result.predictions, [[2.0, 4.0], [2.0, 4.0]])
    for gradient in result.gradients:
        np.testing.assert_array_equal(gradient, np.zeros_like(gradient))
    targets[:, 0] = -2.0
    directed = core.continuous_bc_loss_and_gradients(features, targets, weights, geometry)
    assert directed.loss == 0.5
    np.testing.assert_array_equal(directed.gradients[3], [1.0, 0.0])
    np.testing.assert_array_equal(directed.gradients[2][:, 1], [0.0, 0.0])


def test_should_preserve_physical_bounds_and_own_prediction_and_gradient_arrays():
    features, targets = arrays()
    weights = parameters()
    originals = tuple(value.copy() for value in (*weights, features, targets))
    result = core.continuous_bc_loss_and_gradients(features, targets, weights, spec())
    prediction = core.continuous_bc_predict(features, weights, spec())
    np.testing.assert_array_equal(result.predictions, prediction)
    assert prediction.dtype == np.float64 and prediction.shape == (2, 2)
    assert np.all(prediction >= [-2, 4]) and np.all(prediction <= [6, 8])
    for output in (prediction, result.predictions, *result.gradients):
        assert type(output) is np.ndarray and output.dtype == np.float64
        assert all(not np.shares_memory(output, value) for value in (*weights, features, targets))
    for value, original in zip((*weights, features, targets), originals, strict=True):
        np.testing.assert_array_equal(value, original)


def test_should_apply_independent_two_step_sgd_momentum_oracle():
    model = learner()
    features, targets = arrays()
    first = model.snapshot_state()
    for _ in range(2):
        derivative = core.continuous_bc_loss_and_gradients(
            features, targets, first.parameters, first.policy.spec
        )
        velocity = tuple(
            first.policy.momentum * old + gradient
            for old, gradient in zip(first.velocity, derivative.gradients, strict=True)
        )
        expected = tuple(
            old - first.policy.learning_rate * step
            for old, step in zip(first.parameters, velocity, strict=True)
        )
        diagnostic = model.train_batch(features, targets)
        following = model.snapshot_state()
        assert diagnostic.definition == core.OBJECTIVE
        assert diagnostic.value == pytest.approx(derivative.loss)
        for actual, reference in zip(following.parameters, expected, strict=True):
            np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-15)
        for actual, reference in zip(following.velocity, velocity, strict=True):
            np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-15)
        assert following.completed_updates == first.completed_updates + 1
        assert following.training_examples == first.training_examples + 2
        first = following


def test_should_use_local_seed_without_changing_global_numpy_rng():
    before = np.random.get_state()
    first, second, different = learner(), learner(), learner(seed=18)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    assert_same_state(first.snapshot_state(), second.snapshot_state())
    assert any(
        not np.array_equal(left, right)
        for left, right in zip(
            first.snapshot_state().parameters, different.snapshot_state().parameters, strict=True
        )
    )


def test_should_continue_exactly_after_snapshot_restore_and_fork_with_velocity():
    model = learner()
    features, targets = arrays()
    model.train_batch(features, targets)
    saved = model.snapshot_state()
    assert any(np.any(value) for value in saved.velocity)
    restored = learner()
    restored.restore_state(saved)
    forked = model.fork()
    for candidate in (model, restored, forked):
        candidate.train_batch(features, targets)
    assert_same_state(model.snapshot_state(), restored.snapshot_state())
    assert_same_state(model.snapshot_state(), forked.snapshot_state())
    untouched = model.snapshot_state()
    for value in (*saved.parameters, *saved.velocity):
        value[:] = 100
    assert_same_state(untouched, model.snapshot_state())
    assert_same_state(untouched, restored.snapshot_state())
    forked.train_batch(features, targets)
    assert model.snapshot_state().completed_updates == 2


def test_should_count_all_constant_training_without_parameter_updates():
    geometry = ContinuousActionSpec(2, ("m", "N"), (3.0, 4.0), (3.0, 4.0))
    model = learner(spec=geometry)
    before = model.snapshot_state()
    features, _ = arrays()
    result = model.train_batch(features, np.array([[3.0, 4.0], [3.0, 4.0]]))
    after = model.snapshot_state()
    assert result.value == 0
    assert after.completed_updates == 1 and after.training_examples == 2
    for original, following in zip(before.parameters, after.parameters, strict=True):
        np.testing.assert_array_equal(original, following)
    np.testing.assert_array_equal(model.predict(features), [[3.0, 4.0], [3.0, 4.0]])


def test_should_admit_exact_maximum_batch_and_detach_inputs():
    model = learner()
    features = np.zeros((256, 2), dtype=np.float64)
    targets = np.tile(np.array([[2.0, 6.0]], dtype=np.float64), (256, 1))
    model.train_batch(features, targets)
    before = model.snapshot_state()
    features[:] = 20
    targets[:] = -20
    assert_same_state(before, model.snapshot_state())
    assert before.training_examples == 256 and before.completed_updates == 1


@pytest.mark.parametrize(
    "options",
    [
        {"hidden_dim": 0},
        {"hidden_dim": -1},
        {"hidden_dim": True},
        {"hidden_dim": 2.0},
        {"learning_rate": 0},
        {"learning_rate": -0.1},
        {"learning_rate": np.inf},
        {"learning_rate": np.nan},
        {"momentum": -0.1},
        {"momentum": 1},
        {"momentum": np.nan},
        {"training_role": "probe_test"},
        {"training_role": "final_test"},
        {"training_role": "unknown"},
        {"seed": -1},
        {"seed": 2**32},
        {"seed": True},
        {"hidden_dim": 257},
        {"learning_rate": 1.01},
    ],
)
def test_should_refuse_invalid_policy_before_constructing_learner(options):
    with pytest.raises(ValueError):
        learner(**options)


@pytest.mark.parametrize(
    "bad", ["float32", "integer", "list", "nan", "inf", "width", "empty", "oversize"]
)
def test_should_refuse_malformed_features_without_changing_state(bad):
    features, targets = arrays()
    if bad == "float32":
        features = features.astype(np.float32)
    elif bad == "integer":
        features = features.astype(np.int64)
    elif bad == "list":
        features = features.tolist()
    elif bad in ("nan", "inf"):
        features[0, 0] = np.nan if bad == "nan" else np.inf
    elif bad == "width":
        features = features[:, :1]
    elif bad == "empty":
        features, targets = features[:0], targets[:0]
    else:
        features, targets = np.zeros((257, 2)), np.tile(targets[:1], (257, 1))
    model = learner()
    before = model.snapshot_state()
    with pytest.raises(ValueError):
        model.train_batch(features, targets)
    assert_same_state(before, model.snapshot_state())


@pytest.mark.parametrize("bad", ["dtype", "width", "rows", "nan", "below", "above"])
def test_should_refuse_malformed_or_out_of_bounds_targets_atomically(bad):
    features, targets = arrays()
    if bad == "dtype":
        targets = targets.astype(np.float32)
    elif bad == "width":
        targets = targets[:, :1]
    elif bad == "rows":
        targets = targets[:1]
    else:
        targets[0, 0] = {"nan": np.nan, "below": -2.01, "above": 6.01}[bad]
    model = learner()
    before = model.snapshot_state()
    with pytest.raises(ValueError):
        model.train_batch(features, targets)
    assert_same_state(before, model.snapshot_state())


@pytest.mark.parametrize("bad", ["dtype", "shape", "nan", "list", "length", "overflow"])
def test_should_refuse_invalid_core_parameters_and_nonfinite_arithmetic(bad):
    features, targets = arrays()
    weights = list(parameters())
    if bad == "dtype":
        weights[0] = weights[0].astype(np.float32)
    elif bad == "shape":
        weights[0] = weights[0][:1]
    elif bad == "nan":
        weights[0][0, 0] = np.nan
    elif bad == "list":
        weights[0] = weights[0].tolist()
    elif bad == "length":
        weights.pop()
    else:
        features[:] = 1e308
        weights[0][:] = 1e308
    with pytest.raises(FloatingPointError if bad == "overflow" else ValueError):
        core.continuous_bc_loss_and_gradients(features, targets, tuple(weights), spec())


def episode(geometry, role="train"):
    transition = ContinuousTransition(
        0, (0.2, -0.3), geometry.lower_bounds, 0, (0.3, -0.2), True, False
    )
    return ContinuousEpisode("engineering-1", "synthetic", role, geometry, (transition,))


def test_should_train_only_matching_episode_role_and_full_declared_spec():
    model = learner()
    model.train_episode(episode(spec()))
    assert model.snapshot_state().completed_updates == 1
    before = model.snapshot_state()
    mismatched = replace(spec(), action_units=("m/s", "N"))
    for value in (
        episode(spec(), "probe_train"),
        episode(spec(), "probe_test"),
        episode(spec(), "final_test"),
        episode(mismatched),
    ):
        with pytest.raises(ValueError):
            model.train_episode(value)
        assert_same_state(before, model.snapshot_state())
    isolated = learner(training_role="probe_train")
    isolated.train_episode(episode(spec(), "probe_train"))
    assert isolated.snapshot_state().training_examples == 1


@pytest.mark.parametrize(
    "bad",
    [
        "parameter_dtype",
        "parameter_shape",
        "parameter_nan",
        "velocity_shape",
        "velocity_inf",
        "negative_updates",
        "bool_updates",
        "bad_examples",
        "oversize_updates",
        "policy",
    ],
)
def test_should_refuse_corrupted_restore_without_partially_replacing_state(bad):
    model = learner()
    features, targets = arrays()
    model.train_batch(features, targets)
    before = model.snapshot_state()
    corrupted = model.snapshot_state()
    if bad.startswith("parameter"):
        values = list(corrupted.parameters)
        if bad == "parameter_dtype":
            values[0] = values[0].astype(np.float32)
        elif bad == "parameter_shape":
            values[0] = values[0][:1]
        else:
            values[0][0, 0] = np.nan
        object.__setattr__(corrupted, "parameters", tuple(values))
    elif bad.startswith("velocity"):
        values = list(corrupted.velocity)
        if bad == "velocity_shape":
            values[0] = values[0][:1]
        else:
            values[0][0, 0] = np.inf
        object.__setattr__(corrupted, "velocity", tuple(values))
    elif bad == "policy":
        object.__setattr__(corrupted, "policy", replace(corrupted.policy, learning_rate=0.02))
    else:
        field, value = {
            "negative_updates": ("completed_updates", -1),
            "bool_updates": ("completed_updates", True),
            "bad_examples": ("training_examples", 0),
            "oversize_updates": ("completed_updates", 2**53),
        }[bad]
        object.__setattr__(corrupted, field, value)
    with pytest.raises(ValueError):
        model.restore_state(corrupted)
    assert_same_state(before, model.snapshot_state())


def test_should_own_policy_and_refuse_different_units_even_with_matching_dimensions():
    original = adapter.ContinuousBCPolicy(spec(), hidden_dim=2)
    model = adapter.NumpyContinuousBCLearner(original)
    before = model.snapshot_state()
    object.__setattr__(original.spec, "action_units", ("incorrect", "units"))
    assert model.snapshot_state().policy.spec.action_units == ("rad/s", "N")
    detached_policy = model.policy
    object.__setattr__(detached_policy, "learning_rate", 0.75)
    assert_same_state(before, model.snapshot_state())


def test_should_preserve_model_state_when_the_native_proposal_fails(monkeypatch):
    model = learner()
    before = model.snapshot_state()
    features, targets = arrays()
    charged = core.continuous_bc_loss_and_gradients

    def failing(*args, **kwargs):
        # Charge and execute the admitted proposal before injecting a failure.
        charged(*args, **kwargs)
        raise ValueError("scripted failure after derivative proposal")

    monkeypatch.setattr(adapter.core, "continuous_bc_loss_and_gradients", failing)
    with pytest.raises(TrainingUpdateFailed, match="scripted failure"):
        model.train_batch(features, targets)
    assert_same_state(before, model.snapshot_state())


def test_should_preserve_full_state_when_the_actual_momentum_proposal_overflows():
    model = learner(learning_rate=1.0, momentum=0.9)
    state = model.snapshot_state()
    state.parameters[3][:] = -np.finfo(np.float64).max
    state.velocity[3][:] = np.finfo(np.float64).max
    model.restore_state(replace(state, completed_updates=1, training_examples=1))
    before = model.snapshot_state()
    features = np.zeros((1, 2), dtype=np.float64)
    targets = np.array([spec().lower_bounds], dtype=np.float64)
    with pytest.raises(TrainingUpdateFailed) as caught:
        model.train_batch(features, targets)
    assert caught.value.work["gradient_evaluations_completed"] == 1
    assert caught.value.work["parameter_proposals_attempted"] == 1
    assert caught.value.work["parameter_proposals_completed"] == 0
    assert caught.value.work["published_updates"] == 0
    assert_same_state(before, model.snapshot_state())


def test_should_reject_unrepresentable_halfwidth_but_predict_at_huge_finite_bounds():
    tiny = ContinuousActionSpec(2, ("m",), (0.0,), (np.nextafter(0.0, 1.0).item(),))
    with pytest.raises(ValueError):
        adapter.ContinuousBCPolicy(tiny)
    huge = np.finfo(np.float64).max.item()
    geometry = ContinuousActionSpec(2, ("m",), (-huge,), (huge,))
    for sign in (-1.0, 1.0):
        weights = (np.zeros((2, 1)), np.zeros(1), np.zeros((1, 1)), np.array([sign * 1000]))
        predicted = core.continuous_bc_predict(np.zeros((1, 2)), weights, geometry)
        assert predicted[0, 0] == sign * huge


@pytest.mark.parametrize("field", ["learning_rate", "momentum"])
def test_should_reject_overflow_sized_native_policy_integers_as_invalid_values(field):
    with pytest.raises(ValueError):
        learner(**{field: 10**1000})
