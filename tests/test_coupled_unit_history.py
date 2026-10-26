"""Independent fixed-unit history arithmetic and complete native continuation.

Tiny native examples establish where the statistic is measured. Malformed state
and counter boundaries use pure history records, avoiding unnecessary training.
"""

from dataclasses import FrozenInstanceError, fields
import pickle
from typing import Any

import numpy as np
import pytest

import src.core.tracked_coupled_predictive_coding as tracked
from src.adapters.coupled_checkpoint_codec import CoupledCheckpointCodec
from src.core.checkpoint_codec import CodecBinding, CodecLimits
from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.coupled_energy import CoupledParameters, evaluate_coupled, feedforward_latents
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork, CoupledSnapshot
from src.core.coupled_unit_history import (
    UnitHistory,
    advance_unit_history,
    copy_unit_history,
    initial_unit_history,
)
from src.core.tracked_coupled_predictive_coding import (
    TrackedCoupledCircadianPredictiveCodingNetwork,
    TrackedCoupledPredictiveCodingNetwork,
    TrackedCoupledSnapshot,
)


MAX_UPDATES = 2**53 - 1


def batch():
    return np.array([[1.0, -0.5], [0.2, 0.7]]), np.array([[0.0], [1.0]])


def make_model(family, *, decay=0.5, seed=23, tracked_model=True):
    if family == "pc":
        constructor: Any = (
            TrackedCoupledPredictiveCodingNetwork
            if tracked_model
            else CoupledPredictiveCodingNetwork
        )
        options = {}
    else:
        constructor = (
            TrackedCoupledCircadianPredictiveCodingNetwork
            if tracked_model
            else CoupledCircadianPredictiveCodingNetwork
        )
        options = {
            "config": (
                CoupledCircadianConfig.matched_pc_control()
                if family == "neutral"
                else CoupledCircadianConfig(chemical_buildup_rate=1.0, plasticity_sensitivity=5.0)
            )
        }
    if tracked_model:
        options["history_decay"] = decay
    return constructor(2, (3, 2), seed=seed, precisions=(0.7, 1.3), **options)


def parameter_arrays(parameters):
    return (
        *parameters.hidden_weights,
        *parameters.hidden_biases,
        parameters.output_weight,
        parameters.output_bias,
    )


def assert_history(actual, expected):
    assert type(actual) is UnitHistory
    assert actual.format_version == expected.format_version == 1
    assert actual.ema_decay == expected.ema_decay
    assert actual.updates == expected.updates
    for name in ("unit_ids", "importance_ema", "update_age"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


def assert_native(actual, expected):
    assert actual.precisions == expected.precisions
    for left, right in zip(
        parameter_arrays(actual.parameters), parameter_arrays(expected.parameters)
    ):
        np.testing.assert_array_equal(left, right)
    if expected.wake_state is None:
        assert actual.wake_state is None
    else:
        assert actual.wake_state is not None
        for field in fields(expected.wake_state):
            left, right = (
                getattr(actual.wake_state, field.name),
                getattr(expected.wake_state, field.name),
            )
            if type(right) is np.ndarray:
                np.testing.assert_array_equal(left, right)
            else:
                assert left == right


def assert_complete(actual, expected):
    assert type(actual) is type(expected) is TrackedCoupledSnapshot
    assert actual.format_version == expected.format_version == 2
    assert_native(actual, expected)
    assert_history(actual.unit_history, expected.unit_history)


def raw_gradient(snapshot, features, targets):
    """Reproduce simultaneous fixed inference, independently of training hooks."""
    latents = feedforward_latents(snapshot.parameters, features)
    for _ in range(2):
        evaluation = evaluate_coupled(
            snapshot.parameters, features, targets, latents, snapshot.precisions
        )
        latents = tuple(
            value - 0.02 * gradient for value, gradient in zip(latents, evaluation.latent_gradients)
        )
    return evaluate_coupled(
        snapshot.parameters, features, targets, latents, snapshot.precisions
    ).parameter_gradients.output_weight


@pytest.mark.parametrize("width", [1, 3, 4096])
def test_should_start_with_stable_ids_zero_history_and_linear_storage(width):
    history = initial_unit_history(width)
    assert history.format_version == 1
    assert history.ema_decay == 0.95
    assert history.updates == 0
    assert history.unit_ids.dtype == np.int64
    assert history.importance_ema.dtype == history.update_age.dtype == np.float64
    np.testing.assert_array_equal(history.unit_ids, np.arange(width, dtype=np.int64))
    np.testing.assert_array_equal(history.importance_ema, np.zeros(width))
    np.testing.assert_array_equal(history.update_age, np.zeros(width))
    assert (
        sum(
            array.nbytes for array in (history.unit_ids, history.importance_ema, history.update_age)
        )
        == 24 * width
    )
    assert set(vars(history)) == {
        "format_version",
        "ema_decay",
        "unit_ids",
        "importance_ema",
        "update_age",
        "updates",
    }
    with pytest.raises(FrozenInstanceError):
        setattr(history, "updates", 1)


@pytest.mark.parametrize("decay", [0.0, 0.5, 0.95, np.nextafter(1.0, 0.0).item()])
def test_should_match_independent_ema_recurrence_and_leave_inputs_owned(decay):
    history = initial_unit_history(3, decay)
    expected = np.zeros(3)
    old = []
    for index, values in enumerate(([2.0, -4.0, 0.0], [-1.0, 3.0, -8.0], [0.0, 0.0, 2.0]), 1):
        gradient = np.array(values).reshape(3, 1)
        before = gradient.copy()
        previous = copy_unit_history(history, 3, decay)
        old.append((history, previous))
        history = advance_unit_history(history, gradient)
        expected = decay * expected + (1 - decay) * np.abs(np.array(values))
        np.testing.assert_allclose(history.importance_ema, expected, rtol=2e-15, atol=0)
        np.testing.assert_array_equal(history.update_age, np.full(3, float(index)))
        np.testing.assert_array_equal(history.unit_ids, [0, 1, 2])
        np.testing.assert_array_equal(gradient, before)
        assert history.updates == index
        for name in ("unit_ids", "importance_ema", "update_age"):
            assert not np.shares_memory(getattr(history, name), getattr(previous, name))
    for original, saved in old:
        assert_history(original, saved)
    copied = copy_unit_history(history, 3, decay)
    copied.importance_ema[:] = 99
    copied.update_age[:] = 99
    copied.unit_ids[:] = 99
    np.testing.assert_array_equal(history.unit_ids, [0, 1, 2])
    np.testing.assert_array_equal(history.update_age, [3, 3, 3])
    np.testing.assert_allclose(history.importance_ema, expected)


@pytest.mark.parametrize("width", [True, 0, -1, 4097, 2.0, np.int64(2), None])
def test_should_refuse_invalid_width_before_allocating_history(width):
    with pytest.raises(ValueError):
        initial_unit_history(width)


@pytest.mark.parametrize(
    "decay", [True, -0.1, 1.0, float("inf"), float("nan"), "0.5", np.float64(0.5)]
)
def test_should_refuse_non_native_or_out_of_domain_decay(decay):
    with pytest.raises(ValueError):
        initial_unit_history(2, decay)


@pytest.mark.parametrize(
    "fault",
    [
        "format",
        "format_bool",
        "ids_dtype",
        "ids_order",
        "ids_duplicate",
        "ids_shape",
        "importance_dtype",
        "importance_negative",
        "importance_nan",
        "age_dtype",
        "age_fraction",
        "age_mismatch",
        "updates_bool",
        "updates_overflow",
        "zero_importance",
        "policy",
    ],
)
def test_should_revalidate_complete_history_before_copy_or_advance(fault):
    history = initial_unit_history(2, 0.5)
    values: dict[str, Any] = {
        "format": ("format_version", 2),
        "format_bool": ("format_version", True),
        "ids_dtype": ("unit_ids", np.array([0, 1], dtype=np.int32)),
        "ids_order": ("unit_ids", np.array([1, 0], dtype=np.int64)),
        "ids_duplicate": ("unit_ids", np.array([0, 0], dtype=np.int64)),
        "ids_shape": ("unit_ids", np.array([[0, 1]], dtype=np.int64)),
        "importance_dtype": ("importance_ema", np.zeros(2, dtype=np.float32)),
        "importance_negative": ("importance_ema", np.array([-1.0, 0.0])),
        "importance_nan": ("importance_ema", np.array([float("nan"), 0.0])),
        "age_dtype": ("update_age", np.zeros(2, dtype=np.int64)),
        "age_fraction": ("update_age", np.array([0.5, 0.5])),
        "age_mismatch": ("update_age", np.array([0.0, 1.0])),
        "updates_bool": ("updates", True),
        "updates_overflow": ("updates", MAX_UPDATES + 1),
        "zero_importance": ("importance_ema", np.array([1.0, 0.0])),
        "policy": ("ema_decay", 0.7),
    }
    name, value = values[fault]
    object.__setattr__(history, name, value)
    with pytest.raises(ValueError):
        copy_unit_history(history, 2, 0.5)
    # A different valid decay is still internally consistent for advance, which
    # has no external policy argument. The model-bound copy enforces that policy.
    if fault != "policy":
        with pytest.raises(ValueError):
            advance_unit_history(history, np.ones((2, 1)))


@pytest.mark.parametrize(
    "gradient",
    [
        [[1.0], [2.0]],
        np.ones(2),
        np.ones((1, 2)),
        np.ones((2, 1), dtype=np.float32),
        np.ones((2, 1), dtype=np.int64),
        np.array([[float("nan")], [0.0]]),
        np.array([[float("inf")], [0.0]]),
    ],
)
def test_should_refuse_bad_outgoing_gradient_without_changing_history(gradient):
    history = initial_unit_history(2)
    saved = copy_unit_history(history, 2, 0.95)
    with pytest.raises(ValueError):
        advance_unit_history(history, gradient)
    assert_history(history, saved)


def test_should_refuse_foreign_records_width_and_policy_without_copying():
    history = initial_unit_history(2, 0.5)
    cases: tuple[tuple[Any, int, float], ...] = (
        (object(), 2, 0.5),
        (history, 1, 0.5),
        (history, 2, 0.7),
    )
    for value, width, decay in cases:
        with pytest.raises(ValueError):
            copy_unit_history(value, width, decay)


def test_should_accept_last_exact_age_but_refuse_another_update_atomically():
    history = initial_unit_history(2, 0.5)
    object.__setattr__(history, "updates", MAX_UPDATES)
    object.__setattr__(history, "update_age", np.full(2, float(MAX_UPDATES)))
    saved = copy_unit_history(history, 2, 0.5)
    with pytest.raises(ValueError):
        advance_unit_history(history, np.ones((2, 1)))
    assert_history(history, saved)


def test_should_compute_large_finite_convex_ema_without_intermediate_overflow():
    history = advance_unit_history(initial_unit_history(2, 0.5), np.full((2, 1), 1e308))
    history = advance_unit_history(history, np.full((2, 1), -1e308))
    np.testing.assert_allclose(history.importance_ema, [7.5e307, 7.5e307])
    assert history.updates == 2


@pytest.mark.parametrize("family", ["pc", "neutral", "active"])
def test_should_preserve_default_native_math_when_history_is_enabled(family):
    ordinary, model = make_model(family, tracked_model=False), make_model(family)
    assert model.unit_history_decay == 0.5
    features, targets = batch()
    for index in range(1, 3):
        expected = ordinary.train_epoch(features, targets, 0.01, 2, 0.02)
        actual = model.train_epoch(features, targets, 0.01, 2, 0.02)
        assert actual == expected
        assert_native(model.snapshot_state(), ordinary.snapshot_state())
        history = model.read_unit_history()
        assert history.updates == index
        np.testing.assert_array_equal(history.update_age, [index, index])
    np.testing.assert_array_equal(model.predict_proba(features), ordinary.predict_proba(features))


def test_should_track_absolute_aggregate_gradient_after_batch_cancellation():
    model = TrackedCoupledPredictiveCodingNetwork(1, (1,), seed=23, history_decay=0.0)
    model.restore_parameters(
        CoupledParameters(
            (np.zeros((1, 1)),), (np.ones((1, 1)),), np.zeros((1, 1)), np.zeros((1, 1))
        )
    )
    features, targets = np.zeros((2, 1)), np.array([[0.0], [1.0]])
    before = model.snapshot_state()
    latents = feedforward_latents(before.parameters, features)
    probabilities = model.predict_proba(features)
    per_example = latents[-1] * (probabilities - targets)
    assert float(np.mean(np.abs(per_example))) > 0
    np.testing.assert_array_equal(np.mean(per_example, axis=0), [0.0])
    model.train_epoch(features, targets, 0.01, 2, 0.02)
    np.testing.assert_array_equal(model.read_unit_history().importance_ema, [0.0])


def test_should_measure_raw_outgoing_gradient_before_active_circadian_gates():
    model = make_model("active")
    features, targets = batch()
    expected = np.zeros(2)
    for _ in range(2):
        before = model.snapshot_state()
        gradient = raw_gradient(before, features, targets)
        model.train_epoch(features, targets, 0.01, 2, 0.02)
        after = model.snapshot_state()
        expected = 0.5 * expected + 0.5 * np.abs(gradient[:, 0])
        np.testing.assert_allclose(after.unit_history.importance_ema, expected, rtol=2e-15)
        assert np.all(after.wake_state.plasticity_state < 1)
        np.testing.assert_allclose(
            after.parameters.output_weight,
            before.parameters.output_weight
            - 0.01 * gradient * after.wake_state.plasticity_state[:, None],
            rtol=2e-15,
        )
        assert np.any(
            np.abs(gradient[:, 0]) > np.abs(gradient[:, 0] * after.wake_state.plasticity_state)
        )


def test_should_keep_neutral_cpc_and_pc_histories_exact_across_updates():
    pc, cpc = make_model("pc"), make_model("neutral")
    features, targets = batch()
    for _ in range(3):
        assert pc.train_epoch(features, targets, 0.01, 2, 0.02) == cpc.train_epoch(
            features, targets, 0.01, 2, 0.02
        )
        for left, right in zip(
            parameter_arrays(pc.snapshot_parameters()), parameter_arrays(cpc.snapshot_parameters())
        ):
            np.testing.assert_array_equal(left, right)
        assert_history(pc.read_unit_history(), cpc.read_unit_history())


@pytest.mark.parametrize("family", ["pc", "active"])
@pytest.mark.parametrize("fault", ["input", "candidate", "history"])
def test_should_leave_native_and_history_state_unchanged_on_failed_update(
    monkeypatch, family, fault
):
    model = make_model(family)
    before = model.snapshot_state()
    features, targets = batch()
    problem = RuntimeError("prospective history refused")
    if fault == "input":
        targets[0, 0] = float("nan")
    elif fault == "candidate":

        def fail_candidate(*args, **kwargs):
            raise problem

        monkeypatch.setattr(model, "_prepare_parameters", fail_candidate)
    else:

        def fail_history(*args, **kwargs):
            raise problem

        monkeypatch.setattr(tracked, "advance_unit_history", fail_history)
    with pytest.raises(ValueError if fault == "input" else RuntimeError) as caught:
        model.train_epoch(features, targets, 0.01, 2, 0.02)
    if fault != "input":
        assert caught.value is problem
    assert_complete(model.snapshot_state(), before)


@pytest.mark.parametrize("family", ["pc", "active"])
def test_should_preserve_history_through_predictions_and_parameter_only_restore(family):
    model = make_model(family)
    features, targets = batch()
    model.train_epoch(features, targets, 0.01, 2, 0.02)
    before = model.snapshot_state()
    parameters = model.snapshot_parameters()
    parameters.output_bias[:] += 0.25
    model.predict_proba(features)
    model.restore_parameters(parameters)
    np.testing.assert_array_equal(model.snapshot_parameters().output_bias, parameters.output_bias)
    assert_history(model.read_unit_history(), before.unit_history)
    if before.wake_state is not None:
        after = model.snapshot_state()
        assert after.wake_state.wake_updates == before.wake_state.wake_updates
        np.testing.assert_array_equal(
            after.wake_state.chemical_state, before.wake_state.chemical_state
        )
    parameters.output_bias[:] = 99
    assert not np.any(model.snapshot_parameters().output_bias == 99)
    read = model.read_unit_history()
    read.importance_ema[:] = 99
    assert_history(model.read_unit_history(), before.unit_history)


@pytest.mark.parametrize("family", ["pc", "active"])
def test_should_own_complete_snapshot_and_continue_exactly_after_restore(family):
    original = make_model(family)
    features, targets = batch()
    original.train_epoch(features, targets, 0.01, 2, 0.02)
    snapshot = original.snapshot_state()
    restored = make_model(family, seed=59)
    restored.restore_state(snapshot)
    assert_complete(restored.snapshot_state(), snapshot)
    for name in ("unit_ids", "importance_ema", "update_age"):
        left, right = (
            getattr(snapshot.unit_history, name),
            getattr(restored.read_unit_history(), name),
        )
        assert not np.shares_memory(left, right)
    for left, right in zip(
        parameter_arrays(snapshot.parameters), parameter_arrays(restored.snapshot_parameters())
    ):
        assert not np.shares_memory(left, right)
    assert original.train_epoch(features, targets, 0.01, 2, 0.02) == restored.train_epoch(
        features, targets, 0.01, 2, 0.02
    )
    assert_complete(original.snapshot_state(), restored.snapshot_state())
    saved = original.snapshot_state()
    snapshot.parameters.output_weight[:] = 99
    snapshot.unit_history.importance_ema[:] = 99
    if snapshot.wake_state is not None:
        snapshot.wake_state.chemical_state[:] = 99
    assert_complete(original.snapshot_state(), saved)
    assert_complete(restored.snapshot_state(), saved)


@pytest.mark.parametrize("family", ["pc", "active"])
@pytest.mark.parametrize("fault", ["legacy", "decay", "age", "format", "missing_history"])
def test_should_refuse_incompatible_full_state_without_partial_restore(family, fault):
    model = make_model(family)
    features, targets = batch()
    model.train_epoch(features, targets, 0.01, 2, 0.02)
    before = model.snapshot_state()
    incoming: Any = model.snapshot_state()
    if fault == "legacy":
        incoming = CoupledSnapshot(1, incoming.parameters, incoming.precisions, incoming.wake_state)
    elif fault == "decay":
        object.__setattr__(incoming.unit_history, "ema_decay", 0.7)
    elif fault == "age":
        incoming.unit_history.update_age[0] += 1
    elif fault == "format":
        object.__setattr__(incoming, "format_version", 1)
    else:
        object.__setattr__(incoming, "unit_history", None)
    with pytest.raises(ValueError):
        model.restore_state(incoming)
    assert_complete(model.snapshot_state(), before)


def test_should_refuse_cpc_wake_and_history_count_mismatch_before_commit():
    model = make_model("active")
    before = model.snapshot_state()
    incoming = model.snapshot_state()
    object.__setattr__(incoming.unit_history, "updates", 1)
    incoming.unit_history.update_age[:] = 1
    with pytest.raises(ValueError):
        model.restore_state(incoming)
    assert_complete(model.snapshot_state(), before)


@pytest.mark.parametrize("counter, updates", [(False, 0), (True, 1), (0.0, 0), (1.0, 1)])
def test_should_refuse_equal_but_non_integer_cpc_wake_counter_atomically(
    monkeypatch, counter, updates
):
    model = make_model("active")
    features, targets = batch()
    if updates:
        model.train_epoch(features, targets, 0.01, 2, 0.02)
    assert model.read_unit_history().updates == updates
    model._wake_updates = counter
    before = pickle.dumps(vars(model), protocol=5)
    with pytest.raises(ValueError):
        model.read_unit_history()
    assert pickle.dumps(vars(model), protocol=5) == before

    preparations = []
    original = TrackedCoupledCircadianPredictiveCodingNetwork._prepare_parameters

    def prepare(*args, **kwargs):
        result = original(*args, **kwargs)
        preparations.append(True)
        return result

    monkeypatch.setattr(
        TrackedCoupledCircadianPredictiveCodingNetwork, "_prepare_parameters", prepare
    )
    with pytest.raises(ValueError):
        model.train_epoch(features, targets, 0.01, 2, 0.02)
    assert preparations == [True]
    assert pickle.dumps(vars(model), protocol=5) == before


def test_should_keep_default_instance_fields_and_version_one_snapshot_schema():
    pc = make_model("pc", tracked_model=False)
    cpc = make_model("active", tracked_model=False)
    common = {"_input_dim", "_hidden_dims", "_precisions", "_parameters"}
    assert set(vars(pc)) == common
    assert set(vars(cpc)) == common | {
        "_config",
        "_chemical_state",
        "_plasticity_state",
        "_wake_updates",
    }
    assert [field.name for field in fields(CoupledSnapshot)] == [
        "format_version",
        "parameters",
        "precisions",
        "wake_state",
    ]
    for model in (pc, cpc):
        snapshot = model.snapshot_state()
        assert type(snapshot) is CoupledSnapshot
        assert snapshot.format_version == 1
    assert [field.name for field in fields(TrackedCoupledSnapshot)] == [
        "format_version",
        "parameters",
        "precisions",
        "wake_state",
        "unit_history",
    ]


@pytest.mark.parametrize("family", ["pc", "active"])
def test_should_refuse_tracked_format_in_legacy_codec_and_native_restore(family):
    model = make_model(family)
    snapshot = model.snapshot_state()
    with pytest.raises(ValueError):
        CoupledCheckpointCodec().encode(
            snapshot,
            binding=CodecBinding("a" * 64, "b" * 64),
            limits=CodecLimits(32768, 8192, 8, 64),
        )
    ordinary = make_model(family, tracked_model=False)
    before = ordinary.snapshot_state()
    with pytest.raises(ValueError):
        ordinary.restore_state(snapshot)
    assert_native(ordinary.snapshot_state(), before)
