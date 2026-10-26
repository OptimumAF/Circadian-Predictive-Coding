"""Bounded replacement arithmetic, complete resets, and owned continuation.

Eligibility is tested with explicit history data. Small native cases then check
that renewing a last-hidden unit cannot silently alter protected model state.
"""

from dataclasses import FrozenInstanceError, fields, replace
from math import sqrt
import pickle
from typing import Any

import numpy as np
import pytest

import src.core.renewable_coupled_predictive_coding as renewable
from src.adapters.coupled_checkpoint_codec import CoupledCheckpointCodec
from src.core.checkpoint_codec import CodecBinding, CodecLimits
from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.coupled_energy import evaluate_coupled, feedforward_latents
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork, CoupledSnapshot
from src.core.coupled_unit_renewal import (
    RenewableUnitHistory,
    RenewalPolicy,
    RenewalResult,
    advance_renewal_history,
    copy_renewal_history,
    initial_renewal_history,
    select_renewal_indices,
)
from src.core.renewable_coupled_predictive_coding import (
    RenewableCoupledCircadianPredictiveCodingNetwork,
    RenewableCoupledPredictiveCodingNetwork,
    RenewableCoupledSnapshot,
)
from src.core.tracked_coupled_predictive_coding import (
    TrackedCoupledPredictiveCodingNetwork,
    TrackedCoupledSnapshot,
)


MAX_UPDATES = 2**53 - 1
MAX_ID = 2**63 - 1


def policy(**changes):
    return RenewalPolicy(
        **{"min_age_updates": 2, "max_importance": 1e308, "max_per_event": 2, "seed": 29, **changes}
    )


def model(kind, *, selected_policy=None, seed=17, ordinary=False):
    options: dict[str, Any] = {}
    if not ordinary:
        options["renewal_policy"] = policy() if selected_policy is None else selected_policy
        options["history_decay"] = 0.5
    if kind == "pc":
        constructor: Any = (
            CoupledPredictiveCodingNetwork if ordinary else RenewableCoupledPredictiveCodingNetwork
        )
    else:
        constructor = (
            CoupledCircadianPredictiveCodingNetwork
            if ordinary
            else RenewableCoupledCircadianPredictiveCodingNetwork
        )
        options["config"] = (
            CoupledCircadianConfig.matched_pc_control()
            if kind == "neutral"
            else CoupledCircadianConfig(chemical_buildup_rate=0.5, plasticity_sensitivity=2.0)
        )
    return constructor(2, (4, 3), seed=seed, precisions=(0.7, 1.3), **options)


def batch():
    return np.array([[0.2, -0.4], [-0.5, 0.3]]), np.array([[1.0], [0.0]])


def warm(native, count=2):
    x, y = batch()
    for _ in range(count):
        native.train_epoch(x, y, 0.02, 2, 0.04)


def history(snapshot):
    assert type(snapshot) is RenewableCoupledSnapshot
    assert type(snapshot.unit_history) is RenewableUnitHistory
    return snapshot.unit_history


def arrays(parameters):
    return (
        *parameters.hidden_weights,
        *parameters.hidden_biases,
        parameters.output_weight,
        parameters.output_bias,
    )


def assert_bits(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


def outgoing_gradient(snapshot, features, targets):
    latents = feedforward_latents(snapshot.parameters, features)
    for _ in range(2):
        evaluation = evaluate_coupled(
            snapshot.parameters, features, targets, latents, snapshot.precisions
        )
        latents = tuple(
            value - 0.04 * gradient for value, gradient in zip(latents, evaluation.latent_gradients)
        )
    return evaluate_coupled(
        snapshot.parameters, features, targets, latents, snapshot.precisions
    ).parameter_gradients.output_weight[:, 0]


def assert_history(actual, expected):
    assert actual.format_version == expected.format_version == 2
    assert actual.ema_decay == expected.ema_decay
    assert actual.updates == expected.updates and actual.next_unit_id == expected.next_unit_id
    for name in ("unit_ids", "importance_ema", "update_age"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


def assert_native(actual, expected):
    assert actual.precisions == expected.precisions
    for left, right in zip(arrays(actual.parameters), arrays(expected.parameters), strict=True):
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
    assert type(actual) is type(expected) is RenewableCoupledSnapshot
    assert actual.format_version == expected.format_version == 3
    assert actual.renewal_policy == expected.renewal_policy
    assert_native(actual, expected)
    assert_history(history(actual), history(expected))


@pytest.mark.parametrize("width", [1, 3, 4096])
def test_should_start_owned_fixed_width_history_without_retired_record_storage(width):
    value = initial_renewal_history(width)
    assert value.format_version == 2 and value.ema_decay == 0.95
    assert value.updates == 0 and value.next_unit_id == width
    np.testing.assert_array_equal(value.unit_ids, np.arange(width, dtype=np.int64))
    np.testing.assert_array_equal(value.importance_ema, np.zeros(width))
    np.testing.assert_array_equal(value.update_age, np.zeros(width))
    assert (
        sum(array.nbytes for array in (value.unit_ids, value.importance_ema, value.update_age))
        == 24 * width
    )
    assert set(vars(value)) == {
        "format_version",
        "ema_decay",
        "unit_ids",
        "importance_ema",
        "update_age",
        "updates",
        "next_unit_id",
    }
    with pytest.raises(FrozenInstanceError):
        setattr(value, "next_unit_id", width + 1)


def test_should_own_and_freeze_exact_policy_boundary_values():
    value = RenewalPolicy(MAX_UPDATES, 0.0, 0, 2**32 - 1)
    assert value.min_age_updates == MAX_UPDATES and value.max_per_event == 0
    assert value.seed == 2**32 - 1
    with pytest.raises(FrozenInstanceError):
        setattr(value, "seed", 0)
    object.__setattr__(value, "max_per_event", True)
    with pytest.raises(ValueError):
        select_renewal_indices(initial_renewal_history(3), value, 3)


@pytest.mark.parametrize(
    "name, value",
    [
        ("min_age_updates", 0),
        ("min_age_updates", True),
        ("min_age_updates", MAX_UPDATES + 1),
        ("max_importance", -0.1),
        ("max_importance", float("nan")),
        ("max_importance", True),
        ("max_per_event", -1),
        ("max_per_event", True),
        ("max_per_event", 4097),
        ("seed", -1),
        ("seed", True),
        ("seed", 2**32),
    ],
)
def test_should_refuse_invalid_policy_fields(name, value):
    with pytest.raises(ValueError):
        policy(**{name: value})


def test_should_advance_ema_and_individual_ages_without_reassigning_existing_ids():
    value = RenewableUnitHistory(
        2,
        0.5,
        np.array([9, 2, 7], dtype=np.int64),
        np.array([0.0, 4.0, 2.0]),
        np.array([0.0, 3.0, 1.0]),
        3,
        10,
    )
    saved = copy_renewal_history(value, 3, 0.5)
    gradient = np.array([[2.0], [-6.0], [0.0]])
    after = advance_renewal_history(value, gradient)
    np.testing.assert_array_equal(after.unit_ids, [9, 2, 7])
    np.testing.assert_array_equal(after.importance_ema, [1.0, 5.0, 1.0])
    np.testing.assert_array_equal(after.update_age, [1.0, 4.0, 2.0])
    assert after.updates == 4 and after.next_unit_id == 10
    assert_history(value, saved)
    for name in ("unit_ids", "importance_ema", "update_age"):
        assert not np.shares_memory(getattr(value, name), getattr(after, name))
    np.testing.assert_array_equal(gradient, [[2.0], [-6.0], [0.0]])
    copied = copy_renewal_history(after, 3, 0.5)
    copied.importance_ema.fill(99)
    copied.unit_ids.fill(99)
    copied.update_age.fill(99)
    np.testing.assert_array_equal(after.unit_ids, [9, 2, 7])
    np.testing.assert_array_equal(after.update_age, [1.0, 4.0, 2.0])


@pytest.mark.parametrize(
    "fault",
    [
        "format",
        "ids_dtype",
        "duplicate",
        "negative_id",
        "next_id",
        "importance",
        "fractional_age",
        "old_age",
        "nan_age",
        "updates",
        "updates_bool",
        "next_bool",
        "id_overflow",
        "decay",
    ],
)
def test_should_revalidate_all_history_fields_before_selection_or_copy(fault):
    value = initial_renewal_history(3, 0.5)
    changes: dict[str, Any] = {
        "format": ("format_version", 1),
        "ids_dtype": ("unit_ids", np.arange(3, dtype=np.int32)),
        "duplicate": ("unit_ids", np.array([0, 0, 2], dtype=np.int64)),
        "negative_id": ("unit_ids", np.array([-1, 1, 2], dtype=np.int64)),
        "next_id": ("next_unit_id", 2),
        "importance": ("importance_ema", np.ones(3)),
        "fractional_age": ("update_age", np.array([0.5, 0.0, 0.0])),
        "old_age": ("update_age", np.array([1.0, 0.0, 0.0])),
        "nan_age": ("update_age", np.array([float("nan"), 0.0, 0.0])),
        "updates": ("updates", MAX_UPDATES + 1),
        "updates_bool": ("updates", True),
        "next_bool": ("next_unit_id", True),
        "id_overflow": ("next_unit_id", MAX_ID + 1),
        "decay": ("ema_decay", 1.0),
    }
    name, changed = changes[fault]
    object.__setattr__(value, name, changed)
    with pytest.raises(ValueError):
        copy_renewal_history(value, 3, 0.5)
    with pytest.raises(ValueError):
        select_renewal_indices(value, policy(), 3)


@pytest.mark.parametrize(
    "gradient",
    [
        np.ones(3),
        np.ones((1, 3)),
        np.ones((3, 1), dtype=np.int64),
        np.full((3, 1), float("nan")),
        np.full((3, 1), float("inf")),
    ],
)
def test_should_refuse_invalid_gradient_without_mutation(gradient):
    value = initial_renewal_history(3)
    before = pickle.dumps(value)
    with pytest.raises(ValueError):
        advance_renewal_history(value, gradient)
    assert pickle.dumps(value) == before


def test_should_refuse_age_overflow_without_changing_ids_or_history():
    value = RenewableUnitHistory(
        2,
        0.5,
        np.arange(2, dtype=np.int64),
        np.zeros(2),
        np.full(2, float(MAX_UPDATES)),
        MAX_UPDATES,
        2,
    )
    before = pickle.dumps(value)
    with pytest.raises(ValueError):
        advance_renewal_history(value, np.ones((2, 1)))
    assert pickle.dumps(value) == before


@pytest.mark.parametrize("maximum, expected", [(0, ()), (1, (1,)), (8, (1, 4)), (2**80, (1, 4))])
def test_should_select_only_mature_low_gradient_units_in_stable_id_order(maximum, expected):
    value = RenewableUnitHistory(
        2,
        0.5,
        np.array([20, 2, 8, 1, 9], dtype=np.int64),
        np.array([0.2, 0.1, 0.1, 0.0, 0.1]),
        np.array([5.0, 5.0, 4.0, 1.0, 5.0]),
        5,
        21,
    )
    selected_policy = policy(min_age_updates=5, max_importance=0.1, max_per_event=3)
    before = pickle.dumps((value, selected_policy))
    assert select_renewal_indices(value, selected_policy, maximum) == expected
    assert select_renewal_indices(value, replace(selected_policy, max_per_event=0), 10) == ()
    assert pickle.dumps((value, selected_policy)) == before


@pytest.mark.parametrize("maximum", [True, -1, 1.0, np.int64(1)])
def test_should_refuse_non_native_or_negative_replacement_allowance(maximum):
    with pytest.raises(ValueError):
        select_renewal_indices(initial_renewal_history(3), policy(), maximum)


@pytest.mark.parametrize("kind", ["pc", "neutral", "active"])
def test_should_reset_complete_selected_units_and_preserve_every_protected_native_bit(kind):
    native = model(kind)
    warm(native)
    parameters = native.snapshot_parameters()
    parameters.hidden_weights[0][0, 0] = -0.0
    parameters.output_bias[0, 0] = -0.0
    native.restore_parameters(parameters)
    before = native.snapshot_state()
    stored = history(before)
    expected_indices = tuple(
        sorted(
            range(3),
            key=lambda index: (stored.importance_ema[index], int(stored.unit_ids[index]), index),
        )[:2]
    )
    global_rng = pickle.dumps(np.random.get_state())
    result = native.renew_units(2)
    after = native.snapshot_state()
    assert type(result) is RenewalResult and result.indices == expected_indices
    assert result.old_unit_ids == tuple(int(stored.unit_ids[index]) for index in expected_indices)
    assert result.new_unit_ids == (3, 4)
    with pytest.raises(FrozenInstanceError):
        setattr(result, "indices", ())
    assert pickle.dumps(np.random.get_state()) == global_rng
    for index, new_id in zip(result.indices, result.new_unit_ids, strict=True):
        expected = np.random.default_rng(np.random.SeedSequence([29, new_id])).normal(
            0.0, 1 / sqrt(4), size=4
        )
        np.testing.assert_array_equal(after.parameters.hidden_weights[-1][:, index], expected)
        np.testing.assert_array_equal(after.parameters.hidden_biases[-1][:, index], [0.0])
        np.testing.assert_array_equal(after.parameters.output_weight[index], [0.0])
        assert history(after).unit_ids[index] == new_id
        assert history(after).importance_ema[index] == history(after).update_age[index] == 0
        if after.wake_state is not None:
            assert after.wake_state.chemical_state[index] == 0
            assert after.wake_state.plasticity_state[index] == 1
    retained = [index for index in range(3) if index not in result.indices]
    assert_bits(
        after.parameters.hidden_weights[-1][:, retained],
        before.parameters.hidden_weights[-1][:, retained],
    )
    assert_bits(
        after.parameters.hidden_biases[-1][:, retained],
        before.parameters.hidden_biases[-1][:, retained],
    )
    assert_bits(after.parameters.output_weight[retained], before.parameters.output_weight[retained])
    for name in ("unit_ids", "importance_ema", "update_age"):
        np.testing.assert_array_equal(
            getattr(history(after), name)[retained], getattr(stored, name)[retained]
        )
    for left, right in zip(
        after.parameters.hidden_weights[:-1], before.parameters.hidden_weights[:-1], strict=True
    ):
        assert_bits(left, right)
    for left, right in zip(
        after.parameters.hidden_biases[:-1], before.parameters.hidden_biases[:-1], strict=True
    ):
        assert_bits(left, right)
    assert_bits(after.parameters.output_bias, before.parameters.output_bias)
    assert history(after).updates == stored.updates == 2 and history(after).next_unit_id == 5
    if after.wake_state is not None:
        assert before.wake_state is not None
        assert np.all(before.wake_state.chemical_state > 0)
        assert after.wake_state.wake_updates == before.wake_state.wake_updates == 2
        np.testing.assert_array_equal(
            after.wake_state.chemical_state[retained], before.wake_state.chemical_state[retained]
        )
        np.testing.assert_array_equal(
            after.wake_state.plasticity_state[retained],
            before.wake_state.plasticity_state[retained],
        )


def test_should_not_change_any_state_when_allowance_is_zero_or_no_unit_is_mature():
    native = model("pc")
    before = pickle.dumps(vars(native))
    for maximum in (0, 2):
        result = native.renew_units(maximum)
        assert result.indices == result.old_unit_ids == result.new_unit_ids == ()
        assert pickle.dumps(vars(native)) == before


@pytest.mark.parametrize("kind", ["pc", "neutral", "active"])
def test_should_preserve_default_native_math_until_an_explicit_replacement(kind):
    ordinary, enabled = model(kind, ordinary=True), model(kind)
    x, y = batch()
    expected = np.zeros(3)
    for index in range(1, 3):
        gradient = outgoing_gradient(ordinary.snapshot_state(), x, y)
        assert ordinary.train_epoch(x, y, 0.02, 2, 0.04) == enabled.train_epoch(x, y, 0.02, 2, 0.04)
        assert_native(enabled.snapshot_state(), ordinary.snapshot_state())
        expected = 0.5 * expected + 0.5 * np.abs(gradient)
        np.testing.assert_allclose(enabled.read_unit_history().importance_ema, expected, rtol=2e-15)
        np.testing.assert_array_equal(
            enabled.read_unit_history().update_age, np.full(3, float(index))
        )
    np.testing.assert_array_equal(ordinary.predict_proba(x), enabled.predict_proba(x))


@pytest.mark.parametrize("kind", ["pc", "neutral", "active"])
def test_should_continue_owned_full_snapshots_with_identical_future_ids_and_replacements(kind):
    original = model(kind)
    warm(original)
    original.renew_units(2)
    saved = original.snapshot_state()
    restored = model(kind, seed=59)
    restored.restore_state(saved)
    assert_complete(restored.snapshot_state(), saved)
    for left, right in zip(
        arrays(saved.parameters), arrays(restored.snapshot_state().parameters), strict=True
    ):
        assert not np.shares_memory(left, right)
    for name in ("unit_ids", "importance_ema", "update_age"):
        assert not np.shares_memory(
            getattr(history(saved), name), getattr(restored.read_unit_history(), name)
        )
    x, y = batch()
    assert original.train_epoch(x, y, 0.02, 2, 0.04) == restored.train_epoch(x, y, 0.02, 2, 0.04)
    assert_complete(original.snapshot_state(), restored.snapshot_state())
    assert original.renew_units(1) == restored.renew_units(1)
    assert_complete(original.snapshot_state(), restored.snapshot_state())
    held = original.snapshot_state()
    saved.parameters.output_bias.fill(999)
    history(saved).importance_ema.fill(999)
    if saved.wake_state is not None:
        saved.wake_state.chemical_state.fill(999)
    assert_complete(original.snapshot_state(), held)
    assert_complete(restored.snapshot_state(), held)


@pytest.mark.parametrize("kind", ["pc", "neutral", "active"])
@pytest.mark.parametrize("fault", ["invalid_input", "history"])
def test_should_not_publish_native_state_when_training_or_history_preparation_fails(
    monkeypatch, kind, fault
):
    native = model(kind)
    before = pickle.dumps(vars(native))
    x, y = batch()
    problem = RuntimeError("history preparation failed")
    if fault == "invalid_input":
        y[0, 0] = float("nan")
    else:

        def broken(*args, **kwargs):
            raise problem

        monkeypatch.setattr(renewable, "advance_renewal_history", broken)
    with pytest.raises(ValueError if fault == "invalid_input" else RuntimeError) as caught:
        native.train_epoch(x, y, 0.02, 2, 0.04)
    if fault == "history":
        assert caught.value is problem
    assert pickle.dumps(vars(native)) == before


@pytest.mark.parametrize("kind", ["pc", "active"])
@pytest.mark.parametrize("fault", ["rng", "candidate"])
def test_should_keep_all_fields_unchanged_when_replacement_preparation_fails(
    monkeypatch, kind, fault
):
    native = model(kind)
    warm(native)
    before = pickle.dumps(vars(native))
    problem = RuntimeError("replacement RNG failed")
    if fault == "rng":

        def broken(*args, **kwargs):
            raise problem

        monkeypatch.setattr(renewable.np.random, "default_rng", broken)
    else:
        reset = renewable._reset_parameters

        def broken(*args, **kwargs):
            candidate, candidate_history, result = reset(*args, **kwargs)
            candidate.output_bias[0, 0] = float("nan")
            return candidate, candidate_history, result

        monkeypatch.setattr(renewable, "_reset_parameters", broken)
    with pytest.raises(RuntimeError if fault == "rng" else ValueError) as caught:
        native.renew_units(2)
    if fault == "rng":
        assert caught.value is problem
    assert pickle.dumps(vars(native)) == before


@pytest.mark.parametrize("kind", ["pc", "active"])
@pytest.mark.parametrize("fault", ["policy", "decay", "age", "legacy"])
def test_should_refuse_incompatible_full_snapshot_before_changing_any_native_field(kind, fault):
    native = model(kind)
    warm(native, 1)
    before = pickle.dumps(vars(native))
    incoming: Any = native.snapshot_state()
    if fault == "policy":
        object.__setattr__(incoming, "renewal_policy", policy(seed=30))
    elif fault == "decay":
        object.__setattr__(history(incoming), "ema_decay", 0.7)
    elif fault == "age":
        history(incoming).update_age[0] = 2
    else:
        incoming = CoupledSnapshot(1, incoming.parameters, incoming.precisions, incoming.wake_state)
    with pytest.raises(ValueError):
        native.restore_state(incoming)
    assert pickle.dumps(vars(native)) == before


@pytest.mark.parametrize("counter, updates", [(False, 0), (True, 1)])
def test_should_require_exact_cpc_wake_count_and_leave_all_state_unchanged(counter, updates):
    native = model("active")
    warm(native, updates)
    native._wake_updates = counter
    before = pickle.dumps(vars(native))
    with pytest.raises(ValueError):
        native.read_unit_history()
    x, y = batch()
    with pytest.raises(ValueError):
        native.train_epoch(x, y, 0.02, 2, 0.04)
    assert pickle.dumps(vars(native)) == before


@pytest.mark.parametrize("kind", ["pc", "active"])
def test_should_refuse_new_id_exhaustion_before_resetting_a_mature_unit(kind):
    native = model(kind)
    snapshot = native.snapshot_state()
    object.__setattr__(
        snapshot,
        "unit_history",
        RenewableUnitHistory(
            2, 0.5, np.arange(3, dtype=np.int64), np.zeros(3), np.full(3, 2.0), 2, MAX_ID
        ),
    )
    if snapshot.wake_state is not None:
        object.__setattr__(snapshot.wake_state, "wake_updates", 2)
    native.restore_state(snapshot)
    before = pickle.dumps(vars(native))
    with pytest.raises(ValueError):
        native.renew_units(1)
    assert pickle.dumps(vars(native)) == before


@pytest.mark.parametrize("kind", ["pc", "neutral", "active"])
def test_should_preserve_history_on_prediction_and_parameter_only_restore_and_own_policy(kind):
    caller_policy = policy(max_per_event=1)
    native = model(kind, selected_policy=caller_policy)
    warm(native, 1)
    before = native.snapshot_state()
    reported = native.renewal_policy
    assert reported == caller_policy and reported is not caller_policy
    object.__setattr__(caller_policy, "seed", 999)
    object.__setattr__(reported, "seed", 888)
    assert native.renewal_policy.seed == 29
    parameters = native.snapshot_parameters()
    parameters.output_bias[:] += 0.25
    native.restore_parameters(parameters)
    native.predict_proba(batch()[0])
    assert_history(native.read_unit_history(), history(before))
    if before.wake_state is not None:
        assert native.wake_updates == before.wake_state.wake_updates
        np.testing.assert_array_equal(native.get_chemical_state(), before.wake_state.chemical_state)
    owned = native.read_unit_history()
    owned.unit_ids.fill(999)
    owned.importance_ema.fill(999)
    owned.update_age.fill(999)
    assert_history(native.read_unit_history(), history(before))
    parameters.output_bias.fill(999)
    assert not np.any(native.snapshot_parameters().output_bias == 999)


def test_should_keep_default_and_existing_tracked_snapshot_schemas_unchanged():
    ordinary = CoupledPredictiveCodingNetwork(2, (4, 3), 17)
    tracked = TrackedCoupledPredictiveCodingNetwork(2, (4, 3), 17)
    assert set(vars(ordinary)) == {"_input_dim", "_hidden_dims", "_precisions", "_parameters"}
    assert set(vars(tracked)) == set(vars(ordinary)) | {"_unit_history_decay", "_unit_history"}
    assert type(ordinary.snapshot_state()) is CoupledSnapshot
    assert ordinary.snapshot_state().format_version == 1
    assert type(tracked.snapshot_state()) is TrackedCoupledSnapshot
    assert tracked.snapshot_state().format_version == 2
    assert [field.name for field in fields(CoupledSnapshot)] == [
        "format_version",
        "parameters",
        "precisions",
        "wake_state",
    ]
    assert [field.name for field in fields(TrackedCoupledSnapshot)] == [
        "format_version",
        "parameters",
        "precisions",
        "wake_state",
        "unit_history",
    ]


@pytest.mark.parametrize("kind", ["pc", "active"])
def test_should_refuse_legacy_codec_and_native_consumers_instead_of_dropping_renewal_state(kind):
    snapshot = model(kind).snapshot_state()
    with pytest.raises(ValueError):
        CoupledCheckpointCodec().encode(
            snapshot,
            binding=CodecBinding("a" * 64, "b" * 64),
            limits=CodecLimits(32768, 8192, 8, 64),
        )
    for consumer in (
        model(kind, ordinary=True),
        TrackedCoupledPredictiveCodingNetwork(2, (4, 3), 17),
    ):
        before = pickle.dumps(vars(consumer))
        with pytest.raises(ValueError):
            consumer.restore_state(snapshot)
        assert pickle.dumps(vars(consumer)) == before
