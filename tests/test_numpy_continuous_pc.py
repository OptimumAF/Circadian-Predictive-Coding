"""Owned continuous PC updates on tiny released synthetic arrays.

The public inference oracle supplies accepted partials; momentum arithmetic is
independent. No BP training, scientific data or simulator. Tiny parameter updates
are correctness fixtures, not evidence of adaptation benefit.

Static per full module: 50 cases; 77 PC/BP constructor entries (including one
refusal), 320 seeded normal values, 26 PC training attempts. Conservative bounds
including the two public-inference oracles: 100 energy entries and 100 latent
proposals. Forks use no RNG; simulator/native work and BP training remain zero.
"""

from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import numpy as np
import pytest

from src.adapters import numpy_continuous_bc as bc
from src.adapters import numpy_continuous_pc as adapter
from src.app.actor_shadow import ActorShadowRuntime
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionStopped
from src.core import continuous_pc as energy
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.continuous_pc_inference import infer_continuous_pc
from src.core.coupled_energy import CoupledParameters
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.learner_ports import TrainingUpdateFailed
from src.core.observation_standardization import ObservationStandardization


def spec(*, constant=False):
    return ContinuousActionSpec(2, ("m/s", "N"), (-2.0, 4.0), (6.0, 4.0 if constant else 8.0))


def arrays():
    return np.array([[0.2, -0.3], [0.7, 0.4]]), np.array([[1.0, 5.0], [4.0, 7.0]])


def policy(**changes):
    base = bc.ContinuousBCPolicy(spec(), hidden_dim=2, seed=17, learning_rate=0.03, momentum=0.8)
    options: dict[str, Any] = dict(
        base=base,
        precision=0.8,
        inference_steps=2,
        inference_learning_rate=0.05,
        inference_policy=CoupledInferencePolicy(
            min_steps=1, gradient_tolerance=None, relative_energy_tolerance=None
        ),
    )
    options.update(changes)
    return adapter.ContinuousPCPolicy(**options)


def model(**changes):
    result = adapter.NumpyContinuousPCLearner(policy(**changes))
    before = result.snapshot_state()
    parameters = (
        np.array([[0.1, -0.2], [0.3, 0.15]]),
        np.array([0.02, -0.03]),
        np.array([[0.2, -0.1], [-0.3, 0.25]]),
        np.array([0.05, -0.04]),
    )
    result.restore_state(replace(before, parameters=parameters))
    return result


def assert_same_state(left, right):
    assert left.policy == right.policy
    assert left.completed_updates == right.completed_updates
    assert left.training_examples == right.training_examples
    for name in ("parameters", "velocity"):
        for first, second in zip(getattr(left, name), getattr(right, name), strict=True):
            np.testing.assert_array_equal(first, second)


def coupled(parameters):
    first, first_bias, second, second_bias = parameters
    return CoupledParameters(
        (first.copy(),), (first_bias[None, :].copy(),), second.copy(), second_bias[None, :].copy()
    )


def normalized(features, transform):
    if transform is None:
        return features.copy()
    return (features - np.array(transform.means)) / np.array(transform.scales)


def expected_work(**changes):
    values = dict(
        normalization_attempts=1,
        normalization_returns=1,
        initialization_attempts=1,
        initialization_returns=1,
        attempted_steps=2,
        accepted_steps=2,
        energy_evaluations=3,
        energy_returns=3,
        gradient_norm_evaluations=3,
        energy_comparisons=2,
        inference_returns=1,
        parameter_proposals_attempted=1,
        parameter_proposals_completed=1,
        candidate_predictions_attempted=1,
        candidate_predictions_completed=1,
        published_updates=1,
    )
    values.update(changes)
    return values


def episode(*, role="train", declared=None):
    geometry = spec() if declared is None else declared
    return ContinuousEpisode(
        "synthetic-pc",
        "synthetic-context",
        role,
        geometry,
        (ContinuousTransition(0, (0.2, -0.3), (1.0, 5.0), 1.0, (0.7, 0.4), True, False),),
    )


@pytest.mark.parametrize("robot_axes", [False, True])
def test_should_start_with_exact_bp_parameters_prediction_and_frozen_normalization(robot_axes):
    transform = ObservationStandardization((0.1, -0.2), (2.0, 0.5), 2)
    declared = (
        spec()
        if not robot_axes
        else ContinuousActionSpec(
            2,
            ("normalized_dx", "normalized_dy", "normalized_dz", "normalized_gripper"),
            (-1.0,) * 4,
            (1.0,) * 4,
        )
    )
    base = replace(policy().base, spec=declared, observation_standardization=transform)
    before_global = cast(tuple[Any, ...], np.random.get_state())
    owner = adapter.NumpyContinuousPCLearner(adapter.ContinuousPCPolicy(base))
    reference = bc.NumpyContinuousBCLearner(base)
    after_global = cast(tuple[Any, ...], np.random.get_state())
    assert before_global[0] == after_global[0] and before_global[2:] == after_global[2:]
    np.testing.assert_array_equal(before_global[1], after_global[1])
    actual, expected = owner.snapshot_state(), reference.snapshot_state()
    for name in ("parameters", "velocity"):
        for first, second in zip(getattr(actual, name), getattr(expected, name), strict=True):
            np.testing.assert_array_equal(first, second)
    features, _ = arrays()
    np.testing.assert_array_equal(owner.predict(features), reference.predict(features))
    assert actual.completed_updates == actual.training_examples == 0
    assert actual.policy.base == base
    assert actual.policy.base is not base
    assert actual.policy.base.observation_standardization is not transform
    assert actual.policy.precision == 1.0 and actual.policy.inference_steps == 4
    assert actual.policy.inference_learning_rate == 0.05


def test_should_apply_inferred_partial_momentum_twice_without_changing_inputs():
    transform = ObservationStandardization((0.1, -0.2), (2.0, 0.5), 2)
    owner = model(base=replace(policy().base, observation_standardization=transform))
    features, targets = arrays()
    originals = features.copy(), targets.copy()
    for index in range(2):
        before = owner.snapshot_state()
        oracle = infer_continuous_pc(
            coupled(before.parameters),
            normalized(features, transform),
            targets,
            (before.policy.precision,),
            before.policy.base.spec,
            max_steps=before.policy.inference_steps,
            learning_rate=before.policy.inference_learning_rate,
            policy=before.policy.inference_policy,
        )
        partial = oracle.evaluation.parameter_gradients
        gradients = (
            partial.hidden_weights[0],
            partial.hidden_biases[0][0],
            partial.output_weight,
            partial.output_bias[0],
        )
        expected_velocity = tuple(
            0.8 * old + gradient for old, gradient in zip(before.velocity, gradients, strict=True)
        )
        expected_parameters = tuple(
            old - 0.03 * velocity
            for old, velocity in zip(before.parameters, expected_velocity, strict=True)
        )
        result = owner.train_batch_with_work(features, targets)
        after = owner.snapshot_state()
        assert result.diagnostic.definition == energy.CONTINUOUS_PC_ENERGY_ID
        assert result.diagnostic.value == oracle.evaluation.energy
        assert result.task_loss == oracle.evaluation.task_loss
        assert result.diagnostic.value > result.task_loss
        assert result.inference == oracle.diagnostics
        assert result.work == expected_work()
        assert all(type(value) is int for value in result.work.values())
        assert after.completed_updates == index + 1 and after.training_examples == 2 * (index + 1)
        for actual, expected in zip(after.parameters, expected_parameters, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
        for actual, expected in zip(after.velocity, expected_velocity, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
        result.work["published_updates"] = 999
        assert owner.snapshot_state().completed_updates == index + 1
        with pytest.raises(FrozenInstanceError):
            result.task_loss = 123
    np.testing.assert_array_equal(features, originals[0])
    np.testing.assert_array_equal(targets, originals[1])


def test_should_continue_exactly_through_owned_snapshot_restore_and_rng_free_forks(monkeypatch):
    owner = model()
    features, targets = arrays()
    owner.train_batch(features, targets)
    saved = owner.snapshot_state()
    restored = model()
    restored.restore_state(saved)
    monkeypatch.setattr(bc.np.random, "default_rng", lambda *_: pytest.fail("fork used RNG"))
    child = owner.fork()
    assert child is not owner
    for candidate in (owner, restored, child):
        candidate.train_batch(features, targets)
    assert_same_state(owner.snapshot_state(), restored.snapshot_state())
    assert_same_state(owner.snapshot_state(), child.snapshot_state())
    stable = owner.snapshot_state()
    for value in (*saved.parameters, *saved.velocity):
        value[:] = 100
    detached = owner.policy
    object.__setattr__(detached.base, "learning_rate", 0.9)
    assert_same_state(stable, owner.snapshot_state())
    assert_same_state(stable, restored.snapshot_state())
    child.train_batch(features, targets)
    assert owner.snapshot_state().completed_updates == 2


@pytest.mark.parametrize(
    "bad", ["dtype", "shape", "nan", "velocity", "negative", "bool", "rows", "policy"]
)
def test_should_refuse_invalid_restoration_atomically(bad):
    owner = model()
    before = owner.snapshot_state()
    damaged = owner.snapshot_state()
    if bad in ("dtype", "shape", "nan"):
        values = list(damaged.parameters)
        if bad == "dtype":
            values[0] = values[0].astype(np.float32)
        elif bad == "shape":
            values[0] = values[0][:1]
        else:
            values[0][0, 0] = np.nan
        damaged = replace(damaged, parameters=tuple(values))
    elif bad == "velocity":
        damaged.velocity[0][0, 0] = np.inf
    elif bad == "policy":
        damaged = replace(damaged, policy=replace(damaged.policy, precision=0.9))
    else:
        overrides: dict[str, dict[str, Any]] = {
            "negative": dict(completed_updates=-1),
            "bool": dict(completed_updates=True),
            "rows": dict(training_examples=1),
        }
        damaged = replace(damaged, **overrides[bad])
    with pytest.raises(ValueError):
        owner.restore_state(damaged)
    assert_same_state(before, owner.snapshot_state())


@pytest.mark.parametrize(
    "changes",
    [
        {"precision": True},
        {"precision": 0},
        {"precision": 1e-7},
        {"precision": 1e7},
        {"precision": float("nan")},
        {"inference_steps": True},
        {"inference_steps": 0},
        {"inference_steps": 10001},
        {"inference_learning_rate": True},
        {"inference_learning_rate": 0},
        {"inference_learning_rate": float("inf")},
        {"inference_policy": None},
        {"inference_policy": CoupledInferencePolicy(min_steps=3)},
    ],
)
def test_should_refuse_invalid_pc_settings_before_delegated_rng(monkeypatch, changes):
    monkeypatch.setattr(
        bc, "NumpyContinuousBCLearner", lambda *_: pytest.fail("invalid PC policy reached RNG")
    )
    with pytest.raises(ValueError):
        adapter.NumpyContinuousPCLearner(policy(**changes))


def test_should_refuse_mixed_binary_mode_and_corrupted_nested_policy_before_rng(monkeypatch):
    declared = ContinuousActionSpec(
        2,
        ("normalized_dx", "normalized_dy", "normalized_dz", "normalized_gripper"),
        (-1.0,) * 4,
        (1.0,) * 4,
    )
    mixed = bc.ContinuousBCPolicy(declared, hidden_dim=2, binary_gripper_axis=3)
    monkeypatch.setattr(
        bc, "NumpyContinuousBCLearner", lambda *_: pytest.fail("mixed mode reached RNG")
    )
    with pytest.raises(ValueError):
        adapter.NumpyContinuousPCLearner(policy(base=mixed))
    value = policy()
    object.__setattr__(value.base, "hidden_dim", True)
    with pytest.raises(ValueError):
        adapter.NumpyContinuousPCLearner(value)


@pytest.mark.parametrize(
    "bad",
    [
        "features_dtype",
        "features_shape",
        "targets_dtype",
        "targets_bounds",
        "targets_nan",
        "counter",
    ],
)
def test_should_refuse_admission_before_initializer_and_preserve_state(monkeypatch, bad):
    owner = model()
    features, targets = arrays()
    if bad == "features_dtype":
        features = features.astype(np.float32)
    elif bad == "features_shape":
        features = features[:, :1]
    elif bad == "targets_dtype":
        targets = targets.astype(np.float32)
    elif bad == "targets_bounds":
        targets[0, 0] = 6.1
    elif bad == "targets_nan":
        targets[0, 0] = np.nan
    else:
        owner.restore_state(
            replace(
                owner.snapshot_state(), completed_updates=2**53 - 1, training_examples=2**53 - 1
            )
        )
    before = owner.snapshot_state()
    monkeypatch.setattr(
        adapter.energy,
        "feedforward_continuous_latents",
        lambda *_: pytest.fail("invalid admission initialized latents"),
    )
    with pytest.raises(ValueError):
        owner.train_batch_with_work(features, targets)
    assert_same_state(before, owner.snapshot_state())


@pytest.mark.parametrize(
    "role", ["probe_train", "probe_test", "inner_guard", "outer_selection", "final_test"]
)
def test_should_refuse_other_episode_roles_before_any_numerical_work(monkeypatch, role):
    owner = model()
    before = owner.snapshot_state()
    monkeypatch.setattr(
        adapter.energy,
        "feedforward_continuous_latents",
        lambda *_: pytest.fail("heldout labels reached inference"),
    )
    value = episode()
    object.__setattr__(value, "role", role)
    with pytest.raises(ValueError):
        owner.train_episode(value)
    assert_same_state(before, owner.snapshot_state())


def test_should_train_matching_probe_only_and_require_full_spec():
    owner = model(base=replace(policy().base, training_role="probe_train"))
    diagnostic = owner.train_episode(episode(role="probe_train"))
    assert diagnostic.definition == energy.CONTINUOUS_PC_ENERGY_ID
    assert owner.snapshot_state().training_examples == 1
    before = owner.snapshot_state()
    for value in (
        episode(),
        episode(role="probe_train", declared=replace(spec(), action_units=("m", "N"))),
    ):
        with pytest.raises(ValueError):
            owner.train_episode(value)
    assert_same_state(before, owner.snapshot_state())


@pytest.mark.parametrize(
    "where", ["initializer", "initial_energy", "late_energy", "parameter", "candidate"]
)
def test_should_preserve_full_state_and_exact_consumed_work_when_update_fails(monkeypatch, where):
    owner = model()
    before = owner.snapshot_state()
    features, targets = arrays()
    if where == "initializer":
        monkeypatch.setattr(
            adapter.energy,
            "feedforward_continuous_latents",
            lambda *_: (_ for _ in ()).throw(FloatingPointError("initializer failure")),
        )
        expected = expected_work(
            initialization_returns=0,
            attempted_steps=0,
            accepted_steps=0,
            energy_evaluations=0,
            energy_returns=0,
            gradient_norm_evaluations=0,
            energy_comparisons=0,
            inference_returns=0,
            parameter_proposals_attempted=0,
            parameter_proposals_completed=0,
            candidate_predictions_attempted=0,
            candidate_predictions_completed=0,
            published_updates=0,
        )
    elif where in ("initial_energy", "late_energy"):
        original = adapter.energy.evaluate_continuous_pc
        calls = 0

        def failing(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == (1 if where == "initial_energy" else 3):
                raise ValueError("scripted energy failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(adapter.energy, "evaluate_continuous_pc", failing)
        expected = expected_work(
            attempted_steps=0 if where == "initial_energy" else 2,
            accepted_steps=0 if where == "initial_energy" else 1,
            energy_evaluations=1 if where == "initial_energy" else 3,
            energy_returns=0 if where == "initial_energy" else 2,
            gradient_norm_evaluations=0 if where == "initial_energy" else 2,
            energy_comparisons=0 if where == "initial_energy" else 1,
            inference_returns=0,
            parameter_proposals_attempted=0,
            parameter_proposals_completed=0,
            candidate_predictions_attempted=0,
            candidate_predictions_completed=0,
            published_updates=0,
        )
    elif where == "parameter":
        original_snapshot = adapter._snapshot

        def invalid_candidate(value, owner_policy):
            if value.completed_updates > before.completed_updates:
                raise ValueError("candidate snapshot failure")
            return original_snapshot(value, owner_policy)

        monkeypatch.setattr(adapter, "_snapshot", invalid_candidate)
        expected = expected_work(
            parameter_proposals_completed=0,
            candidate_predictions_attempted=0,
            candidate_predictions_completed=0,
            published_updates=0,
        )
    else:
        monkeypatch.setattr(
            adapter.energy,
            "predict_continuous_pc",
            lambda *_: (_ for _ in ()).throw(FloatingPointError("candidate forward failure")),
        )
        expected = expected_work(candidate_predictions_completed=0, published_updates=0)
    with pytest.raises(TrainingUpdateFailed) as caught:
        owner.train_batch_with_work(features, targets)
    assert caught.value.definition == energy.CONTINUOUS_PC_ENERGY_ID
    assert caught.value.work == expected
    exposed = caught.value.work
    exposed["published_updates"] = 999
    assert caught.value.work == expected
    assert_same_state(before, owner.snapshot_state())


def test_should_publish_the_accepted_initial_fallback_and_count_rejected_energy(monkeypatch):
    owner = model()
    features, targets = arrays()
    before = owner.snapshot_state()
    original = adapter.energy.evaluate_continuous_pc
    accepted: list[energy.ContinuousPCEvaluation] = []

    def fail_candidate(*args, **kwargs):
        if accepted:
            raise FloatingPointError("rejected candidate energy")
        value = original(*args, **kwargs)
        accepted.append(value)
        return value

    monkeypatch.setattr(adapter.energy, "evaluate_continuous_pc", fail_candidate)
    result = owner.train_batch_with_work(features, targets)
    assert result.inference.used_fallback is True and result.inference.reason == "numerical_failure"
    assert result.diagnostic.value == accepted[0].energy
    assert result.work == expected_work(
        attempted_steps=1,
        accepted_steps=0,
        energy_evaluations=2,
        energy_returns=1,
        gradient_norm_evaluations=1,
        energy_comparisons=0,
    )
    after = owner.snapshot_state()
    assert after.completed_updates == before.completed_updates + 1
    np.testing.assert_array_equal(after.parameters[0], before.parameters[0])
    np.testing.assert_array_equal(after.parameters[1], before.parameters[1])
    assert any(np.any(value) for value in after.velocity[2:])


def test_should_retain_paid_work_when_the_real_momentum_proposal_overflows():
    owner = model(base=replace(policy().base, learning_rate=1.0, momentum=0.9))
    state = owner.snapshot_state()
    for value in (*state.parameters, *state.velocity):
        value[:] = 0
    state.parameters[3][:] = -np.finfo(np.float64).max
    state.velocity[3][:] = np.finfo(np.float64).max
    owner.restore_state(replace(state, completed_updates=1, training_examples=1))
    before = owner.snapshot_state()
    with pytest.raises(TrainingUpdateFailed) as caught:
        owner.train_batch(np.zeros((1, 2)), np.array([spec().lower_bounds]))
    assert caught.value.work == expected_work(
        parameter_proposals_completed=0,
        candidate_predictions_attempted=0,
        candidate_predictions_completed=0,
        published_updates=0,
    )
    assert_same_state(before, owner.snapshot_state())


def test_should_charge_real_normalization_overflow_before_initialization():
    huge = np.finfo(np.float64).max.item()
    transform = ObservationStandardization((-huge, 0.0), (0.5, 1.0), 1)
    owner = model(base=replace(policy().base, observation_standardization=transform))
    before = owner.snapshot_state()
    with pytest.raises(TrainingUpdateFailed) as caught:
        owner.train_batch(np.array([[huge, 0.0]]), np.array([[1.0, 5.0]]))
    assert caught.value.work == {
        name: int(name == "normalization_attempts") for name in expected_work()
    }
    assert_same_state(before, owner.snapshot_state())


def test_should_count_constant_axis_training_without_moving_any_parameter():
    declared = ContinuousActionSpec(2, ("m", "N"), (3.0, 4.0), (3.0, 4.0))
    owner = model(base=replace(policy().base, spec=declared))
    before = owner.snapshot_state()
    features, _ = arrays()
    result = owner.train_batch_with_work(features, np.array([[3.0, 4.0], [3.0, 4.0]]))
    after = owner.snapshot_state()
    assert result.task_loss == result.diagnostic.value == 0.0
    assert result.work == expected_work()
    assert after.completed_updates == 1 and after.training_examples == 2
    for left, right in zip(before.parameters, after.parameters, strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(owner.predict(features), [[3.0, 4.0], [3.0, 4.0]])


@pytest.mark.parametrize("future_label", [False, True])
def test_should_release_one_delayed_packet_to_private_candidate_and_keep_actor_stable(
    monkeypatch, future_label
):
    source = model()
    original = source.snapshot_state()
    features, targets = arrays()
    original_prediction = source.predict(features)
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
    runtime = ActorShadowRuntime(
        source,
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=clock,
        budget=budget,
        max_experiences=2,
        max_consolidations=0,
    )
    experience = Experience(
        "packet", "episode", 0, "actor-0", features, "train", ExperiencePermissions(training=True)
    )
    label = LabelArrival("label", "packet", "episode", 2, "actor-0", targets)
    runtime.record_experience(experience)
    if future_label:
        runtime.record_label(label)
    for tick in (0, 1):
        clock.advance_to(tick)
        assert runtime.train_ready() == ()
        assert budget.updates_completed == 0
        assert_same_state(original, runtime.candidate_snapshot().state)
        np.testing.assert_array_equal(
            runtime.actor.predict(features).prediction, original_prediction
        )
    clock.advance_to(2)
    if not future_label:
        assert runtime.train_ready() == ()
        runtime.record_label(label)
    (receipt,) = runtime.train_ready()
    assert (receipt.observed_at, receipt.arrived_at, receipt.applied_at) == (0, 2, 2)
    assert receipt.diagnostic.definition == energy.CONTINUOUS_PC_ENERGY_ID
    assert budget.updates_completed == 1 and runtime.train_ready() == ()
    candidate = runtime.candidate_snapshot().state
    assert candidate.completed_updates == 1 and candidate.training_examples == 2
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(candidate.parameters, original.parameters, strict=True)
    )
    assert_same_state(original, source.snapshot_state())
    assert_same_state(original, runtime.actor.snapshot_state().state)
    np.testing.assert_array_equal(runtime.actor.predict(features).prediction, original_prediction)
    if not future_label:
        runtime.record_experience(
            Experience(
                "second",
                "episode",
                2,
                "actor-0",
                features,
                "train",
                ExperiencePermissions(training=True),
            )
        )
        runtime.record_label(
            LabelArrival("second-label", "second", "episode", 2, "actor-0", targets)
        )
        monkeypatch.setattr(
            adapter.NumpyContinuousPCLearner,
            "train_batch",
            lambda *_: pytest.fail("spent budget admitted another PC update"),
        )
        with pytest.raises(ToyExecutionStopped):
            runtime.train_ready()
        assert budget.updates_completed == 1 and len(runtime.applied_updates) == 1
    features[:] = 100
    targets[:] = -100
    assert_same_state(candidate, runtime.candidate_snapshot().state)


def test_should_stop_failed_runtime_candidate_but_keep_actual_stable_actor_available(monkeypatch):
    source = model()
    original = source.snapshot_state()
    features, targets = arrays()
    expected = source.predict(features)
    clock = LogicalClock(2)
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
    runtime = ActorShadowRuntime(
        source,
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=clock,
        budget=budget,
        max_experiences=1,
        max_consolidations=0,
    )
    runtime.record_experience(
        Experience(
            "packet",
            "episode",
            0,
            "actor-0",
            features,
            "train",
            ExperiencePermissions(training=True),
        )
    )
    runtime.record_label(LabelArrival("label", "packet", "episode", 2, "actor-0", targets))
    monkeypatch.setattr(
        adapter.energy,
        "evaluate_continuous_pc",
        lambda *_: (_ for _ in ()).throw(FloatingPointError("initial energy failed")),
    )
    with pytest.raises(TrainingUpdateFailed) as caught:
        runtime.train_ready()
    assert caught.value.work["energy_evaluations"] == 1
    assert caught.value.work["published_updates"] == 0
    assert budget.updates_completed == 0 and runtime.applied_updates == ()
    with pytest.raises(ValueError, match="stopped"):
        runtime.train_ready()
    np.testing.assert_array_equal(runtime.actor.predict(features).prediction, expected)
    assert_same_state(original, source.snapshot_state())
