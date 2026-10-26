"""Matched continuous CPC controls and owned active wake updates.

Tiny deterministic engineering fixtures; no performance comparison, BP training,
scientific data or native simulator. Public inference supplies accepted partials;
activity, exponential gates and momentum arithmetic are calculated independently.

Static full-module bound: 31 cases, 68 PC/BP constructor entries (four refused),
264 seeded normal values and 20 PC update attempts. Including two public-inference
oracles, energy entries and latent proposals are each bounded by 100. No BP
training, scientific fitting or native work occurs; forks consume no RNG.
"""

from dataclasses import replace
import numpy as np
import pytest

from src.adapters import numpy_continuous_bc as bc
from src.adapters import numpy_continuous_pc as adapter
from src.app.actor_shadow import ActorShadowRuntime
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.continuous_pc_inference import infer_continuous_pc
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_energy import CoupledParameters
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.learner_ports import TrainingUpdateFailed


CPC_WORK = dict(
    chemistry_attempts=1,
    chemistry_returns=1,
    gate_applications_attempted=1,
    gate_applications_returned=1,
)


def spec():
    return ContinuousActionSpec(2, ("m/s", "N"), (-2.0, 4.0), (6.0, 8.0))


def arrays():
    return np.array([[0.2, -0.3], [0.7, 0.4]]), np.array([[1.0, 5.0], [4.0, 7.0]])


def policy(config=None, *, role="train", declared=None):
    return adapter.ContinuousPCPolicy(
        bc.ContinuousBCPolicy(
            spec() if declared is None else declared,
            hidden_dim=2,
            seed=17,
            learning_rate=0.03,
            momentum=0.8,
            training_role=role,
        ),
        precision=0.8,
        inference_steps=2,
        inference_learning_rate=0.05,
        inference_policy=CoupledInferencePolicy(
            min_steps=1, gradient_tolerance=None, relative_energy_tolerance=None
        ),
        circadian=config,
    )


def model(config=None, *, role="train"):
    owner = adapter.NumpyContinuousPCLearner(policy(config, role=role))
    state = owner.snapshot_state()
    parameters = (
        np.array([[0.1, -0.2], [0.3, 0.15]]),
        np.array([0.02, -0.03]),
        np.array([[0.2, -0.1], [-0.3, 0.25]]),
        np.array([0.05, -0.04]),
    )
    owner.restore_state(replace(state, parameters=parameters))
    return owner


def coupled(parameters):
    return CoupledParameters(
        (parameters[0].copy(),),
        (parameters[1][None, :].copy(),),
        parameters[2].copy(),
        parameters[3][None, :].copy(),
    )


def assert_common(first, second):
    assert replace(first.policy, circadian=None) == replace(second.policy, circadian=None)
    assert first.completed_updates == second.completed_updates
    assert first.training_examples == second.training_examples
    for name in ("parameters", "velocity"):
        for a, b in zip(getattr(first, name), getattr(second, name), strict=True):
            np.testing.assert_array_equal(a, b)


def assert_full(first, second):
    assert first.policy == second.policy
    assert_common(first, second)
    if first.wake_state is None:
        assert second.wake_state is None
        return
    assert second.wake_state is not None
    for name in (
        "chemical_decay",
        "chemical_buildup_rate",
        "plasticity_sensitivity",
        "min_plasticity",
        "wake_updates",
    ):
        assert getattr(first.wake_state, name) == getattr(second.wake_state, name)
    np.testing.assert_array_equal(first.wake_state.chemical_state, second.wake_state.chemical_state)
    np.testing.assert_array_equal(
        first.wake_state.plasticity_state, second.wake_state.plasticity_state
    )


def common_work(result):
    return {key: value for key, value in result.work.items() if key not in CPC_WORK}


def test_should_start_actual_four_axis_neutral_with_exact_pc_parameters_and_extra_owned_wake():
    declared = ContinuousActionSpec(
        2,
        ("normalized_dx", "normalized_dy", "normalized_dz", "normalized_gripper"),
        (-1.0,) * 4,
        (1.0,) * 4,
    )
    config = CoupledCircadianConfig.matched_pc_control()
    pc = adapter.NumpyContinuousPCLearner(policy(declared=declared))
    neutral = adapter.NumpyContinuousPCLearner(policy(config, declared=declared))
    first, second = pc.snapshot_state(), neutral.snapshot_state()
    assert_common(first, second)
    assert first.wake_state is None
    np.testing.assert_array_equal(neutral.get_chemical_state(), [0.0, 0.0])
    np.testing.assert_array_equal(neutral.get_plasticity_state(), [1.0, 1.0])
    assert neutral.wake_updates == 0
    assert second.policy.circadian is not config
    object.__setattr__(config, "chemical_decay", 0.5)
    returned = neutral.get_chemical_state()
    returned[:] = 100
    gates = neutral.get_plasticity_state()
    gates[:] = 0
    assert_full(second, neutral.snapshot_state())
    features, _ = arrays()
    np.testing.assert_array_equal(pc.predict(features), neutral.predict(features))
    for getter in (pc.get_chemical_state, pc.get_plasticity_state, lambda: pc.wake_updates):
        with pytest.raises(ValueError):
            getter()


def test_should_match_pc_exactly_through_repeated_neutral_updates_restore_and_rng_free_forks(
    monkeypatch,
):
    pc, neutral = model(), model(CoupledCircadianConfig.matched_pc_control())
    features, targets = arrays()
    for update in range(1, 4):
        first, second = (
            pc.train_batch_with_work(features, targets),
            neutral.train_batch_with_work(features, targets),
        )
        assert first.diagnostic == second.diagnostic and first.task_loss == second.task_loss
        assert first.inference == second.inference
        assert first.work == common_work(second)
        assert not any(key in first.work for key in CPC_WORK)
        assert {key: second.work[key] for key in CPC_WORK} == CPC_WORK
        assert_common(pc.snapshot_state(), neutral.snapshot_state())
        assert neutral.wake_updates == update
        np.testing.assert_array_equal(neutral.get_plasticity_state(), [1.0, 1.0])
        assert np.all(neutral.get_chemical_state() > 0)
        np.testing.assert_array_equal(pc.predict(features), neutral.predict(features))
    initial_pc, initial_neutral = pc.snapshot_state(), neutral.snapshot_state()
    assert any(np.any(value) for value in initial_pc.velocity)
    restored_pc, restored_neutral = model(), model(CoupledCircadianConfig.matched_pc_control())
    restored_pc.restore_state(initial_pc)
    restored_neutral.restore_state(initial_neutral)
    monkeypatch.setattr(bc.np.random, "default_rng", lambda *_: pytest.fail("fork used RNG"))
    child_pc, child_neutral = pc.fork(), neutral.fork()
    results = [
        owner.train_batch_with_work(features, targets)
        for owner in (child_pc, child_neutral, restored_pc, restored_neutral)
    ]
    for owner in (child_neutral, restored_pc, restored_neutral):
        assert_common(child_pc.snapshot_state(), owner.snapshot_state())
    assert_full(child_neutral.snapshot_state(), restored_neutral.snapshot_state())
    assert (
        results[0].diagnostic
        == results[1].diagnostic
        == results[2].diagnostic
        == results[3].diagnostic
    )
    assert results[0].work == common_work(results[1])
    assert_full(initial_pc, pc.snapshot_state())
    assert_full(initial_neutral, neutral.snapshot_state())
    source_chemistry = neutral.get_chemical_state()
    initial_neutral.wake_state.chemical_state[:] = 100
    initial_neutral.wake_state.plasticity_state[:] = 0
    np.testing.assert_array_equal(neutral.get_chemical_state(), source_chemistry)
    assert_full(child_neutral.snapshot_state(), restored_neutral.snapshot_state())
    assert neutral.wake_updates == 3 and child_neutral.wake_updates == 4


def test_should_apply_hand_computed_active_chemistry_gates_and_ungated_old_momentum_twice():
    config = CoupledCircadianConfig(
        chemical_decay=0.6,
        chemical_buildup_rate=0.4,
        plasticity_sensitivity=1.3,
        min_plasticity=0.2,
    )
    owner = model(config)
    before = owner.snapshot_state()
    chemistry = np.array([0.2, 2.0])
    gates = np.clip(np.exp(-1.3 * chemistry), 0.2, 1.0)
    velocity = tuple(
        np.full_like(value, 0.03 * (index + 1)) for index, value in enumerate(before.velocity)
    )
    owner.restore_state(
        replace(
            before,
            completed_updates=1,
            training_examples=2,
            velocity=velocity,
            wake_state=replace(
                before.wake_state, chemical_state=chemistry, plasticity_state=gates, wake_updates=1
            ),
        )
    )
    features, targets = arrays()
    for update in (2, 3):
        before = owner.snapshot_state()
        oracle = infer_continuous_pc(
            coupled(before.parameters),
            features,
            targets,
            (0.8,),
            spec(),
            max_steps=2,
            learning_rate=0.05,
            policy=before.policy.inference_policy,
        )
        next_chemistry = 0.6 * before.wake_state.chemical_state + 0.4 * np.mean(
            np.abs(oracle.latents[-1]), axis=0
        )
        next_gates = np.clip(np.exp(-1.3 * next_chemistry), 0.2, 1.0)
        derivative = oracle.evaluation.parameter_gradients
        ungated = (
            derivative.hidden_weights[0],
            derivative.hidden_biases[0][0],
            derivative.output_weight,
            derivative.output_bias[0],
        )
        gradients = (
            ungated[0] * next_gates,
            ungated[1] * next_gates,
            ungated[2] * next_gates[:, None],
            ungated[3],
        )
        expected_velocity = tuple(
            0.8 * old + gradient for old, gradient in zip(before.velocity, gradients, strict=True)
        )
        expected_parameters = tuple(
            old - 0.03 * speed
            for old, speed in zip(before.parameters, expected_velocity, strict=True)
        )
        result = owner.train_batch_with_work(features, targets)
        after = owner.snapshot_state()
        assert result.diagnostic.value == oracle.evaluation.energy
        assert result.task_loss == oracle.evaluation.task_loss
        assert result.inference == oracle.diagnostics
        assert {key: result.work[key] for key in CPC_WORK} == CPC_WORK
        np.testing.assert_allclose(
            after.wake_state.chemical_state, next_chemistry, rtol=0, atol=1e-15
        )
        np.testing.assert_allclose(
            after.wake_state.plasticity_state, next_gates, rtol=0, atol=1e-15
        )
        assert after.wake_state.wake_updates == after.completed_updates == update
        for actual, expected in zip(after.velocity, expected_velocity, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
        for actual, expected in zip(after.parameters, expected_parameters, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
        np.testing.assert_allclose(
            after.velocity[3], 0.8 * before.velocity[3] + ungated[3], rtol=0, atol=1e-15
        )
        assert np.any(next_gates < 1)


@pytest.mark.parametrize(
    "bad",
    [
        "missing",
        "wrong_type",
        "dtype",
        "shape",
        "negative",
        "nan",
        "wrong_gates",
        "bool_count",
        "count_mismatch",
        "zero_coherence",
        "config",
    ],
)
def test_should_refuse_corrupt_or_missing_wake_without_publishing_any_state(bad):
    owner = model(CoupledCircadianConfig())
    before = owner.snapshot_state()
    wake = owner.snapshot_state().wake_state
    if bad == "missing":
        wake = None
    elif bad == "wrong_type":
        wake = object()
    elif bad in ("dtype", "shape", "negative", "nan", "zero_coherence"):
        chemicals = wake.chemical_state.copy()
        if bad == "dtype":
            chemicals = chemicals.astype(np.float32)
        elif bad == "shape":
            chemicals = chemicals[:1]
        else:
            chemicals[0] = {"negative": -1.0, "nan": np.nan, "zero_coherence": 1.0}[bad]
        wake = replace(wake, chemical_state=chemicals)
        if bad == "zero_coherence":
            wake = replace(wake, plasticity_state=np.clip(np.exp(-0.7 * chemicals), 0.2, 1.0))
    elif bad == "wrong_gates":
        wake.plasticity_state[0] = 0.5
    elif bad == "config":
        wake = replace(wake, chemical_decay=0.5)
    else:
        wake = replace(wake, wake_updates=True if bad == "bool_count" else 1)
    with pytest.raises(ValueError):
        owner.restore_state(replace(before, wake_state=wake))
    assert_full(before, owner.snapshot_state())


def test_should_refuse_cpc_wake_in_pc_and_policy_mode_replacement():
    pc, active = model(), model(CoupledCircadianConfig())
    pc_before, active_before = pc.snapshot_state(), active.snapshot_state()
    with pytest.raises(ValueError):
        pc.restore_state(replace(pc_before, wake_state=active_before.wake_state))
    with pytest.raises(ValueError):
        active.restore_state(replace(active_before, policy=pc_before.policy))
    assert_full(pc_before, pc.snapshot_state())
    assert_full(active_before, active.snapshot_state())


@pytest.mark.parametrize(
    "field,value",
    [
        ("chemical_decay", True),
        ("chemical_buildup_rate", -1.0),
        ("plasticity_sensitivity", float("nan")),
        ("min_plasticity", 0.0),
    ],
)
def test_should_revalidate_corrupted_circadian_policy_before_rng(monkeypatch, field, value):
    original = policy(CoupledCircadianConfig())
    object.__setattr__(original.circadian, field, value)
    monkeypatch.setattr(
        bc,
        "NumpyContinuousBCLearner",
        lambda *_: pytest.fail("invalid circadian policy reached RNG"),
    )
    with pytest.raises(ValueError):
        adapter.NumpyContinuousPCLearner(original)


@pytest.mark.parametrize(
    "where", ["inference", "chemistry", "gate", "wake", "parameter", "candidate"]
)
def test_should_keep_all_wake_optimizer_and_model_state_when_later_work_fails(monkeypatch, where):
    owner = model(CoupledCircadianConfig())
    before = owner.snapshot_state()
    features, targets = arrays()
    if where == "inference":
        monkeypatch.setattr(
            adapter.energy,
            "evaluate_continuous_pc",
            lambda *_: (_ for _ in ()).throw(FloatingPointError("initial inference failed")),
        )
    elif where == "chemistry":
        monkeypatch.setattr(
            adapter.cpc,
            "advance_continuous_wake",
            lambda *_: (_ for _ in ()).throw(FloatingPointError("chemistry failed")),
        )
    elif where == "gate":
        monkeypatch.setattr(
            adapter.cpc,
            "gate_continuous_gradients",
            lambda *_: (_ for _ in ()).throw(ValueError("gate failed")),
        )
    elif where == "wake":
        original = adapter.cpc.advance_continuous_wake

        def corrupt_wake(*args):
            return replace(original(*args), wake_updates=0)

        monkeypatch.setattr(adapter.cpc, "advance_continuous_wake", corrupt_wake)
    elif where == "parameter":
        original_snapshot = adapter._snapshot

        def fail_candidate(value, owner_policy):
            if value.completed_updates > before.completed_updates:
                raise ValueError("candidate validation failed")
            return original_snapshot(value, owner_policy)

        monkeypatch.setattr(adapter, "_snapshot", fail_candidate)
    else:
        monkeypatch.setattr(
            adapter.energy,
            "predict_continuous_pc",
            lambda *_: (_ for _ in ()).throw(FloatingPointError("candidate forward failed")),
        )
    with pytest.raises(TrainingUpdateFailed) as caught:
        owner.train_batch_with_work(features, targets)
    work = caught.value.work
    assert work["chemistry_attempts"] == int(where != "inference")
    assert work["chemistry_returns"] == int(where not in ("inference", "chemistry"))
    assert work["gate_applications_attempted"] == int(where not in ("inference", "chemistry"))
    assert work["gate_applications_returned"] == int(where in ("wake", "parameter", "candidate"))
    assert work["parameter_proposals_attempted"] == int(where in ("wake", "parameter", "candidate"))
    assert work["parameter_proposals_completed"] == int(where == "candidate")
    assert work["candidate_predictions_attempted"] == int(where == "candidate")
    assert work["candidate_predictions_completed"] == work["published_updates"] == 0
    assert work["inference_returns"] == int(where != "inference")
    assert work["energy_evaluations"] == (1 if where == "inference" else 3)
    assert_full(before, owner.snapshot_state())


def test_should_advance_chemistry_from_last_accepted_latents_when_first_proposal_falls_back(
    monkeypatch,
):
    owner = model(CoupledCircadianConfig())
    before = owner.snapshot_state()
    features, targets = arrays()
    original = adapter.energy.evaluate_continuous_pc
    accepted: list[tuple[np.ndarray, ...]] = []

    def reject_candidate(*args, **kwargs):
        if accepted:
            raise FloatingPointError("rejected inferred candidate")
        accepted.append(tuple(value.copy() for value in args[3]))
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter.energy, "evaluate_continuous_pc", reject_candidate)
    result = owner.train_batch_with_work(features, targets)
    assert result.inference.used_fallback and result.inference.accepted_steps == 0
    assert result.work["energy_evaluations"] == 2 and result.work["energy_returns"] == 1
    np.testing.assert_allclose(
        owner.get_chemical_state(),
        0.02 * np.mean(np.abs(accepted[0][-1]), axis=0),
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_array_equal(owner.snapshot_state().parameters[0], before.parameters[0])
    assert owner.wake_updates == 1


@pytest.mark.parametrize("role", ["probe_train", "inner_guard", "probe_test", "final_test"])
def test_should_refuse_unreleased_or_other_role_labels_before_chemistry(monkeypatch, role):
    owner = model(CoupledCircadianConfig())
    before = owner.snapshot_state()
    value = ContinuousEpisode(
        "synthetic",
        "context",
        role,
        spec(),
        (ContinuousTransition(0, (0.2, -0.3), (1.0, 5.0), 1.0, (0.7, 0.4), True, False),),
    )
    monkeypatch.setattr(
        adapter.energy,
        "feedforward_continuous_latents",
        lambda *_: pytest.fail("unreleased labels initialized PC"),
    )
    monkeypatch.setattr(
        adapter.cpc,
        "advance_continuous_wake",
        lambda *_: pytest.fail("unreleased labels advanced chemistry"),
    )
    with pytest.raises(ValueError):
        owner.train_episode(value)
    assert_full(before, owner.snapshot_state())


def test_should_apply_actual_delayed_cpc_candidate_once_without_changing_source_or_serving_wake():
    source = model(CoupledCircadianConfig())
    initial = source.snapshot_state()
    features, targets = arrays()
    prediction = source.predict(features)
    clock = LogicalClock()
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
    for tick in (0, 1):
        clock.advance_to(tick)
        assert runtime.train_ready() == () and budget.updates_completed == 0
        assert_full(initial, runtime.candidate_snapshot().state)
        np.testing.assert_array_equal(runtime.actor.predict(features).prediction, prediction)
    clock.advance_to(2)
    (receipt,) = runtime.train_ready()
    assert (receipt.observed_at, receipt.arrived_at, receipt.applied_at) == (0, 2, 2)
    assert budget.updates_completed == 1 and runtime.train_ready() == ()
    final = runtime.candidate_snapshot().state
    assert final.completed_updates == final.wake_state.wake_updates == 1
    assert final.training_examples == 2 and np.all(final.wake_state.chemical_state > 0)
    assert_full(initial, source.snapshot_state())
    assert_full(initial, runtime.actor.snapshot_state().state)
    np.testing.assert_array_equal(runtime.actor.predict(features).prediction, prediction)
    features[:] = 999
    targets[:] = -999
    assert_full(final, runtime.candidate_snapshot().state)
