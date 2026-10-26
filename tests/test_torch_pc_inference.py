"""Atomic bounded TRAIN episode publication, native state, and paid failures."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import pickle
import random
from typing import Any

import numpy as np
import pytest

from src.adapters import torch_feature_learner as native
from src.adapters import torch_pc_inference as bounded
from src.adapters import torch_pc_weighted as fixed
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeaturePolicy,
)
from src.core import categorical_pc_inference as core
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.demonstration_episode import episode_arrays
from src.core.learner_ports import TrainingUpdateFailed
from src.core.resnet50_variants import CircadianPredictiveCodingHead, PredictiveCodingHead
from test_torch_feature_learner import NAMES, assert_tree, tensors
from test_torch_pc_weighted import (
    ONES,
    WEIGHTS,
    episode,
    install_after_commit,
    owner,
    pc_owner,
    torch as torch,
)


@pytest.fixture(autouse=True)
def work_accounting(monkeypatch, record_property):
    work = dict(
        bounded_core_attempts=0,
        bounded_core_rows=0,
        bounded_configured_latent_row_iterations=0,
        bounded_attempted_latent_row_iterations=0,
        fixed_core_attempts=0,
        fixed_core_rows=0,
        fixed_nominal_latent_row_iterations=0,
        native_calls=0,
        native_rows=0,
        native_nominal_latent_row_iterations=0,
        transition_attempts=0,
        transition_completed=0,
        transition_rows=0,
    )
    proposal = bounded.bounded_class_weighted_pc_update

    def propose(features, targets, parameters, **options):
        work["bounded_core_attempts"] += 1
        work["bounded_core_rows"] += len(features)
        work["bounded_configured_latent_row_iterations"] += (
            len(features) * options["inference_steps"]
        )
        try:
            result = proposal(features, targets, parameters, **options)
        except TrainingUpdateFailed as error:
            work["bounded_attempted_latent_row_iterations"] += error.work.get(
                "attempted_latent_row_iterations", 0
            )
            raise
        work["bounded_attempted_latent_row_iterations"] += (
            result.diagnostics.attempted_latent_row_iterations
        )
        return result

    monkeypatch.setattr(bounded, "bounded_class_weighted_pc_update", propose)
    original = fixed.class_weighted_pc_update

    def fixed_proposal(features, targets, parameters, **options):
        work["fixed_core_attempts"] += 1
        work["fixed_core_rows"] += len(features)
        work["fixed_nominal_latent_row_iterations"] += len(features) * options["inference_steps"]
        return original(features, targets, parameters, **options)

    monkeypatch.setattr(fixed, "class_weighted_pc_update", fixed_proposal)
    for kind in (PredictiveCodingHead, CircadianPredictiveCodingHead):
        original_transition, original_train = kind._apply_training_gradients, kind.train_step

        def transition(head, hidden, *args, _original=original_transition, **options):
            work["transition_attempts"] += 1
            work["transition_rows"] += len(hidden)
            result = _original(head, hidden, *args, **options)
            work["transition_completed"] += 1
            return result

        def train(head, features, targets, rate, steps, inference_rate, _original=original_train):
            work["native_calls"] += 1
            work["native_rows"] += len(features)
            work["native_nominal_latent_row_iterations"] += len(features) * steps
            return _original(head, features, targets, rate, steps, inference_rate)

        monkeypatch.setattr(kind, "_apply_training_gradients", transition)
        monkeypatch.setattr(kind, "train_step", train)
    yield work
    for name, value in work.items():
        record_property(name, value)


def policy(**changes):
    values: dict[str, Any] = dict(
        min_steps=1, gradient_tolerance=None, relative_energy_tolerance=None
    )
    values.update(changes)
    return CoupledInferencePolicy(**values)


def invoke(learner, packet=None, weights=WEIGHTS, *, settings=None):
    return bounded.train_bounded_weighted_episode(
        learner,
        episode() if packet is None else packet,
        weights,
        policy=policy() if settings is None else settings,
    )


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
@pytest.mark.parametrize("steps", [2, 64])
def test_should_match_fixed_complete_native_state_and_own_result_work(torch, method, steps):
    learner, reference, packet = pc_owner(method, steps), pc_owner(method, steps), episode()
    result = invoke(learner, packet, ONES)
    expected = fixed.train_weighted_episode(reference, packet, ONES)
    assert result.diagnostic.value == expected.value
    assert result.inference.reason == "max_steps" and result.inference.accepted_steps == steps
    assert_tree(torch, learner.snapshot_state(), reference.snapshot_state())
    assert learner.completed_updates == 1 and learner.training_examples == 4
    work = result.work
    assert len(work) == 32
    assert all(type(value) in (int, bool) for value in work.values())
    for key in (
        "feature_projection_attempts",
        "native_state_restores_attempted",
        "staging_context_evaluations_attempted",
        "native_transitions_attempted",
        "native_transitions_completed",
        "candidate_validations_attempted",
        "candidate_validations_completed",
        "published_updates",
    ):
        assert work[key] == 1
    work["published_updates"] = 999
    assert result.work["published_updates"] == 1
    assert tensors(torch, result) == []
    with pytest.raises(FrozenInstanceError):
        result.diagnostic = expected


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
@pytest.mark.parametrize("steps", [2, 64])
def test_should_early_stop_at_one_with_full_state_equal_to_one_fixed_step(torch, method, steps):
    learner, reference = pc_owner(method, steps), pc_owner(method, 1)
    result = invoke(learner, settings=policy(gradient_tolerance=1e20))
    fixed.train_weighted_episode(reference, episode(), WEIGHTS)
    actual, expected = learner.snapshot_state(), reference.snapshot_state()
    # The reference requests one fixed step; the bounded owner retains its
    # original maximum. Compare every other field, including full native history.
    assert actual.policy.native_policy.inference_steps == steps
    assert expected.policy.native_policy.inference_steps == 1
    expected = replace(
        expected,
        policy=actual.policy,
        native_state=replace(expected.native_state, policy=actual.native_state.policy),
    )
    assert_tree(torch, actual, expected)
    assert result.inference.reason == "gradient_tolerance"
    assert result.inference.accepted_steps == result.inference.attempted_steps == 1
    assert result.work["attempted_latent_row_iterations"] == 4
    assert result.work["published_updates"] == 1


def test_should_match_detached_core_oracle_and_pc_neutral_parameters(torch, record_property):
    learner, neutral, packet = owner("pc"), owner("neutral_cpc"), episode()
    before = learner.snapshot_state().native_state
    raw, labels = episode_arrays(packet)
    values = native._features(learner._features(raw), 296)
    expected = core.bounded_class_weighted_pc_update(
        values,
        torch.from_numpy(labels[:, 0].astype(np.int64)),
        tuple(before.head_state[name] for name in NAMES),
        class_weights=WEIGHTS,
        learning_rate=0.03,
        inference_steps=2,
        inference_learning_rate=0.04,
        role="train",
        policy=policy(),
    )
    record_property("direct_oracle_core_attempts", 1)
    record_property("direct_oracle_rows", 4)
    record_property("direct_oracle_configured_latent_row_iterations", 8)
    first, second = invoke(learner, packet), invoke(neutral, packet)
    assert first.diagnostic == second.diagnostic
    assert first.inference == second.inference == expected.diagnostics
    for name, candidate in zip(NAMES, expected.update.parameters, strict=True):
        assert torch.equal(learner.snapshot_state().native_state.head_state[name], candidate)
        assert torch.equal(neutral.snapshot_state().native_state.head_state[name], candidate)
    for item in (learner, neutral):
        state = item.snapshot_state().native_state.head_state
        assert state["traffic_steps"] == 1
    state = neutral.snapshot_state().native_state.head_state
    assert state["wake_examples"] == 4 and state["steps_since_sleep"] == 1
    assert state["energy_history"] == [expected.update.legacy_diagnostic]


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
def test_should_publish_one_charged_transition_even_when_no_latent_attempt_is_accepted(
    torch, method
):
    original = owner(method).policy.native_policy
    learner = InvariantTorchFeatureLearner(
        InvariantTorchFeaturePolicy(replace(original, inference_learning_rate=1e6))
    )
    result = invoke(learner)
    assert result.inference.reason == "energy_increase" and result.inference.used_fallback
    assert result.inference.attempted_steps == result.inference.rejected_steps == 1
    assert result.inference.accepted_steps == result.work["accepted_latent_row_iterations"] == 0
    assert result.work["attempted_latent_row_iterations"] == 4
    assert result.work["native_transitions_completed"] == result.work["published_updates"] == 1
    assert learner.completed_updates == 1 and learner.training_examples == 4


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
def test_should_preserve_fork_restore_donor_rng_gradient_mode_and_source(torch, method):
    learner, packet = owner(method), episode()
    donor = learner.snapshot_state()
    child, restored = learner.fork(), owner(method)
    restored.restore_state(donor)
    source = pickle.dumps(packet)
    rng = (torch.get_rng_state().clone(), pickle.dumps(np.random.get_state()), random.getstate())
    with torch.no_grad():
        first, second = invoke(child, packet), invoke(restored, packet)
        assert not torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
    assert first == second
    expected = child.snapshot_state()
    assert_tree(torch, restored.snapshot_state(), expected)
    assert_tree(torch, learner.snapshot_state(), donor)
    assert pickle.dumps(packet) == source
    assert torch.equal(torch.get_rng_state(), rng[0])
    assert pickle.dumps(np.random.get_state()) == rng[1] and random.getstate() == rng[2]
    for value in tensors(torch, expected):
        value.zero_()
    assert_tree(torch, child.snapshot_state(), restored.snapshot_state())
    for left, right in zip(
        tensors(torch, child.snapshot_state()),
        tensors(torch, restored.snapshot_state()),
        strict=True,
    ):
        assert left.data_ptr() != right.data_ptr()


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
@pytest.mark.parametrize("fault", ["core", "before_transition", "transition", "candidate"])
def test_should_charge_failed_staging_and_preserve_unpublished_owner(
    torch, monkeypatch, method, fault
):
    learner, packet = owner(method), episode()
    before, payload = learner.snapshot_state(), learner._native._payload
    committed = []

    def after(head):
        committed.append(head)
        if fault == "transition":
            raise OSError("native transition returned then interrupted")

    install_after_commit(monkeypatch, after)
    if fault == "core":

        def fail(*args, **kwargs):
            raise OSError("core energy port interrupted")

        monkeypatch.setattr(core, "_energy", fail)
    elif fault == "before_transition":

        def refuse(*args, **kwargs):
            raise OSError("native transition refused before entry")

        for kind in (PredictiveCodingHead, CircadianPredictiveCodingHead):
            monkeypatch.setattr(kind, "_apply_training_gradients", refuse)
    elif fault == "candidate":
        original_state = native._state

        def state(*args, **kwargs):
            if committed:
                raise OSError("candidate validation interrupted")
            return original_state(*args, **kwargs)

        monkeypatch.setattr(native, "_state", state)
    with pytest.raises(TrainingUpdateFailed) as caught:
        invoke(learner, packet)
    work = caught.value.work
    assert len(work) == 32 and work["published_updates"] == 0
    assert work["feature_projection_attempts"] == work["native_state_restores_attempted"] == 1
    assert work["failure_work_may_be_unobserved"] is True
    if fault == "core":
        assert work["energy_evaluations_attempted"] == 1
        assert work["energy_evaluations_completed"] == work["native_transitions_attempted"] == 0
    else:
        assert work["parameter_proposals_completed"] == 1
        assert work["native_transitions_attempted"] == 1
        assert work["native_transitions_completed"] == (fault == "candidate")
        assert work["candidate_validations_attempted"] == (fault == "candidate")
        assert work["candidate_validations_completed"] == 0
    assert learner._native._payload is payload
    monkeypatch.undo()
    assert_tree(torch, learner.snapshot_state(), before)


def test_should_refuse_a_foreign_grad_mode_change_after_staging_without_restoring_it(
    torch, monkeypatch
):
    learner = owner("pc")
    before, payload = learner.snapshot_state(), learner._native._payload
    original_mode = torch.is_grad_enabled()
    original_stage = bounded._stage

    def stage(*args, **options):
        result = original_stage(*args, **options)
        torch.set_grad_enabled(not original_mode)
        return result

    monkeypatch.setattr(bounded, "_stage", stage)
    try:
        with pytest.raises(TrainingUpdateFailed) as caught:
            invoke(learner)
        assert torch.is_grad_enabled() is not original_mode
        assert learner._native._payload is payload
        assert_tree(torch, learner.snapshot_state(), before)
        assert caught.value.work["native_transitions_completed"] == 1
        assert caught.value.work["candidate_validations_completed"] == 1
        assert caught.value.work["published_updates"] == 0
        assert caught.value.work["failure_work_may_be_unobserved"] is True
    finally:
        torch.set_grad_enabled(original_mode)


def test_should_keep_known_work_when_a_native_port_raises_untrusted_training_failure(
    torch, monkeypatch
):
    learner = owner("pc")
    before, payload = learner.snapshot_state(), learner._native._payload
    foreign_work = {"foreign_" + str(index): index for index in range(32)}
    primary = TrainingUpdateFailed("foreign native receipt", "foreign_native", foreign_work)

    def fail_after_native_transition(head):
        raise primary

    install_after_commit(monkeypatch, fail_after_native_transition)
    with pytest.raises(TrainingUpdateFailed) as caught:
        invoke(learner)
    work = caught.value.work
    assert caught.value.__cause__ is primary
    assert caught.value.definition == bounded.TORCH_BOUNDED_WEIGHTED_PC_ID
    assert len(work) == 32 and not set(work).intersection(foreign_work)
    assert work["parameter_proposals_completed"] == 1
    assert work["native_transitions_attempted"] == 1
    assert work["native_transitions_completed"] == 0
    assert work["candidate_validations_attempted"] == work["published_updates"] == 0
    assert work["failure_work_may_be_unobserved"] is True
    assert primary.work == foreign_work
    assert learner._native._payload is payload
    monkeypatch.undo()
    assert_tree(torch, learner.snapshot_state(), before)


@pytest.mark.parametrize("original,replacement", [(1, 1.0), (0.0, -0.0)])
def test_should_refuse_equal_numeric_policy_changes_and_preserve_the_foreign_value(
    torch, monkeypatch, original, replacement
):
    learner = owner("pc")
    before, payload = learner.snapshot_state(), learner._native._payload
    settings = policy(gradient_tolerance=original)

    def mutate(head):
        object.__setattr__(settings, "gradient_tolerance", replacement)

    install_after_commit(monkeypatch, mutate)
    with pytest.raises(TrainingUpdateFailed) as caught:
        invoke(learner, settings=settings)
    assert type(settings.gradient_tolerance) is type(replacement)
    if type(replacement) is float:
        assert settings.gradient_tolerance.hex() == replacement.hex()
    else:
        assert settings.gradient_tolerance == replacement
    work = caught.value.work
    assert work["native_transitions_completed"] == work["candidate_validations_completed"] == 1
    assert work["published_updates"] == 0 and work["failure_work_may_be_unobserved"] is True
    assert learner._native._payload is payload
    assert_tree(torch, learner.snapshot_state(), before)


@pytest.mark.parametrize(
    "fault", ["parameter", "gradient", "payload", "episode", "policy", "numpy"]
)
def test_should_refuse_closing_foreign_changes_without_rolling_them_back(torch, monkeypatch, fault):
    learner, packet, settings = owner("pc"), episode(), policy()
    rng = deepcopy(np.random.get_state())
    observed = []

    def mutate(head):
        if fault == "parameter":
            learner._native._head.weight_feature_hidden[0, 0] += 0.125
        elif fault == "gradient":
            learner._native._head.weight_feature_hidden.grad = torch.zeros_like(
                learner._native._head.weight_feature_hidden
            )
        elif fault == "payload":
            learner._native._payload = tuple(list(learner._native._payload))
        elif fault == "episode":
            object.__setattr__(packet, "episode_id", "foreign-TRAIN")
        elif fault == "policy":
            object.__setattr__(settings, "gradient_tolerance", 0.0)
        else:
            np.random.random()
        observed.append(
            (
                learner.snapshot_state(),
                learner._native._payload,
                pickle.dumps(packet),
                pickle.dumps(settings),
                pickle.dumps(np.random.get_state()),
            )
        )

    install_after_commit(monkeypatch, mutate)
    try:
        with pytest.raises(TrainingUpdateFailed) as caught:
            invoke(learner, packet, settings=settings)
        work = caught.value.work
        assert work["published_updates"] == 0
        assert work["native_transitions_completed"] == work["candidate_validations_completed"] == 1
        assert work["failure_work_may_be_unobserved"] is True
        assert len(observed) == 1
        saved, payload, source, saved_policy, saved_rng = observed[0]
        assert_tree(torch, learner.snapshot_state(), saved)
        assert learner._native._payload is payload
        assert pickle.dumps(packet) == source and pickle.dumps(settings) == saved_policy
        assert pickle.dumps(np.random.get_state()) == saved_rng
    finally:
        np.random.set_state(rng)


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test"])
def test_should_refuse_nontrain_roles_before_snapshot(monkeypatch, role):
    learner = owner("pc")
    monkeypatch.setattr(
        learner, "snapshot_state", lambda: pytest.fail("unreleased source reached snapshot")
    )
    with pytest.raises(ValueError):
        invoke(learner, episode(role=role))


@pytest.mark.parametrize("method", ["bp", "gated_cpc"])
def test_should_refuse_unmatched_native_methods_before_snapshot(monkeypatch, method):
    learner = owner(method)
    monkeypatch.setattr(
        learner, "snapshot_state", lambda: pytest.fail("unmatched method reached snapshot")
    )
    with pytest.raises(ValueError):
        invoke(learner)


@pytest.mark.parametrize("settings", [None, object(), CoupledInferencePolicy(min_steps=3)])
def test_should_refuse_invalid_or_overcap_policy_before_snapshot(monkeypatch, settings):
    learner = owner("pc")
    monkeypatch.setattr(
        learner, "snapshot_state", lambda: pytest.fail("invalid policy reached snapshot")
    )
    with pytest.raises(ValueError):
        bounded.train_bounded_weighted_episode(learner, episode(), WEIGHTS, policy=settings)


def test_should_refuse_inference_mode_before_any_native_snapshot(torch, monkeypatch):
    learner = owner("pc")
    monkeypatch.setattr(
        learner, "snapshot_state", lambda: pytest.fail("inference mode reached snapshot")
    )
    with torch.inference_mode():
        with pytest.raises(ValueError):
            invoke(learner)
        assert torch.is_inference_mode_enabled()
