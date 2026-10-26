"""Owned bounded TRAIN schedules and rich receipts in the existing stream."""

from copy import deepcopy
from dataclasses import asdict, replace
import pickle
from typing import Any

import pytest

from src.adapters import torch_pc_inference as backend
from src.adapters import torch_train_weighted_learner as adapter
from src.adapters.invariant_torch_feature_learner import InvariantTorchFeatureLearner
from src.app import coupled_attribution as stream_app
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.categorical_training_diagnostic import (
    CategoricalPCTrainingDiagnostic,
    ScheduledCategoricalUpdateFailed,
)
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.demonstration_class_weights import episode_class_weights
from src.core.demonstration_episode import episode_arrays
from src.core.learner_ports import TrainingDiagnostic, TrainingUpdateFailed
from src.core.resnet50_variants import CircadianPredictiveCodingHead, PredictiveCodingHead
from test_torch_feature_learner import assert_tree, tensors
from test_weighted_pc_stream_adapter import episode, pc_policy, schedule, torch as torch


@pytest.fixture(autouse=True)
def work_accounting(monkeypatch, record_property):
    work = dict(
        core_attempts=0,
        core_rows=0,
        configured_latent_row_iterations=0,
        attempted_latent_row_iterations=0,
        native_calls=0,
        native_rows=0,
        native_nominal_latent_row_iterations=0,
        transition_attempts=0,
        transition_completed=0,
        backend_updates_returned=0,
        backend_publications_returned=0,
    )
    proposal = backend.bounded_class_weighted_pc_update

    def propose(features, targets, parameters, **options):
        work["core_attempts"] += 1
        work["core_rows"] += len(features)
        work["configured_latent_row_iterations"] += len(features) * options["inference_steps"]
        try:
            result = proposal(features, targets, parameters, **options)
        except TrainingUpdateFailed as error:
            work["attempted_latent_row_iterations"] += error.work.get(
                "attempted_latent_row_iterations", 0
            )
            raise
        work["attempted_latent_row_iterations"] += (
            result.diagnostics.attempted_latent_row_iterations
        )
        return result

    monkeypatch.setattr(backend, "bounded_class_weighted_pc_update", propose)
    original = backend.train_bounded_weighted_episode

    def bounded(owner, packet, weights, **options):
        result = original(owner, packet, weights, **options)
        work["backend_updates_returned"] += 1
        work["backend_publications_returned"] += result.work["published_updates"]
        return result

    monkeypatch.setattr(backend, "train_bounded_weighted_episode", bounded)
    monkeypatch.setattr(adapter, "train_bounded_weighted_episode", bounded)
    for kind in (PredictiveCodingHead, CircadianPredictiveCodingHead):
        old_transition, old_train = kind._apply_training_gradients, kind.train_step

        def transition(head, *args, _original=old_transition, **options):
            work["transition_attempts"] += 1
            result = _original(head, *args, **options)
            work["transition_completed"] += 1
            return result

        def train(head, features, targets, rate, steps, inference_rate, _original=old_train):
            work["native_calls"] += 1
            work["native_rows"] += len(features)
            work["native_nominal_latent_row_iterations"] += len(features) * steps
            return _original(head, features, targets, rate, steps, inference_rate)

        monkeypatch.setattr(kind, "_apply_training_gradients", transition)
        monkeypatch.setattr(kind, "train_step", train)
    yield work
    for key, value in work.items():
        record_property(key, value)


def stopping(**changes):
    values: dict[str, Any] = dict(
        min_steps=1, gradient_tolerance=None, relative_energy_tolerance=None
    )
    values.update(changes)
    return CoupledInferencePolicy(**values)


def learner(method="pc", *, records=None, steps=2, settings=None):
    return adapter.TorchTrainWeightedLearner(
        pc_policy(method, steps),
        schedule() if records is None else records,
        inference_policy=stopping() if settings is None else settings,
    )


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
@pytest.mark.parametrize("steps,early", [(2, False), (64, False), (64, True)])
def test_should_preserve_full_direct_backend_state_and_return_rich_owned_receipts(
    torch, method, steps, early
):
    settings = stopping(gradient_tolerance=1e20) if early else stopping()
    source = learner(method, steps=steps, settings=settings)
    reference = InvariantTorchFeatureLearner(pc_policy(method, steps))
    for packet in schedule():
        actual = source.train_batch(*episode_arrays(packet))
        expected = backend.train_bounded_weighted_episode(
            reference, packet, episode_class_weights(packet), policy=settings
        )
        assert type(actual) is CategoricalPCTrainingDiagnostic
        assert (
            actual.definition == expected.diagnostic.definition
            and actual.value == expected.diagnostic.value
        )
        assert actual.inference == expected.inference and actual.inference_policy == settings
        assert actual.work == expected.work
        assert actual.inference.accepted_steps == (1 if early else steps)
        assert_tree(torch, source.snapshot_state(), reference.snapshot_state())
        assert actual.scheduling_work == dict(
            candidate_forks_attempted=1,
            candidate_forks_completed=1,
            publication_restores_attempted=1,
            publication_restores_completed=1,
            publication_validations_attempted=1,
            publication_validations_completed=1,
            published_updates=1,
        )
        native, outer = actual.work, actual.scheduling_work
        native["published_updates"] = outer["published_updates"] = 999
        assert actual.work["published_updates"] == actual.scheduling_work["published_updates"] == 1
        assert tensors(torch, actual) == []
    assert source.completed_updates == 2 and source.training_examples == 7


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
def test_should_preserve_exact_base_diagnostic_and_default_fixed_path(torch, method):
    source = adapter.TorchTrainWeightedLearner(pc_policy(method), schedule())
    explicit_none = adapter.TorchTrainWeightedLearner(
        pc_policy(method), schedule(), inference_policy=None
    )
    assert source.inference_policy is explicit_none.inference_policy is None
    first = source.train_batch(*episode_arrays(schedule()[0]))
    second = explicit_none.train_batch(*episode_arrays(schedule()[0]))
    assert type(first) is type(second) is TrainingDiagnostic and first == second
    assert_tree(torch, source.snapshot_state(), explicit_none.snapshot_state())


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
def test_should_own_stopping_policy_and_preserve_it_across_fork_deepcopy_and_restore(torch, method):
    settings = stopping(gradient_tolerance=1e20)
    source = learner(method, settings=settings)
    initial = source.snapshot_state()
    assert source.inference_policy == settings and source.inference_policy is not settings
    observed = source.inference_policy
    object.__setattr__(observed, "gradient_tolerance", 0.0)
    object.__setattr__(settings, "min_steps", 2)
    assert (
        source.inference_policy.min_steps == 1
        and source.inference_policy.gradient_tolerance == 1e20
    )
    source.train_batch(*episode_arrays(schedule()[0]))
    saved = source.snapshot_state()
    fork, copied = source.fork(), deepcopy(source)
    for child in (fork, copied):
        assert child.inference_policy == source.inference_policy
        assert child._inference_policy is not source._inference_policy
        result = child.train_batch(*episode_arrays(schedule()[1]))
        assert result.inference.accepted_steps == 1
    assert_tree(torch, fork.snapshot_state(), copied.snapshot_state())
    assert_tree(torch, source.snapshot_state(), saved)
    for left, right in zip(
        tensors(torch, source.snapshot_state()), tensors(torch, fork.snapshot_state()), strict=True
    ):
        assert left.data_ptr() != right.data_ptr()
    source.restore_state(initial)
    assert source.inference_policy.gradient_tolerance == 1e20
    source.train_batch(*episode_arrays(schedule()[0]))
    assert_tree(torch, source.snapshot_state(), saved)


@pytest.mark.parametrize("fault", ["equal_replacement", "equal_type_change", "owner_seal"])
def test_should_refuse_private_stopping_metadata_changes_before_native_work(
    torch, work_accounting, fault
):
    source = learner(settings=stopping(gradient_tolerance=1))
    if fault == "equal_replacement":
        source._inference_policy = replace(source._inference_policy)
    elif fault == "equal_type_change":
        object.__setattr__(source._inference_policy, "gradient_tolerance", 1.0)
    else:
        source._inference_policy_seal = stopping(gradient_tolerance=1)
    with pytest.raises(ValueError):
        source.train_batch(*episode_arrays(schedule()[0]))
    assert work_accounting["core_attempts"] == work_accounting["transition_attempts"] == 0


def packets_for(records):
    return tuple(
        stream_app.AttributionBatch(
            "train",
            record.context,
            tuple(record.episode_id + ":" + str(index) for index in range(len(record.transitions))),
            *episode_arrays(record),
            num_classes=7,
        )
        for record in records
    )


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
def test_should_pause_resume_three_delayed_wakes_with_rich_cursor_and_stable_actor(
    torch, method, work_accounting
):
    records = (*schedule(), episode("third", (1, 6), cue=4))
    source = learner(method, records=records, settings=stopping(gradient_tolerance=1e20))
    initial = source.snapshot_state()
    packets = packets_for(records)
    originals = pickle.dumps(packets)
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=3), lambda: 0.0)
    stream = stream_app.AttributionStream(
        source,
        packets,
        (),
        checkpoint_updates=(0, 1, 2, 3),
        label_delay=1,
        budget=budget,
        training_only=True,
    )
    actor, runtime = stream._runtime.actor, stream._runtime
    stream.run_until(1)
    token = stream.pause()
    saved = token.snapshot_state()
    receipt = saved["inbox_cursor"].applied[0]
    assert type(receipt.diagnostic) is CategoricalPCTrainingDiagnostic
    assert (
        receipt.diagnostic.work["published_updates"]
        == receipt.diagnostic.scheduling_work["published_updates"]
        == 1
    )
    paid = budget.updates_completed
    core_calls = work_accounting["core_attempts"]
    with pytest.raises(ValueError, match="paused"):
        stream.run_until(2)
    assert budget.updates_completed == paid == 1 and work_accounting["core_attempts"] == core_calls
    stream.resume(token)
    assert stream._runtime is runtime and runtime.actor is actor and stream._budget is budget
    restored = runtime._inbox.capture_cursor()
    assert restored.applied[0].diagnostic == receipt.diagnostic
    assert restored.applied[0].diagnostic is not receipt.diagnostic
    stream.run_until(3)
    report = stream.finish()
    assert report["completed_updates"] == budget.updates_completed == 3
    assert report["training_examples"] == 9 and report["score_forward_calls"] == 0
    assert report["evaluation_scope"] == "training_only"
    assert [row["arrived_at"] for row in report["applied_updates"]] == [1, 2, 3]
    assert all(
        row["diagnostic"]["inference"]["accepted_steps"] == 1 for row in report["applied_updates"]
    )
    assert all(
        len(row["diagnostic"]["native_work"]) == 32 and len(row["diagnostic"]["stream_work"]) == 7
        for row in report["applied_updates"]
    )
    assert_tree(torch, source.snapshot_state(), initial)
    assert_tree(torch, actor.snapshot_state().state, initial)
    assert pickle.dumps(packets) == originals


@pytest.mark.parametrize(
    "fault", ["before_backend", "after_backend", "outer_restore", "outer_validate", "closing_input"]
)
def test_should_preserve_native_and_outer_paid_work_without_publishing_failed_candidates(
    torch, monkeypatch, fault
):
    source = learner()
    before, owner = source.snapshot_state(), source._learner
    payload = owner._native._payload
    features, targets = episode_arrays(schedule()[0])
    original = adapter.train_bounded_weighted_episode
    completed = []
    publishing = []
    original_publish = source._publish_feature

    def publish(*args, **options):
        publishing.append("publication")
        return original_publish(*args, **options)

    monkeypatch.setattr(source, "_publish_feature", publish)

    def invoke(candidate, packet, weights, **options):
        if fault == "before_backend":
            raise OSError("backend refused before numerical entry")
        result = original(candidate, packet, weights, **options)
        completed.append(result)
        if fault == "after_backend":
            raise TrainingUpdateFailed(
                "returned backend then failed", "untrusted_outer_port", {"foreign": 1}
            )
        if fault == "closing_input":
            features[0, 0] += 0.125
        return result

    monkeypatch.setattr(adapter, "train_bounded_weighted_episode", invoke)
    if fault == "outer_restore":
        restore = adapter.feature_native._restore_components

        def restoring(snapshot):
            if publishing:
                raise OSError("outer publication restore refused")
            return restore(snapshot)

        monkeypatch.setattr(adapter.feature_native, "_restore_components", restoring)
    elif fault == "outer_validate":
        state = adapter.feature_native._state

        def validating(*args, **kwargs):
            if publishing:
                raise OSError("outer publication validation refused")
            return state(*args, **kwargs)

        monkeypatch.setattr(adapter.feature_native, "_state", validating)
    with pytest.raises(ScheduledCategoricalUpdateFailed) as caught:
        source.train_batch(features, targets)
    native, outer = caught.value.work, caught.value.scheduling_work
    assert len(native) <= 32 and len(outer) == 7
    assert "foreign" not in native
    assert native["failure_work_may_be_unobserved"] is True
    assert outer["candidate_forks_completed"] == 1 and outer["published_updates"] == 0
    if fault in ("before_backend", "after_backend"):
        assert outer["publication_restores_attempted"] == 0
    else:
        assert native["published_updates"] == 1
        assert outer["publication_restores_attempted"] == 1
        assert outer["publication_restores_completed"] == (fault != "outer_restore")
        assert outer["publication_validations_attempted"] == (fault != "outer_restore")
        assert outer["publication_validations_completed"] == (fault == "closing_input")
    assert owner._native._payload is payload
    monkeypatch.undo()
    assert_tree(torch, source.snapshot_state(), before)


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
def test_should_preserve_foreign_donor_change_after_paid_candidate_and_refuse_outer_publication(
    torch, monkeypatch, method
):
    source = learner(method)
    owner = source._learner
    before = owner.snapshot_state()
    original = adapter.train_bounded_weighted_episode

    def mutate(candidate, packet, weights, **options):
        result = original(candidate, packet, weights, **options)
        with torch.no_grad():
            owner._native._head.bias_output[0, 0] += 0.125
        return result

    monkeypatch.setattr(adapter, "train_bounded_weighted_episode", mutate)
    with pytest.raises(ScheduledCategoricalUpdateFailed) as caught:
        source.train_batch(*episode_arrays(schedule()[0]))
    expected = deepcopy(before)
    expected.native_state.head_state["bias_output"][0, 0] += 0.125
    assert_tree(torch, owner.snapshot_state(), expected)
    assert source._learner is owner
    assert caught.value.work["published_updates"] == 1
    assert caught.value.scheduling_work["published_updates"] == 0
    with pytest.raises(ValueError):
        source.snapshot_state()


def test_should_stop_the_actual_inbox_after_failed_bounded_update_without_retry(monkeypatch):
    records = (*schedule(), episode("third", (1, 6), cue=4))
    source = learner(records=records)
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=3), lambda: 0.0)
    stream = stream_app.AttributionStream(
        source,
        packets_for(records),
        (),
        checkpoint_updates=(0, 1, 2, 3),
        label_delay=1,
        budget=budget,
        training_only=True,
    )
    calls = []

    def stopped(*args, **options):
        calls.append("backend")
        raise OSError("bounded backend interrupted")

    monkeypatch.setattr(adapter, "train_bounded_weighted_episode", stopped)
    with pytest.raises(ScheduledCategoricalUpdateFailed):
        stream.run_until(1)
    assert calls == ["backend"] and stream._runtime._inbox.stopped is True
    assert budget.updates_completed == 0 and stream._runtime.applied_updates == ()
    with pytest.raises((ValueError, RuntimeError)):
        stream.run_until(1)
    assert calls == ["backend"] and budget.updates_completed == 0


@pytest.mark.parametrize(
    "fault",
    [
        "native_duplicate",
        "core_mismatch",
        "native_bool",
        "native_negative",
        "stream_unknown",
        "stream_bool",
        "stream_overcommit",
        "policy_extra",
        "inference_type",
    ],
)
def test_should_validate_exact_rich_diagnostic_counter_schemas_and_core_agreement(fault):
    result = learner().train_batch(*episode_arrays(schedule()[0]))
    if fault == "native_duplicate":
        changes = dict(native_work=(*result.native_work, result.native_work[0]))
    elif fault == "core_mismatch":
        work = result.work
        work["accepted_steps"] += 1
        changes = dict(native_work=tuple(work.items()))
    elif fault == "native_bool":
        work = result.work
        work["published_updates"] = True
        changes = dict(native_work=tuple(work.items()))
    elif fault == "native_negative":
        work = result.work
        work["native_transitions_attempted"] = -1
        changes = dict(native_work=tuple(work.items()))
    elif fault in ("stream_unknown", "stream_bool", "stream_overcommit"):
        work = result.scheduling_work
        if fault == "stream_unknown":
            work["foreign"] = work.pop("published_updates")
        elif fault == "stream_bool":
            work["published_updates"] = True
        else:
            work["publication_restores_completed"] = 2
        changes = dict(stream_work=tuple(work.items()))
    elif fault == "policy_extra":
        settings = result.inference_policy
        object.__setattr__(settings, "foreign", 1)
        changes = dict(inference_policy=settings)
    else:
        inference = result.inference
        object.__setattr__(inference, "accepted_steps", True)
        changes = dict(inference=inference)
    with pytest.raises(ValueError):
        replace(result, **changes)


def test_should_own_rich_diagnostic_policy_and_work_and_preserve_asdict_fields():
    result = learner().train_batch(*episode_arrays(schedule()[0]))
    settings = stopping()
    rebuilt = replace(result, inference_policy=settings)
    object.__setattr__(settings, "gradient_tolerance", 1e20)
    assert rebuilt.inference_policy.gradient_tolerance is None
    assert rebuilt.work == result.work and rebuilt.scheduling_work == result.scheduling_work
    assert set(asdict(rebuilt)) == {
        "definition",
        "value",
        "inference_policy",
        "inference",
        "native_work",
        "stream_work",
    }


@pytest.mark.parametrize(
    "fault",
    [
        "native_publication",
        "candidate_forks",
        "publication_restores",
        "publication_validations",
        "stopping_reason",
    ],
)
def test_should_require_complete_native_and_outer_publication_in_a_success_receipt(fault):
    result = learner().train_batch(*episode_arrays(schedule()[0]))
    if fault == "native_publication":
        work = result.work
        work["published_updates"] = 0
        changes = dict(native_work=tuple(work.items()))
    elif fault == "stopping_reason":
        assert result.inference.rejected_steps == 0
        assert result.inference.used_fallback is False
        changes = dict(inference=replace(result.inference, reason="energy_increase"))
    else:
        work = result.scheduling_work
        work[fault + "_attempted"] = work[fault + "_completed"] = 0
        assert work["published_updates"] == 1
        changes = dict(stream_work=tuple(work.items()))
    with pytest.raises(ValueError):
        replace(result, **changes)


def test_should_refuse_coordinated_policy_and_seal_edits_against_the_original_choice(
    torch, work_accounting
):
    source = learner(settings=stopping(gradient_tolerance=1))
    before = source.snapshot_state()
    actual, seal = source._inference_policy, source._inference_policy_seal
    object.__setattr__(actual, "gradient_tolerance", 2)
    object.__setattr__(seal, "gradient_tolerance", 2)
    assert source._inference_policy is actual and source._inference_policy_seal is seal
    with pytest.raises(ValueError):
        source.snapshot_state()
    with pytest.raises(ValueError):
        source.train_batch(*episode_arrays(schedule()[0]))
    assert work_accounting["core_attempts"] == work_accounting["transition_attempts"] == 0
    assert actual.gradient_tolerance == seal.gradient_tolerance == 2
    assert_tree(torch, source._learner.snapshot_state(), before)


@pytest.mark.parametrize("fault", ["foreign_subclass", "corrupted_rich"])
def test_should_revalidate_cursor_receipts_and_refuse_unapproved_diagnostic_types(fault):
    records = (*schedule(), episode("third", (1, 6), cue=4))
    source = learner(records=records)
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=3), lambda: 0.0)
    stream = stream_app.AttributionStream(
        source,
        packets_for(records),
        (),
        checkpoint_updates=(0, 1, 2, 3),
        label_delay=1,
        budget=budget,
        training_only=True,
    )
    stream.run_until(1)
    cursor = stream._runtime._inbox.capture_cursor()
    receipt = cursor.applied[0]
    diagnostic: TrainingDiagnostic
    if fault == "foreign_subclass":

        class ForeignDiagnostic(TrainingDiagnostic):
            pass

        diagnostic = ForeignDiagnostic(receipt.diagnostic.definition, receipt.diagnostic.value)
    else:
        diagnostic = deepcopy(receipt.diagnostic)
        object.__setattr__(diagnostic, "value", diagnostic.value + 1.0)
    with pytest.raises(ValueError):
        replace(cursor, applied=(replace(receipt, diagnostic=diagnostic),))


@pytest.mark.parametrize("fault", ["bp", "overcap", "forged"])
def test_should_reject_unqualified_stopping_before_native_construction(monkeypatch, fault):
    settings = stopping(min_steps=3) if fault == "overcap" else stopping()
    if fault == "forged":
        object.__setattr__(settings, "min_steps", True)
    candidate = pc_policy("bp") if fault == "bp" else pc_policy("pc")
    monkeypatch.setattr(
        adapter,
        "InvariantTorchFeatureLearner",
        lambda *a, **k: pytest.fail("invalid stopping allocated native learner"),
    )
    with pytest.raises(ValueError):
        adapter.TorchTrainWeightedLearner(candidate, schedule(), inference_policy=settings)


def test_should_refuse_oversized_backend_work_without_overflowing_its_failure_receipt(
    torch, monkeypatch
):
    source = learner()
    before, owner = source.snapshot_state(), source._learner
    payload = owner._native._payload
    original = adapter.train_bounded_weighted_episode

    def malformed(candidate, packet, weights, **options):
        result = original(candidate, packet, weights, **options)
        return replace(result, _work_items=(*result._work_items, ("foreign", 1)))

    monkeypatch.setattr(adapter, "train_bounded_weighted_episode", malformed)
    with pytest.raises(ScheduledCategoricalUpdateFailed) as caught:
        source.train_batch(*episode_arrays(schedule()[0]))
    assert caught.value.work == {"failure_work_may_be_unobserved": True}
    assert "foreign" not in caught.value.work
    assert caught.value.scheduling_work["candidate_forks_completed"] == 1
    assert caught.value.scheduling_work["publication_restores_attempted"] == 0
    assert caught.value.scheduling_work["published_updates"] == 0
    assert owner._native._payload is payload
    assert_tree(torch, source.snapshot_state(), before)


@pytest.mark.parametrize(
    "fault", ["minimum_over_cap", "minimum_not_reached", "unconfigured_gradient_stop"]
)
def test_should_require_the_successful_stopping_reason_to_agree_with_its_policy(fault):
    result = learner().train_batch(*episode_arrays(schedule()[0]))
    if fault == "minimum_over_cap":
        changes = dict(inference_policy=replace(result.inference_policy, min_steps=10000))
    elif fault == "minimum_not_reached":
        assert result.inference.used_fallback is False
        changes = dict(
            inference_policy=replace(
                result.inference_policy, min_steps=result.inference.accepted_steps + 1
            )
        )
    else:
        assert result.inference_policy.gradient_tolerance is None
        changes = dict(inference=replace(result.inference, reason="gradient_tolerance"))
    with pytest.raises(ValueError):
        replace(result, **changes)
