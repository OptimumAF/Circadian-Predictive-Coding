"""Warm reserved-task schedules through the existing isolated fresh-task probe."""

from copy import deepcopy
from dataclasses import replace
import pickle
import random

import numpy as np
import pytest

from src.adapters import torch_pc_inference as backend
from src.adapters import torch_pc_weighted as fixed
from src.adapters import torch_train_weighted_learner as adapter
from src.adapters.invariant_torch_feature_learner import InvariantTorchFeatureLearner
from src.app.coupled_attribution import AttributionBatch
from src.app.fresh_task_probe import run_fresh_task_probe
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.categorical_training_diagnostic import ScheduledCategoricalUpdateFailed
from src.core import categorical_pc_inference as core
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.demonstration_class_weights import episode_class_weights, probe_episode_class_weights
from src.core.demonstration_episode import DemonstrationEpisode, episode_arrays
from src.core.learner_ports import TrainingUpdateFailed
from src.core.resnet50_variants import CircadianPredictiveCodingHead, PredictiveCodingHead
from test_torch_feature_learner import assert_tree, tensors
from test_torch_pc_weighted import ONES
from test_weighted_pc_stream_adapter import episode, pc_policy, torch as torch


@pytest.fixture(autouse=True)
def work_accounting(monkeypatch, record_property):
    work = dict(
        bounded_core_attempts=0,
        bounded_core_rows=0,
        bounded_configured_latent_row_iterations=0,
        bounded_attempted_latent_row_iterations=0,
        fixed_core_attempts=0,
        fixed_configured_latent_row_iterations=0,
        transition_attempts=0,
        transition_completed=0,
        native_calls=0,
        native_configured_latent_row_iterations=0,
        prediction_calls=0,
        prediction_rows=0,
    )
    proposal = backend.bounded_class_weighted_pc_update

    def bounded(features, targets, parameters, **options):
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

    monkeypatch.setattr(backend, "bounded_class_weighted_pc_update", bounded)
    fixed_proposal = fixed.class_weighted_pc_update

    def ordinary(features, targets, parameters, **options):
        work["fixed_core_attempts"] += 1
        work["fixed_configured_latent_row_iterations"] += len(features) * options["inference_steps"]
        return fixed_proposal(features, targets, parameters, **options)

    monkeypatch.setattr(fixed, "class_weighted_pc_update", ordinary)
    predict = InvariantTorchFeatureLearner.predict

    def prediction(owner, features):
        work["prediction_calls"] += 1
        work["prediction_rows"] += len(features)
        return predict(owner, features)

    monkeypatch.setattr(InvariantTorchFeatureLearner, "predict", prediction)
    for kind in (PredictiveCodingHead, CircadianPredictiveCodingHead):
        original, original_train = kind._apply_training_gradients, kind.train_step

        def transition(head, *args, _original=original, **options):
            work["transition_attempts"] += 1
            result = _original(head, *args, **options)
            work["transition_completed"] += 1
            return result

        def train(head, features, targets, rate, steps, inference_rate, _original=original_train):
            work["native_calls"] += 1
            work["native_configured_latent_row_iterations"] += len(features) * steps
            return _original(head, features, targets, rate, steps, inference_rate)

        monkeypatch.setattr(kind, "_apply_training_gradients", transition)
        monkeypatch.setattr(kind, "train_step", train)
    yield work
    for key, value in work.items():
        record_property(key, value)


def stopping(*, early=False):
    return CoupledInferencePolicy(
        min_steps=1, gradient_tolerance=1e20 if early else None, relative_energy_tolerance=None
    )


def warm(method="pc", steps=2):
    source = InvariantTorchFeatureLearner(pc_policy(method, steps))
    fixed.train_weighted_episode(source, episode("warm-TRAIN"), ONES)
    return source, source.snapshot_state()


def reserved(index, *, role="probe_train"):
    cue = 2 if index % 2 == 0 else 4
    actions = (1, 6) if index % 2 == 0 else (2, 6)
    return replace(
        episode("F-" + str(index), actions, role=role, cue=cue), context="F" + str(index % 2)
    )


def probe_owner(state, records, *, early=False, returns=()):
    return adapter.TorchTrainWeightedLearner.from_probe_snapshot(
        state,
        records,
        inference_policy=stopping(early=early),
        return_episodes=returns,
    )


def batch(record):
    return AttributionBatch(
        record.role,
        record.context,
        tuple(record.episode_id + ":step:" + str(row.step) for row in record.transitions),
        *episode_arrays(record),
        episode=record,
        num_classes=7,
    )


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
@pytest.mark.parametrize("steps,early", [(2, False), (64, False), (64, True)])
def test_should_continue_warm_absolute_state_and_match_direct_reserved_backend(
    torch, method, steps, early
):
    donor, state = warm(method, steps)
    records = (reserved(0), reserved(1))
    source = probe_owner(state, records, early=early)
    reference = InvariantTorchFeatureLearner(state.policy)
    reference.restore_state(state)
    assert source.source_role == "probe_train"
    assert_tree(torch, source.snapshot_state(), state)
    for index, packet in enumerate(records):
        actual = source.train_batch(*episode_arrays(packet))
        expected = backend.train_bounded_probe_episode(
            reference, packet, probe_episode_class_weights(packet), policy=stopping(early=early)
        )
        assert actual.value == expected.diagnostic.value
        assert actual.inference == expected.inference and actual.work == expected.work
        assert actual.inference.accepted_steps == (1 if early else steps)
        assert actual.scheduling_work["published_updates"] == 1
        assert_tree(torch, source.snapshot_state(), reference.snapshot_state())
        assert source.completed_updates == state.completed_updates + index + 1
        assert source.training_examples == state.training_examples + 2 * (index + 1)
    assert_tree(torch, donor.snapshot_state(), state)
    with pytest.raises(ValueError, match="exhausted"):
        source.train_batch(*episode_arrays(records[-1]))


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
def test_should_own_warm_schedule_and_preserve_relative_progress_across_fork_restore_and_copy(
    torch, method
):
    donor, state = warm(method)
    records = (reserved(0), reserved(1))
    original_first = deepcopy(records[0])
    source = probe_owner(state, records)
    object.__setattr__(records[0], "role", "final_test")
    source.train_batch(*episode_arrays(original_first))
    saved = source.snapshot_state()
    fork, copied = source.fork(), deepcopy(source)
    for child in (fork, copied):
        assert (
            child.source_role == "probe_train" and child.inference_policy == source.inference_policy
        )
        child.train_batch(*episode_arrays(records[1]))
    assert_tree(torch, fork.snapshot_state(), copied.snapshot_state())
    assert_tree(torch, source.snapshot_state(), saved)
    source.restore_state(state)
    source.train_batch(*episode_arrays(original_first))
    assert_tree(torch, source.snapshot_state(), saved)
    earlier = InvariantTorchFeatureLearner(state.policy).snapshot_state()
    with pytest.raises(ValueError):
        source.restore_state(earlier)
    for left, right in zip(
        tensors(torch, source.snapshot_state()), tensors(torch, fork.snapshot_state()), strict=True
    ):
        assert left.data_ptr() != right.data_ptr()
    assert_tree(torch, donor.snapshot_state(), state)


@pytest.mark.parametrize(
    "fault",
    ["train", "probe_test", "final_test", "empty", "too_many", "return_count", "return_role"],
)
def test_should_refuse_unreleased_roles_and_invalid_phase_envelopes_before_native_allocation(
    monkeypatch, fault
):
    _, state = warm()
    records: tuple[DemonstrationEpisode, ...] = (reserved(0),)
    returns: tuple[DemonstrationEpisode, ...] = ()
    if fault in ("train", "probe_test", "final_test"):
        records = (reserved(0, role=fault),)
    elif fault == "empty":
        records = ()
    elif fault == "too_many":
        records = tuple(reserved(index) for index in range(17))
    elif fault == "return_count":
        returns = (episode("return-0"),)
    else:
        returns = tuple(reserved(index) for index in range(16))
    monkeypatch.setattr(
        adapter,
        "InvariantTorchFeatureLearner",
        lambda *a, **k: pytest.fail("invalid probe allocated a native owner"),
    )
    with pytest.raises(ValueError):
        probe_owner(state, records, returns=returns)


def test_should_keep_public_train_and_reserved_backend_admission_separate(torch, work_accounting):
    owner, before = warm()
    calls = work_accounting["bounded_core_attempts"]
    with pytest.raises(ValueError):
        backend.train_bounded_weighted_episode(owner, reserved(0), ONES, policy=stopping())
    with pytest.raises(ValueError):
        episode_class_weights(reserved(0))
    for packet in (
        episode("ordinary"),
        reserved(0, role="probe_test"),
        reserved(0, role="final_test"),
    ):
        with pytest.raises(ValueError):
            backend.train_bounded_probe_episode(owner, packet, ONES, policy=stopping())
    assert work_accounting["bounded_core_attempts"] == calls
    assert_tree(torch, owner.snapshot_state(), before)


def test_should_derive_mass_only_from_explicit_reserved_labels_without_rewriting_the_role():
    packet = episode("F-weighted", (1, 1, 6), role="probe_train", cue=2)
    before = pickle.dumps(packet)
    assert probe_episode_class_weights(packet) == (1.0, 0.75, 1.0, 1.0, 1.0, 1.0, 1.5)
    assert packet.role == "probe_train" and pickle.dumps(packet) == before
    for role in ("train", "probe_test", "final_test"):
        with pytest.raises(ValueError):
            probe_episode_class_weights(replace(packet, role=role))


def test_should_detach_the_supplied_warm_snapshot_from_the_live_probe_and_donor(torch):
    donor, state = warm()
    source = probe_owner(state, (reserved(0),))
    before = source.snapshot_state()
    for value in tensors(torch, state):
        value.zero_()
    assert_tree(torch, source.snapshot_state(), before)
    assert_tree(torch, donor.snapshot_state(), before)


@pytest.mark.parametrize("fault", ["order", "labels", "dtype", "future_return"])
def test_should_refuse_wrong_or_future_arrays_without_new_native_work(
    torch, work_accounting, fault
):
    _, state = warm()
    records = (reserved(0), reserved(1))
    returns = tuple(episode("A-" + str(index), (0, 6)) for index in range(16))
    source = probe_owner(state, records, returns=returns)
    features, targets = episode_arrays(
        records[1] if fault == "order" else returns[0] if fault == "future_return" else records[0]
    )
    if fault == "labels":
        targets[0, 0] = 3
    elif fault == "dtype":
        features = features.astype(np.float32)
    before = work_accounting["bounded_core_attempts"]
    with pytest.raises(ValueError):
        source.train_batch(features, targets)
    assert work_accounting["bounded_core_attempts"] == before
    assert_tree(torch, source.snapshot_state(), state)


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
@pytest.mark.parametrize("with_return", [False, True])
def test_should_execute_existing_isolated_probe_and_optional_old_train_return_with_rich_paid_reports(
    torch, monkeypatch, method, with_return
):
    donor, state = warm(method)
    count = 16 if with_return else 2
    records = tuple(reserved(index) for index in range(count))
    evaluation = tuple(batch(reserved(100 + index, role="probe_test")) for index in range(2))
    monitors = (
        tuple(
            batch(
                replace(
                    episode("monitor-" + str(index), (0, 6), cue=index), context="A" + str(index)
                )
            )
            for index in range(2)
        )
        if with_return
        else ()
    )
    templates = tuple(
        batch(replace(item.episode, episode_id="template-" + str(index)))
        for index, item in enumerate(monitors)
    )
    returns = (
        tuple(
            replace(templates[index % 2].episode, episode_id="return-" + str(index))
            for index in range(16)
        )
        if with_return
        else ()
    )
    source = probe_owner(state, records, early=True, returns=returns)
    # Compare full native tensor state structurally and immutable inputs by bytes.
    original_episodes = pickle.dumps((records, evaluation, monitors, templates))
    source_before = source.snapshot_state()
    rng = (torch.get_rng_state().clone(), pickle.dumps(np.random.get_state()), random.getstate())
    roles = []
    probe_port, train_port = (
        adapter.train_bounded_probe_episode,
        adapter.train_bounded_weighted_episode,
    )

    def probe_call(owner, packet, weights, **options):
        roles.append((packet.role, owner.completed_updates))
        return probe_port(owner, packet, weights, **options)

    def train_call(owner, packet, weights, **options):
        roles.append((packet.role, owner.completed_updates))
        return train_port(owner, packet, weights, **options)

    monkeypatch.setattr(adapter, "train_bounded_probe_episode", probe_call)
    monkeypatch.setattr(adapter, "train_bounded_weighted_episode", train_call)
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=count), lambda: 0.0)
    return_budget = (
        ToyBudgetSession(ToyExecutionBudget(max_training_updates=16), lambda: 0.0)
        if with_return
        else None
    )
    report = run_fresh_task_probe(
        source,
        state,
        tuple(batch(item) for item in records),
        evaluation,
        budget=budget,
        checkpoint_updates=(0, 4, 8, 16) if with_return else (0, 1, 2),
        timer=lambda: 0.0,
        decision_diagnostics=True,
        training_readouts=monitors,
        return_training=templates,
        return_budget=return_budget,
    )
    assert report["status"] == "complete", report["reason"]
    assert report["completed_updates"] == budget.updates_completed == count
    assert len(report["diagnostics"]) == count
    assert all(
        len(row["native_work"]) == 32 and len(row["stream_work"]) == 7
        for row in report["diagnostics"]
    )
    assert all(
        np.isfinite(score["ce"])
        for point in report["checkpoints"]
        for score in point["scores"].values()
    )
    if with_return:
        assert return_budget is not None
        returned = report["return_phase"]
        assert returned["status"] == "complete", returned["reason"]
        assert returned["completed_updates"] == return_budget.updates_completed == 16
        assert len(returned["diagnostics"]) == 16
        assert all(
            len(row["native_work"]) == 32 and len(row["stream_work"]) == 7
            for row in returned["diagnostics"]
        )
        assert [role for role, _ in roles] == ["probe_train"] * 16 + ["train"] * 16
        assert [updates for _, updates in roles] == list(
            range(state.completed_updates, state.completed_updates + 32)
        )
    else:
        assert [role for role, _ in roles] == ["probe_train"] * count
    assert_tree(torch, donor.snapshot_state(), state)
    assert_tree(torch, source.snapshot_state(), source_before)
    assert pickle.dumps((records, evaluation, monitors, templates)) == original_episodes
    assert torch.equal(torch.get_rng_state(), rng[0])
    assert pickle.dumps(np.random.get_state()) == rng[1] and random.getstate() == rng[2]


def test_should_keep_paid_reserved_candidate_work_and_foreign_donor_changes_on_refusal(
    torch, monkeypatch
):
    _, state = warm()
    source = probe_owner(state, (reserved(0),))
    owner = source._learner
    before = owner.snapshot_state()
    original = adapter.train_bounded_probe_episode

    def mutate(candidate, packet, weights, **options):
        result = original(candidate, packet, weights, **options)
        with torch.no_grad():
            owner._native._head.bias_output[0, 0] += 0.125
        return result

    monkeypatch.setattr(adapter, "train_bounded_probe_episode", mutate)
    with pytest.raises(ScheduledCategoricalUpdateFailed) as caught:
        source.train_batch(*episode_arrays(reserved(0)))
    assert caught.value.work["published_updates"] == 1
    assert caught.value.scheduling_work["published_updates"] == 0
    expected = deepcopy(before)
    expected.native_state.head_state["bias_output"][0, 0] += 0.125
    assert_tree(torch, owner.snapshot_state(), expected)
    with pytest.raises(ValueError):
        source.snapshot_state()


def test_should_preserve_failed_reserved_native_and_scheduling_work_in_the_actual_probe_report(
    torch, monkeypatch
):
    donor, state = warm()
    records = (reserved(0), reserved(1))
    source = probe_owner(state, records)
    evaluation = tuple(batch(reserved(100 + index, role="probe_test")) for index in range(2))
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)

    def fail(*args, **options):
        raise OSError("reserved latent energy interrupted")

    monkeypatch.setattr(core, "_energy", fail)
    report = run_fresh_task_probe(
        source,
        state,
        tuple(batch(item) for item in records),
        evaluation,
        budget=budget,
        checkpoint_updates=(0, 1, 2),
        timer=lambda: 0.0,
    )
    assert report["status"] == "failed" and report["attempted_updates"] == 1
    assert report["completed_updates"] == budget.updates_completed == 0
    paid = report["failed_native_work"]
    assert len(paid["work"]) == 32 and len(paid["scheduling_work"]) == 7
    assert paid["work"]["energy_evaluations_attempted"] == 1
    assert paid["work"]["energy_evaluations_completed"] == 0
    assert paid["work"]["published_updates"] == paid["scheduling_work"]["published_updates"] == 0
    assert paid["work"]["failure_work_may_be_unobserved"] is True
    assert_tree(torch, donor.snapshot_state(), state)
    assert_tree(torch, source.snapshot_state(), state)
