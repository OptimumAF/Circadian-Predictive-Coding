"""Joint LS rehearsal through ordinary delayed stream; bounded fixture work."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from scripts import context_head_probe as head
from src.adapters.context_linear_fit_learner import ContextLinearFitLearner
from src.adapters.context_linear_fit_decision import ContextLinearFitDecisionPredictor
from src.app.coupled_attribution import AttributionBatch, AttributionStream
from src.app.stream_rehearsal import StreamRehearsal, RecentStreamReplayPolicy
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.context_linear_state import ContextLinearFitPolicy
from src.core.context_joint_replay import JointContextReplayPolicy, compose_joint_packet
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    EpisodeReplaySchema,
)
from src.shared.process_memory import ProcessRssSampler
from src.adapters import linear_fit_learner as linear
from src.app.ticket_workflow import run_workflow_episode
from src.core.ticket_workflow import ACTION_IDS, WorkflowCapabilities, WorkflowState

SCHEMA = EpisodeReplaySchema(9, 3, 6)


@pytest.fixture(scope="module")
def inputs():
    spec = head.DecisionFeatureSpec(head.CONTEXT_FEATURES, head.CATALOG)
    return spec, head.build_head_cohorts(spec)


@pytest.fixture(scope="module")
def legacy_train(inputs):
    """Ordinary training reads original TRAIN teachers, never probe cohorts."""
    result = []
    for version in ("v1", "v2"):
        cases = head.training_cases(policy_version=version, include_policy_context=True)
        start = 3 * head.VOCABULARY.index("legacy")
        selected = cases[start : start + 3]
        assert all(c.role == "train" for c in selected)
        result.append(
            head.build_decision_training_batch(
                selected, inputs[0], f"legacy-train-{version}", allow_subsets=True
            )
        )
    assert all(b.role == "train" for b in result)
    return tuple(result)


def packet(x, y, role, identifier):
    observations = [tuple(float(v) for v in row) for row in x]
    transitions = tuple(
        DemonstrationTransition(
            i,
            observations[i],
            int(y[i, 0]),
            float(i == 2),
            observations[i + 1] if i < 2 else (0.0,) * 9,
            i == 2,
            6,
        )
        for i in range(3)
    )
    episode = DemonstrationEpisode(identifier, identifier, role, transitions, 6)
    return AttributionBatch(
        role,
        identifier,
        tuple(f"{identifier}:step:{i}" for i in range(3)),
        x.copy(),
        y.copy(),
        num_classes=6,
        episode=episode,
    )


def old_packets(inputs):
    return tuple(
        packet(b.features[3 * j : 3 * j + 3], b.targets[3 * j : 3 * j + 3], "train", f"old-{i}-{j}")
        for i, b in enumerate(inputs[1][2])
        for j in range(2)
    )


def owner():
    return ContextLinearFitLearner(ContextLinearFitPolicy(9, 6, (4, 5)))


def joint_policy(control="joint"):
    return JointContextReplayPolicy(9, 6, (4, 5), control)


def sessions(wakes, replay):
    budgets = (
        ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=wakes, max_wall_seconds=30, max_process_rss_bytes=512 * 1024**2
            ),
            lambda: 0.0,
        ),
        ToyBudgetSession(
            ToyExecutionBudget(
                max_replay_examples=replay, max_wall_seconds=30, max_process_rss_bytes=512 * 1024**2
            ),
            lambda: 0.0,
        ),
    )
    sampler = ProcessRssSampler()
    sampler.sample()
    for budget in budgets:
        budget.attach_memory(sampler)
    return budgets


def make_helper(source, budget, replay, total, *, control="joint", bank=(), boundary=None):
    return StreamRehearsal(
        source,
        budget,
        RecentStreamReplayPolicy(1, 12, ReplayRetentionBudget(30, 65536)),
        replay_budget=replay,
        total_updates=total,
        input_dim=9,
        episode_schema=SCHEMA,
        fixed_training_episodes=bank,
        wake_role="probe_train" if bank else "train",
        joint_context=joint_policy(control),
        check_boundary=boundary,
    )


@pytest.mark.parametrize("control", ["joint", "current_only"])
def test_should_compose_owned_same_row_count_and_refuse_cross_context(inputs, control):
    current = inputs[1][0][0]
    old = inputs[1][2]
    x, y = compose_joint_packet(
        joint_policy(control),
        (current.features, current.targets),
        ((old[0].features, old[0].targets),),
    )
    assert x.shape == (9, 9) and y.shape == (9, 1)
    assert not np.shares_memory(x, current.features) and not np.shares_memory(y, old[0].targets)
    if control == "current_only":
        assert np.array_equal(x, np.tile(current.features, (3, 1)))
    else:
        assert np.array_equal(x[3:], old[0].features)
    with pytest.raises(ValueError, match="contexts"):
        compose_joint_packet(
            joint_policy(control),
            (current.features, current.targets),
            ((old[1].features, old[1].targets),),
        )


@pytest.mark.parametrize("control", ["joint", "current_only"])
def test_should_execute_arrived_train_old_new_return_with_exact_consolidation_work(
    inputs, legacy_train, control
):
    old = old_packets(inputs)
    legacy = tuple(
        packet(b.features, b.targets, "train", f"new-{i}") for i, b in enumerate(legacy_train)
    )
    returning = tuple(
        packet(b.features, b.targets, "train", f"return-{i}") for i, b in enumerate(old)
    )
    source = owner()
    before = source.snapshot_state()
    wake, replay = sessions(10, 66)
    helper = make_helper(source, wake, replay, 10, control=control)
    stream = AttributionStream(
        source,
        (*old, *legacy, *returning),
        (),
        checkpoint_updates=(0, 10),
        label_delay=1,
        budget=wake,
        rehearsal=helper,
        training_only=True,
    )
    stream.run_until(10)
    result = stream.finish()
    state = stream._runtime.candidate_snapshot().state
    assert source.snapshot_state() == stream._runtime.actor.snapshot_state().state == before
    assert result["completed_updates"] == wake.updates_completed == 10
    assert state.completed_updates == 18 and state.training_examples == 96
    assert replay.replay_examples_completed == 66
    assert (
        helper.report["attempted_updates"]
        == helper.report["returned_updates"]
        == helper.report["committed_updates"]
        == 8
    )
    assert len(stream.applied_consolidations) == 8
    assert [r["status"] for r in helper.report["events"]].count("skipped") == 2
    assert all(r.observed_at + 1 == r.arrived_at <= r.applied_at for r in stream.applied_updates)
    for event in helper.report["events"]:
        if event["status"] == "complete":
            detail = event["joint_context"]
            assert event["charged_exposures"] == detail["total_fit_rows"]
            assert detail["current_role"] == detail["retained_role"] == "train"
            assert detail["retained_train_rows_used"] == (
                detail["retained_rows_available"] if control == "joint" else 0
            )
    candidate = source.fork()
    candidate.restore_state(state)
    if control == "joint":
        for b in (*old, *legacy):
            assert np.array_equal(np.argmax(candidate.predict(b.features), axis=1), b.targets[:, 0])
    # No retained probe/final records are admitted anywhere in this ordinary bank.
    assert all(
        ep["role"] == "train" for e in helper.report["events"] for ep in e["selected_episodes"]
    )
    if control == "joint":
        reader = ContextLinearFitDecisionPredictor(inputs[0], state)
        tasks = (
            ("v1", "invoice"),
            ("v1", "password"),
            ("v2", "invoice"),
            ("v2", "parcel"),
            ("v1", "legacy"),
            ("v2", "legacy"),
        )
        for version, concept in tasks:
            episode = run_workflow_episode(
                WorkflowState(f"joint-test-{version}-{concept}", concept, policy_version=version),
                reader,
                capabilities=WorkflowCapabilities(ACTION_IDS),
                max_steps=6,
                include_policy_context=True,
            )
            assert episode["complete"] and episode["success"]
            assert episode["training_updates"] == 0
        assert reader.work.decision_returns == 18


@pytest.mark.parametrize("control", ["joint", "current_only"])
def test_should_keep_probe_inputs_out_of_fixed_train_bank(inputs, control):
    old = old_packets(inputs)
    source = owner()
    for b in inputs[1][2]:
        source.train_batch(b.features, b.targets)
    before = source.snapshot_state()
    wake, replay = sessions(1, 9)
    bank = tuple(b.episode for b in old)
    helper = make_helper(source, wake, replay, 1, control=control, bank=bank)
    raw = inputs[1][0][0]
    current = packet(raw.features, raw.targets, "probe_train", "fresh-legacy")
    stream = AttributionStream(
        source,
        (current,),
        (),
        checkpoint_updates=(0, 1),
        label_delay=1,
        budget=wake,
        rehearsal=helper,
        training_only=True,
        training_role="probe_train",
    )
    stream.run_until(1)
    stream.finish()
    event = helper.report["events"][0]
    assert source.snapshot_state() == before
    assert replay.replay_examples_completed == event["charged_exposures"] == 9
    assert (
        event["attempted_updates"] == event["returned_updates"] == event["committed_updates"] == 1
    )
    assert event["joint_context"]["current_role"] == "probe_train"
    assert event["joint_context"]["retained_role"] == "train"
    assert helper.report["fixed_bank"]["native_transition_count"] == 12
    assert len(event["selected_episodes"]) == 2 and all(
        e["role"] == "train" for e in event["selected_episodes"]
    )
    assert stream._runtime.candidate_snapshot().state.training_examples == 24


@pytest.mark.parametrize("fault", ["lost", "pending"])
def test_should_keep_paid_failed_joint_work_without_partial_consolidation(
    inputs, fault, monkeypatch
):
    source = owner()
    for b in inputs[1][2]:
        source.train_batch(b.features, b.targets)
    before = source.snapshot_state()
    wake, replay = sessions(1, 9)
    helper = make_helper(
        source, wake, replay, 1, bank=tuple(b.episode for b in old_packets(inputs))
    )
    b = inputs[1][0][0]
    current = packet(b.features, b.targets, "probe_train", "failed-joint")
    fit, calls = linear.fit_categorical_linear, []

    def failed(*args):
        result = fit(*args)
        calls.append(len(args[0]))
        if len(args[0]) == 9:
            if fault == "lost":
                raise RuntimeError("lost joint solve")
            helper._events[0]["joint_context"]["retained_role"] = "final_test"
        return result

    monkeypatch.setattr(linear, "fit_categorical_linear", failed)
    stream = AttributionStream(
        source,
        (current,),
        (),
        checkpoint_updates=(0, 1),
        label_delay=1,
        budget=wake,
        rehearsal=helper,
        training_only=True,
        training_role="probe_train",
    )
    with pytest.raises(Exception, match="lost joint|pending joint"):
        stream.run_until(1)
    assert calls == [3, 9] and wake.updates_completed == 1 and replay.replay_examples_completed == 9
    assert source.snapshot_state() == before
    assert stream._runtime._candidate.snapshot_state().training_examples == 15
    row = helper.report["events"][0]
    assert (
        row["status"] == "failed"
        and row["attempted_updates"] == 1
        and row["committed_updates"] == 0
    )
    assert row["joint_context"]["retained_role"] == "train"
    assert row["returned_updates"] == (0 if fault == "lost" else 1)
    if fault == "lost":
        assert row["native_failure_work"]["work"]["fit_rows_attempted"] == 9
    with pytest.raises(ValueError):
        stream.run_until(1)
    assert calls == [3, 9]


@pytest.mark.parametrize("fault", ["limits", "counter"])
def test_should_refuse_changed_replay_authority_before_additional_fit(inputs, fault, monkeypatch):
    source = owner()
    wake, replay = sessions(1, 9)
    helper = make_helper(
        source, wake, replay, 1, bank=tuple(b.episode for b in old_packets(inputs))
    )
    if fault == "limits":
        object.__setattr__(replay.budget, "max_replay_examples", 99)
    else:
        replay.replay_examples_completed = 1
    monkeypatch.setattr(linear, "fit_categorical_linear", lambda *a: pytest.fail("fit"))
    with pytest.raises(ValueError, match="budget changed"):
        helper.episode_schema
    assert helper._stopped and wake.updates_completed == 0
    assert (
        (replay.budget.max_replay_examples == 99)
        if fault == "limits"
        else replay.replay_examples_completed == 1
    )


@pytest.mark.parametrize("fault", ["value", "replace"])
def test_should_refuse_changed_preceding_paid_joint_event_before_next_wake(inputs, fault):
    source = owner()
    old = old_packets(inputs)
    b = inputs[1][0][0]
    current = packet(b.features, b.targets, "train", "later")
    wake, replay = sessions(3, 15)
    helper = make_helper(source, wake, replay, 3)
    stream = AttributionStream(
        source,
        (*old[:2], current),
        (),
        checkpoint_updates=(0, 3),
        label_delay=1,
        budget=wake,
        rehearsal=helper,
        training_only=True,
    )
    stream.run_until(2)
    original = helper._events[1]
    pin = deepcopy(original)
    if fault == "value":
        original["returned_updates"] = 99
    else:
        helper._events[1] = deepcopy(original)
    with pytest.raises(ValueError, match="preceding paid"):
        stream.run_until(3)
    assert helper._events[1] is original and original == pin
    assert wake.updates_completed == 2 and replay.replay_examples_completed == 6


def test_should_refuse_budget_renewal_inside_boundary_before_joint_fit(inputs, monkeypatch):
    source = owner()
    wake, replay = sessions(1, 9)
    holder: list[StreamRehearsal] = []
    calls = []

    def boundary():
        if holder and holder[0]._events[0].get("joint_context"):
            object.__setattr__(replay.budget, "max_replay_examples", 99)

    helper = make_helper(
        source,
        wake,
        replay,
        1,
        bank=tuple(b.episode for b in old_packets(inputs)),
        boundary=boundary,
    )
    holder.append(helper)
    b = inputs[1][0][0]
    current = packet(b.features, b.targets, "probe_train", "budget-refusal")
    fit = linear.fit_categorical_linear

    def observed(*args):
        calls.append(len(args[0]))
        return fit(*args)

    monkeypatch.setattr(linear, "fit_categorical_linear", observed)
    stream = AttributionStream(
        source,
        (current,),
        (),
        checkpoint_updates=(0, 1),
        label_delay=1,
        budget=wake,
        rehearsal=helper,
        training_only=True,
        training_role="probe_train",
    )
    with pytest.raises(ValueError, match="budget changed"):
        stream.run_until(1)
    assert calls == [3] and wake.updates_completed == 1 and replay.replay_examples_completed == 0
    assert replay.budget.max_replay_examples == 99


def test_should_refuse_probe_bank_and_changed_joint_policy_before_native_work(inputs, monkeypatch):
    source = owner()
    bank = tuple(b.episode for b in old_packets(inputs))
    wake, replay = sessions(1, 9)
    with pytest.raises(ValueError):
        make_helper(source, wake, replay, 1, bank=(replace(bank[0], role="probe_train"), *bank[1:]))
    helper = make_helper(source, wake, replay, 1, bank=bank)
    helper._joint_context = replace(helper._joint_context)
    monkeypatch.setattr(linear, "fit_categorical_linear", lambda *a: pytest.fail("fit"))
    with pytest.raises(ValueError, match="policy changed"):
        helper.episode_schema


@pytest.mark.parametrize("control", ["joint", "current_only"])
def test_should_refuse_conflicting_labels_in_both_declared_controls(inputs, control):
    b = inputs[1][0][0]
    targets = b.targets.copy()
    targets[0, 0] = (targets[0, 0] + 1) % 6
    with pytest.raises(ValueError, match="labels conflict"):
        compose_joint_packet(
            joint_policy(control), (b.features, b.targets), ((b.features.copy(), targets),)
        )


def test_should_keep_joint_pins_and_paid_authority_through_owned_continuation(inputs):
    source = owner()
    wake, replay = sessions(3, 15)
    helper = make_helper(source, wake, replay, 3)
    b = inputs[1][0][0]
    current = packet(b.features, b.targets, "train", "after-capture")
    stream = AttributionStream(
        source,
        (*old_packets(inputs)[:2], current),
        (),
        checkpoint_updates=(0, 3),
        label_delay=1,
        budget=wake,
        rehearsal=helper,
        training_only=True,
    )
    stream.run_until(2)
    captured = helper._capture_continuation()
    prepared = helper._prepare_continuation(captured)
    helper._commit_continuation(prepared)
    assert helper.report["committed_updates"] == 1
    assert wake.updates_completed == 2 and replay.replay_examples_completed == 6
    object.__setattr__(helper._joint_context, "control", "current_only")
    with pytest.raises(ValueError, match="policy changed"):
        helper._capture_continuation()


@pytest.mark.parametrize("fault", ["truncate", "append", "replace"])
def test_should_restore_original_event_container_before_native_admission(fault, monkeypatch):
    source = owner()
    wake, replay = sessions(2, 12)
    helper = make_helper(source, wake, replay, 2)
    container, rows = helper._events, tuple(helper._events)
    if fault == "truncate":
        helper._events.clear()
    elif fault == "append":
        helper._events.append(deepcopy(helper._events[-1]))
    else:
        helper._events = deepcopy(helper._events)
    monkeypatch.setattr(linear, "fit_categorical_linear", lambda *a: pytest.fail("fit"))
    with pytest.raises(ValueError, match="evidence changed"):
        helper.episode_schema
    assert helper._events is container and len(container) == 2
    assert all(a is b for a, b in zip(container, rows, strict=True))
    assert helper._stopped and wake.updates_completed == replay.replay_examples_completed == 0
