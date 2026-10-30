"""Packet rehearsal in the original serving admission, with explicit consent."""

import pytest

from scripts import context_head_live as live
from scripts import recurring_ticket_workflow_example as entry
from scripts.context_training_streams import WITHHELD_CONTEXT_RECIPES
from src.app.actor_shadow import ActorShadowRuntime
from src.app.coupled_attribution import _record_tick
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.core.adaptation_action import AdaptationAction, AdaptationEnvelope
from src.core.experience import LogicalClock
from src.core.resource_sharing import SharingLimits
from src.shared.process_memory import ProcessRssSampler
from test_context_linear_joint_rehearsal import owner, sessions, make_helper
from test_context_linear_catalog_study import literal_state
from test_context_packet_rehearsal import packets as packets


def setup(packets, *, quota=6):
    source = owner()
    wake, replay = sessions(3, 18)
    helper = make_helper(source, wake, replay, 3)
    maximum = helper.bind(source, wake, 3, 9)
    clock = LogicalClock()
    runtime = ActorShadowRuntime(
        source,
        actor_version="a0",
        candidate_version="c0",
        clock=clock,
        budget=wake,
        max_consolidations=maximum,
    )
    gate = ServingPriorityGate(SharingLimits(1, quota, 1), resource_available=lambda: True)
    shared = ResourceSharedRuntime(
        runtime,
        gate,
        action_envelope=AdaptationEnvelope(
            (AdaptationAction("wait"), AdaptationAction("wake1", 1)), 16, 3, 18
        ),
        wake_rehearsal=helper,
        rehearsal_packets=packets,
    )
    _record_tick(runtime, packets, 0, 1, replay_consent=True)
    clock.advance_to(1)
    _record_tick(runtime, packets, 1, 1, replay_consent=True)
    return source, wake, replay, helper, runtime, gate, shared, clock


@pytest.mark.parametrize(
    "fault",
    [
        "paused",
        "serving",
        "resource",
        "quota",
        "counter",
        "packet",
        "consent",
        "candidate",
        "candidate_owner",
    ],
)
def test_should_refuse_or_defer_without_work_when_original_reservation_is_unavailable(
    packets, fault
):
    _, wake, replay, helper, runtime, gate, shared, _ = setup(
        packets, quota=1 if fault == "quota" else 6
    )
    if fault == "paused":
        gate.pause()
    elif fault == "resource":
        gate._resource = lambda: False
    elif fault == "counter":
        gate._admitted += 1
    elif fault == "packet":
        shared._rehearsal_packets[0].targets[1, 0] = 0
    elif fault == "consent":
        source = runtime._inbox._experiences[("attribution", "batch-0")]
        object.__setattr__(source.permissions, "replay", False)
    elif fault == "candidate":
        runtime._candidate.restore_state(literal_state())
    elif fault == "candidate_owner":
        runtime._candidate = runtime._candidate.fork()
    if fault in ("counter", "packet", "consent", "candidate", "candidate_owner"):
        with pytest.raises(ValueError):
            shared.train_action("wake1")
    elif fault == "serving":
        with gate.serving():
            result = shared.train_action("wake1")
        assert result.training.deferred_reason == "serving_active"
    else:
        result = shared.train_action("wake1")
        assert (
            result.training.deferred_reason
            == {"paused": "paused", "resource": "resource_unavailable", "quota": "work_quota"}[
                fault
            ]
        )
    assert wake.updates_completed == replay.replay_examples_completed == helper._observed == 0
    assert len(runtime.applied_updates) == 0


def test_should_pay_two_reserved_attempts_before_work_and_preserve_whole_lease(
    packets, monkeypatch
):
    source, wake, replay, helper, runtime, gate, shared, clock = setup(packets)
    before = source.snapshot_state()
    gate.pause()
    assert shared.train_action("wake1").training.deferred_reason == "paused"
    gate.resume()
    original = type(source).train_batch
    observations = []

    def train(self, x, y):
        snapshot = gate.snapshot()
        observations.append((len(x), snapshot.training_active, snapshot.admitted_updates))
        return original(self, x, y)

    monkeypatch.setattr(type(source), "train_batch", train)
    for tick in (1, 2, 3):
        if tick > 1:
            clock.advance_to(tick)
            _record_tick(runtime, packets, tick, 1, replay_consent=True)
        result = shared.train_action("wake1")
        assert len(result.training.updates) == 1 and result.training.deferred_reason is None
    assert observations == [(6, True, 2), (3, True, 4), (9, True, 4), (6, True, 6), (9, True, 6)]
    progress = shared.adaptation_snapshot().progress
    assert (progress.native_attempts, progress.native_completions, progress.wake_admissions) == (
        5,
        5,
        3,
    )
    assert (
        progress.replay_examples_charged,
        progress.replay_native_completions,
        progress.replay_commits,
    ) == (18, 2, 2)
    assert gate.snapshot().admitted_updates == 6 and not gate.snapshot().training_active
    assert runtime.actor.snapshot_state().state == source.snapshot_state() == before
    assert (wake.updates_completed, replay.replay_examples_completed, helper._observed) == (
        3,
        18,
        3,
    )
    assert runtime.candidate_snapshot().state.completed_updates == 5
    receipt = runtime._inbox._applied[("attribution", "batch-0")]
    object.__setattr__(receipt, "applied_at", receipt.applied_at + 1)
    with pytest.raises(ValueError, match="preceding paid"):
        shared.train_action("wake1")
    assert wake.updates_completed == 3 and replay.replay_examples_completed == 18
    assert gate.snapshot().admitted_updates == 6


def test_should_preserve_paid_wakes_reservations_and_joint_exposure_after_native_refusal(
    packets, monkeypatch
):
    source, wake, replay, helper, runtime, gate, shared, clock = setup(packets)
    original = type(source).train_batch

    def train(self, x, y):
        if len(x) == 9:
            raise RuntimeError("fixture joint fit refused before numerical call")
        return original(self, x, y)

    monkeypatch.setattr(type(source), "train_batch", train)
    shared.train_action("wake1")
    clock.advance_to(2)
    _record_tick(runtime, packets, 2, 1, replay_consent=True)
    with pytest.raises(RuntimeError, match="fixture joint"):
        shared.train_action("wake1")
    assert wake.updates_completed == 2 and replay.replay_examples_completed == 9
    assert gate.snapshot().admitted_updates == 4 and not gate.snapshot().training_active
    assert helper.report["attempted_updates"] == 1 and helper.report["returned_updates"] == 0
    assert runtime.candidate_snapshot().state.completed_updates == 2
    with pytest.raises(ValueError, match="stopped"):
        shared.train_action("wake1")
    assert wake.updates_completed == 2 and replay.replay_examples_completed == 9


@pytest.mark.parametrize(
    "change",
    [
        dict(_retained_joint=1),
        dict(_linear_fit=False),
        dict(reference_mode="drift"),
        dict(candidate_feedback=True),
        dict(publish_after_learning=True),
    ],
)
def test_should_reject_unsupported_owned_modes_before_source_read(change, monkeypatch):
    monkeypatch.setattr(live, "_source_bytes", lambda *a: pytest.fail("source opened"))
    kwargs = dict(
        _retained_joint=True,
        _linear_fit=True,
        _owned_source=(object(), lambda: None),
        stream_recipe=WITHHELD_CONTEXT_RECIPES[0],
    )
    kwargs.update(change)
    with pytest.raises(ValueError):
        live.run_case(None, **kwargs)


def test_should_execute_original_private_serving_on_literal_source_with_retained_joint_packets():
    state = literal_state()
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        report = live.run_case(
            None,
            _linear_fit=True,
            _owned_source=(state, lambda: None),
            _memory=memory,
            _retained_joint=True,
            stream_recipe=WITHHELD_CONTEXT_RECIPES[0],
        )
    assert report["complete"], report.get("error")
    assert report["source_unchanged"] and report["actor_unchanged"]
    assert report["training_complete"] and report["budget"]["updates"] == 6
    assert report["training_work"] == dict(
        attempted_updates=11, attempted_rows=72, returned_updates=11
    )
    assert report["replay_examples"] == 45 and report["joint_rehearsal"]["committed_updates"] == 5
    assert (
        len(report["applied_updates"]) == 6 and report["candidate_state"]["completed_updates"] == 23
    )
    assert report["final_examples"] == report["probe_updates"] == 0
    assert report["candidate_return_readout"]["counts"][0] >= 1
    assert report["candidate_return_readout"]["counts"][1] >= 2


def test_should_route_public_workflow_mode_without_touching_source(monkeypatch):
    monkeypatch.setattr(
        entry,
        "prepare_training",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("fixture reached training")),
    )
    report = entry.run_example(
        method="linear_fit",
        context_return=True,
        separate_context_heads=True,
        paired_serving="retained_joint",
    )
    assert report["complete"] is False
    assert report["error"] == "RuntimeError: fixture reached training"
