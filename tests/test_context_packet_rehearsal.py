"""Ordinary paid packets retain authentic whole episodes, never split wakes."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from scripts.context_catalog_replay import compile_context_packet
from src.app.coupled_attribution import AttributionStream, _validate_batch, _validated_batches
from src.core.ticket_workflow import ACTION_IDS
from src.core.context_joint_replay import compose_joint_packet, JointContextReplayPolicy
from test_context_catalog_replay import prototype, spec
from test_context_linear_joint_rehearsal import owner, sessions, make_helper


@pytest.fixture(scope="module")
def packets():
    old = prototype()
    template = prototype(("legacy", "password"))
    targets = template.targets[:3].copy()
    targets[1, 0] = ACTION_IDS.index("retire")
    new = replace(
        template,
        context="new-v1",
        features=template.features[:3].copy(),
        targets=targets,
        row_ids=tuple(f"new-{i}" for i in range(3)),
    )
    returning = replace(old, row_ids=tuple(f"return-{i}" for i in range(6)))
    return tuple(compile_context_packet(spec(), b) for b in (old, new, returning))


def stream_for(packets, *, replay_rows=18):
    source = owner()
    wake, replay = sessions(3, replay_rows)
    helper = make_helper(source, wake, replay, 3)
    stream = AttributionStream(
        source,
        packets,
        (),
        checkpoint_updates=(0, 3),
        label_delay=1,
        budget=wake,
        rehearsal=helper,
        training_only=True,
    )
    return source, wake, replay, helper, stream


def test_should_copy_complete_episodes_without_changing_original_row_identity(packets):
    training, heldout = _validated_batches(packets, (), training_only=True)
    assert heldout == ()
    assert [len(b.packet_episodes) for b in training] == [2, 1, 2]
    for actual, original in zip(training, packets, strict=True):
        assert (
            actual.row_ids == original.row_ids
            and actual.packet_episodes == original.packet_episodes
        )
        assert actual.packet_episodes is not original.packet_episodes
        assert not np.shares_memory(actual.features, original.features)
        assert actual.episode is None
        assert all(e.role == "train" and len(e.transitions) == 3 for e in actual.packet_episodes)


@pytest.mark.parametrize(
    "fault",
    ["order", "labels", "context", "duplicate", "list", "single", "partial", "probe", "final"],
)
def test_should_refuse_mismatched_packet_metadata_before_runtime_or_retention(packets, fault):
    batch = deepcopy(packets[0])
    if fault == "order":
        batch = replace(batch, packet_episodes=batch.packet_episodes[::-1])
    elif fault == "labels":
        batch.targets[1, 0] = ACTION_IDS.index("retire")
    elif fault == "context":
        batch = replace(
            batch,
            packet_episodes=(
                replace(batch.packet_episodes[0], context="wrong"),
                batch.packet_episodes[1],
            ),
        )
    elif fault == "duplicate":
        batch = replace(batch, packet_episodes=(batch.packet_episodes[0],) * 2)
    elif fault == "list":
        batch = replace(batch, packet_episodes=list(batch.packet_episodes))
    elif fault == "single":
        batch = replace(batch, episode=batch.packet_episodes[0])
    elif fault == "partial":
        batch = replace(batch, packet_episodes=batch.packet_episodes[:1])
    else:
        batch = replace(batch, role="probe_train" if fault == "probe" else "final_test")
    with pytest.raises(ValueError):
        _validate_batch(batch, batch.role, set())


@pytest.mark.parametrize("fault", ["labels", "order", "role", "schema", "foreign_metadata"])
def test_should_compile_only_existing_authentic_ordinary_train_packets(fault):
    batch, schema = prototype(), spec()
    if fault == "labels":
        batch.targets[1, 0] = ACTION_IDS.index("retire")
    elif fault == "order":
        batch = replace(
            batch, features=batch.features[::-1].copy(), targets=batch.targets[::-1].copy()
        )
    elif fault == "role":
        batch = replace(batch, role="probe_train")
    elif fault == "schema":
        schema = object()
    else:
        batch = replace(batch, packet_episodes=(object(),))
    with pytest.raises(ValueError):
        compile_context_packet(schema, batch)


def test_should_require_joint_rehearsal_before_admitting_packet_stream(packets):
    source = owner()
    wake, _ = sessions(3, 18)
    with pytest.raises(ValueError, match="joint rehearsal"):
        AttributionStream(
            source,
            packets,
            (),
            checkpoint_updates=(0, 3),
            label_delay=1,
            budget=wake,
            training_only=True,
        )
    assert wake.updates_completed == 0 and source.snapshot_state().completed_updates == 0


def test_should_preserve_complete_packet_exposure_requirement_for_current_only(packets):
    old, new, _ = packets
    with pytest.raises(ValueError, match="complete-packet row exposure"):
        compose_joint_packet(
            JointContextReplayPolicy(9, 6, (4, 5), "current_only"),
            (old.features, old.targets),
            ((new.features, new.targets),),
        )


@pytest.mark.parametrize("pause", [False, True])
def test_should_execute_ordinary_packets_with_exact_work_and_owned_continuation(packets, pause):
    source, wake, replay, helper, stream = stream_for(packets)
    before = source.snapshot_state()
    stream.run_until(2)
    assert (wake.updates_completed, replay.replay_examples_completed) == (2, 9)
    if pause:
        token = stream.pause()
        stream.resume(token)
        with pytest.raises(ValueError, match="original unconsumed"):
            stream.resume(token)
    stream.run_until(3)
    state = stream.candidate_snapshot()
    result = stream.finish()
    assert (wake.updates_completed, replay.replay_examples_completed) == (3, 18)
    assert result["completed_updates"] == 3
    assert (state.completed_updates, state.training_examples) == (5, 33)
    assert source.snapshot_state() == stream._runtime.actor.snapshot_state().state == before
    assert len(stream.applied_consolidations) == 2 and helper.retention.example_count <= 30
    events = helper.report["events"]
    assert [e["status"] for e in events] == ["skipped", "complete", "complete"]
    assert [e["charged_exposures"] for e in events] == [0, 9, 9]
    assert [e["joint_context"]["retained_rows_available"] for e in events[1:]] == [6, 3]
    candidate = source.fork()
    candidate.restore_state(state)
    for b in packets[:2]:
        assert np.array_equal(np.argmax(candidate.predict(b.features), axis=1), b.targets[:, 0])


def test_should_keep_completed_wake_cost_when_joint_exposure_budget_refuses(packets):
    source, wake, replay, helper, stream = stream_for(packets, replay_rows=8)
    with pytest.raises(Exception, match="replay"):
        stream.run_until(2)
    assert wake.updates_completed == 2 and replay.replay_examples_completed == 0
    assert helper.report["stopped"] and helper.report["committed_updates"] == 0
    assert source.snapshot_state().completed_updates == 0
    assert stream._runtime.candidate_snapshot().state.completed_updates == 2


def test_should_refuse_changed_original_packet_before_another_paid_wake(packets):
    _, wake, replay, _, stream = stream_for(packets)
    stream.run_until(1)
    original = stream._training[0]
    stream._training = (
        replace(original, packet_episodes=original.packet_episodes[::-1]),
        *stream._training[1:],
    )
    with pytest.raises(ValueError, match="original packet episode"):
        stream.run_until(2)
    assert wake.updates_completed == 1 and replay.replay_examples_completed == 0
