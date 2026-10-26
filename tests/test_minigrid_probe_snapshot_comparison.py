"""Fresh complete count-state validation precedes ordinary value equality.

These literal typed snapshots and evaluator receipts are scripted boundary
fixtures. No native update, prediction, learner model or environment is executed.
"""

from copy import deepcopy
from dataclasses import replace

import pytest

from src.adapters import minigrid_probe_rollout as live
from src.adapters.invariant_action_memory_learner import (
    InvariantMemoryPolicy,
    InvariantMemorySnapshot,
)
from src.core.action_memory import ActionMemoryEntry, ActionMemorySnapshot
from src.core.fifo_action_memory import FifoActionMemoryPolicy, FifoActionMemorySnapshot
from test_minigrid_probe_rollout import Learner, ports as ports, report


def two_entry_learner():
    learner = Learner()
    native = learner.state.native_state
    assert type(native) is ActionMemorySnapshot
    second_key = list((-1.0,) * 296)
    second_key[0] = -0.8  # Canonical public object1; no nuisance wall/direction.
    entries = (
        ActionMemoryEntry((-1.0,) * 296, (72, 0, 0, 0, 0, 0, 0)),
        ActionMemoryEntry(tuple(second_key), (72, 0, 0, 0, 0, 0, 0)),
    )
    learner.state = InvariantMemorySnapshot(
        learner.state.policy, ActionMemorySnapshot(native.policy, entries, 16, 144)
    )
    return learner


def observe(observer, learner, api):
    if api == "qualified":

        def boundary(read_ports=None):
            if read_ports is not None:
                read_ports()

        observer.observe_qualified(learner, "after_probe", boundary)
    else:
        observer.observe(learner, "after_probe", lambda: None)


@pytest.mark.parametrize("api", ["legacy", "qualified"])
def test_should_accept_equal_owned_snapshots_and_read_and_validate_afresh_at_each_boundary(
    monkeypatch, ports, api
):
    learner = two_entry_learner()
    original_snapshot = learner.snapshot_state
    original_copy = live.copy_invariant_memory_snapshot
    snapshots, copies = [], []

    def snapshot():
        current = original_snapshot()
        snapshots.append(current)
        assert current is not learner.state and current == learner.state
        return current

    def copy(state):
        owned = original_copy(state)
        copies.append((state, owned))
        assert owned is not state and owned == state
        return owned

    monkeypatch.setattr(learner, "snapshot_state", snapshot)
    monkeypatch.setattr(live, "copy_invariant_memory_snapshot", copy)

    def evaluate(owner, state, options):
        for _ in range(3):
            before = len(snapshots), len(copies)
            options["check_boundary"]()
            assert (len(snapshots), len(copies)) == (before[0] + 1, before[1] + 1)
        return report()

    ports["callback"] = evaluate
    observer = live.ProbeRollouts(0)
    observe(observer, learner, api)
    observer.check()
    assert observer.records[0]["status"] == "complete"
    assert len(snapshots) >= 3 and len({id(state) for state in snapshots}) == len(snapshots)
    assert len(ports["calls"]) == 1
    assert observer.records[0]["charged_updates"] == observer.records[0]["replay_examples"] == 0


@pytest.mark.parametrize(
    "fault",
    [
        "last_valid_key",
        "last_valid_count_distribution",
        "bool_count",
        "nan_key",
        "nonfloat_key",
        "extra_fields",
        "policy_capacity",
        "encoding",
        "fifo_mismatch",
        "counter_type",
        "exposure_mismatch",
        "delegated_snapshot_alias",
    ],
)
def test_should_refuse_changed_or_malformed_current_snapshots_and_retain_paid_work(ports, fault):
    learner = two_entry_learner()
    original = deepcopy(learner.state)

    def change(owner, delegated, options):
        current = delegated if fault == "delegated_snapshot_alias" else deepcopy(owner.state)
        native = current.native_state
        final = native.entries[-1]
        if fault == "last_valid_key":
            key = list(final.features)
            key[292] = -0.8
            object.__setattr__(final, "features", tuple(key))
        elif fault in {"last_valid_count_distribution", "delegated_snapshot_alias"}:
            object.__setattr__(final, "counts", (71, 0, 0, 0, 0, 0, 1))
        elif fault == "bool_count":
            # Python equality would consider False equal to the saved zero.
            object.__setattr__(final, "counts", (72, 0, 0, 0, 0, 0, False))
        elif fault in {"nan_key", "nonfloat_key"}:
            key = list(final.features)
            key[-1] = float("nan") if fault == "nan_key" else -1
            object.__setattr__(final, "features", tuple(key))
        elif fault == "extra_fields":
            # Dataclass equality alone omits unexpected record attributes.
            object.__setattr__(native, "unexpected", None)
        elif fault == "policy_capacity":
            policy = replace(native.policy, max_entries=63)
            current = InvariantMemorySnapshot(
                InvariantMemoryPolicy(policy), replace(native, policy=policy)
            )
        elif fault == "encoding":
            object.__setattr__(current.policy, "encoding_id", "wrong_public_encoding")
        elif fault == "fifo_mismatch":
            policy = FifoActionMemoryPolicy(
                feature_dim=296, num_classes=7, max_payload_bytes=155136
            )
            object.__setattr__(
                current,
                "native_state",
                FifoActionMemorySnapshot(policy, native.entries, 16, 144, 0, 0),
            )
        elif fault == "counter_type":
            object.__setattr__(native, "completed_updates", 16.0)
        else:
            object.__setattr__(native, "training_examples", 145)
        owner.state = current
        # Return a real-shaped paid receipt first. The subsequent fresh read,
        # not this delegate's retained alias, must authorize publication.
        return report()

    ports["callback"] = change
    observer = live.ProbeRollouts(0)
    api = (
        "qualified"
        if fault in {"bool_count", "nonfloat_key", "counter_type", "delegated_snapshot_alias"}
        else "legacy"
    )
    with pytest.raises(RuntimeError):
        observe(observer, learner, api)
    assert len(ports["calls"]) == 1
    saved = observer.records[0]
    assert saved["status"] == "failed" and saved["reason"]
    assert saved["evaluator_admitted"] is True
    assert saved["report"]["work"]["live_prediction_calls_returned"] == 4
    assert saved["report"]["work"]["internal_projection_array_bytes_returned"] == 9472
    assert saved["charged_updates"] == saved["replay_examples"] == 0
    assert saved["budget_binding_preserved"] is True
    assert saved["timing_complete"] is False
    assert (
        original.native_state.completed_updates == 16
        and original.native_state.training_examples == 144
    )
