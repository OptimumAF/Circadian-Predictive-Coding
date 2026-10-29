"""Existing mixed TRAIN builder/stream with scripted learner and closed task ports."""

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts import run_bp_ticket_serving as demo
from scripts import ticket_catalog_control as control
from scripts.ticket_controller_streams import TicketStreamRecipe
from src.core.catalog_contextual_controller import CatalogEstimate
from src.core.ticket_workflow import ACTION_IDS
from test_ticket_catalog_control import FakeBP, batches, catalog, fitted, request
from test_ticket_catalog_control import driver as driver


def policy(action="replay_wake", n=8):
    p = fitted(n)
    return replace(
        p,
        catalog=catalog(n, replay_rows=3 * n),
        estimates=tuple(
            CatalogEstimate(c, a, float(a == action))
            for c in ("clean", "mistakes")
            for a in p.catalog.arms
        ),
        choices=(action, action),
    )


@pytest.fixture
def mixed(driver, monkeypatch):
    monkeypatch.setattr(demo.context, "BATCHES_PER_PHASE", 8)
    packets = list(batches(8))
    for i in range(8, 16):
        targets = packets[i].targets.copy()
        targets[1, 0] = ACTION_IDS.index("inspect")
        packets[i] = replace(packets[i], targets=targets)
    monkeypatch.setattr(demo.context, "_teacher_batches", lambda spec: (None, None, tuple(packets)))
    for name in ("_candidate_tasks", "run_workflow_episode", "_sealed_cases", "_publish_candidate"):
        monkeypatch.setattr(
            demo, name, lambda *a, **kw: pytest.fail("task/evaluation port opened during TRAIN")
        )
    return driver


@pytest.mark.parametrize("delay,expected_wakes", [(1, 8), (2, 7)])
@pytest.mark.parametrize("action", ["wait", "wake1", "replay_wake"])
def test_should_apply_frozen_catalog_to_actual_mixed_packets_without_readout(
    mixed, delay, expected_wakes, action
):
    recipe = TicketStreamRecipe("literal-recurring", ("new_contexts", "old_return") * 4, delay)
    r = demo.run_case(
        mixed.path, catalog_policy=policy(action), stream_recipe=recipe, deferred_evaluation=True
    )
    assert r["complete"], r.get("error")
    wakes = 0 if action == "wait" else expected_wakes
    replays = wakes if action == "replay_wake" else 0
    assert r["definition"] == "deferred_catalog_stream_v1"
    assert r["episodes"] == [] and "candidate_episodes" not in r and "reference_outcomes" not in r
    assert r["deferred_actor_unchanged"] and r["deferred_promotions"] == 0
    assert r["catalog_actor_unchanged"] and r["catalog_promotions"] == 0
    assert r["stream"]["opportunities"] == 9
    assert r["stream"]["arrived_labels"] == (8 if delay == 1 else 7)
    assert r["stream"]["pending_unapplied"] == 8 - wakes
    assert r["stream"]["unarrived_labels"] == delay - 1
    assert r["controller_work"]["prediction_records_remaining"] == delay - 1
    assert r["catalog_control"]["stream"]["natural_horizon"] == 8 + delay
    assert r["catalog_control"]["stream"]["observed_horizon"] == 9
    assert r["training_stream"]["row_counts"] == [12] * 8
    assert r["training_work"]["attempted_updates"] == wakes + replays
    assert r["training_work"]["attempted_rows"] == 12 * wakes + 3 * replays
    assert r["replay_examples"] == 3 * replays
    assert r["candidate_checkpoint"]["state"]["training_examples"] == 576 + 12 * wakes + 3 * replays
    assert r["catalog_control"]["native_history"]["wake_training_rows"] == 12 * wakes
    assert r["budget"]["limits"]["max_training_updates"] == 8
    assert r["action_snapshot"]["progress"]["decisions_attempted"] == 9
    for event in r["controller_events"]:
        fb = event["observation"]["feedback"]
        if fb is not None:
            assert fb["role"] == "train" and fb["examples"] == 12
            assert fb["arrived_at"] - fb["observed_at"] == delay
            assert fb["arrived_at"] <= event["observation"]["tick"]
    assert mixed.readouts == [] and all(m.snapshot_state() == mixed.initial for m in mixed.made)
    with pytest.raises(ValueError, match="deferred"):
        control.case_work(r)
    with pytest.raises(ValueError, match="deferred"):
        demo._case_measures(r, 1.0)


def test_should_preserve_explicit_noise_and_original_frozen_policy(mixed):
    p = policy("wake1")
    recipe = TicketStreamRecipe("literal-noise", ("old_return",) * 8, 2, True)
    r = demo.run_case(mixed.path, catalog_policy=p, stream_recipe=recipe, deferred_evaluation=True)
    assert r["complete"], r.get("error")
    assert r["known_train_label_corruptions"] == 8
    assert r["training_stream"]["first_targets"][1][0] == ACTION_IDS.index("defer")
    assert r["catalog_control"]["record"]["state_digest"] == p.state_digest
    assert p == policy("wake1")


@pytest.mark.parametrize(
    "kw",
    [
        dict(stream_recipe=object(), deferred_evaluation=True),
        dict(stream_recipe=TicketStreamRecipe("x", ("new_contexts",), 1)),
        dict(deferred_evaluation=True),
        dict(
            stream_recipe=TicketStreamRecipe("x", ("new_contexts",), 1),
            deferred_evaluation=True,
            training_context="old_return",
        ),
    ],
)
def test_should_refuse_invalid_deferred_metadata_before_source_io(monkeypatch, kw):
    monkeypatch.setattr(demo.context, "_source", lambda *a: pytest.fail("source opened"))
    with pytest.raises(ValueError):
        demo.run_case(Path("unused"), catalog_policy=policy(), **kw)


def test_should_keep_trial_protocol_separate_before_source_io(monkeypatch):
    monkeypatch.setattr(demo.context, "_source", lambda *a: pytest.fail("source opened"))
    with pytest.raises(ValueError):
        demo.run_case(
            Path("unused"),
            catalog_trial=request(),
            base_snapshot=object(),
            stream_recipe=TicketStreamRecipe("x", ("new_contexts",), 1),
            deferred_evaluation=True,
        )


def test_should_stop_before_model_construction_for_non_train_packets(mixed, monkeypatch):
    original = demo.build_stream

    def wrong(*args):
        packets = original(*args)
        return (replace(packets[0], role="final_test"), *packets[1:])

    monkeypatch.setattr(demo, "build_stream", wrong)
    r = demo.run_case(
        mixed.path,
        catalog_policy=policy(),
        stream_recipe=TicketStreamRecipe("x", ("new_contexts",) * 8, 1),
        deferred_evaluation=True,
    )
    assert not r["complete"] and mixed.made == []
    assert r["training_work"]["attempted_updates"] == 0


@pytest.mark.parametrize(
    "fault", ["prepared", "scheduled", "recipe", "replacement", "scheduled_replacement"]
)
def test_should_refuse_changed_packet_evidence_after_paid_replay_and_wake(
    mixed, monkeypatch, fault
):
    choose = control.CatalogTicketControl.choose

    def corrupt(owner, observation, progress):
        selected = choose(owner, observation, progress)
        if observation.tick == 2:
            if fault == "recipe":
                object.__setattr__(owner.stream_recipe, "label_delay", 2)
            elif fault == "replacement":
                owner._packets = tuple(list(owner._packets))
            elif fault == "scheduled_replacement":
                owner._schedule_packets = tuple(list(owner._schedule_packets))
            else:
                packets = owner._packets if fault == "prepared" else owner._schedule_packets
                np.copyto(packets[0].targets, 5)
        return selected

    monkeypatch.setattr(control.CatalogTicketControl, "choose", corrupt)
    r = demo.run_case(
        mixed.path,
        catalog_policy=policy(),
        stream_recipe=TicketStreamRecipe("x", ("new_contexts",) * 8, 1),
        deferred_evaluation=True,
    )
    assert not r["complete"] and mixed.made[0].calls == [3, 12]
    assert r["replay_examples"] == 3 and r["budget"]["updates_completed"] == 1
    assert r["training_work"]["attempted_updates"] == r["training_work"]["returned_updates"] == 2
    assert "catalog" in r["error"] and mixed.readouts == []


def test_should_preserve_lost_wake_return_and_replay_charge(mixed, monkeypatch):
    train = FakeBP.train_batch

    def lose_return(owner, features, targets):
        result = train(owner, features, targets)
        if len(features) == 12:
            raise RuntimeError("lost twelve-row native return")
        return result

    monkeypatch.setattr(FakeBP, "train_batch", lose_return)
    r = demo.run_case(
        mixed.path,
        catalog_policy=policy(),
        stream_recipe=TicketStreamRecipe("x", ("new_contexts",) * 8, 1),
        deferred_evaluation=True,
    )
    assert not r["complete"] and mixed.made[0].calls == [3, 12]
    assert r["replay_examples"] == 3
    assert r["catalog_control"]["replay"]["committed_events"] == 1
    assert r["action_snapshot"]["progress"]["unreturned_work_unknown"]
    assert r["training_work"]["unreturned_work_unknown"] and mixed.readouts == []


def test_should_complete_shorter_declared_stream_without_renewing_fit(mixed):
    p = policy()
    r = demo.run_case(
        mixed.path,
        catalog_policy=p,
        stream_recipe=TicketStreamRecipe(
            "short", ("new_contexts", "old_return", "new_contexts"), 2
        ),
        deferred_evaluation=True,
    )
    assert r["complete"], r.get("error")
    assert r["stream"]["opportunities"] == 5 and r["stream"]["unarrived_labels"] == 0
    assert r["stream"]["pending_unapplied"] == 0
    assert r["training_work"]["attempted_rows"] == 45
    assert r["catalog_control"]["record"]["catalog"]["envelope"]["max_decisions"] == 9
    assert r["budget"]["limits"]["max_training_updates"] == 3
    assert p == policy()


@pytest.mark.parametrize(
    "field,value", [("sample_id", "batch-other"), ("arrived_at", 99), ("update_number", 8)]
)
def test_should_refuse_corrupt_fifo_receipt_history_with_all_charges_retained(
    mixed, monkeypatch, field, value
):
    history = control.CatalogTicketControl.require_history

    def corrupt(owner, candidate, wake, receipts=None):
        changed = (replace(receipts[0], **{field: value}), *receipts[1:])
        return history(owner, candidate, wake, changed)

    monkeypatch.setattr(control.CatalogTicketControl, "require_history", corrupt)
    r = demo.run_case(
        mixed.path,
        catalog_policy=policy(),
        stream_recipe=TicketStreamRecipe("x", ("new_contexts",) * 8, 1),
        deferred_evaluation=True,
    )
    assert not r["complete"] and "FIFO" in r["error"]
    assert r["replay_examples"] == 24 and r["training_work"]["attempted_rows"] == 120
    assert len(mixed.made[0].calls) == 16 and mixed.readouts == []


def test_should_leave_existing_readonly_task_consumer_until_after_all_candidates(
    mixed, monkeypatch
):
    results = [
        demo.run_case(
            mixed.path,
            catalog_policy=policy(action),
            stream_recipe=TicketStreamRecipe("x", ("new_contexts",) * 8, 1),
            deferred_evaluation=True,
        )
        for action in ("wake1", "replay_wake")
    ]
    assert all(r["complete"] for r in results) and mixed.readouts == []
    calls = list(mixed.made[0].calls), list(mixed.made[1].calls)

    @dataclass(frozen=True)
    class Work:
        native_forward_attempts: int = 0
        native_forward_returns: int = 0

    reads = []

    def readout(spec, state):
        assert all(r["complete"] for r in results)
        reads.append(state["training_examples"])
        return SimpleNamespace(work=Work(), assert_unchanged=lambda: None)

    monkeypatch.setattr(demo.native, "readout", readout)
    monkeypatch.setattr(
        demo, "_sealed_cases", lambda: tuple(("literal-" + c, c) for c in demo.context.CONCEPTS)
    )
    monkeypatch.setattr(
        demo, "run_workflow_episode", lambda *a, **kw: dict(complete=True, success=False, steps=[])
    )
    for r in results:
        report: dict[str, Any] = dict(sealed_source_reads=[])
        measures = demo._sealed_readout(r, report, demo.LogicalClock())
        assert measures.work_complete and len(report["sealed_episodes"]) == 8
    assert reads == [576, 672, 576, 696]
    assert (mixed.made[0].calls, mixed.made[1].calls) == calls
    assert all(r["final_examples"] == 0 for r in results)


@pytest.mark.parametrize("delay", [1, 2])
@pytest.mark.parametrize("reference", ["periodic", "drift"])
def test_should_preserve_legacy_deferred_reference_horizon_and_report(mixed, delay, reference):
    recipe = TicketStreamRecipe("legacy", ("new_contexts", "old_return") * 4, delay)
    r = demo.run_case(
        mixed.path, reference_mode=reference, stream_recipe=recipe, deferred_evaluation=True
    )
    assert r["complete"], r.get("error")
    assert r["definition"] == "deferred_controller_stream_v1"
    assert r["stream"]["opportunities"] == 8 + delay
    assert r["stream"]["arrived_labels"] == 8
    assert r["controller_work"]["prediction_records_remaining"] == 0
    assert "catalog_control" not in r and "unarrived_labels" not in r["stream"]
    assert r["deferred_actor_unchanged"] and r["episodes"] == []
