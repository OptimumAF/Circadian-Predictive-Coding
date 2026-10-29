"""Correct-ready consumption through real stream/runtime with literal predictions."""

from copy import deepcopy
from typing import Any

import numpy as np
import pytest

from src.app.actor_shadow import ActorShadowRuntime
from src.app.coupled_attribution import AttributionBatch, AttributionStream
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.reference_controller import ReferencePolicy
from tests.test_affine_drift_feedback import Learner


def make(targets=(0, 1, 0, 1), observer=None):
    source = Learner()
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=4, max_wall_seconds=60), lambda: 0.0
    )
    batches = tuple(
        AttributionBatch(
            "train", "A", (f"r{i}",), np.array([[float(i)]]), np.array([[float(y)]]), num_classes=2
        )
        for i, y in enumerate(targets)
    )
    owner = AttributionStream(
        source,
        batches,
        (),
        training_only=True,
        checkpoint_updates=tuple(range(len(batches) + 1)),
        label_delay=1,
        budget=budget,
        observe_checkpoint=observer,
    )
    return owner, source, budget


def run(owner, **kwargs):
    return owner.run_reference_window(
        ReferencePolicy("drift"), ready_feedback=True, acknowledge_correct=True, **kwargs
    )


def test_should_advance_correct_packets_without_optimizer_and_train_later_mistakes():
    seen = []
    owner, source, budget = make(
        observer=lambda n, state, b: seen.append((n, None if b is None else b.row_ids))
    )
    initial = source.snapshot_state(), owner._runtime.actor.snapshot_state()
    r = run(owner)
    assert (
        r["window_complete"]
        and r["consumption_complete"]
        and not r["training_complete"]
        and not r["all_packets_fitted"]
    )
    assert (
        r["completed_updates"],
        r["processed_packets"],
        r["pending_unapplied"],
        r["unfitted_packets"],
    ) == (2, 4, 0, 2)
    assert budget.updates_completed == 2 and budget.replay_examples_completed == 0
    assert r["applied_packet_indices"] == [1, 3] and r["training_examples"] == 2
    assert [a["sample_id"] for a in r["acknowledged_experiences"]] == ["batch-0", "batch-2"]
    assert [a["sample_id"] for a in r["applied_updates"]] == ["batch-1", "batch-3"]
    assert seen == [(0, None), (1, ("r1",)), (2, ("r3",))]
    assert (source.snapshot_state(), owner._runtime.actor.snapshot_state()) == initial
    proof = r["acknowledgement"]["proof"]
    assert proof["prediction_calls_completed"] == proof["prediction_rows_completed"] == 2
    assert len([c for c in source.calls if c[0] == "predict"]) == 6


def test_should_consume_all_correct_data_without_any_native_update():
    owner, _, budget = make((0, 0, 0, 0))
    r = run(owner)
    assert (
        r["consumption_complete"]
        and not r["training_complete"]
        and r["completed_updates"] == budget.updates_completed == 0
    )
    assert len(r["acknowledged_experiences"]) == r["processed_packets"] == 4
    assert r["training_examples"] == 0 and r["unfitted_packets"] == 4
    with pytest.raises(ValueError):
        run(owner)


def test_should_keep_default_waiting_schema_and_fifo_behavior():
    owner, _, budget = make((0, 0, 0, 0))
    r = owner.run_reference_window(ReferencePolicy("drift"), ready_feedback=True)
    assert budget.updates_completed == 0 and r["pending_unapplied"] == 4
    assert "acknowledgement" not in r and "processed_packets" not in r


@pytest.mark.parametrize("flag", [1, None, (), "yes"])
def test_should_refuse_nonbool_flag_before_predictions(flag):
    owner, source, _ = make()
    with pytest.raises(ValueError):
        owner.run_reference_window(
            ReferencePolicy("drift"), ready_feedback=True, acknowledge_correct=flag
        )
    assert not source.calls


def test_should_refuse_acknowledgement_without_ready_feedback():
    owner, source, _ = make()
    with pytest.raises(ValueError):
        owner.run_reference_window(ReferencePolicy("periodic"), acknowledge_correct=True)
    assert not source.calls


def test_should_not_use_earlier_zero_mistake_scalar_when_fresh_proof_disagrees():
    owner, source, budget = make((0, 0, 0, 0))
    calls = []

    def predict(model, features):
        calls.append(True)
        return np.full((1, 1), float(len(calls) % 2 == 0))

    source.hooks["predict"] = predict
    r = run(owner)
    assert budget.updates_completed == 0 and r["processed_packets"] == 0
    assert r["acknowledged_experiences"] == []
    assert r["acknowledgement"]["attempts"] == r["acknowledgement"]["returned"] == 4


@pytest.mark.parametrize("fault", ["lost", "nonfinite", "candidate", "source", "actor", "clock"])
def test_should_fail_closed_on_bad_or_lost_fresh_proof(fault):
    owner, source, budget = make((0, 0, 0, 0))
    calls = []

    def predict(model, features):
        calls.append(True)
        if len(calls) == 2:
            if fault == "lost":
                raise RuntimeError("lost proof prediction")
            if fault == "candidate":
                owner._runtime._candidate.state += 1
            elif fault == "source":
                source.state += 1
            elif fault == "actor":
                owner._runtime.actor._learner.state += 1
            elif fault == "clock":
                owner._clock.advance_to(99)
            else:
                return np.full((1, 1), np.nan)
        return np.zeros((1, 1))

    source.hooks["predict"] = predict
    with pytest.raises((ValueError, RuntimeError)):
        run(owner)
    r = owner.reference_report
    assert r["status"] == "failed" and budget.updates_completed == 0
    assert r["acknowledgement"]["attempts"] == 1 and r["acknowledgement"]["returned"] == 0
    assert r["acknowledged_experiences"] == []
    if fault == "lost":
        assert r["acknowledgement"]["proof"]["unreturned_prediction_unknown"]
    with pytest.raises(ValueError):
        run(owner)


def test_should_preserve_actual_consumption_if_ack_return_is_lost(monkeypatch):
    owner, _, budget = make((0, 0, 0, 0))
    original = ActorShadowRuntime.acknowledge_ready

    def lose(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("lost acknowledgement return")

    monkeypatch.setattr(ActorShadowRuntime, "acknowledge_ready", lose)
    with pytest.raises(RuntimeError):
        run(owner)
    r = owner.reference_report
    assert r["processed_packets"] == 1 and r["loop_processed_packets"] == 0
    assert len(r["acknowledged_experiences"]) == 1 and budget.updates_completed == 0
    assert r["acknowledgement"]["unreturned_work_unknown"]


@pytest.mark.parametrize("replace_owner", [False, True, "ledger", "during_proof"])
def test_should_refuse_changed_preceding_ack_and_preserve_known_prefix(replace_owner):
    box: dict[str, Any] = {}

    def observe(n, state, batch):
        if n == 1 and replace_owner != "during_proof":
            ledger = box["owner"]._runtime._inbox._acknowledged
            key = next(iter(ledger))
            if replace_owner == "ledger":
                box["owner"]._runtime._inbox._acknowledged = dict(ledger)
            elif replace_owner:
                ledger[key] = deepcopy(ledger[key])
            else:
                object.__setattr__(ledger[key], "acknowledged_at", 99)

    owner, source, budget = make(observer=observe)
    box["owner"] = owner
    if replace_owner == "during_proof":
        calls = []

        def predict(model, features):
            calls.append(True)
            if len(calls) == 5:
                ledger = owner._runtime._inbox._acknowledged
                object.__setattr__(next(iter(ledger.values())), "acknowledged_at", 99)
            return np.zeros((1, 1))

        source.hooks["predict"] = predict
    with pytest.raises(ValueError):
        run(owner)
    r = owner.reference_report
    assert r["status"] == "failed" and budget.updates_completed == 1
    assert r["known_acknowledgements"][0]["acknowledged_at"] == 1
