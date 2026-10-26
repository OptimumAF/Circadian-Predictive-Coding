"""Private completed-readout ownership, using preaccepted fake training arms.

The existing evaluator and fake public environment run normally. Training is
stubbed here because these tests target storage/publication, not stream learning;
the existing integration suite retains the real-stream dose and receipt checks.
"""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_learning_pilot as pilot
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from test_minigrid_learning_pilot import (
    POINTS,
    evaluator as evaluator,
    pilot_environment,
)
from test_torch_feature_rehearsal_pilot import Sampler


@pytest.fixture
def storage(monkeypatch, evaluator):
    captured = pilot_environment(monkeypatch, evaluator)
    rows: list[dict[str, Any]] = []
    callbacks: list[str] = []
    row_factory = pilot._row

    def monitor(label):
        callbacks.append(label)
        for row in rows:
            for slot in [*row["readouts"], row["final_evaluation"]]:
                if (
                    type(slot) is dict
                    and "scores" in slot
                    and "episodes" in slot
                    and slot.get("status") in ("complete", "failed")
                ):
                    raise AssertionError(
                        "decoded completed evidence exposed before external callbacks ended"
                    )

    def row(method):
        value = row_factory(method)
        rows.append(value)
        return value

    def accepted_arm(row, packets, sampler, deadline, check_boundary):
        check_boundary()
        source = pilot._learner(row["method"])
        states: list[tuple[int, Any]] = []
        for point in POINTS:
            state = source.snapshot_state()
            state.completed_updates, state.training_examples = point, 9 * point
            states.append((point, state))
        budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=128), lambda: 0.0)
        budget.updates_completed = 128
        stream = SimpleNamespace(completed_updates=128)
        row.update(
            status="trained",
            charged_updates=128,
            native_training_examples=1152,
            source_unchanged=True,
            actor_unchanged=True,
            checkpoint_updates=list(POINTS),
            stream_result=dict(completed_updates=128, training_examples=1152),
            wake_calls_attempted=128,
            wake_calls_returned=128,
        )
        captured["streams"].append((stream, source, packets, budget))
        captured["finished"].append(stream)
        return dict(
            row=row,
            source=source,
            states=states,
            stream=stream,
            budget=budget,
            guard=lambda: monitor("completed_arm_guard"),
        )

    class MonitoredSampler(Sampler):
        def sample(self):
            monitor("rss_sample")
            return super().sample()

        def snapshot(self):
            monitor("rss_snapshot")
            return super().snapshot()

        def __exit__(self, *args):
            monitor("sampler_exit")
            return super().__exit__(*args)

    def readback(*args):
        monitor("readback")

    original_environment = evaluator.work["on_environment"]

    def environment(game):
        monitor("environment_reset")
        original_environment(game)

    monkeypatch.setattr(pilot, "_row", row)
    monkeypatch.setattr(pilot, "_train_arm", accepted_arm)
    monkeypatch.setattr(pilot, "_readback", readback)
    monkeypatch.setattr(pilot, "ProcessRssSampler", MonitoredSampler)
    evaluator.work["on_environment"] = environment
    evaluator.work["on_step"] = lambda game: monitor("environment_step")
    evaluator.work["on_predict"] = lambda owner: monitor("prediction")
    return SimpleNamespace(
        evaluator=evaluator,
        work=evaluator.work,
        captured=captured,
        rows=rows,
        callbacks=callbacks,
        monitor=monitor,
    )


def test_should_detach_completed_helper_payload_and_publish_full_original_tuple_schema_last(
    monkeypatch, storage
):
    evaluate = pilot.evaluate_checkpoint
    borrowed: list[dict[str, Any]] = []
    changed: list[bool] = []

    def capture(source, state, **kwargs):
        result = evaluate(source, state, **kwargs)
        if not borrowed:
            borrowed.append(result)
        return result

    def mutate_retired_helper_result(owner):
        storage.monitor("prediction")
        if borrowed and not changed:
            assert type(storage.rows[0]["readouts"][0]) is bytes
            borrowed[0]["scores"][0]["ce"] = 1234.0
            borrowed[0]["episodes"][0]["predictions"][0]["observation"] = ()
            changed.append(True)

    monkeypatch.setattr(pilot, "evaluate_checkpoint", capture)
    storage.work["on_predict"] = mutate_retired_helper_result
    report = pilot.run_pilot()
    assert report["status"] == "complete", report["reason"]
    assert changed == [True] and borrowed[0]["scores"][0]["ce"] == 1234.0
    assert not storage.work.get("trains")
    assert len(storage.rows) == len(report["methods"]) == 5
    assert len(storage.work["environments"]) == 160
    assert "sampler_exit" in storage.callbacks and "readback" in storage.callbacks
    for row in report["methods"]:
        assert [point["updates"] for point in row["readouts"]] == list(POINTS)
        assert row["final_evaluation"]["status"] == "complete"
        for point in [*row["readouts"], row["final_evaluation"]]:
            assert type(point) is dict and point["status"] == "complete"
            for episode in point["episodes"]:
                assert type(episode["predictions"][0]["observation"]) is tuple
                assert len(episode["predictions"][0]["observation"]) == 148
    first = report["methods"][0]["readouts"][0]
    assert first["scores"][0]["ce"] == pytest.approx(float(np.log(2)))
    assert first["work"]["raw_score_forwards_completed"] == 6
    assert first["work"]["live_prediction_calls_returned"] == 12


@pytest.mark.parametrize("replacement", ["equal_bytes", "invalid_bytes", "public_dict"])
def test_should_refuse_replaced_completed_slot_on_next_prediction_and_keep_paid_original(
    storage, replacement
):
    changed: list[bool] = []

    def corrupt(owner):
        storage.monitor("prediction")
        slot = storage.rows[0]["readouts"][0]
        if type(slot) is bytes and not changed:
            if replacement == "equal_bytes":
                substitute: Any = bytes(bytearray(slot))
                assert substitute == slot and substitute is not slot
            elif replacement == "invalid_bytes":
                substitute = b"not a trusted pickle payload"
            else:
                substitute = dict(status="complete", fabricated=True)
            storage.rows[0]["readouts"][0] = substitute
            changed.append(True)

    storage.work["on_predict"] = corrupt
    report = pilot.run_pilot()
    assert report["status"] == "incomplete"
    assert "fixed readout slots" in report["reason"]
    assert changed == [True]
    original = report["methods"][0]["readouts"][0]
    assert original["status"] == "complete" and original["updates"] == 0
    assert original["scores"][0]["ce"] == pytest.approx(float(np.log(2)))
    assert original["work"]["live_prediction_calls_returned"] == 12
    partial = report["methods"][0]["readouts"][1]
    assert partial["status"] == "failed"
    assert partial["work"]["raw_score_forwards_completed"] == 1
    assert len(storage.work["predicts"]) == 19
    assert len(storage.work["environments"]) == 6
    assert not any(game.role == "final_test" for game in storage.work["environments"])
    assert all(row["charged_updates"] == 128 for row in report["methods"])
    assert not storage.work.get("trains")


def test_should_refuse_pending_slot_mutation_without_accepting_its_unobserved_claim(storage):
    changed: list[bool] = []

    def corrupt(owner):
        storage.monitor("prediction")
        if not changed:
            pending = storage.rows[0]["readouts"][1]
            assert type(pending) is dict and pending["status"] == "not_run"
            pending["updates"] = 96
            changed.append(True)

    storage.work["on_predict"] = corrupt
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and "fixed readout slots" in report["reason"]
    first = report["methods"][0]["readouts"][0]
    assert first["status"] == "failed" and first["work"]["raw_score_forwards_completed"] == 1
    assert report["methods"][0]["readouts"][1] == dict(updates=32, role="train", status="not_run")
    assert len(storage.work["predicts"]) == 1 and not storage.work.get("environments")


def test_should_refuse_replaced_completed_final_slot_before_more_final_physics(storage):
    changed: list[bool] = []

    def corrupt(owner):
        storage.monitor("prediction")
        slot = storage.rows[0]["final_evaluation"]
        if type(slot) is bytes and not changed:
            storage.rows[0]["final_evaluation"] = b"untrusted final bytes"
            changed.append(True)

    storage.work["on_predict"] = corrupt
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and "fixed readout slots" in report["reason"]
    first = report["methods"][0]["final_evaluation"]
    assert first["status"] == "complete" and first["scores"] == []
    assert len(first["episodes"]) == 2
    second = report["methods"][1]["final_evaluation"]
    assert second["status"] == "failed"
    assert second["work"]["live_prediction_calls_returned"] == 1
    assert second["work"]["environment_steps_returned"] == 0
    finals = [game for game in storage.work["environments"] if game.role == "final_test"]
    assert len(finals) == 3 and sum(len(game.actions) for game in finals) == 4
    assert all(game.closed for game in storage.work["environments"])


def test_should_decode_only_private_sealed_payloads_even_when_public_slot_contains_untrusted_bytes(
    monkeypatch, storage
):
    malicious = b"not a private pickle"
    decoded: list[bytes] = []
    changed: list[bool] = []
    loads = pilot.pickle.loads

    def private_load(payload, *args, **kwargs):
        assert payload is not malicious, "public slot payload reached pickle.loads"
        decoded.append(payload)
        return loads(payload, *args, **kwargs)

    def corrupt(owner):
        storage.monitor("prediction")
        if type(storage.rows[0]["readouts"][0]) is bytes and not changed:
            storage.rows[0]["readouts"][0] = malicious
            changed.append(True)

    monkeypatch.setattr(pilot.pickle, "loads", private_load)
    storage.work["on_predict"] = corrupt
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and changed == [True]
    assert decoded and all(payload is not malicious for payload in decoded)
    assert report["methods"][0]["readouts"][0]["status"] == "complete"


@pytest.mark.parametrize("boundary", ["readback", "sampler"])
def test_should_refuse_completed_payload_replacement_at_final_callback_before_publication(
    monkeypatch, storage, boundary
):
    changed: list[bool] = []

    def replace_completed():
        if changed:
            return
        assert all(type(row["final_evaluation"]) is bytes for row in storage.rows)
        original = storage.rows[0]["readouts"][0]
        assert type(original) is bytes
        replacement = bytes(bytearray(original))
        assert replacement == original and replacement is not original
        storage.rows[0]["readouts"][0] = replacement
        changed.append(True)

    if boundary == "readback":
        readback = pilot._readback

        def final_readback(*args):
            readback(*args)
            replace_completed()

        monkeypatch.setattr(pilot, "_readback", final_readback)
    else:
        sampler_type = pilot.ProcessRssSampler
        snapshot = sampler_type.snapshot

        def final_snapshot(self):
            result = snapshot(self)
            if "sampler_exit" in storage.callbacks:
                replace_completed()
            return result

        monkeypatch.setattr(sampler_type, "snapshot", final_snapshot)

    report = pilot.run_pilot()
    assert changed == [True]
    assert report["status"] == "incomplete" and "fixed readout slots" in report["reason"]
    assert report["comparisons"] is None
    assert all(row["outcomes"] is None for row in report["methods"])
    assert all(row["acquisition_conditions"] is None for row in report["methods"])
    first = report["methods"][0]["readouts"][0]
    assert first["status"] == "complete"
    assert first["work"]["raw_score_forwards_completed"] == 6
    assert first["work"]["live_prediction_calls_returned"] == 12
    assert first["scores"][0]["ce"] == pytest.approx(float(np.log(2)))
    assert all(row["final_evaluation"]["status"] == "complete" for row in report["methods"])
    assert len(storage.work["environments"]) == 160
    assert not storage.work.get("trains")
