"""Paid-data export boundaries; scripted counters, no native fitting or games."""

from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest

from src.adapters import minigrid_context_control as control
from src.app.coupled_attribution import AttributionStream
from test_context_neural_control import scripted as scripted, sources


@pytest.mark.parametrize("when", ["before_seal", "after_seal"])
def test_should_preserve_owned_paid_prefix_when_real_finalization_fails(
    scripted, monkeypatch, when
):
    observed: list[Any] = []
    captured: dict[str, Any] = {}

    def real_stream(*args, **kwargs):
        owner = AttributionStream(*args, **kwargs)
        captured["owner"] = owner
        actual_finish = owner.finish

        def finish():
            assert len(observed) == 1
            assert observed[0].completed_updates == 128
            if when == "after_seal":
                actual_finish()
            raise RuntimeError("intentional finalization failure")

        monkeypatch.setattr(owner, "finish", finish)
        return owner

    monkeypatch.setattr(control, "AttributionStream", real_stream)
    report = control.run_neural_a_control(sources()[:2], endpoint_observer=observed.append)
    assert report["status"] == "failed" and "intentional finalization failure" in report["reason"]
    assert report["finalization"] == "failed"
    assert report["paid_prefix"]["status"] == "unqualified"
    assert report["paid_prefix"]["completed_updates"] == 128
    assert report["native_endpoint"] is report["games"] is None
    assert report["checkpoints"] == [] and len(report["committed_receipts"]) == 128
    assert report["training_budget"]["updates_completed"] == 128
    assert report["training_budget"]["replay_examples_completed"] == 0
    state = captured["owner"]._runtime.candidate_snapshot().state
    assert observed[0] == state and observed[0] is not state
    assert all(a is not b for a, b in zip(observed[0].heads, state.heads))
    assert "games" not in scripted["calls"]


def test_should_preserve_paid_prefix_and_stop_when_observer_raises(scripted):
    observed: list[Any] = []

    def observe(state):
        observed.append(state)
        raise RuntimeError("intentional observer failure after saving owned data")

    report = control.run_neural_a_control(sources()[:2], endpoint_observer=observe)
    assert report["status"] == "failed" and "observer failure" in report["reason"]
    assert len(observed) == 1 and observed[0].training_examples == 1152
    assert report["paid_prefix"]["status"] == "unqualified"
    assert report["finalization"] == "not_started"
    assert scripted["stream_owner"].finished is False
    assert report["native_endpoint"] is report["games"] is None


def test_should_detach_exported_data_and_require_finalization_before_qualification(scripted):
    received: list[Any] = []

    def observe(state):
        assert scripted["stream_owner"].finished is False
        received.append(state)
        object.__setattr__(state, "heads", ())

    report = control.run_neural_a_control(sources()[:2], endpoint_observer=observe)
    assert report["status"] == "complete", report["reason"]
    assert received[0].heads == ()
    assert report["paid_prefix"]["completed_updates"] == 128
    assert report["paid_prefix"]["status"] == "unqualified"
    assert report["finalization"] == "complete"
    assert report["native_endpoint"]["completed_updates"] == 128


@pytest.mark.parametrize("fault", ["missing", "sequence", "clock", "policy", "failed"])
def test_should_refuse_unvalidated_prefix_before_observer_or_finish(scripted, monkeypatch, fault):
    owner_type = control.AttributionStream
    original_run = owner_type.run_until

    def run(owner, n):
        original_run(owner, n)
        if n != 128:
            return
        if fault == "missing":
            owner.applied_updates.pop()
        elif fault in ("sequence", "clock"):
            fields = dict(update_number=127) if fault == "sequence" else dict(applied_at=129)
            owner.applied_updates[-1] = replace(owner.applied_updates[-1], **fields)
        else:
            state = owner._candidate_state

            def changed():
                value = state()
                return (
                    replace(value, policy="replaced")
                    if fault == "policy"
                    else replace(value, failed_work=("unobserved",))
                )

            monkeypatch.setattr(owner, "_candidate_state", changed)

    monkeypatch.setattr(owner_type, "run_until", run)
    observed: list[Any] = []
    report = control.run_neural_a_control(sources()[:2], endpoint_observer=observed.append)
    assert report["status"] == "failed" and observed == []
    assert report["paid_prefix"] is report["native_endpoint"] is report["games"] is None
    assert report["finalization"] == "not_started"
    assert scripted["stream_owner"].finished is False


def test_should_keep_exported_prefix_but_refuse_changed_receipts_at_finish(scripted, monkeypatch):
    owner_type = control.AttributionStream
    original_finish = owner_type.finish

    def finish(owner):
        result = original_finish(owner)
        owner.applied_updates.pop()
        return result

    monkeypatch.setattr(owner_type, "finish", finish)
    observed: list[Any] = []
    report = control.run_neural_a_control(sources()[:2], endpoint_observer=observed.append)
    assert report["status"] == "failed" and report["finalization"] == "failed"
    assert len(observed) == 1 and observed[0].completed_updates == 128
    assert len(report["committed_receipts"]) == 128
    assert report["paid_prefix"]["status"] == "unqualified"
    assert report["native_endpoint"] is report["games"] is None


def test_should_refuse_observer_runtime_replacement_before_finish(scripted):
    def observe(state):
        owner = scripted["stream_owner"]
        owner._runtime = deepcopy(owner._runtime)

    report = control.run_neural_a_control(sources()[:2], endpoint_observer=observe)
    assert report["status"] == "failed" and "runtime was replaced" in report["reason"]
    assert report["paid_prefix"]["status"] == "unqualified"
    assert report["finalization"] == "not_started"
    assert scripted["stream_owner"].finished is False
    assert report["games"] is report["native_endpoint"] is None
