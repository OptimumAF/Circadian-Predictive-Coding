"""Finite genuine serving workers, original gate and no outcome feedback."""

from dataclasses import replace
import json
import os
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any, cast

import pytest

from scripts import run_bp_ticket_serving as demo
from scripts.ticket_serving_timeline import LiveServingPlan, TicketServingTimeline
from src.app.resource_sharing import ServingPriorityGate
from src.core.resource_sharing import SharingLimits
from test_ticket_workflow import decision


class LiteralServingFixture:
    def __init__(self, gate, before=None):
        self.gate, self.before = gate, before
        self.calls = []

    def serve(self, request):
        assert self.gate.snapshot().active_serving_requests == 1
        self.calls.append(request)
        if self.before is not None:
            self.before()
        return SimpleNamespace(
            generation=0, actor_version="fixture", native_updates=0, result=decision(request)
        )


def timeline(plan=None, before=None, require_source=None):
    gate = ServingPriorityGate(SharingLimits(1, 2, 1), resource_available=lambda: True)
    predictor = LiteralServingFixture(gate, before)
    live = TicketServingTimeline(
        plan or LiveServingPlan(4, (1,)), gate, predictor, require_source or (lambda: None)
    )
    return live, gate, predictor


@pytest.mark.parametrize(
    "changes",
    [
        {"opportunities": True},
        {"opportunities": 0},
        {"opportunities": 65},
        {"contention_ticks": [1]},
        {"contention_ticks": (1, 1)},
        {"contention_ticks": (2, 1)},
        {"contention_ticks": (True,)},
        {"contention_ticks": (24,)},
        {"wait_seconds": True},
        {"wait_seconds": 0},
        {"wait_seconds": float("nan")},
        {"wait_seconds": float("inf")},
        {"wait_seconds": 2**2048},
    ],
)
def test_bad_live_plan_refuses_before_any_worker_or_native_work(changes):
    with pytest.raises(ValueError):
        replace(LiveServingPlan(), **changes)


def test_reserved_real_serving_lease_defers_native_admission_and_then_completes_episode():
    live, gate, predictor = timeline(LiveServingPlan(1, (0,)))
    try:
        with live.opportunity(0):
            assert gate.snapshot().active_serving_requests == 1
            assert predictor.calls == []  # Prediction waits for the paid admission decision.
            with gate.training() as admitted:
                assert not admitted.allowed and admitted.reason == "serving_active"
        row = live.snapshot()["records"][0]
        assert row["reserved_before_admission"] and row["complete"]
        assert row["episode"]["success"] and len(predictor.calls) == 3
        assert row["training_active_at_lease"] is False
        assert gate.snapshot().admitted_updates == gate.snapshot().active_serving_requests == 0
    finally:
        live.close()


def test_serving_can_run_while_an_original_native_admission_is_active():
    entered, release = Event(), Event()

    def before():
        entered.set()
        assert release.wait(5), "main test did not release request"

    live, gate, predictor = timeline(LiveServingPlan(1, ()), before)
    try:
        with live.opportunity(0):
            with gate.training() as admitted:
                assert admitted.allowed
                live.native_started()
                assert entered.wait(5)
                assert (
                    gate.snapshot().training_active and gate.snapshot().active_serving_requests == 1
                )
                release.set()
        row = live.snapshot()["records"][0]
        assert row["training_active_at_lease"] and row["requests"][0]["training_active_at_start"]
        assert all(
            p["actor_version"] == "fixture" and p["native_updates"] == 0 for p in row["requests"]
        )
        assert live.snapshot()["latency"]["attempts"] == 3
    finally:
        release.set()
        live.close()


def test_native_fault_keeps_already_started_serving_prefix_and_refuses_next_opportunity():
    live, gate, _ = timeline(LiveServingPlan(2, ()))
    try:
        with pytest.raises(RuntimeError, match="native fault"):
            with live.opportunity(0):
                with gate.training():
                    live.native_started()
                    raise RuntimeError("native fault")
        assert live.snapshot()["records"][0]["complete"]
        with pytest.raises(ValueError, match="stopped"):
            with live.opportunity(1):
                pytest.fail("retried stopped timeline")
        assert len(live.records) == 1 and gate.snapshot().active_serving_requests == 0
    finally:
        live.close()


def test_missing_typed_return_keeps_attempt_unknown_work_and_does_not_retry():
    def fail():
        raise RuntimeError("lost typed return")

    live, gate, predictor = timeline(LiveServingPlan(2, ()), fail)
    try:
        with pytest.raises(RuntimeError, match="lost typed return|incomplete"):
            with live.opportunity(0):
                pass
        r = live.snapshot()
        assert len(predictor.calls) == 1 and not r["records"][0]["complete"]
        assert r["unreturned_work_unknown"] and not r["records"][0]["requests"][0]["complete"]
        assert gate.snapshot().active_serving_requests == 0
        with pytest.raises(ValueError, match="stopped"):
            with live.opportunity(1):
                pass
    finally:
        live.close()


def test_failed_controller_choice_does_not_start_a_new_serving_request():
    live, gate, predictor = timeline(LiveServingPlan(1, ()))
    try:
        with pytest.raises(RuntimeError, match="choice"):
            with live.opportunity(0):
                raise RuntimeError("choice failure")
        assert predictor.calls == live.records == []
    finally:
        live.close()


@pytest.mark.parametrize("owner", ["plan", "gate", "predictor"])
def test_replaced_original_live_owner_refused_before_dispatch(owner):
    live, gate, predictor = timeline(LiveServingPlan(1, ()))
    try:
        setattr(live, owner, object())
        with pytest.raises(ValueError, match="original"):
            with live.opportunity(0):
                pass
        assert predictor.calls == []
    finally:
        live.close()


def test_snapshots_are_owned_and_finite_tick_order_is_enforced():
    live, _, _ = timeline(LiveServingPlan(1, ()))
    try:
        with pytest.raises(ValueError, match="sequence"):
            with live.opportunity(1):
                pass
        with live.opportunity(0):
            pass
        snap = live.snapshot()
        snap["records"].clear()
        assert len(live.records) == 1
        with pytest.raises(ValueError):
            with live.opportunity(1):
                pass
    finally:
        live.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reference_mode": "periodic"},
        {"deferred_evaluation": True},
        {"counterfactual": True},
        {"training_context": "old_return"},
        {"contextual_policy": object()},
    ],
)
def test_live_mode_refuses_incompatible_paths_before_source_or_models(kwargs, monkeypatch):
    monkeypatch.setattr(demo.context, "_source", lambda p: pytest.fail("source access"))
    args = dict(reference_mode="drift", serving_plan=LiveServingPlan()) | kwargs
    with pytest.raises(ValueError, match="live serving"):
        demo.run_case(Path("unused"), **cast(Any, args))


def test_cli_explicit_live_mode_and_existing_output_are_preserved(tmp_path, monkeypatch):
    calls = []

    def run(p, **kwargs):
        calls.append(kwargs)
        return {"complete": True}

    monkeypatch.setattr(demo, "run_case", run)
    output = tmp_path / "live.json"
    assert (
        demo.main(["--source", "unused", "--output", str(output), "--serving-during-updates"]) == 0
    )
    assert calls == [dict(reference_mode="drift", serving_plan=LiveServingPlan())]
    with pytest.raises(FileExistsError):
        demo.main(["--source", "unused", "--output", str(output), "--serving-during-updates"])
    with pytest.raises(SystemExit):
        demo.main(
            [
                "--source",
                "unused",
                "--output",
                str(tmp_path / "bad.json"),
                "--serving-during-updates",
                "--reference",
                "drift",
            ]
        )


@pytest.fixture(scope="module")
def native_result():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(demo.context, "BATCHES_PER_PHASE", 2)
        result = demo.run_case(
            Path("artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json"),
            reference_mode="drift",
            serving_plan=LiveServingPlan(4, (1,)),
        )
    if os.environ.get("LIVE_SERVING_NATIVE_REPORT"):
        Path(os.environ["LIVE_SERVING_NATIVE_REPORT"]).write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf8"
        )
    assert result["complete"], result.get("error")
    return result


def test_native_live_episode_versions_original_causal_receipts_and_priorities(native_result):
    r = native_result
    assert r["training_work"] == dict(
        attempted_updates=2, attempted_rows=12, returned_updates=2, unreturned_work_unknown=False
    )
    assert r["stream"]["opportunities"] == 4 and r["stream"]["pending_unapplied"] == 0
    assert r["action_snapshot"]["progress"]["decisions_attempted"] == 4
    live = r["live_serving"]
    assert live["closed"] and not live["stopped"] and not live["unreturned_work_unknown"]
    assert len(live["records"]) == 4 and all(v["complete"] for v in live["records"])
    assert r["controller_events"][1]["poll"]["training"]["deferred_reason"] == "serving_active"
    assert sum(v["training_active_at_lease"] for v in live["records"]) == 2
    requests = [p for v in live["records"] for p in v["requests"]]
    assert all(
        p["complete"]
        and p["generation"] == 0
        and p["actor_version"] == "qualified-96"
        and p["native_updates"] == 96
        for p in requests
    )
    assert live["latency"]["attempts"] == len(requests)
    assert 0 <= live["latency"]["p50"] <= live["latency"]["p95"] <= live["latency"]["maximum"]
    assert r["source_unchanged"] and r["numpy_rng_unchanged"] and r["torch_rng_unchanged"]
    assert r["final_examples"] == r["reserved_probe_updates"] == r["replay_examples"] == 0


def test_native_live_task_records_do_not_become_feedback_or_promotion_guards(native_result):
    r = native_result
    assert r["controller_work"]["feedback_rows"] == 12
    assert [e["observation"]["feedback"]["sample_id"] for e in r["controller_events"][1:]] == [
        "batch-0",
        "batch-1",
        "batch-1",
    ]
    assert all(g["role"] != "live_service" for g in r["guards"])
    assert all(u["sample_id"].startswith("batch-") for u in r["applied_updates"])
    assert all(row["role"] == "live_service" for row in r["live_serving"]["records"])
