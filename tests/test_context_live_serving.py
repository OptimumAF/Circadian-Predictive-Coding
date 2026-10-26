"""Context worker ports, fixed schedule and failures; paid snapshots, no refits."""

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import context_head_live as demo
from scripts import run_bp_ticket_serving as cli
from scripts.context_head_probe import head_states
from scripts.context_head_serving import TASKS
from scripts.ticket_serving_timeline import LiveServingPlan, TicketServingTimeline
from src.adapters.context_perceptron import ContextPerceptronLearner
from src.adapters.context_serving import ContextServingPredictor
from src.app.resource_sharing import ServingPriorityGate
from src.app.serving_promotion import PromotableActor
from src.core.decision_features import DecisionFeatureSpec
from src.core.resource_sharing import SharingLimits
from src.core.serving_ports import ServingConfiguration
from src.core.ticket_workflow import CATALOG
from test_ticket_workflow import decision


@pytest.mark.parametrize(
    "tasks",
    [
        [],
        (),
        (("v1", "invoice"),),
        (("v1", "invoice"),) * 3,
        (["v1", "invoice"],) * 2,
        (("v1",),) * 2,
        (("v1", "invoice", "x"),) * 2,
        (("v3", "invoice"),) * 2,
        (("v1", "missing"),) * 2,
        ((True, "invoice"),) * 2,
        (("v1", True),) * 2,
    ],
)
def test_should_refuse_invalid_immutable_context_plan_before_worker(tasks, monkeypatch):
    monkeypatch.setattr(
        "scripts.ticket_serving_timeline.ThreadPoolExecutor",
        lambda **kw: pytest.fail("worker allocated"),
    )
    with pytest.raises(ValueError, match="public context tasks"):
        LiveServingPlan(2, (), context_tasks=tasks)


class Literal:
    def __init__(self):
        self.requests = []

    def serve(self, request):
        self.requests.append(request)
        words = request.query.split()
        selected = (
            "open"
            if "sealed" in words
            else "commit"
            if "staged" in words
            else (
                "route"
                if "policy_v1" in words and "invoice" in words
                else "inspect"
                if "invoice" in words or "password" in words
                else "defer"
            )
        )
        return SimpleNamespace(
            generation=0,
            actor_version="fixture",
            native_updates=12,
            result=decision(request, selected),
        )


def make(plan, predictor=None, require=None):
    gate = ServingPriorityGate(SharingLimits(1, 2, 1), resource_available=lambda: True)
    predictor = predictor or Literal()
    return TicketServingTimeline(plan, gate, predictor, require or (lambda: None)), gate, predictor


def test_should_tag_each_request_with_its_declared_observed_context():
    tasks = (("v1", "invoice"), ("v2", "parcel"))
    live, _, p = make(LiveServingPlan(2, (), context_tasks=tasks))
    try:
        for tick in range(2):
            with live.opportunity(tick):
                pass
        rows = live.snapshot()["records"]
        assert all(r["complete"] and r["episode"]["success"] for r in rows)
        assert [r["policy_version"] for r in rows] == ["v1", "v2"]
        assert all(r["public_context"] for r in rows)
        for i, q in enumerate(p.requests):
            assert ("policy_v1" if i < 3 else "policy_v2") in q.query.split()
    finally:
        live.close()


def test_should_preserve_default_v2_queries_without_context_markers():
    live, _, p = make(LiveServingPlan(1, ()))
    try:
        with live.opportunity(0):
            pass
        row = live.snapshot()["records"][0]
        assert "public_context" not in row and "policy_version" not in row
        assert row["episode"]["initial_state"]["policy_version"] == "v2"
        assert all("policy_" not in q.query for q in p.requests)
    finally:
        live.close()


def test_should_refuse_tampered_context_plan_before_dispatch():
    plan = LiveServingPlan(1, (), context_tasks=(("v1", "invoice"),))
    live, _, p = make(plan)
    try:
        object.__setattr__(plan, "context_tasks", (("v2", "parcel"),))
        with pytest.raises(ValueError, match="original"):
            with live.opportunity(0):
                pass
        assert p.requests == []
    finally:
        live.close()


def test_should_preserve_context_serving_prefix_on_worker_source_failure():
    calls = []

    def require():
        calls.append(1)
        if len(calls) == 3:
            raise ValueError("source changed")

    live, gate, p = make(
        LiveServingPlan(1, (), context_tasks=(("v1", "invoice"),)), require=require
    )
    try:
        with pytest.raises(RuntimeError, match="source changed|incomplete"):
            with live.opportunity(0):
                pass
        row = live.snapshot()["records"][0]
        assert not row["complete"] and len(row["requests"]) == 1
        assert row["requests"][0]["complete"] and len(p.requests) == 1
        assert gate.snapshot().active_serving_requests == 0
    finally:
        live.close()


def test_should_bind_actual_context_reader_and_complete_counter_to_worker():
    state = head_states(
        json.loads(Path("artifacts/runs/context-head-control-20261023/actual.json").read_bytes())
    )[0]
    learner = ContextPerceptronLearner(state.policy)
    learner.restore_state(state)
    actor = PromotableActor(
        learner,
        version="qualified-12",
        configuration=ServingConfiguration(3, 2),
        feature_digest=lambda x: sha256(x.tobytes()).hexdigest(),
        metadata={},
    )
    spec = DecisionFeatureSpec(demo.CONTEXT_FEATURES, CATALOG)
    p = ContextServingPredictor(actor, spec)
    live, _, _ = make(LiveServingPlan(6, (), context_tasks=TASKS), predictor=p)
    try:
        for tick in range(6):
            with live.opportunity(tick):
                pass
        rows = live.snapshot()["records"]
        assert all(r["complete"] for r in rows)
        assert [r["episode"]["success"] for r in rows] == [True, True, True, True, False, True]
        assert all(
            q["generation"] == 0
            and q["actor_version"] == "qualified-12"
            and q["native_updates"] == 12
            for r in rows
            for q in r["requests"]
        )
        assert p.work["returns"] == sum(len(r["requests"]) for r in rows)
        assert learner.snapshot_state() == actor.serving_snapshot().model_state == state
    finally:
        live.close()


@pytest.mark.parametrize(
    "intervals,expected",
    [
        ([], (0, 0, 0)),
        ([dict(model_started_at=2.0, model_finished_at=3.0)], (1, 0, 1)),
        ([dict(model_started_at=4.0, model_finished_at=5.0)], (0, 0, 0)),
    ],
)
def test_should_measure_observed_overlap_without_demanding_positive_result(intervals, expected):
    rows = [
        dict(
            requests=[
                dict(complete=True, started_at=2.5, seconds=1.0),
                dict(complete=False, started_at=2.6, seconds=0.1),
            ]
        )
    ]
    r = demo.overlap_readback(rows, intervals)
    assert r["request_attempts"] == 2 and r["complete_requests"] == 1
    assert (
        r["started_during_model"],
        r["finished_during_model"],
        r["intersected_model"],
    ) == expected


def test_should_preserve_original_fixed_schedule_and_contexts():
    p = demo.live_plan()
    assert p.opportunities == 14 and p.contention_ticks == (2,)
    assert p.context_tasks == demo.LIVE_TASKS
    assert p.context_tasks[7:13] == TASKS


def test_should_reject_replaced_source_before_teacher_or_worker(tmp_path, monkeypatch):
    p = tmp_path / "bad.json"
    p.write_text("{}")
    monkeypatch.setattr(demo, "prepare_data", lambda *a: pytest.fail("teacher"))
    monkeypatch.setattr(demo, "TicketServingTimeline", lambda *a: pytest.fail("worker"))
    r = demo.run_case(p)
    assert not r["complete"] and "replaced" in r["error"]
    assert not r["native_calls"] and r["training_work"]["attempted_updates"] == 0


def test_should_optin_cli_preserve_exclusive_output_and_native_default(tmp_path, monkeypatch):
    calls = []

    def run(p):
        calls.append(p)
        return dict(complete=True)

    monkeypatch.setattr(demo, "run_case", run)
    p = tmp_path / "live.json"
    assert cli.main(["--source", "unused", "--output", str(p), "--context-head-serving-live"]) == 0
    with pytest.raises(FileExistsError):
        cli.main(["--source", "unused", "--output", str(p), "--context-head-serving-live"])
    assert len(calls) == 1
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--source",
                "unused",
                "--output",
                str(tmp_path / "bad"),
                "--context-head-serving-live",
                "--context-head-serving",
            ]
        )
