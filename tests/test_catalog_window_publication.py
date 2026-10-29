"""Actual promotion controller/evaluator and live workers, with literal learners."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
import pytest

from scripts import run_bp_ticket_serving as demo
from scripts import ticket_catalog_control as control
from scripts.ticket_serving_timeline import LiveServingPlan
from src.app.promotion_guard_evaluation import PromotionGuardEvaluator
from src.app.serving_promotion import ServingPromotionController
from src.app.ticket_workflow import run_workflow_episode
from src.core.promotion_guard import GuardBatch
from src.core.ticket_workflow import ACTION_IDS
from test_catalog_live_serving import serving as serving
from test_catalog_ticket_stream import policy
from test_ticket_catalog_control import FakeBP, digest, request
from test_ticket_catalog_control import driver as driver


class GuardLearner(FakeBP):
    def predict(self, features):
        p = np.zeros((len(features), 6))
        for i, row in enumerate(features):
            p[i, int(row[0] == 3 or self.state.completed_updates > 96)] = 1
        return p


def promotion_digest(state):
    return sha256(digest(state).encode()).hexdigest()


@pytest.fixture
def publication(serving, monkeypatch):
    owners, controllers = [], []
    bind = control.CatalogTicketControl.bind

    def bound(owner, runtime, source, gate):
        shared = bind(owner, runtime, source, gate)
        owners.append((owner, runtime, shared, gate))
        return shared

    monkeypatch.setattr(control.CatalogTicketControl, "bind", bound)
    monkeypatch.setattr(demo, "run_workflow_episode", run_workflow_episode)
    original_public: Any = demo.TorchServingPredictor

    def public(actor, spec):
        port = original_public(actor, spec)
        port.release.set()  # Original before/midway phase readouts are synchronous.
        return port

    monkeypatch.setattr(demo, "TorchServingPredictor", public)

    def guards(spec, now):
        return tuple(
            GuardBatch(
                name,
                tuple((name, str(i)) for i in range(2)),
                now - 1,
                np.tile([marker, 0, 0, 0, 0, 0, 0], (2, 1)),
                np.ones((2, 1)),
                "inner_guard",
            )
            for name, marker in (("literal-new", 2), ("literal-old", 3))
        )

    def evaluator(spec, clock):
        return PromotionGuardEvaluator(
            build_learner=build,
            utility=lambda p, y: float(np.mean(np.argmax(p, axis=1) == y[:, 0])),
            actions=lambda p: tuple(ACTION_IDS[i] for i in np.argmax(p, axis=1)),
            state_valid=lambda state: bool(
                np.isfinite(state.weight) and np.isfinite(state.momentum)
            ),
            prediction_valid=lambda p: bool(np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1)),
            resource_bytes=lambda state: 128,
            state_digest=promotion_digest,
            clock=monotonic,
        )

    build = lambda state: GuardLearner()
    monkeypatch.setattr(demo, "build_native", build)
    monkeypatch.setattr(demo, "categorical_guard", evaluator)
    monkeypatch.setattr(demo, "guards", guards)

    def controller(*args, **kwargs):
        kwargs["state_digest"] = promotion_digest
        result = ServingPromotionController(*args, **kwargs)
        controllers.append(result)
        return result

    monkeypatch.setattr(demo, "ServingPromotionController", controller)
    serving.owners, serving.controllers = owners, controllers
    return serving


def run(publication, **changes):
    args = (
        dict(
            catalog_policy=policy("wake1"),
            serving_plan=LiveServingPlan(9),
            publish_after_window=True,
        )
        | changes
    )
    return demo.run_case(publication.path, **args)


def test_should_publish_after_window_without_draining_or_reusing_pending(publication):
    r = run(publication)
    assert r["complete"], r.get("error")
    assert r["definition"] == "live_catalog_bp_ticket_publication_v1"
    row = r["window_publication"]
    assert row["completed"] and row["accepted"] and not row["publication_unknown"]
    assert row["retired"] and len(row["pending_unapplied"]) == 2
    assert [p["sample_id"] for p in row["pending_unapplied"]] == ["batch-6", "batch-7"]
    assert all(p["source_version"] == "qualified-96" for p in row["pending_unapplied"])
    assert r["training_work"]["returned_updates"] == 6
    assert r["training_work"]["attempted_rows"] == 36 and r["replay_examples"] == 0
    assert r["stream"]["pending_unapplied"] == 2 and len(r["applied_updates"]) == 6
    assert row["receipt"] == r["promotion_receipt"]
    assert row["receipt"]["generation"] == 1
    assert row["receipt"]["previous_version"] == "qualified-96"
    assert row["receipt"]["actor_version"] == "c0"
    post = [v for v in r["episodes"] if v["phase"] == "window-published"]
    assert len(post) == 4 and all(v["episode"]["complete"] for v in post)
    assert all(
        p["generation"] == 1 and p["actor_version"] == "c0" and p["native_updates"] == 102
        for v in post
        for p in v["publications"]
    )
    owner, runtime, shared, gate = publication.owners[0]
    assert gate.snapshot().paused and gate.snapshot().admitted_updates == 6
    assert runtime._retired and runtime._inbox._historical_completed_updates == 6
    assert len(runtime._inbox._experiences) == len(runtime._inbox._labels) == 8
    with pytest.raises(ValueError):
        runtime.train_ready()
    with pytest.raises(ValueError):
        shared.train_action("wake1")
    with pytest.raises(ValueError, match="retired"):
        owner.choose(None, None)
    assert not r["catalog_actor_unchanged"] and r["catalog_actor_unchanged_before_publication"]
    assert r["catalog_promotions"] == 1 and r["source_unchanged"]
    assert r["numpy_rng_unchanged"] and r["torch_rng_unchanged"]
    with pytest.raises(ValueError, match="live catalog"):
        demo._case_measures(r, 1.0)
    with pytest.raises(ValueError, match="terminal"):
        control.case_work(r)


def test_should_retire_rejected_window_without_retraining_or_publishing(publication, monkeypatch):
    monkeypatch.setattr(demo, "policy", lambda: replace(demo_policy, min_new_gain=2.0))
    r = run(publication)
    assert r["complete"], r.get("error")
    assert not r["window_publication"]["accepted"] and r["window_publication"]["retired"]
    assert not r["window_publication"]["publication_unknown"]
    assert r["catalog_promotions"] == 0 and r["catalog_actor_unchanged"]
    assert r["training_work"]["returned_updates"] == 6 and r["stream"]["pending_unapplied"] == 2
    assert "promotion_receipt" not in r
    with pytest.raises(ValueError, match="terminal"):
        control.case_work(r)
    assert all(
        p["generation"] == 0
        for v in r["episodes"]
        if v["phase"] == "window-rejected"
        for p in v["publications"]
    )


demo_policy = demo.policy()


@pytest.mark.parametrize(
    "fault",
    [
        "lost",
        "missing",
        "generation",
        "previous",
        "version",
        "boolean",
        "copy",
        "metadata_float",
        "config_float",
    ],
)
def test_should_preserve_unknown_or_mismatched_commit_without_readout(
    publication, monkeypatch, fault
):
    commit = ServingPromotionController.commit

    def broken(controller, runtime, ticket):
        receipt = commit(controller, runtime, ticket)
        if fault == "lost":
            raise RuntimeError("lost publication return")
        if fault == "missing":
            return None
        if fault in ("metadata_float", "config_float"):
            with controller._actor._payload_exclusive():
                bundle = controller._actor._slot.bundle
                if fault == "metadata_float":
                    bundle.metadata["native_history"] = 102.0
                else:
                    object.__setattr__(
                        bundle.configuration,
                        "cache_ttl_ticks",
                        float(bundle.configuration.cache_ttl_ticks),
                    )
            return receipt
        changed: dict[str, Any] = {
            "generation": {"generation": 2, "boolean": True}.get(fault, receipt.generation),
            "previous_version": "wrong" if fault == "previous" else receipt.previous_version,
            "actor_version": "wrong" if fault == "version" else receipt.actor_version,
        }
        return replace(receipt, **changed)

    monkeypatch.setattr(ServingPromotionController, "commit", broken)
    r = run(publication)
    assert not r["complete"]
    assert r["window_publication"]["publication_unknown"] and r["window_publication"]["retired"]
    assert not any(v["phase"].startswith("window-") for v in r["episodes"])
    assert publication.owners[0][1]._retired and publication.owners[0][3].snapshot().paused
    assert r["catalog_promotions"] is None and r["catalog_actor_unchanged"] is None
    assert r["training_work"]["returned_updates"] == 6 and r["stream"]["pending_unapplied"] == 2


@pytest.mark.parametrize(
    "changes",
    [
        dict(catalog_policy=None),
        dict(serving_plan=None),
        dict(catalog_trial=request(), base_snapshot=object()),
        dict(publish_on_drain=True),
        dict(deferred_evaluation=True),
        dict(followup_plan=object()),
        dict(publish_after_window=1),
        dict(publish_after_window=None),
    ],
)
def test_should_refuse_unsupported_terminal_publication_before_source(monkeypatch, changes):
    monkeypatch.setattr(demo.context, "_source", lambda p: pytest.fail("source opened"))
    args = (
        dict(catalog_policy=policy(), serving_plan=LiveServingPlan(9), publish_after_window=True)
        | changes
    )
    with pytest.raises(ValueError):
        demo.run_case(Path("unused"), **args)


@pytest.mark.parametrize(
    "action,updates,rows,pending,accepted",
    [
        ("wait", 0, 0, 8, False),
        ("replay_wake", 6, 18, 8, True),
    ],
)
def test_should_preserve_full_native_history_for_replay_or_zero_work(
    publication, action, updates, rows, pending, accepted
):
    r = run(publication, catalog_policy=policy(action))
    assert r["complete"], r.get("error")
    assert r["training_work"]["returned_updates"] == updates
    assert r["training_work"]["attempted_rows"] == rows
    assert len(r["window_publication"]["pending_unapplied"]) == pending
    assert r["window_publication"]["accepted"] is accepted
    assert publication.owners[0][1]._inbox._historical_completed_updates == 0
    if accepted:
        assert publication.ports[0].actor.serving_snapshot().metadata["native_history"] == 102
        assert r["replay_examples"] == 18
        assert r["catalog_control"]["native_history"]["replay_updates"] == 6


@pytest.mark.parametrize(
    "fault",
    [
        "pending_equal",
        "pending_value",
        "receipt_equal",
        "report_equal",
        "budget",
        "admissions",
        "progress",
        "pause_type",
    ],
)
def test_should_refuse_changes_to_paid_prefix_before_guard_publication(
    publication, monkeypatch, fault
):
    publish = demo._publish_candidate

    def changed(runtime, controller, spec, now, report, require, updates):
        inbox = runtime._inbox
        if fault.startswith("pending"):
            key = next(k for k in inbox._labels if k[1] == "batch-6")
            if fault == "pending_equal":
                inbox._labels[key] = replace(inbox._labels[key])
            else:
                inbox._labels[key].targets[0, 0] = 5
        elif fault == "receipt_equal":
            key = next(iter(inbox._applied))
            inbox._applied[key] = replace(inbox._applied[key])
        elif fault == "report_equal":
            report["candidate_checkpoint"] = deepcopy(report["candidate_checkpoint"])
        elif fault == "budget":
            runtime._budget.budget = replace(runtime._budget.budget, max_training_updates=9)
        elif fault == "admissions":
            publication.owners[0][3]._admitted += 1
        elif fault == "pause_type":
            publication.owners[0][3]._paused = 1
        else:
            report["action_snapshot"]["progress"]["decisions_attempted"] = 8
        return publish(runtime, controller, spec, now, report, require, updates)

    monkeypatch.setattr(demo, "_publish_candidate", changed)
    r = run(publication)
    assert not r["complete"] and r["window_publication"]["retired"]
    assert publication.ports[0].actor.serving_snapshot().generation == 0
    assert r["training_work"]["returned_updates"] == 6
    assert not any(v["phase"].startswith("window-") for v in r["episodes"])


@pytest.mark.parametrize("fault", ["receipt_bool", "record_equal", "actor", "resume"])
def test_should_refuse_changed_publication_during_postwindow_service(
    publication, monkeypatch, fault
):
    publish = demo._publish_candidate

    def attach(*args):
        receipt = publish(*args)
        report = args[4]
        port = publication.ports[0]

        def after():
            if fault == "receipt_bool":
                report["promotion_receipt"]["generation"] = True
            elif fault == "record_equal":
                report["window_publication"] = deepcopy(report["window_publication"])
            elif fault == "resume":
                publication.owners[0][3].resume()
            else:
                with port.actor._payload_exclusive():
                    model = port.actor._payload_references().models[0]
                    model.state = replace(model.state, momentum=999.0)

        port.after = after
        return receipt

    monkeypatch.setattr(demo, "_publish_candidate", attach)
    r = run(publication)
    assert not r["complete"] and publication.owners[0][1]._retired
    assert r["training_work"]["returned_updates"] == 6
    post = [v for v in r["episodes"] if v["phase"] == "window-published"]
    assert len(post) == 1 and not post[0]["episode"]["complete"]
    assert publication.ports[0].work["returns"] == 48


def test_should_preserve_known_commit_if_ledger_retirement_fails(publication, monkeypatch):
    publish = demo._publish_candidate

    def fail(*args):
        receipt = publish(*args)

        def lost():
            raise RuntimeError("lost ledger retirement")

        monkeypatch.setattr(args[0]._inbox, "_retire_ledger", lost)
        return receipt

    monkeypatch.setattr(demo, "_publish_candidate", fail)
    r = run(publication)
    assert not r["complete"] and r["catalog_promotions"] == 1
    row = r["window_publication"]
    assert row["accepted"] and not row["publication_unknown"]
    assert row["runtime_retired"] and not row["retired"]
    assert "lost ledger retirement" in row["retirement_error"]
    owner, runtime, shared, gate = publication.owners[0]
    assert gate.snapshot().paused
    gate.resume()
    with pytest.raises(ValueError, match="retired"):
        runtime.train_ready()
    with pytest.raises(ValueError):
        shared.train_action("wake1")
    with pytest.raises(ValueError, match="retired"):
        owner.choose(None, None)
    assert not any(v["phase"].startswith("window-") for v in r["episodes"])
