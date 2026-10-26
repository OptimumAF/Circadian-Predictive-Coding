"""Single causal publication, genuine guard refusal and no feedback rebinding."""

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from scripts import run_bp_ticket_serving as demo
from scripts.ticket_serving_timeline import LiveServingPlan
from src.core.reference_controller import ArrivedTrainingFeedback
from src.core.serving_ports import PromotionReceipt


class ActorFixture:
    def __init__(self):
        self.version, self.generation = "old", 0

    def serving_snapshot(self):
        return SimpleNamespace(actor_version=self.version, generation=self.generation)

    def publish(self):
        self.version, self.generation = "new", self.generation + 1
        return PromotionReceipt(self.generation, "old", "new")


def feedback():
    return ArrivedTrainingFeedback("train", "batch-1", "old", 1, 2, 1, 6)


def test_verified_publication_invalidates_feedback_without_rebinding_or_losing_old_evidence():
    actor, original = ActorFixture(), feedback()
    report: dict[str, Any] = {}
    calls = []

    def publish(owner, tick):
        calls.append((owner, tick))
        return actor.publish()

    owner = object()
    assert demo._publish_at_drain(actor, owner, publish, 3, original, report, 2) is None
    assert calls == [(owner, 3)]
    row = report["drain_publication"]
    assert row["completed"] and row["accepted"] and not row["publication_unknown"]
    assert row["wake_updates"] == 2 and row["pending_predictions"] == 0
    invalidated = report["feedback_invalidations"][0]
    assert invalidated["feedback"]["actor_version"] == original.actor_version == "old"
    assert invalidated["actor_version"] == actor.version == "new"
    with pytest.raises(ValueError, match="already attempted"):
        demo._publish_at_drain(actor, owner, publish, 4, original, report, 2)
    assert len(calls) == 1


def test_guard_refusal_preserves_original_feedback_and_never_retries_at_tail():
    actor, old = ActorFixture(), feedback()
    report: dict[str, Any] = {}
    calls = []

    def refuse(owner, tick):
        calls.append(tick)
        return None

    assert demo._publish_at_drain(actor, object(), refuse, 3, old, report, 2) is old
    assert report["drain_publication"]["completed"] and not report["drain_publication"]["accepted"]
    assert not report["drain_publication"]["publication_unknown"]
    assert report["feedback_invalidations"] == [] and actor.version == "old"
    with pytest.raises(ValueError, match="already attempted"):
        demo._publish_at_drain(actor, object(), refuse, 5, old, report, 2)
    assert calls == [3]


@pytest.mark.parametrize(
    "fault",
    [
        "lost_return",
        "missing_receipt",
        "wrong_generation",
        "wrong_previous",
        "wrong_type",
        "coerced_generation",
        "coerced_after",
    ],
)
def test_uncertain_or_mismatched_publication_is_preserved_and_cannot_be_retried(fault):
    actor = ActorFixture()
    report: dict[str, Any] = {}

    def publish(owner, tick):
        receipt = actor.publish()
        if fault == "lost_return":
            raise RuntimeError("lost commit return")
        if fault == "missing_receipt":
            return None
        if fault == "wrong_generation":
            return replace(receipt, generation=2)
        if fault == "wrong_previous":
            return replace(receipt, previous_version="other")
        if fault == "coerced_generation":
            return replace(receipt, generation=True)
        if fault == "coerced_after":
            cast(Any, actor).generation = 1.0
            return receipt
        return object()

    with pytest.raises((RuntimeError, ValueError)):
        demo._publish_at_drain(actor, object(), publish, 3, feedback(), report, 2)
    row = report["drain_publication"]
    assert not row["completed"] and row["publication_unknown"]
    assert row["actor_version_after"] == "new" and report["feedback_invalidations"] == []
    with pytest.raises(ValueError, match="already attempted"):
        demo._publish_at_drain(actor, object(), publish, 4, None, report, 2)


def test_stale_feedback_refused_before_publication_instead_of_relabelled():
    actor = ActorFixture()
    report: dict[str, Any] = {}
    with pytest.raises(ValueError, match="stale feedback"):
        demo._publish_at_drain(
            actor,
            object(),
            lambda *a: pytest.fail("callback entered"),
            3,
            replace(feedback(), actor_version="foreign"),
            report,
            2,
        )
    assert actor.version == "old" and not report["drain_publication"]["publication_unknown"]


@pytest.mark.parametrize("value", [False, 0.0])
def test_non_native_original_generation_is_refused_before_callback(value):
    actor = ActorFixture()
    actor.generation = value
    report: dict[str, Any] = {}
    with pytest.raises(ValueError, match="native actor generation"):
        demo._publish_at_drain(
            actor, object(), lambda *a: pytest.fail("callback entered"), 3, feedback(), report, 2
        )
    assert not report["drain_publication"]["publication_unknown"]


@dataclass(frozen=True)
class DecisionFixture:
    accepted: bool = False
    reasons: tuple[str, ...] = ("fixture refusal",)


@dataclass(frozen=True)
class GuardFixture:
    decision: DecisionFixture = DecisionFixture()


def test_factored_original_publication_uses_same_guards_ids_metadata_and_no_live_outcomes():
    spec = demo.DecisionFeatureSpec(demo.FEATURES, demo.CATALOG)
    order, calls = [], []
    ticket = SimpleNamespace(report=GuardFixture(DecisionFixture(True, ())))
    owner = object()
    receipt = PromotionReceipt(1, "old", "new")

    class Controller:
        def prepare(self, runtime, new, old, **kwargs):
            order.append("prepare")
            calls.append((runtime, new, old, kwargs))
            return ticket

        def commit(self, runtime, given):
            order.append("commit")
            assert runtime is owner and given is ticket
            return receipt

    report = {"live_serving": object()}  # No inspectable live outcomes are needed.
    result = demo._publish_candidate(
        owner, Controller(), spec, 3, report, lambda: order.append("input"), 2
    )
    assert result is receipt and order == ["input", "prepare", "input", "commit"]
    runtime, new, old, kwargs = calls[0]
    assert kwargs["now"] == 3 and len(kwargs["training_ids"]) == 576
    assert kwargs["metadata"] == dict(origin=demo.context.SOURCE_SHA, native_history=98)
    assert len(new.features) == len(old.features) == 6
    assert report["promotion_receipt"] == dict(
        generation=1, previous_version="old", actor_version="new"
    )


def test_measured_guard_refusal_returns_without_entering_commit():
    class Controller:
        def prepare(self, *args, **kwargs):
            raise demo.PromotionRejected(cast(Any, GuardFixture()))

        def commit(self, *args):
            pytest.fail("commit after refusal")

    report: dict[str, Any] = {}
    result = demo._publish_candidate(
        object(),
        Controller(),
        demo.DecisionFeatureSpec(demo.FEATURES, demo.CATALOG),
        3,
        report,
        lambda: None,
        2,
    )
    assert result is None and not report["guard_report"]["decision"]["accepted"]
    assert "promotion_receipt" not in report


@pytest.mark.parametrize(
    "kwargs",
    [
        {"publish_on_drain": 1},
        {"publish_on_drain": True},
        {"publish_on_drain": True, "serving_plan": LiveServingPlan(), "reference_mode": "periodic"},
        {
            "publish_on_drain": True,
            "serving_plan": LiveServingPlan(),
            "reference_mode": "drift",
            "deferred_evaluation": True,
        },
    ],
)
def test_invalid_publication_mode_is_refused_before_source_or_native_access(kwargs, monkeypatch):
    monkeypatch.setattr(demo.context, "_source", lambda p: pytest.fail("source accessed"))
    with pytest.raises(ValueError, match="drained publication|live serving"):
        demo.run_case(Path("unused"), **kwargs)


def test_new_cli_is_explicit_and_does_not_overwrite_existing_output(tmp_path, monkeypatch):
    calls = []

    def run(source, **kwargs):
        calls.append(kwargs)
        return {"complete": True}

    monkeypatch.setattr(demo, "run_case", run)
    output = tmp_path / "published.json"
    assert (
        demo.main(["--source", "unused", "--output", str(output), "--publish-during-serving"]) == 0
    )
    assert calls == [
        dict(reference_mode="drift", serving_plan=LiveServingPlan(), publish_on_drain=True)
    ]
    with pytest.raises(FileExistsError):
        demo.main(["--source", "unused", "--output", str(output), "--publish-during-serving"])
    with pytest.raises(SystemExit):
        demo.main(
            [
                "--source",
                "unused",
                "--output",
                str(tmp_path / "bad.json"),
                "--publish-during-serving",
                "--serving-during-updates",
            ]
        )


@pytest.fixture(scope="module")
def native_refusal():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(demo.context, "BATCHES_PER_PHASE", 2)
        result = demo.run_case(
            Path("artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json"),
            reference_mode="drift",
            serving_plan=LiveServingPlan(6, (1,)),
            publish_on_drain=True,
        )
    if os.environ.get("DRAINED_PUBLICATION_NATIVE_REPORT"):
        Path(os.environ["DRAINED_PUBLICATION_NATIVE_REPORT"]).write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf8"
        )
    assert result["complete"], result.get("error")
    return result


def test_genuine_native_guard_refusal_preserves_spent_work_old_actor_and_tail(native_refusal):
    r = native_refusal
    assert r["training_work"] == dict(
        attempted_updates=2, attempted_rows=12, returned_updates=2, unreturned_work_unknown=False
    )
    row = r["drain_publication"]
    assert row["tick"] == 3 and row["wake_updates"] == 2 and row["pending_predictions"] == 0
    assert row["completed"] and not row["accepted"] and not row["publication_unknown"]
    assert r["feedback_invalidations"] == [] and "promotion_receipt" not in r
    assert [v["phase"] for v in r["publications"]] == ["before", "stream-rejected"]
    assert r["stream"]["opportunities"] == 6 and r["stream"]["pending_unapplied"] == 0
    assert r["sharing_snapshot"]["admitted_updates"] == r["budget"]["updates_completed"] == 2
    assert r["action_snapshot"]["progress"]["decisions_attempted"] == 6
    assert all(
        p["generation"] == 0 and p["actor_version"] == "qualified-96"
        for v in r["live_serving"]["records"]
        for p in v["requests"]
    )
    assert r["source_unchanged"] and r["numpy_rng_unchanged"] and r["torch_rng_unchanged"]
    assert r["final_examples"] == r["reserved_probe_updates"] == r["replay_examples"] == 0
