"""Literal combined native-trial procedure without constructing native models."""

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts import run_bp_ticket_serving as demo
from scripts import ticket_workflow_native as native
from scripts.ticket_serving_timeline import LiveServingPlan, FollowupTrainingPlan
from scripts.ticket_controller_streams import TicketStreamRecipe
from src.app.actor_shadow import StableActor
from src.adapters import torch_bp_record
from src.core.contextual_controller import ContextualChoice, ContextualEstimate, FittedController
from src.core.controller_objective import ControllerObjective
from src.core.learner_ports import TrainingDiagnostic
from test_ticket_workflow import decision


@dataclass(frozen=True)
class LiteralState:
    completed_updates: int = 96
    training_examples: int = 576


class LiteralLearner:
    def __init__(self, before=None, after=None):
        self.state = LiteralState()
        self.before, self.after = before, after

    def fork(self):
        other = LiteralLearner(self.before, self.after)
        other.restore_state(self.state)
        return other

    def snapshot_state(self):
        return deepcopy(self.state)

    def restore_state(self, state):
        self.state = deepcopy(state)

    def train_batch(self, x, y):
        if self.before:
            self.before(x)
        self.state = LiteralState(
            self.state.completed_updates + 1, self.state.training_examples + len(x)
        )
        if self.after:
            self.after()
        return TrainingDiagnostic("scripted_train", 0.0)

    def predict(self, x):
        return np.tile(np.ones(6) / 6, (len(x), 1))


class LiteralActor(StableActor):
    def __init__(self, source, **kw):
        super().__init__(source, version=kw["version"])
        self.configuration, self.metadata = kw["configuration"], kw["metadata"]

    def serving_snapshot(self):
        return SimpleNamespace(
            generation=0,
            actor_version=self.version,
            model_state=self.snapshot_state().state,
            configuration=self.configuration,
            metadata=deepcopy(self.metadata),
            cache={},
            last_cache_tick=0,
        )


@dataclass
class LiteralWork:
    native_forward_attempts: int = 0
    decision_returns: int = 0


class LiteralReader:
    def __init__(self, spec, state):
        self.state = deepcopy(state)
        self.pin = deepcopy(state)
        self.work = LiteralWork()

    def __call__(self, request):
        self.work.native_forward_attempts += 1
        words = request.query.split()
        chosen = None
        if words[0] == "ready" and "retire" in {c.candidate_id for c in request.allowed_candidates}:
            chosen = "inspect" if words[1] == "password" else "route"
        result = decision(request, chosen)
        self.work.decision_returns += 1
        return result

    def assert_unchanged(self):
        assert self.state == self.pin


class LiteralRng(list):
    def clone(self):
        return LiteralRng(self)


@pytest.fixture
def literal(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_bytes(b"literal-original")
    prior = dict(training_work={}, costs={}, checkpoints=[dict(state=asdict(LiteralState()))])
    monkeypatch.setattr(demo.context, "_source", lambda path: (path.read_bytes(), deepcopy(prior)))
    monkeypatch.setattr(demo.context, "bp_snapshot_from_record", lambda d: LiteralState(**d))
    monkeypatch.setattr(torch_bp_record, "bp_snapshot_from_record", lambda d: LiteralState(**d))
    monkeypatch.setattr(
        native,
        "require_torch",
        lambda: SimpleNamespace(get_rng_state=lambda: LiteralRng([0]), equal=lambda a, b: a == b),
    )
    monkeypatch.setattr(
        native, "counted_learner", lambda method, d, k, before, after: LiteralLearner(before, after)
    )
    monkeypatch.setattr(native, "snapshot_record", asdict)
    monkeypatch.setattr(native, "same_state", lambda a, b: a == b)
    monkeypatch.setattr(
        native,
        "checkpoint",
        lambda point, state: dict(
            source_updates=point, source_rows=state.training_examples, state=asdict(state)
        ),
    )
    monkeypatch.setattr(native, "readout", LiteralReader)
    monkeypatch.setattr(demo, "PromotableActor", LiteralActor)
    monkeypatch.setattr(demo, "TorchServingPredictor", lambda *a: SimpleNamespace(work={}))
    monkeypatch.setattr(demo, "ServingPromotionController", lambda *a, **kw: object())
    monkeypatch.setattr(demo, "snapshot_digest", lambda state: "a" * 64)
    monkeypatch.setattr(demo, "_sealed_cases", lambda: pytest.fail("legacy final source opened"))
    return source


@pytest.mark.parametrize("deferred", [False, True])
def test_full_scripted_trial_keeps_actual_counts_and_defers_all_readouts_when_requested(
    literal, monkeypatch, deferred
):
    if deferred:
        monkeypatch.setattr(
            native, "readout", lambda *a: pytest.fail("deferred task source opened")
        )
    r = demo.run_case(
        literal, reference_mode="periodic", policy_change=True, deferred_evaluation=deferred
    )
    assert r["complete"], r.get("error")
    assert (
        r["budget"]["limits"]["max_training_updates"] == 12
        and r["budget"]["updates_completed"] == 6
    )
    assert r["training_work"]["attempted_rows"] == 36 and len(r["applied_updates"]) == 6
    assert (
        r["stream"]["opportunities"] == 13
        and r["stream"]["arrived_labels"] == 12
        and r["stream"]["pending_unapplied"] == 6
    )
    assert r["policy_capture_complete"] is True
    assert [
        (p["opportunity"], p["policy_version"], p["applied_receipts"])
        for p in r["policy_checkpoints"]
    ] == [(0, "v1", 0), (4, "v1", 2), (8, "v2", 4), (12, "v2", 6)]
    assert r["action_snapshot"]["envelope"]["max_decisions"] == 13
    assert (
        r["action_snapshot"]["envelope"]["max_wake_updates"]
        == r["sharing_snapshot"]["limits"]["max_admitted_updates"]
        == 12
    )
    assert r["source_unchanged"] and r["numpy_rng_unchanged"] and r["torch_rng_unchanged"]
    assert r["episodes"] == [] and all(
        r[k] == 0
        for k in ("final_examples", "reserved_probe_updates", "replay_examples", "external_actions")
    )
    if deferred:
        assert r["deferred_actor_unchanged"] and r["deferred_promotions"] == 0
        assert "policy_development_episodes" not in r and "policy_summary" not in r
    else:
        assert (
            r["policy_actor_unchanged"]
            and r["policy_promotions"] == 0
            and r["policy_readout_complete"]
        )
        assert len(r["policy_development_episodes"]) == 15
        assert "ordinary_success_by_update" not in r["policy_summary"]
        assert set(r["policy_summary"]["ordinary_success_by_opportunity"]) == {0, 4, 8, 12}
        assert all(
            row["role"] == "outer_selection" and row["episode"]["training_updates"] == 0
            for row in r["policy_development_episodes"]
        )
        from scripts.recurring_ticket_workflow_example import summarize_policy_changes

        legacy_rows = [
            dict(row, source_updates=row["opportunity"]) for row in r["policy_development_episodes"]
        ]
        old_axis = summarize_policy_changes(legacy_rows)
        assert (
            old_axis["ordinary_success_by_update"]
            == r["policy_summary"]["ordinary_success_by_opportunity"]
        )


@pytest.mark.parametrize(
    "change",
    [
        {"policy_change": 1},
        {"reference_mode": None},
        {"serving_plan": LiveServingPlan()},
        {"followup_plan": FollowupTrainingPlan()},
        {"publish_on_drain": True},
        {"stream_recipe": TicketStreamRecipe("other", ("old_return",), 1)},
        {"training_context": "old_return"},
        {"counterfactual": True},
        {"base_snapshot": object()},
    ],
)
def test_invalid_policy_trial_refuses_before_source_or_model_access(change, monkeypatch):
    monkeypatch.setattr(demo.context, "_source", lambda *a: pytest.fail("source opened"))
    kw: dict[str, Any] = dict(reference_mode="periodic", policy_change=True)
    kw.update(change)
    with pytest.raises(ValueError):
        demo.run_case(Path("not-opened.json"), **kw)


def test_bound_contextual_policy_is_accepted_but_stale_full_state_is_refused(literal):
    estimates = tuple(
        ContextualEstimate(c, a, 0.0) for c in ("clean", "mistakes") for a in ("wait", "wake1")
    )
    fit = FittedController(
        "qualified-96",
        "a" * 64,
        ControllerObjective(),
        1,
        estimates,
        ContextualChoice("wait", "wait"),
    )
    r = demo.run_case(literal, policy_change=True, contextual_policy=fit, deferred_evaluation=True)
    assert r["complete"] and r["budget"]["updates_completed"] == 0
    assert [p["applied_receipts"] for p in r["policy_checkpoints"]] == [0] * 4
    for stale in (replace(fit, learner_version="other"), replace(fit, state_digest="b" * 64)):
        bad = demo.run_case(
            literal, policy_change=True, contextual_policy=stale, deferred_evaluation=True
        )
        assert not bad["complete"] and "original learner version/payload" in bad["error"]
        assert bad["training_work"]["attempted_updates"] == 0


@pytest.mark.parametrize(
    "report",
    [
        dict(policy_change=True),
        dict(policy_change=None),
        dict(policy_checkpoints=[]),
        dict(definition="policy_change_native_reference_v1"),
    ],
)
def test_legacy_final_refuses_before_clock_and_source_payload(report):
    clock = SimpleNamespace(now=lambda: pytest.fail("final clock read"))
    with pytest.raises(ValueError, match="legacy task"):
        demo._sealed_readout(report, {}, clock)
    with pytest.raises(ValueError, match="legacy task"):
        demo._case_measures(report, 0.0)


def test_lost_fifth_scripted_update_preserves_paid_prefix_and_partial_phase_history(
    literal, monkeypatch
):
    original = LiteralLearner.train_batch
    calls = []

    def fail(self, x, y):
        calls.append(1)
        if len(calls) == 5:
            if self.before:
                self.before(x)
            raise RuntimeError("lost policy native return")
        return original(self, x, y)

    monkeypatch.setattr(LiteralLearner, "train_batch", fail)
    r = demo.run_case(literal, reference_mode="periodic", policy_change=True)
    assert not r["complete"] and "lost policy native return" in r["error"]
    assert r["budget"]["updates_completed"] == 4 and r["training_work"]["attempted_updates"] == 5
    assert r["training_work"]["unreturned_work_unknown"] and len(r["applied_updates"]) == 4
    assert not r["policy_capture_complete"] and len(r["policy_checkpoints"]) == 3
    assert "policy_development_episodes" not in r


def test_bad_phase_metadata_refuses_before_native_state_decoder(literal, monkeypatch):
    original = demo.run_case(
        literal, reference_mode="periodic", policy_change=True, deferred_evaluation=True
    )
    monkeypatch.setattr(
        torch_bp_record, "bp_snapshot_from_record", lambda *a: pytest.fail("native decoder used")
    )
    for field, value in (
        ("opportunity", True),
        ("policy_version", "v1"),
        ("applied_receipts", True),
    ):
        r = deepcopy(original)
        r["policy_checkpoints"][-1][field] = value
        with pytest.raises(ValueError):
            native.policy_development_readout(None, r, lambda: None)
    for field in ("completed_updates", "training_examples"):
        r = deepcopy(original)
        state = r["policy_checkpoints"][-1]["checkpoint"]["state"]
        state[field] = float(state[field])
        with pytest.raises(ValueError):
            native.policy_development_readout(None, r, lambda: None)


def test_replaced_prior_capture_during_readout_stops_with_paid_episode_prefix(literal):
    r = demo.run_case(
        literal, reference_mode="periodic", policy_change=True, deferred_evaluation=True
    )

    def require():
        if r.get("policy_development_episodes"):
            r["policy_checkpoints"][0]["policy_version"] = "v2"

    with pytest.raises(ValueError, match="evidence changed"):
        native.policy_development_readout(None, r, require)
    assert len(r["policy_development_episodes"]) == 1 and not r["policy_readout_complete"]


def test_cli_explicit_mode_preserves_default_and_occupied_output(tmp_path, monkeypatch):
    calls = []

    def run(source, **kw):
        calls.append(kw)
        return dict(complete=True)

    monkeypatch.setattr(demo, "run_case", run)
    target = tmp_path / "policy.json"
    assert (
        demo.main(
            [
                "--source",
                "saved.json",
                "--output",
                str(target),
                "--policy-change-reference",
                "periodic",
            ]
        )
        == 0
    )
    assert calls == [dict(reference_mode="periodic", policy_change=True)]
    with pytest.raises(FileExistsError):
        demo.main(
            [
                "--source",
                "saved.json",
                "--output",
                str(target),
                "--policy-change-reference",
                "periodic",
            ]
        )
    assert len(calls) == 1
    assert demo.main(["--source", "saved.json", "--output", str(tmp_path / "default.json")]) == 0
    assert calls[-1] == {}


def test_sole_native_periodic_policy_trial():
    output = os.environ.get("NATIVE_POLICY_TRIAL_REPORT")
    if not output:
        pytest.skip("separate frozen native case after software gates")
    r = demo.run_case(
        Path("artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json"),
        reference_mode="periodic",
        policy_change=True,
    )
    Path(output).write_text(json.dumps(r, indent=2, allow_nan=False) + "\n", encoding="utf8")
    assert r["complete"], r.get("error")
    assert r["training_work"]["returned_updates"] == r["budget"]["updates_completed"] == 6
    assert r["training_work"]["attempted_rows"] == 36 and len(r["applied_updates"]) == 6
    assert (
        r["policy_capture_complete"]
        and r["policy_readout_complete"]
        and len(r["policy_development_episodes"]) == 15
    )
    assert [(p["opportunity"], p["applied_receipts"]) for p in r["policy_checkpoints"]] == [
        (0, 0),
        (4, 2),
        (8, 4),
        (12, 6),
    ]
    assert (
        r["stream"]["pending_unapplied"] == 6
        and r["candidate_checkpoint"]["state"]["completed_updates"] == 102
    )
    assert (
        r["source_unchanged"]
        and r["policy_actor_unchanged"]
        and r["numpy_rng_unchanged"]
        and r["torch_rng_unchanged"]
    )
    assert (
        r["policy_promotions"]
        == r["final_examples"]
        == r["reserved_probe_updates"]
        == r["replay_examples"]
        == r["external_actions"]
        == 0
    )
