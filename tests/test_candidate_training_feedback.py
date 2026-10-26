"""Causal TRAIN-only candidate diagnostics through the original schedule."""

import json
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts import context_head_live as live
from scripts import run_bp_ticket_serving as cli
from scripts.candidate_training_feedback import read_candidate_feedback
from scripts.context_head_probe import head_states
from scripts.context_head_serving import prepare_data
from scripts.run_ticket_workflow import CONTEXT_FEATURES
from src.adapters.context_perceptron import ContextPerceptronLearner
from src.adapters.context_serving import build_context
from src.app.actor_shadow import StableActor
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.contextual_controller import ContextualChoice
from src.core.decision_features import DecisionFeatureSpec
from src.core.ticket_workflow import CATALOG

SOURCE = Path("artifacts/runs/context-head-control-20261023/actual.json")


def predict(state, features):
    learner = build_context(state)
    learner.restore_state(state)
    return learner.predict(features)


def setup():
    state = head_states(json.loads(SOURCE.read_bytes()))[0]
    source = ContextPerceptronLearner(state.policy)
    source.restore_state(state)
    actor = StableActor(source, version="qualified-12")
    batches, _ = prepare_data(DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG))
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=2, max_wall_seconds=30), monotonic
    )
    return state, source, actor, batches, budget


@pytest.mark.parametrize("flag", [None, 0, 1, "yes"])
def test_should_refuse_nonboolean_optin_before_io(flag, monkeypatch):
    monkeypatch.setattr(live, "_source_bytes", lambda p: pytest.fail("source entered"))
    with pytest.raises(ValueError):
        live.run_case("unused", reference_mode="drift", candidate_feedback=flag)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        dict(reference_mode="periodic"),
        dict(reference_mode="drift", contextual_policy=ContextualChoice("wait", "wake1")),
    ],
)
def test_should_refuse_changed_feedback_for_other_policies_before_io(kwargs, monkeypatch):
    monkeypatch.setattr(live, "_source_bytes", lambda p: pytest.fail("source entered"))
    with pytest.raises(ValueError):
        live.run_case("unused", candidate_feedback=True, **kwargs)


@pytest.mark.parametrize("arrived,completed", [(0, 0), (2, 2)])
def test_should_never_open_candidate_or_packet_before_labels_or_after_drain(arrived, completed):
    class Poison:
        def __getitem__(self, key):
            pytest.fail("TRAIN packet entered")

    result = read_candidate_feedback(
        None,
        Poison(),
        arrived=arrived,
        completed=completed,
        tick=2,
        label_delay=1,
        actor_version="original",
        predict=None,
        paid={},
        require_input=lambda: pytest.fail("budget entered"),
    )
    assert result is None


@pytest.mark.parametrize("role", ["final_test", "probe_train", "inner_guard"])
def test_should_refuse_nontrain_roles_without_entering_model(role):
    with pytest.raises(ValueError, match="ordinary TRAIN"):
        read_candidate_feedback(
            None,
            [SimpleNamespace(role=role)],
            arrived=1,
            completed=0,
            tick=1,
            label_delay=1,
            actor_version="original",
            predict=None,
            paid={},
            require_input=lambda: None,
        )


def test_should_detect_old_packet_regression_after_new_update_with_causal_metadata():
    state, source, actor, batches, budget = setup()
    report: dict[str, Any] = {}
    runtime, _ = cli._reference_schedule(
        source,
        actor,
        batches,
        budget,
        "drift",
        report,
        budget.before_final,
        lambda: None,
        candidate_predict=predict,
    )
    paid = report["candidate_feedback"]
    assert (
        paid["attempts"] == paid["returns"] == 2
        and paid["rows_attempted"] == paid["rows_returned"] == 9
    )
    assert [
        (
            r["index"],
            r["label_arrived_at"],
            r["read_at"],
            r["wake_updates"],
            r["native_updates"],
            r["mistakes"],
        )
        for r in paid["records"]
    ] == [(0, 1, 1, 0, 12, 1), (1, 2, 2, 1, 13, 1)]
    assert [e["action_id"] for e in report["controller_events"]] == ["wait", "wake1", "wake1"]
    assert all(r["complete"] and r["candidate_version"] == "c0" for r in paid["records"])
    assert (
        runtime.candidate_snapshot().state.completed_updates == 14 and budget.updates_completed == 2
    )
    assert source.snapshot_state() == actor.snapshot_state().state == state
    assert report["stream"]["pending_unapplied"] == 0


@pytest.mark.parametrize("fault", ["raise", "invalid", "mutate"])
def test_should_retain_attempt_and_stop_before_training_on_invalid_diagnostic(fault):
    state, source, actor, batches, budget = setup()
    report: dict[str, Any] = {}

    def bad(snapshot, features):
        if fault == "raise":
            raise ValueError("lost diagnostic return")
        if fault == "invalid":
            return np.zeros((len(features), 6))
        p = predict(snapshot, features)
        object.__setattr__(snapshot, "completed_updates", 999)
        return p

    with pytest.raises(ValueError):
        cli._reference_schedule(
            source,
            actor,
            batches,
            budget,
            "drift",
            report,
            budget.before_final,
            lambda: None,
            candidate_predict=bad,
        )
    assert (
        budget.updates_completed == 0
        and source.snapshot_state() == actor.snapshot_state().state == state
    )
    paid = report["candidate_feedback"]
    assert paid["attempts"] == 1 and paid["rows_attempted"] == 3 and paid["rows_returned"] == 0
    assert paid["returns"] == int(fault != "raise") and not paid["records"][0]["complete"]
    assert "error" in paid["records"][0] and paid["records"][0]["seconds"] >= 0
    assert report["action_snapshot"]["progress"]["native_attempts"] == 0


def test_should_preserve_api_default_without_new_diagnostics(monkeypatch):
    seen = []

    def schedule(*args, **kwargs):
        seen.append(kwargs)
        return SimpleNamespace(applied_updates=[]), 14

    monkeypatch.setattr(cli, "_reference_schedule", schedule)
    r = live.run_case(SOURCE, reference_mode="drift")
    assert r["complete"] and "candidate_feedback" not in r and "candidate_predict" not in seen[0]


@pytest.mark.parametrize("mode,callback", [("periodic", predict), ("drift", True)])
def test_should_reject_unsupported_callback_mode_before_runtime(mode, callback, monkeypatch):
    monkeypatch.setattr(cli, "ActorShadowRuntime", lambda *a, **k: pytest.fail("runtime entered"))
    with pytest.raises(ValueError, match="candidate feedback"):
        cli._reference_schedule(
            None, None, (), None, mode, {}, lambda: None, lambda: None, candidate_predict=callback
        )


def test_should_require_and_dispatch_explicit_cli_optin(tmp_path, monkeypatch):
    seen = []

    def case(path, **kwargs):
        seen.append(kwargs)
        return dict(complete=True)

    monkeypatch.setattr(live, "run_case", case)
    for tail in ([], ["--context-head-serving-live"]):
        with pytest.raises(SystemExit):
            cli.main(
                [
                    "--source",
                    str(SOURCE),
                    "--output",
                    str(tmp_path / "bad.json"),
                    "--context-candidate-feedback",
                    *tail,
                ]
            )
        assert not (tmp_path / "bad.json").exists()
    output = tmp_path / "out.json"
    assert (
        cli.main(
            [
                "--source",
                str(SOURCE),
                "--output",
                str(output),
                "--context-head-serving-live",
                "--context-live-reference",
                "drift",
                "--context-candidate-feedback",
            ]
        )
        == 0
    )
    assert seen == [dict(reference_mode="drift", candidate_feedback=True)]
