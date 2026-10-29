"""Actual command/stream/controller/actor on literal data; no numerical or physics work."""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

from scripts import minigrid_affine_serving as app
from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters.minigrid_context_linear_learner import ContextLinearLearner
from src.app.serving_promotion import PromotableActor
from src.app.actor_shadow import ActorShadowRuntime
from src.app.toy_execution_budget import ToyBudgetSession
from src.core.learner_ports import TrainingDiagnostic
from tests.test_arrived_cue_catalogue import episode
from test_affine_reference_readout import literal_state
from test_context_linear_causal_games import PhaseEnvironment
from test_minigrid_learning_pilot import Environment

FLAG = "--example-affine-published"
WORK = dict(
    owners=0,
    snapshots=0,
    trains=0,
    train_rows=0,
    predicts=0,
    predict_rows=0,
    episodes=0,
    steps=0,
    source_reads=0,
    preflights=0,
    forbidden=0,
)


def invoke(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["affine", *arguments])
    cli.main()


@pytest.fixture
def case(monkeypatch):
    work: dict[str, Any] = dict(time=0.0, trace=[], paid=[], owners=[])
    original_init = ContextLinearLearner.__init__
    original_snapshot = ContextLinearLearner.snapshot_state
    original_paid = app.paid_report_guard
    original_budget = ToyBudgetSession.__init__
    packets = tuple(
        episode(f"minigrid-source-{context}-{cue}", color, context=context, action=2)
        for context in ("A", "B", "C_train")
        for cue, color in (("blue", 2), ("red", 0))
    )
    work["packets"] = packets

    def forbidden(*args, **kwargs):
        WORK["forbidden"] += 1
        raise AssertionError("genuine solver, forward or native physics forbidden")

    def empty(owner):
        WORK["owners"] += 1
        assert WORK["owners"] <= 160
        work["owners"].append(owner)
        work["trace"].append("model")
        original_init(owner)

    def snapshot(owner):
        WORK["snapshots"] += 1
        assert WORK["snapshots"] <= 6000
        return original_snapshot(owner)

    def train(owner, x, y):
        WORK["trains"] += 1
        WORK["train_rows"] += len(x)
        assert WORK["trains"] <= 48 and WORK["train_rows"] <= 432
        assert x.shape == (9, 296) and y.shape == (9, 1)
        work.setdefault("train_inputs", []).append((x.copy(), y.copy()))
        owner._state = literal_state(owner._state.completed_updates + 1)
        if work.get("train_error"):
            raise RuntimeError("scripted native fit lost return")
        if work.get("train_late"):
            work["time"] = 150.0
        return TrainingDiagnostic("literal_train_no_solver", 0.0)

    def predict(owner, x):
        WORK["predicts"] += 1
        WORK["predict_rows"] += len(x)
        assert WORK["predicts"] <= 128 and WORK["predict_rows"] <= 1152
        result = np.full((len(x), 7), 1 / 7, dtype=np.float64)
        if owner._state.completed_updates and not work.get("guard_reject"):
            result[:] = 0
            result[:, 2] = 1
        if work.get("guard_reject") and owner._state.completed_updates:
            result[:] = 0
            result[:, 6] = 1
        elif work.get("guard_reject"):
            result[:] = 0
            result[:, 2] = 1
        callback = work.get("on_predict")
        if callback:
            callback(owner, x)
        return result

    class LiteralEnvironment(PhaseEnvironment):
        def __init__(self, *args, **kwargs):
            WORK["episodes"] += 1
            assert WORK["episodes"] <= 32
            super().__init__(*args, **kwargs)

        def step(self, action):
            WORK["steps"] += 1
            assert WORK["steps"] <= 64
            return super().step(action)

    def preflight():
        WORK["preflights"] += 1
        work["trace"].append("preflight")
        if work.get("missing"):
            raise RuntimeError("literal missing optional runtime")

    def budget_init(owner, *args, **kwargs):
        work["trace"].append("budget")
        original_budget(owner, *args, **kwargs)
        work.setdefault("budgets", []).append(owner)

    def read(path, *, expected_sha256):
        WORK["source_reads"] += 1
        work["trace"].append("source")
        assert expected_sha256 == app.TRAIN_SHA and path == Path("literal-bundled")
        if work.get("source_error"):
            raise ValueError("source byte digest mismatch")
        return work.get("source_override", packets)

    def paid(value):
        work["paid"].append(value)
        callback = work.get("on_paid")
        result = original_paid(value)
        if callback:
            callback(value)
        return result

    monkeypatch.setattr(ContextLinearLearner, "__init__", empty)
    monkeypatch.setattr(ContextLinearLearner, "snapshot_state", snapshot)
    monkeypatch.setattr(ContextLinearLearner, "train_batch", train)
    monkeypatch.setattr(ContextLinearLearner, "predict", predict)
    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", LiteralEnvironment)
    monkeypatch.setattr(Environment, "work", work)
    monkeypatch.setattr(app, "_require_minigrid", preflight)
    monkeypatch.setattr(
        app, "example_affine_training", lambda: nullcontext(Path("literal-bundled"))
    )
    monkeypatch.setattr(cli, "read_training_sources", read)
    monkeypatch.setattr(app, "monotonic", lambda: work["time"])
    monkeypatch.setattr(app, "paid_report_guard", paid)
    monkeypatch.setattr(ToyBudgetSession, "__init__", budget_init)
    return work


def test_should_run_actual_command_to_publication_public_actor_and_two_literal_games(
    case, monkeypatch, tmp_path, capsys
):
    target = tmp_path / "published.json"
    before = deepcopy(case["packets"])
    invoke(monkeypatch, [FLAG, "--output", str(target)])
    result = json.loads(target.read_text())
    assert json.loads(capsys.readouterr().out) == result
    assert result["status"] == "complete" and result["preservation"] is True
    assert result["summary"] == dict(episodes=2, successes=2)
    assert result["training"]["completed_updates"] == 2
    assert result["candidate"]["native_rows"] == 18
    assert result["publication"]["generation"] == 1
    assert result["guards"]["decision"]["accepted"]
    assert (
        result["work"]["guard_predictions_returned"] == 4
        and result["work"]["guard_rows_returned"] == 36
    )
    assert [g["serving"]["actor_version"] for g in result["games"]] == ["c0", "c0"]
    assert all(e.closed and e.actions == [2, 2] for e in case["environments"])
    assert case["packets"] == before and case["trace"][:4] == [
        "preflight",
        "budget",
        "budget",
        "source",
    ]
    assert [r["arrived_at"] for r in result["candidate"]["receipts"]] == [1, 2]
    assert result["budgets"]["train"]["updates"] == 2
    assert result["budgets"]["readout"]["updates"] == result["budgets"]["readout"]["replay"] == 0
    assert all(result["budgets"][name]["original_controls"] for name in ("train", "readout"))
    assert case["owners"][0].snapshot_state().completed_updates == 0


def test_should_describe_without_runtime_source_budget_or_model_work(case, monkeypatch, capsys):
    invoke(monkeypatch, [FLAG, "--describe"])
    description = json.loads(capsys.readouterr().out)
    assert description == app.describe_published_workflow()
    assert description["fit_calls"] == 2 and description["fit_rows"] == 18
    assert case["trace"] == [] and not case["owners"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--source", "foreign.json"],
        ["--retained-source", "foreign.json"],
        ["--fit-updates", "2"],
        ["--fit-seconds", "1"],
        ["--cue-relative"],
        ["--train-a-rollout"],
        ["--example-affine-play"],
        ["--context-linear-reference", "periodic"],
        ["--describe", "--output", "unused.json"],
    ],
)
def test_should_refuse_other_modes_and_overrides_before_work(case, monkeypatch, arguments):
    with pytest.raises(SystemExit) as caught:
        invoke(monkeypatch, [FLAG, *arguments])
    assert caught.value.code == 2 and case["trace"] == []


def test_should_refuse_existing_output_before_runtime_and_keep_bytes(case, monkeypatch, tmp_path):
    target = tmp_path / "existing.json"
    target.write_text("unchanged")
    with pytest.raises(FileExistsError):
        invoke(monkeypatch, [FLAG, "--output", str(target)])
    assert target.read_text() == "unchanged" and case["trace"] == []


def test_should_save_preflight_failure_with_no_source_model_or_budget(
    case, monkeypatch, tmp_path, capsys
):
    case["missing"] = True
    target = tmp_path / "missing.json"
    with pytest.raises(SystemExit) as caught:
        invoke(monkeypatch, [FLAG, "--output", str(target)])
    result = json.loads(target.read_text())
    assert caught.value.code == 2 and result == json.loads(capsys.readouterr().out)
    assert result["status"] == "failed" and result["phase"] == "preflight"
    assert result["budgets"] is result["candidate"] is result["training"] is None
    assert case["trace"] == ["preflight"] and not case["owners"]


@pytest.mark.parametrize("fault", ["source_error", "order", "role", "length"])
def test_should_refuse_bad_source_before_model_construction(case, fault):
    if fault == "source_error":
        case[fault] = True
    else:
        packets = list(case["packets"])
        if fault == "order":
            packets.reverse()
        elif fault == "role":
            packets[0] = replace(packets[0], role="final_test")
        else:
            packets = packets[:4]
        case["source_override"] = tuple(packets)
    result = app.run_published_workflow()
    assert result["status"] == "failed" and result["phase"] == "source"
    assert not case["owners"] and not result["games"]


@pytest.mark.parametrize("fault", ["train_error", "train_late"])
def test_should_preserve_paid_or_unknown_training_prefix_without_guards_or_games(case, fault):
    case[fault] = True
    result = app.run_published_workflow()
    assert result["status"] == "failed" and result["phase"] == "train"
    assert result["work"]["train_work_may_be_unobserved"]
    assert not result["games"] and result["guards"] is None and result["summary"] is None
    if fault == "train_late":
        assert (
            result["candidate"]["native_updates"] == 1 and result["candidate"]["native_rows"] == 9
        )
        assert result["budgets"]["train"]["updates"] == 1
    else:
        assert "error" in result["candidate"] and result["committed_receipts"] == []
        assert result["budgets"]["train"]["updates"] == 0
        assert case["owners"][2]._state.completed_updates == 1


def test_should_keep_guard_rejection_and_full_training_cost_without_publishing(case):
    case["guard_reject"] = True
    result = app.run_published_workflow()
    assert result["status"] == "failed" and result["phase"] == "guard"
    assert not result["guards"]["decision"]["accepted"]
    assert result["candidate"]["native_rows"] == 18 and result["budgets"]["train"]["updates"] == 2
    assert result["publication"] is None and not case.get("environments")


@pytest.mark.parametrize("fault", ["step_error", "close_error"])
def test_should_preserve_failed_game_prefix_and_close_without_second_game(case, fault):
    case[fault] = True
    result = app.run_published_workflow()
    assert result["status"] == "failed" and result["phase"] == "game-blue"
    assert result["failed_game"]["cue"] == "blue" and result["summary"] is None
    assert case["environments"][0].closed and len(case["environments"]) == 1
    assert result["publication"]["generation"] == 1 and result["candidate"]["native_updates"] == 2
    assert result["failed_game"]["report"]["serving"]["actions_attempted"] >= 1


@pytest.mark.parametrize("target", ["training", "guards", "publication", "game"])
def test_should_refuse_corrupted_or_equal_replaced_paid_evidence_before_more_work(case, target):
    selected = []

    def on_paid(value):
        matches = (
            target == "training"
            and "applied_updates" in value
            or target == "guards"
            and "candidate_revision" in value
            or target == "publication"
            and "previous_version" in value
            or target == "game"
            and "serving" in value
        )
        if matches:
            selected.append(value)

    def mutate(environment):
        if target == "game" and len(case["environments"]) != 2:
            return
        if target == "training":
            selected[0]["training_examples"] += 1
        elif target == "guards":
            selected[0]["decision"] = deepcopy(selected[0]["decision"])
        elif target == "publication":
            selected[0]["generation"] += 1
        else:
            selected[0]["episodes"] = deepcopy(selected[0]["episodes"])

    case["on_paid"] = on_paid
    case["on_environment"] = mutate
    result = app.run_published_workflow()
    assert result["status"] == "failed" and result["summary"] is None
    assert (
        "paid" in result["reason"]
        or "changed" in result["reason"]
        or "replaced" in result["reason"]
    )
    assert result["training"]["training_examples"] == 18
    assert len(result["games"]) == (1 if target == "game" else 0)
    assert all(e.closed for e in case["environments"])


def test_should_keep_original_default_descriptor_dispatch(case, monkeypatch, capsys):
    invoke(monkeypatch, ["--describe"])
    result = json.loads(capsys.readouterr().out)
    assert result != app.describe_published_workflow() and case["trace"] == []


def test_should_refuse_actor_mutation_from_guard_callback_before_publication(case):
    def corrupt(owner, x):
        if len(x) == 9:
            case["owners"][1]._state = literal_state(2)

    case["on_predict"] = corrupt
    result = app.run_published_workflow()
    assert result["status"] == "failed" and result["phase"] == "guard"
    assert result["publication"] is None and not result["games"]


def test_should_call_actual_public_actor_and_supply_only_history_rows(case, monkeypatch):
    actual = PromotableActor.predict
    observed = []

    def predict(owner, x):
        observed.append((owner, x.copy()))
        return actual(owner, x)

    monkeypatch.setattr(PromotableActor, "predict", predict)
    result = app.run_published_workflow()
    assert result["status"] == "complete" and len(observed) == 4
    assert all(x.shape == (1, 296) for _, x in observed)
    assert all(owner is observed[0][0] for owner, _ in observed)


def test_should_report_failed_memory_setup_without_source_or_model_work(case, monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("literal memory sampling unavailable")

    monkeypatch.setattr(app, "ProcessRssSampler", unavailable)
    result = app.run_published_workflow()
    assert result["status"] == "failed" and result["budgets"] is None
    assert "sampling unavailable" in result["reason"] and case["trace"] == ["preflight"]


def test_should_refuse_changed_original_budget_after_one_paid_action(case):
    def change_budget(environment):
        case["budgets"][0].started_at = 1.0

    case["on_step"] = change_budget
    result = app.run_published_workflow()
    assert result["status"] == "failed" and result["phase"] == "game-blue"
    assert result["failed_game"]["report"]["serving"]["actions_returned"] == 1
    assert len(case["environments"]) == 1 and case["environments"][0].closed
    assert result["budgets"]["train"]["updates"] == 2 and result["candidate"]["native_rows"] == 18
    assert not result["budgets"]["readout"]["original_controls"] and result["summary"] is None


def test_should_keep_completed_game_prefix_but_fail_a_lost_final_candidate_read(case, monkeypatch):
    run = app._run_ready_workflow
    snapshot = ActorShadowRuntime.candidate_snapshot
    observed = {}
    complete_reads = 0

    def ready(report):
        observed["report"] = report
        return run(report)

    def read(owner):
        nonlocal complete_reads
        if observed["report"]["status"] == "complete":
            complete_reads += 1
            if complete_reads == 2:
                raise ValueError("lost final candidate readback")
        return snapshot(owner)

    monkeypatch.setattr(app, "_run_ready_workflow", ready)
    monkeypatch.setattr(ActorShadowRuntime, "candidate_snapshot", read)
    result = app.run_published_workflow()
    assert complete_reads == 2 and result["status"] == "failed" and result["summary"] is None
    assert not result["preservation"] and "lost final" in result["capture_error"]
    assert len(result["games"]) == 2 and all(g["status"] == "complete" for g in result["games"])
    assert len(result["committed_receipts"]) == result["budgets"]["train"]["updates"] == 2
