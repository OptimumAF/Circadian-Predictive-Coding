"""Separate-head control through original delayed stream and permission sandbox."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts import recurring_ticket_workflow_example as study
from src.adapters.context_perceptron import (
    ContextPerceptronDecisionPredictor,
)


@pytest.fixture(scope="module")
def completed():
    return study.run_example(context_return=True, separate_context_heads=True)


def test_should_execute_original_fixed_data_receipts_and_complete_owned_heads(completed):
    r = completed
    assert r["complete"], r.get("error")
    assert (
        r["separate_context_heads"]
        and r["source_unchanged"]
        and r["actor_unchanged"]
        and r["numpy_rng_unchanged"]
    )
    assert r["training_work"] == dict(
        attempted_updates=12,
        attempted_rows=72,
        returned_updates=12,
        unreturned_work_unknown=False,
        budget_committed_updates=12,
    )
    assert len(r["applied_updates"]) == 12
    assert all(
        x["arrived_at"] == x["observed_at"] + 1 <= x["applied_at"] for x in r["applied_updates"]
    )
    assert [x["source_updates"] for x in r["checkpoints"]] == [0, 4, 8, 12]
    states = [x["state"] for x in r["checkpoints"]]
    assert [[h["completed_updates"] for h in s["heads"]] for s in states] == [
        [0, 0],
        [4, 0],
        [4, 4],
        [8, 4],
    ]
    assert [[h["training_examples"] for h in s["heads"]] for s in states] == [
        [0, 0],
        [24, 0],
        [24, 24],
        [48, 24],
    ]
    assert states[0]["heads"][1] == states[1]["heads"][1]
    assert states[1]["heads"][0] == states[2]["heads"][0]
    assert states[2]["heads"][1] == states[3]["heads"][1]
    assert all(
        sum(h["training_examples"] for h in s["heads"]) == s["training_examples"] for s in states
    )
    assert all(
        sum(h["mistake_updates"] for h in s["heads"]) == s["mistake_updates"] for s in states
    )
    assert r["budget"]["updates_completed"] == r["budget"]["limits"]["max_training_updates"] == 12
    assert r["costs"]["model_coefficient_bytes"] == 768 and r["head_routing"]["coefficients"] == 96
    assert r["costs"]["attempted_training_affine_terms"] == 72 * 8 * 6
    assert r["costs"]["retained_checkpoint_coefficient_bytes"] == 4 * 2 * 8 * 6 * 8


def test_should_keep_same_task_readouts_and_saved_training_protocol_without_selecting_on_outcomes(
    completed,
):
    r = completed
    assert len(r["episodes"]) == 24 and all(x["episode"]["complete"] for x in r["episodes"])
    for p in study.CHECKPOINTS:
        ordinary = [
            x
            for x in r["episodes"]
            if x["source_updates"] == p and x["episode"]["scenario"] == "ordinary"
        ]
        assert [(x["policy_version"], x["concept"]) for x in ordinary] == list(study.CONTEXT_TASKS)
    assert r["summary"] == study.summarize_context_return(r["episodes"])
    assert all(x["episode"]["training_updates"] == 0 for x in r["episodes"])
    assert all(
        r[k] == 0
        for k in ("replay_examples", "probe_updates", "final_examples", "external_actions")
    )
    prior = Path("artifacts/runs/context-return-stream-20261023/actual.json")
    assert (
        hashlib.sha256(prior.read_bytes()).hexdigest()
        == "581a68d8badbaed81fb6dbf85b78aab3e08798149de86448feb31936e3d7b4b0"
    )
    old = json.loads(prior.read_bytes())
    assert r["training_data"] == old["training_data"] and r["phases"] == old["phases"]
    assert (
        r["summary"]["no_learning_success_reused_from_update0"]
        == old["summary"]["no_learning_success_reused_from_update0"]
    )
    assert len(study._encode_report(r)) <= 512 * 1024


@pytest.mark.parametrize(
    "kw",
    [
        dict(separate_context_heads=1),
        dict(separate_context_heads=True),
        dict(context_return=True, separate_context_heads=True, replay="recent"),
        dict(context_return=True, separate_context_heads=True, method="bp"),
    ],
)
def test_should_refuse_unsupported_separate_head_modes_before_data(kw, monkeypatch):
    monkeypatch.setattr(study, "prepare_training", lambda **kw: pytest.fail("data entered"))
    with pytest.raises(ValueError):
        study.run_example(**kw)


def test_should_preserve_missing_typed_return_and_stop_before_any_update(monkeypatch):
    monkeypatch.setattr(
        ContextPerceptronDecisionPredictor,
        "__call__",
        lambda *a: (_ for _ in ()).throw(RuntimeError("lost typed return")),
    )
    r = study.run_example(context_return=True, separate_context_heads=True)
    assert not r["complete"] and r["training_work"]["attempted_updates"] == 0
    assert len(r["episodes"]) == 1 and not r["episodes"][0]["episode"]["complete"]
    assert r["episode_work"]["prediction_calls_attempted"] == 1
    assert r["episode_work"]["prediction_calls_returned"] == 0


def test_should_preserve_corrupted_paid_prefix_and_refuse_before_more_work(monkeypatch):
    saved = []
    original = study.run_workflow_episode

    def episode(*a, **kw):
        result = original(*a, **kw)
        saved.append(result)
        if len(saved) == 2:
            saved[0]["success"] = not saved[0]["success"]
        return result

    monkeypatch.setattr(study, "run_workflow_episode", episode)
    r = study.run_example(context_return=True, separate_context_heads=True)
    assert not r["complete"] and "paid evidence" in r["error"]
    assert len(saved) == 2 and r["training_work"]["attempted_updates"] == 0
    assert r["episodes"][0]["episode"]["success"] is False


def test_should_dispatch_cli_and_refuse_incompatible_options_before_reservation(
    tmp_path, monkeypatch
):
    seen = []

    def run(**kw):
        seen.append(kw)
        return dict(complete=True)

    monkeypatch.setattr(study, "run_example", run)
    p = tmp_path / "case.json"
    assert study.main(["--context-return", "--separate-context-heads", "--output", str(p)]) == 0
    assert seen == [dict(context_return=True, separate_context_heads=True)]
    with pytest.raises(FileExistsError):
        study.main(["--context-return", "--separate-context-heads", "--output", str(p)])
    bad = tmp_path / "bad.json"
    with pytest.raises(SystemExit):
        study.main(
            [
                "--context-return",
                "--separate-context-heads",
                "--replay",
                "coverage",
                "--output",
                str(bad),
            ]
        )
    assert not bad.exists()
