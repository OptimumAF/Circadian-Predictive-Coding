"""Arrival-only LS/ACK composition; pure gates, then one real CLI case."""

import json
from pathlib import Path
import sys
from time import monotonic

import pytest

from scripts import context_head_live as live
from scripts import context_head_rehearsal as source
from scripts import recurring_ticket_workflow_example as entry
from src.app.toy_execution_budget import ToyBudgetSession
from src.shared.process_memory import ProcessRssSampler


@pytest.mark.parametrize(
    "options",
    [
        dict(adaptive_serving=1),
        dict(adaptive_serving=None),
        dict(method="perceptron"),
        dict(separate_context_heads=False),
        dict(paired_serving=None),
        dict(private_adaptation=True),
        dict(replay="recent"),
        dict(record_train_fit=True),
    ],
)
def test_should_refuse_invalid_adaptive_api_before_data(options, monkeypatch):
    monkeypatch.setattr(entry, "prepare_training", lambda **kw: pytest.fail("data opened"))
    args = dict(
        method="linear_fit",
        context_return=True,
        separate_context_heads=True,
        paired_serving="joint",
        adaptive_serving=True,
    )
    args.update(options)
    with pytest.raises(ValueError):
        entry.run_example(**args)


@pytest.mark.parametrize("adaptive", [False, True])
def test_should_forward_adaptive_cli_only_when_selected_and_preserve_output(
    adaptive, tmp_path, monkeypatch
):
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return dict(complete=True)

    monkeypatch.setattr(entry, "run_example", fake)
    output = tmp_path / "out.json"
    args = [
        "--output",
        str(output),
        "--method",
        "linear_fit",
        "--context-return",
        "--separate-context-heads",
        "--paired-serving",
        "joint",
    ]
    if adaptive:
        args.append("--adaptive-serving")
    assert entry.main(args) == 0
    assert calls == [
        dict(
            method="linear_fit",
            context_return=True,
            separate_context_heads=True,
            paired_serving="joint",
            **(dict(adaptive_serving=True) if adaptive else {}),
        )
    ]
    with pytest.raises(FileExistsError):
        entry.main(args)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "flags",
    [
        [],
        ["--method", "linear_fit"],
        ["--context-return", "--separate-context-heads", "--paired-serving", "joint"],
    ],
)
def test_should_refuse_missing_adaptive_cli_prerequisites_before_output(
    flags, tmp_path, monkeypatch
):
    monkeypatch.setattr(entry, "run_example", lambda **kw: pytest.fail("model opened"))
    output = tmp_path / "out.json"
    with pytest.raises(SystemExit):
        entry.main(["--output", str(output), "--adaptive-serving", *flags])
    assert not output.exists()


def test_should_translate_owned_adaptive_mode_to_existing_drift_and_ack_ports(monkeypatch):
    calls = []
    guard = lambda: None

    def prepare(state, *args, **kwargs):
        assert kwargs["linear_fit"] is True
        assert kwargs["training_key"] == "serving_source_training"
        return state, guard

    def run(path, **kwargs):
        calls.append((path, kwargs))
        return dict(complete=True)

    monkeypatch.setattr(source, "prepare_owned_source", prepare)
    monkeypatch.setattr(live, "run_case", run)
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        budget = ToyBudgetSession(source._limits(12), monotonic)
        budget.attach_memory(memory)
        result = live.run_owned_case(
            None,
            None,
            {},
            budget,
            control="joint",
            check_boundary=guard,
            memory=memory,
            linear_fit=True,
            adaptive=True,
        )
    assert result["complete"] and result["owned_source_verified"]
    path, kwargs = calls[0]
    assert path is None and kwargs["publish_after_learning"] is False
    assert kwargs["reference_mode"] == "drift"
    assert kwargs["candidate_feedback"] is kwargs["acknowledge_correct"] is True
    assert kwargs["_linear_fit"] is True
    assert "followup_plan" not in kwargs
    assert budget.updates_completed == 0


@pytest.mark.parametrize("adaptive,linear", [(1, True), (None, True), (True, False)])
def test_should_refuse_owned_adaptive_without_exact_ls_optin(adaptive, linear, monkeypatch):
    monkeypatch.setattr(
        source, "prepare_owned_source", lambda *a, **kw: pytest.fail("source opened")
    )
    result = live.run_owned_case(
        None,
        None,
        None,
        None,
        control="joint",
        check_boundary=lambda: None,
        memory=None,
        linear_fit=linear,
        adaptive=adaptive,
    )
    assert not result["complete"]


@pytest.mark.parametrize(
    "changes",
    [
        dict(arrived_shift=True),
        dict(task_feedback=True),
        dict(publish_after_learning=True),
        dict(candidate_feedback=False),
        dict(acknowledge_correct=False),
        dict(reference_mode="periodic"),
    ],
)
def test_should_refuse_mixed_owned_adaptive_modes_before_source(changes, monkeypatch):
    monkeypatch.setattr(live, "_source_bytes", lambda _: pytest.fail("source opened"))
    args = dict(
        _linear_fit=True,
        _owned_source=(None, lambda: None),
        _memory=object(),
        stream_recipe=live.CONTEXT_PAIRED_REHEARSAL_RECIPES[1],
        reference_mode="drift",
        candidate_feedback=True,
        acknowledge_correct=True,
    )
    args.update(changes)
    with pytest.raises(ValueError):
        live.run_case(None, **args)


def test_should_execute_one_full_adaptive_ls_cli_with_real_fit_and_ack_receipts():
    output = (
        Path("artifacts/runs/context-linear-adaptive-serving-20261029")
        / f"{sys.platform}-integration.json"
    )
    status = entry.main(
        [
            "--output",
            str(output),
            "--method",
            "linear_fit",
            "--context-return",
            "--separate-context-heads",
            "--paired-serving",
            "joint",
            "--adaptive-serving",
        ]
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0 and result["complete"], result.get("error")
    child = result["paired_serving"]
    assert child["training_control"] == "arrived_candidate_error_ack_v1"
    assert child["source_unchanged"] and child["owned_source_verified"]
    progress = child["receipt_progress"]
    assert progress["complete"] and progress["consumed"] == progress["total"] == 6
    assert progress["actual_updates"] + progress["acknowledged"] == 6
    assert child["budget"]["limits"]["max_training_updates"] == 6
    assert child["budget"]["updates"] == progress["actual_updates"]
    assert child["training_work"]["returned_updates"] == progress["actual_updates"]
    assert len(child["applied_updates"]) == progress["actual_updates"]
    assert len(child["acknowledged_experiences"]) == progress["acknowledged"]
    for row in child["candidate_feedback"]["records"]:
        assert row["role"] == "train" and row["label_arrived_at"] <= row["read_at"]
    assert child["rollback_equal_source"]
    published = [e for e in child["episodes"] if e["phase"] == "published"]
    assert len(published) == 6 and all(e["episode"]["success"] for e in published)


def test_should_refuse_adaptive_owned_without_named_paired_recipe_before_copy(monkeypatch):
    from src.core import context_linear_state

    monkeypatch.setattr(
        context_linear_state, "copy_context_linear_state", lambda _: pytest.fail("state copied")
    )
    monkeypatch.setattr(live, "prepare_data", lambda *a, **kw: pytest.fail("producer opened"))
    with pytest.raises(ValueError, match="exact paired recipe"):
        live.run_case(
            None,
            _linear_fit=True,
            _owned_source=(None, lambda: None),
            _memory=object(),
            reference_mode="drift",
            candidate_feedback=True,
            acknowledge_correct=True,
        )
