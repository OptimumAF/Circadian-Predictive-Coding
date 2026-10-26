"""Fixed-mode plumbing, phase matching and paid evidence; scripted reports only."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import context_head_live as live
from scripts import context_reference_comparison as comparison
from scripts import run_bp_ticket_serving as cli
from src.core.controller_objective import ControllerObjective
from src.core.reference_controller import ReferencePolicy

SOURCE = Path("artifacts/runs/context-head-control-20261023/actual.json")
PERIODIC = Path("artifacts/runs/context-head-live-20261023/actual.json")


@pytest.fixture(scope="module")
def accepted():
    return json.loads(PERIODIC.read_bytes())


def scripted_case(accepted, mode):
    """Literal metadata/readout fixture, not an executed alternate controller."""
    r = deepcopy(accepted)
    if mode == "periodic":
        return r
    n = int(mode == "drift")
    r["reference_policy"] = asdict(ReferencePolicy(mode))
    r["training_work"] = dict(attempted_updates=n, attempted_rows=3 * n, returned_updates=n)
    r["budget"]["updates"] = n
    r["training_complete"] = False
    r["stream"].update(completed_updates=n, pending_unapplied=2 - n)
    r["native_calls"] = r["native_calls"][:n]
    r["applied_updates"] = r["applied_updates"][:n]
    r["episodes"] = []
    for k in (
        "guard_report",
        "drain_publication",
        "promotion_receipt",
        "feedback_invalidations",
        "rollback",
        "rollback_generation",
        "rollback_equal_source",
    ):
        r.pop(k, None)
    for row in r["live_serving"]["records"]:
        if row["policy_version"] == "v1" and row["concept"] == "legacy":
            row["episode"]["success"] = False
        for q in row["requests"]:
            q.update(generation=0, actor_version="qualified-12", native_updates=12)
    r["typed_work"] = dict.fromkeys(("attempts", "returns", "readouts", "typed_attempts"), 46)
    r["elapsed_seconds"] = 1.5
    r["memory"]["peak_bytes"] = 40 * 1024**2
    return r


@pytest.mark.parametrize("mode", [True, None, [], "invalid"])
def test_should_refuse_invalid_reference_before_source_sampler_models(mode, monkeypatch):
    monkeypatch.setattr(live, "_source_bytes", lambda p: pytest.fail("source entered"))
    monkeypatch.setattr(live, "ProcessRssSampler", lambda **kw: pytest.fail("sampler entered"))
    with pytest.raises(ValueError):
        live.run_case("unused", reference_mode=mode)


@pytest.mark.parametrize("mode", ["no_learning", "drift", "budget_aware", "periodic"])
def test_should_distinguish_finite_reference_execution_from_planned_training(mode, monkeypatch):
    seen = []

    def schedule(source, actor, batches, budget, choice, report, *args, **kwargs):
        seen.append(choice)
        report["stream"] = dict(pending_unapplied=2, completed_updates=0)
        return SimpleNamespace(applied_updates=[]), 14

    monkeypatch.setattr(cli, "_reference_schedule", schedule)
    r = live.run_case(SOURCE, reference_mode=mode)
    assert seen == [mode] and not r["training_complete"]
    assert r["budget"]["updates"] == 0 and r["stream"]["pending_unapplied"] == 2
    assert r["complete"] is (mode != "periodic")
    if mode == "periodic":
        assert "planned updates" in r["error"]
    assert r["source_unchanged"]


def test_should_read_accepted_periodic_and_same_scenario_window(accepted):
    assert comparison.task_window(accepted) == dict(old=4, new=2, updates=2, rows=9, pending=0)
    assert comparison.task_window(scripted_case(accepted, "no_learning")) == dict(
        old=4, new=1, updates=0, rows=0, pending=2
    )


def test_should_use_phase_matched_no_learning_counts_and_original_objective(accepted):
    cases = [scripted_case(accepted, m) for m in comparison.MODES]
    # Earlier live failures and conditional rollback aren't substituted for the
    # same task/scenario window. Full cost and violation probes still include them.
    cases[1]["live_serving"]["records"][0]["episode"]["success"] = False
    scores = comparison.score_cases(cases)
    assert [r["mode"] for r in scores] == list(comparison.MODES)
    assert [r["utility"]["normalized_new_gain"] for r in scores] == [0, 0.5, 0, 0]
    assert all(
        r["measures"]["new_before"] == 1 and r["measures"]["old_before"] == 4 for r in scores
    )
    assert all(r["utility"]["normalized_old_drop"] == 0 for r in scores)
    assert all(r["utility"]["objective_id"] == ControllerObjective().objective_id for r in scores)
    assert scores[1]["measures"]["execution_seconds"] == accepted["elapsed_seconds"]


@pytest.mark.parametrize(
    "fault",
    [
        "incomplete",
        "unknown",
        "role",
        "context",
        "scenario",
        "request",
        "attempt",
        "pending",
        "native",
        "settings",
    ],
)
def test_should_refuse_incomplete_or_changed_work_before_utility(accepted, fault):
    cases = [scripted_case(accepted, m) for m in comparison.MODES]
    r = cases[0]
    if fault == "incomplete":
        r["complete"] = False
    elif fault == "unknown":
        r["live_serving"]["unreturned_work_unknown"] = True
    elif fault == "role":
        r["live_serving"]["records"][0]["role"] = "final_test"
    elif fault == "context":
        r["live_serving"]["records"][0]["policy_version"] = "v2"
    elif fault == "scenario":
        r["live_serving"]["records"][12]["scenario"] = "ordinary"
    elif fault == "request":
        r["live_serving"]["records"][0]["requests"][0]["complete"] = False
    elif fault == "attempt":
        r["training_work"]["attempted_updates"] = True
    elif fault == "pending":
        r["stream"]["pending_unapplied"] = 0
    elif fault == "native":
        r["native_calls"] = [dict(complete=False)]
    else:
        r["reference_policy"]["reserved_updates"] = 0
    with pytest.raises(ValueError):
        comparison.score_cases(cases)


def test_should_run_only_missing_controls_and_keep_periodic_bytes(accepted, monkeypatch):
    calls = []

    def run(path, *, reference_mode):
        calls.append(reference_mode)
        return scripted_case(accepted, reference_mode)

    monkeypatch.setattr(live, "run_case", run)
    before = PERIODIC.read_bytes()
    result = comparison.run_comparison(SOURCE, PERIODIC)
    assert result["complete"], result.get("error")
    assert calls == list(comparison.MISSING) and not result["pending_modes"]
    assert PERIODIC.read_bytes() == before
    assert result["reused_periodic_work"] == dict(updates=2, rows=9, mistakes=2)
    assert result["cases"][0] == accepted


@pytest.mark.parametrize("fault", ["incomplete", "malformed", "mode"])
def test_should_keep_paid_failed_case_and_stop_following_admission(accepted, monkeypatch, fault):
    calls = []

    def run(path, *, reference_mode):
        calls.append(reference_mode)
        r = scripted_case(accepted, reference_mode)
        if fault == "incomplete":
            r.update(complete=False, error="paid failure")
        elif fault == "malformed":
            r["training_work"]["attempted_updates"] = 1
        else:
            r["reference_policy"]["mode"] = "periodic"
        return r

    monkeypatch.setattr(live, "run_case", run)
    result = comparison.run_comparison(SOURCE, PERIODIC)
    assert not result["complete"] and calls == ["no_learning"]
    assert len(result["cases"]) == 2 and result["scores"] == []
    assert result["pending_modes"] == list(comparison.MISSING)


@pytest.mark.parametrize("fault", ["corrupt", "replace"])
def test_should_refuse_corrupted_or_equal_replaced_preceding_case(accepted, monkeypatch, fault):
    original = comparison.paid_report_guard
    rows = []
    calls = []

    def guard(row):
        rows.append(row)
        return original(row)

    def run(path, *, reference_mode):
        calls.append(reference_mode)
        if fault == "corrupt":
            rows[0]["prior_training"]["rows"] = 0
        else:
            rows[0]["prior_training"] = deepcopy(rows[0]["prior_training"])
        return scripted_case(accepted, reference_mode)

    monkeypatch.setattr(comparison, "paid_report_guard", guard)
    monkeypatch.setattr(live, "run_case", run)
    result = comparison.run_comparison(SOURCE, PERIODIC)
    assert not result["complete"] and calls == ["no_learning"] and not result["scores"]
    assert "corrupted" in result["error"]
    assert result["cases"][0]["prior_training"] == accepted["prior_training"]


@pytest.mark.parametrize("which", ["source", "periodic"])
def test_should_refuse_replaced_inputs_before_any_control(tmp_path, monkeypatch, which):
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    monkeypatch.setattr(live, "run_case", lambda *a, **k: pytest.fail("control started"))
    r = comparison.run_comparison(
        bad if which == "source" else SOURCE, bad if which == "periodic" else PERIODIC
    )
    assert not r["complete"] and "replaced" in r["error"] and r["cases"] == []


def test_should_optin_cli_modes_required_inputs_and_preserve_output(tmp_path, monkeypatch):
    calls: list[Any] = []

    def compare(a, b):
        calls.append((a, b))
        return dict(complete=True)

    def run(a, **kw):
        calls.append((a, kw))
        return dict(complete=True)

    monkeypatch.setattr(comparison, "run_comparison", compare)
    monkeypatch.setattr(live, "run_case", run)
    out = tmp_path / "report.json"
    argv = [
        "--source",
        "unused",
        "--output",
        str(out),
        "--context-controller-comparison",
        "--periodic-report",
        "saved",
    ]
    assert cli.main(argv) == 0
    with pytest.raises(FileExistsError):
        cli.main(argv)
    assert len(calls) == 1
    assert (
        cli.main(
            [
                "--source",
                "unused",
                "--output",
                str(tmp_path / "live.json"),
                "--context-head-serving-live",
                "--context-live-reference",
                "drift",
            ]
        )
        == 0
    )
    assert calls[-1][1] == dict(reference_mode="drift")
    for extra in (
        ["--context-controller-comparison"],
        ["--periodic-report", "saved"],
        ["--context-live-reference", "drift"],
    ):
        with pytest.raises(SystemExit):
            cli.main(["--source", "unused", "--output", str(tmp_path / "bad"), *extra])
