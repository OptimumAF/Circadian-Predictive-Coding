"""Literal saved-record fixtures and scalar protocol tests, no comparison models."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import context_stream_comparison as study
from scripts import context_head_live as live
from scripts import run_bp_ticket_serving as cli
from scripts.context_training_streams import CONTEXT_TRAINING_RECIPES
from scripts.ticket_serving_timeline import serving_opportunity, serving_preview, LiveServingPlan
from src.core.reference_controller import ReferencePolicy
from test_context_live_serving import make
from test_ticket_workflow import decision

SOURCE = Path("artifacts/runs/context-head-control-20261023/actual.json")
PERIODIC = Path("artifacts/runs/context-training-recipes-20261023/actual.json")
FIT = Path("artifacts/runs/context-controller-learning-20261023/actual.json")


@pytest.fixture(scope="module")
def evidence():
    return json.loads(PERIODIC.read_bytes()), study.restore_scalar_policy(
        json.loads(FIT.read_bytes())
    )


def scripted(base, recipe, mode, fit):
    """Metadata fixture ONLY, not an executed alternative control."""
    c = deepcopy(base)
    if mode == "periodic":
        return c
    n = 0 if mode == "no_learning" else 2
    if mode == "fitted":
        c.pop("reference_policy", None)
        c["contextual_policy"] = asdict(fit)
    else:
        c["reference_policy"] = asdict(
            ReferencePolicy("drift" if mode == "candidate_drift" else mode)
        )
    c["training_work"] = dict(attempted_updates=n, attempted_rows=9 if n else 0, returned_updates=n)
    c["budget"]["updates"] = n
    c["training_complete"] = False
    c["stream"].update(completed_updates=n, pending_unapplied=6 - n)
    c["native_calls"] = c["native_calls"][:n]
    c["applied_updates"] = c["applied_updates"][:n]
    for k in (
        "guard_report",
        "promotion_receipt",
        "drain_publication",
        "candidate_state",
        "feedback_invalidations",
        "rollback",
        "rollback_equal_source",
        "rollback_generation",
    ):
        c.pop(k, None)
    c["episodes"] = []
    for row in c["live_serving"]["records"]:
        if row["policy_version"] == "v1" and row["concept"] == "legacy":
            row["episode"]["success"] = False
        for q in row["requests"]:
            q.update(generation=0, actor_version="qualified-12", native_updates=12)
    if mode == "candidate_drift":
        records = [
            dict(
                complete=True,
                role="train",
                index=i,
                label_observed_at=i,
                label_arrived_at=i + 2,
                read_at=i + 2,
                rows=3 if i == 0 else 6,
                base_actor_version="qualified-12",
                candidate_version="c0",
                wake_updates=i,
                native_updates=12 + i,
            )
            for i in (0, 1)
        ]
        c["candidate_feedback"] = dict(
            row_limit=144, attempts=2, returns=2, rows_attempted=9, rows_returned=9, records=records
        )
    return c


def test_should_preview_literal_full_schedule_without_worker_or_model():
    tasks = [
        ("v1", "invoice"),
        ("v1", "legacy"),
        ("v1", "password"),
        ("v2", "invoice"),
        ("v2", "parcel"),
        ("v1", "legacy"),
        ("v1", "invoice"),
        ("v1", "invoice"),
        ("v1", "password"),
        ("v2", "invoice"),
        ("v2", "parcel"),
        ("v1", "legacy"),
        ("v2", "legacy"),
        ("v1", "legacy"),
        ("v1", "invoice"),
        ("v2", "invoice"),
        ("v1", "password"),
        ("v1", "invoice"),
        ("v1", "password"),
        ("v2", "invoice"),
        ("v2", "parcel"),
        ("v1", "legacy"),
        ("v2", "legacy"),
        ("v1", "legacy"),
    ]
    scenarios = ["ordinary"] * 24
    for tick in (4, 12, 20):
        scenarios[tick] = "stale_once"
    for tick in (6, 14, 22):
        scenarios[tick] = "revoke_commit"
    p = study.protocol_snapshot()
    assert [(r["policy_version"], r["concept"]) for r in p["opportunities"]] == tasks
    assert [r["scenario"] for r in p["opportunities"]] == scenarios
    assert [r["tick"] for r in p["opportunities"] if r["contention"]] == [2]
    assert p["common_ticks"] == list(range(17, 23))
    assert serving_opportunity(LiveServingPlan(32, ()), 28)["scenario"] == "ordinary"


@pytest.mark.parametrize("tick", [True, -1, 24, 1.5])
def test_should_refuse_invalid_preview_tick(tick):
    with pytest.raises(ValueError):
        serving_opportunity(live.recipe_plan(), tick)


def test_should_execute_the_same_preview_using_literal_worker():
    class LiteralPolicy:
        def serve(self, request):
            words = request.query.split()
            # Why: this fixture covers both catalogs, including v1-only retire.
            rules = (
                dict(invoice="route", legacy="retire", parcel="route", password="inspect")
                if "policy_v1" in words
                else dict(invoice="inspect", legacy="defer", parcel="defer", password="inspect")
            )
            selected = (
                "open"
                if words[0] == "sealed"
                else "commit"
                if words[0] == "staged"
                else rules[words[1]]
            )
            return SimpleNamespace(
                generation=0,
                actor_version="fixture",
                native_updates=12,
                result=decision(request, selected),
            )

    plan = live.recipe_plan()
    timeline, _, _ = make(plan, LiteralPolicy())
    try:
        for tick in range(24):
            with timeline.opportunity(tick):
                pass
        rows = timeline.snapshot()["records"]
        assert [(r["tick"], r["policy_version"], r["concept"], r["scenario"]) for r in rows] == [
            (p["tick"], p["policy_version"], p["concept"], p["scenario"])
            for p in serving_preview(plan)
        ]
    finally:
        timeline.close()


def test_should_validate_limited_periodic_data_without_replaying_models(evidence):
    prior, fit = evidence
    assert study.case_outcomes(
        prior["cases"][0], CONTEXT_TRAINING_RECIPES[0], "periodic", fit
    ) == dict(old=4, new=1, updates=6, rows=27, pending=0)
    assert study.case_outcomes(
        prior["cases"][1], CONTEXT_TRAINING_RECIPES[1], "periodic", fit
    ) == dict(old=4, new=0, updates=6, rows=27, pending=0)


def test_should_score_all_cases_against_same_scenario_no_learning_counts(evidence):
    prior, fit = evidence
    rows = [
        dict(recipe_id=r.recipe_id, mode=m, result=scripted(base, r, m, fit))
        for r, base in zip(CONTEXT_TRAINING_RECIPES, prior["cases"])
        for m in study.MODES
    ]
    scores = study.score_cases(rows, fit)
    assert len(scores) == 12 and all(
        s["measures"]["new_before"] == 0 and s["measures"]["old_before"] == 4 for s in scores
    )
    assert scores[1]["mode"] == "periodic" and scores[1]["utility"]["normalized_new_gain"] == 0.5
    assert scores[7]["utility"]["normalized_new_gain"] == 0
    assert all(s["measures"]["violations"] == 0 for s in scores)
    with pytest.raises(ValueError):
        study.score_cases(rows[:-1], fit)


@pytest.mark.parametrize(
    "fault",
    ["incomplete", "unknown", "scenario", "role", "request", "work", "noise", "policy", "cost"],
)
def test_should_refuse_changed_or_unknown_records_before_utility(fault, evidence):
    prior, fit = evidence
    c = deepcopy(prior["cases"][0])
    recipe = CONTEXT_TRAINING_RECIPES[0]
    if fault == "incomplete":
        c["complete"] = False
    elif fault == "unknown":
        c["live_serving"]["unreturned_work_unknown"] = True
    elif fault == "scenario":
        c["live_serving"]["records"][22]["scenario"] = "ordinary"
    elif fault == "role":
        c["live_serving"]["records"][17]["role"] = "final_test"
    elif fault == "request":
        c["live_serving"]["records"][17]["requests"][0]["complete"] = False
    elif fault == "work":
        c["training_work"]["attempted_updates"] = 7
    elif fault == "noise":
        c["synthetic_label_noise"] = [{}]
    elif fault == "policy":
        c["reference_policy"]["period"] = 3
    else:
        c["memory"]["peak_bytes"] = None
    with pytest.raises((ValueError, TypeError)):
        study.case_outcomes(c, recipe, "periodic", fit)


def test_should_run_only_missing_ten_and_never_refit_or_periodic(evidence, monkeypatch):
    prior, fit = evidence
    seen = []

    def case(path, **kwargs):
        recipe = kwargs["stream_recipe"]
        mode = (
            "fitted"
            if "contextual_policy" in kwargs
            else "candidate_drift"
            if kwargs.get("candidate_feedback")
            else kwargs["reference_mode"]
        )
        seen.append((recipe.recipe_id, mode))
        return scripted(prior["cases"][int(recipe.noisy_old_invoice)], recipe, mode, fit)

    monkeypatch.setattr(live, "run_case", case)
    r = study.run_comparison(SOURCE, PERIODIC, FIT)
    assert r["complete"], r.get("error")
    assert seen == [(p.recipe_id, m) for p in CONTEXT_TRAINING_RECIPES for m in study.MISSING]
    assert (
        r["cases"][0]["result"] == prior["cases"][0]
        and r["cases"][1]["result"] == prior["cases"][1]
    )
    assert r["cases"][0]["origin"] == "retrospective84engineering"


def test_should_retain_incomplete_paid_application_and_stop(evidence, monkeypatch):
    seen = []

    def case(*args, **kwargs):
        seen.append(1)
        return dict(complete=False, error="lost native return")

    monkeypatch.setattr(live, "run_case", case)
    r = study.run_comparison(SOURCE, PERIODIC, FIT)
    assert not r["complete"] and len(seen) == 1 and len(r["cases"]) == 3 and r["scores"] == []
    assert r["cases"][-1]["result"]["error"] == "lost native return"


@pytest.mark.parametrize("replace_equal", [False, True])
def test_should_restore_corrupted_or_replaced_preceding_paid_case(
    evidence, replace_equal, monkeypatch
):
    prior, fit = evidence
    retained = []
    original = study.paid_report_guard

    def seal(row):
        retained.append(row)
        return original(row)

    seen = []

    def case(path, **kwargs):
        seen.append(1)
        row = retained[0]["result"]
        if replace_equal:
            row["live_serving"] = deepcopy(row["live_serving"])
        else:
            row["training_work"]["attempted_rows"] = 999
        recipe = kwargs["stream_recipe"]
        return scripted(prior["cases"][0], recipe, "no_learning", fit)

    monkeypatch.setattr(study, "paid_report_guard", seal)
    monkeypatch.setattr(live, "run_case", case)
    r = study.run_comparison(SOURCE, PERIODIC, FIT)
    assert not r["complete"] and len(seen) == 1 and r["scores"] == []
    assert r["cases"][0]["result"] == prior["cases"][0]


def test_should_refuse_changed_inputs_before_admitting_models(tmp_path, monkeypatch):
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    monkeypatch.setattr(live, "run_case", lambda *a, **k: pytest.fail("model case entered"))
    r = study.run_comparison(SOURCE, bad, FIT)
    assert not r["complete"] and r["cases"] == [] and "replaced" in r["error"]


def test_should_provide_cli_preview_without_opening_source(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "_source_bytes", lambda p: pytest.fail("source opened"))
    output = tmp_path / "preview.json"
    assert (
        cli.main(["--source", "unused", "--context-stream-protocol", "--output", str(output)]) == 0
    )
    assert len(json.loads(output.read_bytes())["opportunities"]) == 24


def test_should_require_pinned_reports_and_dispatch_comparison(tmp_path, monkeypatch):
    output = tmp_path / "out.json"
    with pytest.raises(SystemExit):
        cli.main(["--source", str(SOURCE), "--context-stream-comparison", "--output", str(output)])
    assert not output.exists()
    seen: list[Path] = []

    def comparison(*paths):
        seen.extend(paths)
        return dict(complete=True)

    monkeypatch.setattr(study, "run_comparison", comparison)
    assert (
        cli.main(
            [
                "--source",
                str(SOURCE),
                "--context-stream-comparison",
                "--periodic-report",
                str(PERIODIC),
                "--fitted-report",
                str(FIT),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert seen == [SOURCE, PERIODIC, FIT]
