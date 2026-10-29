"""Literal source records and scripted learners through the existing comparison."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import run_bp_ticket_serving as demo
from scripts import ticket_controller_comparison as comparison
from scripts import ticket_controller_learning as learning
from scripts.ticket_controller_streams import TicketStreamRecipe
from src.core.controller_objective import ControllerMeasures
from src.core.ticket_workflow import ACTION_IDS
from test_ticket_catalog_control import batches, catalog
from test_ticket_catalog_control import driver as driver


@pytest.fixture
def fitted_source(driver, monkeypatch, tmp_path):
    original_readout, original_measures = demo._candidate_tasks, demo._case_measures

    def readout(spec, runtime, report, require):
        original_readout(spec, runtime, report, require)
        report["reference_outcomes"]["new_after"] = int(
            report["training_context"] == "new_contexts"
            and runtime.candidate_snapshot().wake_updates > 0
        )

    monkeypatch.setattr(demo, "_candidate_tasks", readout)
    monkeypatch.setattr(
        demo,
        "_case_measures",
        lambda r, sec: replace(
            original_measures(r, sec),
            execution_seconds=1 + r["training_work"]["attempted_updates"],
            peak_rss_bytes=1024,
        ),
    )
    study = learning.run_study(driver.path, catalog=catalog(2))
    assert study["complete"], study.get("error")
    assert study["fitted_policy"]["choices"] == ("wait", "wake1")
    study = json.loads(json.dumps(study))
    study["prior_training_work"] = dict(
        attempted_updates=96, attempted_rows=576, returned_updates=96, unreturned_work_unknown=False
    )
    path = tmp_path / "literal-catalog-study.json"
    path.write_text(json.dumps(study), encoding="utf8")
    templates = list(batches())
    for index in (2, 3):
        targets = templates[index].targets.copy()
        targets[1, 0] = ACTION_IDS.index("inspect")
        templates[index] = replace(templates[index], targets=targets)
    monkeypatch.setattr(
        demo.context, "_teacher_batches", lambda spec: (None, None, tuple(templates))
    )
    recipes = (
        TicketStreamRecipe("literal-mix", ("new_contexts", "old_return"), 1),
        TicketStreamRecipe("literal-delay", ("old_return",) * 2, 2, True),
    )
    monkeypatch.setattr(comparison, "WITHHELD_RECIPES", recipes)
    train: list[tuple[dict[str, Any], dict[str, Any]]] = []
    opened: list[dict[str, Any]] = []
    run = demo.run_case

    def training(*args, **kw):
        assert not opened
        result = run(*args, **kw)
        train.append((kw, result))
        return result

    def readonly(result, report, clock):
        assert len(train) == 10 and all(r["complete"] for _, r in train)
        assert result["episodes"] == [] and "candidate_episodes" not in result
        opened.append(result)
        report["sealed_source_reads"].append(dict(at=clock.now(), role="literal-final-port"))
        return ControllerMeasures(0, 0, 2, 2, 2, 2, 0.01, 1024, 0, True)

    monkeypatch.setattr(demo, "run_case", training)
    monkeypatch.setattr(demo, "_sealed_readout", readonly)
    monkeypatch.setattr(demo, "_candidate_tasks", lambda *a: pytest.fail("early tasks"))
    driver.made.clear()
    return dict(driver=driver, path=path, study=study, train=train, opened=opened)


def test_should_admit_exact_complete_catalog_and_keep_all_discarded_costs(fitted_source):
    raw, policy, costs = comparison._frozen_fit(fitted_source["path"], catalog_fit=True)
    r = fitted_source["study"]
    assert raw == fitted_source["path"].read_bytes()
    assert policy.choices == ("wait", "wake1")
    assert costs["fit_training_updates"] == sum(
        t["native_work"]["attempted_updates"] for t in r["trials"]
    )
    assert costs["fit_training_rows"] == sum(
        t["native_work"]["attempted_rows"] for t in r["trials"]
    )
    assert costs["fit_replay_examples"] == 12
    assert costs["whole_source_catalog_work"] == r["total_catalog_work"]
    assert costs["whole_source_study_costs"] == r["costs"]
    assert costs["source_pretraining_work"] == r["prior_training_work"]
    assert costs["all_in_utility"] is None and costs["exact_fit_overhead_separable"] is False
    assert fitted_source["driver"].made == []


def test_should_match_all_control_opportunities_and_finish_before_readout(fitted_source):
    r = comparison.run_comparison(
        fitted_source["driver"].path, fitted_source["path"], catalog_fit=True
    )
    assert r["complete"], r.get("error")
    assert r["definition"] == "catalog_controller_comparison_v1"
    assert r["protocol"]["common_opportunities"] == {"literal-mix": 3, "literal-delay": 3}
    assert len(r["cases"]) == len(r["final_readouts"]) == 10
    assert len(fitted_source["train"]) == len(fitted_source["opened"]) == 10
    for kw, case in fitted_source["train"]:
        assert kw["decision_limit"] == 3
        assert case["stream"]["opportunities"] == 3
        assert case["stream"]["unarrived_labels"] == (kw["stream_recipe"].label_delay - 1)
        assert ("catalog_policy" in kw) == (case["definition"] == "deferred_catalog_stream_v1")
        assert "contextual_policy" not in kw and kw["deferred_evaluation"]
    assert r["new_controller_fits"] == r["final_training_updates"] == r["promotions"] == 0
    assert r["numpy_rng_unchanged"] and r["torch_rng_unchanged"]
    assert all(f["all_in_utility"] is None for f in r["final_readouts"])


@pytest.mark.parametrize(
    "fault",
    [
        "incomplete",
        "coverage",
        "digest",
        "catalog",
        "utility",
        "work",
        "unknown",
        "total",
        "fitted",
        "role",
        "clock",
        "metric_count",
        "monitoring",
        "preparation",
        "boolean_work",
    ],
)
def test_should_refuse_corrupt_fit_before_opening_any_candidate(fitted_source, fault):
    r = deepcopy(fitted_source["study"])
    if fault == "incomplete":
        r["complete"] = False
    elif fault == "coverage":
        r["trials"][1] = deepcopy(r["trials"][0])
    elif fault == "digest":
        r["trials"][0]["request"]["state_digest"] = "foreign"
    elif fault == "catalog":
        r["catalog"]["envelope"]["max_decisions"] = 99
    elif fault == "utility":
        r["trials"][0]["measures"]["new_after"] = 2
    elif fault == "work":
        r["trials"][1]["native_work"]["attempted_rows"] += 1
    elif fault == "unknown":
        r["trials"][1]["result"]["training_work"]["unreturned_work_unknown"] = True
    elif fault == "total":
        r["total_catalog_work"]["attempted_rows"] += 1
    elif fault == "fitted":
        r["training_state"]["fitted"]["state_digest"] = "foreign"
    elif fault == "role":
        r["trials"][0]["request"]["role"] = "final_test"
    elif fault == "clock":
        r["trials"][0]["request"]["started_at"] = r["fitted_at"] + 1
    elif fault == "metric_count":
        r["trials"][0]["measures"]["new_count"] = 3
    elif fault == "monitoring":
        r["context_monitoring"][0]["role"] = "final_test"
    elif fault == "preparation":
        r["trials"][0]["result"]["catalog_control"]["preparation"]["prepared_rows"] = -3
    else:
        r["trials"][0]["native_work"]["replay_examples_charged"] = False
    fitted_source["path"].write_text(json.dumps(r), encoding="utf8")
    with pytest.raises((ValueError, TypeError)):
        comparison._frozen_fit(fitted_source["path"], catalog_fit=True)
    assert (
        fitted_source["driver"].made == []
        and not fitted_source["train"]
        and not fitted_source["opened"]
    )


@pytest.mark.parametrize("limit", [True, 0, 1, 5, "3"])
def test_should_refuse_bad_decision_limits_before_source(monkeypatch, limit):
    monkeypatch.setattr(demo.context, "_source", lambda *a: pytest.fail("source opened"))
    with pytest.raises(ValueError):
        demo.run_case(
            Path("unused"),
            reference_mode="drift",
            stream_recipe=TicketStreamRecipe("x", ("new_contexts",) * 2, 2),
            deferred_evaluation=True,
            decision_limit=limit,
        )


@pytest.mark.parametrize("bad", [1, None, "yes"])
def test_should_refuse_non_boolean_optin_without_any_source(monkeypatch, bad):
    monkeypatch.setattr(comparison, "_frozen_fit", lambda *a, **kw: pytest.fail("source opened"))
    with pytest.raises(ValueError):
        comparison.run_comparison(Path("unused"), Path("unused"), catalog_fit=bad)


def test_should_keep_policy_change_mode_separate_before_any_source(monkeypatch):
    monkeypatch.setattr(
        comparison, "_run_policy_comparison", lambda *a: pytest.fail("legacy mode dispatched")
    )
    with pytest.raises(ValueError):
        comparison.run_comparison(
            Path("unused"), Path("unused"), policy_change=True, catalog_fit=True
        )


def test_should_stop_failed_paid_candidate_before_more_work_or_final(fitted_source, monkeypatch):
    run = demo.run_case

    def fail(*args, **kw):
        r = run(*args, **kw)
        r["training_work"]["unreturned_work_unknown"] = True
        r["complete"] = False
        return r

    monkeypatch.setattr(demo, "run_case", fail)
    r = comparison.run_comparison(
        fitted_source["driver"].path, fitted_source["path"], catalog_fit=True
    )
    assert not r["complete"] and len(r["cases"]) == 1 and fitted_source["opened"] == []
    assert r["models_frozen_at"] is None


def test_should_refuse_replaced_preceding_paid_cost_before_third_case(fitted_source, monkeypatch):
    admit = comparison._catalog_comparison_case

    def mutate(*args):
        work = admit(*args)
        if len(fitted_source["train"]) == 2:
            first = fitted_source["train"][0][1]
            first["training_work"] = deepcopy(first["training_work"])
        return work

    monkeypatch.setattr(comparison, "_catalog_comparison_case", mutate)
    r = comparison.run_comparison(
        fitted_source["driver"].path, fitted_source["path"], catalog_fit=True
    )
    assert not r["complete"] and len(r["cases"]) == 2 and fitted_source["opened"] == []


def test_should_refuse_changed_fit_blob_after_first_paid_candidate(fitted_source, monkeypatch):
    run = demo.run_case

    def mutate(*args, **kw):
        r = run(*args, **kw)
        fitted_source["path"].write_bytes(b"replaced-fit")
        return r

    monkeypatch.setattr(demo, "run_case", mutate)
    r = comparison.run_comparison(
        fitted_source["driver"].path, fitted_source["path"], catalog_fit=True
    )
    assert not r["complete"] and len(r["cases"]) == 1 and fitted_source["opened"] == []


def test_should_refuse_changed_frozen_candidate_after_first_readout(fitted_source, monkeypatch):
    read = demo._sealed_readout

    def mutate(result, report, clock):
        measures = read(result, report, clock)
        result["candidate_checkpoint"]["state"]["training_examples"] += 1
        return measures

    monkeypatch.setattr(demo, "_sealed_readout", mutate)
    r = comparison.run_comparison(
        fitted_source["driver"].path, fitted_source["path"], catalog_fit=True
    )
    assert not r["complete"] and len(r["cases"]) == 10 and len(fitted_source["opened"]) == 1
    assert "incremental_utility" not in r["final_readouts"][0]


def test_should_refuse_replaced_preceding_readout_before_scoring_second(fitted_source, monkeypatch):
    read = demo._sealed_readout
    reports = []

    def mutate(result, report, clock):
        measures = read(result, report, clock)
        reports.append(report)
        if len(reports) == 2:
            reports[0]["incremental_measures"] = deepcopy(reports[0]["incremental_measures"])
        return measures

    monkeypatch.setattr(demo, "_sealed_readout", mutate)
    r = comparison.run_comparison(
        fitted_source["driver"].path, fitted_source["path"], catalog_fit=True
    )
    assert not r["complete"] and len(fitted_source["opened"]) == 2
    assert "incremental_utility" not in r["final_readouts"][1]


def test_should_keep_catalog_cap_before_source_io(fitted_source, monkeypatch):
    _, p, _ = comparison._frozen_fit(fitted_source["path"], catalog_fit=True)
    monkeypatch.setattr(demo.context, "_source", lambda *a: pytest.fail("source opened"))
    with pytest.raises(ValueError):
        demo.run_case(
            Path("unused"),
            catalog_policy=p,
            stream_recipe=TicketStreamRecipe("x", ("new_contexts",) * 2, 2),
            deferred_evaluation=True,
            decision_limit=4,
        )


def test_should_report_shorter_explicit_catalog_horizon_without_renewal(fitted_source):
    _, p, _ = comparison._frozen_fit(fitted_source["path"], catalog_fit=True)
    # This direct TRAIN-only capture has no task readout; the outer fixture's
    # wrapper records it but never calls the comparison's final callback.
    r = demo.run_case(
        fitted_source["driver"].path,
        catalog_policy=p,
        stream_recipe=TicketStreamRecipe("short-cap", ("new_contexts",) * 2, 1),
        deferred_evaluation=True,
        decision_limit=2,
    )
    assert r["complete"], r.get("error")
    assert r["stream"]["opportunities"] == 2 and r["stream"]["unarrived_labels"] == 1
    assert r["catalog_control"]["stream"]["horizon_limited"]
    assert r["catalog_control"]["stream"]["decision_cap"] == 3


def test_should_reject_catalog_source_when_optin_is_absent(fitted_source):
    with pytest.raises(ValueError, match="original completed"):
        comparison._frozen_fit(fitted_source["path"])


def test_should_preserve_legacy_four_trial_fit_shape(driver, monkeypatch, tmp_path):
    monkeypatch.setattr(demo, "_publish_candidate", lambda *a: None)
    study = learning.run_study(driver.path)
    assert study["complete"], study.get("error")
    path = tmp_path / "literal-legacy.json"
    path.write_text(json.dumps(study), encoding="utf8")
    _, policy, historical = comparison._frozen_fit(path)
    assert type(policy).__name__ == "FittedController" and len(study["trials"]) == 4
    assert "fit_replay_examples" not in historical and historical["all_in_utility"] is None
