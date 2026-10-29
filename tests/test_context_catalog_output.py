"""Literal metadata only; no source, learner, fitter execution or task calls."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from scripts.context_catalog_control import context_catalog, ContextCatalogControl
from scripts.context_catalog_learning import controller_record, write_catalog_report
from src.core.catalog_contextual_controller import (
    CatalogEstimate,
    CatalogTrialWork,
    FittedCatalogController,
    fit_catalog_choices,
)
from src.core.controller_objective import ControllerObjective
from src.app.counterfactual_controller import CatalogCounterfactualRequest


def snapshot():
    catalog = context_catalog()
    rows = tuple(
        CatalogEstimate(c, a, float(a == "wait"))
        for c in ("clean", "mistakes")
        for a in catalog.arms
    )
    fitted = FittedCatalogController(
        "qualified-12",
        "literal-digest",
        ControllerObjective(),
        86,
        rows,
        fit_catalog_choices(catalog, rows),
        catalog=catalog,
    )
    return dict(
        attempts=6,
        completed=6,
        discarded_trials=6,
        stopped=False,
        rows=rows,
        fitted=fitted,
        clock=86,
        catalog=catalog,
        failed_work=None,
    )


def test_should_project_literal_completed_controller_to_detached_json():
    original = snapshot()
    result = controller_record(original)
    decoded = json.loads(json.dumps(result, allow_nan=False))
    assert decoded["completed"] == 6 and len(decoded["rows"]) == 6
    assert decoded["fitted"]["learner_version"] == "qualified-12"
    result["rows"][0]["mean_utility"] = 99
    assert original["rows"][0].mean_utility != 99


def test_should_project_partial_work_without_inventing_a_fit():
    original = snapshot()
    original.update(
        rows=original["rows"][:1],
        fitted=None,
        completed=1,
        stopped=True,
        failed_work=CatalogTrialWork(1, 0, 6, 0, 6, 1, 3, False),
    )
    result = controller_record(original)
    assert result["fitted"] is None and result["failed_work"]["work_complete"] is False
    assert json.loads(json.dumps(result))["failed_work"]["attempted_updates"] == 1


@pytest.mark.parametrize("fault", ["too_many", "row", "fit", "catalog", "work"])
def test_should_refuse_unbounded_or_foreign_controller_metadata(fault):
    data = snapshot()
    if fault == "too_many":
        data["rows"] = data["rows"] + data["rows"][:1]
    elif fault == "row":
        data["rows"] = ({},)
    else:
        data[{"fit": "fitted", "catalog": "catalog", "work": "failed_work"}[fault]] = object()
    with pytest.raises(ValueError):
        controller_record(data)


def test_should_write_completed_json_using_projected_controller(tmp_path):
    target = tmp_path / "result.json"
    report = dict(complete=True, controller=controller_record(snapshot()))
    assert write_catalog_report(target, report)
    assert json.loads(target.read_text())["controller"]["completed"] == 6


@pytest.mark.parametrize("fault", ["unknown", "oversize", "nan"])
def test_should_save_failed_scalar_costs_before_refusing_detailed_output(tmp_path, fault):
    target = tmp_path / "failed.json"
    report = dict(
        complete=True,
        trials=[
            dict(
                result=dict(
                    training_work=dict(attempted_updates=2, attempted_rows=9, returned_updates=1),
                    budget=dict(updates=0, started_at=1.0, last_clock=2.0),
                    replay_examples=6,
                )
            )
        ],
    )
    report["bad"] = (
        object() if fault == "unknown" else "x" * 32768 if fault == "oversize" else float("nan")
    )
    observed = dict(seconds=3.0, peak_rss_bytes=1234, counts=dict(train_passes=2, training_rows=9))
    before = deepcopy(report["trials"])
    assert not write_catalog_report(target, report, observed_work=observed, max_bytes=32768)
    saved = json.loads(target.read_text())
    assert saved["complete"] is False and saved["producer_complete"] is True
    assert not saved["detailed_outcomes_available"] and not saved["validated_receipts_available"]
    assert saved["reported_case_costs"][0]["training_work"]["attempted_rows"] == 9
    assert saved["observer_counts"]["train_passes"] == 2
    assert saved["observer_reported"]["elapsed_seconds"] == 3.0
    assert saved["observer_counts"]["source_reads"] is None
    assert report["trials"] == before and target.stat().st_size <= 32768


def test_should_preserve_unknown_counters_in_failed_output(tmp_path):
    target = tmp_path / "unknown.json"
    report = dict(
        complete=False,
        trials=[dict(result=dict(training_work=dict(attempted_updates=True)))],
        bad=object(),
    )
    assert not write_catalog_report(target, report)
    saved = json.loads(target.read_text())
    assert saved["reported_case_costs"][0]["training_work"]["attempted_updates"] is None
    assert saved["observer_counts"]["train_passes"] is None


def test_should_never_replace_existing_output(tmp_path):
    target = tmp_path / "old.json"
    target.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        write_catalog_report(target, dict(complete=True))
    assert target.read_bytes() == b"original"


@pytest.mark.parametrize("field", ["record", "catalog", "prototype"])
def test_should_refuse_replaced_constructor_owners_before_any_source_or_model(field):
    request = CatalogCounterfactualRequest(
        "qualified-12", "literal-digest", "mistakes", "wait", 0, catalog=context_catalog()
    )
    owner = ContextCatalogControl(request, base_snapshot={})
    if field == "record":
        owner.record = replace(request, catalog=request.catalog)
    elif field == "catalog":
        owner.catalog = deepcopy(owner.catalog)
    else:
        owner._old = object()
    with pytest.raises(ValueError, match="owner replaced"):
        owner.require_pin()


def test_should_keep_original_constructor_bindings_without_native_work():
    request = CatalogCounterfactualRequest(
        "qualified-12", "literal-digest", "mistakes", "wait", 0, catalog=context_catalog()
    )
    owner = ContextCatalogControl(request, base_snapshot={})
    owner.require_pin()
    assert owner.record is request and owner.catalog is request.catalog
    assert owner.replay is owner.replay_budget is None
