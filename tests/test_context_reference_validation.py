"""Frozen role/config adapters with scalar report stubs; no model/teacher work."""

from copy import deepcopy
from dataclasses import asdict, replace
from functools import lru_cache
import json
from inspect import signature
from pathlib import Path

import pytest

from scripts import context_reference_validation as study
from scripts.context_fifo_reference_study import case_outcome
from scripts.context_fifo_reference_study import score_cases
from src.app.controller_chronology import ControllerChronology

DEV = Path("artifacts/runs/context-fifo-reference-20261024/actual.json")
SOURCE = Path("examples/context_head_source/source.json")


@lru_cache
def known():
    return json.loads(DEV.read_text(encoding="utf8"))


def test_should_preserve_all_accepted102_scores_without_model_work():
    assert score_cases(known()["cases"]) == known()["scores"]


@pytest.mark.parametrize("name", ["sealed_callback", "fitting_callback"])
def test_should_not_accept_final_or_fitting_callbacks(name):
    with pytest.raises(TypeError):
        signature(study.run_study).bind(SOURCE, DEV, **{name: lambda: pytest.fail("entered")})


def validation_case(mode, recipe=None):
    """Change metadata in known reports to exercise adapters, never task evidence."""
    recipe = study.VALIDATION_RECIPE if recipe is None else recipe
    c = deepcopy(known()["cases"][mode])
    c["context_recipe"] = asdict(recipe)
    stream = c["training_stream"]
    stream.update(
        contexts=[f"{recipe.recipe_id}:{x}" for x in recipe.contexts],
        row_counts=[3 if x == "new-v1" else 6 for x in recipe.contexts],
        label_delay=recipe.label_delay,
    )
    stream["first_targets"][1][0] = 1.0  # Explicit false ready-invoice target.
    native = c["applied_updates"]
    for row in (*native, *c.get("acknowledged_experiences", [])):
        i = int(row["sample_id"].split("-")[1])
        row["arrived_at"] = i + recipe.label_delay
    examples, mistakes = 72, 16
    for call, receipt in zip(c["native_calls"], native, strict=True):
        i = int(receipt["sample_id"].split("-")[1])
        call["rows"] = 3 if recipe.contexts[i] == "new-v1" else 6
        examples += call["rows"]
        mistakes += int(receipt["diagnostic"]["value"])
        call["state"].update(training_examples=examples, mistake_updates=mistakes)
    c["training_work"]["attempted_rows"] = examples - 72
    if mode == "candidate_ack":
        f = c["candidate_feedback"]
        for row in f["records"]:
            i = row["index"]
            row["label_arrived_at"] = i + recipe.label_delay
            row["rows"] = 3 if recipe.contexts[i] == "new-v1" else 6
        f["rows_attempted"] = f["rows_returned"] = sum(x["rows"] for x in f["records"])
    c["synthetic_label_noise"] = [
        dict(
            role="train",
            kind="synthetic_label_corruption",
            batch=i,
            semantic_row="v1:invoice:ready",
            source_row_id="update-1-1",
            row_id=f"{recipe.recipe_id}:batch{i}:v1:invoice:ready",
            clean_action="route",
            corrupted_action="defer",
        )
        for i, x in enumerate(recipe.contexts)
        if x == "old-v1"
    ]
    return c


@pytest.mark.parametrize("mode", study.MODES)
def test_should_bind_full_fixed_execution_and_distinct_roles(mode):
    execution = study.ReferenceExecution(mode)
    record, options, p = execution.record(), execution.kwargs(), study.protocol(execution)
    assert options["reference_mode"] == record["policy"]["mode"] == execution.policy.mode
    assert (
        options.get("candidate_feedback", False)
        == record["candidate_feedback"]
        == (mode == "candidate_ack")
    )
    assert options.get("acknowledge_correct", False) == record["acknowledge_correct"]
    assert tuple(x.role for x in p.streams) == (
        "controller_train",
        "controller_validation",
        "sealed_evaluation",
    )
    assert len({x.source_group for x in p.streams}) == 3
    assert options["stream_recipe"] == study.VALIDATION_RECIPE
    assert options["stream_recipe"] is not study.VALIDATION_RECIPE


@pytest.mark.parametrize("mode", [True, None, "arrived_shift", "fitted"])
def test_should_refuse_undeclared_execution_before_any_payload(mode):
    with pytest.raises(ValueError):
        study.ReferenceExecution(mode)


@pytest.mark.parametrize("mode", study.MODES)
def test_should_preserve_development_defaults_and_admit_only_declared_new_recipe(mode):
    case_outcome(known()["cases"][mode], mode)
    c = validation_case(mode)
    with pytest.raises(ValueError):
        case_outcome(c, mode)
    outcome = case_outcome(c, mode, recipe=study.VALIDATION_RECIPE)
    assert outcome["old"] == 4
    study._noise(c)


@pytest.mark.parametrize("fault", ["count", "role", "name", "target", "delay"])
def test_should_refuse_wrong_validation_source_or_corruption_provenance(fault):
    c = validation_case("no_learning")
    if fault == "count":
        c["synthetic_label_noise"].pop()
    if fault == "role":
        c["synthetic_label_noise"][0]["role"] = "final_test"
    if fault == "name":
        c["synthetic_label_noise"][0]["semantic_row"] = "v1:password:ready"
    if fault == "target":
        c["synthetic_label_noise"][0]["corrupted_action"] = "route"
    if fault == "delay":
        c["training_stream"]["label_delay"] = 2
    with pytest.raises(ValueError):
        study._noise(c)


def test_should_keep_validation_out_of_training_and_never_supply_final_callback(monkeypatch):
    events, chronologies = [], []
    original = ControllerChronology.run_development

    def phase(self, stream_id, operation):
        snap = self.snapshot()
        role = snap["protocol"].streams[len(snap["rows"])].role
        events.append(role)
        chronologies.append(self)
        return original(self, stream_id, operation)

    def run(path, **kwargs):
        assert events.count("controller_train") == 3
        mode = "candidate_ack" if kwargs.get("acknowledge_correct") else kwargs["reference_mode"]
        return validation_case(mode, kwargs["stream_recipe"])

    monkeypatch.setattr(ControllerChronology, "run_development", phase)
    monkeypatch.setattr(
        ControllerChronology, "evaluate_final", lambda *a, **k: pytest.fail("sealed opened")
    )
    monkeypatch.setattr(study.live, "run_case", run)
    r = study.run_study(SOURCE, DEV)
    assert r["complete"], r.get("error")
    assert events == ["controller_train"] * 3 + ["controller_validation"] * 3
    assert r["sealed_sources_opened"] == r["fitting_callbacks"] == 0
    assert len(r["validation_scores"]) == 3
    for mode in study.MODES:
        s = r["chronologies"][mode]
        assert s["clock"] == s["frozen_at"] == 2 and len(s["rows"]) == 2
        assert [x["metadata"]["stream"]["role"] for x in s["rows"]] == [
            "controller_train",
            "controller_validation",
        ]
    for c in chronologies[-3:]:
        with pytest.raises(ValueError):
            c.training_rows()
        with pytest.raises(ValueError):
            c.run_development("again", lambda p: pytest.fail("retry"))


@pytest.mark.parametrize("failure", study.MODES)
def test_should_latch_failed_paid_validation_and_stop_remaining_cases(monkeypatch, failure):
    seen = []

    def run(path, **kwargs):
        mode = "candidate_ack" if kwargs.get("acknowledge_correct") else kwargs["reference_mode"]
        seen.append(mode)
        c = validation_case(mode)
        if mode == failure:
            c["complete"] = False
        return c

    monkeypatch.setattr(study.live, "run_case", run)
    r = study.run_study(SOURCE, DEV)
    assert not r["complete"] and not r["validation_scores"]
    assert seen == list(study.MODES[: study.MODES.index(failure) + 1])
    assert failure in r["validation_cases"] and r["chronologies"][failure]["stopped"]
    assert r["sealed_sources_opened"] == 0


def test_should_refuse_config_replacement_after_paid_call_before_metrics(monkeypatch):
    calls = []

    def run(path, **kwargs):
        calls.append(1)
        c = validation_case("no_learning", kwargs["stream_recipe"])
        monkeypatch.setattr(
            study, "VALIDATION_RECIPE", replace(study.VALIDATION_RECIPE, label_delay=2)
        )
        return c

    monkeypatch.setattr(study.live, "run_case", run)
    r = study.run_study(SOURCE, DEV)
    assert not r["complete"] and not r["validation_scores"] and len(calls) == 1
    assert "no_learning" in r["validation_cases"] and "binding changed" in r["error"]


def test_should_refuse_replaced_development_before_validation_source(tmp_path, monkeypatch):
    p = tmp_path / "bad.json"
    p.write_text("{}", encoding="utf8")
    monkeypatch.setattr(study.live, "run_case", lambda *a, **k: pytest.fail("validation entered"))
    r = study.run_study(SOURCE, p)
    assert not r["complete"] and not r["validation_cases"] and not r["development"]
