"""Scripted live catalog window into the existing real disposable probe loop."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from scripts import run_bp_ticket_serving as demo
from scripts import probe_ticket_policy_learning as recipe
from scripts.ticket_serving_timeline import LiveServingPlan
from test_catalog_live_recipes import recipes as recipes, run
from test_catalog_live_serving import serving as serving
from test_ticket_catalog_control import driver as driver, FakeBP
from test_categorical_replay_action import State
from test_ticket_policy_probe import FakeReader


@pytest.mark.parametrize(
    "change",
    [
        dict(probe_after_window=1),
        dict(probe_after_window=None),
        dict(probe_after_window=True),
        dict(probe_after_window=True, reference_mode="drift", serving_plan=LiveServingPlan()),
        dict(probe_after_window=True, publish_after_window=True),
    ],
)
def test_should_refuse_invalid_probe_modes_before_source_io(monkeypatch, change):
    monkeypatch.setattr(demo.context, "_source", lambda *a: pytest.fail("source IO"))
    with pytest.raises(ValueError):
        demo.run_case(Path("unused"), **change)


@pytest.fixture
def probing(recipes, monkeypatch):
    from scripts import ticket_catalog_probe as composition

    original = demo.native.counted_learner
    recipes.probe_owners = []

    def construct(method, width, classes, before, after):
        if not recipes.made:
            return original(method, width, classes, before, after)
        owner = FakeBP(before, after)
        owner.restore_state(State())
        recipes.probe_owners.append(owner)
        return owner

    monkeypatch.setattr(demo.native, "counted_learner", construct)
    monkeypatch.setattr(demo.native, "readout", FakeReader)
    start_probe = composition.run_catalog_probe

    def observed(report, **kwargs):
        recipes.probe_report, recipes.probe_context = report, kwargs
        return start_probe(report, **kwargs)

    monkeypatch.setattr(composition, "run_catalog_probe", observed)
    recipes.composition = composition
    return recipes


def test_should_probe_current_owned_candidate_with_separate_callbacks_and_return(probing):
    r = run(probing, probe_after_window=True, serving_plan=LiveServingPlan(9, (2, 6)))
    assert r["complete"], r.get("error")
    assert r["catalog_window_complete"] is True and r["reserved_probe_updates"] == 64
    assert r["training_work"]["attempted_updates"] == 5
    assert r["training_work"]["attempted_rows"] == 60
    assert r["budget"]["updates_completed"] == 5
    probe = r["isolated_probe"]
    assert probe["complete"] and len(probe["cases"]) == 2
    assert probe["diagnostic_seed_owners"] == dict(
        attempts=2, returns=2, unreturned_work_unknown=False
    )
    assert probe["training_work"] == dict(
        attempted_updates=64, returned_updates=64, attempted_rows=192, unreturned_work_unknown=False
    )
    assert [
        (p["mode"], p["prior_native_updates"], p["prior_native_rows"]) for p in probe["cases"]
    ] == [("warm", 101, 636), ("fresh", 0, 0)]
    assert all(len(p["episodes"]) == 8 for p in probe["cases"])
    assert all(p["probe"]["return_phase"]["completed_updates"] == 16 for p in probe["cases"])
    assert len(probing.probe_owners) == 2
    assert probing.probe_owners[0].snapshot_state().completed_updates == 101
    assert probing.probe_owners[1].snapshot_state().completed_updates == 0
    assert probing.made[0].snapshot_state() == probing.initial
    assert r["catalog_actor_unchanged"] and r["source_unchanged"]
    assert r["numpy_rng_unchanged"] and r["torch_rng_unchanged"]
    assert r["live_serving"]["closed"] and r["stream"]["unarrived_labels"] == 1
    assert r["catalog_promotions"] == r["final_examples"] == 0
    with pytest.raises(ValueError):
        demo._case_measures(r, 1.0)


def test_should_not_start_probe_after_unknown_preceding_window(probing, monkeypatch):
    real = demo._candidate_tasks

    def fail(*a, **kw):
        real(*a, **kw)
        raise RuntimeError("unknown preceding readout")

    monkeypatch.setattr(demo, "_candidate_tasks", fail)
    r = run(probing, probe_after_window=True)
    assert not r["complete"] and not probing.probe_owners
    assert r["isolated_probe"]["status"] == "not_run"
    assert r["training_work"]["attempted_updates"] > 0


@pytest.mark.parametrize("fault", ["late_update", "first_result", "source", "rng"])
def test_should_stop_after_paid_probe_failure_or_replaced_evidence(probing, monkeypatch, fault):
    real = recipe.run_arm

    def damage(*args, **kwargs):
        if fault == "late_update":
            owner = args[0]
            # The fault travels through the real fork to its first update.
            owner.faults["raise_replay"] = True
        result = real(*args, **kwargs)
        if fault == "first_result":
            result["probe"]["status"] = "failed"
        elif fault == "source":
            probing.path.write_bytes(b"replaced")
        elif fault == "rng":
            np.random.seed(123)
        return result

    numpy_before = deepcopy(np.random.get_state())
    monkeypatch.setattr(recipe, "run_arm", damage)
    try:
        r = run(probing, probe_after_window=True)
    finally:
        np.random.set_state(numpy_before)
    assert not r["complete"] and len(probing.probe_owners) == 1
    assert r["training_work"]["attempted_updates"] == 7
    assert r["isolated_probe"]["training_work"]["attempted_updates"] > 0


def test_should_preserve_default_without_constructing_diagnostic_owners(probing):
    r = run(probing)
    assert r["complete"] and not probing.probe_owners
    assert r["reserved_probe_updates"] == 0 and "isolated_probe" not in r


@pytest.mark.parametrize("fault", ["paid_value", "paid_replacement", "sharing", "clock", "pending"])
def test_should_refuse_changed_original_evidence_before_diagnostic_construction(
    probing, monkeypatch, fault
):
    original = recipe.build_cohorts

    def changed(spec):
        result = original(spec)
        r, context = probing.probe_report, probing.probe_context
        if fault == "paid_value":
            r["applied_updates"][0]["sample_id"] = "changed"
        elif fault == "paid_replacement":
            r["applied_updates"] = deepcopy(r["applied_updates"])
        elif fault == "sharing":
            context["gate"].pause()
        elif fault == "clock":
            context["budget"].last_clock += 1
        else:
            cursor = context["runtime"]._inbox._borrow_cursor()
            object.__setattr__(cursor.labels[-1], "model_version", "changed")
        return result

    monkeypatch.setattr(recipe, "build_cohorts", changed)
    r = run(probing, probe_after_window=True)
    assert not r["complete"] and not probing.probe_owners
    assert r["catalog_window_complete"] is True
    assert r["isolated_probe"]["training_work"]["attempted_updates"] == 0
    if fault in ("paid_value", "paid_replacement"):
        assert r["applied_updates"][0]["sample_id"] == "batch-0"
        assert r["isolated_probe"]["paid_window_evidence_refused"] is True


@pytest.mark.parametrize("fault", ["arm_value", "arm_replacement", "counts", "cases", "root"])
def test_should_protect_first_accepted_arm_before_fresh_probe_updates(probing, monkeypatch, fault):
    construct = demo.native.counted_learner

    def changed(*args):
        seed = construct(*args)
        if len(probing.probe_owners) == 2:
            paid = probing.probe_report["isolated_probe"]
            if fault == "arm_value":
                paid["cases"][0]["probe"]["completed_updates"] = 0
            elif fault == "arm_replacement":
                paid["cases"][0]["probe"] = deepcopy(paid["cases"][0]["probe"])
            elif fault == "cases":
                paid["cases"] = list(paid["cases"])
            elif fault == "root":
                probing.probe_report["isolated_probe"] = deepcopy(paid)
            else:
                paid["training_work"]["attempted_updates"] = 0
        return seed

    monkeypatch.setattr(demo.native, "counted_learner", changed)
    r = run(probing, probe_after_window=True)
    assert not r["complete"] and len(probing.probe_owners) == 2
    assert len(r["isolated_probe"]["cases"]) == 1
    assert r["isolated_probe"]["training_work"]["attempted_updates"] == 32
    assert r["isolated_probe"]["training_work"]["returned_updates"] == 32
    assert r["isolated_probe"]["cases"][0]["probe"]["completed_updates"] == 16
    if fault != "counts":
        assert r["isolated_probe"]["paid_evidence_refused"] is True


@pytest.mark.parametrize("fault", ["factory", "torch"])
def test_should_capture_failed_probe_setup_without_discarding_original_window(
    probing, monkeypatch, fault
):
    construct = demo.native.counted_learner
    torch = demo.native.require_torch
    calls = 0

    def failed_constructor(*args):
        if getattr(probing, "probe_report", None) is not None:
            raise RuntimeError("lost diagnostic constructor return")
        return construct(*args)

    def failed_torch():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("diagnostic setup unavailable")
        return torch()

    monkeypatch.setattr(
        demo.native,
        "counted_learner" if fault == "factory" else "require_torch",
        failed_constructor if fault == "factory" else failed_torch,
    )
    r = run(probing, probe_after_window=True)
    assert not r["complete"] and r["catalog_window_complete"] is True
    assert r["training_work"]["returned_updates"] == 7
    assert not probing.probe_owners
    assert r["isolated_probe"]["diagnostic_seed_owners"]["attempts"] == int(fault == "factory")
    assert r["isolated_probe"]["diagnostic_seed_owners"]["unreturned_work_unknown"] == (
        fault == "factory"
    )
