"""Synthetic literal source, actual contextual catalog/live/executor integration."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

import pytest

from scripts import context_head_live as live
from scripts.context_catalog_control import ContextCatalogControl, context_catalog
from scripts.context_catalog_learning import catalog_measures
from src.adapters.context_serving import snapshot_digest
from src.app.counterfactual_controller import CatalogCounterfactualRequest
from src.core.catalog_contextual_controller import (
    CatalogEstimate,
    FittedCatalogController,
    fit_catalog_choices,
)
from src.core.categorical_perceptron import PerceptronPolicy, PerceptronSnapshot
from src.core.context_perceptron import ContextPerceptronPolicy, ContextPerceptronSnapshot
from src.core.controller_objective import ControllerObjective
from src.core.ticket_workflow import ACTION_IDS


def initial():
    """Handwritten fixture only, never decoded from a qualified research artifact."""
    words = ("invoice", "legacy", "parcel", "password", "ready", "sealed", "staged")
    heads = []
    for index in range(2):
        weights = [[0.0] * 6 for _ in range(8)]
        actions = dict(
            sealed="open",
            staged="commit",
            invoice="route" if index == 0 else "inspect",
            password="inspect",
            parcel="defer",
        )
        if index:
            actions["legacy"] = "retire"
        for word, action in actions.items():
            weights[words.index(word)][ACTION_IDS.index(action)] = 2.0
        heads.append(
            PerceptronSnapshot(
                PerceptronPolicy(7, 6),
                tuple(tuple(r) for r in weights),
                8 if index == 0 else 4,
                48 if index == 0 else 24,
                12 if index == 0 else 4,
            )
        )
    return ContextPerceptronSnapshot(
        ContextPerceptronPolicy(9, 6, (4, 5)), (heads[0], heads[1]), 12, 72, 16
    )


@pytest.fixture
def source(monkeypatch):
    state = initial()
    raw = b"{}"
    monkeypatch.setattr(live, "_source_bytes", lambda path: raw)
    monkeypatch.setattr(live, "HEAD_SOURCE_SHA", sha256(raw).hexdigest())
    monkeypatch.setattr(live, "head_states", lambda data: (deepcopy(state),))
    return state


def request(state=None, action="replay6-wake1", context="mistakes"):
    return CatalogCounterfactualRequest(
        "qualified-12",
        snapshot_digest(state or initial()),
        context,
        action,
        0,
        catalog=context_catalog(),
    )


def fitted(state=None):
    catalog = context_catalog()
    estimates = tuple(
        CatalogEstimate(c, a, float(a == ("wait" if c == "clean" else "replay6-wake1")))
        for c in ("clean", "mistakes")
        for a in catalog.arms
    )
    return FittedCatalogController(
        "qualified-12",
        snapshot_digest(state or initial()),
        ControllerObjective(),
        0,
        estimates,
        fit_catalog_choices(catalog, estimates),
        catalog=catalog,
    )


@pytest.mark.parametrize(
    "action,passes,replay_rows", [("wait", 0, 0), ("wake1", 1, 0), ("replay6-wake1", 2, 6)]
)
def test_should_run_full_catalog_trial_through_real_original_live_schedule(
    source, action, passes, replay_rows
):
    r = live.run_case("unused", catalog_trial=request(source, action), catalog_base=source)
    assert r["complete"], (r.get("error"), r.get("task_outcomes"))
    history = r["catalog_control"]["native_history"]
    if action == "replay6-wake1":
        # Serving may defer the wake after a successful, permanently paid replay.
        assert 1 <= history["replay_updates"] <= 2
        assert history["wake_updates"] in (0, 1)
        assert r["replay_examples"] == 6 * history["replay_updates"]
    else:
        assert r["training_work"]["returned_updates"] == passes
        assert r["replay_examples"] == replay_rows
    assert r["source_unchanged"] and r["actor_unchanged"] and "promotion_receipt" not in r
    assert len(r["live_serving"]["records"]) == 14
    measures, work = catalog_measures(r)
    work.require_complete(context_catalog(), action)
    assert (
        work.returned_updates == history["wake_updates"] + history["replay_updates"]
        and measures.work_complete
    )
    assert history["completed_updates"] == 12 + work.returned_updates
    assert len(r["task_outcomes"]["before"]) == len(r["task_outcomes"]["after"]) == 6


def test_should_apply_original_frozen_catalog_and_retain_pending_clean_work(source):
    r = live.run_case("unused", catalog_policy=fitted(source))
    assert r["complete"], r.get("error")
    catalog_measures(r)
    assert r["actor_unchanged"] and r["source_unchanged"]
    assert r["catalog_control"]["phase"] == "private_application"
    assert r["stream"]["arrived_labels"] == 2
    assert r["budget"]["updates"] in (0, 1)
    assert r["stream"]["pending_unapplied"] == 2 - r["budget"]["updates"]
    assert r["replay_examples"] == 6


def test_should_preserve_default_guarded_live_path_without_catalog_fields(source):
    r = live.run_case("unused")
    assert r["complete"], r.get("error")
    assert r["budget"]["updates"] == 2 and "catalog_control" not in r
    assert r["source_unchanged"]


@pytest.mark.parametrize("fault", ["version", "context", "role", "action", "catalog", "base"])
def test_should_refuse_wrong_request_before_source_or_model_work(monkeypatch, fault):
    req, base = request(), initial()
    if fault == "version":
        req = replace(req, learner_version="qualified-96")
    elif fault == "context":
        req = replace(req, context="other")
    elif fault == "role":
        req = replace(req, role="final_test")
    elif fault == "action":
        req = replace(req, action_id="rate")
    elif fault == "catalog":
        req = replace(req, catalog=True)
    else:
        base = None
    monkeypatch.setattr(live, "_source_bytes", lambda *a: pytest.fail("source opened"))
    with pytest.raises((ValueError, AttributeError)):
        live.run_case("unused", catalog_trial=req, catalog_base=base)


@pytest.mark.parametrize(
    "options",
    [
        dict(reference_mode="drift"),
        dict(candidate_feedback=True),
        dict(acknowledge_correct=True),
        dict(task_feedback=True),
        dict(arrived_shift=True),
        dict(stream_recipe=True),
        dict(catalog_policy=True),
    ],
)
def test_should_refuse_conflicting_catalog_modes_before_io(monkeypatch, options):
    monkeypatch.setattr(live, "_source_bytes", lambda *a: pytest.fail("source opened"))
    with pytest.raises(ValueError):
        live.run_case("unused", catalog_trial=request(), catalog_base=initial(), **options)


def test_should_refuse_changed_full_base_before_model_construction(source, monkeypatch):
    monkeypatch.setattr(
        live, "ContextPerceptronLearner", lambda *a: pytest.fail("model constructed")
    )
    r = live.run_case(
        "unused", catalog_trial=request(source), catalog_base=replace(source, mistake_updates=17)
    )
    assert not r["complete"] and "source/base" in r["error"]
    assert r["training_work"]["attempted_updates"] == 0


def test_should_refuse_replaced_catalog_record_even_when_equal_valued():
    req = request()
    owner = ContextCatalogControl(req, base_snapshot=initial())
    object.__setattr__(req, "catalog", deepcopy(req.catalog))
    with pytest.raises(ValueError, match="replaced"):
        owner.require_pin()


def test_should_refuse_history_without_original_executor_or_packets():
    owner = ContextCatalogControl(request(), base_snapshot=initial())
    with pytest.raises(ValueError, match="bound packets"):
        owner.require_history(None, None)


def test_should_retain_failed_replay_work_and_runtime_receipts(source, monkeypatch):
    original = live.ContextPerceptronLearner.train_batch

    def failed(owner, features, targets):
        original(owner, features, targets)
        raise RuntimeError("lost context catalog native return")

    monkeypatch.setattr(live.ContextPerceptronLearner, "train_batch", failed)
    r = live.run_case("unused", catalog_trial=request(source), catalog_base=source)
    assert not r["complete"] and "lost context catalog native return" in r["error"]
    assert r["replay_examples"] == 6 and r["budget"]["updates"] == 0
    assert (
        r["training_work"]["attempted_updates"] == 1 and r["training_work"]["returned_updates"] == 0
    )
    assert r["applied_updates"] == [] and r["source_unchanged"] and r["actor_unchanged"]
    replay = r["catalog_control"]["replay"]
    assert replay["stopped"] and replay["native_completions"] == replay["committed_events"] == 0


def test_should_refuse_unobserved_counter_types_roles_and_readout_work(source):
    r = live.run_case("unused", catalog_trial=request(source, "wait"), catalog_base=source)
    assert r["complete"], r.get("error")
    for fault in ("boolean", "role", "readout"):
        changed = deepcopy(r)
        if fault == "boolean":
            changed["catalog_control"]["native_history"]["wake_updates"] = False
        elif fault == "role":
            changed["task_outcomes"]["before"][0]["role"] = "final_test"
        else:
            changed["typed_work"]["returns"] -= 1
        with pytest.raises(ValueError):
            catalog_measures(changed)
