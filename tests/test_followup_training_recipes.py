"""Synthetic packet/scalar schedule checks, with no native learner or teacher."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import run_bp_ticket_serving as demo
from scripts.ticket_controller_streams import (
    FOLLOWUP_TRAINING_RECIPES,
    build_followup_stream,
)
from scripts.ticket_serving_timeline import FollowupTrainingPlan, LiveServingPlan
from src.app.coupled_attribution import AttributionBatch
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from test_post_publication_learning import LiteralLearner, LiteralActor, LiteralLive, batches

CLEAN, NOISY = (r for _, r in FOLLOWUP_TRAINING_RECIPES)


def originals():
    return tuple(
        AttributionBatch(
            "train",
            f"original-{i}",
            tuple(f"row-{i}-{j}" for j in range(6)),
            np.full((6, 7), i % 2, dtype=float),
            np.full((6, 1), demo.ACTION_IDS.index("inspect"), dtype=float),
            num_classes=6,
        )
        for i in range(16)
    )


def test_should_preserve_first_window_and_match_second_except_marked_target():
    source = originals()
    pin = deepcopy(source)
    clean, noisy = (build_followup_stream(r, source) for r in (CLEAN, NOISY))
    assert [len(b.features) for b in clean] == [6] * 8 + [12] * 8
    assert demo._same_value(source, pin)
    assert demo._same_value(clean[:8], source[:8]) and demo._same_value(noisy[:8], source[:8])
    assert len(set(k for b in clean for k in b.row_ids)) == 144
    for a, b in zip(clean[8:], noisy[8:], strict=True):
        assert a.context == b.context and a.row_ids == b.row_ids
        np.testing.assert_array_equal(a.features, b.features)
        assert np.flatnonzero(a.targets[:, 0] != b.targets[:, 0]).tolist() == [7]
        assert a.targets[7, 0] == demo.ACTION_IDS.index("inspect")
        assert b.targets[7, 0] == demo.ACTION_IDS.index("defer")
        assert b.role == "train" and not np.shares_memory(b.targets, source[8].targets)
    noisy[8].targets[0, 0] = 99
    assert noisy[9].targets[0, 0] != 99 and demo._same_value(source, pin)


@pytest.mark.parametrize("role", ["probe_train", "inner_guard", "outer_selection", "final_test"])
def test_should_refuse_nontrain_originals_without_relabeling(role):
    source = list(originals())
    source[8] = replace(source[8], role=role)
    with pytest.raises(ValueError, match="role must be train"):
        build_followup_stream(NOISY, tuple(source))


@pytest.mark.parametrize("fault", ["count", "recipe", "boolean", "payload"])
def test_should_refuse_changed_contract_or_owned_payload(fault):
    recipe = deepcopy(NOISY)
    source = originals()
    if fault == "count":
        with pytest.raises(ValueError, match="sixteen"):
            build_followup_stream(recipe, source[:-1])
        return
    owned = build_followup_stream(recipe, source)
    recipe_pin, data_pin = deepcopy(recipe), deepcopy(owned)
    if fault == "recipe":
        object.__setattr__(recipe.recipe, "label_delay", 1)
    if fault == "boolean":
        object.__setattr__(recipe, "retained_invoice_noise", 1)
    if fault == "payload":
        owned[8].targets[7, 0] = 99
    with pytest.raises(ValueError):
        demo._require_followup_input(recipe, recipe_pin, owned, data_pin)


@pytest.mark.parametrize("first_delay,second_delay", [(1, None), (1, 1), (1, 2), (2, None)])
def test_should_keep_each_window_clock_and_original_lifetime_owner(
    monkeypatch, first_delay, second_delay
):
    source, owners = LiteralLearner(), []
    actor = LiteralActor(source)
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    gate = demo.ServingPriorityGate(demo.SharingLimits(1, 2, 1), resource_available=lambda: True)
    live = LiteralLive(gate)
    report: dict[str, Any] = {}
    factory = demo.ActorShadowRuntime

    def runtime(*args, **kw):
        owner = factory(*args, **kw)
        owners.append(owner)
        return owner

    monkeypatch.setattr(demo, "ActorShadowRuntime", runtime)
    owner, horizon = demo._reference_schedule(
        source,
        actor,
        batches(),
        budget,
        "periodic",
        report,
        lambda: None,
        lambda: None,
        label_delay=first_delay,
        sharing_gate=gate,
        live=live,
        on_drain=lambda r, tick: actor.publish(r),
        followup_plan=FollowupTrainingPlan(1, 4),
        **({} if second_delay is None else dict(followup_label_delay=second_delay)),
    )
    delay = first_delay if second_delay is None else second_delay
    assert owners == [owner] and owner._budget is budget and horizon == 10
    assert budget.updates_completed == gate.snapshot().admitted_updates == 2
    receipts = owner.applied_updates
    assert [(r.actor_version, r.observed_at, r.arrived_at) for r in receipts] == [
        ("old", 0, first_delay),
        ("c0", 4, 4 + delay),
    ]
    assert [r.update_number for r in receipts] == [1, 2]
    assert report["followup_training"]["status"] == "complete"
    assert report["action_snapshot"]["envelope"]["max_wake_updates"] == 2
    assert source.snapshot_state() == 0 and actor.snapshot_state().state == 1
    assert owner.candidate_snapshot().state == 2 and len(report["feedback_invalidations"]) == 1
    if delay == first_delay:
        assert "label_delay" not in report["followup_training"]


@pytest.mark.parametrize("value", [True, 0, 3, 1.0])
def test_should_refuse_invalid_second_delay_before_models(value):
    with pytest.raises(ValueError, match="delay"):
        demo._reference_schedule(
            None,
            None,
            (),
            None,
            "periodic",
            {},
            lambda: None,
            lambda: None,
            followup_label_delay=value,
            followup_plan=FollowupTrainingPlan(),
        )


def test_should_refuse_changed_generation_before_delayed_second_label(monkeypatch):
    source = LiteralLearner()
    actor = LiteralActor(source)
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    gate = demo.ServingPriorityGate(demo.SharingLimits(1, 2, 1), resource_available=lambda: True)
    live = LiteralLive(gate)
    report: dict[str, Any] = {}

    def require():
        if len(report.get("controller_events", [])) == 5:
            actor.generation = 2

    with pytest.raises(ValueError, match="publication"):
        demo._reference_schedule(
            source,
            actor,
            batches(),
            budget,
            "periodic",
            report,
            require,
            lambda: None,
            sharing_gate=gate,
            live=live,
            on_drain=lambda r, t: actor.publish(r),
            followup_plan=FollowupTrainingPlan(1, 4),
            followup_label_delay=2,
        )
    assert budget.updates_completed == 1 and len(report["applied_updates"]) == 1
    assert report["followup_training"]["pending_prediction_indices"] == [1]
    assert report["followup_training"]["status"] == "incomplete"


@pytest.mark.parametrize(
    "changes",
    [
        {"followup_plan": None},
        {"followup_training_recipe": object()},
        {"live_training_recipe": NOISY},
        {"reference_mode": "drift"},
        {"serving_plan": LiveServingPlan(33)},
    ],
)
def test_should_refuse_unsupported_recipe_modes_before_source_access(monkeypatch, changes):
    monkeypatch.setattr(demo.context, "_source", lambda *a: pytest.fail("source accessed"))
    kw: dict[str, Any] = dict(
        reference_mode="periodic",
        serving_plan=LiveServingPlan(48),
        publish_on_drain=True,
        followup_plan=FollowupTrainingPlan(),
        followup_training_recipe=NOISY,
    )
    kw.update(changes)
    with pytest.raises(ValueError):
        demo.run_case(Path("unused"), **kw)


@pytest.mark.parametrize("name,recipe", FOLLOWUP_TRAINING_RECIPES)
def test_should_admit_explicit_recipe_metadata_before_native_work(monkeypatch, name, recipe):
    def source(path):
        raise ValueError("scripted source stop before native work")

    monkeypatch.setattr(demo.context, "_source", source)
    report = demo.run_case(
        Path("unused"),
        reference_mode="periodic",
        serving_plan=LiveServingPlan(48),
        publish_on_drain=True,
        followup_plan=FollowupTrainingPlan(),
        followup_training_recipe=recipe,
    )
    assert not report["complete"] and "scripted source stop" in report["error"]
    assert report["followup_training_corruption"]["planned_corrupted_exposures"] == (
        8 if name == "noisy" else 0
    )
    assert (
        report["final_examples"]
        == report["reserved_probe_updates"]
        == report["replay_examples"]
        == 0
    )


@pytest.mark.parametrize("name,recipe", FOLLOWUP_TRAINING_RECIPES)
def test_should_pass_explicit_cli_recipe_and_keep_default_old_flag_absent(
    tmp_path, monkeypatch, name, recipe
):
    calls = []

    def run(path, **kw):
        calls.append(kw)
        return dict(complete=True)

    monkeypatch.setattr(demo, "run_case", run)
    output = tmp_path / (name + ".json")
    assert (
        demo.main(
            [
                "--source",
                "saved.json",
                "--continue-training-after-publication",
                "--followup-training-recipe",
                name,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert calls[0]["followup_training_recipe"] == recipe
    assert calls[0]["followup_plan"] == FollowupTrainingPlan()
    with pytest.raises(SystemExit) as error:
        demo.main(
            [
                "--source",
                "saved.json",
                "--followup-training-recipe",
                name,
                "--output",
                str(tmp_path / "invalid.json"),
            ]
        )
    assert error.value.code == 2 and not (tmp_path / "invalid.json").exists()
