"""Named contextual TRAIN recipes, default preservation and paid-case admission."""

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts import context_head_live as live
from scripts import context_recipe_validation as validation
from scripts import run_bp_ticket_serving as cli
from scripts.context_head_serving import prepare_data, TASKS
from scripts.context_training_streams import (
    CONTEXT_TRAINING_RECIPES,
    build_context_stream,
)
from scripts.run_ticket_workflow import CONTEXT_FEATURES
from src.core.decision_features import DecisionFeatureSpec
from src.core.ticket_workflow import ACTION_IDS, CATALOG

SOURCE = Path("artifacts/runs/context-head-control-20261023/actual.json")


@pytest.fixture(scope="module")
def prototypes():
    return prepare_data(DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG))[0]


@pytest.mark.parametrize(
    "changes",
    [
        dict(contexts=[]),
        dict(contexts=()),
        dict(contexts=("final_test",)),
        dict(contexts=("new-v1",) * 9),
        dict(label_delay=True),
        dict(label_delay=3),
        dict(noisy_old_invoice=1),
        dict(contexts=("new-v1",), noisy_old_invoice=True),
    ],
)
def test_should_refuse_invalid_recipe_before_io(changes, monkeypatch):
    monkeypatch.setattr(live, "_source_bytes", lambda p: pytest.fail("source entered"))
    with pytest.raises(ValueError):
        replace(CONTEXT_TRAINING_RECIPES[0], **changes)
    with pytest.raises(ValueError):
        live.run_case("unused", stream_recipe={})


@pytest.mark.parametrize("recipe", CONTEXT_TRAINING_RECIPES)
def test_should_build_owned_named_train_packets_without_mutating_clean_truth(recipe, prototypes):
    new, old = prototypes
    before = deepcopy(prototypes)
    packets, noise = build_context_stream(recipe, new, old)
    assert len(packets) == 6 and [len(p.features) for p in packets] == [3, 6] * 3
    assert sum(len(p.features) for p in packets) == 27
    assert len({i for p in packets for i in p.row_ids}) == 27
    for i, p in enumerate(packets):
        base = prototypes[i % 2]
        assert p.role == "train" and p.num_classes == 6 and p.episode is None
        assert not np.shares_memory(p.features, base.features) and not np.shares_memory(
            p.targets, base.targets
        )
        assert np.array_equal(p.features, base.features)
        changed = np.flatnonzero(p.targets[:, 0] != base.targets[:, 0])
        if recipe.noisy_old_invoice and i % 2:
            assert len(changed) == 1 and p.row_ids[changed[0]].endswith("v1:invoice:ready")
        else:
            assert len(changed) == 0
    assert len(noise) == (3 if recipe.noisy_old_invoice else 0)
    assert all(n["clean_action"] == "route" and n["corrupted_action"] == "defer" for n in noise)
    assert all(
        np.array_equal(a.targets, b.targets) for a, b in zip(prototypes, before, strict=True)
    )


def test_should_corrupt_named_row_after_reordering_instead_of_native_offset(prototypes):
    new, old = prototypes
    order = [5, 4, 3, 2, 0, 1]
    old = replace(
        old,
        features=old.features[order].copy(),
        targets=old.targets[order].copy(),
        row_ids=tuple(old.row_ids[i] for i in order),
    )
    packets, noise = build_context_stream(CONTEXT_TRAINING_RECIPES[1], new, old)
    assert np.flatnonzero(packets[1].targets[:, 0] != old.targets[:, 0]).tolist() == [5]
    assert (
        noise[0]["source_row_id"] == old.row_ids[5]
        and noise[0]["semantic_row"] == "v1:invoice:ready"
    )


@pytest.mark.parametrize("role", ["final_test", "probe_train", "inner_guard", "outer_selection"])
def test_should_never_relabel_other_roles_as_train(role, prototypes):
    with pytest.raises(ValueError):
        build_context_stream(
            CONTEXT_TRAINING_RECIPES[0], replace(prototypes[0], role=role), prototypes[1]
        )


@pytest.mark.parametrize("fault", ["shape", "truth", "context", "duplicate"])
def test_should_refuse_foreign_geometry_truth_or_row_identity(fault, prototypes):
    new, old = deepcopy(prototypes)
    if fault == "shape":
        new = replace(new, features=new.features[:, :7])
    elif fault == "truth":
        old.targets[1, 0] = ACTION_IDS.index("defer")
    elif fault == "context":
        old.features[:, CONTEXT_FEATURES.index("policy_v1")] = 0
    else:
        old.features[1] = old.features[0]
        old.targets[1] = old.targets[0]
    with pytest.raises(ValueError):
        build_context_stream(CONTEXT_TRAINING_RECIPES[0], new, old)


def test_should_bind_recipe_plan_delay_and_budget_once_without_model_updates(monkeypatch):
    seen = []

    def schedule(source, actor, batches, budget, mode, report, *args, **kwargs):
        seen.append(
            (
                len(batches),
                budget.budget.max_training_updates,
                budget.budget.max_wall_seconds,
                kwargs["label_delay"],
                kwargs["live"].plan.opportunities,
            )
        )
        for _ in batches:
            budget.record_update()
        return SimpleNamespace(applied_updates=[]), 24

    monkeypatch.setattr(cli, "_reference_schedule", schedule)
    r = live.run_case(SOURCE, stream_recipe=CONTEXT_TRAINING_RECIPES[0])
    assert r["complete"] and r["training_complete"] and seen == [(6, 6, 60, 2, 24)]
    assert r["budget"]["updates"] == 6 and r["source_unchanged"]
    assert tuple(r["live_plan"]["context_tasks"][17:23]) == TASKS
    assert (
        r["training_work"]["attempted_updates"] == 0
    )  # Scripted budget fixture, not scientific training.


def test_should_keep_nonperiodic_pending_training_explicit(monkeypatch):
    monkeypatch.setattr(
        cli, "_reference_schedule", lambda *a, **k: (SimpleNamespace(applied_updates=[]), 24)
    )
    r = live.run_case(
        SOURCE, reference_mode="no_learning", stream_recipe=CONTEXT_TRAINING_RECIPES[0]
    )
    assert r["complete"] and not r["training_complete"] and r["budget"]["updates"] == 0


def scripted(recipe):
    return dict(
        complete=True,
        source_sha256=validation.HEAD_SOURCE_SHA,
        source_unchanged=True,
        context_recipe=asdict(recipe),
        training_work=dict(attempted_updates=6, attempted_rows=27, returned_updates=6),
    )


def test_should_execute_only_two_declared_cases_in_order(monkeypatch):
    seen = []

    def case(path, *, stream_recipe):
        seen.append(stream_recipe.recipe_id)
        return scripted(stream_recipe)

    monkeypatch.setattr(live, "run_case", case)
    r = validation.run_validation(SOURCE)
    assert r["complete"] and seen == [p.recipe_id for p in CONTEXT_TRAINING_RECIPES]


@pytest.mark.parametrize("failure", ["incomplete", "wrong_recipe"])
def test_should_retain_failed_paid_case_and_stop_admission(failure, monkeypatch):
    calls = []

    def case(path, *, stream_recipe):
        calls.append(1)
        r = scripted(stream_recipe)
        if failure == "incomplete":
            r["complete"] = False
        else:
            r["context_recipe"]["noisy_old_invoice"] = True
        return r

    monkeypatch.setattr(live, "run_case", case)
    r = validation.run_validation(SOURCE)
    assert not r["complete"] and len(calls) == len(r["cases"]) == 1


def test_should_restore_corrupted_prior_evidence_and_stop_following_work(monkeypatch):
    first: list[dict[str, Any]] = []

    def case(path, *, stream_recipe):
        r = scripted(stream_recipe)
        if first:
            first[0]["training_work"]["attempted_rows"] = 999
        else:
            first.append(r)
        return r

    monkeypatch.setattr(live, "run_case", case)
    r = validation.run_validation(SOURCE)
    assert not r["complete"] and "corrupted" in r["error"]
    assert len(r["cases"]) == 2 and r["cases"][0]["training_work"]["attempted_rows"] == 27


def test_should_require_explicit_cli_mode_for_recipe_and_dispatch_validation(tmp_path, monkeypatch):
    output = tmp_path / "out.json"
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--source",
                str(SOURCE),
                "--output",
                str(output),
                "--context-stream-recipe",
                CONTEXT_TRAINING_RECIPES[0].recipe_id,
            ]
        )
    assert not output.exists()
    seen = []

    def case(path, **kwargs):
        seen.append(kwargs)
        return dict(complete=True)

    monkeypatch.setattr(live, "run_case", case)
    assert (
        cli.main(
            [
                "--source",
                str(SOURCE),
                "--output",
                str(output),
                "--context-head-serving-live",
                "--context-stream-recipe",
                CONTEXT_TRAINING_RECIPES[0].recipe_id,
            ]
        )
        == 0
    )
    assert seen == [dict(stream_recipe=CONTEXT_TRAINING_RECIPES[0])]
    monkeypatch.setattr(validation, "run_validation", lambda path: dict(complete=True))
    assert (
        cli.main(
            [
                "--source",
                str(SOURCE),
                "--output",
                str(tmp_path / "validation.json"),
                "--context-recipe-validation",
            ]
        )
        == 0
    )
