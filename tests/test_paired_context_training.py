"""Ordinary TRAIN paired-head recipes; private probe data is never used."""

from copy import deepcopy
from dataclasses import replace, asdict

import numpy as np
import pytest

from scripts import context_head_serving as serving
from scripts import context_head_live as live
from scripts import context_training_streams as streams
from scripts.context_stream_comparison import protocol_snapshot
from src.app.coupled_attribution import _same_value
from src.core.decision_features import DecisionFeatureSpec
from src.core.ticket_workflow import CATALOG
from test_context_catalog_control import source as source_fixture


@pytest.fixture
def spec():
    return DecisionFeatureSpec(serving.CONTEXT_FEATURES, CATALOG)


@pytest.fixture
def packets(spec):
    return serving.prepare_data(spec, head_task_transfer=True)


@pytest.fixture
def source(monkeypatch):
    return source_fixture.__wrapped__(monkeypatch)


def build(recipe, packets):
    return streams.build_context_stream(recipe, *packets[:2], second_head=packets[2:])


def test_should_prepare_four_owned_train_prototypes_and_matching_inner_guards(spec, monkeypatch):
    from scripts import context_head_probe
    from scripts import context_head_rehearsal

    monkeypatch.setattr(
        context_head_probe, "build_head_cohorts", lambda *a, **kw: pytest.fail("probe data used")
    )
    monkeypatch.setattr(
        context_head_rehearsal, "prepare_cohorts", lambda *a, **kw: pytest.fail("private bank used")
    )
    packets, guards = serving.prepare_data(spec, head_task_transfer=True)
    assert [b.context for b in packets] == ["new-v1", "old-v1", "new-v2", "old-v2"]
    assert [len(b.features) for b in packets] == [3, 6, 3, 6]
    assert all(b.role == "train" and b.episode is None and b.num_classes == 6 for b in packets)
    assert len({i for b in packets for i in b.row_ids}) == 18
    assert [len(b.features) for b in guards] == [6, 12]
    assert all(b.role == "inner_guard" for b in guards)
    assert np.array_equal(
        guards[0].features, np.concatenate((packets[0].features, packets[2].features))
    )
    assert np.array_equal(
        guards[1].targets, np.concatenate((packets[1].targets, packets[3].targets))
    )
    arrays = [a for b in (*packets, *guards) for a in (b.features, b.targets)]
    assert all(not np.shares_memory(a, b) for i, a in enumerate(arrays) for b in arrays[i + 1 :])


@pytest.mark.parametrize("index", [0, 1])
def test_should_match_frozen_six_packet_rows_and_heads_with_common_return(index, packets):
    data, _ = packets
    pin = deepcopy(data)
    recipe = streams.CONTEXT_PAIRED_REHEARSAL_RECIPES[index]
    rows, noise = build(recipe, data)
    assert noise == [] and recipe.label_delay == 1
    assert [len(b.features) for b in rows] == [9, 9, 9, 9, 6, 6]
    assert sum(len(b.features) for b in rows) == len({i for b in rows for i in b.row_ids}) == 48
    assert all(b.role == "train" and b.episode is None and b.num_classes == 6 for b in rows)
    for i, row in enumerate(rows[:4]):
        new, old = data[(i % 2) * 2 : (i % 2) * 2 + 2]
        expected = np.concatenate(
            (new.features,) * 3 if index == 0 else (new.features, old.features)
        )
        assert np.array_equal(row.features, expected)
    assert np.array_equal(rows[4].features, data[1].features)
    assert np.array_equal(rows[5].targets, data[3].targets)
    assert _same_value(data, pin)
    assert [sum(np.count_nonzero(b.features[:, c]) for b in rows) for c in (4, 5)] == [24, 24]
    assert [sum(np.count_nonzero(b.features[:, c]) > 0 for b in rows) for c in (4, 5)] == [3, 3]
    arrays = [a for b in rows for a in (b.features, b.targets)]
    assert all(not np.shares_memory(a, b) for i, a in enumerate(arrays) for b in arrays[i + 1 :])
    data[0].features[:] = 0
    assert np.array_equal(
        rows[0].features,
        np.concatenate(
            (pin[0].features,) * 3 if index == 0 else (pin[0].features, pin[1].features)
        ),
    )


@pytest.mark.parametrize(
    "fault", ["role", "truth", "ids", "head", "missing", "duplicate", "order", "episode"]
)
def test_should_refuse_foreign_or_corrupted_second_head_before_composition(fault, packets):
    data, _ = packets
    data = list(data)
    b = data[2]
    if fault == "role":
        b = replace(b, role="probe_train")
    elif fault == "truth":
        b.targets[1, 0] = serving.ACTION_IDS.index("retire")
    elif fault == "ids":
        b = replace(b, row_ids=("foreign", *b.row_ids[1:]))
    elif fault == "head":
        b.features[:, 4], b.features[:, 5] = 1, 0
    elif fault == "missing":
        b = replace(b, features=b.features[:2], targets=b.targets[:2], row_ids=b.row_ids[:2])
    elif fault == "duplicate":
        b = replace(b, row_ids=(b.row_ids[0],) * 3)
    elif fault == "order":
        b = replace(b, targets=b.targets[::-1].copy())
    else:
        b = replace(b, episode=object())
    data[2] = b
    with pytest.raises(ValueError):
        build(streams.CONTEXT_PAIRED_REHEARSAL_RECIPES[1], tuple(data))


@pytest.mark.parametrize("role", ["final_test", "probe_train", "inner_guard", "outer_selection"])
def test_should_never_relabel_foreign_teacher_cases_as_ordinary_train(spec, monkeypatch, role):
    original = serving.training_cases
    monkeypatch.setattr(
        serving, "training_cases", lambda **kw: tuple(replace(c, role=role) for c in original(**kw))
    )
    with pytest.raises(ValueError):
        serving.prepare_data(spec, head_task_transfer=True)


@pytest.mark.parametrize("index", [0, 1])
def test_should_preview_both_heads_tasks_and_exact_original_budget_without_io(index, monkeypatch):
    recipe = streams.CONTEXT_PAIRED_REHEARSAL_RECIPES[index]
    monkeypatch.setattr(live, "_source_bytes", lambda p: pytest.fail("source read"))
    monkeypatch.setattr(live, "prepare_data", lambda *a, **kw: pytest.fail("teacher entered"))
    result = protocol_snapshot(recipe=recipe)
    assert result["complete"]
    assert result["training_exposures"] == dict(
        updates=6, rows=48, new_rows=36 if index == 0 else 12, old_rows=12 if index == 0 else 36
    )
    assert result["wake_budget"] == dict(
        max_training_updates=6, max_wall_seconds=60, max_process_rss_bytes=512 * 1024**2
    )
    assert len(result["plan"]["context_tasks"]) == 24
    assert not any(c == "legacy" for _, c in result["plan"]["context_tasks"])
    assert asdict(live.recipe_plan()) != result["plan"]


@pytest.mark.parametrize(
    "kw",
    [
        dict(reference_mode="drift"),
        dict(candidate_feedback=True),
        dict(task_feedback=True),
        dict(arrived_shift=True),
    ],
)
def test_should_refuse_unsupported_paired_feedback_before_source_io(monkeypatch, kw):
    monkeypatch.setattr(live, "_source_bytes", lambda p: pytest.fail("source read"))
    with pytest.raises(ValueError):
        live.run_case("unused", stream_recipe=streams.CONTEXT_PAIRED_REHEARSAL_RECIPES[1], **kw)


def test_should_keep_legacy_factory_and_recipes_and_refuse_implicit_paired_input(spec, packets):
    old, guards = serving.prepare_data(spec)
    assert [b.context for b in old] == ["new-v1", "old-v1"]
    assert all("legacy" in i or "update-" in i for b in old for i in b.row_ids)
    rows, _ = streams.build_context_stream(streams.CONTEXT_REHEARSAL_RECIPES[1], *old)
    assert [len(b.features) for b in rows] == [9, 9, 6]
    data, _ = packets
    with pytest.raises(ValueError):
        streams.build_context_stream(streams.CONTEXT_PAIRED_REHEARSAL_RECIPES[1], *data[:2])
    with pytest.raises(ValueError):
        build(streams.ContextTrainingRecipe("arbitrary", ("joint-v1", "joint-v2"), 1), data)


@pytest.mark.parametrize("index", [0, 1])
def test_should_execute_existing_live_train_budget_and_preserve_source_and_rollback(source, index):
    rng = deepcopy(np.random.get_state())
    result = live.run_case(
        "literal-source", stream_recipe=streams.CONTEXT_PAIRED_REHEARSAL_RECIPES[index]
    )
    assert result["complete"], result.get("error")
    assert result["training_work"] == dict(
        attempted_updates=6, attempted_rows=48, returned_updates=6
    )
    assert result["train_task_family"] == "paired_head_tasks_v1"
    assert result["declared_tasks"] == serving.PAIRED_TASKS
    assert result["source_unchanged"] and _same_value(np.random.get_state(), rng)
    assert result["final_examples"] == result["probe_updates"] == result["replay_examples"] == 0
    assert result["budget"]["updates"] == result["budget"]["limits"]["max_training_updates"] == 6
    assert result["candidate_state"]["completed_updates"] == 18
    assert result["candidate_state"]["training_examples"] == 120
    assert [
        (h["completed_updates"], h["training_examples"]) for h in result["candidate_state"]["heads"]
    ] == [(11, 72), (7, 48)]
    assert len(result["applied_updates"]) == 6
    assert [(r["policy_version"], r["concept"]) for r in result["live_serving"]["records"]] == list(
        live.recipe_plan(head_task_transfer=True).context_tasks
    )
    assert result["final_actor"]["model_state"] == asdict(source)
    if "promotion_receipt" in result:
        assert result["rollback_equal_source"]
        assert [(e["policy_version"], e["concept"]) for e in result["episodes"]] == list(
            serving.PAIRED_TASKS
        )


@pytest.mark.parametrize("value", [1, None, "yes"])
def test_should_refuse_nonboolean_data_optin_before_teacher(spec, monkeypatch, value):
    monkeypatch.setattr(serving, "training_cases", lambda **kw: pytest.fail("teacher entered"))
    with pytest.raises(ValueError):
        serving.prepare_data(spec, head_task_transfer=value)


def test_should_keep_failed_paired_native_charge_without_publication(source, monkeypatch):
    original = live.ContextPerceptronLearner.train_batch

    def failed(owner, features, targets):
        original(owner, features, targets)
        raise RuntimeError("lost paired TRAIN return")

    monkeypatch.setattr(live.ContextPerceptronLearner, "train_batch", failed)
    result = live.run_case(
        "literal-source", stream_recipe=streams.CONTEXT_PAIRED_REHEARSAL_RECIPES[1]
    )
    assert not result["complete"] and "lost paired TRAIN return" in result["error"]
    assert result["training_work"] == dict(
        attempted_updates=1, attempted_rows=9, returned_updates=0
    )
    assert result["budget"]["updates"] == 0 and result["applied_updates"] == []
    assert "promotion_receipt" not in result and result["source_unchanged"]


def test_should_keep_original_no_learning_actor_and_zero_update_budget_consumption(source):
    result = live.run_case(
        "literal-source",
        reference_mode="no_learning",
        stream_recipe=streams.CONTEXT_PAIRED_REHEARSAL_RECIPES[1],
    )
    assert result["complete"], result.get("error")
    assert result["training_work"] == dict(
        attempted_updates=0, attempted_rows=0, returned_updates=0
    )
    assert result["budget"]["updates"] == 0 and result["source_unchanged"]
    assert result["final_actor"]["model_state"] == asdict(source)


def test_should_cover_all_paired_tasks_in_late_ordinary_worker_slots():
    from scripts.ticket_serving_timeline import serving_preview

    rows = serving_preview(live.recipe_plan(head_task_transfer=True))
    late = [r for r in rows if r["tick"] >= 13 and r["scenario"] == "ordinary"]
    assert [r["tick"] for r in late] == [13, 15, 16, 17, 18, 19, 21, 23]
    assert {(r["policy_version"], r["concept"]) for r in late} == set(serving.PAIRED_TASKS)
    original = serving_preview(live.recipe_plan())
    assert [(r["scenario"], r["contention"]) for r in rows] == [
        (r["scenario"], r["contention"]) for r in original
    ]
