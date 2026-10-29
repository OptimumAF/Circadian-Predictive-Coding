"""Equal-row TRAIN packet composition and original live runner, literal source."""

from copy import deepcopy
from dataclasses import replace, asdict

import numpy as np
import pytest

from scripts import context_head_live as live
from scripts.context_training_streams import (
    CONTEXT_REHEARSAL_RECIPES,
    ContextTrainingRecipe,
    build_context_stream,
)
from src.core.ticket_workflow import ACTION_IDS
import test_context_catalog_control as fixture_module
from test_context_catalog_replay import prototype

source = fixture_module.source


@pytest.fixture
def packets():
    template = prototype(("legacy", "password"))
    targets = template.targets[:3].copy()
    targets[1, 0] = ACTION_IDS.index("retire")
    new = replace(
        template,
        context="new-v1",
        row_ids=tuple(f"update-0-{i}" for i in range(3)),
        features=template.features[:3].copy(),
        targets=targets,
    )
    return new, prototype()


@pytest.mark.parametrize("index", [0, 1])
def test_should_own_exact_equal_row_learning_packets_and_common_old_return(index, packets):
    new, old = packets
    pin = deepcopy(packets)
    recipe = CONTEXT_REHEARSAL_RECIPES[index]
    rows, noise = build_context_stream(recipe, new, old)
    assert noise == [] and recipe.label_delay == 1
    assert [len(b.features) for b in rows] == [9, 9, 6]
    assert len({identifier for b in rows for identifier in b.row_ids}) == 24
    expected_x = np.concatenate((new.features,) * 3 if index == 0 else (new.features, old.features))
    expected_y = np.concatenate((new.targets,) * 3 if index == 0 else (new.targets, old.targets))
    for b in rows[:2]:
        assert b.role == "train" and b.episode is None and b.num_classes == 6
        assert b.features.dtype == b.targets.dtype == np.float64
        assert np.array_equal(b.features, expected_x) and np.array_equal(b.targets, expected_y)
        assert not np.shares_memory(b.features, new.features) and not np.shares_memory(
            b.features, old.features
        )
    assert not np.shares_memory(rows[0].targets, rows[1].targets)
    assert np.array_equal(rows[-1].features, old.features)
    assert np.array_equal(rows[-1].targets, old.targets)
    assert all(np.array_equal(a.targets, b.targets) for a, b in zip(packets, pin, strict=True))
    assert rows[0].row_ids[0].endswith("exposure0:v1:legacy:sealed")
    assert (
        rows[0]
        .row_ids[3]
        .endswith("exposure1:v1:legacy:sealed" if index == 0 else "exposure1:v1:invoice:sealed")
    )
    new.features[:] = old.features[:] = 0
    assert np.array_equal(rows[0].features, expected_x)


def test_should_keep_old_named_return_semantics_when_prototype_order_changes(packets):
    new, old = packets
    order = [5, 4, 3, 2, 1, 0]
    changed = replace(
        old,
        features=old.features[order].copy(),
        targets=old.targets[order].copy(),
        row_ids=tuple(old.row_ids[i] for i in order),
    )
    rows, _ = build_context_stream(CONTEXT_REHEARSAL_RECIPES[1], new, changed)
    assert rows[0].row_ids[3].endswith("exposure1:v1:password:staged")
    assert np.array_equal(rows[0].features[3:], changed.features)
    assert np.array_equal(rows[-1].targets, changed.targets)


@pytest.mark.parametrize("context", ["current9-v1", "joint-v1"])
def test_should_bound_eight_packets_to_seventy_two_rows(context, packets):
    rows, _ = build_context_stream(ContextTrainingRecipe("max9", (context,) * 8), *packets)
    assert len(rows) == 8 and sum(len(b.features) for b in rows) == 72
    assert len({i for b in rows for i in b.row_ids}) == 72


@pytest.mark.parametrize("role", ["final_test", "probe_train", "inner_guard", "outer_selection"])
def test_should_never_convert_other_roles_to_joint_train(role, packets):
    for part in (0, 1):
        changed = list(packets)
        changed[part] = replace(changed[part], role=role)
        with pytest.raises(ValueError):
            build_context_stream(CONTEXT_REHEARSAL_RECIPES[1], *changed)


@pytest.mark.parametrize(
    "fault", ["truth", "ids", "geometry", "context", "missing_context", "nan", "foreign_episode"]
)
def test_should_validate_original_prototypes_before_composition(fault, packets):
    new, old = packets
    if fault == "truth":
        old.targets[1, 0] = ACTION_IDS.index("retire")
    elif fault == "ids":
        new = replace(new, row_ids=("other", *new.row_ids[1:]))
    elif fault == "geometry":
        new = replace(new, features=new.features[:, :7])
    elif fault == "context":
        old = replace(old, context="new-v1")
    elif fault == "missing_context":
        old.features[:, 4] = 0
    elif fault == "nan":
        old.features[0, 0] = np.nan
    else:
        old = replace(old, episode=object())
    with pytest.raises(ValueError):
        build_context_stream(CONTEXT_REHEARSAL_RECIPES[1], new, old)


@pytest.mark.parametrize(
    "changes",
    [
        dict(contexts=("joint-v1",) * 9),
        dict(contexts=("joint-v1", "old-v1"), noisy_old_invoice=True),
        dict(contexts=("current9-v1", "old-v1"), noisy_old_invoice=True),
        dict(label_delay=True),
        dict(contexts=("joint-v2",)),
    ],
)
def test_should_refuse_invalid_or_ambiguous_modes_before_io(changes, monkeypatch):
    monkeypatch.setattr(live, "_source_bytes", lambda *a: pytest.fail("source opened"))
    with pytest.raises(ValueError):
        replace(CONTEXT_REHEARSAL_RECIPES[1], **changes)


@pytest.mark.parametrize("index", [0, 1])
def test_should_execute_learning_and_return_through_original_live_runner(source, index):
    recipe = CONTEXT_REHEARSAL_RECIPES[index]
    result = live.run_case("unused", stream_recipe=recipe)
    assert result["complete"], result.get("error")
    assert result["training_complete"] and result["budget"]["updates"] == 3
    assert result["training_work"] == dict(
        attempted_updates=3, attempted_rows=24, returned_updates=3
    )
    assert [c["rows"] for c in result["native_calls"]] == [9, 9, 6]
    assert all(c["complete"] for c in result["native_calls"])
    assert result["candidate_state"]["completed_updates"] == 15
    assert result["candidate_state"]["training_examples"] == 96
    assert result["candidate_state"]["heads"][1] == asdict(source.heads[1])
    assert result["source_unchanged"] and result["synthetic_label_noise"] == []
    assert result["replay_examples"] == result["probe_updates"] == result["final_examples"] == 0
    assert result["stream"]["pending_unapplied"] == 0 and result["stream"]["arrived_labels"] == 3
    assert len(result["live_serving"]["records"]) == 24
    for i, receipt in enumerate(result["applied_updates"]):
        assert receipt["sample_id"] == f"batch-{i}" and receipt["event_id"] == f"label-{i}"
        assert receipt["observed_at"] == i and receipt["arrived_at"] == i + 1
        assert receipt["applied_at"] >= receipt["arrived_at"]
    if "rollback_equal_source" in result:
        assert result["rollback_equal_source"]


def test_should_preserve_failed_native_work_without_claiming_complete_return(source, monkeypatch):
    original = live.ContextPerceptronLearner.train_batch

    def failed(owner, features, targets):
        original(owner, features, targets)
        raise RuntimeError("lost joint native return")

    monkeypatch.setattr(live.ContextPerceptronLearner, "train_batch", failed)
    result = live.run_case("unused", stream_recipe=CONTEXT_REHEARSAL_RECIPES[1])
    assert not result["complete"] and "lost joint native return" in result["error"]
    assert result["training_work"] == dict(
        attempted_updates=1, attempted_rows=9, returned_updates=0
    )
    assert result["budget"]["updates"] == 0 and result["applied_updates"] == []
    assert result["source_unchanged"]
