"""Incident stream/dispatch contracts on literal TRAIN; no native model work."""

from time import monotonic

import numpy as np
import pytest

from scripts import context_head_live as live
from scripts import context_head_rehearsal as source
from scripts import recurring_ticket_workflow_example as entry
from scripts.context_training_streams import (
    CONTEXT_INCIDENT_RECIPES,
    CONTEXT_PAIRED_REHEARSAL_RECIPES,
    CONTEXT_STREAM_RECIPES,
    build_context_stream,
    ContextTrainingRecipe,
)
from scripts.run_ticket_workflow import CONTEXT_FEATURES
from src.app.coupled_attribution import AttributionBatch
from src.app.toy_execution_budget import ToyBudgetSession
from src.core.temporal_decision_sequence import policy_for_phase
from src.core.ticket_workflow import ACTION_IDS
from src.shared.process_memory import ProcessRssSampler


def prototype(version, concepts, context):
    features, targets, ids = [], [], []
    rules = policy_for_phase("after_A" if version == "v1" else "after_return")["rules"]
    for concept in concepts:
        for stage in ("sealed", "ready", "staged"):
            words = {stage, f"policy_{version}"}
            if stage == "ready":
                words.add(concept)
            features.append([float(w in words) for w in CONTEXT_FEATURES])
            action = (
                "open" if stage == "sealed" else "commit" if stage == "staged" else rules[concept]
            )
            targets.append([float(ACTION_IDS.index(action))])
            ids.append(f"train:{version}:{concept}:{stage}")
    return AttributionBatch(
        "train", context, tuple(ids), np.array(features), np.array(targets), num_classes=6
    )


@pytest.mark.parametrize("index,total,last", [(0, 24, (3, 3)), (1, 36, (9, 9))])
def test_should_build_exact_owned_prefix_and_same_stream_control_packets(index, total, last):
    new = prototype("v1", ("parcel",), "new-v1")
    old = prototype("v1", ("invoice", "password"), "old-v1")
    second = (
        prototype("v2", ("password",), "new-v2"),
        prototype("v2", ("invoice", "parcel"), "old-v2"),
    )
    recipe = CONTEXT_INCIDENT_RECIPES[index]
    assert recipe not in CONTEXT_STREAM_RECIPES
    assert len(CONTEXT_PAIRED_REHEARSAL_RECIPES) == 2
    batches, noise = build_context_stream(recipe, new, old, second_head=second)
    assert len(batches) == 8 and sum(len(b.features) for b in batches) == total
    assert [len(b.features) for b in batches] == [3] * 6 + list(last)
    assert not noise and recipe.label_delay == 1
    assert all(b.role == "train" for b in batches)
    assert len({row for b in batches for row in b.row_ids}) == total
    assert all("v1:invoice:" in row for b in batches[:6] for row in b.row_ids)
    assert all(np.array_equal(b.features, batches[0].features) for b in batches[:6])
    batches[0].features[:] = -1
    assert (old.features >= 0).all() and (batches[1].features >= 0).all()


@pytest.mark.parametrize(
    "changes",
    [
        dict(incident_stream="bad"),
        dict(incident_stream=True),
        dict(adaptive_serving=False),
        dict(paired_serving="current"),
        dict(calibrated_serving=None),
        dict(calibrated_serving=1),
        dict(incident_stream=None),
        dict(method="perceptron"),
        dict(private_adaptation=True),
    ],
)
def test_should_refuse_invalid_incident_workflow_before_data(changes, monkeypatch):
    monkeypatch.setattr(entry, "prepare_training", lambda **kw: pytest.fail("data opened"))
    args = dict(
        method="linear_fit",
        context_return=True,
        separate_context_heads=True,
        paired_serving="joint",
        adaptive_serving=True,
        incident_stream="shift",
        calibrated_serving=True,
    )
    args.update(changes)
    with pytest.raises(ValueError):
        entry.run_example(**args)


@pytest.mark.parametrize("stream", ["stationary", "shift"])
@pytest.mark.parametrize("calibrated", [False, True])
def test_should_forward_owned_case_to_existing_adaptive_schedule(stream, calibrated, monkeypatch):
    calls = []
    guard = lambda: None
    monkeypatch.setattr(source, "prepare_owned_source", lambda state, *a, **kw: (state, guard))

    def run(path, **kwargs):
        calls.append((path, kwargs))
        return dict(complete=True)

    monkeypatch.setattr(live, "run_case", run)
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        budget = ToyBudgetSession(source._limits(12), monotonic)
        budget.attach_memory(memory)
        report = live.run_owned_case(
            None,
            None,
            {},
            budget,
            control="joint",
            check_boundary=guard,
            memory=memory,
            linear_fit=True,
            adaptive=True,
            incident_stream=stream,
            calibrated=calibrated,
        )
    assert report["complete"] and report["owned_source_verified"]
    assert budget.updates_completed == 0
    path, kw = calls[0]
    assert path is None and kw["_linear_fit"] and kw["reference_mode"] == "drift"
    assert (
        kw["candidate_feedback"] and kw["acknowledge_correct"] and not kw["publish_after_learning"]
    )
    assert kw["incident_stream"] == stream and kw["calibrated_serving"] is calibrated
    assert kw["stream_recipe"] == CONTEXT_INCIDENT_RECIPES[stream == "shift"]


@pytest.mark.parametrize("stream", ["stationary", "shift"])
@pytest.mark.parametrize("calibrated", [False, True])
def test_should_forward_cli_and_preserve_existing_output(stream, calibrated, tmp_path, monkeypatch):
    calls = []

    def example(**kwargs):
        calls.append(kwargs)
        return dict(complete=True)

    monkeypatch.setattr(entry, "run_example", example)
    output = tmp_path / "case.json"
    args = [
        "--output",
        str(output),
        "--method",
        "linear_fit",
        "--context-return",
        "--separate-context-heads",
        "--paired-serving",
        "joint",
        "--adaptive-serving",
        "--incident-stream",
        stream,
    ]
    if calibrated:
        args.append("--calibrated-serving")
    assert entry.main(args) == 0
    assert calls[0]["incident_stream"] == stream and calls[0]["calibrated_serving"] is calibrated
    with pytest.raises(FileExistsError):
        entry.main(args)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "options",
    [
        dict(calibrated_serving=True),
        dict(incident_stream="shift", stream_recipe=CONTEXT_INCIDENT_RECIPES[0]),
        dict(incident_stream="shift", calibrated_serving=1),
        dict(stream_recipe=CONTEXT_INCIDENT_RECIPES[1]),
    ],
)
def test_should_refuse_mismatched_live_controls_before_source(options, monkeypatch):
    monkeypatch.setattr(live, "_source_bytes", lambda p: pytest.fail("source opened"))
    args = dict(
        _linear_fit=True,
        _owned_source=(None, lambda: None),
        _memory=object(),
        reference_mode="drift",
        candidate_feedback=True,
        acknowledge_correct=True,
    )
    args.update(options)
    with pytest.raises(ValueError, match="incident"):
        live.run_case(None, **args)


def test_should_refuse_prefix_recipe_alias_and_missing_second_head():
    with pytest.raises(ValueError):
        ContextTrainingRecipe("other", ("prefix-v1",) * 8, 1)
    with pytest.raises(ValueError, match="second-head"):
        build_context_stream(CONTEXT_INCIDENT_RECIPES[0], None, None)
