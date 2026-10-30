"""Owned live catalog/readout ports; no model training or task evaluation."""

from time import monotonic
from typing import Any

import pytest

from scripts import context_catalog_control as control
from scripts import context_controller_learning as readouts
from scripts import context_head_live as live
from scripts import context_head_rehearsal as source
from scripts import recurring_ticket_workflow_example as entry
from src.adapters import context_linear_serving as binding
from src.adapters.context_linear_fit_learner import ContextLinearFitLearner
from src.adapters.context_perceptron import ContextPerceptronLearner
from src.app.counterfactual_controller import CatalogCounterfactualRequest
from src.app.toy_execution_budget import ToyBudgetSession
from src.core.context_linear_state import ContextLinearFitPolicy
from src.core.context_perceptron import ContextPerceptronPolicy
from src.shared.process_memory import ProcessRssSampler


def options():
    return dict(
        method="linear_fit",
        context_return=True,
        separate_context_heads=True,
        catalog_action="replay6-wake1",
        catalog_context="mistakes",
    )


@pytest.mark.parametrize(
    "change",
    [
        dict(catalog_action=True),
        dict(catalog_action="other"),
        dict(catalog_context=None),
        dict(catalog_context="final_test"),
        dict(method="perceptron"),
        dict(context_return=False),
        dict(separate_context_heads=False),
        dict(private_adaptation=True),
        dict(paired_serving="joint"),
        dict(adaptive_serving=True),
        dict(incident_stream="shift"),
        dict(replay="recent"),
        dict(record_train_fit=True),
        dict(catalog_action=None),
    ],
)
def test_should_refuse_invalid_catalog_api_before_data(change, monkeypatch):
    monkeypatch.setattr(entry, "prepare_training", lambda **kw: pytest.fail("data opened"))
    args = options()
    args.update(change)
    with pytest.raises(ValueError):
        entry.run_example(**args)


@pytest.mark.parametrize("action", ["wait", "wake1", "replay6-wake1"])
@pytest.mark.parametrize("context", ["clean", "mistakes"])
def test_should_forward_cli_to_owned_catalog_and_preserve_output(
    action, context, tmp_path, monkeypatch
):
    calls = []

    def example(**kwargs):
        calls.append(kwargs)
        return dict(complete=True, training_work=dict(attempted_updates=0, attempted_rows=0))

    monkeypatch.setattr(entry, "run_example", example)
    path = tmp_path / "out.json"
    args = [
        "--output",
        str(path),
        "--method",
        "linear_fit",
        "--context-return",
        "--separate-context-heads",
        "--catalog-action",
        action,
        "--catalog-context",
        context,
    ]
    assert entry.main(args) == 0
    assert calls == [
        dict(
            context_return=True,
            separate_context_heads=True,
            method="linear_fit",
            catalog_action=action,
            catalog_context=context,
        )
    ]
    with pytest.raises(FileExistsError):
        entry.main(args)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "flags",
    [
        ["--catalog-action", "wake1"],
        ["--catalog-context", "clean"],
        ["--catalog-action", "wake1", "--catalog-context", "clean", "--method", "perceptron"],
    ],
)
def test_should_refuse_missing_cli_prerequisites_before_output(flags, tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "run_example", lambda **kw: pytest.fail("source opened"))
    path = tmp_path / "never.json"
    with pytest.raises(SystemExit):
        entry.main(["--output", str(path), *flags])
    assert not path.exists()


def test_should_build_trial_from_current_qualified_owned_source(monkeypatch):
    calls = []
    state = object()
    guard = lambda: None
    monkeypatch.setattr(source, "prepare_owned_source", lambda value, *a, **kw: (value, guard))
    monkeypatch.setattr(binding, "snapshot_digest", lambda value: "current-owned-digest")

    def run(path, **kwargs):
        calls.append((path, kwargs))
        return dict(complete=True)

    monkeypatch.setattr(live, "run_case", run)
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        budget = ToyBudgetSession(source._limits(12), monotonic)
        budget.attach_memory(memory)
        result = live.run_owned_case(
            state,
            None,
            {},
            budget,
            control="catalog",
            linear_fit=True,
            catalog_action="replay6-wake1",
            catalog_context="mistakes",
            check_boundary=guard,
            memory=memory,
        )
    assert result["owned_source_verified"] and budget.updates_completed == 0
    path, kw = calls[0]
    assert path is None and kw["stream_recipe"] is None and not kw["publish_after_learning"]
    assert kw["_owned_source"] == (state, guard) and kw["catalog_base"] is state
    assert kw["_linear_fit"] and kw["resume_catalog_wake"]
    req = kw["catalog_trial"]
    assert req.learner_version == "context-ls-12" and req.state_digest == "current-owned-digest"
    assert req.context == "mistakes" and req.action_id == "replay6-wake1"
    assert req.catalog == control.context_catalog()


@pytest.mark.parametrize(
    "changes",
    [
        dict(linear_fit=False),
        dict(adaptive=True),
        dict(catalog_context=None),
        dict(catalog_action=1),
        dict(control="joint"),
    ],
)
def test_should_refuse_invalid_owned_catalog_before_qualification(changes, monkeypatch):
    monkeypatch.setattr(
        source, "prepare_owned_source", lambda *a, **kw: pytest.fail("source opened")
    )
    args = dict(control="catalog", linear_fit=True, catalog_action="wake1", catalog_context="clean")
    args.update(changes)
    result = live.run_owned_case(
        None, None, {}, None, check_boundary=lambda: None, memory=None, **args
    )
    assert result["complete"] is False and result["training_work"]["attempted_updates"] == 0


@pytest.mark.parametrize("linear", [False, True])
def test_should_forward_explicit_family_to_live_catalog_before_source(linear, monkeypatch):
    calls = []

    def constructor(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("stop before source")

    monkeypatch.setattr(control, "ContextCatalogControl", constructor)
    monkeypatch.setattr(live, "_source_bytes", lambda p: pytest.fail("source opened"))
    req = CatalogCounterfactualRequest(
        "context-ls-12" if linear else "qualified-12",
        "digest",
        "mistakes",
        "wake1",
        0,
        catalog=control.context_catalog(),
    )
    extra = (
        dict(_linear_fit=True, _owned_source=(object(), lambda: None), _memory=object())
        if linear
        else {}
    )
    with pytest.raises(RuntimeError, match="stop before source"):
        live.run_case(None, catalog_trial=req, catalog_base=object(), **extra)
    assert calls[0].get("linear_fit", False) is linear


@pytest.mark.parametrize("linear", [False, True])
@pytest.mark.parametrize("lost", [False, True])
def test_should_use_family_private_readout_and_retain_failed_native_costs(
    linear, lost, monkeypatch
):
    learner = (
        ContextLinearFitLearner(ContextLinearFitPolicy(9, 6, (4, 5)))
        if linear
        else ContextPerceptronLearner(ContextPerceptronPolicy(9, 6, (4, 5)))
    )
    state = learner.snapshot_state()
    report: dict[str, Any] = dict(task_outcomes=dict(before=[]), task_readout_work={})

    def episodes(reader, phase, check, rows, **kwargs):
        check()
        assert kwargs == dict(role="controller_train")
        assert phase == "before" and rows is report["task_outcomes"]["before"]
        if lost:
            raise RuntimeError("lost private readout")

    monkeypatch.setattr(readouts, "_episodes", episodes)
    monkeypatch.setattr(readouts, "_task_counts", lambda *a: [0, 0])
    if lost:
        with pytest.raises(RuntimeError, match="lost private"):
            readouts._task_readout(state, "before", lambda: None, report, linear_fit=linear)
    else:
        assert readouts._task_readout(state, "before", lambda: None, report, linear_fit=linear) == [
            0,
            0,
        ]
    assert report["task_readout_work"]["before"]["attempts"] == 0
    assert ("task_readout_native_work" in report) is linear
    if linear:
        assert report["task_readout_native_work"]["before"]["decision_attempts"] == 0


@pytest.mark.parametrize("flag", [None, 0, 1, "yes"])
def test_should_refuse_nonboolean_readout_family_before_state_access(flag):
    with pytest.raises(ValueError, match="Boolean"):
        readouts._task_readout(None, None, None, None, linear_fit=flag)
