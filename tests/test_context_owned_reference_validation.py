"""Actual deferred collector on scripted ports; no native model/task work."""

from dataclasses import asdict, replace
from time import monotonic
from typing import Any

import pytest

from scripts import context_stream_comparison as study
from scripts import context_controller_learning as endpoints
from scripts import context_head_live as live
from scripts import context_head_rehearsal as source
from scripts import recurring_ticket_workflow_example as entry
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.decision_features import DecisionFeatureSpec
from src.core.reference_controller import ReferencePolicy
from scripts.run_ticket_workflow import CONTEXT_FEATURES
from src.core.ticket_workflow import CATALOG
from src.shared.process_memory import ProcessRssSampler
from test_context_linear_catalog_study import literal_state


def options():
    return dict(
        method="linear_fit",
        context_return=True,
        separate_context_heads=True,
        reference_validation=True,
    )


@pytest.mark.parametrize(
    "change",
    [
        dict(reference_validation=1),
        dict(method="perceptron"),
        dict(context_return=False),
        dict(separate_context_heads=False),
        dict(private_adaptation=True),
        dict(paired_serving="joint"),
        dict(adaptive_serving=True),
        dict(catalog_study=True),
        dict(catalog_action="wake1", catalog_context="clean"),
        dict(incident_stream="shift"),
        dict(replay="recent"),
        dict(record_train_fit=True),
    ],
)
def test_should_refuse_invalid_workflow_before_data(change, monkeypatch):
    monkeypatch.setattr(entry, "prepare_training", lambda **kw: pytest.fail("data opened"))
    args = options()
    args.update(change)
    with pytest.raises(ValueError):
        entry.run_example(**args)


def flags(path):
    return [
        "--output",
        str(path),
        "--method",
        "linear_fit",
        "--context-return",
        "--separate-context-heads",
        "--reference-validation",
    ]


def test_should_dispatch_normal_cli_and_preserve_new_output(tmp_path, monkeypatch):
    calls = []

    def example(**kw):
        calls.append(kw)
        return dict(complete=True, training_work=dict(attempted_updates=0, attempted_rows=0))

    monkeypatch.setattr(entry, "run_example", example)
    path = tmp_path / "out.json"
    assert entry.main(flags(path)) == 0 and calls == [options()]
    with pytest.raises(FileExistsError):
        entry.main(flags(path))
    assert len(calls) == 1


def test_should_preview_actual_protocol_without_source_or_model(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "run_example", lambda **kw: pytest.fail("source opened"))
    monkeypatch.setattr(study, "_source_bytes", lambda *a: pytest.fail("history opened"))
    import json

    path = tmp_path / "protocol.json"
    assert entry.main([*flags(path), "--reference-validation-protocol"]) == 0
    protocol = json.loads(path.read_text())
    assert protocol["case_count"] == 6 and not protocol["runtime_authority"]
    assert protocol["modes"] == list(study.OWNED_LINEAR_MODES)
    assert len(protocol["opportunities"]) == 24
    assert study.protocol_snapshot(withheld=True)["case_count"] == 14


@pytest.mark.parametrize(
    "change",
    [
        dict(_linear_fit=False),
        dict(_owned_source=None),
        dict(_memory=None),
        dict(input_path="saved"),
        dict(fitted_path="saved"),
        dict(periodic_path="saved"),
        dict(withheld=False),
    ],
)
def test_should_refuse_invalid_owned_comparison_before_io(change, monkeypatch):
    monkeypatch.setattr(study, "_source_bytes", lambda *a: pytest.fail("history opened"))
    args = dict(
        input_path=None,
        periodic_path=None,
        fitted_path=None,
        withheld=True,
        _owned_source=(literal_state(), lambda: None),
        _memory=object(),
        _linear_fit=True,
    )
    args.update(change)
    with pytest.raises(ValueError):
        study.run_comparison(**args)


def scripted(monkeypatch, fault=None):
    calls = []
    results: list[dict[str, Any]] = []
    readouts = []

    def run(path, **kw):
        calls.append(kw)
        if len(calls) == 2 and fault == "corrupt":
            results[0]["training_work"]["attempted_rows"] = 999
        kw["_owned_source"][1]()
        if len(calls) == 2 and fault == "lost":
            raise RuntimeError("lost result")
        initial = kw["_owned_source"][0]
        mode = kw["reference_mode"]
        n = 0 if mode == "no_learning" else 6 if mode == "periodic" else 2
        rows = 0 if n == 0 else 27 if n == 6 else 9
        state = replace(
            initial,
            heads=(
                replace(initial.heads[0], completed_updates=8 + n, training_examples=48 + rows),
                initial.heads[1],
            ),
            completed_updates=12 + n,
            training_examples=72 + rows,
        )
        kw["_capture_candidate"](state)
        sizes = [] if n == 0 else ([3, 6] * 3 if n == 6 else [3, 6])
        result: dict[str, Any] = dict(
            complete=True,
            source_unchanged=True,
            actor_unchanged=True,
            role="reference_validation_train",
            context_recipe=asdict(kw["stream_recipe"]),
            reference_policy=asdict(ReferencePolicy(mode)),
            candidate_state=asdict(state),
            native_calls=[dict(rows=r, complete=True) for r in sizes],
            receipt_progress=dict(complete=True, consumed=6),
            training_work=dict(attempted_updates=n, returned_updates=n, attempted_rows=rows),
        )
        results.append(result)
        if len(calls) == 1 and fault == "boolean_work":
            result["training_work"]["attempted_rows"] = False
        if len(calls) == 2 and fault == "incomplete":
            result["complete"] = False
        return result

    def readout(state, phase, require, report, **kw):
        assert len(calls) == 6 and all(r["complete"] for r in results)
        assert kw == dict(linear_fit=True)
        readouts.append(report)
        if len(readouts) == 2 and fault == "corrupt_readout":
            readouts[0]["task_readout_work"]["tampered"] = 1
        require()
        report["task_readout_work"][phase] = dict(attempts=0)
        return [1, 4]

    monkeypatch.setattr(study.live, "run_case", run)
    monkeypatch.setattr(endpoints, "_task_readout", readout)
    monkeypatch.setattr(study, "_source_bytes", lambda *a: pytest.fail("history opened"))
    return calls, results, readouts


@pytest.mark.parametrize(
    "fault", [None, "lost", "corrupt", "incomplete", "corrupt_readout", "boolean_work"]
)
def test_should_collect_before_releasing_endpoints_and_close_on_paid_failure(fault, monkeypatch):
    calls, results, readouts = scripted(monkeypatch, fault)
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        report = study.run_comparison(
            None,
            None,
            None,
            withheld=True,
            _owned_source=(literal_state(), lambda: None),
            _memory=memory,
            _linear_fit=True,
        )
    if fault is None:
        assert report["complete"] and len(readouts) == 7
        assert report["endpoint_release_after_cases"] == 6
        assert report["training_work"] == dict(
            attempted_updates=16, attempted_rows=72, returned_updates=16
        )
        assert [(r["recipe_id"], r["mode"]) for r in report["cases"]] == [
            (r.recipe_id, m)
            for r in study.WITHHELD_CONTEXT_RECIPES
            for m in study.OWNED_LINEAR_MODES
        ]
    elif fault == "boolean_work":
        assert not report["complete"] and len(calls) == 1 and not readouts
        assert report["delegated_work_unknown"]
    elif fault == "corrupt_readout":
        assert not report["complete"] and len(calls) == 6 and len(readouts) == 2
        assert "tampered" not in report["validation"][0]["readout"]["task_readout_work"]
    else:
        assert not report["complete"] and len(calls) == 2 and not readouts
        assert report["endpoint_release_after_cases"] is None
        if fault in ("lost", "corrupt"):
            assert report["delegated_work_unknown"]
            assert all(v is None for v in report["training_work"].values())
        if fault == "corrupt":
            assert report["cases"][0]["result"]["training_work"]["attempted_rows"] == 0


def test_should_qualify_original_born_long_lived_source_without_renewal(monkeypatch):
    calls = []

    class Stop(Exception):
        pass

    def witness(session, binding, limits, count):
        calls.append((session, limits, count))
        raise Stop

    monkeypatch.setattr("src.app.probe_endpoint_observer._BudgetWitness", witness)
    budget = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=12, max_wall_seconds=240, max_process_rss_bytes=512 * 1024**2
        ),
        monotonic,
    )
    birth = budget.started_at
    with pytest.raises(Stop):
        source.prepare_owned_source(
            literal_state(),
            DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG),
            {},
            budget,
            check_boundary=lambda: None,
            training_key="serving_source_training",
            linear_fit=True,
            reference_validation=True,
        )
    assert calls[0][0] is budget and calls[0][1].max_wall_seconds == 240 and calls[0][2] == 12
    assert budget.started_at == birth and budget.updates_completed == 0


def test_should_forward_original_owned_validation_to_existing_comparison(monkeypatch):
    calls = []
    state = object()
    guard = lambda: None
    monkeypatch.setattr(source, "prepare_owned_source", lambda *a, **kw: (state, guard))

    def run(*paths, **kw):
        calls.append((paths, kw))
        return dict(complete=True)

    monkeypatch.setattr(study, "run_comparison", run)
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        budget = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=12, max_wall_seconds=240, max_process_rss_bytes=512 * 1024**2
            ),
            monotonic,
        )
        budget.attach_memory(memory)
        result = live.run_owned_case(
            state,
            None,
            {},
            budget,
            control="reference_validation",
            check_boundary=guard,
            memory=memory,
            linear_fit=True,
        )
    assert result["complete"] and result["owned_source_verified"]
    assert calls[0][0] == (None, None, None) and calls[0][1]["_owned_source"] == (state, guard)
