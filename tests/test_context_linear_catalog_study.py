"""Real finite trainer/paid guards on scripted ports; no native model work."""

from copy import deepcopy
from dataclasses import asdict
from time import monotonic
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts import context_catalog_learning as study
from scripts import context_head_live as live
from scripts import context_head_rehearsal as source
from scripts import recurring_ticket_workflow_example as entry
from scripts.run_ticket_workflow import CONTEXT_FEATURES
from src.app.coupled_attribution import AttributionBatch
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.actor_ports import VersionedState
from src.core.categorical_linear_fit import CategoricalLinearFit
from src.core.categorical_linear_state import LinearFitPolicy, LinearFitSnapshot
from src.core.context_linear_state import ContextLinearFitPolicy, ContextLinearFitSnapshot
from src.core.catalog_contextual_controller import CatalogTrialWork
from src.core.controller_objective import ControllerMeasures
from src.core.decision_features import DecisionFeatureSpec
from src.core.ticket_workflow import CATALOG
from src.shared.process_memory import ProcessRssSampler


def literal_state():
    model = CategoricalLinearFit(7, 6, 6, ((0.0,) * 6,) * 8, 0, (0.0,) * 6, 0.0)
    policy = LinearFitPolicy(7, 6)
    return ContextLinearFitSnapshot(
        ContextLinearFitPolicy(9, 6, (4, 5)),
        (LinearFitSnapshot(policy, model, 8, 48), LinearFitSnapshot(policy, model, 4, 24)),
        12,
        72,
    )


def scripted(monkeypatch, fault=None):
    results: list[dict[str, Any]] = []
    invocations = []

    class Actor:
        def __init__(self, learner, *, version):
            self.state = learner.snapshot_state()
            self.version = version

        def snapshot_state(self):
            return VersionedState(self.version, deepcopy(self.state))

        def predict(self, features):
            p = np.zeros((len(features), 6))
            p[:, int(features[0, 0])] = 1.0
            return SimpleNamespace(actor_version=self.version, prediction=p)

    def packets(spec):
        batches = tuple(
            AttributionBatch(
                "train",
                name,
                tuple(f"row-{i}-{j}" for j in range(n)),
                np.full((n, 9), float(1 - i)),
                np.zeros((n, 1)),
                num_classes=6,
            )
            for i, (name, n) in enumerate((("new-v1", 3), ("old-v1", 6)))
        )
        return batches, ()

    def run(path, **kwargs):
        invocations.append(kwargs)
        if "_owned_source" in kwargs:
            kwargs["_owned_source"][1]()
        if len(invocations) == 2 and fault == "lost":
            raise RuntimeError("lost delegated result")
        if len(invocations) == 2 and fault == "corrupt":
            results[0]["training_work"]["attempted_updates"] = 99
            kwargs["_owned_source"][1]()
        trial = kwargs.get("catalog_trial")
        record = trial if trial else kwargs["catalog_policy"]
        action = "wait" if trial is None else trial.action_id
        rows = 3 if trial and trial.context == "mistakes" else 6
        count = 0 if action == "wait" else 2 if action == "replay6-wake1" else 1
        optimized = 0 if count == 0 else rows + (6 if count == 2 else 0)
        result = dict(
            complete=True,
            role="controller_train" if trial else "frozen_application_development",
            catalog_control=dict(record=asdict(record)),
            training_work=dict(
                attempted_updates=count, attempted_rows=optimized, returned_updates=count
            ),
            stub_work=dict(count=count, rows=optimized, replay=6 if count == 2 else 0),
        )
        results.append(result)
        return result

    def measures(result):
        w = result["stub_work"]
        return ControllerMeasures(1, 1, 2, 4, 4, 4, 0.01, 1, 0, True), CatalogTrialWork(
            w["count"], w["count"], w["rows"], w["rows"], w["replay"], 1, 1
        )

    monkeypatch.setattr(study, "StableActor", Actor)
    monkeypatch.setattr(study, "prepare_data", packets)
    monkeypatch.setattr(study.live, "run_case", run)
    monkeypatch.setattr(study, "catalog_measures", measures)
    return invocations, results


@pytest.mark.parametrize("fault", [None, "lost", "corrupt"])
def test_should_run_existing_trainer_and_stop_on_lost_or_corrupt_paid_trial(fault, monkeypatch):
    calls, results = scripted(monkeypatch, fault)
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        report = study.run_catalog_study(
            None, _owned_source=(literal_state(), lambda: None), _memory=memory, _linear_fit=True
        )
    if fault is None:
        assert report["complete"] and report["source_unchanged"] and report["actor_unchanged"]
        assert report["delegated_calls"] == dict(attempts=7, returns=7)
        assert (
            report["controller"]["completed"] == 6 and report["controller"]["discarded_trials"] == 6
        )
        assert report["fitted_policy"]["learner_version"] == "context-ls-12"
        assert report["training_work"] == dict(
            attempted_updates=6, attempted_rows=30, returned_updates=6
        )
        assert len(report["trials"]) == 6 and report["application"] is results[-1]
        assert [(x["catalog_trial"].context, x["catalog_trial"].action_id) for x in calls[:-1]] == [
            (context, action)
            for context in ("mistakes", "clean")
            for action in ("wait", "wake1", "replay6-wake1")
        ]
        assert all(x["_linear_fit"] and x["resume_catalog_wake"] for x in calls)
    else:
        assert not report["complete"] and len(calls) == 2 and len(report["trials"]) == 1
        assert report["application"] is None and report["delegated_work_unknown"]
        assert all(value is None for value in report["training_work"].values())
        if fault == "corrupt":
            assert report["trials"][0]["result"]["training_work"]["attempted_updates"] == 0


def test_should_preserve_saved_perceptron_decoder_and_default_child_dispatch(monkeypatch):
    from hashlib import sha256
    from src.core.categorical_perceptron import PerceptronPolicy, PerceptronSnapshot
    from src.core.context_perceptron import ContextPerceptronPolicy, ContextPerceptronSnapshot

    state = ContextPerceptronSnapshot(
        ContextPerceptronPolicy(9, 6, (4, 5)),
        (
            PerceptronSnapshot(PerceptronPolicy(7, 6), ((0.0,) * 6,) * 8, 8, 48, 12),
            PerceptronSnapshot(PerceptronPolicy(7, 6), ((0.0,) * 6,) * 8, 4, 24, 4),
        ),
        12,
        72,
        16,
    )
    calls, _ = scripted(monkeypatch)
    raw = b"{}"
    monkeypatch.setattr(study.live, "_source_bytes", lambda path: raw)
    monkeypatch.setattr(study.live, "HEAD_SOURCE_SHA", sha256(raw).hexdigest())
    monkeypatch.setattr(study.live, "head_states", lambda value: (deepcopy(state),))
    report = study.run_catalog_study("fixture-perceptron")
    assert report["complete"] and report["fitted_policy"]["learner_version"] == "qualified-12"
    assert report["prior_training"]["mistakes"] == 16
    assert all("_linear_fit" not in call and "_owned_source" not in call for call in calls)


@pytest.mark.parametrize(
    "options",
    [
        dict(_linear_fit=True),
        dict(_owned_source=(None, lambda: None)),
        dict(_linear_fit=1),
        dict(_memory=object()),
        dict(_linear_fit=True, _owned_source=(None, lambda: None), _memory=object()),
    ],
)
def test_should_refuse_invalid_owned_study_before_source(options, monkeypatch):
    monkeypatch.setattr(study.live, "_source_bytes", lambda p: pytest.fail("source read"))
    with pytest.raises(ValueError):
        study.run_catalog_study("unrelated-path", **options)


@pytest.mark.parametrize(
    "options",
    [
        dict(catalog_study=1),
        dict(method="perceptron"),
        dict(context_return=False),
        dict(separate_context_heads=False),
        dict(catalog_action="wait"),
        dict(paired_serving="joint"),
        dict(private_adaptation=True),
        dict(adaptive_serving=True),
        dict(replay="recent"),
    ],
)
def test_should_refuse_invalid_study_workflow_before_data(options, monkeypatch):
    monkeypatch.setattr(entry, "prepare_training", lambda **kw: pytest.fail("data opened"))
    args = dict(
        method="linear_fit", context_return=True, separate_context_heads=True, catalog_study=True
    )
    args.update(options)
    with pytest.raises(ValueError):
        entry.run_example(**args)


def test_should_forward_study_cli_and_preserve_existing_output(tmp_path, monkeypatch):
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return dict(complete=True)

    monkeypatch.setattr(entry, "run_example", run)
    output = tmp_path / "study.json"
    args = [
        "--output",
        str(output),
        "--method",
        "linear_fit",
        "--context-return",
        "--separate-context-heads",
        "--catalog-study",
    ]
    assert entry.main(args) == 0
    assert calls == [
        dict(
            context_return=True,
            separate_context_heads=True,
            catalog_study=True,
            method="linear_fit",
        )
    ]
    with pytest.raises(FileExistsError):
        entry.main(args)
    assert len(calls) == 1


def test_should_forward_current_qualified_study_source_and_original_sampler(monkeypatch):
    calls = []
    state = object()
    guard = lambda: None

    def prepare(value, *args, **kwargs):
        assert kwargs["catalog_study"] and kwargs["linear_fit"]
        return value, guard

    def run(path, **kwargs):
        calls.append((path, kwargs))
        return dict(complete=True)

    monkeypatch.setattr(source, "prepare_owned_source", prepare)
    monkeypatch.setattr(study, "run_catalog_study", run)
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
            control="catalog_study",
            linear_fit=True,
            check_boundary=guard,
            memory=memory,
        )
    assert result["owned_source_verified"] and calls[0][0] is None
    assert calls[0][1]["_owned_source"] == (state, guard) and calls[0][1]["_memory"] is memory


@pytest.mark.parametrize("mode,wall", [(False, 30), (True, 240)])
def test_should_witness_declared_born_source_limits_without_mutating_budget(
    mode, wall, monkeypatch
):
    from src.app import probe_endpoint_observer

    captured = []

    class Stop(Exception):
        pass

    def witness(session, binding, limits, expected):
        captured.append((session, limits, expected))
        raise Stop

    monkeypatch.setattr(probe_endpoint_observer, "_BudgetWitness", witness)
    budget = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=12, max_wall_seconds=wall, max_process_rss_bytes=512 * 1024**2
        ),
        monotonic,
    )
    born = (id(budget.budget), budget.budget, budget.started_at, id(budget.clock))
    with pytest.raises(Stop):
        source.prepare_owned_source(
            literal_state(),
            DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG),
            {},
            budget,
            check_boundary=lambda: None,
            training_key="serving_source_training",
            linear_fit=True,
            catalog_study=mode,
        )
    assert captured[0][1].max_wall_seconds == wall and captured[0][2] == 12
    assert born == (id(budget.budget), budget.budget, budget.started_at, id(budget.clock))
