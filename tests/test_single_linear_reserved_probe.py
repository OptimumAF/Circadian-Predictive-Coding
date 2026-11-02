"""Ordinary shared-head probe transport; literal source, no adaptation study."""

from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

from scripts import recurring_ticket_workflow_example as entry
from scripts import ticket_native_probe as probe
from scripts.ticket_linear_probe import MinimumChangeSingleProbeLearner
from src.adapters.linear_fit_learner import MINIMUM_CHANGE_RESIDUAL_ID
from src.adapters.linear_fit_learner import LinearFitLearner, LINEAR_RESIDUAL_ID
from src.core.categorical_linear_fit import CategoricalLinearFit
from src.core.categorical_linear_state import LinearFitPolicy, LinearFitSnapshot
from src.core.learner_ports import TrainingDiagnostic
from test_ordinary_linear_reserved_probe import (
    invoke,
    scripted_reads,
    source_fixture as separate_fixture,
)

OPTIONS = dict(
    method="linear_fit",
    context_return=True,
    reserved_probe=True,
    probe_update="minimum_change",
    separate_context_heads=False,
)
SCRIPTED = dict(calls=0, rows=0)


def source_fixture():
    _, spec, report, budget, memory, clock = separate_fixture()
    model = CategoricalLinearFit(9, 6, 6, ((0.0,) * 6,) * 10, 0, (0.0,) * 6, 6.0)
    state = LinearFitSnapshot(LinearFitPolicy(9, 6), model, 12, 72)
    report.update(
        definition="linear_context_return_ticket_workflow_v1",
        separate_context_heads=False,
        checkpoints=[dict(state=asdict(state))],
    )
    return state, spec, report, budget, memory, clock


@pytest.mark.parametrize("train_update", [None, "minimum_change"])
def test_shared_api_reaches_data_with_explicit_probe_intent(train_update, monkeypatch):
    def reached(**kwargs):
        raise RuntimeError("shared data admission")

    monkeypatch.setattr(entry, "prepare_training", reached)
    result = entry.run_example(**OPTIONS, train_update=train_update)
    assert not result["complete"] and "shared data admission" in result["error"]


@pytest.mark.parametrize(
    "change",
    [
        dict(probe_update=None),
        dict(probe_update=True),
        dict(probe_update="cold"),
        dict(replay="coverage"),
        dict(replay="recent"),
        dict(private_adaptation=True),
        dict(paired_serving="retained_joint_probe"),
        dict(probe_initialization="ce"),
        dict(context_return=False),
        dict(record_train_fit=True),
    ],
)
def test_shared_api_refuses_other_modes_before_data(change, monkeypatch):
    monkeypatch.setattr(entry, "prepare_training", lambda **kw: pytest.fail("data"))
    with pytest.raises(ValueError):
        entry.run_example(**(OPTIONS | change))


@pytest.mark.parametrize("train_update", [False, True])
def test_shared_cli_preserves_head_and_train_choices(train_update, tmp_path, monkeypatch):
    calls = []

    def record(**kwargs):
        calls.append(kwargs)
        return dict(complete=True)

    monkeypatch.setattr(entry, "run_example", record)
    flags = [
        "--method",
        "linear_fit",
        "--context-return",
        "--reserved-probe",
        "--probe-update",
        "minimum_change",
        "--output",
        str(tmp_path / "out.json"),
    ]
    if train_update:
        flags += ["--train-update", "minimum_change"]
    assert entry.main(flags) == 0
    assert calls == [OPTIONS | (dict(train_update="minimum_change") if train_update else {})]
    with pytest.raises(FileExistsError):
        entry.main(flags)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "extra",
    [
        ["--replay", "coverage"],
        ["--private-adaptation"],
        ["--paired-serving", "retained_joint_probe"],
    ],
)
def test_shared_cli_refuses_compositions_before_dispatch(extra, tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "run_example", lambda **kw: pytest.fail("dispatch"))
    with pytest.raises(SystemExit):
        entry.main(
            [
                "--method",
                "linear_fit",
                "--context-return",
                "--reserved-probe",
                "--probe-update",
                "minimum_change",
                "--output",
                str(tmp_path / "out.json"),
                *extra,
            ]
        )


@pytest.mark.parametrize(
    "change",
    [
        dict(definition="separate_context_linear_ticket_workflow_v1"),
        dict(separate_context_heads=True),
        dict(reserved_probe_update="cold"),
        dict(replay_mode="coverage"),
        dict(replay_examples=6),
        dict(final_examples=1),
    ],
)
def test_shared_source_admission_refuses_foreign_authority(change, monkeypatch):
    source = source_fixture()
    source[2].update(change)
    monkeypatch.setattr(probe, "build_head_cohorts", lambda *a, **kw: pytest.fail("data"))
    with pytest.raises(ValueError, match="checkpoint12"):
        invoke(source)


@pytest.mark.parametrize("fault", ["checkpoint", "intent", "head", "replay", "clock"])
def test_shared_source_changes_refuse_more_work(fault, monkeypatch):
    source = source_fixture()

    def data(*args, **kwargs):
        if fault == "checkpoint":
            source[2]["checkpoints"][-1] = deepcopy(source[2]["checkpoints"][-1])
        elif fault == "intent":
            source[2]["reserved_probe_update"] = "cold"
        elif fault == "head":
            source[2]["separate_context_heads"] = True
        elif fault == "replay":
            source[2]["replay_mode"] = "coverage"
        else:
            source[-1][0] = 31.0
        return ((),) * 4

    monkeypatch.setattr(probe, "build_head_cohorts", data)
    result = invoke(source)
    assert not result["complete"] and not result["sessions"]
    assert result["preserved_training_work"] == dict(
        attempted_updates=0, attempted_rows=0, returned_updates=0
    )


def scripted_fit(owner, x, y):
    SCRIPTED["calls"] += 1
    SCRIPTED["rows"] += len(x)
    return TrainingDiagnostic(MINIMUM_CHANGE_RESIDUAL_ID, 0.0)


def test_shared_scripted_fresh_then_return_uses_private_owner_and_original_budgets(monkeypatch):
    source = source_fixture()
    pin = deepcopy(source[0])
    monkeypatch.setattr(probe, "build_head_cohorts", lambda *a, **kw: ((),) * 4)
    monkeypatch.setattr(probe, "_read_tasks", scripted_reads)
    monkeypatch.setattr(MinimumChangeSingleProbeLearner, "train_batch", scripted_fit)

    def fake(seed, state, training, evaluation, **kw):
        child = seed.fork()
        assert child is not seed and child.snapshot_state() == state
        for budget, rows, phase in (
            (kw["budget"], 3, "after_probe"),
            (kw["return_budget"], 6, "after_return"),
        ):
            for _ in range(16):
                child.train_batch(np.zeros((rows, 9)), np.zeros((rows, 1)))
                budget.updates_completed += 1  # Scripted receipt slots, not native fits.
            kw["endpoint_observer"](child, phase, lambda *a, **kw: None)
        return dict(status="complete", return_phase=dict(status="complete"))

    monkeypatch.setattr(probe, "run_fresh_task_probe", fake)
    result = invoke(source)
    assert result["complete"], result.get("error")
    assert result["preserved_training_work"] == dict(
        attempted_updates=32, attempted_rows=144, returned_updates=32
    )
    assert [s["updates"] for s in result["sessions"]] == [16, 16]
    assert source[0] == pin and source[3].updates_completed == 12
    assert result["source_train_update"] is None


@pytest.mark.parametrize("fault", ["replace", "corrupt", "work"])
def test_shared_prior_success_evidence_must_survive_before_next_update(fault, monkeypatch):
    source = source_fixture()
    row: dict[str, Any] = {}
    monkeypatch.setattr(probe, "build_head_cohorts", lambda *a, **kw: ((),) * 4)
    monkeypatch.setattr(probe, "_read_tasks", scripted_reads)
    monkeypatch.setattr(
        MinimumChangeSingleProbeLearner, "train_batch", lambda *a: pytest.fail("fit")
    )

    def corrupt(seed, *args, **kwargs):
        if fault == "replace":
            row["episodes"]["before_probe"] = deepcopy(row["episodes"]["before_probe"])
        elif fault == "corrupt":
            row["episodes"]["before_probe"][0]["episode"]["success"] = True
        else:
            row["task_readout_work"]["before_probe"]["decision_attempts"] = 999
        seed.fork().train_batch(np.zeros((3, 9)), np.zeros((3, 1)))

    monkeypatch.setattr(probe, "run_fresh_task_probe", corrupt)
    result = invoke(source, row=row)
    assert not result["complete"] and "preceding linear reserved readout" in result["error"]
    assert result["preserved_training_work"]["attempted_updates"] == 0
    assert (
        result["preserved_readouts"]["before_probe"]["episodes"][0]["episode"]["success"] is False
    )


@pytest.mark.parametrize("throw", [False, True])
def test_shared_tiny_fit_or_failure_preserves_source_and_permanent_attempt(throw, monkeypatch):
    source = source_fixture()
    monkeypatch.setattr(probe, "build_head_cohorts", lambda *a, **kw: ((),) * 4)
    monkeypatch.setattr(probe, "_read_tasks", scripted_reads)
    if throw:

        def fail(*args):
            raise RuntimeError("lost private fit")

        monkeypatch.setattr(MinimumChangeSingleProbeLearner, "train_batch", fail)

    def partial(seed, state, *args, **kwargs):
        child = seed.fork()
        child.train_batch(np.zeros((1, 9)), np.zeros((1, 1)))
        assert seed.snapshot_state() == state and child.completed_updates == 13
        return dict(status="incomplete")

    monkeypatch.setattr(probe, "run_fresh_task_probe", partial)
    result = invoke(source)
    assert not result["complete"]
    assert result["preserved_training_work"] == dict(
        attempted_updates=1, attempted_rows=1, returned_updates=0 if throw else 1
    )
    assert source[0].completed_updates == source[3].updates_completed == 12


def test_shared_paid_report_cannot_be_reused():
    with pytest.raises(ValueError, match="paid work cannot be renewed"):
        invoke(source_fixture(), row=dict(complete=False))


def test_shared_endpoint_uses_actual_frozen_linear_readout_without_training():
    state, spec, *_ = source_fixture()
    owner = MinimumChangeSingleProbeLearner(state.policy)
    owner.restore_state(state)
    row: dict[str, Any] = dict(episodes={}, task_readout_work={})
    probe._read_tasks(owner, "before_probe", spec, lambda: None, row, linear=True)
    assert len(row["episodes"]["before_probe"]) == 6
    assert all(item["complete"] for item in row["episodes"]["before_probe"])
    work = row["task_readout_work"]["before_probe"]
    assert 0 < work["decision_attempts"] == work["decision_returns"] == work["affine_returns"]
    assert owner.snapshot_state() == state


@pytest.mark.parametrize("train_update", [None, "minimum_change"])
def test_shared_runtime_reaches_private_fresh_return_without_spending_source_authority(
    train_update, monkeypatch
):
    def source_fit(owner, features, targets):
        SCRIPTED["calls"] += 1
        SCRIPTED["rows"] += len(features)
        before = owner.snapshot_state()
        model = CategoricalLinearFit(
            9, 6, len(features), ((0.0,) * 6,) * 10, 0, (0.0,) * len(features), 0.0
        )
        owner._state = replace(
            before,
            model=model,
            completed_updates=before.completed_updates + 1,
            training_examples=before.training_examples + len(features),
        )
        return TrainingDiagnostic(
            MINIMUM_CHANGE_RESIDUAL_ID if train_update else LINEAR_RESIDUAL_ID, 0.0
        )

    monkeypatch.setattr(LinearFitLearner, "train_batch", source_fit)
    monkeypatch.setattr(LinearFitLearner, "train_batch_minimum_change", source_fit)
    monkeypatch.setattr(MinimumChangeSingleProbeLearner, "train_batch", scripted_fit)
    monkeypatch.setattr(
        entry,
        "run_workflow_episode",
        lambda *a, **kw: dict(
            complete=True,
            success=False,
            scenario=kw.get("scenario", "ordinary"),
            training_updates=0,
            work=dict(prediction_calls_validated=0),
        ),
    )
    monkeypatch.setattr(probe, "_read_tasks", scripted_reads)

    def fresh(seed, state, training, evaluation, **kwargs):
        for session, rows, phase in (
            (kwargs["budget"], 3, "after_probe"),
            (kwargs["return_budget"], 6, "after_return"),
        ):
            for _ in range(16):
                seed.fork().train_batch(np.zeros((rows, 9)), np.zeros((rows, 1)))
                session.updates_completed += 1
            kwargs["endpoint_observer"](seed, phase, lambda *a, **kw: None)
        return dict(status="complete", return_phase=dict(status="complete"))

    monkeypatch.setattr(probe, "run_fresh_task_probe", fresh)
    result = entry.run_example(**OPTIONS, train_update=train_update)
    assert result["complete"], result.get("error")
    assert result["budget"]["updates_completed"] == 12 and len(result["applied_updates"]) == 12
    assert result["training_work"]["attempted_updates"] == 12
    child = result["reserved_probe"]
    assert child["complete"] and child["source_train_update"] == train_update
    assert child["preserved_training_work"] == dict(
        attempted_updates=32, attempted_rows=144, returned_updates=32
    )
    assert all(
        result[k] is True
        for k in (
            "source_unchanged",
            "actor_unchanged",
            "numpy_rng_unchanged",
            "python_rng_unchanged",
        )
    )
