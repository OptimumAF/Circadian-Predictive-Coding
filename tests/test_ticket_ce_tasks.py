"""Ordinary CE command and guarded tiny composition; not a native12 study."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from scripts import recurring_ticket_workflow_example as entry
from scripts import ticket_ce_tasks as tasks
from scripts.run_ticket_workflow import CONTEXT_FEATURES, CONTEXT_TASKS
from src.adapters.torch_categorical_ce_readout import (
    CategoricalCEFit,
    CategoricalCEFitFailed,
    CEReadoutWork,
)
from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.decision_features import DecisionFeatureSpec
from src.core.ticket_workflow import CATALOG
from src.shared.process_memory import ProcessRssSampler


def forbidden(*args, **kwargs):
    raise AssertionError("unexpected work")


@pytest.mark.parametrize(
    "change",
    [
        dict(train_readout=True),
        dict(train_readout=False),
        dict(train_readout=0),
        dict(train_readout="other"),
        dict(train_readout=[]),
        dict(train_readout={}),
        dict(method="pc"),
        dict(method="neutral_cpc"),
        dict(method="gated_cpc"),
        dict(method="linear_fit"),
        dict(context_return=False),
        dict(replay="recent"),
        dict(reserved_probe=True),
        dict(probe_initialization="ce"),
        dict(train_update="minimum_change"),
        dict(paired_serving="joint"),
        dict(private_adaptation=True),
        dict(record_train_fit=True),
        dict(class_balanced_bp=True),
        dict(separate_context_heads=True),
        dict(policy_change=True),
        dict(catalog_study=True),
        dict(adaptive_serving=True),
        dict(incident_stream="shift"),
        dict(calibrated_serving=True),
        dict(reference_validation=True),
        dict(probe_update="minimum_change"),
    ],
)
def test_options_refuse_before_data_or_models(change, monkeypatch):
    monkeypatch.setattr(entry, "prepare_training", forbidden)
    options = dict(method="bp", context_return=True, train_readout="ce")
    options.update(change)
    with pytest.raises(ValueError, match="TRAIN CE readout"):
        entry.run_example(**options)


@pytest.mark.parametrize(
    "flags",
    [
        [],
        ["--context-return"],
        ["--context-return", "--method", "pc"],
        ["--context-return", "--method", "bp", "--reserved-probe"],
    ],
)
def test_cli_bad_mode_refused_before_output_or_dispatch(flags, tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "run_example", forbidden)
    output = tmp_path / "new.json"
    with pytest.raises(SystemExit):
        entry.main(["--output", str(output), "--train-readout", "ce", *flags])
    assert not output.exists()


def test_existing_output_refused_before_dispatch(tmp_path, monkeypatch):
    output = tmp_path / "paid.json"
    output.write_text("original")
    monkeypatch.setattr(entry, "run_example", forbidden)
    with pytest.raises(FileExistsError):
        entry.main(
            ["--output", str(output), "--context-return", "--method", "bp", "--train-readout", "ce"]
        )
    assert output.read_text() == "original"


@pytest.mark.parametrize("mode", [None, "ce"])
def test_cli_preserves_default_and_records_paid_ce_failure(mode, tmp_path, monkeypatch):
    from scripts import context_catalog_learning as writer

    calls, writes = [], []
    paid = dict(complete=False, fit_work=dict(backward_attempts=1, backward_rows=12))
    report = dict(
        complete=False,
        training_work=dict(returned_updates=12),
        train_readout={},
        preserved_train_readout=paid,
    )

    def run(**kwargs):
        calls.append(kwargs)
        return report

    def write(path, actual, **kwargs):
        assert actual is report
        writes.append(kwargs)
        return False

    monkeypatch.setattr(entry, "run_example", run)
    monkeypatch.setattr(writer, "write_catalog_report", write)
    flags = ["--output", str(tmp_path / "new.json"), "--context-return", "--method", "bp"]
    if mode:
        flags += ["--train-readout", mode]
    assert entry.main(flags) == 1
    assert calls == [
        dict(context_return=True, method="bp", **(dict(train_readout=mode) if mode else {}))
    ]
    if mode:
        assert writes[0]["observed_work"]["train_readout"] is paid
        assert "train_readout_budget" in writes[0]["observed_work"]
        assert writes[0]["max_bytes"] == 4 * 1024**2
    else:
        assert not writes


class Stream:
    """Explicit synthetic clocks/receipts; no source training is executed."""

    def __init__(self, source, budget):
        self._runtime = SimpleNamespace(
            _candidate=source,
            _inbox=SimpleNamespace(_applied={i: dict(updates=1, rows=6) for i in range(12)}),
        )
        self._budget = budget
        self.completed_updates = 12

    @property
    def applied_updates(self):
        return tuple(deepcopy(tuple(self._runtime._inbox._applied.values())))


@pytest.fixture
def context():
    source = TorchFeatureLearner(TorchFeaturePolicy("bp", 9, 3, 17, num_classes=6))
    state = source.snapshot_state()
    assert state.gradients is not None and state.optimizer_state is not None
    # This synthetic valid state tests ownership/counters, not learned native12.
    gradients = {name: torch.zeros_like(state.head_state[name]) for name in state.gradients}
    optimizer = deepcopy(state.optimizer_state)
    optimizer["state"] = {
        i: dict(momentum_buffer=value.clone()) for i, value in enumerate(gradients.values())
    }
    state = replace(
        state,
        completed_updates=12,
        training_examples=72,
        gradients=gradients,
        optimizer_state=optimizer,
    )
    source.restore_state(state)
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        budget = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=12, max_wall_seconds=30, max_process_rss_bytes=512 * 1024**2
            ),
            lambda: 0.0,
        )
        budget.attach_memory(memory)
        for _ in range(12):
            budget.record_update()
        stream = Stream(source, budget)
        row: dict[str, Any] = {}
        report = dict(
            definition="native_context_return_ticket_workflow_v1",
            method="bp",
            source_unchanged=True,
            actor_unchanged=True,
            final_examples=0,
            probe_updates=0,
            replay_examples=0,
            train_readout_requested="ce",
            train_readout_source_training=dict(completed_updates=12, applied_update_count=12),
            train_readout=row,
        )
        yield (
            stream,
            source,
            state,
            DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG),
            report,
            budget,
            memory,
            row,
        )


def run(context, boundary=lambda: None):
    stream, source, state, spec, report, budget, memory, row = context
    row, guard = tasks.run_train_readout(
        stream, state, spec, report, budget, check_boundary=boundary, memory=memory, row=row
    )
    return row


def test_actual_fit_and_existing_task_loop_are_isolated_and_separately_paid(context):
    stream, source, state, spec, report, budget, memory, row = context
    before = deepcopy(state)
    numpy_rng = entry.np.random.get_state()
    torch_rng = torch.get_rng_state().clone()
    calls = []
    result = run(context, lambda: calls.append(1))
    assert result is row and row["complete"] is True
    assert [(r["policy_version"], r["concept"]) for r in row["episodes"]] == list(CONTEXT_TASKS)
    assert all(
        r["role"] == "controller_train" and r["episode"]["complete"] for r in row["episodes"]
    )
    assert row["work"] == dict(
        fit_attempts=1,
        fit_returns=1,
        reader_attempts=1,
        reader_returns=1,
        task_attempts=4,
        task_returns=4,
    )
    assert row["fit_work"]["step_returns"] == 1 and row["fit_work"]["backward_rows"] > 0
    assert 1 <= row["readout_work"]["decision_returns"] <= 24
    assert tasks.native.same_state(source.snapshot_state(), before)
    current_rng = entry.np.random.get_state()
    assert isinstance(numpy_rng, tuple) and isinstance(current_rng, tuple)
    assert (
        numpy_rng[0] == current_rng[0]
        and entry.np.array_equal(numpy_rng[1], current_rng[1])
        and numpy_rng[2:] == current_rng[2:]
    )
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert budget.updates_completed == 12 and budget.started_at == 0
    assert report["train_readout_budget"]["elapsed_seconds"] >= 0 and len(calls) > 10
    assert "preserved_train_readout" not in report
    with pytest.raises(ValueError, match="fresh"):
        run(context)


@pytest.fixture
def scripted(context, monkeypatch):
    stream, source, state, spec, report, budget, memory, row = context
    fit = CategoricalCEFit(
        deepcopy(state),
        spec,
        tuple((0.0,) * 6 for _ in range(4)),
        tuple((0.0,) * 6 for _ in range(12)),
        (1.8,),
        1.8,
        dict(step_returns=1, backward_rows=12),
    )
    monkeypatch.setattr(tasks, "fit_categorical_ce_readout", lambda *a, **kw: fit)
    monkeypatch.setattr(tasks.native, "prepare_train_fit", lambda s: ((), None))

    class Reader:
        def __init__(self, fit):
            self.work = CEReadoutWork()

        def assert_unchanged(self):
            pass

    monkeypatch.setattr(tasks, "CategoricalCEReadoutPredictor", Reader)
    monkeypatch.setattr(
        tasks, "run_workflow_episode", lambda *a, **kw: dict(complete=True, success=False)
    )
    return fit


@pytest.mark.parametrize(
    "fault",
    [
        "intent",
        "report_replaced",
        "receipt_replaced",
        "receipt_changed",
        "budget_birth",
        "budget_owner",
        "source_changed",
        "fit_work_replaced",
        "paid_row_changed",
        "paid_row_replaced",
    ],
)
def test_changed_authority_or_paid_evidence_refuses_next_task_and_preserves_costs(
    context, scripted, fault
):
    stream, source, state, spec, report, budget, memory, row = context
    changed = False

    def boundary():
        nonlocal changed
        if changed or row.get("work", {}).get("task_returns") != 1:
            return
        changed = True
        if fault == "intent":
            report["train_readout_requested"] = "other"
        elif fault == "report_replaced":
            report["train_readout"] = deepcopy(row)
        elif fault == "receipt_replaced":
            stream._runtime._inbox._applied[0] = deepcopy(stream._runtime._inbox._applied[0])
        elif fault == "receipt_changed":
            stream._runtime._inbox._applied[0]["rows"] = 7
        elif fault == "budget_birth":
            budget.started_at = -1
        elif fault == "budget_owner":
            budget.budget = deepcopy(budget.budget)
        elif fault == "source_changed":
            source.restore_state(replace(state, training_examples=73))
        elif fault == "fit_work_replaced":
            object.__setattr__(scripted, "work", deepcopy(scripted.work))
        elif fault == "paid_row_changed":
            row["episodes"][0]["episode"]["success"] = True
        else:
            row["episodes"][0] = deepcopy(row["episodes"][0])

    with pytest.raises(ValueError):
        run(context, boundary)
    saved = report["preserved_train_readout"]
    assert saved["complete"] is False
    assert saved["work"]["task_attempts"] == saved["work"]["task_returns"] == 1
    assert saved["episodes"][0]["episode"]["success"] is False
    assert saved["fit_work"]["backward_rows"] == 12
    assert changed


def test_negative_task_outcomes_do_not_select_or_stop_later_tasks(context, scripted):
    row = run(context)
    assert row["complete"] and len(row["episodes"]) == 4
    assert all(r["episode"]["success"] is False for r in row["episodes"])


@pytest.mark.parametrize(
    "fault",
    ["row_value", "row_owner", "work_owner", "episode_owner", "budget_value", "budget_owner"],
)
def test_retained_guard_refuses_late_finish_clock_mutation_and_keeps_paid_record(
    context, scripted, fault
):
    stream, source, state, spec, report, budget, memory, row = context
    armed = False

    def clock():
        if armed:
            if fault == "row_value":
                row["episodes"][0]["episode"]["success"] = True
            elif fault == "row_owner":
                report["train_readout"] = deepcopy(row)
            elif fault == "work_owner":
                row["work"] = deepcopy(row["work"])
            elif fault == "episode_owner":
                row["episodes"][0] = deepcopy(row["episodes"][0])
            elif fault == "budget_value":
                report["train_readout_budget"]["elapsed_seconds"] = -1
            else:
                report["train_readout_budget"] = deepcopy(report["train_readout_budget"])
        return 0.0

    budget.clock = clock
    _, keep = tasks.run_train_readout(
        stream, state, spec, report, budget, check_boundary=lambda: None, memory=memory, row=row
    )
    armed = True
    budget.before_final()  # Same original clock boundary called by stream.finish().
    with pytest.raises(ValueError):
        keep()
    saved = report["preserved_train_readout"]
    assert not saved["complete"] and len(saved["episodes"]) == 4
    assert saved["work"]["fit_returns"] == 1 and saved["work"]["task_returns"] == 4
    assert saved["episodes"][0]["episode"]["success"] is False
    assert saved["fit_work"]["backward_rows"] == 12
    assert report["preserved_train_readout_budget"]["elapsed_seconds"] >= 0
    assert budget.started_at == 0 and budget.updates_completed == 12


def test_failed_fit_retains_attempt_and_gradient_cost_without_constructing_reader(
    context, monkeypatch
):
    def fail(*args, **kwargs):
        raise CategoricalCEFitFailed("fixed failure", dict(backward_attempts=2, backward_rows=24))

    monkeypatch.setattr(tasks, "fit_categorical_ce_readout", fail)
    monkeypatch.setattr(tasks, "CategoricalCEReadoutPredictor", forbidden)
    with pytest.raises(CategoricalCEFitFailed):
        run(context)
    saved = context[4]["preserved_train_readout"]
    assert saved["work"] == dict(fit_attempts=1)
    assert saved["fit_work"]["backward_rows"] == 24 and not saved["episodes"]


@pytest.mark.parametrize(
    "fault", ["final", "method", "source_incomplete", "counter", "receipt_count", "intent"]
)
def test_source_admission_precedes_train_label_preparation(context, monkeypatch, fault):
    stream, source, state, spec, report, budget, memory, row = context
    monkeypatch.setattr(tasks.native, "prepare_train_fit", forbidden)
    if fault == "final":
        report["final_examples"] = 1
    elif fault == "method":
        report["method"] = "pc"
    elif fault == "source_incomplete":
        report["source_unchanged"] = False
    elif fault == "counter":
        budget.updates_completed = 11
    elif fault == "receipt_count":
        stream._runtime._inbox._applied.pop(0)
    else:
        report["train_readout_requested"] = "other"
    with pytest.raises(ValueError, match="checkpoint12"):
        run(context)
    assert not row
