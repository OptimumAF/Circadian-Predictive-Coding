"""Ordinary option/composition correctness; no genuine source12/F16 study."""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from scripts import recurring_ticket_workflow_example as entry
from scripts import ticket_ce_probe as ce_run
from scripts import ticket_native_probe as probe
from src.adapters import torch_feature_learner as native
from src.adapters.torch_categorical_ce_readout import CategoricalCEFitFailed
from src.adapters.torch_decision import _same_values
from src.app.decision_reference_evaluation import DecisionCase
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.decision_features import DecisionFeatureSpec
from src.core.typed_decision import DecisionCandidate, DecisionRequest


def forbidden(*args, **kwargs):
    raise AssertionError("unexpected work")


@pytest.mark.parametrize("value", [True, False, 0, "other", [], {}])
def test_option_type_and_value_refused_before_models(value, monkeypatch):
    monkeypatch.setattr(entry, "prepare_training", forbidden)
    with pytest.raises(ValueError, match="CE probe initialization"):
        entry.run_example(
            method="bp", context_return=True, reserved_probe=True, probe_initialization=value
        )


@pytest.mark.parametrize(
    "change",
    [
        dict(reserved_probe=False),
        dict(context_return=False),
        dict(method="pc"),
        dict(method="neutral_cpc"),
        dict(method="gated_cpc"),
        dict(method="perceptron"),
        dict(replay="recent"),
        dict(record_train_fit=True),
    ],
)
def test_incompatible_mode_refused_before_models(change, monkeypatch):
    monkeypatch.setattr(entry, "prepare_training", forbidden)
    options = dict(method="bp", context_return=True, reserved_probe=True, probe_initialization="ce")
    options.update(change)
    with pytest.raises(ValueError):
        entry.run_example(**options)


@pytest.mark.parametrize("initialized", [True, False])
def test_cli_forwarding_preserves_default_and_paid_failure_fallback(
    initialized, tmp_path, monkeypatch
):
    calls, writes = [], []
    fit_work = dict(complete=False, fit_work=dict(backward_attempts=1, backward_rows=3))
    report = dict(
        complete=False,
        probe_initialization=fit_work,
        training_work=dict(returned_updates=12),
        reserved_probe={},
    )

    def run(**kwargs):
        calls.append(kwargs)
        return report

    def write(path, actual, **kwargs):
        writes.append(kwargs)
        assert actual is report
        return False

    from scripts import context_catalog_learning as writer

    monkeypatch.setattr(entry, "run_example", run)
    monkeypatch.setattr(writer, "write_catalog_report", write)
    flags = [
        "--context-return",
        "--method",
        "bp",
        "--reserved-probe",
        "--output",
        str(tmp_path / "new.json"),
    ]
    if initialized:
        flags += ["--probe-initialization", "ce"]
    assert entry.main(flags) == 1
    expected = dict(context_return=True, method="bp", reserved_probe=True)
    if initialized:
        expected["probe_initialization"] = "ce"
        assert writes[0]["observed_work"]["probe_initialization"] is fit_work
    else:
        assert "probe_initialization" not in writes[0]["observed_work"]
    assert calls == [expected] and writes[0]["max_bytes"] == 4 * 1024**2


@pytest.mark.parametrize(
    "flags",
    [
        [],
        ["--reserved-probe", "--method", "pc", "--context-return"],
        ["--reserved-probe", "--method", "bp"],
        ["--reserved-probe", "--method", "bp", "--context-return", "--class-balanced-bp"],
    ],
)
def test_bad_cli_refused_before_dispatch(flags, tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "run_example", forbidden)
    with pytest.raises(SystemExit):
        entry.main(["--output", str(tmp_path / "bad.json"), "--probe-initialization", "ce", *flags])


def test_existing_output_refused_before_dispatch(tmp_path, monkeypatch):
    path = tmp_path / "used.json"
    path.write_text("paid")
    monkeypatch.setattr(entry, "run_example", forbidden)
    with pytest.raises(FileExistsError):
        entry.main(
            [
                "--output",
                str(path),
                "--context-return",
                "--method",
                "bp",
                "--reserved-probe",
                "--probe-initialization",
                "ce",
            ]
        )
    assert path.read_text() == "paid"


class Stream:
    """Scripted receipt/public-view seams; original native12 is not claimed."""

    def __init__(self, learner=None, budget=None):
        self._runtime = SimpleNamespace(
            _candidate=learner,
            _inbox=SimpleNamespace(_applied={"a": dict(updates=1, rows=3, nested=[1])}),
        )
        self._budget = budget
        self.on_read = lambda: None

    @property
    def applied_updates(self):
        self.on_read()
        return deepcopy(tuple(self._runtime._inbox._applied.values()))


@pytest.mark.parametrize(
    "fault", [None, "receipt_replace", "ledger_replace", "mutation", "extra", "during_read"]
)
def test_authoritative_receipts_refuse_changes_but_accept_public_copies(fault):
    stream = Stream()
    guard = ce_run._receipt_guard(stream)
    ledger = stream._runtime._inbox._applied
    if fault == "receipt_replace":
        ledger["a"] = deepcopy(ledger["a"])
    elif fault == "ledger_replace":
        stream._runtime._inbox._applied = dict(ledger)
    elif fault == "mutation":
        ledger["a"]["nested"].append(2)
    elif fault == "extra":
        ledger["b"] = dict(updates=2, rows=3)
    elif fault == "during_read":
        stream.on_read = lambda: ledger["a"]["nested"].append(2)
    if fault is None:
        guard()
        assert stream.applied_updates == stream.applied_updates
    else:
        with pytest.raises(ValueError, match="receipt"):
            guard()
        if fault in ("mutation", "during_read"):
            assert ledger["a"]["nested"] == [1, 2]


@pytest.fixture(scope="module")
def owner():
    catalog = tuple(DecisionCandidate(n, n) for n in ("a", "b", "c"))
    spec = DecisionFeatureSpec(("alpha", "beta"), catalog)
    cases = tuple(
        DecisionCase(DecisionRequest(str(i), q, catalog), t, "fixture", "train")
        for i, (q, t) in enumerate((("alpha", "a"), ("beta", "b"), ("alpha beta", "c")))
    )
    from src.adapters.perceptron_decision import build_decision_training_batch

    batch = build_decision_training_batch(cases, spec, "tiny", allow_subsets=True)
    learner = native.TorchFeatureLearner(native.TorchFeaturePolicy("bp", 2, 3, 17, num_classes=3))
    learner.train_batch(batch.features, batch.targets)
    return learner, spec, cases, batch


def inputs(owner, monkeypatch):
    learner, spec, cases, batch = owner
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=1, max_wall_seconds=30), lambda: 0.0
    )
    budget.record_update()
    memory = SimpleNamespace()
    # Scripted RSS/source-admission seams admit the tiny2D fixture, never native12.
    monkeypatch.setattr(ce_run, "_require_probe_source", lambda *a, **kw: None)
    monkeypatch.setattr(
        ToyBudgetSession, "attach_memory", lambda s, m: setattr(s, "process_rss_sampler", m)
    )
    monkeypatch.setattr(ToyBudgetSession, "_check_memory", lambda s: None)
    budget.attach_memory(cast(Any, memory))
    monkeypatch.setattr(ce_run.native, "prepare_train_fit", lambda s: (cases, batch))
    stream = Stream(learner, budget)
    row: dict[str, Any] = {}
    report = dict(
        probe_initialization=row,
        reserved_probe_source_training=dict(completed_updates=1, applied_update_count=1),
    )
    return stream, learner.snapshot_state(), spec, report, budget, memory, row


def test_tiny_actual_fit_derivation_and_private_update_preserve_source(owner, monkeypatch):
    stream, state, spec, report, budget, memory, row = inputs(owner, monkeypatch)
    seen: list[Any] = []

    def private_phase(original, schema, source_report, source_budget, **kwargs):
        assert original is state and source_budget is budget and source_report is report
        actual = kwargs["ce_initialization"]
        seen.append(actual)
        private = native.TorchFeatureLearner(state.policy)
        private.restore_state(actual.state)
        assert np.array_equal(
            private.predict_logits(owner[3].features), np.array(actual.fit.training_logits)
        )
        private.train_batch(owner[3].features, owner[3].targets)
        assert private.completed_updates == 2
        kwargs["row"].update(complete=False, preserved_training_work=dict(returned_updates=1))
        return kwargs["row"]

    monkeypatch.setattr(ce_run, "run_owned_probe", private_phase)
    private_row: dict[str, Any] = {}
    result = ce_run.run_initialized_probe(
        stream,
        state,
        spec,
        report,
        budget,
        check_boundary=lambda: None,
        memory=memory,
        row=private_row,
        initialization_row=row,
    )
    actual = seen[0]
    assert result is private_row and result["complete"] is False
    assert row["complete"] and row["source_unchanged"]
    assert row["preserved_work"] == dict(
        fit_attempts=1, fit_returns=1, initialization_attempts=1, initialization_returns=1
    )
    assert row["fit_work"] == row["preserved_fit_work"] == actual.fit.work
    assert actual.state.optimizer_state is not state.optimizer_state
    assert _same_values(actual.state.optimizer_state, state.optimizer_state)
    assert _same_values(actual.state.gradients, state.gradients)
    assert budget.updates_completed == 1 and budget.started_at == 0.0
    assert _same_values(stream._runtime._candidate.snapshot_state(), state)


def scripted_fit(monkeypatch):
    fit = SimpleNamespace(
        source={"source": 1},
        spec={"spec": 1},
        coefficients=((1.0,),),
        work=dict(step_returns=1),
        training_logits=((1.0,),),
    )
    initialization = SimpleNamespace(fit=deepcopy(fit), state={"private": 1})
    monkeypatch.setattr(ce_run, "fit_categorical_ce_readout", lambda *a, **kw: fit)
    monkeypatch.setattr(ce_run, "initialize_categorical_ce_probe", lambda *a, **kw: initialization)
    monkeypatch.setattr(ce_run, "ce_initialization_record", lambda i: dict(definition="scripted"))
    return fit, initialization


@pytest.mark.parametrize(
    "fault",
    [
        "fit_failure",
        "receipt_replace",
        "fit_replace",
        "initialization_replace",
        "work_replace",
        "budget_replace",
        "source_tag_replace",
    ],
)
def test_auxiliary_failure_preserves_costs_and_refuses_more_work(owner, monkeypatch, fault):
    stream, state, spec, report, budget, memory, row = inputs(owner, monkeypatch)
    fit, initialization = scripted_fit(monkeypatch)
    if fault == "fit_failure":

        def fail(*a, **kw):
            raise CategoricalCEFitFailed("lost closure", dict(backward_attempts=1, backward_rows=3))

        monkeypatch.setattr(ce_run, "fit_categorical_ce_readout", fail)
    elif fault == "receipt_replace":

        def replace_receipt(*a, **kw):
            ledger = stream._runtime._inbox._applied
            ledger["a"] = deepcopy(ledger["a"])
            return fit

        monkeypatch.setattr(ce_run, "fit_categorical_ce_readout", replace_receipt)
    elif fault == "initialization_replace":

        def replace_initialization(i):
            i.fit.work = deepcopy(i.fit.work)
            return dict(definition="scripted")

        monkeypatch.setattr(ce_run, "ce_initialization_record", replace_initialization)

    def boundary():
        if row.get("work", {}).get("fit_returns") == 1:
            if fault == "fit_replace":
                fit.work = deepcopy(fit.work)
            elif fault == "work_replace":
                row["work"] = deepcopy(row["work"])
            elif fault == "budget_replace":
                budget.clock = lambda: 0.0
            elif fault == "source_tag_replace":
                report["reserved_probe_source_training"] = deepcopy(
                    report["reserved_probe_source_training"]
                )

    with pytest.raises((ValueError, CategoricalCEFitFailed)):
        ce_run.prepare_ce_initialization(
            stream, state, spec, report, budget, check_boundary=boundary, memory=memory, row=row
        )
    assert row["complete"] is False and budget.updates_completed == 1
    assert row["preserved_work"]["fit_attempts"] == 1
    if fault == "fit_failure":
        assert row["fit_work"] == dict(backward_attempts=1, backward_rows=3)
        assert row["preserved_work"].get("initialization_attempts", 0) == 0
    else:
        assert row["preserved_fit_work"] == dict(step_returns=1)
    assert _same_values(stream._runtime._candidate.snapshot_state(), state)


def test_auxiliary_paid_report_cannot_be_reused(monkeypatch):
    monkeypatch.setattr(ce_run, "_require_probe_source", forbidden)
    row: dict[str, Any] = dict(complete=False, fit_work=dict(backward_attempts=1))
    with pytest.raises(ValueError, match="paid work cannot be renewed"):
        ce_run.prepare_ce_initialization(
            None, None, None, None, None, check_boundary=None, memory=None, row=row
        )
    assert row["fit_work"]["backward_attempts"] == 1


def test_auxiliary_source_admission_precedes_data_and_owner_work(monkeypatch):
    monkeypatch.setattr(ce_run.native, "prepare_train_fit", forbidden)
    monkeypatch.setattr(ce_run, "_unchanged_owner", forbidden)
    with pytest.raises(ValueError, match="checkpoint12"):
        ce_run.prepare_ce_initialization(
            None, None, None, {}, None, check_boundary=None, memory=None, row={}
        )


@pytest.mark.parametrize("fault", ["paid_row", "initialization", "receipt"])
def test_completed_initialization_cannot_be_changed_at_private_handoff(owner, monkeypatch, fault):
    stream, state, spec, report, budget, memory, row = inputs(owner, monkeypatch)
    _, initialization = scripted_fit(monkeypatch)
    monkeypatch.setattr(ce_run, "run_owned_probe", forbidden)

    def boundary():
        if row.get("complete") is True:
            if fault == "paid_row":
                row["fit_work"] = deepcopy(row["fit_work"])
            elif fault == "initialization":
                initialization.fit.work = deepcopy(initialization.fit.work)
            else:
                ledger = stream._runtime._inbox._applied
                ledger["a"] = deepcopy(ledger["a"])

    with pytest.raises(ValueError):
        ce_run.run_initialized_probe(
            stream,
            state,
            spec,
            report,
            budget,
            check_boundary=boundary,
            memory=memory,
            row={},
            initialization_row=row,
        )
    assert row["preserved_fit_work"] == dict(step_returns=1)


def test_receipt_authority_remains_active_during_private_phase(owner, monkeypatch):
    stream, state, spec, report, budget, memory, row = inputs(owner, monkeypatch)
    scripted_fit(monkeypatch)

    def private(*args, **kwargs):
        kwargs["check_initialization"]()
        ledger = stream._runtime._inbox._applied
        ledger["a"] = deepcopy(ledger["a"])
        kwargs["check_initialization"]()
        raise AssertionError("foreign replacement admitted")

    monkeypatch.setattr(ce_run, "run_owned_probe", private)
    with pytest.raises(ValueError, match="receipt"):
        ce_run.run_initialized_probe(
            stream,
            state,
            spec,
            report,
            budget,
            check_boundary=lambda: None,
            memory=memory,
            row={},
            initialization_row=row,
        )


@pytest.mark.parametrize("fault", ["value", "replace", "receipt_after_clock"])
def test_private_admission_refuses_changed_auxiliary_evidence_after_source_clock(
    monkeypatch, fault
):
    """Scripted native12 admission only; no actual source/probe constructors."""
    from src.adapters.torch_categorical_ce_initialization import CategoricalCEInitialization

    memory = SimpleNamespace()
    prior: dict[str, Any] = dict(complete=True, fit_work=dict(backward_attempts=1))
    state = SimpleNamespace(policy=SimpleNamespace(method="bp"))
    report: dict[str, Any] = dict(
        definition="scripted",
        method="bp",
        reserved_probe_requested=True,
        reserved_probe_source_training=dict(completed_updates=12, applied_update_count=12),
        checkpoints=[dict(state=state)],
        probe_initialization=prior,
    )
    armed = False
    stream = Stream()
    receipt_guard = ce_run._receipt_guard(stream)

    def clock():
        if armed:
            if fault == "value":
                prior["fit_work"]["backward_attempts"] = 2
            elif fault == "replace":
                report["probe_initialization"] = deepcopy(prior)
            else:
                ledger = stream._runtime._inbox._applied
                ledger["a"] = deepcopy(ledger["a"])
        return 0.0

    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=12), clock)
    budget.updates_completed = 12
    budget.process_rss_sampler = cast(Any, memory)
    monkeypatch.setattr(ToyBudgetSession, "_check_memory", lambda s: None)
    monkeypatch.setattr(probe, "_require_probe_source", lambda *a, **kw: None)
    monkeypatch.setattr(probe, "build_head_cohorts", forbidden)
    monkeypatch.setattr(probe, "copy_ce_initialization", forbidden)
    fit = SimpleNamespace(source={}, work={})
    initialization = CategoricalCEInitialization(
        "scripted", "scripted", cast(Any, fit), cast(Any, {}), 1
    )
    armed = True
    result = probe.run_owned_probe(
        state,
        None,
        report,
        budget,
        check_boundary=lambda: None,
        memory=memory,
        ce_initialization=initialization,
        check_initialization=receipt_guard if fault == "receipt_after_clock" else None,
    )
    assert (
        result["complete"] is False and result["preserved_training_work"]["attempted_updates"] == 0
    )
    if fault == "value":
        assert prior["fit_work"]["backward_attempts"] == 2
    elif fault == "replace":
        assert report["probe_initialization"] is not prior
    else:
        assert "receipt" in result["error"]
