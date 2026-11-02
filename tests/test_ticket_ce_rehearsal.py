"""Ordinary CE composition with scripted ports; no numerical model work."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pytest

from scripts import recurring_ticket_workflow_example as entry
from scripts import ticket_ce_probe as initialization
from scripts import ticket_ce_updates as updates
from scripts import ticket_native_probe as probe
from src.app.coupled_attribution import _same_value
from src.core.probe_update import ProbeUpdateFailed, ProbeUpdateReceipt
from test_ticket_ce_task_guards import Fit, State, metadata as _metadata_fixture

metadata = _metadata_fixture
COUNTS = dict(private_owners=0, callbacks=0, current_rows=0)


def forbidden(*args, **kwargs):
    pytest.fail("unbudgeted source or model work")


@pytest.mark.parametrize(
    "change",
    [
        dict(method="pc"),
        dict(reserved_probe=False),
        dict(context_return=False),
        dict(probe_initialization=None),
        dict(probe_output_only=True),
        dict(probe_update_diagnostics=True),
        dict(separate_context_heads=True),
        dict(replay="recent"),
        dict(train_update="minimum_change"),
    ],
)
def test_incompatible_ce_modes_refuse_before_source(change, monkeypatch):
    monkeypatch.setattr(entry, "prepare_training", forbidden)
    options = dict(
        method="bp",
        context_return=True,
        reserved_probe=True,
        probe_initialization="ce",
        probe_update="ce_rehearsal",
    )
    options.update(change)
    with pytest.raises(ValueError):
        entry.run_example(**options)


def test_existing_cli_forwards_only_explicit_ce_and_writes_paid_failure(tmp_path, monkeypatch):
    seen, written = [], []
    paid = dict(attempted_updates=1, events=[])
    report = dict(complete=False, reserved_probe=dict(preserved_ce_rehearsal_work=paid))

    def run(**kw):
        seen.append(kw)
        return report

    monkeypatch.setattr(entry, "run_example", run)
    from scripts import context_catalog_learning as writer

    def write(path, value, **kw):
        written.append(kw)
        return False

    monkeypatch.setattr(writer, "write_catalog_report", write)
    flags = [
        "--context-return",
        "--method",
        "bp",
        "--reserved-probe",
        "--probe-initialization",
        "ce",
        "--probe-update",
        "ce_rehearsal",
        "--output",
        str(tmp_path / "failed.json"),
    ]
    assert entry.main(flags) == 1
    assert seen == [
        dict(
            context_return=True,
            method="bp",
            reserved_probe=True,
            probe_initialization="ce",
            probe_update="ce_rehearsal",
        )
    ]
    assert written[0]["observed_work"]["probe_ce_rehearsal"] is paid
    assert written[0]["max_bytes"] == 4 * 1024**2


@pytest.mark.parametrize("extra", [[], ["--probe-output-only"], ["--probe-update-diagnostics"]])
def test_bad_ce_cli_refuses_before_dispatch(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(entry, "run_example", forbidden)
    flags = [
        "--context-return",
        "--method",
        "bp",
        "--reserved-probe",
        "--probe-update",
        "ce_rehearsal",
        "--output",
        str(tmp_path / "bad.json"),
    ]
    if extra:
        flags += ["--probe-initialization", "ce", *extra]
    with pytest.raises(SystemExit):
        entry.main(flags)


def test_ce_existing_output_refused_before_source(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "run_example", forbidden)
    path = tmp_path / "existing.json"
    path.write_text("preserved", encoding="utf-8")
    with pytest.raises(FileExistsError):
        entry.main(
            [
                "--context-return",
                "--method",
                "bp",
                "--reserved-probe",
                "--probe-initialization",
                "ce",
                "--probe-update",
                "ce_rehearsal",
                "--output",
                str(path),
            ]
        )
    assert path.read_text(encoding="utf-8") == "preserved"


@dataclass(frozen=True)
class Initialization:
    fit: Any
    state: Any


@pytest.fixture
def ordinary(metadata, monkeypatch):
    control: dict[str, Any] = dict(calls=[], owners=[], reports=[])

    class Private:
        def __init__(self, policy):
            self.policy = policy
            self.state = State(policy, 12, 72)
            self.value = 0
            control["owners"].append(self)
            COUNTS["private_owners"] += 1

        def snapshot_state(self):
            return deepcopy(self.state)

        def restore_state(self, state):
            self.state = deepcopy(state)

        def fork(self):
            child = Private(self.policy)
            child.restore_state(self.state)
            return child

        def predict(self, features):
            return np.full((len(features), 6), 1 / 6)

        def train_batch(self, *args):
            pytest.fail("CE mode called private native SGD")

    monkeypatch.setattr(probe, "TorchFeatureSnapshot", State)
    monkeypatch.setattr(probe, "TorchFeatureLearner", Private)
    monkeypatch.setattr(probe.native, "snapshot_record", asdict)
    monkeypatch.setattr(probe, "CategoricalCEInitialization", Initialization)
    monkeypatch.setattr(probe, "copy_ce_initialization", deepcopy)
    monkeypatch.setattr(probe, "ce_initialization_record", lambda i: dict(scripted=True))

    def prepare(stream, state, spec, source_report, source_budget, **kwargs):
        row = kwargs["row"]
        row.update(complete=True, work={}, fit_work={})
        control.update(
            source=stream._runtime._candidate,
            source_pin=deepcopy(state),
            source_report=source_report,
            source_budget=source_budget,
            actor=stream._runtime.actor,
            actor_pin=stream._runtime.actor.snapshot_state(),
            initialization=row,
        )
        value = Initialization(
            Fit(source=deepcopy(state), spec=spec, coefficients=((0.0,),), work={}, final_loss=0.0),
            deepcopy(state),
        )
        return value, lambda: kwargs["check_boundary"]()

    monkeypatch.setattr(initialization, "prepare_ce_initialization", prepare)

    def read_tasks(owner, phase, spec, require, row, *, boundary=None):
        require()
        row.setdefault("episodes", {})[phase] = [
            dict(
                policy_version=v,
                concept=c,
                role="controller_train",
                phase=phase,
                complete=True,
                episode=dict(complete=True, success=False, scenario="ordinary", training_updates=0),
            )
            for v, c in probe.PAIRED_TASKS
        ]
        row.setdefault("task_readout_work", {})[phase] = dict(
            decision_attempts=6,
            decision_returns=6,
            native_forward_attempts=0,
            native_forward_returns=0,
        )
        require()

    monkeypatch.setattr(probe, "_read_tasks", read_tasks)

    def apply(owner, current, retained, *, phase, check_boundary):
        index = len(control["calls"])
        COUNTS["callbacks"] += 1
        COUNTS["current_rows"] += len(current.features)
        assert current.role == ("probe_train" if phase == "fresh" else "train")
        assert all(b.role == "train" for b in retained)
        assert owner is not control["source"]
        assert _same_value(owner.state, control["source_pin"])
        control["calls"].append((phase, len(current.features), owner))
        fault = control.get("fault")
        work = dict(
            native_sgd_updates=0,
            closure_attempts=1,
            backward_attempts=1,
            forward_rows=len(current.features) + sum(len(b.features) for b in retained),
            backward_rows=len(current.features) + sum(len(b.features) for b in retained),
            probe_gradient_rows=len(current.features) if phase == "fresh" else 0,
            current_train_gradient_rows=len(current.features) if phase == "return" else 0,
            retained_train_gradient_rows=sum(len(b.features) for b in retained),
            application_attempts=0,
            application_returns=0,
            application_verified=0,
        )
        if fault == "early":
            raise ProbeUpdateFailed(
                "scripted paid closure failure", ProbeUpdateReceipt(updates.DEFINITION, work, False)
            )
        if index == 1:
            public = control["source_report"]["reserved_probe"]
            if fault == "replace":
                public["ce_rehearsal_work"] = deepcopy(public["ce_rehearsal_work"])
            elif fault == "cost":
                public["ce_rehearsal_work"]["events"][0]["work"]["backward_rows"] += 1
            elif fault == "clock":
                control["source_budget"].clock = lambda: 0.0
            elif fault == "initialization":
                control["initialization"]["work"] = {}
            elif fault == "intent":
                control["source_report"]["reserved_probe_update"] = "minimum_change"
        check_boundary()
        owner.value += 1
        work.update(application_attempts=1, application_returns=1, application_verified=1)
        receipt = ProbeUpdateReceipt(updates.DEFINITION, work, True, 0.1)
        if fault == "late":
            raise ProbeUpdateFailed("scripted failed guard after apply", receipt)
        return receipt

    monkeypatch.setattr(updates, "apply_ce_probe_update", apply)
    return control


def invoke():
    return entry.run_example(
        method="bp",
        context_return=True,
        reserved_probe=True,
        probe_initialization="ce",
        probe_update="ce_rehearsal",
    )


def test_real_ordinary_stream_probe_return_finish_and_ce_work_are_separate(ordinary):
    report = invoke()
    assert report["complete"], report.get("error")
    private = report["reserved_probe"]
    assert private["complete"] and report["source_unchanged"] and report["actor_unchanged"]
    paid = private["preserved_ce_rehearsal_work"]
    assert (paid["attempted_updates"], paid["attempted_rows"], paid["applied_updates"]) == (
        32,
        144,
        32,
    )
    assert private["preserved_training_work"] == dict(
        attempted_updates=0, attempted_rows=0, returned_updates=0
    )
    assert report["probe_updates"] == 0
    assert report["probe_auxiliary_updates"] == paid
    assert [p["updates"] for p in private["sessions"]] == [16, 16]
    assert [c[0] for c in ordinary["calls"]] == ["fresh"] * 16 + ["return"] * 16
    assert len({id(c[2]) for c in ordinary["calls"]}) == 1
    assert _same_value(ordinary["source"].snapshot_state(), ordinary["source_pin"])
    assert _same_value(ordinary["actor"].snapshot_state(), ordinary["actor_pin"])
    costs = report["reserved_probe_summary"]["reported_costs"]["ce_rehearsal"]
    assert costs["application_slots"]["native_sgd_updates"] == 0
    assert costs["auxiliary_work"]["backward_rows"] > 144


@pytest.mark.parametrize(
    "fault", ["early", "late", "replace", "cost", "clock", "initialization", "intent"]
)
def test_failed_ce_receipts_keep_costs_without_more_work_or_rebaselining(ordinary, fault):
    ordinary["fault"] = fault
    report = invoke()
    assert not report["complete"]
    private = report["reserved_probe"]
    paid = private["preserved_ce_rehearsal_work"]
    assert paid["attempted_updates"] == len(ordinary["calls"]) <= 2
    assert paid["events"][0]["work_known"]
    assert paid["events"][0]["work"]["backward_attempts"] == 1
    assert private["preserved_training_work"]["returned_updates"] == 0
    assert [s["updates"] for s in private["sessions"]] == [int(fault != "early"), 0]
    assert not private["complete"]
    if fault == "cost":
        assert (
            paid["events"][0]["work"]["backward_rows"] + 1
            == private["ce_rehearsal_work"]["events"][0]["work"]["backward_rows"]
        )


@pytest.mark.parametrize(
    "record", [None, {}, {"attempted_updates": 1}, {"attempted_updates": 1, "events": []}]
)
def test_missing_or_incomplete_ce_receipts_remain_unknown(record):
    from src.app.reserved_probe_summary import summarize_reserved_probe
    from test_reserved_probe_summary import recorded

    report = recorded()
    report["reserved_probe_update"] = "ce_rehearsal"
    if record is not None:
        report["reserved_probe"]["preserved_ce_rehearsal_work"] = record
    costs = summarize_reserved_probe(report)["reported_costs"]["ce_rehearsal"]
    assert all(value is None for value in costs["auxiliary_work"].values())
    assert costs["unknown_work_events"] is None


def test_explicit_complete_zero_ce_work_is_zero_without_inventing_unknown_work():
    from src.app.reserved_probe_summary import summarize_reserved_probe
    from test_reserved_probe_summary import recorded

    report = recorded()
    report["reserved_probe_update"] = "ce_rehearsal"
    report["reserved_probe"]["preserved_ce_rehearsal_work"] = dict(
        attempted_updates=0,
        attempted_rows=0,
        returned_updates=0,
        applied_updates=0,
        native_sgd_updates=0,
        events=[],
    )
    costs = summarize_reserved_probe(report)["reported_costs"]["ce_rehearsal"]
    assert all(value == 0 for value in costs["auxiliary_work"].values())
    assert costs["unknown_work_events"] == 0


def test_partial_ce_event_keeps_missing_gradient_work_unknown():
    from src.app.reserved_probe_summary import summarize_reserved_probe
    from test_reserved_probe_summary import recorded

    report = recorded()
    report["reserved_probe_update"] = "ce_rehearsal"
    report["reserved_probe"]["preserved_ce_rehearsal_work"] = dict(
        attempted_updates=1, events=[dict(work_known=False)]
    )
    costs = summarize_reserved_probe(report)["reported_costs"]["ce_rehearsal"]
    assert all(value is None for value in costs["auxiliary_work"].values())
    assert costs["unknown_work_events"] == 1
