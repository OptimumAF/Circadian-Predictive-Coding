"""Protected in-memory probe composition; scripted probe port, no probe fits."""

from copy import deepcopy
from dataclasses import asdict, replace, dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import context_retained_probe as composed
from scripts import context_head_live as live
from scripts import recurring_ticket_workflow_example as entry
from scripts.context_training_streams import WITHHELD_CONTEXT_RECIPES
from src.adapters.context_linear_fit_learner import ContextLinearFitLearner
from src.app.serving_promotion import PromotableActor
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.actor_ports import CandidateState
from src.core.experience import LogicalClock, Experience, LabelArrival, ExperiencePermissions
from src.core.serving_ports import ServingConfiguration
from src.shared.process_memory import ProcessRssSampler
from test_context_linear_transfer_probe import donor
from test_context_retained_publication import qualified_literal
from test_context_catalog_replay import spec


@dataclass(frozen=True)
class Receipt:
    index: int


@dataclass(frozen=True)
class Cursor:
    experiences: tuple
    labels: tuple
    applied: tuple


def scope_fixture():
    learner = donor()
    actor = PromotableActor(
        learner,
        version="fixture",
        configuration=ServingConfiguration(3, 2),
        feature_digest=lambda _: "fixture",
        metadata={"scope": "literal probe composition fixture"},
    )
    wake = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=6, max_wall_seconds=60), lambda: 0.0
    )
    replay = ToyBudgetSession(
        ToyExecutionBudget(max_replay_examples=72, max_wall_seconds=60), lambda: 0.0
    )
    wake.updates_completed, replay.replay_examples_completed = 6, 45
    runtime = SimpleNamespace(
        actor=actor,
        _candidate=learner,
        _inbox=object(),
        _budget=wake,
        applied_updates=tuple(Receipt(i) for i in range(6)),
        applied_consolidations=tuple(Receipt(i) for i in range(5)),
        acknowledged_experiences=(),
    )
    runtime.candidate_snapshot = lambda: CandidateState(
        "fixture", "c0", runtime._candidate.snapshot_state(), 6, 5, 5
    )
    sources = tuple(
        Experience(
            f"batch-{i}",
            "attribution",
            i,
            "fixture",
            ((0.0,),),
            "train",
            ExperiencePermissions(training=True, replay=True),
        )
        for i in range(6)
    )
    labels = tuple(
        LabelArrival(
            f"label-{i}", f"batch-{i}", "attribution", i + 1, "fixture", ((0.0,),), role="train"
        )
        for i in range(6)
    )
    runtime._inbox = SimpleNamespace(
        _clock=LogicalClock(24),
        _budget=wake,
        _learner=learner,
        applied_updates=runtime.applied_updates,
    )
    runtime._inbox._borrow_cursor = lambda: Cursor(sources, labels, runtime._inbox.applied_updates)
    runtime._consolidations = list(runtime.applied_consolidations)
    runtime.with_candidate_snapshot = lambda operation: operation(runtime.candidate_snapshot())
    helper = SimpleNamespace(
        _runtime=runtime, _replay_budget=replay, report=dict(charged_exposures=45)
    )
    report = dict(
        native_calls=[dict(rows=9, complete=True)],
        training_work=dict(attempted_updates=11, attempted_rows=72, returned_updates=11),
        joint_rehearsal=deepcopy(helper.report),
        candidate_return_readout=dict(counts=[2, 4]),
        candidate_state=asdict(learner.snapshot_state()),
        action_snapshot={},
        sharing_snapshot={},
        stream={},
    )
    record: dict[str, Any] = {}
    report["isolated_transfer_probe"] = record
    return runtime, helper, report, record


def scripted(donor, state, cohorts, schema, guard, memory, name, *, row, **options):
    assert type(donor) is ContextLinearFitLearner
    assert state.completed_updates == 23 and state.training_examples == 144
    assert options == dict(
        separate_context_heads=True, linear_fit=True, head_tasks=composed.TRANSFER_HEAD_TASKS
    )
    guard()
    row["probe"] = dict(
        attempted_updates=16,
        completed_updates=16,
        attempted_training_examples=48,
        completed_training_examples=48,
        return_phase=dict(
            attempted_updates=16,
            completed_updates=16,
            attempted_training_examples=96,
            completed_training_examples=96,
        ),
    )
    return row


def test_should_use_plain_independent_donor_and_preserve_original_scope(monkeypatch):
    runtime, helper, report, record = scope_fixture()
    before = runtime.candidate_snapshot()
    monkeypatch.setattr(composed, "run_arm", scripted)
    composed.run_probe(runtime, spec(), helper, lambda: None, None, report, record)
    assert record["donor_unchanged"] and record["retained_scope_verified"]
    assert record["private_work"] == dict(
        attempted_updates=32,
        completed_updates=32,
        attempted_training_examples=144,
        completed_training_examples=144,
        unreturned_work_unknown=False,
    )
    assert runtime.candidate_snapshot() == before and runtime._budget.updates_completed == 6


@pytest.mark.parametrize(
    "fault",
    [
        "candidate",
        "actor",
        "budget",
        "receipt",
        "clock",
        "paid",
        "outer",
        "arm",
        "header",
        "consent",
        "label",
    ],
)
def test_should_refuse_mutated_retained_scope_before_private_work(fault, monkeypatch):
    runtime, helper, report, record = scope_fixture()

    def bad(*args, **kwargs):
        if fault == "candidate":
            runtime._candidate = runtime._candidate.fork()
        elif fault == "actor":
            object.__setattr__(
                runtime.actor._slot, "generation", runtime.actor._slot.generation + 1
            )
        elif fault == "budget":
            object.__setattr__(runtime._budget.budget, "max_training_updates", 7)
        elif fault == "receipt":
            runtime.applied_updates = (
                deepcopy(runtime.applied_updates[0]),
                *runtime.applied_updates[1:],
            )
            runtime._inbox.applied_updates = runtime.applied_updates
        elif fault == "clock":
            runtime._inbox._clock.advance_to(25)
        elif fault == "consent":
            object.__setattr__(
                runtime._inbox._borrow_cursor().experiences[0].permissions, "replay", False
            )
        elif fault == "label":
            object.__setattr__(runtime._inbox._borrow_cursor().labels[0], "arrived_at", 99)
        elif fault == "paid":
            report["native_calls"][0]["complete"] = False
        elif fault == "outer":
            report["isolated_transfer_probe"] = {}
        elif fault == "arm":
            record["arm"] = {}
        else:
            record["donor_native_updates"] = 12
        args[4]()
        pytest.fail("admitted work after changed scope")

    monkeypatch.setattr(composed, "run_arm", bad)
    with pytest.raises(ValueError, match="retained donor"):
        composed.run_probe(runtime, spec(), helper, lambda: None, None, report, record)
    assert (
        record["retained_scope_verified"] is None
        and record["private_work"]["unreturned_work_unknown"]
    )
    assert record["private_work"]["attempted_updates"] is None


def test_should_keep_missing_private_result_unknown_and_original_error(monkeypatch):
    runtime, helper, report, record = scope_fixture()

    def lost(*a, **k):
        raise RuntimeError("lost delegated probe result")

    monkeypatch.setattr(composed, "run_arm", lost)
    with pytest.raises(RuntimeError, match="lost delegated"):
        composed.run_probe(runtime, spec(), helper, lambda: None, None, report, record)
    assert record["donor_unchanged"] and record["retained_scope_verified"] is None
    assert (
        record["private_work"]["unreturned_work_unknown"]
        and record["private_work"]["completed_updates"] is None
    )


def test_should_refuse_a_probe_delegate_that_changes_the_donor(monkeypatch):
    runtime, helper, report, record = scope_fixture()

    def changed(seed, *args, **kwargs):
        result = scripted(seed, *args, **kwargs)
        seed._state = replace(
            seed._state,
            completed_updates=24,
            heads=(replace(seed._state.heads[0], completed_updates=20), seed._state.heads[1]),
        )
        return result

    monkeypatch.setattr(composed, "run_arm", changed)
    with pytest.raises(ValueError, match="owned donor"):
        composed.run_probe(runtime, spec(), helper, lambda: None, None, report, record)
    assert record["donor_unchanged"] is False and record["retained_scope_verified"] is None


@pytest.mark.parametrize(
    "mode,seconds",
    [("retained_joint_probe", 120), ("retained_joint", 30), ("retained_joint_publish", 30)],
)
def test_should_declare_source_lifetime_at_birth_only_for_explicit_combined_mode(
    mode, seconds, monkeypatch
):
    seen = []

    def capture(limits, *a, **kw):
        seen.append(limits.max_wall_seconds)
        raise RuntimeError("fixture source budget boundary")

    monkeypatch.setattr(entry, "ToyBudgetSession", capture)
    result = entry.run_example(
        method="linear_fit", context_return=True, separate_context_heads=True, paired_serving=mode
    )
    assert result["complete"] is False and "fixture source budget boundary" in result["error"]
    assert seen == [seconds]


@pytest.mark.parametrize(
    "change",
    [
        dict(_probe_retained=1),
        dict(_retained_joint=False),
        dict(_publish_retained=True),
        dict(acknowledge_correct=True),
    ],
)
def test_should_refuse_mixed_probe_modes_before_source(change, monkeypatch):
    monkeypatch.setattr(live, "_source_bytes", lambda *a: pytest.fail("source opened"))
    kwargs = dict(
        _linear_fit=True,
        _owned_source=(object(), lambda: None),
        _retained_joint=True,
        _probe_retained=True,
        stream_recipe=WITHHELD_CONTEXT_RECIPES[0],
    )
    kwargs.update(change)
    with pytest.raises(ValueError):
        live.run_case(None, **kwargs)


def test_should_finish_retained_serving_before_delegating_isolated_probe(monkeypatch):
    calls = []

    def probe(*args, **kwargs):
        calls.append((args[1].completed_updates, args[1].training_examples))
        return scripted(*args, **kwargs)

    monkeypatch.setattr(composed, "run_arm", probe)
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        result = live.run_case(
            None,
            _linear_fit=True,
            _owned_source=(qualified_literal(), lambda: None),
            _memory=memory,
            _retained_joint=True,
            _probe_retained=True,
            stream_recipe=WITHHELD_CONTEXT_RECIPES[0],
        )
    assert result["complete"], result.get("error")
    assert calls == [(23, 144)] and result["source_unchanged"] and result["actor_unchanged"]
    assert result["training_work"] == dict(
        attempted_updates=11, attempted_rows=72, returned_updates=11
    )
    assert (
        result["isolated_transfer_probe"]["donor_unchanged"]
        and result["isolated_transfer_probe"]["retained_scope_verified"]
    )
    assert result["isolated_transfer_probe"]["private_work"]["completed_updates"] == 32
    assert (
        result["live_serving"]["closed"]
        and result["final_examples"] == result["probe_updates"] == 0
    )
