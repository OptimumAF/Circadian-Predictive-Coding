"""Task-return selection and paid failures; metadata fixtures are not studies."""

from copy import deepcopy
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, cast

import pytest

from scripts import context_head_live as live
from scripts import context_controller_learning as learning
from scripts import run_bp_ticket_serving as cli
from scripts.candidate_task_feedback import CandidateTaskFeedback
from src.core.contextual_controller import ContextualChoice, choose_contextual
from src.core.reference_controller import (
    ArrivedTaskFeedback,
    ReferenceObservation,
    ReferencePolicy,
    choose_reference,
)
from test_candidate_training_feedback import setup
from test_controller_task_trials import task_rows


def observation(feedback, **changes):
    values = dict(
        tick=3,
        remaining_opportunities=2,
        ready_updates=1,
        remaining_wake=1,
        actor_version="base",
        feedback=feedback,
    )
    return ReferenceObservation(**(values | changes))


@pytest.mark.parametrize("failed,action", [(0, "wait"), (1, "wake1"), (6, "wake1")])
def test_should_select_fixed_drift_from_task_units_without_entering_classifier_fit(failed, action):
    feedback = ArrivedTaskFeedback("controller_train", "tasks", "base", 2, 3, failed, 6)
    obs = observation(feedback)
    assert choose_reference(ReferencePolicy("drift"), obs).action_id == action
    with pytest.raises(ValueError, match="classifier contextual"):
        choose_contextual(ContextualChoice("wait", "wake1"), obs)
    with pytest.raises(ValueError, match="stale"):
        observation(feedback, actor_version="replaced")
    with pytest.raises(ValueError, match="arrived"):
        observation(feedback, tick=2)


@pytest.mark.parametrize(
    "role", ["train", "final_test", "controller_validation", "probe_train", "inner_guard"]
)
def test_should_refuse_wrong_task_role_before_count_measurement(role):
    with pytest.raises(ValueError, match="controller_train"):
        ArrivedTaskFeedback(role, "id", "base", 0, 1, cast(Any, object()), cast(Any, object()))


@pytest.mark.parametrize(
    "changes",
    [
        dict(tasks=0),
        dict(failed_tasks=7),
        dict(failed_tasks=True),
        dict(observed_at=4),
        dict(arrived_at=-1),
    ],
)
def test_should_refuse_incomplete_or_noncausal_task_scalar(changes):
    feedback = ArrivedTaskFeedback("controller_train", "tasks", "base", 2, 3, 1, 6)
    with pytest.raises(ValueError):
        replace(feedback, **changes)


def metadata_readout(state, phase, require, row):
    require()
    row["task_outcomes"][phase] = task_rows(phase, state.completed_updates, new=2)
    row["task_readout_work"][phase] = dict(attempts=18, returns=18, readouts=18, typed_attempts=18)
    require()
    return [2, 4]


def scalar_reader(readout=metadata_readout, limit=2):
    @dataclass
    class State:
        completed_updates: int = 12

    @dataclass
    class Candidate:
        actor_version: str
        learner_version: str
        wake_updates: int
        state: State

    snapshot = Candidate(
        actor_version="base",
        learner_version="c0",
        wake_updates=0,
        state=State(),
    )
    runtime = SimpleNamespace(candidate_snapshot=lambda: deepcopy(snapshot))
    reader = CandidateTaskFeedback(readout, limit)
    kwargs = dict(
        arrived=1,
        completed=0,
        tick=2,
        label_delay=2,
        actor=SimpleNamespace(version="base"),
        require_input=lambda: None,
    )
    return reader, runtime, [SimpleNamespace(role="train")], kwargs


def test_should_preserve_original_clocks_and_not_acknowledge_correct_pending_packet():
    reader, runtime, batches, kwargs = scalar_reader(limit=1)
    feedback = reader.read(runtime, batches, **kwargs)
    assert type(feedback) is ArrivedTaskFeedback and feedback.failed_tasks == 0
    row = reader.report["records"][0]
    assert (row["label_observed_at"], row["label_arrived_at"], row["read_at"]) == (0, 2, 2)
    assert (feedback.observed_at, feedback.arrived_at) == (2, 2)
    assert row["complete"] and row["native_updates"] == 12 and row["wake_updates"] == 0
    assert not hasattr(runtime, "acknowledged_experiences")
    assert choose_reference(ReferencePolicy("drift"), observation(feedback)).action_id == "wait"
    with pytest.raises(ValueError, match="allowance"):
        reader.read(runtime, batches, **kwargs)


def test_should_avoid_packet_and_candidate_before_labels_or_after_drain():
    reader, _, _, kwargs = scalar_reader()

    class Poison:
        def __getitem__(self, key):
            pytest.fail("packet entered")

    assert reader.read(None, Poison(), **(kwargs | dict(arrived=0))) is None
    assert reader.report["records"] == []


@pytest.mark.parametrize("role", ["probe_train", "final_test", "inner_guard"])
def test_should_refuse_unarrived_or_wrong_role_before_candidate(role):
    reader, _, _, kwargs = scalar_reader()
    with pytest.raises(ValueError, match="ordinary TRAIN"):
        reader.read(None, [SimpleNamespace(role=role)], **kwargs)
    with pytest.raises(ValueError, match="ordinary TRAIN"):
        reader.read(None, [SimpleNamespace(role="train")], **(kwargs | dict(tick=1)))


@pytest.mark.parametrize(
    "fault", ["raise", "role", "incomplete", "version", "work", "count", "extra", "candidate"]
)
def test_should_retain_paid_failure_and_refuse_next_task_read(fault):
    holder: dict[str, Any] = {}

    def bad(state, phase, require, row):
        assert holder["reader"].report["records"][-1] is row
        if fault == "raise":
            raise ValueError("lost typed return")
        counts = metadata_readout(state, phase, require, row)
        if fault == "role":
            row["role"] = "final_test"
        if fault == "incomplete":
            row["task_outcomes"][phase][0]["complete"] = False
        if fault == "version":
            row["task_outcomes"][phase][0]["reads"][0]["actor_version"] = "replaced"
        if fault == "work":
            row["task_readout_work"][phase]["returns"] = 0
        if fault == "count":
            counts = [99, 4]
        if fault == "extra":
            row["task_outcomes"]["final_test"] = object()
        if fault == "candidate":
            state.completed_updates = 999
        return counts

    reader, runtime, batches, kwargs = scalar_reader(bad)
    holder["reader"] = reader
    with pytest.raises(ValueError):
        reader.read(runtime, batches, **kwargs)
    row = reader.report["records"][0]
    assert not row["complete"] and "error" in row and row["seconds"] >= 0
    with pytest.raises(ValueError, match="incomplete|changed"):
        reader.read(None, None, **kwargs)
    assert len(reader.report["records"]) == 1


@pytest.mark.parametrize("fault", ["mutate", "replace", "records", "header", "remove"])
def test_should_refuse_preceding_paid_evidence_change_before_more_candidate_work(fault):
    reader, runtime, batches, kwargs = scalar_reader()
    reader.read(runtime, batches, **kwargs)
    rows = reader.report["records"]
    original = deepcopy(rows[0])
    if fault == "mutate":
        rows[0]["failed_tasks"] = 6
    if fault == "replace":
        rows[0] = deepcopy(rows[0])
    if fault == "records":
        reader.report["records"] = list(rows)
    if fault == "header":
        reader.report["attempt_limit"] = 999
    if fault == "remove":
        rows.clear()
    with pytest.raises(ValueError, match="paid"):
        reader.read(None, None, **kwargs)
    if fault in ("mutate", "replace", "remove"):
        assert rows[0] == original


def test_should_execute_actual_private_task_returns_before_two_bounded_updates():
    state, source, actor, batches, budget = setup()
    report: dict[str, Any] = {}
    runtime, _ = cli._reference_schedule(
        source,
        actor,
        batches,
        budget,
        "drift",
        report,
        budget.before_final,
        lambda: None,
        task_readout=learning._task_readout,
    )
    assert [e["action_id"] for e in report["controller_events"]] == ["wait", "wake1", "wake1"]
    rows = report["task_feedback"]["records"]
    assert len(rows) == 2 and all(r["complete"] and r["tasks"] == 6 for r in rows)
    assert [r["native_updates"] for r in rows] == [12, 13]
    assert all(
        e["observation"]["feedback"]["role"] == "controller_train"
        for e in report["controller_events"][1:]
    )
    assert budget.updates_completed == 2 and runtime.acknowledged_experiences == ()
    assert source.snapshot_state() == actor.snapshot_state().state == state


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(task_feedback=1),
        dict(task_feedback=True),
        dict(task_feedback=True, reference_mode="drift", candidate_feedback=True),
        dict(
            task_feedback=True,
            reference_mode="drift",
            contextual_policy=ContextualChoice("wait", "wake1"),
        ),
    ],
)
def test_should_refuse_unsupported_api_before_source(kwargs, monkeypatch):
    monkeypatch.setattr(live, "_source_bytes", lambda p: pytest.fail("source entered"))
    with pytest.raises(ValueError, match="task feedback"):
        live.run_case("unused", **kwargs)


def test_should_dispatch_explicit_task_feedback_cli_and_refuse_other_modes(tmp_path, monkeypatch):
    seen = []

    def case(path, **kwargs):
        seen.append(kwargs)
        return dict(complete=True)

    monkeypatch.setattr(live, "run_case", case)
    output = tmp_path / "out.json"
    base = ["--source", "unused", "--output", str(output), "--context-task-feedback"]
    with pytest.raises(SystemExit):
        cli.main(base)
    assert not output.exists()
    assert (
        cli.main([*base, "--context-head-serving-live", "--context-live-reference", "drift"]) == 0
    )
    assert seen == [dict(reference_mode="drift", task_feedback=True)]


@pytest.mark.parametrize("fault", ["actor", "reentrant"])
def test_should_stop_private_work_on_actor_swap_or_reentrant_reader(fault):
    holder: dict[str, Any] = {}

    def bad(state, phase, require, row):
        metadata_readout(state, phase, require, row)
        if fault == "actor":
            holder["kwargs"]["actor"].version = "changed"
        else:
            holder["reader"].read(None, None, **holder["kwargs"])
        require()
        pytest.fail("continued after actor swap/reentry")

    reader, runtime, batches, kwargs = scalar_reader(bad)
    holder.update(reader=reader, kwargs=kwargs)
    with pytest.raises(ValueError, match="actor version|reentrant"):
        reader.read(runtime, batches, **kwargs)
    assert len(reader.report["records"]) == 1
    assert not reader.report["records"][0]["complete"]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(mode="periodic"),
        dict(candidate_predict=lambda: None),
        dict(followup_plan=True),
        dict(policy_change=True),
    ],
)
def test_should_refuse_incompatible_task_schedule_before_runtime(kwargs, monkeypatch):
    monkeypatch.setattr(cli, "ActorShadowRuntime", lambda *a, **k: pytest.fail("runtime entered"))
    values = dict(mode="drift", task_readout=metadata_readout)
    values.update(kwargs)
    with pytest.raises(ValueError, match="task feedback"):
        cli._reference_schedule(
            None,
            None,
            (),
            None,
            report={},
            require_input=lambda: None,
            midway=lambda: None,
            **values,
        )
