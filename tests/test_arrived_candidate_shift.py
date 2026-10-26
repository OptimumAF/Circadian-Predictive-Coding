"""Pure/stub causal shift scheduling; no actual models, teachers or physics."""

from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
import pytest

from scripts import context_head_live as live
from scripts import run_bp_ticket_serving as cli
from scripts.arrived_candidate_shift import ArrivedCandidateShift
from src.app.actor_shadow import StableActor
from src.app.coupled_attribution import AttributionBatch
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.learner_ports import TrainingDiagnostic
from src.core.reference_controller import (
    ArrivedShiftFeedback,
    ArrivedTrainingFeedback,
    ReferenceObservation,
    ReferencePolicy,
    choose_reference,
)


@dataclass(frozen=True)
class State:
    completed_updates: int = 12


@dataclass(frozen=True)
class Snapshot:
    actor_version: str = "actor"
    learner_version: str = "c0"
    wake_updates: int = 0
    state: State = State()


class Runtime:
    def __init__(self):
        self.state = Snapshot()

    def candidate_snapshot(self):
        return deepcopy(self.state)


class ScriptedLearner:
    def __init__(self):
        self.state = State()

    def snapshot_state(self):
        return deepcopy(self.state)

    def restore_state(self, state):
        self.state = deepcopy(state)

    def fork(self):
        child = ScriptedLearner()
        child.restore_state(self.state)
        return child

    def train_batch(self, features, targets):
        self.state = replace(self.state, completed_updates=self.state.completed_updates + 1)
        return TrainingDiagnostic("scripted_test_only", 0.0)

    def predict(self, features):
        result = np.zeros((len(features), 6))
        result[:, 0] = 1.0
        return result


def packets(n=8):
    return tuple(
        AttributionBatch(
            "train",
            "A" if i < 3 else "B",
            (f"row-{i}",),
            np.full((1, 2), 0.0 if i < 3 else 1.0),
            np.zeros((1, 1)),
            num_classes=6,
        )
        for i in range(n)
    )


def prediction(state, features):
    if features[0, 0] == 1:
        return np.full((len(features), 6), 1 / 6)
    result = np.zeros((len(features), 6))
    result[:, 0] = 1.0
    return result


def read(reader, runtime=None, data=None, **changes):
    args = dict(
        arrived=1,
        completed=0,
        tick=1,
        label_delay=1,
        actor_version="actor",
        require_input=lambda: None,
    )
    args.update(changes)
    return reader.read(runtime or Runtime(), packets() if data is None else data, **args)


def test_should_observe_each_new_arrival_once_while_learning_waits():
    reader = ArrivedCandidateShift(prediction)
    runtime = Runtime()
    for k in range(1, 9):
        feedback = read(reader, runtime, arrived=k, tick=k)
        assert feedback.batch_index == k and feedback.wake_updates == 0
        assert read(reader, runtime, arrived=k, tick=k + 1) is None
    report = reader.report
    assert report["attempts"] == report["returns"] == report["rows_returned"] == 8
    assert [r["index"] for r in report["records"]] == list(range(8))
    assert [r["native_updates"] for r in report["records"]] == [12] * 8
    assert report["records"][5]["decision"]["alarms"] == (True, True, True)


def test_should_never_open_packet_or_candidate_without_a_new_label():
    class Poison:
        def __getitem__(self, key):
            pytest.fail("unarrived packet entered")

    assert read(ArrivedCandidateShift(prediction), data=Poison(), arrived=0, tick=0) is None


@pytest.mark.parametrize(
    "changes",
    [
        dict(tick=0),
        dict(arrived=2),
        dict(arrived=True),
        dict(label_delay=True),
        dict(completed=2),
        dict(label_delay=2, tick=1),
    ],
)
def test_should_refuse_noncausal_counts_before_prediction(changes):
    reader = ArrivedCandidateShift(lambda *a: pytest.fail("prediction entered"))
    with pytest.raises(ValueError):
        read(reader, **changes)
    assert reader.report["attempts"] == 0


@pytest.mark.parametrize("role", ["final_test", "probe_train", "inner_guard", "controller_train"])
def test_should_refuse_nontrain_roles_without_prediction(role):
    reader = ArrivedCandidateShift(lambda *a: pytest.fail("prediction entered"))
    with pytest.raises(ValueError, match="TRAIN"):
        read(reader, data=(replace(packets()[0], role=role),))
    assert reader.report["attempts"] == 0


@pytest.mark.parametrize("change", [dict(actor_version="stale"), dict(wake_updates=1)])
def test_should_refuse_stale_candidate_before_model_work(change):
    runtime = Runtime()
    runtime.state = replace(runtime.state, **change)
    reader = ArrivedCandidateShift(lambda *a: pytest.fail("prediction entered"))
    with pytest.raises(ValueError, match="stale"):
        read(reader, runtime)
    assert reader.report["attempts"] == 0


@pytest.mark.parametrize(
    "fault", ["raise", "invalid", "mutate_state", "mutate_owner", "mutate_data"]
)
def test_should_charge_failure_and_latch_before_any_training(fault):
    runtime, data = Runtime(), packets()

    def bad(state, features):
        if fault == "raise":
            raise ValueError("lost return")
        if fault == "invalid":
            return np.zeros((1, 6))
        result = prediction(state, features)
        if fault == "mutate_state":
            object.__setattr__(state, "completed_updates", 999)
        if fault == "mutate_owner":
            runtime.state = replace(runtime.state, state=State(999))
        if fault == "mutate_data":
            data[0].targets[:] = 4
        return result

    reader = ArrivedCandidateShift(bad)
    with pytest.raises(ValueError):
        read(reader, runtime, data)
    report = reader.report
    assert report["attempts"] == report["rows_attempted"] == 1
    assert report["returns"] == int(fault != "raise") and report["stopped"]
    assert not report["records"][0]["complete"] and "seconds" in report["records"][0]
    with pytest.raises(ValueError, match="stopped"):
        read(reader, runtime, data)
    assert reader.report["attempts"] == 1


def test_should_refuse_replaced_paid_report_and_changed_state_before_admission():
    reader, runtime = ArrivedCandidateShift(prediction), Runtime()
    read(reader, runtime)
    detached = reader.report
    detached["attempts"] = 0
    with pytest.raises(ValueError, match="report changed"):
        reader.require_paid(detached)
    reader.require_paid(reader.report)
    runtime.state = replace(runtime.state, state=State(13))
    with pytest.raises(ValueError, match="before admission"):
        reader.require_current(runtime)


def test_should_refuse_duplicate_rows_width_change_and_spent_row_cap_without_prediction():
    reader, runtime = ArrivedCandidateShift(prediction), Runtime()
    read(reader, runtime)
    for bad in (
        replace(packets()[1], row_ids=packets()[0].row_ids),
        replace(packets()[1], features=np.zeros((1, 3))),
        replace(packets()[1], features=np.zeros((48, 2)), targets=np.zeros((48, 1))),
    ):
        with pytest.raises(ValueError):
            read(reader, runtime, (packets()[0], bad), arrived=2, tick=2)
    assert reader.report["attempts"] == 1


@pytest.mark.parametrize(
    "ready,credit,alarms,expected",
    [
        (1, 1, (True, False, False), "wake1"),
        (0, 1, (True, False, False), "wait"),
        (1, 0, (True, False, False), "wait"),
        (1, 1, (False, False, False), "wait"),
    ],
)
def test_should_request_only_existing_wake_with_ready_work_and_credit(
    ready, credit, alarms, expected
):
    feedback = ArrivedShiftFeedback("train", "s", "actor", 1, 1, "c0", 0, 1, alarms)
    obs = ReferenceObservation(1, 8, ready, credit, "actor", feedback)
    assert choose_reference(ReferencePolicy("drift"), obs).action_id == expected
    ordinary = ArrivedTrainingFeedback("train", "s", "actor", 0, 1, 1, 6)
    assert choose_reference(
        ReferencePolicy("drift"), replace(obs, feedback=ordinary)
    ).action_id == ("wake1" if ready and credit else "wait")


@pytest.mark.parametrize(
    "changes",
    [
        dict(role="final_test"),
        dict(role="probe_train"),
        dict(alarms=(1, False, False)),
        dict(alarms=[True, False, False]),
        dict(batch_index=0),
        dict(wake_updates=True),
    ],
)
def test_should_refuse_malformed_scalar_shift_feedback(changes):
    feedback = ArrivedShiftFeedback("train", "s", "actor", 1, 1, "c0", 0, 1, (False,) * 3)
    with pytest.raises(ValueError):
        replace(feedback, **changes)


@pytest.mark.parametrize(
    "changes", [dict(actor_version="stale"), dict(observed_at=2, arrived_at=2)]
)
def test_should_refuse_stale_or_future_shift_feedback_in_observation(changes):
    feedback = ArrivedShiftFeedback("train", "s", "actor", 1, 1, "c0", 0, 1, (False,) * 3)
    with pytest.raises(ValueError):
        ReferenceObservation(1, 8, 1, 8, "actor", replace(feedback, **changes))


@pytest.mark.parametrize("fault", ["native", "version", "valid_advance"])
def test_should_bind_native_origin_and_candidate_version_across_arrivals(fault):
    reader, runtime = ArrivedCandidateShift(prediction), Runtime()
    read(reader, runtime)
    runtime.state = replace(runtime.state, state=State(13))
    if fault == "version":
        runtime.state = replace(runtime.state, learner_version="other", state=State())
    if fault == "valid_advance":
        runtime.state = replace(runtime.state, wake_updates=1)
        assert read(reader, runtime, arrived=2, completed=1, tick=2).wake_updates == 1
    else:
        with pytest.raises(ValueError, match="origin changed"):
            read(reader, runtime, arrived=2, tick=2)
        assert reader.report["attempts"] == 1


def test_should_preserve_ordinary_schedule_without_new_schema_or_diagnostics():
    source = ScriptedLearner()
    actor = StableActor(source, version="actor")
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=8, max_wall_seconds=30), monotonic
    )
    report: dict[str, Any] = {}
    cli._reference_schedule(
        source, actor, packets(), budget, "drift", report, budget.before_final, lambda: None
    )
    assert "arrived_shift_feedback" not in report
    assert all(e["action_id"] == "wait" for e in report["controller_events"])
    assert budget.updates_completed == 0


def test_should_refuse_selection_side_report_change_before_training(monkeypatch):
    source = ScriptedLearner()
    actor = StableActor(source, version="actor")
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=8, max_wall_seconds=30), monotonic
    )
    report: dict[str, Any] = {}
    original = cli.choose_reference

    def choose(policy, observation):
        result = original(policy, observation)
        if result.action_id == "wake1":
            report["arrived_shift_feedback"]["attempts"] = 0
        return result

    monkeypatch.setattr(cli, "choose_reference", choose)
    with pytest.raises(ValueError, match="report changed"):
        cli._reference_schedule(
            source,
            actor,
            packets(),
            budget,
            "drift",
            report,
            budget.before_final,
            lambda: None,
            arrived_shift_predict=prediction,
        )
    assert budget.updates_completed == 0


def test_should_wake_from_wait_through_original_delayed_schedule_without_changing_actor():
    source = ScriptedLearner()
    actor = StableActor(source, version="actor")
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=8, max_wall_seconds=30), monotonic
    )
    report: dict[str, Any] = {}
    runtime, _ = cli._reference_schedule(
        source,
        actor,
        packets(),
        budget,
        "drift",
        report,
        budget.before_final,
        lambda: None,
        arrived_shift_predict=prediction,
    )
    assert [e["action_id"] for e in report["controller_events"]] == ["wait"] * 6 + ["wake1"] * 3
    assert budget.updates_completed == 3
    assert runtime.candidate_snapshot().state.completed_updates == 15
    assert source.snapshot_state() == actor.snapshot_state().state == State()
    assert report["stream"]["pending_unapplied"] == 5
    assert report["arrived_shift_feedback"]["attempts"] == 8
    assert [r["wake_updates"] for r in report["arrived_shift_feedback"]["records"]] == [0] * 6 + [
        1,
        2,
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(mode="periodic"),
        dict(candidate_predict=prediction),
        dict(task_readout=prediction),
        dict(acknowledge_correct=True),
        dict(policy_change=True),
    ],
)
def test_should_refuse_incompatible_schedule_modes_before_runtime(kwargs, monkeypatch):
    monkeypatch.setattr(cli, "ActorShadowRuntime", lambda *a, **k: pytest.fail("runtime entered"))
    args = dict(mode="drift", arrived_shift_predict=prediction)
    args.update(kwargs)
    with pytest.raises(ValueError, match="arrived shift"):
        cli._reference_schedule(
            None, None, (), None, report={}, require_input=lambda: None, midway=lambda: None, **args
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(arrived_shift=1),
        dict(arrived_shift=None),
        dict(arrived_shift=True),
        dict(arrived_shift=True, reference_mode="drift", task_feedback=True),
        dict(arrived_shift=True, reference_mode="drift", candidate_feedback=True),
    ],
)
def test_should_refuse_invalid_live_optin_before_source(kwargs, monkeypatch):
    monkeypatch.setattr(live, "_source_bytes", lambda *a: pytest.fail("source entered"))
    with pytest.raises(ValueError, match="arrived shift"):
        live.run_case("unused", **kwargs)


def test_should_require_explicit_cli_mode_and_preserve_default_dispatch(tmp_path, monkeypatch):
    seen = []

    def case(path, **kwargs):
        seen.append(kwargs)
        return dict(complete=True)

    monkeypatch.setattr(live, "run_case", case)
    source = str(Path("examples/context_head_source/source.json"))
    for flags in (
        [],
        ["--context-head-serving-live"],
        [
            "--context-head-serving-live",
            "--context-live-reference",
            "drift",
            "--context-candidate-feedback",
        ],
    ):
        output = tmp_path / "refused.json"
        with pytest.raises(SystemExit):
            cli.main(
                ["--source", source, "--output", str(output), "--context-arrived-shift", *flags]
            )
        assert not output.exists()
    for flags in ([], ["--context-arrived-shift", "--context-live-reference", "drift"]):
        output = tmp_path / f"ok-{len(seen)}.json"
        assert (
            cli.main(
                ["--source", source, "--output", str(output), "--context-head-serving-live", *flags]
            )
            == 0
        )
    assert seen == [{}, dict(reference_mode="drift", arrived_shift=True)]
