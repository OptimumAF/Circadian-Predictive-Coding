"""Known-A rollout routing and paid receipt retention without native learning.

Fake returned game receipts are engineering inputs. Existing fake TRAIN/probe
composition supplies the benchmark endpoint; no model or physics benefit is
inferred from these orchestration tests.
"""

from copy import deepcopy

import numpy as np
import pytest

from src.adapters import minigrid_learnability as pilot
from src.adapters import minigrid_train_rollout as rollout
from src.adapters.minigrid_policy_evaluation import MiniGridEvaluationFailed
from test_minigrid_a_return import RETURN_FLAGS
from test_minigrid_learning_pilot import evaluator as evaluator
from test_minigrid_matched_a_only import METHODS, matched_environment
from test_minigrid_pc_qualification import environment
from test_torch_feature_rehearsal_pilot import forbidden


def live_report(method="public_invariant_bp", *, status="complete", predictions=4):
    return dict(
        status=status,
        reason=None if status == "complete" else "injected live failure",
        method=method,
        updates=128,
        native_training_examples=1152,
        role="knownTRAIN_A_only",
        scores=[],
        episodes=[
            dict(
                context="A",
                cue=cue,
                source_role="train",
                status="complete"
                if status == "complete"
                else "failed"
                if index == 0
                else "not_run",
                success=False if status == "complete" else None,
                **{"return": 0.0 if status == "complete" else None},
                environment=dict(environment_steps=2 if status == "complete" else predictions)
                if index == 0 or status == "complete"
                else None,
            )
            for index, cue in enumerate(("blue", "red"))
        ],
        work=dict(
            raw_score_forwards_completed=0,
            raw_score_rows_completed=0,
            live_prediction_calls_returned=predictions,
            environment_resets_completed=2 if status == "complete" else 1,
            environment_steps_attempted=predictions,
            environment_steps_returned=predictions,
            failure_work_may_be_unobserved=status != "complete",
        ),
    )


def live_record(method="public_invariant_bp", *, status="complete", predictions=4):
    return dict(
        status=status,
        reason=None if status == "complete" else "partial rollout retained",
        checkpoint_updates=128,
        report=live_report(method, status=status, predictions=predictions),
        charged_updates=0,
        replay_examples=0,
        budget_binding_preserved=True,
        elapsed_seconds=0.0,
    )


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], np.bool_(True)])
def test_should_reject_nonboolean_rollout_requests_before_resources(monkeypatch, value):
    for name in ("require_torch", "ProcessRssSampler", "MiniGridKeyDoor", "run_train_rollout"):
        monkeypatch.setattr(pilot, name, forbidden)
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**RETURN_FLAGS, train_a_rollout=value)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"public_history": True},
        {"public_history": True, "invariant_history": True, "qualification_method": "bp"},
        {
            "public_history": True,
            "invariant_history": True,
            "pc_qualification": True,
            "acquisition_only": True,
        },
    ],
)
def test_should_require_the_explicit_bound_singleton_a_only_route(monkeypatch, options):
    for name in ("require_torch", "ProcessRssSampler", "MiniGridKeyDoor", "run_train_rollout"):
        monkeypatch.setattr(pilot, name, forbidden)
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**options, train_a_rollout=True)


@pytest.mark.parametrize("method", METHODS)
def test_should_preserve_training_and_add_only_the_known_a_plan_to_each_method(monkeypatch, method):
    for name in ("require_torch", "ProcessRssSampler", "MiniGridKeyDoor", "run_train_rollout"):
        monkeypatch.setattr(pilot, name, forbidden)
    options = dict(RETURN_FLAGS, qualification_method=method)
    previous = pilot.describe_protocol(**options)
    selected = pilot.describe_protocol(**options, train_a_rollout=True)
    assert selected["methods"] == previous["methods"] == [f"public_invariant_{method}"]
    for key in (
        "seed",
        "learning_rate",
        "momentum",
        "inference_steps",
        "inference_learning_rate",
        "cpc_configs",
        "task_sequence",
        "checkpoint_updates",
        "feature_dim_by_method",
    ):
        assert selected[key] == previous[key]
    for key in (
        "native_train_calls",
        "training_rows",
        "score_forwards",
        "score_rows",
        "teacher_episodes",
        "final_episodes",
    ):
        assert selected["planned_work"][key] == previous["planned_work"][key]
    assert selected["train_a_rollout"]["checkpoint_updates"] == 128
    assert selected["train_a_rollout"]["cues"] == ["blue", "red"]
    assert selected["train_a_rollout"]["role"] == "train"
    assert selected["planned_work"]["learner_live_episodes"] == 2
    assert selected["planned_work"]["maximum_live_prediction_forwards"] == 32
    assert selected["planned_work"]["maximum_live_environment_steps"] == 32
    assert selected["planned_work"]["maximum_total_prediction_forwards"] == 126


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"public_history": True},
        {"public_history": True, "invariant_history": True, "qualification_method": "pc"},
        RETURN_FLAGS,
    ],
)
def test_should_leave_default_descriptors_equal_when_the_new_flag_is_false(options):
    assert pilot.describe_protocol(**options) == pilot.describe_protocol(
        **options, train_a_rollout=False
    )


@pytest.mark.parametrize("method", METHODS)
def test_should_use_benchmark128_after_both_disposable_probe_returns(
    monkeypatch, evaluator, method
):
    case = matched_environment(monkeypatch, evaluator, method)
    calls = []

    def read_live(arm, *, check_boundary):
        check_boundary()
        assert arm is case.work["arms"][0]
        point, state = arm["states"][-1]
        assert point == state.completed_updates == 128 and state.training_examples == 1152
        assert arm["source"].snapshot_state().completed_updates == 0
        assert len(case.native["trains"]) == 64 and len(case.native["predicts"]) == 94
        assert all(
            slot["status"] == "complete" and slot["return_charged_updates"] == 16
            for slot in case.work["slots"]
        )
        calls.append((arm, state))
        return live_record(f"public_invariant_{method}"), lambda: None

    monkeypatch.setattr(pilot, "run_train_rollout", read_live)
    report = pilot.run_pilot(
        **dict(RETURN_FLAGS, qualification_method=method), train_a_rollout=True
    )
    assert report["status"] == "complete", report["reason"]
    assert len(calls) == 1
    row = report["methods"][0]
    live = row["train_a_rollout"]
    assert live["status"] == "complete" and live["checkpoint_updates"] == 128
    assert live["charged_updates"] == live["replay_examples"] == 0
    assert live["report"]["method"] == f"public_invariant_{method}"
    assert [slot["source_role"] for slot in live["report"]["episodes"]] == ["train", "train"]
    assert row["charged_updates"] == 128 and row["native_training_examples"] == 1152
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert report["work"]["native_train_calls"] == 192 and report["work"]["training_rows"] == 1728
    assert report["work"]["total_score_forwards_completed"] == 94
    assert report["work"]["total_prediction_forwards_completed"] == 98
    assert report["work"]["learner_live_episodes"] == 2
    assert report["work"]["train_a_rollout"]["environment_steps_returned"] == 4
    assert report["scientific_advantage_verified"] is False


def test_should_preserve_a_returned_partial_rollout_without_repeating_training(
    monkeypatch, evaluator
):
    case = matched_environment(monkeypatch, evaluator, "bp")
    paid = live_record(status="failed", predictions=1)
    monkeypatch.setattr(pilot, "run_train_rollout", lambda arm, **options: (paid, lambda: None))
    report = pilot.run_pilot(**RETURN_FLAGS, train_a_rollout=True)
    assert report["status"] == "incomplete" and report["reason"]
    assert report["methods"][0]["train_a_rollout"] == paid
    assert report["work"]["train_a_rollout"]["live_prediction_calls_returned"] == 1
    assert len(case.native["trains"]) == 64 and len(case.native["predicts"]) == 94
    assert report["methods"][0]["probes"][1]["return_charged_updates"] == 16


def test_should_allow_plain_a_only_acquisition_without_requiring_old_fit_or_a_return(
    monkeypatch, evaluator
):
    case = environment(monkeypatch, evaluator, method="public_invariant_bp", acquisition_only=True)
    calls = []

    def read_live(arm, *, check_boundary):
        check_boundary()
        assert len(case.native["trains"]) == 32 and len(case.native["predicts"]) == 46
        assert arm["states"][-1][0] == arm["states"][-1][1].completed_updates == 128
        calls.append(arm)
        return live_record(), lambda: None

    monkeypatch.setattr(pilot, "run_train_rollout", read_live)
    report = pilot.run_pilot(
        public_history=True,
        invariant_history=True,
        qualification_method="bp",
        acquisition_only=True,
        train_a_rollout=True,
    )
    assert report["status"] == "complete", report["reason"]
    assert len(calls) == 1 and report["methods"][0]["train_a_rollout"]["status"] == "complete"
    assert report["work"]["native_train_calls"] == 160 and report["work"]["training_rows"] == 1440
    assert report["work"]["total_prediction_forwards_completed"] == 50
    assert report["work"]["learner_live_episodes"] == 2


def test_should_refuse_corrupted_preceding_probe_evidence_before_live_admission(
    monkeypatch, evaluator
):
    case = matched_environment(monkeypatch, evaluator, "bp")
    monkeypatch.setattr(pilot, "run_train_rollout", forbidden)

    def replace_prior(owner):
        if len(case.native.get("trains", [])) == 33:
            prior = case.work["slots"][0]
            prior["report"]["return_phase"] = deepcopy(prior["report"]["return_phase"])

    case.native["on_train"] = replace_prior
    report = pilot.run_pilot(**RETURN_FLAGS, train_a_rollout=True)
    assert report["status"] == "incomplete"
    live = report["methods"][0]["train_a_rollout"]
    assert live["status"] == "not_run" and live["report"] is None
    assert len(case.native["trains"]) == 33
    assert report["methods"][0]["probes"][0]["return_charged_updates"] == 16


@pytest.mark.parametrize("fault", ["evaluator", "final_boundary", "budget_clock"])
def test_should_preserve_the_thin_helpers_paid_receipt_and_original_zero_update_budget(
    monkeypatch, fault
):
    monkeypatch.setattr(rollout, "monotonic", lambda: 1.0)
    source, state = object(), object()
    arm = dict(source=source, states=[(0, object()), (128, state)])
    calls = []
    late = [False]
    paid = live_report(
        status="failed" if fault == "evaluator" else "complete",
        predictions=1 if fault == "evaluator" else 4,
    )

    def boundary():
        if late[0] and fault == "final_boundary":
            raise ValueError("late preceding evidence changed")

    def evaluate(actual_source, actual_state, **options):
        assert actual_source is source and actual_state is state
        assert options["updates"] == 128 and options["known_a_only"] is True
        budget = options["budget"]
        assert budget.budget.max_training_updates == budget.budget.max_replay_examples == 0
        options["check_boundary"]()
        calls.append(budget)
        if fault == "evaluator":
            raise MiniGridEvaluationFailed(paid)
        if fault == "budget_clock":
            budget.clock = lambda: 1.0
        late[0] = True
        return paid

    monkeypatch.setattr(rollout, "evaluate_neural_representation_checkpoint", evaluate)
    record, guard = rollout.run_train_rollout(arm, check_boundary=boundary)
    assert record["status"] == "failed" and record["reason"]
    assert record["report"] == paid and record["report"] is not paid
    assert len(calls) == 1 and calls[0].updates_completed == calls[0].replay_examples_completed == 0
    assert record["charged_updates"] == record["replay_examples"] == 0
    assert record["budget_binding_preserved"] is (fault != "budget_clock")
    paid["work"]["live_prediction_calls_returned"] = 99
    assert record["report"]["work"]["live_prediction_calls_returned"] != 99
    if fault == "budget_clock":
        with pytest.raises(ValueError):
            guard()


def test_should_retain_completed_rollout_work_if_a_later_guard_refuses_publication(
    monkeypatch, evaluator
):
    case = matched_environment(monkeypatch, evaluator, "bp")
    paid = live_record()

    def late_guard():
        raise ValueError("rollout binding changed after returned receipt")

    monkeypatch.setattr(pilot, "run_train_rollout", lambda arm, **options: (paid, late_guard))
    report = pilot.run_pilot(**RETURN_FLAGS, train_a_rollout=True)
    assert report["status"] == "incomplete" and "rollout binding changed" in report["reason"]
    assert report["methods"][0]["train_a_rollout"] == paid
    assert report["work"]["train_a_rollout"]["live_prediction_calls_returned"] == 4
    assert len(case.native["trains"]) == 64
