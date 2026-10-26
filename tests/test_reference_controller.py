"""Causal reference choices and tiny end-to-end native task observations."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from scripts import run_bp_ticket_serving as demo
from src.core.reference_controller import (
    REFERENCE_MODES,
    ArrivedTrainingFeedback,
    ReferenceObservation,
    ReferencePolicy,
    choose_reference,
)


def observation(**changes):
    feedback = ArrivedTrainingFeedback("train", "sample", "actor", 0, 1, 1, 6)
    return replace(ReferenceObservation(2, 7, 1, 8, "actor", feedback), **changes)


@pytest.mark.parametrize(
    "mode,expected",
    [("no_learning", "wait"), ("periodic", "wake1"), ("drift", "wake1"), ("budget_aware", "wait")],
)
def test_fixed_references_choose_from_the_same_causal_observation(mode, expected):
    assert choose_reference(ReferencePolicy(mode), observation()).action_id == expected


def test_drift_requires_arrived_mistakes_above_frozen_threshold():
    policy = ReferencePolicy("drift")
    assert choose_reference(policy, observation(feedback=None)).action_id == "wait"
    feedback = replace(observation().feedback, mistakes=1, examples=10)
    assert choose_reference(policy, observation(feedback=feedback)).action_id == "wait"
    assert (
        choose_reference(policy, observation(feedback=replace(feedback, mistakes=2))).action_id
        == "wake1"
    )


def test_budget_aware_preserves_reserve_and_spends_only_at_late_opportunities():
    p = ReferencePolicy("budget_aware")
    assert choose_reference(p, observation(tick=5, remaining_opportunities=4)).action_id == "wake1"
    assert (
        choose_reference(
            p, observation(tick=8, remaining_opportunities=1, remaining_wake=4)
        ).action_id
        == "wait"
    )


@pytest.mark.parametrize("mode", REFERENCE_MODES)
@pytest.mark.parametrize("change", [{"ready_updates": 0}, {"remaining_wake": 0}])
def test_all_references_respect_readiness_and_hard_remaining_credit(mode, change):
    assert choose_reference(ReferencePolicy(mode), observation(**change)).action_id == "wait"


@pytest.mark.parametrize(
    "change",
    [
        {"mode": "learned"},
        {"period": True},
        {"period": 0},
        {"drift_threshold": True},
        {"drift_threshold": float("nan")},
        {"drift_threshold": -1},
        {"reserved_updates": -1},
    ],
)
def test_malformed_policy_is_refused(change):
    with pytest.raises(ValueError):
        replace(ReferencePolicy("periodic"), **change)


@pytest.mark.parametrize("role", ["final_test", "outer_selection", "inner_guard", "probe_train"])
def test_nontrain_feedback_is_refused_before_measurement_fields(role):
    class Poison:
        def __lt__(self, other):
            pytest.fail("read nontrain feedback")

        def __float__(self):
            pytest.fail("read nontrain feedback")

    with pytest.raises(ValueError, match="TRAIN"):
        ArrivedTrainingFeedback(
            role, "sample", "actor", 0, 1, cast(Any, Poison()), cast(Any, Poison())
        )


def test_future_and_stale_feedback_are_refused_even_for_no_learning():
    with pytest.raises(ValueError, match="not arrived"):
        observation(tick=0)
    with pytest.raises(ValueError, match="stale"):
        observation(actor_version="replacement")
    forged = observation()
    object.__setattr__(forged.feedback, "role", "final_test")
    with pytest.raises(ValueError, match="TRAIN"):
        choose_reference(ReferencePolicy("no_learning"), forged)


@pytest.mark.parametrize(
    "change",
    [
        {"mistakes": 7},
        {"examples": 0},
        {"observed_at": 2},
        {"mistakes": True},
        {"arrived_at": False},
    ],
)
def test_malformed_feedback_is_refused(change):
    with pytest.raises(ValueError):
        replace(observation().feedback, **change)


@pytest.fixture(scope="module")
def completed():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(demo.context, "BATCHES_PER_PHASE", 2)
        spec = demo.DecisionFeatureSpec(demo.FEATURES, demo.CATALOG)
        templates = demo.context._teacher_batches(spec)
        patch.setattr(demo.context, "_teacher_batches", lambda s: deepcopy(templates))
        results = {
            mode: demo.run_case(
                Path("artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json"),
                reference_mode=mode,
            )
            for mode in REFERENCE_MODES
        }
    assert all(r["complete"] for r in results.values()), {
        m: r.get("error") for m, r in results.items()
    }
    return results


@pytest.mark.parametrize(
    "mode,updates", [("no_learning", 0), ("periodic", 1), ("drift", 2), ("budget_aware", 0)]
)
def test_reference_schedule_accounts_all_opportunities_without_forced_draining(
    completed, mode, updates
):
    r = completed[mode]
    assert r["definition"] == "reference_bp_ticket_serving_v1"
    assert r["training_work"] == dict(
        attempted_updates=updates,
        attempted_rows=6 * updates,
        returned_updates=updates,
        unreturned_work_unknown=False,
    )
    assert r["stream"]["opportunities"] == 3 and r["stream"]["pending_unapplied"] == 2 - updates
    assert r["stream"]["registered_batches"] == r["stream"]["arrived_labels"] == 2
    assert (
        r["budget"]["updates_completed"] == updates
        and r["budget"]["limits"]["max_training_updates"] == 2
    )
    assert r["candidate_checkpoint"]["state"]["completed_updates"] == 96 + updates
    assert r["candidate_checkpoint"]["state"]["training_examples"] == 576 + 6 * updates
    assert r["action_snapshot"]["progress"]["decisions_attempted"] == 3
    assert r["action_snapshot"]["progress"]["wake_admissions"] == updates
    assert [u["sample_id"] for u in r["applied_updates"]] == [f"batch-{i}" for i in range(updates)]
    assert all(
        u["arrived_at"] <= u["applied_at"]
        and u["arrived_at"] > u["observed_at"]
        and u["actor_version"] == "qualified-96"
        for u in r["applied_updates"]
    )
    assert r["source_unchanged"] and r["numpy_rng_unchanged"] and r["torch_rng_unchanged"]
    assert r["final_examples"] == r["reserved_probe_updates"] == r["replay_examples"] == 0


def test_reference_feedback_and_overhead_are_common_causal_and_separate_from_task_outcomes(
    completed,
):
    common = None
    for r in completed.values():
        events = r["controller_events"]
        work = r["controller_work"]
        assert len(events) == work["selections_attempted"] == work["selections_returned"] == 3
        assert events[0]["observation"]["feedback"] is None and events[0]["action_id"] == "wait"
        feedback = [e["observation"]["feedback"] for e in events[1:]]
        assert all(
            f["role"] == "train"
            and f["arrived_at"] == i + 1
            and f["observed_at"] == i
            and f["actor_version"] == "qualified-96"
            for i, f in enumerate(feedback)
        )
        assert common is None or common == feedback
        common = feedback
        assert work["prediction_attempts"] == work["prediction_returns"] == 2
        assert work["prediction_rows"] == work["feedback_rows"] == 12
        assert work["prediction_records_peak"] <= 2 and work["prediction_records_remaining"] == 0
        assert all(
            work[k] >= 0 for k in ("selection_seconds", "observation_seconds", "admission_seconds")
        )
        assert (
            len(r["candidate_episodes"]) == 4
            and r["candidate_readout_work"]["work"]["decision_returns"] == 12
        )
        assert (
            r["reference_outcomes"]["new_before"] == 1
            and r["reference_outcomes"]["old_before"] == 2
        )
        assert r["reference_outcomes"]["new_gain"] == r["reference_outcomes"]["new_after"] - 1


def test_reference_selection_failure_keeps_observed_prefix_and_paid_work(monkeypatch):
    original = demo.choose_reference
    calls = []

    def choose(*args):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("selection fault")
        return original(*args)

    monkeypatch.setattr(demo, "choose_reference", choose)
    r = demo.run_case(
        Path("artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json"),
        reference_mode="periodic",
    )
    assert not r["complete"] and "selection fault" in r["error"]
    assert (
        r["controller_work"]["selections_attempted"] == 2
        and r["controller_work"]["selections_returned"] == 1
    )
    assert (
        len(r["controller_events"]) == 2
        and r["action_snapshot"]["progress"]["decisions_attempted"] == 1
    )
    assert r["training_work"]["attempted_updates"] == 0


def test_reference_lost_native_return_preserves_attempt_and_unknown_charge(monkeypatch):
    from src.adapters.torch_feature_learner import TorchFeatureLearner

    original = TorchFeatureLearner.train_batch

    def lost(*args):
        original(*args)
        raise RuntimeError("lost native return")

    monkeypatch.setattr(TorchFeatureLearner, "train_batch", lost)
    r = demo.run_case(
        Path("artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json"),
        reference_mode="drift",
    )
    assert not r["complete"] and "lost native return" in r["error"]
    assert r["training_work"] == dict(
        attempted_updates=1, attempted_rows=6, returned_updates=0, unreturned_work_unknown=True
    )
    p = r["action_snapshot"]["progress"]
    assert p["wake_admissions"] == p["native_attempts"] == 1 and p["native_completions"] == 0
    assert p["unreturned_work_unknown"] and r["budget"]["updates_completed"] == 0


def test_reference_cli_optin_preserves_original_dispatch_and_occupied_output(tmp_path, monkeypatch):
    calls = []

    def run(p, **kw):
        calls.append((p, kw))
        return {"complete": False}

    monkeypatch.setattr(demo, "run_case", run)
    output = tmp_path / "reference.json"
    assert (
        demo.main(["--source", "unused", "--reference", "no_learning", "--output", str(output)])
        == 1
    )
    assert calls == [(Path("unused"), {"reference_mode": "no_learning"})]
    with pytest.raises(FileExistsError):
        demo.main(["--source", "unused", "--output", str(output)])
    with pytest.raises(SystemExit):
        demo.main(["--source", "unused", "--reference", "tuned", "--output", str(tmp_path / "bad")])
