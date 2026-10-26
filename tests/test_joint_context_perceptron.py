"""Small real perceptron fixtures, with full8pass case budgeted separately."""

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import run_ticket_workflow as study
from scripts import qualify_ticket_bp_fit as native_study
from src.adapters.perceptron_learner import PerceptronLearner


def teachers():
    return {
        v: study.training_cases(policy_version=v, include_policy_context=True) for v in ("v1", "v2")
    }


@pytest.fixture(scope="module")
def completed():
    holder: dict[str, Any] = {}
    original = study.AttributionStream

    def capture(*args, **kwargs):
        stream = original(*args, **kwargs)
        holder.update(stream=stream, seed=args[0])
        return stream

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(study, "PASSES", 2)
        patch.setattr(study, "AttributionStream", capture)
        result = study.run_example(joint_policy_context=True)
    assert result["complete"], result.get("error")
    return result, holder


def test_should_reuse_exact_saved_native_training_inputs_without_native_fit():
    saved = json.loads(
        Path("artifacts/runs/context-bp-qualification-20261023/actual.json").read_bytes()
    )["training_data"]
    spec = study.DecisionFeatureSpec(study.CONTEXT_FEATURES, study.CATALOG)
    cases, batch = study.build_joint_context_batch(spec, teachers())
    other_cases, other_batch = native_study._context_teacher(spec)
    assert cases == other_cases
    assert np.array_equal(batch.features, other_batch.features)
    assert np.array_equal(batch.targets, other_batch.targets)
    assert (
        batch.features.tolist() == saved["features"] and batch.targets.tolist() == saved["targets"]
    )
    assert [dict(query=c.request.query, target_id=c.target_id) for c in cases] == saved["rows"]
    assert native_study.CONTEXT_TASKS == study.CONTEXT_TASKS


def test_should_keep_source_actor_clocks_receipts_and_zero_randomness(completed):
    r, holder = completed
    assert r["definition"] == "local_joint_context_perceptron_v1" and r["policy_context_in_query"]
    assert r["training_work"]["committed_updates"] == r["budget"]["updates_completed"] == 2
    assert r["training_work"]["committed_rows"] == 24
    assert r["numpy_rng_unchanged"] and r["source_seed_actor_unchanged"]
    assert holder["seed"].snapshot_state().completed_updates == 0
    receipts = holder["stream"].applied_updates
    assert len(receipts) == 2 and all(a.applied_at > a.observed_at for a in receipts)
    assert [(s["update"], s["state"]["training_examples"]) for s in r["snapshots"]] == [
        (0, 0),
        (2, 24),
    ]
    assert (
        r["stream"]["evaluation_scope"] == "training_only"
        and r["stream"]["score_forward_calls"] == 0
    )
    assert (
        r["costs"]["model_random_draws"] == 0
        and r["costs"]["retained_snapshot_coefficient_bytes"] == 960
    )
    assert r["costs"]["committed_training_affine_terms"] == 24 * 60
    assert r["costs"]["attempted_teacher_readout_affine_terms"] == 24 * 50


def test_should_cover_all_teacher_tasks_scenarios_and_keep_actual_truth_separate(completed):
    r, _ = completed
    expected = [
        (p, v, c, s)
        for p in (0, 2)
        for v, c in study.CONTEXT_TASKS
        for s in ("ordinary", "stale_once", "revoke_commit")
    ]
    assert [
        (x["source_updates"], x["policy_version"], x["concept"], x["episode"]["scenario"])
        for x in r["episodes"]
    ] == expected
    assert r["teacher_read_work"] == dict(attempted=24, returned=24)
    assert [len(p["rows"]) for p in r["teacher_fit"]] == [12, 12]
    assert all(
        e["episode"]["complete"] and e["episode"]["policy_context_in_query"] for e in r["episodes"]
    )
    assert all(
        r[k] == 0
        for k in ("final_examples", "probe_updates", "replay_examples", "external_actions")
    )
    assert r["training_case_generation"]["demonstration_transitions"] == 24
    assert json.loads(study._encode_report(r))["complete"]
    # No desired accuracy/success/margin sign in correctness fixtures.


@pytest.mark.parametrize("fault", ["role", "group", "missing", "schema"])
def test_should_refuse_bad_joint_teacher_before_model(fault, monkeypatch):
    supplied = teachers()
    spec = study.DecisionFeatureSpec(study.CONTEXT_FEATURES, study.CATALOG)
    if fault in ("role", "group"):
        object.__setattr__(
            supplied["v1"][0], "role" if fault == "role" else "source_group", "final_test"
        )
    elif fault == "missing":
        supplied.pop("v1")
    else:
        spec = study.DecisionFeatureSpec(study.FEATURES, study.CATALOG)
    monkeypatch.setattr(
        study, "PerceptronLearner", lambda *a: pytest.fail("bad data reached model")
    )
    with pytest.raises(ValueError):
        study.build_joint_context_batch(spec, supplied)


def test_should_preserve_lost_update_prefix_without_task_readouts(monkeypatch):
    original = PerceptronLearner.train_batch
    calls = 0

    def fail(self, features, targets):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second pass return lost")
        return original(self, features, targets)

    monkeypatch.setattr(study, "PASSES", 2)
    monkeypatch.setattr(PerceptronLearner, "train_batch", fail)
    r = study.run_example(joint_policy_context=True)
    assert not r["complete"] and "return lost" in r["error"]
    assert (
        r["training_work"]["committed_updates"] == 1 and r["training_work"]["committed_rows"] == 12
    )
    assert r["training_work"]["unobserved_attempted_work_unknown"] is True and r["episodes"] == []
    assert r["budget"]["updates_completed"] == 1


def test_should_keep_attempted_teacher_read_on_missing_return(monkeypatch):
    monkeypatch.setattr(study, "PASSES", 2)
    monkeypatch.setattr(
        study.PerceptronDecisionPredictor,
        "__call__",
        lambda *a: (_ for _ in ()).throw(RuntimeError("teacher return lost")),
    )
    r = study.run_example(joint_policy_context=True)
    assert not r["complete"] and "teacher return lost" in r["error"]
    assert r["training_work"]["committed_updates"] == 2
    assert r["teacher_read_work"] == dict(attempted=1, returned=0) and r["episodes"] == []


def test_should_retain_missing_task_return_after_complete_teacher_reads(monkeypatch):
    original = study.PerceptronDecisionPredictor.__call__
    calls = 0

    def fail(self, request):
        nonlocal calls
        calls += 1
        if calls == 13:
            raise RuntimeError("task return lost")
        return original(self, request)

    monkeypatch.setattr(study, "PASSES", 2)
    monkeypatch.setattr(study.PerceptronDecisionPredictor, "__call__", fail)
    r = study.run_example(joint_policy_context=True)
    assert not r["complete"] and r["teacher_read_work"] == dict(attempted=12, returned=12)
    assert (
        len(r["episodes"]) == 1
        and r["episodes"][0]["episode"]["unreturned_prediction_work_unknown"]
    )


def test_should_refuse_corrupted_paid_episode_and_restore_original(monkeypatch):
    original_fit = study.fit_reference
    original_read = study.PerceptronDecisionPredictor.__call__
    holder: dict[str, Any] = {}
    calls = 0

    def capture(report, *args, **kwargs):
        holder["report"] = report
        return original_fit(report, *args, **kwargs)

    def corrupt(self, request):
        nonlocal calls
        calls += 1
        result = original_read(self, request)
        if calls == 19:
            holder["report"]["episodes"][0]["episode"]["success"] = True
        return result

    monkeypatch.setattr(study, "PASSES", 2)
    monkeypatch.setattr(study, "fit_reference", capture)
    monkeypatch.setattr(study.PerceptronDecisionPredictor, "__call__", corrupt)
    r = study.run_example(joint_policy_context=True)
    assert not r["complete"] and len(r["episodes"]) == 2
    assert r["episodes"][0]["episode"]["success"] is False
    assert "paid evidence" in r["episodes"][1]["episode"]["error"]
    assert r["teacher_read_work"]["returned"] == 12


def test_should_refuse_changed_frozen_predictor_before_more_reads(monkeypatch):
    original = study.PerceptronDecisionPredictor.__call__

    def corrupt(self, request):
        result = original(self, request)
        object.__setattr__(self, "state", replace(self.state, completed_updates=1))
        return result

    monkeypatch.setattr(study, "PASSES", 2)
    monkeypatch.setattr(study.PerceptronDecisionPredictor, "__call__", corrupt)
    r = study.run_example(joint_policy_context=True)
    assert not r["complete"] and "frozen predictor" in r["error"]
    assert r["teacher_read_work"] == dict(attempted=1, returned=0)


@pytest.mark.parametrize("fault", ["started_at", "clock"])
def test_should_refuse_changed_original_budget_origin(monkeypatch, fault):
    original_fit = study.fit_reference
    original_read = study.PerceptronDecisionPredictor.__call__
    holder: dict[str, Any] = {}

    def capture(report, budget, **options):
        holder["budget"] = budget
        return original_fit(report, budget, **options)

    def corrupt(self, request):
        result = original_read(self, request)
        if fault == "started_at":
            holder["budget"].started_at -= 100.0
        else:
            before = holder["budget"].clock
            holder["budget"].clock = lambda: before()
        return result

    monkeypatch.setattr(study, "PASSES", 2)
    monkeypatch.setattr(study, "fit_reference", capture)
    monkeypatch.setattr(study.PerceptronDecisionPredictor, "__call__", corrupt)
    r = study.run_example(joint_policy_context=True)
    assert not r["complete"] and "original budget" in r["error"]
    assert r["budget"]["updates_completed"] == 2
    assert r["teacher_read_work"] == dict(attempted=1, returned=0)
    assert r["episodes"] == []


@pytest.mark.parametrize(
    "options",
    [
        dict(joint_policy_context=1),
        dict(joint_policy_context=None),
        dict(include_policy_context=True, joint_policy_context=True),
    ],
)
def test_should_refuse_invalid_modes_before_teachers_or_models(monkeypatch, options):
    monkeypatch.setattr(
        study, "training_cases", lambda **k: pytest.fail("bad mode reached teacher")
    )
    with pytest.raises(ValueError):
        study.run_example(**options)


def test_should_dispatch_joint_cli_without_changing_original_modes(tmp_path, monkeypatch):
    calls: list[dict[str, Any]] = []

    def run(**options):
        calls.append(options)
        return dict(complete=True)

    monkeypatch.setattr(study, "run_example", run)
    output = tmp_path / "joint.json"
    assert study.main(["--joint-policy-context", "--output", str(output)]) == 0
    assert calls == [dict(joint_policy_context=True)]
    with pytest.raises(FileExistsError):
        study.main(["--joint-policy-context", "--output", str(output)])
    with pytest.raises(SystemExit):
        study.main(
            ["--policy-context", "--joint-policy-context", "--output", str(tmp_path / "bad.json")]
        )
    assert len(calls) == 1
