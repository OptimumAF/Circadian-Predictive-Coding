"""True same-policy recurrence through the existing delayed stream and sandbox."""

from copy import deepcopy
from dataclasses import dataclass, replace
import json

import numpy as np
import pytest

from scripts import recurring_ticket_workflow_example as study
from scripts import run_ticket_workflow as teacher
from scripts.ticket_controller_streams import build_context_return_stream
from test_ticket_workflow import decision
from test_versioned_policy_stream import LiteralLearner, LiteralState


@pytest.fixture(scope="module")
def completed():
    return study.run_example(context_return=True)


def test_should_execute_fixed_same_policy_readouts_and_original_delayed_training(completed):
    r = completed
    assert r["complete"], r.get("error")
    assert r["source_unchanged"] and r["actor_unchanged"] and r["numpy_rng_unchanged"]
    assert r["training_work"]["attempted_updates"] == r["training_work"]["returned_updates"] == 12
    assert r["training_work"]["attempted_rows"] == 72
    assert [p["source_rows"] for p in r["checkpoints"]] == [0, 24, 48, 72]
    assert [p["policy_version"] for p in r["training_data"]] == ["v1"] * 4 + ["v2"] * 4 + ["v1"] * 4
    assert len({i for b in r["training_data"] for i in b["row_ids"]}) == 72
    assert all(b["role"] == "train" for b in r["training_data"])
    assert len(r["applied_updates"]) == 12
    assert all(
        x["arrived_at"] == x["observed_at"] + 1 <= x["applied_at"] for x in r["applied_updates"]
    )
    assert r["budget"]["updates_completed"] == r["budget"]["limits"]["max_training_updates"] == 12
    assert len(r["episodes"]) == 24
    for p in study.CHECKPOINTS:
        rows = [x for x in r["episodes"] if x["source_updates"] == p]
        assert [(x["policy_version"], x["concept"]) for x in rows[:4]] == list(study.CONTEXT_TASKS)
    for row in r["episodes"]:
        e = row["episode"]
        assert e["complete"] and e["policy_context_in_query"] and e["training_updates"] == 0
        assert row["policy_version"] == e["initial_state"]["policy_version"]
    assert r["summary"] == study.summarize_context_return(r["episodes"])
    assert r["costs"]["attempted_training_affine_terms"] == 72 * 10 * 6
    assert r["costs"]["retained_checkpoint_coefficient_bytes"] == 4 * 10 * 6 * 8
    assert all(
        r[k] == 0
        for k in ("probe_updates", "final_examples", "replay_examples", "external_actions")
    )
    assert len(teacher._encode_report(r)) <= 512 * 1024


def test_should_restore_exact_a_truth_and_owned_contextual_packets():
    spec, batches = study.prepare_training(context_return=True)
    assert spec.vocabulary == teacher.CONTEXT_FEATURES
    assert len(batches) == 12 and all(b.features.shape == (6, 9) for b in batches)
    assert np.array_equal(batches[0].features, batches[8].features)
    assert np.array_equal(batches[0].targets, batches[8].targets)
    assert batches[0].targets[1, 0] != batches[4].targets[1, 0]
    before = batches[1].features.copy()
    batches[0].features[0, 0] += 10
    assert np.array_equal(before, batches[1].features)
    assert not np.shares_memory(batches[0].targets, batches[8].targets)


@pytest.mark.parametrize("role", ["probe_train", "final_test", "outer_selection"])
def test_should_refuse_nontrain_before_projecting_or_learning(role, monkeypatch):
    original = teacher.training_cases
    monkeypatch.setattr(
        teacher, "training_cases", lambda **kw: tuple(replace(c, role=role) for c in original(**kw))
    )
    with pytest.raises(ValueError):
        build_context_return_stream(
            study.DecisionFeatureSpec(teacher.CONTEXT_FEATURES, study.CATALOG)
        )


@pytest.mark.parametrize(
    "kw",
    [
        dict(context_return=1),
        dict(context_return=True, policy_change=True),
        dict(context_return=True, method="unknown"),
        dict(context_return=True, replay="unknown"),
    ],
)
def test_should_refuse_unsupported_modes_before_work(kw, monkeypatch):
    monkeypatch.setattr(study, "prepare_training", lambda **kw: pytest.fail("data entered"))
    with pytest.raises(ValueError):
        study.run_example(**kw)


@pytest.fixture(scope="module")
def packet_template():
    return study.prepare_training(context_return=True)


@pytest.fixture
def scripted(monkeypatch, packet_template):
    monkeypatch.setattr(study, "PerceptronLearner", LiteralLearner)
    monkeypatch.setattr(study, "prepare_training", lambda **kw: deepcopy(packet_template))

    @dataclass
    class Predictor:
        state: LiteralState

        def __init__(self, spec, state):
            self.state = deepcopy(state)

        def __call__(self, request):
            words = request.query.split()
            selected = None
            if words[0] == "ready":
                selected = "inspect" if words[1] == "password" or "policy_v2" in words else "route"
                if words[1] == "parcel":
                    selected = "defer"
            return decision(request, selected)

    monkeypatch.setattr(study, "PerceptronDecisionPredictor", Predictor)
    return Predictor


def test_should_not_call_retention_recovery_when_no_acquired_skill_was_lost(scripted):
    r = study.run_example(context_return=True)
    assert r["complete"], r.get("error")
    assert all(v is None for v in r["summary"]["acquired_a_recovered_on_return"].values())
    assert all(r["summary"]["b_ability_before_training"].values())
    assert all(v == 0 for v in r["summary"]["b_acquisition_change"].values())


def test_should_report_true_conditional_loss_and_recovery_without_changing_task_truth():
    rows = [
        dict(
            source_updates=p,
            policy_version=v,
            concept=c,
            episode=dict(scenario="ordinary", success=(p >= 4 and (p != 8 or v == "v2"))),
        )
        for p in study.CHECKPOINTS
        for v, c in study.CONTEXT_TASKS
    ]
    s = study.summarize_context_return(rows)
    assert all(s["acquired_a_lost_during_b"].values())
    assert all(s["acquired_a_recovered_on_return"].values())
    assert s["a_success_change_during_b"]["v1:invoice"] == -1
    assert s["a_return_change"]["v1:invoice"] == 1


@pytest.mark.parametrize("fault", ["paid", "replace_paid", "input", "predictor", "clock"])
def test_should_stop_on_corrupted_evidence_before_more_work(fault, scripted, monkeypatch):
    original_episode, original_call = study.run_workflow_episode, scripted.__call__
    saved = []
    reports = []
    owners = []
    inputs = []
    original_budget = study.ToyBudgetSession

    def record_dict(*a, **kw):
        result = dict(*a, **kw)
        if "definition" in kw:
            reports.append(result)
        return result

    monkeypatch.setattr(study, "dict", record_dict, raising=False)

    def budget(*a, **kw):
        owner = original_budget(*a, **kw)
        owners.append(owner)
        return owner

    def episode(*a, **kw):
        result = original_episode(*a, **kw)
        saved.append(result)
        return result

    calls = 0

    def predict(self, request):
        nonlocal calls
        calls += 1
        result = original_call(self, request)
        if len(saved) == 1:
            if fault in ("paid", "replace_paid"):
                if fault == "replace_paid":
                    reports[0]["episodes"][0] = deepcopy(reports[0]["episodes"][0])
                else:
                    saved[0]["success"] = False
            elif fault == "predictor":
                self.state = replace(self.state, completed_updates=999)
            elif fault == "clock":
                owners[0].started_at += 1
        return result

    if fault == "input":
        prepare = study.prepare_training

        def changed(**kw):
            spec, batches = prepare(**kw)
            inputs.append(batches)
            return spec, batches

        monkeypatch.setattr(study, "prepare_training", changed)

        def episode(*a, **kw):
            result = original_episode(*a, **kw)
            inputs[0][0].targets[0, 0] = 99
            return result

    else:
        monkeypatch.setattr(study, "ToyBudgetSession", budget)
    monkeypatch.setattr(study, "run_workflow_episode", episode)
    monkeypatch.setattr(scripted, "__call__", predict)
    r = study.run_example(context_return=True)
    assert not r["complete"]
    errors = r["error"] + " ".join(x["episode"].get("error", "") for x in r["episodes"])
    assert "context" in errors
    assert r["training_work"]["attempted_updates"] == 0
    if fault in ("paid", "replace_paid"):
        assert r["episodes"][0]["episode"]["success"] is True
        assert len(r["episodes"][0]["episode"]["steps"]) == 3
    assert calls <= 4


def test_should_preserve_failed_prediction_attempt_and_stop(scripted, monkeypatch):
    monkeypatch.setattr(
        scripted, "__call__", lambda *a: (_ for _ in ()).throw(RuntimeError("lost return"))
    )
    r = study.run_example(context_return=True)
    assert not r["complete"] and r["training_work"]["attempted_updates"] == 0
    assert len(r["episodes"]) == 1 and not r["episodes"][0]["episode"]["complete"]
    assert r["episode_work"]["prediction_calls_attempted"] == 1
    assert r["episode_work"]["prediction_calls_returned"] == 0


def test_should_keep_cli_output_exclusive_and_refuse_bad_modes_before_reserving(
    tmp_path, monkeypatch
):
    p = tmp_path / "case.json"
    with pytest.raises(SystemExit):
        study.main(["--context-return", "--method", "bp", "--replay", "recent", "--output", str(p)])
    assert not p.exists()
    seen = []

    def run(**kw):
        seen.append(kw)
        return dict(complete=True)

    monkeypatch.setattr(study, "run_example", run)
    assert study.main(["--context-return", "--output", str(p)]) == 0
    assert seen == [dict(context_return=True)] and json.loads(p.read_bytes())["complete"]
    with pytest.raises(FileExistsError):
        study.main(["--context-return", "--output", str(p)])
