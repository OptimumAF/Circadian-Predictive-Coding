"""Versioned sandbox truth, isolated TRAIN packets and original stream integration."""

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts import recurring_ticket_workflow_example as study
from scripts import run_ticket_workflow as teacher
from scripts.ticket_controller_streams import POLICY_CHANGE_PHASES, build_policy_change_stream
from src.core.learner_ports import TrainingDiagnostic
from src.core.temporal_decision_sequence import VOCABULARY
from src.core.ticket_workflow import (
    ACTION_IDS,
    CATALOG,
    WorkflowCapabilities,
    WorkflowState,
    dispatch_workflow,
    workflow_success,
)
from test_ticket_workflow import decision


def test_default_v2_teacher_is_exact_saved_accepted_bytes_and_explicit_default():
    prior = json.loads(
        Path("artifacts/runs/local-ticket-workflow-20261022/actual.json").read_bytes()
    )
    actual = json.loads(json.dumps([asdict(c) for c in teacher.training_cases()]))
    assert actual == prior["training_cases"]
    assert teacher.training_cases() == teacher.training_cases(policy_version="v2")


@pytest.mark.parametrize(
    "version,rules",
    [
        ("v1", {"invoice": "route", "legacy": "retire", "parcel": "route", "password": "inspect"}),
        ("v2", {"invoice": "inspect", "legacy": "defer", "parcel": "defer", "password": "inspect"}),
    ],
)
def test_teacher_labels_complete_actual_sandbox_transitions_with_correct_active_tools(
    version, rules
):
    cases = teacher.training_cases(policy_version=version)
    caps = WorkflowCapabilities(ACTION_IDS)
    for index, concept in enumerate(VOCABULARY):
        state = WorkflowState("literal-truth", concept, policy_version=version)
        for row, action in zip(
            cases[index * 3 : index * 3 + 3], ("open", rules[concept], "commit"), strict=True
        ):
            assert row.role == "train" and row.target_id == action
            assert row.target_id in [c.candidate_id for c in row.request.allowed_candidates]
            state = dispatch_workflow(state, row.target_id, state.revision, caps).state
        assert workflow_success(state)


@pytest.mark.parametrize("version", [None, True, 1, "v3", "V1", "final_test", "probe_train"])
def test_invalid_teacher_version_refuses_before_request_or_payload(version, monkeypatch):
    monkeypatch.setattr(teacher, "workflow_request", lambda *a: pytest.fail("request built"))
    with pytest.raises(ValueError):
        teacher.training_cases(policy_version=version)


def test_wrong_but_permitted_teacher_target_is_not_silently_declared_truth(monkeypatch):
    original = teacher.policy_for_phase

    def wrong(phase):
        p = original(phase)
        p["rules"]["invoice"] = "route"
        return p

    monkeypatch.setattr(teacher, "policy_for_phase", wrong)
    with pytest.raises(ValueError, match="did not solve"):
        teacher.training_cases()


def test_fixed_recipe_keeps_owned_temporal_labels_and_unchanged_password():
    spec = study.DecisionFeatureSpec(study.FEATURES, CATALOG)
    batches = build_policy_change_stream(spec)
    assert len(batches) == 12 and sum(len(b.features) for b in batches) == 72
    assert len({row for b in batches for row in b.row_ids}) == 72
    for phase, (_, version, concepts) in enumerate(POLICY_CHANGE_PHASES):
        first = batches[phase * 4]
        assert first.role == "train" and first.num_classes == 6 and first.episode is None
        assert all(
            b.context == POLICY_CHANGE_PHASES[phase][0] for b in batches[phase * 4 : phase * 4 + 4]
        )
        actions = [ACTION_IDS[int(i)] for i in first.targets[:, 0]]
        assert actions[4] == "inspect" and concepts[1] == "password"
        assert actions[1] == ("route" if version == "v1" else "defer" if phase == 1 else "inspect")
    assert np.array_equal(batches[0].features, batches[8].features)
    assert batches[0].targets[1, 0] != batches[8].targets[1, 0]
    before = batches[1].targets.copy()
    batches[0].targets[1, 0] = 99
    assert np.array_equal(before, batches[1].targets)
    assert not np.shares_memory(batches[0].features, batches[1].features)


@pytest.mark.parametrize("role", ["probe_train", "outer_selection", "inner_guard", "final_test"])
def test_builder_refuses_nontrain_teacher_before_projection(role, monkeypatch):
    cases = teacher.training_cases()
    monkeypatch.setattr(
        teacher, "training_cases", lambda **kw: tuple(replace(c, role=role) for c in cases)
    )
    with pytest.raises(ValueError):
        build_policy_change_stream(study.DecisionFeatureSpec(study.FEATURES, CATALOG))


@pytest.mark.parametrize("bad", [None, object(), 0])
def test_bad_schema_refuses_before_teacher_or_model_read(bad, monkeypatch):
    monkeypatch.setattr(teacher, "training_cases", lambda **kw: pytest.fail("teacher read"))
    with pytest.raises(ValueError):
        build_policy_change_stream(bad)


@dataclass(frozen=True)
class LiteralState:
    completed_updates: int = 0
    training_examples: int = 0
    mistake_updates: int = 0


class LiteralLearner:
    def __init__(self, policy):
        self.policy = policy
        self.state = LiteralState()

    def snapshot_state(self):
        return deepcopy(self.state)

    def restore_state(self, state):
        self.state = deepcopy(state)

    def fork(self):
        child = LiteralLearner(self.policy)
        child.restore_state(self.state)
        return child

    def train_batch(self, features, targets):
        self.state = LiteralState(
            self.state.completed_updates + 1, self.state.training_examples + len(features)
        )
        return TrainingDiagnostic("scripted_train", 0.0)

    def predict(self, features):
        return np.tile(np.ones(6) / 6, (len(features), 1))


class LiteralPredictor:
    def __init__(self, spec, state):
        self.state = deepcopy(state)

    def __call__(self, request):
        words = request.query.split()
        if words[0] == "ready" and "retire" in {c.candidate_id for c in request.allowed_candidates}:
            selected = "inspect" if words[1] == "password" else "route"
            return decision(request, selected)
        return decision(request)


@pytest.fixture
def literal(monkeypatch):
    monkeypatch.setattr(study, "PerceptronLearner", LiteralLearner)
    monkeypatch.setattr(study, "PerceptronDecisionPredictor", LiteralPredictor)
    return study


def test_scripted_combined_runner_uses_original_stream_clock_budget_receipts_and_policy_readouts(
    literal,
):
    r = literal.run_example(policy_change=True)
    assert r["complete"], r.get("error")
    assert r["source_unchanged"] and r["actor_unchanged"] and r["numpy_rng_unchanged"]
    assert r["budget"]["limits"]["max_training_updates"] == r["budget"]["updates_completed"] == 12
    assert len(r["applied_updates"]) == 12 and r["training_work"]["attempted_rows"] == 72
    assert [p["source_rows"] for p in r["checkpoints"]] == [0, 24, 48, 72]
    assert [row["policy_version"] for row in r["training_data"]] == ["v1"] * 4 + ["v2"] * 8
    assert all(
        row["arrived_at"] == row["observed_at"] + 1 <= row["applied_at"]
        for row in r["applied_updates"]
    )
    assert [
        row["episode"]["initial_state"]["policy_version"]
        for row in r["episodes"]
        if row["concept"] == "password"
    ] == ["v1", "v1", "v2", "v2"]
    assert all(
        row["source_group"] == "development-policy-workflow-traces"
        and row["role"] == "outer_selection"
        and row["episode"]["training_updates"] == 0
        for row in r["episodes"]
    )
    assert "old_task_success_forgetting" not in r["summary"]
    assert all(
        r[k] == 0
        for k in ("final_examples", "probe_updates", "replay_examples", "external_actions")
    )


def test_failed_fifth_scripted_update_preserves_original_spent_prefix_and_no_future_readout(
    literal, monkeypatch
):
    original = LiteralLearner.train_batch
    calls = []

    def fail(self, x, y):
        calls.append(1)
        if len(calls) == 5:
            raise RuntimeError("lost policy-update return")
        return original(self, x, y)

    monkeypatch.setattr(LiteralLearner, "train_batch", fail)
    r = literal.run_example(policy_change=True)
    assert not r["complete"] and "lost policy-update return" in r["error"]
    assert (
        r["training_work"]["attempted_updates"] == 5 and r["training_work"]["returned_updates"] == 4
    )
    assert r["training_work"]["unreturned_work_unknown"] and r["budget"]["updates_completed"] == 4
    assert (
        len(r["applied_updates"]) == 4 and max(row["source_updates"] for row in r["episodes"]) == 4
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"policy_change": 1},
        {"policy_change": True, "replay": "recent"},
        {"policy_change": True, "replay": "coverage"},
    ],
)
def test_invalid_recipe_mode_refuses_before_models_or_teachers(kwargs, monkeypatch):
    monkeypatch.setattr(study, "prepare_training", lambda **kw: pytest.fail("teacher read"))
    with pytest.raises(ValueError):
        study.run_example(**kwargs)


def test_explicit_cli_preserves_default_dispatch_and_occupied_output(tmp_path, monkeypatch):
    calls = []

    def run(**kw):
        calls.append(kw)
        return dict(complete=True)

    monkeypatch.setattr(study, "run_example", run)
    assert study.main(["--output", str(tmp_path / "default.json")]) == 0 and calls == [{}]
    target = tmp_path / "policy.json"
    assert study.main(["--output", str(target), "--policy-change"]) == 0
    assert calls[-1] == dict(replay="none", method="perceptron", policy_change=True)
    before = len(calls)
    with pytest.raises(FileExistsError):
        study.main(["--output", str(target), "--policy-change"])
    assert len(calls) == before
    with pytest.raises(SystemExit):
        study.main(
            ["--output", str(tmp_path / "bad.json"), "--policy-change", "--replay", "recent"]
        )
    assert not (tmp_path / "bad.json").exists()


def test_sole_fixed_perceptron_policy_case():
    output = os.environ.get("VERSIONED_POLICY_CASE_REPORT")
    if not output:
        pytest.skip("separate frozen scalar case after software gates")
    r = study.run_example(policy_change=True)
    Path(output).write_text(json.dumps(r, indent=2, allow_nan=False) + "\n", encoding="utf8")
    assert r["complete"], r.get("error")
    assert r["source_unchanged"] and r["actor_unchanged"] and r["numpy_rng_unchanged"]
    assert r["training_work"] == dict(
        attempted_updates=12,
        attempted_rows=72,
        returned_updates=12,
        unreturned_work_unknown=False,
        budget_committed_updates=12,
    )
    assert len(r["applied_updates"]) == 12 and [p["source_rows"] for p in r["checkpoints"]] == [
        0,
        24,
        48,
        72,
    ]
    assert (
        r["final_examples"]
        == r["probe_updates"]
        == r["replay_examples"]
        == r["external_actions"]
        == 0
    )
    assert len(r["episodes"]) == 15 and all(row["episode"]["complete"] for row in r["episodes"])
