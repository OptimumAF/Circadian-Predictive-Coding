"""Existing generic probe on private contextual heads; no native BP experiment."""

from copy import deepcopy
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import probe_ticket_policy_learning as recipe
from scripts import context_head_probe as head
from src.core.context_perceptron import create_context_snapshot, copy_context_snapshot
from src.core.learner_ports import TrainingDiagnostic
from src.core.typed_decision import CandidateScore, DecisionResult

SOURCE = Path("artifacts/runs/context-head-control-20261023/actual.json")


@pytest.fixture(scope="module")
def saved():
    return json.loads(SOURCE.read_bytes())


@pytest.fixture(scope="module")
def spec():
    return head.DecisionFeatureSpec(head.CONTEXT_FEATURES, head.CATALOG)


def test_should_build_detached_train_derived_roles_and_homogeneous_contexts(spec):
    training, evaluation, monitors, returning = head.build_head_cohorts(spec)
    assert len(training) == 16 and all(
        b.role == "probe_train" and len(b.features) == 3 for b in training
    )
    assert [b.context for b in training] == ["legacy-v1", "legacy-v2"] * 8
    assert all(b.role == "probe_test" for b in evaluation)
    assert all(b.role == "train" and len(b.features) == 6 for b in (*monitors, *returning))
    batches = (*training, *evaluation, *monitors, *returning)
    assert len({x for b in batches for x in b.row_ids}) == 78
    arrays = [x for b in batches for x in (b.features, b.targets)]
    assert sum(a.nbytes for a in arrays) == 6240
    assert all(not np.shares_memory(a, b) for i, a in enumerate(arrays) for b in arrays[i + 1 :])
    for i, b in enumerate(training):
        assert (b.features[:, (4, 5)] == ((1, 0) if i % 2 == 0 else (0, 1))).all()
    for b, old in zip(returning, monitors, strict=True):
        assert (
            b.context == old.context
            and np.array_equal(b.features, old.features)
            and np.array_equal(b.targets, old.targets)
        )


@pytest.mark.parametrize("fault", ["points", "bool", "rows", "policy", "head", "extra"])
def test_should_refuse_wrong_qualified_source_metadata(saved, fault):
    s = deepcopy(saved)
    if fault == "points":
        s["checkpoints"].reverse()
    elif fault == "bool":
        s["checkpoints"][0]["source_updates"] = False
    elif fault == "rows":
        s["checkpoints"][-1]["source_rows"] -= 1
    elif fault == "policy":
        s["checkpoints"][-1]["state"]["policy"]["context_columns"] = [0, 1]
    elif fault == "head":
        s["checkpoints"][-1]["state"]["heads"][0]["completed_updates"] += 1
    else:
        s["checkpoints"][-1]["state"]["heads"][0]["extra"] = 1
    with pytest.raises(ValueError):
        head.head_states(s)


def test_should_refuse_replaced_source_before_teacher_or_learner(tmp_path, monkeypatch):
    p = tmp_path / "replaced.json"
    p.write_bytes(SOURCE.read_bytes() + b" ")
    monkeypatch.setattr(head, "build_head_cohorts", lambda *a: pytest.fail("teacher entered"))
    monkeypatch.setattr(head, "ContextPerceptronLearner", lambda *a: pytest.fail("learner entered"))
    r = recipe.run_case(p, separate_context_heads=True)
    assert not r["complete"] and "replaced" in r["error"]
    assert r["training_work"]["attempted_updates"] == 0


@pytest.mark.parametrize("role", ["final_test", "probe_train", "outer_selection"])
def test_should_refuse_nontrain_teacher_before_projection(spec, role, monkeypatch):
    original = head.training_cases
    monkeypatch.setattr(
        head, "training_cases", lambda **kw: tuple(replace(c, role=role) for c in original(**kw))
    )
    with pytest.raises(ValueError):
        head.build_head_cohorts(spec)


@pytest.fixture(scope="module")
def completed():
    return recipe.run_case(SOURCE, separate_context_heads=True)


def test_should_run_original_fixed_probe_return_and_preserve_all_source_owners(completed):
    r = completed
    assert r["complete"], r.get("error")
    assert [x["mode"] for x in r["cases"]] == ["warm", "fresh"]
    assert [x["prior_native_updates"] for x in r["cases"]] == [12, 0]
    assert r["training_work"] == dict(
        attempted_updates=64, attempted_rows=288, returned_updates=64, unreturned_work_unknown=False
    )
    assert r["source_owners_unchanged"] and r["source_bytes_unchanged"] and r["numpy_rng_unchanged"]
    assert r["source_prior_training"] == dict(updates=12, rows=72, mistakes=16)
    assert r["model_coefficients"] == 96 and r["model_coefficient_bytes"] == 768
    assert r["input_array_bytes"] == 6240
    for arm in r["cases"]:
        recipe._require_arm(arm, separate_context_heads=True)
        assert [s["updates"] for s in arm["sessions"]] == [16, 16]
        assert all(s["limits"] == recipe.asdict(recipe.HEAD_LIMITS) for s in arm["sessions"])
        assert [p["updates"] for p in arm["probe"]["checkpoints"]] == [0, 4, 8, 16]
        returning = arm["probe"]["return_phase"]
        assert [p["updates"] for p in returning["checkpoints"]] == [0, 4, 8, 16]
        assert (
            arm["probe"]["completed_training_examples"] == 48
            and returning["completed_training_examples"] == 96
        )
        assert len(arm["readout_work"]) == 2 and len(arm["episodes"]) == 12
        prior = arm["prior_native_updates"], arm["prior_native_rows"]
        assert [(w["native_updates"], w["native_rows"]) for w in arm["readout_work"]] == [
            (prior[0] + 16, prior[1] + 48),
            (prior[0] + 32, prior[1] + 144),
        ]
        for w in arm["readout_work"]:
            assert (
                w["work"]["attempted_calls"]
                == w["work"]["returned_calls"]
                == w["work"]["validated_calls"]
            )
            state = w["state"]
            assert sum(h["training_examples"] for h in state["heads"]) == state["training_examples"]
        assert all(
            x["episode"]["training_updates"] == 0 and x["episode"]["policy_context_in_query"]
            for x in arm["episodes"]
        )
    assert r["final_examples"] == r["replay_examples"] == r["promotions"] == 0
    assert len(recipe._encode_report(r)) <= 512 * 1024


@dataclass
class FakeReader:
    spec: Any
    state: Any

    def __call__(self, request):
        allowed = sorted(request.allowed_candidates, key=lambda c: c.candidate_id)
        return DecisionResult(
            request.request_id,
            allowed[0].candidate_id,
            tuple(CandidateScore(c.candidate_id, 0.0, 1 / len(allowed)) for c in allowed),
            0,
            len(allowed),
        )


class FakeContext:
    """Counter-only owned port, no perceptron optimization or successful tasks."""

    def __init__(self, policy):
        self.policy = policy
        self.state = create_context_snapshot(policy)

    def snapshot_state(self):
        return copy_context_snapshot(self.state)

    def restore_state(self, state):
        self.state = copy_context_snapshot(state)

    @property
    def completed_updates(self):
        return self.state.completed_updates

    @property
    def training_examples(self):
        return self.state.training_examples

    def train_batch(self, features, targets):
        index = int(features[0, 5])
        heads = list(self.state.heads)
        h = heads[index]
        heads[index] = replace(
            h,
            completed_updates=h.completed_updates + 1,
            training_examples=h.training_examples + len(features),
        )
        self.state = replace(
            self.state,
            heads=(heads[0], heads[1]),
            completed_updates=self.state.completed_updates + 1,
            training_examples=self.state.training_examples + len(features),
        )
        return TrainingDiagnostic("scripted_no_head_optimization", 0.0)

    def predict(self, features):
        return np.full((len(features), 6), 1 / 6)


@pytest.fixture
def scripted(monkeypatch):
    monkeypatch.setattr(head, "ContextPerceptronLearner", FakeContext)
    monkeypatch.setattr(recipe, "ContextPerceptronDecisionPredictor", FakeReader)


def test_should_preserve_failed_update_and_original_budget_prefix(scripted, monkeypatch):
    original = FakeContext.train_batch
    calls = 0

    def fail(self, x, y):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("lost head update return")
        return original(self, x, y)

    monkeypatch.setattr(FakeContext, "train_batch", fail)
    r = recipe.run_case(SOURCE, separate_context_heads=True)
    assert not r["complete"] and len(r["cases"]) == 1
    assert (
        r["training_work"]["attempted_updates"] == 3 and r["training_work"]["returned_updates"] == 2
    )
    assert r["training_work"]["unreturned_work_unknown"]
    assert [s["updates"] for s in r["cases"][0]["sessions"]] == [2, 0]
    assert r["cases"][0]["probe"]["return_phase"]["status"] == "not_run"


def test_should_keep_endpoint_failure_and_prevent_return_or_next_arm(scripted, monkeypatch):
    monkeypatch.setattr(
        FakeReader,
        "__call__",
        lambda *a: (_ for _ in ()).throw(RuntimeError("lost endpoint return")),
    )
    r = recipe.run_case(SOURCE, separate_context_heads=True)
    assert not r["complete"] and len(r["cases"]) == 1
    assert r["training_work"]["returned_updates"] == 16
    assert [s["updates"] for s in r["cases"][0]["sessions"]] == [16, 0]
    assert r["cases"][0]["readout_work"][0]["work"]["attempted_calls"] == 1
    assert r["cases"][0]["readout_work"][0]["work"]["returned_calls"] == 0


def test_should_refuse_changed_original_probe_data_before_first_update(scripted, monkeypatch):
    saved = []
    prepare, construct = head.build_head_cohorts, FakeContext.__init__

    def build(spec):
        result = prepare(spec)
        saved.append(result)
        return result

    def corrupt(self, policy):
        construct(self, policy)
        saved[0][0][0].targets[0, 0] = 5.0

    monkeypatch.setattr(head, "build_head_cohorts", build)
    monkeypatch.setattr(FakeContext, "__init__", corrupt)
    r = recipe.run_case(SOURCE, separate_context_heads=True)
    assert not r["complete"] and "input data changed" in r["error"]
    assert r["training_work"]["attempted_updates"] == 0


@pytest.mark.parametrize("replace_equal", [False, True])
def test_should_refuse_corrupted_or_replaced_paid_arm_before_fresh_training(
    scripted, monkeypatch, replace_equal
):
    rows = []
    original_arm, original_init = recipe.run_arm, FakeContext.__init__

    def arm(*a, **kw):
        result = original_arm(*a, **kw)
        rows.append(result)
        return result

    def construct(self, policy):
        if rows:
            if replace_equal:
                rows[0]["probe"]["diagnostics"] = deepcopy(rows[0]["probe"]["diagnostics"])
            else:
                rows[0]["probe"]["diagnostics"][0]["value"] = 999.0
        original_init(self, policy)

    monkeypatch.setattr(recipe, "run_arm", arm)
    monkeypatch.setattr(FakeContext, "__init__", construct)
    r = recipe.run_case(SOURCE, separate_context_heads=True)
    assert not r["complete"] and "paid probe evidence" in r["error"]
    assert r["training_work"]["returned_updates"] == 32
    assert r["cases"][0]["probe"]["diagnostics"][0]["value"] == 0.0


def test_should_dispatch_explicit_cli_keep_default_and_protect_source_output(tmp_path, monkeypatch):
    seen = []

    def run(*a, **kw):
        seen.append(kw)
        return dict(complete=True)

    monkeypatch.setattr(recipe, "run_case", run)
    p = tmp_path / "head.json"
    assert (
        recipe.main(["--input", str(SOURCE), "--output", str(p), "--separate-context-heads"]) == 0
    )
    assert seen == [dict(separate_context_heads=True)]
    with pytest.raises(FileExistsError):
        recipe.main(["--input", str(SOURCE), "--output", str(p), "--separate-context-heads"])
    with pytest.raises(ValueError):
        recipe.main(["--input", str(SOURCE), "--output", str(SOURCE), "--separate-context-heads"])
    assert recipe.main(["--input", str(SOURCE), "--output", str(tmp_path / "native.json")]) == 0
    assert seen[-1] == {}


def test_should_refuse_ambiguous_api_mode_before_source_read():
    with pytest.raises(ValueError):
        recipe.run_case(Path("not-read"), separate_context_heads=1)
