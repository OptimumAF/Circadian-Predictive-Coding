"""Scripted integration of the real generic probe and actual ticket sandbox."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.probe_ticket_policy_learning as recipe
from src.adapters.torch_decision import TorchDecisionWork
from src.core.learner_ports import TrainingDiagnostic
from src.core.typed_decision import CandidateScore, DecisionResult

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json"
)


class FakeLearner:
    def __init__(self, before=lambda x: None, after=lambda: None):
        self.state = SimpleNamespace(completed_updates=0, training_examples=0)
        self.before, self.after = before, after

    def fork(self):
        child = FakeLearner(self.before, self.after)
        child.restore_state(self.state)
        return child

    def restore_state(self, state):
        self.state = deepcopy(state)

    def snapshot_state(self):
        return deepcopy(self.state)

    def train_batch(self, features, targets):
        self.before(features)
        self.state.completed_updates += 1
        self.state.training_examples += len(features)
        self.after()
        return TrainingDiagnostic("scripted_no_native_learning", 1.0)

    def predict(self, features):
        return np.full((len(features), 6), 1.0 / 6)


class FakeReader:
    def __init__(self, *args):
        self.work = TorchDecisionWork()

    def assert_unchanged(self):
        pass

    def __call__(self, request):
        candidates = sorted(request.allowed_candidates, key=lambda c: c.candidate_id)
        self.work = TorchDecisionWork(*(self.work.decision_attempts + 1 for _ in range(4)))
        return DecisionResult(
            request.request_id,
            candidates[0].candidate_id,
            tuple(CandidateScore(c.candidate_id, 0.0, 1.0 / len(candidates)) for c in candidates),
            0,
            len(candidates),
        )


class FakeRng(np.ndarray):
    def clone(self):
        return self.copy()


@pytest.fixture
def scripted(monkeypatch):
    monkeypatch.setattr(recipe.native, "same_state", lambda a, b: a == b)
    monkeypatch.setattr(recipe.native, "readout", FakeReader)
    monkeypatch.setattr(
        recipe.native,
        "require_torch",
        lambda: SimpleNamespace(
            get_rng_state=lambda: np.array([17]).view(FakeRng), equal=np.array_equal
        ),
    )
    monkeypatch.setattr(
        recipe,
        "bp_snapshot_from_record",
        lambda r: SimpleNamespace(
            completed_updates=r["completed_updates"], training_examples=r["training_examples"]
        ),
    )
    created = []

    def construct(method, d, k, before, after):
        assert (method, d, k) == ("bp", 7, 6)
        learner = FakeLearner(before, after)
        created.append(learner)
        return learner

    monkeypatch.setattr(recipe.native, "counted_learner", construct)
    return created


@pytest.fixture(scope="module")
def spec():
    return recipe.DecisionFeatureSpec(recipe.FEATURES, recipe.CATALOG)


def test_should_build_detached_roles_and_identical_return_templates(spec):
    training, evaluation, monitors, returning = recipe.build_cohorts(spec)
    assert len(training) == 16 and all(b.role == "probe_train" for b in training)
    assert [b.context for b in training] == ["new-invoice", "new-password"] * 8
    assert all(b.role == "probe_test" for b in evaluation)
    assert all(b.role == "train" for b in (*monitors, *returning))
    all_batches = (*training, *evaluation, *monitors, *returning)
    assert len({r for b in all_batches for r in b.row_ids}) == 66
    arrays = [a for b in all_batches for a in (b.features, b.targets)]
    assert sum(a.nbytes for a in arrays) == 4224
    assert all(not np.shares_memory(a, b) for i, a in enumerate(arrays) for b in arrays[i + 1 :])
    for monitor, template in zip(monitors, returning, strict=True):
        assert np.array_equal(monitor.features, template.features)
        assert np.array_equal(monitor.targets, template.targets)


@pytest.fixture(scope="module")
def saved():
    return json.loads(SOURCE.read_bytes())


@pytest.mark.parametrize("fault", ["points", "method", "seed", "bool", "rows"])
def test_should_refuse_replaced_source_metadata_before_native(fault, saved):
    s = deepcopy(saved)
    if fault == "points":
        s["checkpoints"].reverse()
    elif fault == "method":
        s["checkpoints"][0]["state"]["policy"]["method"] = "pc"
    elif fault == "seed":
        s["checkpoints"][0]["state"]["policy"]["seed"] = 19
    elif fault == "bool":
        s["checkpoints"][0]["state"]["completed_updates"] = False
    else:
        s["checkpoints"][1]["state"]["training_examples"] -= 1
    with pytest.raises(ValueError):
        recipe._require_source_records(s)


def test_should_execute_real_probe_return_and_tasks_without_changing_sources(scripted):
    before = SOURCE.read_bytes()
    result = recipe.run_case(SOURCE)
    assert result["complete"], result.get("error")
    assert [r["mode"] for r in result["cases"]] == ["warm", "fresh"]
    assert [r["prior_native_updates"] for r in result["cases"]] == [96, 0]
    assert result["training_work"] == dict(
        attempted_updates=64, attempted_rows=192, returned_updates=64, unreturned_work_unknown=False
    )
    assert [s.state.completed_updates for s in scripted] == [96, 0]
    assert SOURCE.read_bytes() == before
    for row in result["cases"]:
        recipe._require_arm(row)
        assert len(row["episodes"]) == 8 and len(row["readout_work"]) == 2
        assert [s["updates"] for s in row["sessions"]] == [16, 16]
        # Scripted uniform predictions deliberately fail actual workflows.
        assert all(not e["episode"]["success"] for e in row["episodes"])


def test_should_keep_paid_failed_probe_and_close_later_arm(scripted, monkeypatch):
    original = FakeLearner.train_batch
    calls = []

    def failing(self, features, targets):
        calls.append(1)
        if len(calls) == 3:
            self.before(features)
            raise RuntimeError("scripted unknown native work")
        return original(self, features, targets)

    monkeypatch.setattr(FakeLearner, "train_batch", failing)
    result = recipe.run_case(SOURCE)
    assert not result["complete"] and len(result["cases"]) == len(scripted) == 1
    row = result["cases"][0]
    assert row["probe"]["completed_updates"] == 2
    assert row["probe"]["return_phase"]["status"] == "not_run"
    assert result["training_work"]["attempted_updates"] == 3
    assert result["training_work"]["unreturned_work_unknown"] is True


def test_should_preserve_endpoint_failure_and_prevent_return(scripted, monkeypatch):
    def failing(self, request):
        raise RuntimeError("scripted task reader failure")

    monkeypatch.setattr(FakeReader, "__call__", failing)
    result = recipe.run_case(SOURCE)
    assert not result["complete"] and len(scripted) == 1
    row = result["cases"][0]
    assert row["probe"]["completed_updates"] == 16
    assert row["probe"]["return_phase"]["status"] == "not_run"
    assert row["readout_work"][0]["work"] is not None


def test_should_refuse_nontrain_templates_before_fork(scripted, monkeypatch):
    original = recipe.build_cohorts

    def corrupted(spec):
        training, *rest = original(spec)
        return (replace(training[0], role="final_test"), *training[1:]), *rest

    monkeypatch.setattr(recipe, "build_cohorts", corrupted)
    result = recipe.run_case(SOURCE)
    assert not result["complete"] and result["training_work"]["attempted_updates"] == 0
    assert "probe_train" in result["error"]


def test_should_refuse_replaced_source_before_teacher_or_native(tmp_path, monkeypatch):
    p = tmp_path / "replaced.json"
    p.write_bytes(SOURCE.read_bytes() + b" ")
    monkeypatch.setattr(
        recipe, "build_cohorts", lambda *a: pytest.fail("replaced source reached teacher")
    )
    result = recipe.run_case(p)
    assert not result["complete"] and "replaced" in result["error"]


def test_should_protect_existing_output_and_input_before_work(tmp_path, monkeypatch):
    p = tmp_path / "existing.json"
    p.write_text("preserve")
    monkeypatch.setattr(
        recipe, "run_case", lambda *a: pytest.fail("occupied output reached native")
    )
    with pytest.raises(FileExistsError):
        recipe.main(["--input", str(SOURCE), "--output", str(p)])
    assert p.read_text() == "preserve"
    with pytest.raises(ValueError, match="sealed"):
        recipe.main(["--input", str(SOURCE), "--output", str(SOURCE)])


def test_should_refuse_corrupted_paid_evidence_before_second_arm(scripted, monkeypatch):
    original = recipe.run_arm
    saved_row = []

    def capture(*args, **kwargs):
        row = original(*args, **kwargs)
        saved_row.append(row)
        return row

    original_construct = recipe.native.counted_learner

    def corrupt(*args):
        if saved_row:
            saved_row[0]["probe"]["diagnostics"][0]["value"] += 1.0
        return original_construct(*args)

    monkeypatch.setattr(recipe, "run_arm", capture)
    monkeypatch.setattr(recipe.native, "counted_learner", corrupt)
    result = recipe.run_case(SOURCE)
    assert not result["complete"] and "paid probe evidence changed" in result["error"]
    assert result["training_work"]["attempted_updates"] == 32
    assert result["cases"][0]["probe"]["diagnostics"][0]["value"] == 1.0
