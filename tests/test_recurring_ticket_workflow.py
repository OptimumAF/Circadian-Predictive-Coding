"""Chronological complete task behavior via actual existing conventional ports."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

import scripts.recurring_ticket_workflow_example as study
from src.adapters.perceptron_learner import PerceptronLearner
from src.adapters.perceptron_decision import PerceptronDecisionPredictor


@pytest.fixture(scope="module")
def completed():
    return study.run_example()


def test_should_interleave_fixed_training_phases_with_both_full_task_readouts(completed):
    r = completed
    assert r["complete"] is True, r.get("error")
    assert [row["source_updates"] for row in r["checkpoints"]] == [0, 4, 8, 12]
    assert [row["source_rows"] for row in r["checkpoints"]] == [0, 12, 24, 36]
    assert [row["context"] for row in r["training_data"]] == ["A"] * 4 + ["B"] * 4 + [
        "A_return"
    ] * 4
    assert all(row["role"] == "train" and row["rows"] == 3 for row in r["training_data"])
    assert len({i for row in r["training_data"] for i in row["row_ids"]}) == 36
    assert len(r["episodes"]) == 11 and all(row["episode"]["complete"] for row in r["episodes"])
    for point in study.CHECKPOINTS:
        slots = [
            (row["episode"]["scenario"], row["concept"])
            for row in r["episodes"]
            if row["source_updates"] == point
        ]
        assert tuple(slots) == study.episode_slots(point)
    assert r["summary"] == study.summarize(r["episodes"])


def test_should_keep_delayed_receipts_costs_and_no_final_or_probe_training(completed):
    r = completed
    assert r["training_work"] == dict(
        attempted_updates=12,
        attempted_rows=36,
        returned_updates=12,
        unreturned_work_unknown=False,
        budget_committed_updates=12,
    )
    stream = r["stream"]
    assert stream["completed_updates"] == 12 and stream["training_examples"] == 36
    assert stream["evaluation_scope"] == "training_only" and stream["score_forward_calls"] == 0
    assert len(stream["applied_updates"]) == 12
    assert all(row["applied_at"] > row["observed_at"] for row in stream["applied_updates"])
    assert (
        r["probe_updates"]
        == r["final_examples"]
        == r["replay_examples"]
        == r["external_actions"]
        == 0
    )
    assert r["numpy_rng_unchanged"] is True
    calls = sum(row["episode"]["work"]["prediction_calls_attempted"] for row in r["episodes"])
    assert calls == r["episode_work"]["prediction_calls_attempted"] <= 66
    assert r["costs"]["attempted_training_affine_terms"] == 36 * 8 * 6
    assert r["costs"]["retained_checkpoint_coefficient_bytes"] == 4 * 8 * 6 * 8
    assert all(row["episode"]["training_updates"] == 0 for row in r["episodes"])


def test_should_refuse_final_cases_before_any_fit(monkeypatch):
    cases = study.training_cases()
    monkeypatch.setattr(
        study, "training_cases", lambda: (replace(cases[0], role="final_test"), *cases[1:])
    )
    monkeypatch.setattr(
        PerceptronLearner, "train_batch", lambda *a: pytest.fail("forbidden final reached fit")
    )
    r = study.run_example()
    assert r["complete"] is False and "final and probe roles refused" in r["error"]
    assert r["training_work"]["attempted_updates"] == 0 and r["episodes"] == []


def test_should_stop_and_preserve_paid_prefix_on_failed_fifth_update(monkeypatch):
    original = PerceptronLearner.train_batch
    calls = 0

    def fail(self, features, targets):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("controlled update return lost")
        return original(self, features, targets)

    monkeypatch.setattr(PerceptronLearner, "train_batch", fail)
    r = study.run_example()
    assert r["complete"] is False and "return lost" in r["error"]
    assert calls == r["training_work"]["attempted_updates"] == 5
    assert (
        r["training_work"]["attempted_rows"] == 15 and r["training_work"]["returned_updates"] == 4
    )
    assert r["training_work"]["unreturned_work_unknown"] is True
    assert [row["source_updates"] for row in r["checkpoints"]] == [0, 4]
    assert len(r["episodes"]) == 4
    assert len(r["applied_updates"]) == 4 and r["budget"]["updates_completed"] == 4


def test_should_preserve_paid_prediction_failure_without_fitting(monkeypatch):
    def fail(*args):
        raise RuntimeError("prediction return lost")

    monkeypatch.setattr(PerceptronDecisionPredictor, "__call__", fail)
    r = study.run_example()
    assert r["complete"] is False and r["training_work"]["attempted_updates"] == 0
    assert len(r["episodes"]) == 1 and not r["episodes"][0]["episode"]["complete"]
    assert r["episode_work"]["prediction_calls_attempted"] == 1
    assert r["episode_work"]["prediction_calls_returned"] == 0


def test_should_stop_if_a_typed_readout_changes_the_training_candidate(monkeypatch):
    trained: list[PerceptronLearner] = []
    original_train, original_predict = (
        PerceptronLearner.train_batch,
        PerceptronDecisionPredictor.__call__,
    )

    def train(self, features, targets):
        result = original_train(self, features, targets)
        trained[:] = [self]
        return result

    def corrupt(self, request):
        result = original_predict(self, request)
        if trained:
            owner = trained[0]
            owner._state = replace(
                owner._state,
                completed_updates=owner.completed_updates + 1,
                training_examples=owner.training_examples + 3,
            )
        return result

    monkeypatch.setattr(PerceptronLearner, "train_batch", train)
    monkeypatch.setattr(PerceptronDecisionPredictor, "__call__", corrupt)
    r = study.run_example()
    assert r["complete"] is False and "source or budget" in r["error"]
    assert r["training_work"]["returned_updates"] == 4
    assert len(r["episodes"]) == 3 and r["episode_work"]["prediction_calls_attempted"] >= 1


def test_should_refuse_occupied_cli_output_before_model_work(tmp_path, monkeypatch):
    path = tmp_path / "original.json"
    path.write_text("preserve")
    monkeypatch.setattr(study, "run_example", lambda: pytest.fail("occupied output reached model"))
    with pytest.raises(FileExistsError):
        study.main(["--output", str(path)])
    assert path.read_text() == "preserve"


def test_should_write_failed_cli_report_and_nonzero_status(tmp_path, monkeypatch):
    path = tmp_path / "failure.json"
    monkeypatch.setattr(study, "run_example", lambda: dict(complete=False, error="paid failure"))
    assert study.main(["--output", str(path)]) == 1
    assert json.loads(path.read_bytes()) == dict(complete=False, error="paid failure")


def test_should_preserve_saved_no_replay_baseline_after_replay_integration(completed):
    from copy import deepcopy

    baseline = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "artifacts/runs/recurring-ticket-workflow-20261022/actual.json"
        ).read_bytes()
    )

    def stable(value):
        result = deepcopy(value)
        for key in ("elapsed_seconds", "sampled_process_rss"):
            result["costs"].pop(key)
        for key in ("started_at", "last_clock"):
            result["budget"].pop(key)
        return json.loads(json.dumps(result))

    assert stable(completed) == stable(baseline)
