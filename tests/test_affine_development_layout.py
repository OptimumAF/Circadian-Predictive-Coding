"""Literal main parameters/scripted episodes, never native transfer evidence."""

from dataclasses import asdict
from hashlib import sha256
import json
import importlib.util
from pathlib import Path
from typing import Any

import pytest

from scripts import minigrid_affine_development as development
from scripts import minigrid_affine_workflow as workflow
from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters.minigrid_context_linear_learner import ContextLinearLearner
from src.core.minigrid_key_door import predefined_layouts
from test_context_linear_causal_games import PhaseEnvironment, budget
from test_minigrid_learning_pilot import Environment
from test_ordinary_fourth_task import state

literal_capture: dict[str, Any] = {}


class DevelopmentEnvironment(PhaseEnvironment):
    @property
    def report(self):
        return dict(
            super().report,
            layout=asdict(self.layout),
            role=self.role,
            seed=self.seed,
            max_steps=self.max_steps,
        )


def record():
    return dict(
        definition="portable_affine_recurring_train_v1",
        status="complete",
        reason=None,
        attempted_phases=["collection", "training", "games"],
        protocol=workflow.describe_workflow(recurring_training=True),
        training=dict(
            definition="minigrid_arrived_recurring_capacity3_train_stream_v1",
            status="complete",
            source_unchanged=True,
            actor_unchanged=True,
            failure_work=None,
            receipt_capture_error=None,
            stream=dict(
                evaluation_scope="training_only", completed_updates=10, training_examples=252
            ),
            endpoints=[
                dict(updates=n, state=asdict(state()) if n == 10 else None)
                for n in (0, 2, 4, 6, 8, 10)
            ],
            committed_receipts=[
                dict(update_number=i + 1, observed_at=i, arrived_at=i + 1, applied_at=i + 1)
                for i in range(10)
            ],
            budget=dict(
                updates_completed=10, started_at=111.0, initial_limits=dict(max_training_updates=10)
            ),
            catalogue=dict(committed_rows=dict(arrivals=90, replay=162, native=252)),
        ),
        games=dict(
            status="complete",
            state_unchanged=True,
            role="knownTRAIN_GA_return",
            episodes=[dict(status="complete", environment=dict(closed=True)) for _ in range(4)],
        ),
        workflow_budget=dict(started_at=100.0, elapsed_seconds=74.0),
    )


def literal_source(path, monkeypatch):
    raw = (json.dumps(record()) + "\n").encode()
    path.write_bytes(raw)
    monkeypatch.setattr(development, "SOURCE_BYTES", len(raw))
    monkeypatch.setattr(development, "SOURCE_SHA", sha256(raw).hexdigest())


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    folder = tmp_path_factory.mktemp("a4")
    source, output = folder / "literal-source.json", folder / "literal-output.json"
    work: dict[str, Any] = {}
    with pytest.MonkeyPatch.context() as patch:
        literal_source(source, patch)
        patch.setattr(Environment, "work", work)
        patch.setattr(evaluation, "MiniGridKeyDoor", DevelopmentEnvironment)
        patch.setattr(
            "sys.argv",
            [
                "affine",
                "--context-linear-a4-development",
                "--source",
                str(source),
                "--output",
                str(output),
            ],
        )
        cli.main()
        report = json.loads(output.read_text())
        literal_capture.update(definition="scripted_A4_CLI_not_native_v1", report=report)
    return report, work


def test_should_execute_cli_using_original_loop_on_fixed_development_plan(completed):
    report, _ = completed
    assert report["status"] == "complete", report["reason"]
    games = report["games"]
    assert games["role"] == "development_A4_from_ordinary_MAIN"
    assert games["state_unchanged"] is True
    assert [(e["context"], e["cue"], e["source_role"]) for e in games["episodes"]] == [
        ("A", "blue", "probe_test"),
        ("A", "red", "probe_test"),
    ]
    assert all(e["environment"]["closed"] for e in games["episodes"])
    assert games["work"]["raw_score_forwards_attempted"] == 0
    assert games["work"]["environment_steps_attempted"] == 4
    assert "development initial frame" in games["public_history"]["storage_scope"]
    assert predefined_layouts()[0].branch_row == 3
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runs/affine-development-layout-20261025/execute.py"
    )
    spec = importlib.util.spec_from_file_location("a4_result_reader", path)
    assert spec is not None and spec.loader is not None
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    summary = reader.summarize(report)
    assert summary["complete"] and summary["observed_cues"] == ["blue", "red"]
    assert summary["successes"] == 2 and summary["steps"] == [2, 2]
    # The reader must accept complete negative outcomes, not require a win.
    negative = json.loads(json.dumps(report))
    for entry in negative["games"]["episodes"]:
        entry["success"], entry["return"] = False, 0.0
        entry["environment"]["total_return"] = 0.0
    assert reader.summarize(negative)["successes"] == 0


def test_should_preserve_original_training_authority_only_as_provenance(completed):
    report = completed[0]
    assert report["protocol"]["new_fit_calls"] == report["protocol"]["new_training_rows"] == 0
    assert report["source"]["training_authority_restored"] is False
    assert report["source"]["training_budget"]["started_at"] == 111.0
    assert len(report["source"]["training_receipts"]) == 10
    assert report["source"]["catalogue"]["committed_rows"] == dict(
        arrivals=90, replay=162, native=252
    )
    assert (
        report["budget"]["updates_completed"] == report["budget"]["replay_examples_completed"] == 0
    )
    assert report["budget"]["started_at"] == report["budget"]["observed_budget_started_at"]
    assert report["budget"]["initial_limits"] == report["budget"]["limits"]


@pytest.mark.parametrize(
    "fault",
    ["private", "status", "source", "clock", "checkpoint", "catalogue", "games", "protocol"],
)
def test_should_refuse_invalid_ordinary_prefix_before_a_model_or_environment(monkeypatch, fault):
    value = record()
    if fault == "private":
        value["definition"] = "portable_affine_private_adaptation_v1"
    elif fault == "status":
        value["status"] = "failed"
    elif fault == "source":
        value["training"]["source_unchanged"] = False
    elif fault == "clock":
        value["training"]["committed_receipts"][0]["applied_at"] = 0
    elif fault == "checkpoint":
        value["training"]["endpoints"][-1]["state"]["completed_updates"] = 8
    elif fault == "catalogue":
        value["training"]["catalogue"]["committed_rows"]["replay"] = 0
    elif fault == "games":
        value["games"]["episodes"][0]["environment"]["closed"] = False
    else:
        value["protocol"]["new_fit_calls"] = 0
    monkeypatch.setattr(
        development, "ContextLinearLearner", lambda: pytest.fail("allocated source")
    )
    with pytest.raises((ValueError, KeyError)):
        development._admit_record(value)


@pytest.mark.parametrize("flag,recurring", [(1, True), (True, False)])
def test_should_refuse_ambiguous_development_modes_before_loop(monkeypatch, flag, recurring):
    monkeypatch.setattr(evaluation, "_run_checkpoint", lambda *a, **k: pytest.fail("entered loop"))
    with pytest.raises(ValueError, match="development"):
        evaluation.evaluate_context_linear_stream_checkpoint(
            ContextLinearLearner(),
            state(),
            budget=budget(),
            check_boundary=lambda: None,
            recurring_catalogue=recurring,
            development_a4=flag,
        )


def test_should_keep_original_recurring_plan_and_metadata_unchanged(monkeypatch):
    def run(source, current, **options):
        assert options["role"] == "knownTRAIN_GA_return"
        assert [(l.context, l.branch_row, c, r) for l, c, r in options["live_plan"]] == [
            (name, row, cue, "train")
            for name, row in (("G_train", 5), ("A", 3))
            for cue in ("blue", "red")
        ]
        assert options["return_metadata"] == dict(
            source_fit_calls=10,
            source_fit_rows=252,
            new_fit_calls=0,
            new_optimizer_updates=0,
            model_encoding=current.policy.encoding_id,
            readout_scope="known G_train then returned A TRAIN; ten252-row causal fit; no refit/final/probe",
        )
        options["extra_check"]()
        return dict(status="complete")

    monkeypatch.setattr(evaluation, "_run_checkpoint", run)
    assert (
        evaluation.evaluate_context_linear_stream_checkpoint(
            ContextLinearLearner(),
            state(),
            budget=budget(),
            check_boundary=lambda: None,
            recurring_catalogue=True,
        )["status"]
        == "complete"
    )


def test_should_refuse_changed_zero_update_clock_before_more_actions(monkeypatch, tmp_path):
    path = tmp_path / "literal.json"
    literal_source(path, monkeypatch)

    def play(source, current, *, budget, check_boundary, **options):
        budget.started_at -= 1
        check_boundary()
        pytest.fail("clock drift admitted")

    monkeypatch.setattr(development, "evaluate_context_linear_stream_checkpoint", play)
    report = development.run_saved_development(path)
    assert report["status"] == "failed" and "authority" in report["reason"]
    assert report["source"]["training_budget"]["started_at"] == 111.0


def test_should_retain_original_receipts_when_provenance_callback_is_corrupted(
    monkeypatch, tmp_path
):
    path = tmp_path / "literal.json"
    literal_source(path, monkeypatch)
    original = development.paid_report_guard

    def seal(value):
        check, detached = original(value)
        if "training_receipts" in value:
            value["training_receipts"][0]["applied_at"] = 99
        return check, detached

    monkeypatch.setattr(development, "paid_report_guard", seal)
    monkeypatch.setattr(
        development,
        "evaluate_context_linear_stream_checkpoint",
        lambda *a, **k: pytest.fail("played corrupt prefix"),
    )
    report = development.run_saved_development(path)
    assert report["status"] == "failed"
    assert report["source"]["training_receipts"][0]["applied_at"] == 1


def test_should_refuse_wrong_source_bytes_before_decode_or_work(monkeypatch, tmp_path):
    path = tmp_path / "wrong.json"
    path.write_text("{}")
    monkeypatch.setattr(
        development, "_admit_record", lambda *a: pytest.fail("decoded refused bytes")
    )
    report = development.run_saved_development(path)
    assert report["status"] == "failed" and "qualified127" in report["reason"]
