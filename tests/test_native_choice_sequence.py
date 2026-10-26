"""Frozen recurring-task delivery and metrics using recording scalar fakes.

No Torch, upstream package, model, RNG, gradients or optimizer is imported or run.
The fake learner records opaque requests; it does not simulate neural learning.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from math import log
from typing import Any

import pytest

from scripts import run_open_jev_sequence as runner
from src.core.learner_ports import TrainingDiagnostic, TrainingUpdateFailed
from src.core.native_choice_sequence import (
    PASSES,
    PHASES,
    ChoiceSequenceExample,
    sequence_examples,
    validate_sequence_examples,
)


@pytest.fixture(scope="module")
def corpus():
    return sequence_examples()


def test_should_freeze_four_train_and_six_development_examples_with_separate_template_groups(
    corpus,
):
    assert type(corpus) is tuple and len(corpus) == 10
    assert len({record.example_id for record in corpus}) == 10
    assert len({record.request.request_id for record in corpus}) == 10
    assert all(type(record) is ChoiceSequenceExample for record in corpus)
    train = [record for record in corpus if record.role == "train"]
    development = [record for record in corpus if record.role == "outer_selection"]
    assert len(train) == 4 and len(development) == 6
    assert {record.source_group for record in train} == {"train_marker_templates"}
    assert {record.source_group for record in development} == {
        "development_marker_templates",
        "development_anchor_templates",
    }
    for context, code, target in (("A", "zorbex", "shipping"), ("B", "neldax", "password")):
        assert {
            (record.request.query, record.target_id)
            for record in train
            if record.context == context
        } == {
            (f"Case marker: {code}", target),
            (f"Intake code {code}. Route this request.", target),
        }
        assert {
            (record.request.query, record.target_id)
            for record in development
            if record.context == context
        } == {
            (f"Case marker: {code}; please assist.", target),
            (f"A support ticket marked {code} needs routing.", target),
        }
    assert {
        (record.request.query, record.target_id)
        for record in development
        if record.context == "anchor"
    } == {("invoice refund", "billing"), ("reset my password", "password")}
    assert PHASES == (("after_A", "A"), ("after_B", "B"), ("after_return", "A"))
    assert PASSES == 3
    validate_sequence_examples(corpus)
    with pytest.raises(FrozenInstanceError):
        setattr(corpus[0], "target_id", "billing")


def test_should_keep_original_support_catalog_and_no_label_fields_in_each_request(corpus):
    expected = (
        ("billing", "invoice billing payment refund"),
        ("password", "password reset login account"),
        ("shipping", "shipping parcel delivery tracking"),
    )
    for record in corpus:
        assert (
            tuple(
                (item.candidate_id, item.description) for item in record.request.allowed_candidates
            )
            == expected
        )
        assert set(vars(record.request)) == {"request_id", "query", "allowed_candidates"}
        assert record.target_id in {item.candidate_id for item in record.request.allowed_candidates}


@pytest.mark.parametrize("fault", ["final", "probe", "overlap", "target", "catalog", "duplicate"])
def test_should_refuse_protected_or_unmatched_corpus_metadata_before_work(corpus, fault):
    changed = list(deepcopy(corpus))
    if fault == "final":
        object.__setattr__(changed[0], "role", "final_test")
    elif fault == "probe":
        object.__setattr__(changed[0], "role", "probe_train")
    elif fault == "overlap":
        development = next(record for record in changed if record.role == "outer_selection")
        object.__setattr__(development, "source_group", "train_marker_templates")
    elif fault == "target":
        object.__setattr__(changed[0], "target_id", "not-allowed")
    elif fault == "catalog":
        object.__setattr__(changed[0].request.allowed_candidates[0], "description", "replacement")
    else:
        changed[-1] = changed[0]
    with pytest.raises(ValueError):
        validate_sequence_examples(tuple(changed))


class RecordingLearner:
    def __init__(self, timeline: list[Any], *, fail_at: int | None = None):
        self.timeline, self.fail_at = timeline, fail_at
        self.training: list[Any] = []
        self.last_update: dict[str, Any] | None = None
        self._work = {
            "training_attempts": 0,
            "forward_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "committed_updates": 0,
        }

    @property
    def work(self):
        return dict(self._work)

    def train_batch(self, features, targets):
        self.timeline.append(("train", features.query))
        self.training.append((deepcopy(features), targets))
        for name in ("training_attempts", "forward_calls", "backward_calls", "optimizer_steps"):
            self._work[name] += 1
        if len(self.training) == self.fail_at:
            raise TrainingUpdateFailed(
                "controlled fake recurring update failure", "fixture_native_choice", self.work
            )
        self._work["committed_updates"] += 1
        self.last_update = {"loss": 0.5, "gradient_norm_before_clip": 0.25}
        return TrainingDiagnostic("fixture_native_choice", 0.5)


def native_reply(wire):
    allowed = wire["questions"]["decision"]["criteria"]
    query = wire["state"]
    selected = (
        "shipping"
        if "zorbex" in query
        else "password"
        if "neldax" in query or "password" in query
        else "billing"
    )
    return {
        "model": "Qwen/Qwen3.5-2B",
        "answers": {
            "decision": {
                "type": "choice",
                "choice": selected,
                "probabilities": {key: 0.8 if key == selected else 0.1 for key in allowed},
                "confidence": 0.7,
            }
        },
        "usage": {"input_tokens": 42, "output_tokens": 0},
        "metadata": {
            "base_revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
            "method": "lora_decision_head",
            "temperature": 1.518796342858676,
            "max_length": 256,
            "backbone_parameter_dtypes": ["torch.float32"],
            "candidate_sequences": 3,
            "prefix_cache": {"enabled": False, "mode": "independent_candidates"},
        },
    }


@pytest.fixture(scope="module")
def full_run():
    timeline: list[Any] = []
    owner = RecordingLearner(timeline)
    wires: list[Any] = []
    events: list[Any] = []

    def predict(wire):
        timeline.append(("predict", wire["state"]))
        wires.append(deepcopy(wire))
        reply = native_reply(wire)
        # Deliberately poor but valid: unsuccessful criteria cannot stop phases.
        reply["answers"]["decision"].update(
            choice="billing", probabilities={"billing": 0.8, "password": 0.1, "shipping": 0.1}
        )
        return reply

    report: dict[str, Any] = {}
    runner.run_sequence(owner, predict, report, events.append)
    return report, owner, timeline, wires, events


def test_should_run_all_three_fixed_phases_despite_unsuccessful_marker_readouts(full_run):
    report, owner, timeline, wires, _ = full_run
    assert len(owner.training) == 18 and len(wires) == 24
    assert [event[0] for event in timeline] == ["predict"] * 6 + (
        ["train"] * 6 + ["predict"] * 6
    ) * 3
    assert len(report["readouts"]) == 24
    for phase, context in PHASES:
        rows = [
            row for row in report["readouts"] if row["phase"] == phase and row["context"] == context
        ]
        assert len(rows) == 2 and all(row["correct"] is False for row in rows)
    assert owner.work["committed_updates"] == 18


def test_should_train_only_the_two_context_templates_three_passes_per_phase(full_run, corpus):
    _, owner, _, _, _ = full_run
    expected: list[Any] = []
    for _, context in PHASES:
        training = [
            record for record in corpus if record.role == "train" and record.context == context
        ]
        for _ in range(PASSES):
            expected.extend((record.request.query, record.target_id) for record in training)
    assert [(features.query, target) for features, target in owner.training] == expected
    assert sum(target == "shipping" for _, target in owner.training) == 12
    assert sum(target == "password" for _, target in owner.training) == 6
    development_queries = {
        record.request.query for record in corpus if record.role == "outer_selection"
    }
    assert not development_queries.intersection(features.query for features, _ in owner.training)


def test_should_keep_labels_and_group_metadata_out_of_native_prediction_wires(full_run, corpus):
    report, _, _, wires, _ = full_run
    development = [record for record in corpus if record.role == "outer_selection"]
    assert [wire["state"] for wire in wires] == [record.request.query for record in development] * 4
    for wire in wires:
        assert set(wire) == {"model", "state", "questions"}
        assert set(wire["questions"]["decision"]) == {"type", "instructions", "criteria"}
        assert list(wire["questions"]["decision"]["criteria"]) == [
            "billing",
            "password",
            "shipping",
        ]
    assert all(row["role"] == "outer_selection" for row in report["readouts"])
    assert {row["source_group"] for row in report["readouts"]} == {
        "development_marker_templates",
        "development_anchor_templates",
    }
    json.dumps(report, allow_nan=False)


def test_should_keep_eighteen_unique_deliveries_prelabel_zero_and_original_source_lineage(
    full_run, corpus
):
    report, owner, _, _, _ = full_run
    rows = report["delivery"]
    assert len(rows) == len({row["exposure_id"] for row in rows}) == 18
    assert [row["observed_at"] for row in rows] == list(range(0, 36, 2))
    assert all(row["prelabel_updates"] == 0 for row in rows)
    assert {row["source_group"] for row in rows} == {"train_marker_templates"}
    assert {row["source_example_id"] for row in rows} == {
        record.example_id for record in corpus if record.role == "train"
    }
    receipts = [row["receipt"] for row in rows]
    assert [receipt["update_number"] for receipt in receipts] == list(range(1, 19))
    assert [receipt["arrived_at"] for receipt in receipts] == list(range(1, 36, 2))
    assert all(receipt["arrived_at"] == receipt["applied_at"] for receipt in receipts)
    assert all(receipt["sample_id"] == row["exposure_id"] for receipt, row in zip(receipts, rows))
    assert report["native_work"] == owner.work
    assert report["app_committed_updates"] == 18 and report["inbox_stopped"] is False
    assert report["sequence_complete"] is True


def test_should_refuse_invalid_corpus_before_constructing_an_inbox_or_calling_learner(
    monkeypatch, corpus
):
    changed = deepcopy(corpus)
    object.__setattr__(changed[0], "role", "final_test")
    monkeypatch.setattr(runner, "sequence_examples", lambda: changed)
    calls: list[None] = []

    def forbidden(*args, **kwargs):
        calls.append(None)
        raise AssertionError("invalid corpus reached runtime work")

    monkeypatch.setattr(runner, "ExperienceInbox", forbidden)
    owner = RecordingLearner([])
    with pytest.raises(ValueError):
        runner.run_sequence(owner, forbidden, {}, lambda event: None)
    assert calls == [] and owner.training == []


def test_should_keep_prior_phase_readouts_and_seven_commits_when_eighth_update_fails():
    timeline: list[Any] = []
    owner = RecordingLearner(timeline, fail_at=8)
    readouts: list[Any] = []

    def predict(wire):
        readouts.append(deepcopy(wire))
        return native_reply(wire)

    report: dict[str, Any] = {}
    with pytest.raises(TrainingUpdateFailed):
        runner.run_sequence(owner, predict, report, lambda event: None)
    assert len(owner.training) == 8 and owner.work["committed_updates"] == 7
    assert len(readouts) == len(report["readouts"]) == 12
    assert {row["phase"] for row in report["readouts"]} == {"baseline", "after_A"}
    assert report["sequence_complete"] is False
    json.dumps(report, allow_nan=False)


def test_should_stop_after_first_unreturned_native_readout_before_any_training():
    owner = RecordingLearner([])
    calls: list[None] = []

    def predict(wire):
        calls.append(None)
        raise RuntimeError("controlled first recurring readout failure")

    report: dict[str, Any] = {}
    with pytest.raises(RuntimeError, match="controlled first recurring readout failure"):
        runner.run_sequence(owner, predict, report, lambda event: None)
    assert calls == [None] and owner.training == []
    assert len(report["readouts"]) == 1
    assert "response_json" not in report["readouts"][0]
    json.dumps(report, allow_nan=False)


def test_should_retain_first_paid_receipt_when_postcommit_progress_callback_stops_sequence():
    owner = RecordingLearner([])
    predictions: list[None] = []

    def predict(wire):
        predictions.append(None)
        return native_reply(wire)

    def progress(event):
        if event == "training_finished":
            raise RuntimeError("controlled postcommit progress stop")

    report: dict[str, Any] = {}
    with pytest.raises(RuntimeError, match="controlled postcommit progress stop"):
        runner.run_sequence(owner, predict, report, progress)
    assert len(predictions) == len(report["readouts"]) == 6
    assert len(owner.training) == 1
    assert report["native_work"]["committed_updates"] == report["app_committed_updates"] == 1
    assert len(report["delivery"]) == 1
    assert report["delivery"][0]["receipt"]["update_number"] == 1
    assert report["delivery"][0]["training"] == owner.last_update
    assert report["sequence_complete"] is False
    json.dumps(report, allow_nan=False)


def oracle_readouts(corpus):
    rows: list[dict[str, Any]] = []
    development = [record for record in corpus if record.role == "outer_selection"]
    for phase in ("baseline", "after_A", "after_B", "after_return"):
        seen: dict[str, int] = {}
        for record in development:
            index = seen.get(record.context, 0)
            seen[record.context] = index + 1
            if record.context == "A":
                probability = {"baseline": 0.1, "after_A": 0.8, "after_return": 0.9}.get(
                    phase, 0.8 if index == 0 else 0.1
                )
            elif record.context == "B":
                probability = 0.8 if phase in ("after_B", "after_return") else 0.1
            else:
                probability = {
                    "baseline": 0.8,
                    "after_A": 0.79,
                    "after_B": 0.76,
                    "after_return": 0.79,
                }[phase]
            rows.append(
                {
                    "phase": phase,
                    "context": record.context,
                    "example_id": record.example_id,
                    "correct": probability > 0.5,
                    "nll": -log(probability),
                    "nll_status": "finite",
                }
            )
    return rows


def test_should_compute_signed_acquisition_retention_and_return_from_all_declared_rows(corpus):
    rows = oracle_readouts(corpus)
    before = deepcopy(rows)
    summary = runner.summarize_sequence(rows)
    assert rows == before
    phases = summary["by_phase"]
    assert all(group["examples"] == 2 for groups in phases.values() for group in groups.values())
    assert phases["after_B"]["A"]["accuracy"] == 0.5
    assert phases["after_B"]["A"]["nll_mean"] == pytest.approx((-log(0.8) - log(0.1)) / 2)
    deltas = summary["contrasts"]["phase_deltas"]
    assert deltas["A_acquisition"]["accuracy"] == deltas["B_acquisition"]["accuracy"] == 1.0
    assert deltas["A_acquisition"]["nll_mean"] == pytest.approx(log(0.1) - log(0.8))
    assert deltas["A_retention"]["accuracy"] == -0.5
    assert deltas["A_retention"]["nll_mean"] > 0
    assert deltas["A_return"]["accuracy"] == 0.5
    assert deltas["A_return"]["nll_mean"] < 0
    assert summary["criteria"] == {
        "A_after_phase_success": True,
        "B_after_phase_success": True,
        "A_retained_if_acquired": False,
        "A_return_success": True,
        "A_recovery_demonstrated": True,
        "anchors_preserved": True,
    }
    for phase in ("baseline", "after_A", "after_B", "after_return"):
        assert summary["unchanged_control"][phase] == phases["baseline"]
    assert summary["unchanged_control_definition"] == "baseline_reused_no_additional_calls"
    assert summary["contrasts"]["trained_minus_unchanged"]["after_A"]["A"]["accuracy"] == 1.0


def test_should_keep_infinite_nll_in_group_and_undefined_contrasts_without_dropping_a_row(corpus):
    rows = oracle_readouts(corpus)
    changed = next(row for row in rows if row["phase"] == "baseline" and row["context"] == "A")
    changed.update(nll=None, nll_status="infinite")
    summary = runner.summarize_sequence(rows)
    group = summary["by_phase"]["baseline"]["A"]
    assert group["examples"] == 2 and group["nll_mean"] is None
    assert group["nll_status"] == "infinite"
    assert summary["contrasts"]["phase_deltas"]["A_acquisition"]["nll_mean"] is None
    contrast = summary["contrasts"]["trained_minus_unchanged"]["baseline"]["A"]
    assert (
        contrast["nll_mean"] is None
        and contrast["nll_status"] == "nonfinite_comparison_unavailable"
    )
    json.dumps(summary, allow_nan=False)


def test_should_distinguish_return_success_from_recovery_and_enforce_anchor_nll_guard(corpus):
    rows = oracle_readouts(corpus)
    for row in rows:
        if row["phase"] == "after_B" and row["context"] == "A":
            row.update(correct=True, nll=-log(0.8))
    summary = runner.summarize_sequence(rows)
    assert summary["criteria"]["A_retained_if_acquired"] is True
    assert summary["criteria"]["A_return_success"] is True
    assert summary["criteria"]["A_recovery_demonstrated"] is False
    for row in rows:
        if row["phase"] == "after_B" and row["context"] == "anchor":
            row["nll"] = -log(0.8) + 0.11
    guarded = runner.summarize_sequence(rows)
    assert guarded["criteria"]["anchors_preserved"] is False


@pytest.mark.parametrize("fault", ["partial", "duplicate", "context", "nan"])
def test_should_refuse_incomplete_or_unvalidated_summary_rows(corpus, fault):
    rows = oracle_readouts(corpus)
    if fault == "partial":
        rows.pop()
    elif fault == "duplicate":
        rows[1]["example_id"] = rows[0]["example_id"]
    elif fault == "context":
        rows[0]["context"] = "anchor"
    else:
        rows[0]["nll"] = float("nan")
    with pytest.raises(ValueError):
        runner.summarize_sequence(rows)


def test_should_revalidate_corpus_before_optional_native_cli_loading(monkeypatch, tmp_path, corpus):
    changed = deepcopy(corpus)
    object.__setattr__(changed[0], "role", "final_test")
    monkeypatch.setattr(runner, "sequence_examples", lambda: changed)
    loads: list[None] = []

    def load(*args, **kwargs):
        loads.append(None)
        raise AssertionError("invalid corpus reached native model load")

    monkeypatch.setattr(runner, "_load", load)
    output = tmp_path / "refused.json"
    assert (
        runner.main(
            [
                "--source",
                str(tmp_path / "source"),
                "--checkpoint",
                str(tmp_path / "checkpoint"),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    report = json.loads(output.read_text())
    assert loads == [] and report["model_load_attempts"] == 0
    assert report["complete"] is False and report["error"]
