"""Scalar projection-stream qualification; no model, tensor or optimizer math."""

from copy import deepcopy
import hashlib
import json
from math import log
from pathlib import Path
from typing import Any

import pytest

from scripts import run_open_jev_optimizer_control as runner
from src.core.learner_ports import TrainingUpdateFailed
from tests.test_native_choice_sequence import native_reply
from tests.test_native_optimizer_reset_control import (
    ResetLearner,
    cli_args,
    reset_reference as source_reference,
    summary_rows,
)


class ProjectionLearner(ResetLearner):
    def __init__(self, timeline, *, fail_at=None, projection_failure=False):
        super().__init__(timeline, fail_at=fail_at)
        self.projection_failure = projection_failure
        self._projection_work = dict(projection_attempts=0, projections_completed=0)
        self.last_adapter_projection = None

    @property
    def adapter_projection_work(self):
        return dict(self._projection_work)

    def project_adapters(self, rank):
        self.timeline.append(("project", self.work["committed_updates"], rank))
        self._projection_work["projection_attempts"] += 1
        if self.projection_failure:
            raise TrainingUpdateFailed(
                "controlled projection failure",
                "fixture_projection",
                {**self.work, **self.adapter_projection_work},
            )
        # The native projection owns the reset. Calling reset_optimizer_state
        # here would hide an accidental second reset in the orchestration.
        self._reset_work["optimizer_reset_attempts"] += 1
        self._reset_work["optimizer_resets_completed"] += 1
        self._projection_work["projections_completed"] += 1
        receipt = {
            "definition": "open_jev_owned_adapter_projection_v1",
            "effective_rank": rank,
            "optimizer_reset_work": self.optimizer_reset_work,
            "adapter_projection_work": self.adapter_projection_work,
        }
        self.last_adapter_projection = deepcopy(receipt)
        return receipt


@pytest.fixture(scope="module")
def reference():
    return source_reference()


@pytest.fixture(scope="module")
def reset_control(reference):
    groups = deepcopy(reference["sequential_summary"])
    groups["B"] = {"examples": 2, "accuracy": 0.5, "nll_mean": 1.0, "nll_status": "finite"}
    return {"source_sha256": "4" * 64, "by_context": groups}


@pytest.fixture(scope="module")
def completed(reference, reset_control):
    timeline: list[Any] = []
    owner = ProjectionLearner(timeline)
    wires: list[Any] = []
    progress: list[str] = []

    def predict(wire):
        timeline.append(("predict", wire["state"]))
        wires.append(deepcopy(wire))
        return native_reply(wire)

    report: dict[str, Any] = {}
    runner.run_optimizer_control(
        owner,
        predict,
        reference,
        report,
        progress.append,
        projection_rank=4,
        reset_reference=reset_control,
    )
    return report, owner, timeline, wires, progress


def test_should_project_once_after_a_and_train_b_without_an_additional_direct_reset(completed):
    report, owner, timeline, wires, progress = completed
    assert [event[0] for event in timeline] == ["train"] * 6 + ["project"] + ["train"] * 6 + [
        "predict"
    ] * 6
    assert timeline[6] == ("project", 6, 4)
    assert len(owner.training) == 12 and len(wires) == 6
    assert [target for _, target in owner.training] == ["shipping"] * 6 + ["password"] * 6
    assert (
        progress.count("before_adapter_projection")
        == progress.count("adapter_projection_finished")
        == 1
    )
    assert "before_optimizer_reset" not in progress and "optimizer_reset_finished" not in progress
    assert report["projection_receipt"] == owner.last_adapter_projection
    assert report["adapter_projection_work"] == {
        "projection_attempts": 1,
        "projections_completed": 1,
    }
    assert (
        report["reset_receipt"]
        == report["optimizer_reset_work"]
        == {"optimizer_reset_attempts": 1, "optimizer_resets_completed": 1}
    )
    assert report["control_complete"] is True and report["app_committed_updates"] == 12


def test_should_keep_delayed_receipts_and_projection_specific_phase_labels(completed):
    report, _, _, wires, _ = completed
    rows = report["delivery"]
    assert len(rows) == len({row["exposure_id"] for row in rows}) == 12
    assert [row["phase"] for row in rows] == ["after_A"] * 6 + ["after_B_projection"] * 6
    assert [row["observed_at"] for row in rows] == list(range(0, 24, 2))
    assert [row["receipt"]["arrived_at"] for row in rows] == list(range(1, 24, 2))
    assert [row["receipt"]["update_number"] for row in rows] == list(range(1, 13))
    assert all(row["prelabel_updates"] == 0 for row in rows)
    assert all(row["receipt"]["arrived_at"] == row["receipt"]["applied_at"] for row in rows)
    assert all(
        row["phase"] == "after_adapter_projection_B" and row["role"] == "outer_selection"
        for row in report["readouts"]
    )
    assert all(set(wire) == {"model", "state", "questions"} for wire in wires)
    assert report["native_work"]["committed_updates"] == 12
    json.dumps(report, allow_nan=False)


def projection_rows():
    rows = summary_rows()
    for row in rows:
        row["phase"] = "after_adapter_projection_B"
    return rows


def test_should_report_signed_projection_minus_reset_and_keep_warm_baseline_comparisons(
    reference, reset_control
):
    before = deepcopy((reference, reset_control))
    summary = runner.summarize_projection_control(projection_rows(), reference, reset_control)
    assert summary["by_context"]["B"]["accuracy"] == 1.0
    assert summary["contrasts"]["projection_minus_reset"]["B"]["accuracy"] == 0.5
    assert summary["contrasts"]["projection_minus_reset"]["B"]["nll_mean"] == pytest.approx(
        -log(0.9) - 1.0
    )
    assert summary["contrasts"]["projection_minus_baseline"]["B"]["accuracy"] == 1.0
    assert summary["contrasts"]["B_acquisition"]["nll_mean"] == pytest.approx(log(0.1) - log(0.9))
    assert summary["criteria"] == {
        "B_after_phase_success": True,
        "A_retained": False,
        "anchors_preserved": True,
    }
    assert summary["recovery"] == "not_tested"
    assert (reference, reset_control) == before


def test_should_preserve_infinite_nll_and_apply_original_anchor_tolerance(reference, reset_control):
    rows = projection_rows()
    rows[2].update(nll=None, nll_status="infinite")
    for row in rows:
        if row["context"] == "anchor":
            row["nll"] = reference["baseline_summary"]["anchor"]["nll_mean"] + 0.11
    result = runner.summarize_projection_control(rows, reference, reset_control)
    assert result["by_context"]["B"]["nll_mean"] is None
    assert result["contrasts"]["projection_minus_reset"]["B"]["nll_mean"] is None
    assert (
        result["contrasts"]["projection_minus_reset"]["B"]["nll_status"]
        == "nonfinite_comparison_unavailable"
    )
    assert result["criteria"]["anchors_preserved"] is False
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("fault", ["binding", "projection", "after_projection", "b_update"])
def test_should_retain_paid_receipt_before_callbacks_and_stop_without_retries(
    reference, reset_control, fault
):
    owner = ProjectionLearner(
        [], fail_at=9 if fault == "b_update" else None, projection_failure=fault == "projection"
    )
    predictions: list[None] = []

    def predict(wire):
        predictions.append(None)
        return native_reply(wire)

    def progress(event):
        if fault == "binding" and event == "before_adapter_projection":
            raise ValueError("controlled projection binding refusal")
        if fault == "after_projection" and event == "adapter_projection_finished":
            raise RuntimeError("controlled postprojection callback failure")

    report: dict[str, Any] = {}
    with pytest.raises((RuntimeError, ValueError)):
        runner.run_optimizer_control(
            owner,
            predict,
            reference,
            report,
            progress,
            projection_rank=4,
            reset_reference=reset_control,
        )
    attempted, committed, projections, completed = {
        "binding": (6, 6, 0, 0),
        "projection": (6, 6, 1, 0),
        "after_projection": (6, 6, 1, 1),
        "b_update": (9, 8, 1, 1),
    }[fault]
    assert len(owner.training) == attempted and report["app_committed_updates"] == committed
    assert sum("receipt" in row for row in report["delivery"]) == committed
    assert report["adapter_projection_work"] == {
        "projection_attempts": projections,
        "projections_completed": completed,
    }
    assert report["optimizer_reset_work"] == {
        "optimizer_reset_attempts": completed,
        "optimizer_resets_completed": completed,
    }
    if completed:
        assert report["projection_receipt"] == owner.last_adapter_projection
        assert report["reset_receipt"] == owner.optimizer_reset_work
    assert predictions == [] and report["control_complete"] is False and "summary" not in report
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "fault", ["bool", "negative", "too_large", "no_reset", "rehearsal", "missing_reference"]
)
def test_should_refuse_invalid_projection_options_before_runtime_or_training(
    reference, reset_control, monkeypatch, fault
):
    options: dict[str, Any] = {"projection_rank": 4, "reset_reference": reset_control}
    if fault in ("bool", "negative", "too_large"):
        options["projection_rank"] = {"bool": True, "negative": -1, "too_large": 9}[fault]
    elif fault == "no_reset":
        options["reset_optimizer"] = False
    elif fault == "rehearsal":
        options["rehearsal"] = True
    else:
        options["reset_reference"] = None
    calls: list[None] = []

    def forbidden(*args, **kwargs):
        calls.append(None)
        raise AssertionError("invalid projection mode reached runtime")

    monkeypatch.setattr(runner, "ExperienceInbox", forbidden)
    owner = ProjectionLearner([])
    with pytest.raises(ValueError):
        runner.run_optimizer_control(owner, forbidden, reference, {}, lambda event: None, **options)
    assert calls == [] and owner.training == []


@pytest.mark.parametrize("fault", ["spent", "receipt"])
def test_should_refuse_inherited_projection_history_before_runtime(
    reference, reset_control, monkeypatch, fault
):
    owner = ProjectionLearner([])
    if fault == "spent":
        owner._projection_work["projection_attempts"] = 1
    else:
        owner.last_adapter_projection = {"definition": "old historical receipt"}

    def forbidden(*args, **kwargs):
        raise AssertionError("nonfresh projection reached runtime")

    monkeypatch.setattr(runner, "ExperienceInbox", forbidden)
    with pytest.raises(ValueError):
        runner.run_optimizer_control(
            owner,
            forbidden,
            reference,
            {},
            lambda event: None,
            projection_rank=4,
            reset_reference=reset_control,
        )
    assert owner.training == []


@pytest.mark.parametrize(
    "extra",
    [
        ["--projection-rank", "4"],
        ["--reset-reference", "reset.json"],
        ["--projection-rank", "4", "--reset-reference", "reset.json", "--rehearsal"],
        ["--projection-rank", "4", "--reset-reference", "reset.json", "--optimizer-reset", "0"],
        ["--projection-rank", "4", "--reset-reference", "reset.json", "--update-health"],
        ["--projection-rank", "9", "--reset-reference", "reset.json"],
    ],
)
def test_should_refuse_invalid_cli_projection_selection_before_output_or_loading(
    tmp_path, monkeypatch, extra
):
    calls: list[None] = []

    def forbidden(*args, **kwargs):
        calls.append(None)
        raise AssertionError("invalid projection CLI reached loading")

    monkeypatch.setattr(runner, "load_reference", forbidden)
    monkeypatch.setattr(runner, "load_reset_reference", forbidden)
    monkeypatch.setattr(runner, "_load", forbidden)
    output = tmp_path / "not-created.json"
    with pytest.raises(SystemExit) as error:
        runner.main(cli_args(tmp_path, output) + extra)
    assert error.value.code == 2 and calls == [] and not output.exists()


@pytest.fixture(scope="module")
def accepted_reset():
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runs/native-optimizer-reset-20261022/actual.json"
    )
    with path.open("rb") as stream:
        body = stream.read(1024 * 1024 + 1)
    assert len(body) <= 1024 * 1024
    assert (
        hashlib.sha256(body).hexdigest()
        == "04c2b8d5b387c5d0617a68e37516246a5560961b5f04a1bea7a44bb99db18b45"
    )
    return path, body, json.loads(body)


def test_should_reuse_exact_saved_reset_endpoint_without_model_or_baseline_readouts(accepted_reset):
    path, body, source = accepted_reset
    observed = runner.load_reset_reference(path)
    assert observed == {
        "source_sha256": hashlib.sha256(body).hexdigest(),
        "by_context": source["summary"]["by_context"],
    }
    original = deepcopy(source)
    observed["by_context"]["B"]["accuracy"] = -1
    assert source == original


@pytest.mark.parametrize("fault", ["changed_bytes", "oversized"])
def test_should_refuse_changed_or_oversized_reset_reference_before_decoding(
    accepted_reset, tmp_path, fault
):
    _, body, _ = accepted_reset
    candidate = tmp_path / "changed-reset.json"
    candidate.write_bytes(body + b" " if fault == "changed_bytes" else b" " * (1024 * 1024 + 1))
    with pytest.raises(ValueError):
        runner.load_reset_reference(candidate)


@pytest.mark.parametrize("fault", ["definition", "incomplete", "readout_phase"])
def test_should_validate_reset_metadata_and_readout_binding_even_when_synthetic_bytes_are_hash_bound(
    accepted_reset, tmp_path, monkeypatch, fault
):
    _, _, source = accepted_reset
    changed = deepcopy(source)
    if fault == "definition":
        changed["definition"] = "unrelated_control_v1"
    elif fault == "incomplete":
        changed["control_complete"] = False
    else:
        changed["readouts"][0]["phase"] = "after_adapter_projection_B"
    body = json.dumps(changed, allow_nan=False).encode()
    candidate = tmp_path / "synthetic-reset.json"
    candidate.write_bytes(body)
    monkeypatch.setattr(runner, "RESET_REFERENCE_SHA256", hashlib.sha256(body).hexdigest())
    with pytest.raises(ValueError):
        runner.load_reset_reference(candidate)
