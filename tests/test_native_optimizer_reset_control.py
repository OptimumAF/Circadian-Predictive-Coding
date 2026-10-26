"""Fixed A/reset/B delivery and inherited comparisons using scalar fakes.

No Torch, model, tensor, RNG or optimizer is used. The fake reset records one
declared intervention; it does not approximate the native optimizer algorithm.
"""

from copy import deepcopy
import json
from math import log
from typing import Any

import pytest

from scripts import run_open_jev_b_control as cold
from scripts import run_open_jev_optimizer_control as runner
from scripts.run_open_jev_sequence import summarize_sequence
from src.core.learner_ports import TrainingUpdateFailed
from src.core.native_choice_sequence import sequence_examples
from tests.test_native_choice_sequence import RecordingLearner, native_reply, oracle_readouts
from tests.test_native_cold_b_control import reference_data, reference_payload, write_reference


def reset_reference():
    reference = reference_data()
    summary = summarize_sequence(oracle_readouts(sequence_examples()))
    reference["after_A_summary"] = summary["by_phase"]["after_A"]
    reference["after_A_weights"] = deepcopy(reference["baseline_weights"])
    for index, row in enumerate(reference["after_A_weights"].values()):
        row.update(
            after_sha256=("e" if index == 0 else "f") * 64, max_absolute_delta=0.1, l2_delta=0.2
        )
    return reference


@pytest.fixture(scope="module")
def reference():
    return reset_reference()


class ResetLearner(RecordingLearner):
    def __init__(self, timeline, *, fail_at=None, reset_failure=False):
        super().__init__(timeline, fail_at=fail_at)
        self.reset_failure = reset_failure
        self._reset_work = dict(optimizer_reset_attempts=0, optimizer_resets_completed=0)

    @property
    def optimizer_reset_work(self):
        return dict(self._reset_work)

    def reset_optimizer_state(self):
        self.timeline.append(("reset", self.work["committed_updates"]))
        self._reset_work["optimizer_reset_attempts"] += 1
        if self.reset_failure:
            raise TrainingUpdateFailed(
                "controlled reset failure",
                "fixture_reset",
                {**self.work, **self.optimizer_reset_work},
            )
        self._reset_work["optimizer_resets_completed"] += 1
        return self.optimizer_reset_work


@pytest.fixture(scope="module")
def full_control(reference):
    timeline: list[Any] = []
    owner = ResetLearner(timeline)
    wires: list[Any] = []
    progress: list[Any] = []

    def predict(wire):
        timeline.append(("predict", wire["state"]))
        wires.append(deepcopy(wire))
        reply = native_reply(wire)
        reply["answers"]["decision"].update(
            choice="billing", probabilities={"billing": 0.8, "password": 0.1, "shipping": 0.1}
        )
        return reply

    report: dict[str, Any] = {}
    runner.run_optimizer_control(owner, predict, reference, report, progress.append)
    return report, owner, timeline, wires, progress


def test_should_intervene_exactly_once_after_six_a_updates_before_six_b_updates(full_control):
    report, owner, timeline, wires, progress = full_control
    assert [entry[0] for entry in timeline] == ["train"] * 6 + ["reset"] + ["train"] * 6 + [
        "predict"
    ] * 6
    assert timeline[6] == ("reset", 6)
    assert [(request.query, target) for request, target in owner.training] == [
        ("Case marker: zorbex", "shipping"),
        ("Intake code zorbex. Route this request.", "shipping"),
    ] * 3 + [
        ("Case marker: neldax", "password"),
        ("Intake code neldax. Route this request.", "password"),
    ] * 3
    assert len(wires) == 6 and len(report["readouts"]) == 6
    assert (
        progress.count("before_optimizer_reset") == progress.count("optimizer_reset_finished") == 1
    )
    assert report["summary"]["criteria"]["B_after_phase_success"] is False
    assert report["app_committed_updates"] == 12
    assert report["control_complete"] is True
    assert (
        report["optimizer_reset_work"]
        == owner.optimizer_reset_work
        == {"optimizer_reset_attempts": 1, "optimizer_resets_completed": 1}
    )


def test_should_keep_unique_delayed_deliveries_and_never_remeasure_inherited_endpoints(
    full_control,
):
    report, owner, _, wires, _ = full_control
    rows = report["delivery"]
    assert len(rows) == len({row["exposure_id"] for row in rows}) == 12
    assert [row["observed_at"] for row in rows] == list(range(0, 24, 2))
    assert all(row["prelabel_updates"] == 0 for row in rows)
    assert [row["phase"] for row in rows] == ["after_A"] * 6 + ["after_B_reset"] * 6
    receipts = [row["receipt"] for row in rows]
    assert [row["update_number"] for row in receipts] == list(range(1, 13))
    assert [row["arrived_at"] for row in receipts] == list(range(1, 24, 2))
    assert all(row["arrived_at"] == row["applied_at"] for row in receipts)
    development = [example for example in sequence_examples() if example.role == "outer_selection"]
    assert [wire["state"] for wire in wires] == [example.request.query for example in development]
    assert all(set(wire) == {"model", "state", "questions"} for wire in wires)
    assert all(
        row["phase"] == "after_optimizer_reset_B" and row["role"] == "outer_selection"
        for row in report["readouts"]
    )
    assert report["native_work"] == owner.work
    assert report["inbox_stopped"] is False
    json.dumps(report, allow_nan=False)


def summary_rows():
    from tests.test_native_cold_b_control import cold_rows

    rows = cold_rows()
    for row in rows:
        row["phase"] = "after_optimizer_reset_B"
    return rows


def test_should_report_reset_acquisition_and_retention_separately_from_inherited_warm_changes(
    reference,
):
    rows = summary_rows()
    original = deepcopy(reference)
    summary = runner.summarize_optimizer_control(rows, reference)
    assert reference == original
    assert summary["baseline_reused"] == reference["baseline_summary"]
    assert summary["after_A_reused"] == reference["after_A_summary"]
    assert summary["warm_after_B"] == reference["sequential_summary"]
    assert all(group["examples"] == 2 for group in summary["by_context"].values())
    contrasts = summary["contrasts"]
    assert contrasts["reset_minus_baseline"]["B"]["accuracy"] == 1.0
    assert contrasts["reset_minus_warm_after_B"]["B"]["accuracy"] == 0.0
    assert contrasts["reset_B_acquisition"]["accuracy"] == 1.0
    assert contrasts["reset_B_acquisition"]["nll_mean"] == pytest.approx(log(0.1) - log(0.9))
    assert contrasts["warm_B_acquisition"] == reference["sequential_B_acquisition"]
    assert summary["criteria"] == {
        "B_after_phase_success": True,
        "A_retained": False,
        "anchors_preserved": True,
    }
    assert not any("return" in key or "recovery" in key for key in summary["criteria"])
    assert summary["recovery"] == "not_tested"
    summary["after_A_reused"]["A"]["accuracy"] = 0.25
    assert reference == original


def test_should_retain_infinite_nll_and_apply_accuracy_and_anchor_nll_guards(reference):
    rows = summary_rows()
    rows[2].update(nll=None, nll_status="infinite")
    for row in rows:
        if row["context"] == "A":
            row.update(correct=True, nll=-log(0.8))
        elif row["context"] == "anchor":
            row["nll"] = reference["baseline_summary"]["anchor"]["nll_mean"] + 0.11
    summary = runner.summarize_optimizer_control(rows, reference)
    assert summary["by_context"]["B"]["examples"] == 2
    assert summary["by_context"]["B"]["nll_mean"] is None
    assert summary["by_context"]["B"]["nll_status"] == "infinite"
    assert summary["contrasts"]["reset_B_acquisition"]["nll_mean"] is None
    assert summary["criteria"]["A_retained"] is True
    assert summary["criteria"]["anchors_preserved"] is False
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("fault", ["missing", "foreign", "changed"])
def test_should_refuse_unmatched_after_a_weight_binding_before_reset(reference, fault):
    observed = deepcopy(reference["after_A_weights"])
    if fault == "missing":
        observed.pop("head.weight")
    elif fault == "foreign":
        observed["foreign.weight"] = deepcopy(observed["head.weight"])
    else:
        observed["head.weight"]["after_sha256"] = "9" * 64
    with pytest.raises(ValueError):
        runner.validate_after_A_weights(observed, reference)


def test_should_admit_exact_after_a_hashes_without_requiring_they_equal_the_cold_weights(reference):
    observed = deepcopy(reference["after_A_weights"])
    runner.validate_after_A_weights(observed, reference)
    assert observed["head.weight"]["before_sha256"] != observed["head.weight"]["after_sha256"]


@pytest.mark.parametrize(
    "fault", ["binding", "reset", "after_reset", "b_update", "readout", "postcommit"]
)
def test_should_retain_paid_prefix_and_stop_without_retry_at_declared_boundaries(reference, fault):
    owner = ResetLearner(
        [], fail_at=9 if fault == "b_update" else None, reset_failure=fault == "reset"
    )
    predictions: list[None] = []

    def predict(wire):
        predictions.append(None)
        if fault == "readout" and len(predictions) == 2:
            raise RuntimeError("controlled second reset-control readout failure")
        return native_reply(wire)

    def progress(event):
        if fault == "binding" and event == "before_optimizer_reset":
            raise ValueError("controlled after-A binding refusal")
        if fault == "after_reset" and event == "optimizer_reset_finished":
            raise RuntimeError("controlled postreset stop")
        if fault == "postcommit" and event == "training_finished":
            raise RuntimeError("controlled postcommit stop")

    report: dict[str, Any] = {}
    with pytest.raises((RuntimeError, ValueError)):
        runner.run_optimizer_control(owner, predict, reference, report, progress)
    attempted, committed, resets, completed, predictions_count = {
        "binding": (6, 6, 0, 0, 0),
        "reset": (6, 6, 1, 0, 0),
        "after_reset": (6, 6, 1, 1, 0),
        "b_update": (9, 8, 1, 1, 0),
        "readout": (12, 12, 1, 1, 2),
        "postcommit": (1, 1, 0, 0, 0),
    }[fault]
    assert len(owner.training) == attempted and owner.work["committed_updates"] == committed
    assert report["app_committed_updates"] == committed
    assert (
        owner.optimizer_reset_work
        == report["optimizer_reset_work"]
        == {"optimizer_reset_attempts": resets, "optimizer_resets_completed": completed}
    )
    assert len(predictions) == predictions_count and "summary" not in report
    assert report["control_complete"] is False
    assert sum("receipt" in row for row in report["delivery"]) == committed
    if completed:
        assert report["reset_receipt"] == owner.optimizer_reset_work
    if fault == "readout":
        assert "result" in report["readouts"][0] and "response_json" not in report["readouts"][1]
    json.dumps(report, allow_nan=False)


def test_should_revalidate_protected_corpus_before_runtime_or_native_work(reference, monkeypatch):
    examples = deepcopy(sequence_examples())
    object.__setattr__(examples[0], "role", "final_test")
    monkeypatch.setattr(runner, "sequence_examples", lambda: examples)
    calls: list[None] = []

    def forbidden(*args, **kwargs):
        calls.append(None)
        raise AssertionError("invalid corpus reached optimizer control work")

    monkeypatch.setattr(runner, "ExperienceInbox", forbidden)
    owner = ResetLearner([])
    with pytest.raises(ValueError):
        runner.run_optimizer_control(owner, forbidden, reference, {}, lambda event: None)
    assert calls == [] and owner.training == []


def test_should_opt_in_after_a_reference_fields_without_changing_default_loader_schema(
    tmp_path, monkeypatch
):
    payload = reference_payload()
    reference = reset_reference()
    payload["summary"]["by_phase"]["after_A"] = reference["after_A_summary"]
    payload["weight_checkpoints"].append(
        dict(after_updates=6, model_training=False, weights=reference["after_A_weights"])
    )
    path, body, _ = write_reference(tmp_path, monkeypatch, payload)
    original = cold.load_reference(path)
    selected = cold.load_reference(path, include_after_A=True)
    assert set(selected) == set(original) | {"after_A_summary", "after_A_weights"}
    assert selected["after_A_summary"] == reference["after_A_summary"]
    assert selected["after_A_weights"] == reference["after_A_weights"]
    selected["after_A_weights"]["head.weight"]["after_sha256"] = "0" * 64
    assert path.read_bytes() == body


def test_should_refuse_inherited_reset_charge_before_constructing_control_runtime(
    reference, monkeypatch
):
    owner = ResetLearner([])
    owner._reset_work["optimizer_reset_attempts"] = 1
    calls: list[None] = []

    def forbidden(*args, **kwargs):
        calls.append(None)
        raise AssertionError("nonfresh reset allowance reached control work")

    monkeypatch.setattr(runner, "ExperienceInbox", forbidden)
    with pytest.raises(ValueError):
        runner.run_optimizer_control(owner, forbidden, reference, {}, lambda event: None)
    assert calls == [] and owner.training == []


def test_should_refuse_after_a_reference_that_was_not_in_evaluation_mode(tmp_path, monkeypatch):
    payload = reference_payload()
    reference = reset_reference()
    payload["summary"]["by_phase"]["after_A"] = reference["after_A_summary"]
    payload["weight_checkpoints"].append(
        dict(after_updates=6, model_training=True, weights=reference["after_A_weights"])
    )
    path, _, _ = write_reference(tmp_path, monkeypatch, payload)
    with pytest.raises(ValueError):
        cold.load_reference(path, include_after_A=True)


def cli_args(tmp_path, output):
    return [
        "--reference",
        str(tmp_path / "reference.json"),
        "--source",
        str(tmp_path / "source"),
        "--checkpoint",
        str(tmp_path / "checkpoint"),
        "--output",
        str(output),
    ]


def test_should_preserve_existing_output_before_reference_or_native_model_loading(
    tmp_path, monkeypatch
):
    output = tmp_path / "existing.json"
    output.write_bytes(b"preserve previous control")
    calls: list[None] = []

    def forbidden(*args, **kwargs):
        calls.append(None)
        raise AssertionError("existing output reached load")

    monkeypatch.setattr(runner, "load_reference", forbidden)
    monkeypatch.setattr(runner, "_load", forbidden)
    with pytest.raises(SystemExit) as error:
        runner.main(cli_args(tmp_path, output))
    assert error.value.code == 2 and calls == []
    assert output.read_bytes() == b"preserve previous control"


def test_should_request_after_a_reference_before_any_native_model_load(tmp_path, monkeypatch):
    reads: list[Any] = []
    loads: list[None] = []

    def read(path, *, include_after_A=False):
        reads.append((path, include_after_A))
        raise ValueError("controlled after-A reference refusal")

    def forbidden(*args, **kwargs):
        loads.append(None)
        raise AssertionError("invalid reference reached native load")

    monkeypatch.setattr(runner, "load_reference", read)
    monkeypatch.setattr(runner, "_load", forbidden)
    output = tmp_path / "refused.json"
    assert runner.main(cli_args(tmp_path, output)) == 1
    assert reads == [(tmp_path / "reference.json", True)] and loads == []
    report = json.loads(output.read_text())
    assert report["complete"] is False and report["model_load_attempts"] == 0
    assert "controlled after-A reference refusal" in report["error"]
