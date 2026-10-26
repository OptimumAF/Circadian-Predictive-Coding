"""Bounded TRAIN-only rehearsal through real orchestration and scalar fakes.

No Torch, model, optimizer, tensor, RNG or inference is imported or executed.
Recording learners expose paid calls without simulating learned behavior.
"""

from copy import deepcopy
import json
from math import log
from typing import Any

import pytest

from scripts import run_open_jev_optimizer_control as runner
from scripts import run_open_jev_sequence as sequence
from src.core.learner_ports import TrainingUpdateFailed
from src.core.native_choice_sequence import sequence_examples
from tests.test_native_choice_sequence import native_reply
from tests.test_native_optimizer_reset_control import ResetLearner, reset_reference
from tests.test_native_cold_b_control import cold_rows


@pytest.fixture(scope="module")
def reference():
    return reset_reference()


@pytest.fixture(scope="module")
def controls(reference):
    results = {}
    for reset in (False, True):
        timeline: list[Any] = []
        owner = ResetLearner(timeline)
        wires: list[Any] = []
        events: list[Any] = []

        def predict(wire):
            timeline.append(("predict", wire["state"]))
            wires.append(deepcopy(wire))
            reply = native_reply(wire)
            # Valid adverse decisions cannot change the prescribed later work.
            reply["answers"]["decision"].update(
                choice="billing", probabilities={"billing": 0.8, "password": 0.1, "shipping": 0.1}
            )
            return reply

        report: dict[str, Any] = {}
        runner.run_optimizer_control(
            owner, predict, reference, report, events.append, rehearsal=True, reset_optimizer=reset
        )
        results[reset] = report, owner, timeline, wires, events
    return results


@pytest.mark.parametrize("reset", [False, True])
def test_should_run_identical_a_b_replay_exposures_with_only_the_declared_reset_difference(
    controls, reset
):
    report, owner, timeline, wires, events = controls[reset]
    a = [
        ("Case marker: zorbex", "shipping"),
        ("Intake code zorbex. Route this request.", "shipping"),
    ]
    b = [
        ("Case marker: neldax", "password"),
        ("Intake code neldax. Route this request.", "password"),
    ]
    assert [(request.query, target) for request, target in owner.training] == a * 3 + [
        b[0],
        a[0],
        b[1],
        a[1],
    ] * 3
    assert [entry[0] for entry in timeline] == ["train"] * 6 + (["reset"] if reset else []) + [
        "train"
    ] * 12 + ["predict"] * 6
    assert events.count("before_optimizer_reset") == 1
    assert events.count("optimizer_reset_finished") == int(reset)
    assert owner.optimizer_reset_work == {
        "optimizer_reset_attempts": int(reset),
        "optimizer_resets_completed": int(reset),
    }
    assert report["rehearsal"] is True and report["optimizer_reset_enabled"] is reset
    assert report["rehearsal_retained_source_examples"] == 2
    assert report["control_complete"] is True and report["app_committed_updates"] == 18
    assert len(wires) == 6 and report["summary"]["criteria"]["B_after_phase_success"] is False


@pytest.mark.parametrize("reset", [False, True])
def test_should_keep_one_delayed_budget_unique_exposure_ids_and_original_replay_lineage(
    controls, reset
):
    report, owner, _, _, _ = controls[reset]
    rows = report["delivery"]
    assert len(rows) == len({row["exposure_id"] for row in rows}) == 18
    assert [row["observed_at"] for row in rows] == list(range(0, 36, 2))
    assert all(row["prelabel_updates"] == 0 for row in rows)
    assert [row["phase"] for row in rows] == ["after_A"] * 6 + ["after_B_rehearsal"] * 12
    receipts = [row["receipt"] for row in rows]
    assert [row["update_number"] for row in receipts] == list(range(1, 19))
    assert [row["arrived_at"] for row in receipts] == list(range(1, 36, 2))
    assert all(receipt["arrived_at"] == receipt["applied_at"] for receipt in receipts)
    replay = [row for row in rows if row.get("exposure_kind") == "replay"]
    assert len(replay) == report["replay_examples"] == report["replay_exposures_recorded"] == 6
    assert [row["source_example_id"] for row in replay] == ["train-A-0", "train-A-1"] * 3
    assert [row["replay_source_exposure_id"] for row in replay] == [
        "after_A-pass0-train-A-0",
        "after_A-pass0-train-A-1",
    ] * 3
    assert all(
        row["permissions"] == {"training": True, "replay": True, "evaluation": False}
        for row in rows[:6]
    )
    assert all(
        row["permissions"]["training"] is True and row["permissions"]["evaluation"] is False
        for row in replay
    )
    assert report["native_work"] == owner.work and report["inbox_stopped"] is False
    json.dumps(report, allow_nan=False)


def test_should_score_only_original_development_cases_after_all_training_and_keep_labels_off_wire(
    controls,
):
    development = [example for example in sequence_examples() if example.role == "outer_selection"]
    for report, _, _, wires, _ in controls.values():
        assert [wire["state"] for wire in wires] == [
            example.request.query for example in development
        ]
        for wire in wires:
            assert set(wire) == {"model", "state", "questions"}
            assert set(wire["questions"]["decision"]) == {"type", "instructions", "criteria"}
        assert all(
            row["phase"] == "after_rehearsal_B" and row["role"] == "outer_selection"
            for row in report["readouts"]
        )
        assert report["summary"]["recovery"] == "not_tested"


def rehearsal_rows():
    rows = cold_rows()
    for row in rows:
        row["phase"] = "after_rehearsal_B"
        if row["context"] == "A":
            row.update(correct=True, nll=-log(0.8))
    return rows


@pytest.mark.parametrize("reset", [False, True])
def test_should_use_generic_endpoint_contrasts_and_keep_all_context_metrics_for_both_arms(
    reference, reset
):
    rows = rehearsal_rows()
    original = deepcopy(reference)
    summary = runner.summarize_rehearsal(rows, reference, reset_optimizer=reset)
    assert reference == original
    assert summary["baseline_reused"] == reference["baseline_summary"]
    assert summary["after_A_reused"] == reference["after_A_summary"]
    assert summary["continuous_no_replay_after_B"] == reference["sequential_summary"]
    assert all(value["examples"] == 2 for value in summary["by_context"].values())
    contrasts = summary["contrasts"]
    assert set(contrasts) == {
        "endpoint_minus_baseline",
        "endpoint_minus_warm_no_replay",
        "B_acquisition",
    }
    assert contrasts["endpoint_minus_baseline"]["B"]["accuracy"] == 1.0
    assert contrasts["B_acquisition"]["nll_mean"] == pytest.approx(log(0.1) - log(0.9))
    assert summary["criteria"] == {
        "B_after_phase_success": True,
        "A_retained": True,
        "anchors_preserved": True,
    }
    assert summary["recovery"] == "not_tested"
    summary["baseline_reused"]["A"]["accuracy"] = 0.25
    assert reference == original


def test_should_keep_infinite_nll_and_anchor_drift_as_failed_or_unavailable_evidence(reference):
    rows = rehearsal_rows()
    rows[2].update(nll=None, nll_status="infinite")
    for row in rows:
        if row["context"] == "anchor":
            row["nll"] = reference["baseline_summary"]["anchor"]["nll_mean"] + 0.11
    summary = runner.summarize_rehearsal(rows, reference, reset_optimizer=False)
    assert summary["by_context"]["B"]["examples"] == 2
    assert summary["by_context"]["B"]["nll_mean"] is None
    assert summary["contrasts"]["B_acquisition"]["nll_mean"] is None
    assert summary["contrasts"]["B_acquisition"]["nll_status"] == "nonfinite_comparison_unavailable"
    assert summary["criteria"]["anchors_preserved"] is False
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("fault", ["first_replay", "after_replay_commit", "second_readout"])
def test_should_keep_extra_replay_charges_and_stop_without_later_work_after_paid_failure(
    reference, fault
):
    owner = ResetLearner([], fail_at=8 if fault == "first_replay" else None)
    predictions: list[None] = []

    def predict(wire):
        predictions.append(None)
        if fault == "second_readout" and len(predictions) == 2:
            raise RuntimeError("controlled unreturned rehearsal readout")
        return native_reply(wire)

    def progress(event):
        if (
            fault == "after_replay_commit"
            and event == "training_finished"
            and owner.work["committed_updates"] == 8
        ):
            raise RuntimeError("controlled postreplay commit stop")

    report: dict[str, Any] = {}
    with pytest.raises((RuntimeError, TrainingUpdateFailed)):
        runner.run_optimizer_control(
            owner, predict, reference, report, progress, rehearsal=True, reset_optimizer=False
        )
    attempts, commits, replay_commits, replay_recorded, prediction_count = {
        "first_replay": (8, 7, 0, 1, 0),
        "after_replay_commit": (8, 8, 1, 1, 0),
        "second_readout": (18, 18, 6, 6, 2),
    }[fault]
    assert len(owner.training) == attempts and owner.work["committed_updates"] == commits
    assert report["app_committed_updates"] == commits
    assert sum("receipt" in row for row in report["delivery"]) == commits
    assert report["replay_examples"] == replay_commits
    assert report["replay_exposures_recorded"] == replay_recorded
    assert len(predictions) == prediction_count
    assert report["control_complete"] is False and "summary" not in report
    assert owner.optimizer_reset_work["optimizer_reset_attempts"] == 0
    if fault == "after_replay_commit":
        assert report["delivery"][-1]["receipt"]["update_number"] == 8
        assert report["delivery"][-1]["training"] == owner.last_update
    elif fault == "second_readout":
        assert "result" in report["readouts"][0]
        assert "response_json" not in report["readouts"][1]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "fault", ["consent", "receipt", "source_id", "receipt_time", "receipt_identity"]
)
def test_should_refuse_unqualified_old_source_evidence_after_a_before_any_b_or_replay(
    reference, fault
):
    owner = ResetLearner([])
    report: dict[str, Any] = {}
    predictions: list[None] = []

    def predict(wire):
        predictions.append(None)
        return native_reply(wire)

    def progress(event):
        if event == "before_optimizer_reset":
            first = report["delivery"][0]
            if fault == "consent":
                first["permissions"]["replay"] = False
            elif fault == "receipt":
                first.pop("receipt")
            elif fault == "receipt_time":
                first["receipt"]["applied_at"] += 1
            elif fault == "receipt_identity":
                first["receipt"]["sample_id"] = "replaced-paid-source"
            else:
                first["source_example_id"] = "train-B-0"

    with pytest.raises(ValueError):
        runner.run_optimizer_control(
            owner, predict, reference, report, progress, rehearsal=True, reset_optimizer=False
        )
    assert (
        len(owner.training)
        == owner.work["committed_updates"]
        == report["app_committed_updates"]
        == 6
    )
    assert len(report["delivery"]) == 6 and predictions == []
    assert report["replay_examples"] == report["replay_exposures_recorded"] == 0
    assert report["control_complete"] is False


@pytest.mark.parametrize("fault", ["final", "probe", "development", "foreign_label"])
def test_should_refuse_protected_or_changed_old_source_tuple_before_replay_work(
    reference, monkeypatch, fault
):
    original = runner._deliver_phase

    def deliver(learner, inbox, clock, examples, phase, context, report, progress, **kwargs):
        if context == "B":
            selected = deepcopy(
                tuple(
                    example
                    for example in examples
                    if example.role == "train" and example.context == "A"
                )
            )
            if fault == "foreign_label":
                object.__setattr__(selected[0], "target_id", "billing")
            else:
                object.__setattr__(
                    selected[0],
                    "role",
                    {
                        "final": "final_test",
                        "probe": "probe_train",
                        "development": "outer_selection",
                    }[fault],
                )
            kwargs["rehearsal_examples"] = selected
        return original(learner, inbox, clock, examples, phase, context, report, progress, **kwargs)

    monkeypatch.setattr(runner, "_deliver_phase", deliver)
    owner = ResetLearner([])
    report: dict[str, Any] = {}
    with pytest.raises(ValueError):
        runner.run_optimizer_control(
            owner,
            native_reply,
            reference,
            report,
            lambda event: None,
            rehearsal=True,
            reset_optimizer=False,
        )
    assert len(owner.training) == owner.work["committed_updates"] == 6
    assert report["readouts"] == [] and len(report["delivery"]) == 6
    assert report["replay_examples"] == report["replay_exposures_recorded"] == 0


@pytest.mark.parametrize(
    "options",
    [
        {"rehearsal": 1},
        {"rehearsal": None},
        {"rehearsal": True, "reset_optimizer": 0},
        {"reset_optimizer": "1"},
        {"rehearsal": False, "reset_optimizer": False},
    ],
)
def test_should_refuse_invalid_mode_options_before_corpus_runtime_or_learner_work(
    reference, monkeypatch, options
):
    calls: list[None] = []

    def forbidden(*args, **kwargs):
        calls.append(None)
        raise AssertionError("invalid flags reached corpus or runtime")

    monkeypatch.setattr(runner, "sequence_examples", forbidden)
    monkeypatch.setattr(runner, "ExperienceInbox", forbidden)
    owner = ResetLearner([])
    with pytest.raises(ValueError):
        runner.run_optimizer_control(owner, forbidden, reference, {}, lambda event: None, **options)
    assert calls == [] and owner.training == []


def test_should_refuse_granting_replay_permission_to_new_b_phase_before_any_delivery():
    with pytest.raises(ValueError):
        sequence._deliver_phase(
            None,
            None,
            None,
            sequence_examples(),
            "after_B_rehearsal",
            "B",
            {},
            lambda event: None,
            replay_permission=True,
        )


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


def test_should_parse_rehearsal_without_reset_and_bind_eighteen_update_loader_allowances_before_model_work(
    reference, tmp_path, monkeypatch
):
    reads: list[Any] = []
    loads: list[Any] = []

    def read(path, *, include_after_A=False):
        reads.append((path, include_after_A))
        return deepcopy(reference)

    def load(source, checkpoint, progress, **options):
        loads.append((source, checkpoint, options))
        raise RuntimeError("controlled premodel rehearsal stop")

    monkeypatch.setattr(runner, "load_reference", read)
    monkeypatch.setattr(runner, "_load", load)
    output = tmp_path / "selected.json"
    assert runner.main(cli_args(tmp_path, output) + ["--rehearsal", "--optimizer-reset", "0"]) == 1
    assert reads == [(tmp_path / "reference.json", True)]
    assert len(loads) == 1
    assert loads[0][:2] == (tmp_path / "source", tmp_path / "checkpoint")
    assert loads[0][2]["prediction_limit"] == 6
    assert loads[0][2]["forward_limit"] == 36
    assert loads[0][2]["token_limit"] == 18432
    report = json.loads(output.read_text())
    assert report["definition"] == "native_choice_rehearsal_v1"
    assert report["complete"] is False and report["model_load_attempts"] == 1
    assert "controlled premodel rehearsal stop" in report["error"]


@pytest.mark.parametrize("reset", ["2", "0"])
def test_should_refuse_invalid_or_unselected_cli_reset_arm_before_any_input_or_model_load(
    tmp_path, monkeypatch, reset
):
    calls: list[None] = []

    def forbidden(*args, **kwargs):
        calls.append(None)
        raise AssertionError("invalid CLI reset reached source or model")

    monkeypatch.setattr(runner, "load_reference", forbidden)
    monkeypatch.setattr(runner, "_load", forbidden)
    output = tmp_path / "refused.json"
    with pytest.raises(SystemExit) as error:
        runner.main(cli_args(tmp_path, output) + ["--optimizer-reset", reset])
    assert error.value.code == 2 and calls == [] and not output.exists()
