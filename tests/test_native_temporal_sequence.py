"""Scalar native temporal orchestration; no tensors, models or numerical work."""

from copy import deepcopy
import json
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import run_open_jev_temporal_sequence as runner
from src.core.temporal_decision_sequence import temporal_examples
from tests.test_native_choice_sequence import RecordingLearner, native_reply


class TemporalLearner(RecordingLearner):
    """Opaque fake snapshots restore state, never permanent paid counters."""

    def __init__(self, timeline, *, fail_at=None, fail_capture=None, fail_restore=None):
        super().__init__(timeline, fail_at=fail_at)
        self.live_checkpoint = 0
        self.captures: list[Any] = []
        self.restores: list[Any] = []
        self.fail_capture, self.fail_restore = fail_capture, fail_restore

    def train_batch(self, features, targets):
        result = super().train_batch(features, targets)
        self.live_checkpoint = self.work["committed_updates"]
        return result

    def snapshot_state(self):
        self.timeline.append(("snapshot", self.live_checkpoint))
        self.captures.append(self.live_checkpoint)
        if len(self.captures) == self.fail_capture:
            raise RuntimeError("controlled fake snapshot failure")
        return {"fixture_checkpoint": self.live_checkpoint}

    def restore_state(self, snapshot):
        self.timeline.append(("restore", snapshot["fixture_checkpoint"]))
        self.restores.append(deepcopy(snapshot))
        if len(self.restores) == self.fail_restore:
            raise RuntimeError("controlled fake restore failure")
        self.live_checkpoint = snapshot["fixture_checkpoint"]


def workflow_reply(wire, checkpoint):
    reply = native_reply(wire)
    identifiers = sorted(wire["questions"]["decision"]["criteria"])
    selected = identifiers[(checkpoint // 12) % len(identifiers)]
    reply["answers"]["decision"].update(
        choice=selected,
        probabilities={identity: 0.8 if identity == selected else 0.1 for identity in identifiers},
        confidence=0.7,
    )
    return reply


@pytest.fixture(scope="module")
def completed():
    timeline: list[Any] = []
    owner = TemporalLearner(timeline)
    wires: list[Any] = []
    events: list[Any] = []

    def predict(wire):
        assert len(owner.training) == owner.work["committed_updates"] == 36
        timeline.append(("predict", owner.live_checkpoint))
        wires.append(deepcopy(wire))
        return workflow_reply(wire, owner.live_checkpoint)

    report: dict[str, Any] = {}
    runner.run_sequence(owner, predict, report, events.append)
    return report, owner, timeline, wires, events


def test_should_complete_all_thirty_six_train_rows_before_eighty_current_policy_readouts(completed):
    report, owner, timeline, wires, _ = completed
    expected = [
        row
        for phase in ("after_A", "after_B", "after_return")
        for exposure in range(4)
        for row in temporal_examples(phase, "train", exposure=exposure)
    ]
    assert [(request, target) for request, target in owner.training] == [
        (row.request, row.target_id) for row in expected
    ]
    assert len(wires) == len(report["readouts"]) == 80
    first_prediction = next(index for index, event in enumerate(timeline) if event[0] == "predict")
    assert sum(event[0] == "train" for event in timeline[:first_prediction]) == 36
    assert report["sequence_complete"] is True and report["app_committed_updates"] == 36
    assert report["native_work"] == dict.fromkeys(
        (
            "training_attempts",
            "forward_calls",
            "backward_calls",
            "optimizer_steps",
            "committed_updates",
        ),
        36,
    )
    assert report["inbox_stopped"] is False and report["logical_tick"] == 72
    json.dumps(report, allow_nan=False)


def test_should_bind_each_delayed_label_to_its_observation_policy_and_keep_wires_unlabeled(
    completed,
):
    report, owner, _, wires, _ = completed
    rows = report["delivery"]
    assert len(rows) == len({row["exposure_id"] for row in rows}) == 36
    assert [row["policy_version"] for row in rows] == ["v1"] * 12 + ["v2"] * 24
    assert [row["phase"] for row in rows] == ["after_A"] * 12 + ["after_B"] * 12 + [
        "after_return"
    ] * 12
    assert [row["observed_at"] for row in rows] == list(range(0, 72, 2))
    assert [row["receipt"]["arrived_at"] for row in rows] == list(range(1, 72, 2))
    assert [row["receipt"]["update_number"] for row in rows] == list(range(1, 37))
    assert all(
        row["prelabel_updates"] == 0 and row["source_group"] == "train-token" for row in rows
    )
    assert all(row["receipt"]["arrived_at"] == row["receipt"]["applied_at"] for row in rows)
    assert [target for _, target in owner.training[:12]] == ["route", "retire", "inspect"] * 4
    assert [target for _, target in owner.training[24:]] == ["inspect", "defer", "inspect"] * 4
    for wire in wires:
        assert set(wire) == {"model", "state", "questions"}
        question = wire["questions"]["decision"]
        assert set(question) == {"type", "instructions", "criteria"}
        assert question["instructions"] == "Choose the ticket workflow tool."
        assert len(question["criteria"]) == 3
        assert not {"target", "role", "phase", "policy_version", "obsolete_target_id"}.intersection(
            wire
        )


def test_should_capture_four_bounded_states_and_restore_controls_without_renewing_paid_history(
    completed,
):
    report, owner, _, _, _ = completed
    assert owner.captures == [0, 12, 24, 36]
    assert [row["fixture_checkpoint"] for row in owner.restores] == [
        0,
        0,
        12,
        0,
        24,
        0,
        12,
        36,
        0,
        12,
        36,
    ]
    assert owner.live_checkpoint == 36 and owner.work["committed_updates"] == 36
    assert report["snapshot_work"] == {
        "capture_attempts": 4,
        "captures_completed": 4,
        "restore_attempts": 11,
        "restores_completed": 11,
        "peak_snapshots": 4,
        "retained_numeric_bytes": None,
        "released_snapshots": 4,
    }
    assert [row["source_updates"] for row in report["checkpoints"]] == [
        {"learned": 0, "unchanged": 0},
        {"learned": 12, "unchanged": 0},
        {"learned": 24, "unchanged": 0, "frozen_A": 12},
        {"learned": 36, "unchanged": 0, "frozen_A": 12},
    ]
    for checkpoint in report["checkpoints"]:
        for control in checkpoint["source_updates"]:
            evaluation = checkpoint[control]
            assert evaluation["complete"] is True and len(evaluation["cases"]) == 4
            assert evaluation["work"]["prediction_calls_validated"] == 8
            assert evaluation["candidate_order_control"]["all_equal"] is True
            assert evaluation["learning_active"] is None
            for row in evaluation["cases"]:
                assert row["policy_version"] == (
                    "v1" if checkpoint["phase"] in ("baseline", "after_A") else "v2"
                )
    points = {row["phase"]: row for row in report["checkpoints"]}
    for phase, row in points.items():
        for control in ("unchanged", "frozen_A"):
            if control in row:
                delta = report["contrasts"]["by_phase"][phase]["learned_minus_" + control]
                assert (
                    delta["accuracy"]
                    == row["learned"]["metrics"]["accuracy"] - row[control]["metrics"]["accuracy"]
                )


@pytest.mark.parametrize(
    "fault",
    ["postcommit", "train", "capture", "after_capture", "restore", "readout", "changed_wire"],
)
def test_should_preserve_paid_prefixes_and_stop_without_retry_at_native_boundaries(fault):
    owner = TemporalLearner(
        [],
        fail_at=14 if fault == "train" else None,
        fail_capture=3 if fault == "capture" else None,
        fail_restore=2 if fault == "restore" else None,
    )
    predictions: list[None] = []
    report: dict[str, Any] = {}

    def predict(wire):
        predictions.append(None)
        if fault == "readout" and len(predictions) == 2:
            raise RuntimeError("controlled unreturned temporal prediction")
        result = workflow_reply(wire, owner.live_checkpoint)
        if fault == "changed_wire":
            wire["state"] = "changed callback input"
        return result

    def progress(event):
        if fault == "postcommit" and event == "training_finished":
            raise RuntimeError("controlled stop after paid TRAIN receipt")
        if (
            fault == "after_capture"
            and event == "snapshot_captured"
            and report["snapshot_work"]["captures_completed"] == 4
        ):
            raise RuntimeError("controlled stop after paid fourth snapshot")

    with pytest.raises((RuntimeError, ValueError)):
        runner.run_sequence(owner, predict, report, progress)
    trained, committed, capture_attempts, captures, restore_attempts, restores, prediction_calls = {
        "postcommit": (1, 1, 1, 1, 0, 0, 0),
        "train": (14, 13, 2, 2, 0, 0, 0),
        "capture": (24, 24, 3, 2, 0, 0, 0),
        "after_capture": (36, 36, 4, 4, 0, 0, 0),
        "restore": (36, 36, 4, 4, 2, 1, 8),
        "readout": (36, 36, 4, 4, 1, 1, 2),
        "changed_wire": (36, 36, 4, 4, 1, 1, 1),
    }[fault]
    assert len(owner.training) == trained and report["app_committed_updates"] == committed
    assert report["native_work"]["training_attempts"] == trained
    assert report["native_work"]["committed_updates"] == committed
    assert sum("receipt" in row for row in report["delivery"]) == committed
    work = report["snapshot_work"]
    assert work["capture_attempts"] == capture_attempts and work["captures_completed"] == captures
    assert work["restore_attempts"] == restore_attempts and work["restores_completed"] == restores
    assert work["released_snapshots"] == captures
    assert len(predictions) == prediction_calls and report["sequence_complete"] is False
    assert "contrasts" not in report
    if fault == "restore":
        assert report["checkpoints"][0]["learned"]["complete"] is True
    if fault == "readout":
        assert "result" in report["readouts"][0] and "response_json" not in report["readouts"][1]
    if fault == "changed_wire":
        assert "response_json" in report["readouts"][0]
        assert report["readouts"][0]["request"]["query"] == "invoice please"
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "fault", ["receipt", "latest_training", "native_work", "boolean_work", "policy"]
)
def test_should_refuse_changed_preceding_evidence_before_another_train_update(fault):
    owner = TemporalLearner([])
    report: dict[str, Any] = {}

    def progress(event):
        if event == "training_finished":
            if fault == "receipt":
                report["delivery"][0]["receipt"]["update_number"] = 99
            elif fault == "latest_training":
                report["delivery"][0]["training"]["loss"] = 99.0
            elif fault == "native_work":
                owner._work["forward_calls"] = -1
            elif fault == "boolean_work":
                owner._work["forward_calls"] = True
            else:
                report["delivery"][0]["policy_version"] = "v2"

    def forbidden_prediction(wire):
        raise AssertionError("unexpected prediction")

    with pytest.raises(ValueError):
        runner.run_sequence(owner, forbidden_prediction, report, progress)
    assert len(owner.training) == report["app_committed_updates"] == 1
    assert owner.work["committed_updates"] == 1
    assert report["snapshot_work"]["capture_attempts"] == 1
    assert report["readouts"] == [] and report["sequence_complete"] is False


@pytest.mark.parametrize("boundary", ["first_restore", "final_restore"])
def test_should_recheck_paid_counters_after_restore_callbacks_before_scoring_or_completion(
    boundary,
):
    owner = TemporalLearner([])
    report: dict[str, Any] = {}
    predictions: list[None] = []

    def predict(wire):
        predictions.append(None)
        return workflow_reply(wire, owner.live_checkpoint)

    def progress(event):
        target = 1 if boundary == "first_restore" else 11
        if event == "snapshot_restored" and report["snapshot_work"]["restores_completed"] == target:
            owner._work["forward_calls"] = -1

    with pytest.raises(ValueError):
        runner.run_sequence(owner, predict, report, progress)
    assert len(owner.training) == report["app_committed_updates"] == 36
    assert len(predictions) == (0 if boundary == "first_restore" else 80)
    assert report["snapshot_work"]["restores_completed"] == (
        1 if boundary == "first_restore" else 11
    )
    assert report["snapshot_work"]["released_snapshots"] == 4
    assert report["sequence_complete"] is False and "contrasts" not in report
    if boundary == "final_restore":
        assert all(
            checkpoint[control]["complete"]
            for checkpoint in report["checkpoints"]
            for control in checkpoint["source_updates"]
        )


@pytest.mark.parametrize("fault", ["role", "policy", "source_group", "order"])
def test_should_refuse_malformed_temporal_corpus_before_inbox_or_snapshot(monkeypatch, fault):
    original = runner.temporal_examples

    def malformed(phase, role, *, exposure=0):
        rows = deepcopy(original(phase, role, exposure=exposure))
        if phase == "after_A" and role == "train" and exposure == 0:
            if fault == "role":
                object.__setattr__(rows[0], "role", "final_test")
            elif fault == "policy":
                object.__setattr__(rows[0], "policy_version", "v2")
            elif fault == "source_group":
                object.__setattr__(rows[0], "source_group", "development-please")
            else:
                rows = tuple(reversed(rows))
        return rows

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid corpus reached paid temporal work")

    monkeypatch.setattr(runner, "temporal_examples", malformed)
    monkeypatch.setattr(runner, "ExperienceInbox", forbidden)
    owner = TemporalLearner([])
    with pytest.raises(ValueError):
        runner.run_sequence(owner, forbidden, {}, lambda event: None)
    assert owner.training == owner.captures == []


def test_should_refuse_development_case_as_native_train_without_relaxing_temporal_validation():
    with pytest.raises(ValueError):
        runner._validate_train(temporal_examples("after_B", "outer_selection")[0])


def test_should_refuse_metadata_only_oversized_snapshot_before_training(monkeypatch):
    class NumericMetadata:
        def numel(self):
            return runner.SNAPSHOT_BYTES + 1

        def element_size(self):
            return 1

    owner = TemporalLearner([])
    monkeypatch.setattr(owner, "snapshot_state", lambda: SimpleNamespace(buffer=NumericMetadata()))
    report: dict[str, Any] = {}
    with pytest.raises(ValueError, match="storage"):
        runner.run_sequence(owner, lambda wire: None, report, lambda event: None)
    assert owner.training == [] and report["app_committed_updates"] == 0
    assert (
        report["snapshot_work"]["capture_attempts"]
        == report["snapshot_work"]["captures_completed"]
        == 1
    )
    assert report["snapshot_work"]["retained_numeric_bytes"] == runner.SNAPSHOT_BYTES + 1
    assert report["snapshot_work"]["released_snapshots"] == 1


def test_should_refuse_existing_cli_output_before_corpus_or_model_loading(tmp_path, monkeypatch):
    output = tmp_path / "keep.json"
    output.write_bytes(b"previous native temporal report")

    def forbidden(*args, **kwargs):
        raise AssertionError("existing output reached native work")

    monkeypatch.setattr(runner, "_training", forbidden)
    monkeypatch.setattr(runner, "_load", forbidden)
    with pytest.raises(SystemExit) as error:
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
    assert error.value.code == 2 and output.read_bytes() == b"previous native temporal report"
