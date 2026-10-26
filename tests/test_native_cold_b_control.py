"""Cold-B control boundaries using scalar records and recording fakes only.

No Torch, upstream model, optimizer, RNG, tensor or native inference is run.
The reused reference is synthetic and explicitly bound to temporary JSON bytes.
"""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from math import log
from typing import Any

import pytest

from scripts import run_open_jev_b_control as runner
from src.core.learner_ports import TrainingUpdateFailed
from src.core.native_choice_sequence import sequence_examples
from tests.test_native_choice_sequence import RecordingLearner, native_reply, oracle_readouts


def reference_data():
    corpus = sequence_examples()
    rows = oracle_readouts(corpus)
    examples = {example.example_id: example for example in corpus}
    for row in rows:
        example = examples[row["example_id"]]
        row.update(
            role=example.role,
            request=asdict(example.request),
            target=example.target_id,
            source_group=example.source_group,
        )
    from scripts.run_open_jev_sequence import summarize_sequence

    summary = summarize_sequence(rows)
    weights = {
        "head.weight": {
            "elements": 2,
            "before_sha256": "a" * 64,
            "after_sha256": "a" * 64,
            "max_absolute_delta": 0.0,
            "l2_delta": 0.0,
        },
        "backbone.lora_A.weight": {
            "elements": 1,
            "before_sha256": "b" * 64,
            "after_sha256": "b" * 64,
            "max_absolute_delta": 0.0,
            "l2_delta": 0.0,
        },
    }
    return {
        "source_sha256": "c" * 64,
        "baseline_readouts": [deepcopy(row) for row in rows if row["phase"] == "baseline"],
        "baseline_summary": summary["by_phase"]["baseline"],
        "sequential_summary": summary["by_phase"]["after_B"],
        "sequential_B_acquisition": summary["contrasts"]["phase_deltas"]["B_acquisition"],
        "baseline_weights": weights,
        "initial_rng_sha256": "d" * 64,
    }


def cold_rows():
    rows = []
    for example in sequence_examples():
        if example.role == "outer_selection":
            probability = {"A": 0.2, "B": 0.9, "anchor": 0.79}[example.context]
            rows.append(
                dict(
                    phase="after_cold_B",
                    context=example.context,
                    example_id=example.example_id,
                    correct=example.context != "A",
                    nll=-log(probability),
                    nll_status="finite",
                    role=example.role,
                    request=asdict(example.request),
                    target=example.target_id,
                    source_group=example.source_group,
                )
            )
    return rows


@pytest.fixture(scope="module")
def reference():
    return reference_data()


@pytest.fixture(scope="module")
def full_control(reference):
    timeline: list[Any] = []
    owner = RecordingLearner(timeline)
    wires: list[Any] = []

    def predict(wire):
        timeline.append(("predict", wire["state"]))
        wires.append(deepcopy(wire))
        reply = native_reply(wire)
        # A valid adverse endpoint must not cancel any declared exposure/readout.
        reply["answers"]["decision"].update(
            choice="billing", probabilities={"billing": 0.8, "password": 0.1, "shipping": 0.1}
        )
        return reply

    report: dict[str, Any] = {}
    runner.run_b_control(owner, predict, reference, report, lambda event: None)
    return report, owner, timeline, wires


def test_should_train_only_two_b_templates_three_times_before_six_development_readouts(
    full_control,
):
    report, owner, timeline, wires = full_control
    assert [item[0] for item in timeline] == ["train"] * 6 + ["predict"] * 6
    assert [(request.query, target) for request, target in owner.training] == [
        ("Case marker: neldax", "password"),
        ("Intake code neldax. Route this request.", "password"),
    ] * 3
    development = [item for item in sequence_examples() if item.role == "outer_selection"]
    assert [wire["state"] for wire in wires] == [item.request.query for item in development]
    assert len(report["readouts"]) == 6
    assert all(row["phase"] == "after_cold_B" for row in report["readouts"])
    assert report["summary"]["criteria"]["B_after_phase_success"] is False
    assert report["control_complete"] is True


def test_should_keep_native_requests_label_free_and_original_development_roles(full_control):
    report, _, _, wires = full_control
    for wire in wires:
        assert set(wire) == {"model", "state", "questions"}
        assert set(wire["questions"]["decision"]) == {"type", "instructions", "criteria"}
        assert list(wire["questions"]["decision"]["criteria"]) == [
            "billing",
            "password",
            "shipping",
        ]
    assert all(row["role"] == "outer_selection" for row in report["readouts"])
    assert {row["context"] for row in report["readouts"]} == {"A", "B", "anchor"}
    json.dumps(report, allow_nan=False)


def test_should_keep_six_unique_delayed_receipts_without_prior_a_updates(full_control):
    report, owner, _, _ = full_control
    delivery = report["delivery"]
    assert len(delivery) == len({row["exposure_id"] for row in delivery}) == 6
    assert [row["observed_at"] for row in delivery] == [0, 2, 4, 6, 8, 10]
    assert all(row["prelabel_updates"] == 0 for row in delivery)
    assert {row["source_example_id"] for row in delivery} == {"train-B-0", "train-B-1"}
    receipts = [row["receipt"] for row in delivery]
    assert [row["update_number"] for row in receipts] == list(range(1, 7))
    assert [row["arrived_at"] for row in receipts] == [1, 3, 5, 7, 9, 11]
    assert all(row["arrived_at"] == row["applied_at"] for row in receipts)
    assert report["app_committed_updates"] == owner.work["committed_updates"] == 6
    assert report["native_work"] == owner.work
    assert report["inbox_stopped"] is False
    assert report["reused_baseline"] == reference_data()["baseline_readouts"]


def test_should_compare_all_contexts_to_reused_baseline_and_sequential_endpoint(reference):
    rows = cold_rows()
    original_rows, original_reference = deepcopy(rows), deepcopy(reference)
    summary = runner.summarize_b_control(rows, reference)
    assert rows == original_rows and reference == original_reference
    assert summary["baseline_reused"] == reference["baseline_summary"]
    assert summary["sequential_after_B"] == reference["sequential_summary"]
    assert set(summary["by_context"]) == {"A", "B", "anchor"}
    assert all(group["examples"] == 2 for group in summary["by_context"].values())
    assert summary["by_context"]["B"]["accuracy"] == 1.0
    assert summary["by_context"]["B"]["nll_mean"] == pytest.approx(-log(0.9))
    contrasts = summary["contrasts"]
    assert contrasts["cold_minus_baseline"]["B"]["accuracy"] == 1.0
    assert contrasts["cold_minus_baseline"]["B"]["nll_mean"] == pytest.approx(log(0.1) - log(0.9))
    assert contrasts["cold_minus_sequential_after_B"]["B"]["accuracy"] == 0.0
    assert contrasts["cold_minus_sequential_after_B"]["B"]["nll_mean"] == pytest.approx(
        log(0.8) - log(0.9)
    )
    assert contrasts["sequential_B_acquisition"] == reference["sequential_B_acquisition"]
    assert summary["criteria"] == {"B_after_phase_success": True, "anchors_preserved": True}
    assert not any("retention" in key for key in summary["criteria"])
    summary["baseline_reused"]["B"]["accuracy"] = 0.25
    assert reference == original_reference


def test_should_keep_infinite_nll_and_unavailable_delta_without_dropping_a_case(reference):
    rows = cold_rows()
    rows[2].update(nll=None, nll_status="infinite")
    summary = runner.summarize_b_control(rows, reference)
    assert summary["by_context"]["B"] == {
        "examples": 2,
        "accuracy": 1.0,
        "nll_mean": None,
        "nll_status": "infinite",
    }
    for key in ("cold_minus_baseline", "cold_minus_sequential_after_B"):
        assert summary["contrasts"][key]["B"]["nll_mean"] is None
        assert summary["contrasts"][key]["B"]["nll_status"] == "nonfinite_comparison_unavailable"
    json.dumps(summary, allow_nan=False)


def test_should_require_correct_anchor_decisions_and_no_excess_nll_drift(reference):
    rows = cold_rows()
    for row in rows:
        if row["context"] == "anchor":
            row["nll"] = reference["baseline_summary"]["anchor"]["nll_mean"] + 0.11
    assert runner.summarize_b_control(rows, reference)["criteria"]["anchors_preserved"] is False
    for row in rows:
        if row["context"] == "anchor":
            row.update(nll=-log(0.9), correct=False)
    assert runner.summarize_b_control(rows, reference)["criteria"]["anchors_preserved"] is False


@pytest.mark.parametrize("fault", ["partial", "duplicate", "context", "nan", "bool"])
def test_should_refuse_partial_or_unvalidated_cold_summary_rows(reference, fault):
    rows = cold_rows()
    if fault == "partial":
        rows.pop()
    elif fault == "duplicate":
        rows[1]["example_id"] = rows[0]["example_id"]
    elif fault == "context":
        rows[0]["context"] = "B"
    elif fault == "nan":
        rows[0]["nll"] = float("nan")
    else:
        rows[0]["correct"] = 1
    with pytest.raises(ValueError):
        runner.summarize_b_control(rows, reference)


def test_should_bind_all_initial_weight_names_and_unchanged_hashes(reference):
    observed = deepcopy(reference["baseline_weights"])
    runner.validate_initial_weights(observed, reference)
    assert observed == reference["baseline_weights"]


@pytest.mark.parametrize("fault", ["missing", "extra", "before", "after"])
def test_should_refuse_unmatched_or_changed_initial_weights(reference, fault):
    observed = deepcopy(reference["baseline_weights"])
    if fault == "missing":
        observed.pop("head.weight")
    elif fault == "extra":
        observed["unbound.weight"] = deepcopy(observed["head.weight"])
    elif fault == "before":
        observed["head.weight"]["before_sha256"] = "e" * 64
    else:
        observed["head.weight"]["after_sha256"] = "e" * 64
    with pytest.raises(ValueError):
        runner.validate_initial_weights(observed, reference)


def test_should_keep_two_commits_when_third_b_update_fails_without_any_readout(reference):
    owner = RecordingLearner([], fail_at=3)
    calls: list[None] = []

    def predict(wire):
        calls.append(None)
        return native_reply(wire)

    report: dict[str, Any] = {}
    with pytest.raises(TrainingUpdateFailed):
        runner.run_b_control(owner, predict, reference, report, lambda event: None)
    assert calls == [] and report["readouts"] == []
    assert len(owner.training) == len(report["delivery"]) == 3
    assert owner.work["committed_updates"] == report["app_committed_updates"] == 2
    assert [row["receipt"]["update_number"] for row in report["delivery"][:2]] == [1, 2]
    assert "receipt" not in report["delivery"][2]
    assert report["delivery"][2]["error"]
    json.dumps(report, allow_nan=False)


def test_should_retain_paid_receipt_before_postcommit_progress_refusal(reference):
    owner = RecordingLearner([])

    def progress(event):
        if event == "training_finished":
            raise RuntimeError("controlled postcommit cold-B stop")

    report: dict[str, Any] = {}
    with pytest.raises(RuntimeError, match="controlled postcommit cold-B stop"):
        runner.run_b_control(owner, native_reply, reference, report, progress)
    assert len(owner.training) == 1 and report["readouts"] == []
    assert report["app_committed_updates"] == report["native_work"]["committed_updates"] == 1
    assert report["delivery"][0]["receipt"]["update_number"] == 1
    assert report["delivery"][0]["training"] == owner.last_update


def test_should_keep_all_six_training_receipts_and_first_readout_when_second_predictor_raises(
    reference,
):
    owner = RecordingLearner([])
    calls: list[None] = []

    def predict(wire):
        calls.append(None)
        if len(calls) == 2:
            raise RuntimeError("controlled second cold readout failure")
        return native_reply(wire)

    report: dict[str, Any] = {}
    with pytest.raises(RuntimeError, match="controlled second cold readout failure"):
        runner.run_b_control(owner, predict, reference, report, lambda event: None)
    assert len(owner.training) == len(report["delivery"]) == 6
    assert len(calls) == len(report["readouts"]) == 2
    assert "result" in report["readouts"][0]
    assert "response_json" not in report["readouts"][1]
    assert report["app_committed_updates"] == 6
    assert "summary" not in report
    json.dumps(report, allow_nan=False)


def test_should_revalidate_corpus_before_constructing_cold_control_runtime(monkeypatch, reference):
    changed = deepcopy(sequence_examples())
    object.__setattr__(changed[0], "role", "final_test")
    monkeypatch.setattr(runner, "sequence_examples", lambda: changed)
    calls: list[None] = []

    def forbidden(*args, **kwargs):
        calls.append(None)
        raise AssertionError("invalid corpus reached cold control work")

    monkeypatch.setattr(runner, "ExperienceInbox", forbidden)
    owner = RecordingLearner([])
    with pytest.raises(ValueError):
        runner.run_b_control(owner, forbidden, reference, {}, lambda event: None)
    assert calls == [] and owner.training == []


@pytest.mark.parametrize(
    "counter",
    [
        "training_attempts",
        "forward_calls",
        "backward_calls",
        "optimizer_steps",
        "committed_updates",
    ],
)
def test_should_refuse_any_inherited_native_work_before_inbox_prediction_or_training(
    monkeypatch, reference, counter
):
    owner = RecordingLearner([])
    owner._work[counter] = 1
    calls: list[None] = []

    def forbidden(*args, **kwargs):
        calls.append(None)
        raise AssertionError("nonfresh native learner reached cold control work")

    monkeypatch.setattr(runner, "ExperienceInbox", forbidden)
    with pytest.raises(ValueError):
        runner.run_b_control(owner, forbidden, reference, {}, lambda event: None)
    assert calls == [] and owner.training == []


def reference_payload():
    reference = reference_data()
    return dict(
        definition="native_choice_recurring_v1",
        complete=True,
        sequence_complete=True,
        app_committed_updates=18,
        examples=[asdict(example) for example in sequence_examples()],
        readouts=reference["baseline_readouts"],
        weight_checkpoints=[
            dict(after_updates=0, model_training=False, weights=reference["baseline_weights"])
        ],
        events=[dict(event="prediction_attempt", rng_state_sha256=reference["initial_rng_sha256"])],
        summary=dict(
            by_phase=dict(
                baseline=reference["baseline_summary"], after_B=reference["sequential_summary"]
            ),
            contrasts=dict(phase_deltas=dict(B_acquisition=reference["sequential_B_acquisition"])),
        ),
    )


def write_reference(tmp_path, monkeypatch, payload):
    path = tmp_path / "synthetic-reference.json"
    body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
    path.write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()
    monkeypatch.setattr(runner, "REFERENCE_SHA256", digest)
    return path, body, digest


def test_should_load_owned_hash_bound_reference_without_replaying_its_predictions(
    tmp_path, monkeypatch
):
    payload = reference_payload()
    path, body, digest = write_reference(tmp_path, monkeypatch, payload)
    result = runner.load_reference(path)
    assert result["source_sha256"] == digest
    assert result["initial_rng_sha256"] == "d" * 64
    assert result["baseline_readouts"] == json.loads(body)["readouts"]
    assert result["baseline_weights"] == payload["weight_checkpoints"][0]["weights"]
    assert result["baseline_summary"] == payload["summary"]["by_phase"]["baseline"]
    assert result["sequential_summary"] == payload["summary"]["by_phase"]["after_B"]
    assert (
        result["sequential_B_acquisition"]
        == payload["summary"]["contrasts"]["phase_deltas"]["B_acquisition"]
    )
    result["baseline_readouts"][0]["target"] = "replacement"
    assert path.read_bytes() == body
    assert payload["readouts"][0]["target"] != "replacement"


@pytest.mark.parametrize(
    "fault", ["hash", "oversize", "incomplete", "updates", "corpus", "readout", "order", "warm"]
)
def test_should_refuse_unbound_or_incompatible_saved_reference(tmp_path, monkeypatch, fault):
    payload = reference_payload()
    if fault == "incomplete":
        payload["complete"] = False
    elif fault == "updates":
        payload["app_committed_updates"] = 17
    elif fault == "corpus":
        payload["examples"][0]["target_id"] = "billing"
    elif fault == "readout":
        payload["readouts"][0]["request"]["query"] = "replacement query"
    elif fault == "order":
        request = payload["readouts"][0]["request"]
        request["allowed_candidates"] = tuple(reversed(request["allowed_candidates"]))
    elif fault == "warm":
        payload["weight_checkpoints"][0]["after_updates"] = 6
    path, body, _ = write_reference(tmp_path, monkeypatch, payload)
    if fault == "hash":
        path.write_bytes(body + b" ")
    elif fault == "oversize":
        huge = b" " * (runner.MAX_REFERENCE_BYTES + 1)
        path.write_bytes(huge)
        monkeypatch.setattr(runner, "REFERENCE_SHA256", hashlib.sha256(huge).hexdigest())
    with pytest.raises(ValueError):
        runner.load_reference(path)


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


def test_should_refuse_existing_output_before_reading_reference_or_loading_model(
    tmp_path, monkeypatch
):
    output = tmp_path / "existing.json"
    output.write_bytes(b"preserve existing evidence")
    calls: list[None] = []

    def forbidden(*args, **kwargs):
        calls.append(None)
        raise AssertionError("existing output reached input or model load")

    monkeypatch.setattr(runner, "load_reference", forbidden)
    monkeypatch.setattr(runner, "_load", forbidden)
    with pytest.raises(SystemExit) as error:
        runner.main(cli_args(tmp_path, output))
    assert error.value.code == 2 and calls == []
    assert output.read_bytes() == b"preserve existing evidence"


def test_should_publish_reference_refusal_without_attempting_native_model_load(
    tmp_path, monkeypatch
):
    calls: list[None] = []

    def reference_failure(path):
        raise ValueError("controlled mismatched reference")

    def forbidden(*args, **kwargs):
        calls.append(None)
        raise AssertionError("reference refusal reached model load")

    monkeypatch.setattr(runner, "load_reference", reference_failure)
    monkeypatch.setattr(runner, "_load", forbidden)
    output = tmp_path / "failed.json"
    assert runner.main(cli_args(tmp_path, output)) == 1
    report = json.loads(output.read_text())
    assert report["complete"] is False and report["model_load_attempts"] == 0
    assert "controlled mismatched reference" in report["error"]
    assert calls == []
