"""Scalar saved-reply calibration oracles; no scientific files or models."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from math import log, sqrt
from typing import Any

import pytest

from scripts import summarize_native_consolidation as summary
from src.adapters.open_jev_choice import build_open_jev_choice, decode_open_jev_probabilities
from src.core.native_choice_sequence import INSTRUCTIONS, sequence_examples
from tests.test_native_choice_sequence import native_reply


PHASES = {
    "warm": "after_B",
    "reset": "after_optimizer_reset_B",
    "projection": "after_adapter_projection_B",
    "consolidation": "after_adapter_consolidation_B",
}
DEFINITIONS = {
    "warm": "native_choice_recurring_v1",
    "reset": "native_optimizer_reset_control_v1",
    "projection": "native_adapter_projection_control_v1",
    "consolidation": "native_adapter_consolidation_control_v1",
}


def readout(example, phase, *, outcome="correct"):
    wire = build_open_jev_choice(example.request, INSTRUCTIONS)
    raw = native_reply(wire)
    identities = [candidate.candidate_id for candidate in example.request.allowed_candidates]
    other = [identity for identity in identities if identity != example.target_id]
    if outcome == "correct":
        probabilities = {
            identity: 0.8 if identity == example.target_id else 0.1 for identity in identities
        }
        selected = example.target_id
    else:
        probabilities = {
            example.target_id: 0.0 if outcome == "zero" else 0.1,
            other[0]: 0.9 if outcome == "zero" else 0.8,
            other[1]: 0.1,
        }
        selected = other[0]
    raw["answers"]["decision"].update(
        choice=selected,
        probabilities=probabilities,
        confidence=(max(probabilities.values()) - 1 / 3) / (1 - 1 / 3),
    )
    body = json.dumps(raw, allow_nan=False, separators=(",", ":")).encode()
    decoded = decode_open_jev_probabilities(example.request, body)
    probability = probabilities[example.target_id]
    return {
        "phase": phase,
        "context": example.context,
        "role": "outer_selection",
        "request": json.loads(json.dumps(asdict(example.request))),
        "example_id": example.example_id,
        "source_group": example.source_group,
        "response": raw,
        "response_json": body.decode(),
        "result": json.loads(json.dumps(asdict(decoded))),
        "target": example.target_id,
        "correct": selected == example.target_id,
        "nll": None if probability == 0 else -log(probability),
        "nll_status": "infinite" if probability == 0 else "finite",
    }


def rank_receipt(rank):
    singular = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    return {
        "definition": "open_jev_owned_adapter_projection_v1"
        if rank == 4
        else "open_jev_owned_adapter_consolidation_v1",
        "effective_rank": rank,
        "modules": [
            {
                "module": "backbone.literal_layer",
                "allocated_rank": 8,
                "effective_rank": rank,
                "scaling": 2.0,
                "numerical_rank": 8,
                "singular_values": singular,
                "discarded_frobenius": sqrt(sum(value * value for value in singular[rank:])),
            }
        ],
    }


def reports():
    examples = sequence_examples()
    development = [example for example in examples if example.role == "outer_selection"]
    result: dict[str, Any] = {}
    for name, phase in PHASES.items():
        updates = 18 if name == "warm" else 12
        rows = [readout(example, phase) for example in development]
        if name == "warm":
            rows = [readout(example, "baseline") for example in development] + rows
        source = {
            "definition": DEFINITIONS[name],
            "complete": True,
            "sequence_complete" if name == "warm" else "control_complete": True,
            "app_committed_updates": updates,
            "examples": json.loads(json.dumps([asdict(example) for example in examples])),
            "readouts": rows,
            "native_work": {
                "training_attempts": updates,
                "forward_calls": updates + 6,
                "backward_calls": updates,
                "optimizer_steps": updates,
                "committed_updates": updates,
            },
            "outer_seconds": 1.25,
            "events": [],
            "fixture_source_note": "synthetic saved scalar observations, not native outcomes",
        }
        if name in ("projection", "consolidation"):
            source[name + "_receipt"] = rank_receipt(4 if name == "projection" else 8)
        if name == "warm":
            source["events"] = [
                {"observed_work": {"decision_forward_calls": 9}, "peak_rss_bytes": 128},
                {
                    "observed_work": {
                        "decision_forward_calls": 24,
                        "candidate_sequence_lengths": [2, 3],
                    },
                    "outer_peak_rss_bytes": 256,
                },
            ]
        result[name] = source
    return result


@pytest.fixture(scope="module")
def saved_reports():
    return reports()


@pytest.fixture(scope="module")
def completed(saved_reports):
    original = deepcopy(saved_reports)
    result = summary.summarize_native_reports(saved_reports)
    assert saved_reports == original
    return result


def test_should_match_hand_computed_brier_raw_nll_and_max_probability_ece(completed):
    assert completed["definition"] == "native_consolidation_calibration_summary_v1"
    assert completed["complete"] is True
    assert set(completed["arms"]) == {"unchanged", "warm", "reset", "projection", "consolidation"}
    for arm in completed["arms"].values():
        for metric in [arm["overall"], *arm["by_context"].values()]:
            assert metric["accuracy"] == metric["candidate_recall"] == 1.0
            assert metric["brier_mean"] == pytest.approx(0.06)
            assert metric["nll_mean"] == pytest.approx(-log(0.8))
            assert metric["ece_5bin"] == pytest.approx(0.2)
            bins = metric["calibration_bins"]
            assert len(bins) == 5
            assert [bucket["support"] for bucket in bins[:4]] == [0, 0, 0, 0]
            assert all(
                bucket["mean_confidence"] is bucket["accuracy"] is bucket["absolute_gap"] is None
                for bucket in bins[:4]
            )
            assert bins[4]["mean_confidence"] == pytest.approx(0.8)


def test_should_keep_all_contexts_original_criteria_and_detached_source_metadata(
    completed, saved_reports
):
    assert all(set(arm["by_context"]) == {"A", "B", "anchor"} for arm in completed["arms"].values())
    assert all(
        arm["criteria"]
        == {"B_after_phase_success": True, "A_retained": True, "anchors_preserved": True}
        for arm in completed["arms"].values()
    )
    changed = deepcopy(completed)
    changed["arms"]["warm"]["overall"]["accuracy"] = -1
    assert completed["arms"]["warm"]["overall"]["accuracy"] == 1.0
    assert saved_reports["warm"]["native_work"]["committed_updates"] == 18
    json.dumps(completed, allow_nan=False)


def test_should_report_signed_native_minus_control_deltas_without_reversing_lower_is_better_metrics(
    saved_reports,
):
    changed = deepcopy(saved_reports)
    for name in ("warm", "projection"):
        for index, row in enumerate(changed[name]["readouts"]):
            if row["phase"] == PHASES[name] and row["context"] == "B":
                example = next(
                    example
                    for example in sequence_examples()
                    if example.example_id == row["example_id"]
                )
                changed[name]["readouts"][index] = readout(example, PHASES[name], outcome="wrong")
    result = summary.summarize_native_reports(changed)
    warm = result["arms"]["warm"]["overall"]
    assert warm["accuracy"] == pytest.approx(2 / 3)
    assert warm["brier_mean"] == pytest.approx((4 * 0.06 + 2 * 1.46) / 6)
    assert warm["nll_mean"] == pytest.approx((-4 * log(0.8) - 2 * log(0.1)) / 6)
    assert result["contrasts"]["consolidation_minus_warm"]["overall"]["accuracy"] == pytest.approx(
        1 / 3
    )
    assert result["contrasts"]["consolidation_minus_warm"]["overall"]["nll_mean"] < 0
    assert result["contrasts"]["projection_minus_reset"]["overall"]["accuracy"] == pytest.approx(
        -1 / 3
    )
    assert result["contrasts"]["projection_minus_reset"]["overall"]["brier_mean"] > 0


def test_should_keep_zero_target_probability_infinite_nll_and_full_support(saved_reports):
    changed = deepcopy(saved_reports)
    for index, row in enumerate(changed["consolidation"]["readouts"]):
        if row["context"] == "B":
            example = next(
                example
                for example in sequence_examples()
                if example.example_id == row["example_id"]
            )
            changed["consolidation"]["readouts"][index] = readout(
                example, PHASES["consolidation"], outcome="zero"
            )
    result = summary.summarize_native_reports(changed)
    observed = result["arms"]["consolidation"]["overall"]
    assert observed["candidate_recall"] == 1.0 and observed["zero_target_probability_count"] == 2
    assert observed["nll_mean"] is None
    assert observed["nll_status"] == "infinite_due_to_zero_target_probability"
    assert sum(bucket["support"] for bucket in observed["calibration_bins"]) == 6
    assert result["contrasts"]["consolidation_minus_warm"]["overall"]["nll_mean"] is None
    assert (
        result["contrasts"]["consolidation_minus_warm"]["overall"]["nll_status"]
        == "nonfinite_comparison_unavailable"
    )
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "fault",
    [
        "role",
        "source_group",
        "request",
        "target",
        "correct",
        "nll",
        "capture",
        "probability_mass",
        "saved_result",
    ],
)
def test_should_refuse_unbound_or_stale_saved_observations(saved_reports, fault):
    changed = deepcopy(saved_reports)
    row = changed["reset"]["readouts"][0]
    if fault == "role":
        row["role"] = "final_test"
    elif fault == "source_group":
        row["source_group"] = "foreign-evaluation-group"
    elif fault == "request":
        row["request"]["query"] = "replaced input"
    elif fault == "target":
        row["target"] = "password" if row["target"] != "password" else "billing"
    elif fault == "correct":
        row["correct"] = False
    elif fault == "nll":
        row["nll"] = 0.0
    elif fault == "capture":
        row["response"]["usage"]["input_tokens"] += 1
    elif fault == "probability_mass":
        row["response"]["answers"]["decision"]["probabilities"]["billing"] = 0.9
        row["response_json"] = json.dumps(row["response"])
    else:
        row["result"]["selected_id"] = "unknown"
    before = deepcopy(changed)
    with pytest.raises(ValueError):
        summary.summarize_native_reports(changed)
    assert changed == before


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate_module",
        "allocated_rank",
        "effective_rank",
        "negative_spectrum",
        "unsorted_spectrum",
        "nonfinite",
        "numerical_rank",
        "tail",
    ],
)
def test_should_refuse_invalid_or_inconsistent_pretruncation_rank_receipts(saved_reports, fault):
    changed = deepcopy(saved_reports)
    receipt = changed["projection"]["projection_receipt"]
    module = receipt["modules"][0]
    if fault == "duplicate_module":
        receipt["modules"].append(deepcopy(module))
    elif fault == "allocated_rank":
        module["allocated_rank"] = 7
    elif fault == "effective_rank":
        module["effective_rank"] = 8
    elif fault == "negative_spectrum":
        module["singular_values"][-1] = -1.0
    elif fault == "unsorted_spectrum":
        module["singular_values"][0], module["singular_values"][1] = (
            module["singular_values"][1],
            module["singular_values"][0],
        )
    elif fault == "nonfinite":
        module["singular_values"][0] = float("inf")
    elif fault == "numerical_rank":
        module["numerical_rank"] = 9
    else:
        module["discarded_frobenius"] = 0.0
    with pytest.raises(ValueError):
        summary.summarize_native_reports(changed)


def test_should_preserve_pretruncation_spectrum_and_historical_work_without_claiming_installed_rank(
    completed, saved_reports
):
    projection = completed["rank_characterization"]["projection"]
    consolidation = completed["rank_characterization"]["consolidation"]
    assert projection["allocated_rank"] == consolidation["allocated_rank"] == 8
    assert (
        projection["requested_projection_rank"] == 4
        and consolidation["requested_projection_rank"] == 8
    )
    assert projection["modules"][0]["discarded_frobenius"] == pytest.approx(sqrt(30))
    assert consolidation["modules"][0]["discarded_frobenius"] == 0.0
    assert projection["numerical_rank_scope"] == "reported_float64_operator_before_truncation"
    assert projection["installed_float32_numerical_rank"] == "unmeasured"
    assert (
        completed["historical_source_work"]["warm"]["native_work"]
        == saved_reports["warm"]["native_work"]
    )
    history = completed["historical_source_work"]
    assert history["warm"]["observed_work"] == saved_reports["warm"]["events"][-1]["observed_work"]
    assert history["warm"]["peak_rss_bytes"] == 256
    assert history["reset"]["observed_work"] is history["reset"]["peak_rss_bytes"] is None
    assert completed["derived_work"] == {
        "model_initializations": 0,
        "training_updates": 0,
        "prediction_calls": 0,
        "random_draws": 0,
        "scored_rows": 30,
        "spectrum_modules": 2,
    }
    original = deepcopy(saved_reports)
    work = completed["historical_source_work"]["warm"]["native_work"]
    spectrum = completed["rank_characterization"]["projection"]["modules"][0]["singular_values"]
    lengths = history["warm"]["observed_work"]["candidate_sequence_lengths"]
    old_count, old_singular, old_length = work["committed_updates"], spectrum[0], lengths[0]
    try:
        work["committed_updates"], spectrum[0], lengths[0] = -1, -1, -1
        assert saved_reports == original
    finally:
        work["committed_updates"], spectrum[0], lengths[0] = old_count, old_singular, old_length


@pytest.mark.parametrize("fault", ["extra_arm", "boolean_updates"])
def test_should_refuse_wrong_arm_namespace_or_non_native_work_counts(saved_reports, fault):
    changed = deepcopy(saved_reports)
    if fault == "extra_arm":
        changed["selected_winner"] = deepcopy(changed["warm"])
    else:
        changed["warm"]["app_committed_updates"] = True
    with pytest.raises(ValueError):
        summary.summarize_native_reports(changed)


def write_inputs(tmp_path, monkeypatch, saved_reports):
    sources = deepcopy(summary.SOURCES)
    for name, spec in sources.items():
        body = json.dumps(saved_reports[name], allow_nan=False).encode()
        spec["path"] = name + ".json"
        spec["sha256"] = hashlib.sha256(body).hexdigest()
        (tmp_path / spec["path"]).write_bytes(body)
    monkeypatch.setattr(summary, "SOURCES", sources)
    return sources


def test_should_publish_pinned_synthetic_inputs_with_zero_model_derivation_and_original_bytes(
    saved_reports, tmp_path, monkeypatch, capsys
):
    sources = write_inputs(tmp_path, monkeypatch, saved_reports)
    monkeypatch.setattr(summary, "read_process_rss_bytes", lambda: None)
    original = {name: (tmp_path / spec["path"]).read_bytes() for name, spec in sources.items()}
    output = tmp_path / "summary.json"
    assert summary.main(["--runs-dir", str(tmp_path), "--output", str(output)]) == 0
    result = json.loads(output.read_bytes())
    assert result["complete"] is True and result["derived_work"]["model_initializations"] == 0
    assert result["derivation_boundary_rss_bytes"] is None
    assert set(result["source_reports"]) == set(sources)
    for name, spec in sources.items():
        assert result["source_reports"][name]["sha256"] == spec["sha256"]
        assert result["source_reports"][name]["bytes"] == len(original[name])
        assert (tmp_path / spec["path"]).read_bytes() == original[name]
    assert json.loads(capsys.readouterr().out)["complete"] is True


def test_should_refuse_existing_output_before_any_input_read(tmp_path, monkeypatch):
    output = tmp_path / "existing.json"
    output.write_bytes(b"keep original")

    def forbidden(*args, **kwargs):
        raise AssertionError("existing output reached input reads")

    with monkeypatch.context() as patch:
        patch.setattr(summary.Path, "open", forbidden)
        with pytest.raises(ValueError, match="fresh"):
            summary.main(["--runs-dir", str(tmp_path), "--output", str(output)])
    assert output.read_bytes() == b"keep original"


@pytest.mark.parametrize("fault", ["hash", "oversized"])
def test_should_refuse_changed_or_oversized_inputs_before_summary(
    saved_reports, tmp_path, monkeypatch, fault
):
    sources = write_inputs(tmp_path, monkeypatch, saved_reports)
    first = tmp_path / sources["warm"]["path"]
    first.write_bytes(
        first.read_bytes() + b" " if fault == "hash" else b" " * (summary.MAX_REPORT_BYTES + 1)
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("unbound input reached summary")

    monkeypatch.setattr(summary, "summarize_native_reports", forbidden)
    output = tmp_path / "not-created.json"
    with pytest.raises(ValueError):
        summary.main(["--runs-dir", str(tmp_path), "--output", str(output)])
    assert not output.exists()
