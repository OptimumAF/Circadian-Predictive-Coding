"""Saved-source diagnosis without model construction or further fitting."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.diagnose_ticket_training as diagnosis

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "artifacts/runs/native-recurring-tickets-20261023/bp-actual.json"
)


def saved():
    return json.loads(SOURCE.read_bytes())


@pytest.fixture(scope="module")
def completed():
    source = saved()
    before = deepcopy(source)
    rng = deepcopy(np.random.get_state())
    result = diagnosis.diagnose(source)
    assert source == before
    current = np.random.get_state()
    assert type(rng) is tuple and type(current) is tuple
    assert rng[0] == current[0] and np.array_equal(rng[1], current[1]) and rng[2:] == current[2:]
    return result


def test_should_cover_every_phase_and_exposure_without_native_work(completed):
    r = completed
    assert r["complete"] and r["role"] == "offline_train_diagnosis_only"
    assert (
        r["training_updates"]
        == r["model_initializations"]
        == r["random_draws"]
        == r["probe_updates"]
        == r["final_examples"]
        == 0
    )
    assert r["work"] == dict(projection_calls=70, projection_rows=90, nominal_affine_terms=3510)
    assert [p["source_updates"] for p in r["checkpoints"]] == [0, 4, 8, 12]
    assert [p["exposed_rows"] for p in r["checkpoints"]] == [0, 3, 6, 6]
    for point in r["checkpoints"]:
        assert len(point["rows"]) == 6
        assert [(x["concept"], x["phase"]) for x in point["rows"]] == [
            (c, p) for c in ("invoice", "parcel") for p in ("sealed", "ready", "staged")
        ]
        assert len(point["global_probability_margins"]["per_class"]) == 6
        assert all(type(row["numerically_ambiguous"]) is bool for row in point["rows"])
    assert r["source_primary_summary"] == saved()["summary"]


def test_should_bind_reference_to_every_observed_native_raw_score(completed):
    parity = completed["native_parity"]
    assert parity["rows"] == 66 and parity["scores"] == 330
    assert parity["max_absolute_raw_error"] <= 1e-6 and parity["absolute_tolerance"] == 1e-6
    assert completed["alignment"] == dict(
        teacher_public_equal_rows=6, conflicting_feature_targets=0
    )


def test_should_reduce_literal_logits_with_stable_ce_and_ties():
    probs, ce, raw, summary = diagnosis._summarize_logits(
        np.array([[0.0, np.log(2), 0.0], [1000.0, 1000.0, 1000.0]]), np.array([[1.0], [2.0]])
    )
    assert probs[0].tolist() == pytest.approx([0.25, 0.5, 0.25])
    assert ce.tolist() == pytest.approx([np.log(2), np.log(3)])
    assert raw.tolist() == pytest.approx([np.log(2), 0])
    assert summary["target_margins"] == pytest.approx([0.25, 0])
    assert summary["argmax_correct_rows"] == 1


@pytest.mark.parametrize(
    "fault", ["method", "pending", "counter", "dtype", "shape", "nan", "rounded", "trace", "work"]
)
def test_should_refuse_incompatible_source_before_interpreting_fit(fault):
    source = saved()
    if fault == "method":
        source["method"] = "pc"
    elif fault == "pending":
        source["probe_updates"] = 1
    elif fault == "counter":
        source["checkpoints"][1]["state"]["completed_updates"] = 5
    elif fault in ("dtype", "shape", "nan", "rounded"):
        value = source["checkpoints"][1]["state"]["head_state"]["bias_output"]
        if fault == "dtype":
            value["dtype"] = "torch.float64"
        elif fault == "shape":
            value["shape"] = [6]
        elif fault == "nan":
            value["values"][0][0] = float("nan")
        else:
            value["values"][0][0] = 0.123456789
    elif fault == "trace":
        source["episodes"].pop()
    else:
        source["training_data"][0]["role"] = "final_test"
    with pytest.raises(ValueError):
        diagnosis.diagnose(source)


def test_should_refuse_replaced_parameters_when_paid_native_evidence_disagrees():
    source = saved()
    source["checkpoints"][0]["state"]["head_state"]["bias_output"]["values"][0][0] = 8.0
    work = dict(projection_calls=0, projection_rows=0, nominal_affine_terms=0)
    with pytest.raises(ValueError, match="parity"):
        diagnosis.diagnose(source, work=work)
    assert work == dict(projection_calls=1, projection_rows=1, nominal_affine_terms=39)


def test_should_refuse_corrupt_native_record():
    source = saved()
    source["episodes"][0]["episode"]["steps"][0]["result"]["scores"][0]["raw_score"] += 0.1
    with pytest.raises(ValueError, match="parity"):
        diagnosis.diagnose(source)


def test_should_refuse_final_teacher_before_projection(monkeypatch):
    cases = diagnosis.training_cases()
    object.__setattr__(cases[0], "role", "final_test")
    monkeypatch.setattr(diagnosis, "training_cases", lambda: cases)
    with pytest.raises(ValueError, match="TRAIN teacher"):
        diagnosis.diagnose(saved())


def test_should_refuse_feature_alignment_change(monkeypatch):
    original = diagnosis.encode_decision

    def changed(request, spec):
        values, visits = original(request, spec)
        if request.query.endswith("please"):
            values[0, 0] = 1 - values[0, 0]
        return values, visits

    monkeypatch.setattr(diagnosis, "encode_decision", changed)
    with pytest.raises(ValueError, match="alignment"):
        diagnosis.diagnose(saved())


def test_should_refuse_conflicting_teacher_feature_targets(monkeypatch):
    cases = list(diagnosis.training_cases())
    cases[1] = replace(cases[1], request=replace(cases[1].request, query=cases[0].request.query))
    monkeypatch.setattr(diagnosis, "training_cases", lambda: tuple(cases))
    with pytest.raises(ValueError, match="conflicting"):
        diagnosis.diagnose(saved())


def test_should_reserve_fresh_output_before_diagnosis(tmp_path, monkeypatch):
    path = tmp_path / "existing.json"
    path.write_text("preserve")
    monkeypatch.setattr(
        diagnosis, "diagnose", lambda *a, **k: pytest.fail("occupied output reached projection")
    )
    with pytest.raises(FileExistsError):
        diagnosis.main(["--input", str(SOURCE), "--output", str(path)])
    assert path.read_text() == "preserve"


def test_should_preserve_failed_cli_work_and_source(tmp_path):
    path = tmp_path / "bad.json"
    source = saved()
    source["checkpoints"][0]["state"]["head_state"]["bias_output"]["values"][0][0] = 8.0
    path.write_text(json.dumps(source))
    before = path.read_bytes()
    output = tmp_path / "failure.json"
    assert diagnosis.main(["--input", str(path), "--output", str(output)]) == 1
    result = json.loads(output.read_bytes())
    assert not result["complete"] and result["work"]["projection_rows"] == 1
    assert path.read_bytes() == before
