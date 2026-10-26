"""Saved v1 fit evidence remains separate from native learners and final data."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.diagnose_ticket_training as diagnosis

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "artifacts/runs/phase-matched-controllers-20261023/actual.json"
)


def saved():
    data = SOURCE.read_bytes()
    assert hashlib.sha256(data).hexdigest() == diagnosis.POLICY_STUDY_SHA
    return json.loads(data)


@pytest.fixture(scope="module")
def completed():
    source = saved()
    before, rng = deepcopy(source), deepcopy(np.random.get_state())
    result = diagnosis.diagnose_policy_study(source)
    assert source == before
    after = np.random.get_state()
    assert type(rng) is tuple and type(after) is tuple
    assert rng[0] == after[0] and np.array_equal(rng[1], after[1]) and rng[2:] == after[2:]
    return result


def test_should_read_all_fixed_states_with_original_paid_parity(completed):
    assert completed["complete"]
    points = completed["points"]
    assert [p["mode"] for p in points] == [
        "source",
        "no_learning",
        "periodic",
        "drift",
        "budget_aware",
        "fitted",
    ]
    assert [p["new_updates"] for p in points] == [0, 0, 2, 4, 0, 4]
    assert [p["opportunity"] for p in points] == [0, 4, 4, 4, 4, 4]
    assert [p["native_updates"] for p in points] == [96, 96, 98, 100, 96, 100]
    assert [p["native_rows"] for p in points] == [576, 576, 588, 600, 576, 600]
    assert completed["parameter_array_bytes"] == 1152
    assert completed["work"] == dict(
        projection_calls=60, projection_rows=90, nominal_affine_terms=3510
    )
    assert all(
        p["rows"] == 9
        and p["scores"] == 45
        and p["max_absolute_raw_error"] <= 1e-6
        and p["absolute_tolerance"] == 1e-6
        for p in completed["native_parity"]
    )
    assert all(
        completed[k] == 0
        for k in (
            "training_updates",
            "probe_updates",
            "final_examples",
            "model_initializations",
            "random_draws",
        )
    )
    for p in points:
        rows = p["v1_teacher_rows"]
        assert [(r["concept"], r["phase"]) for r in rows] == [
            (c, s) for c in ("invoice", "password") for s in ("sealed", "ready", "staged")
        ]
        assert all(r["new_train_exposures"] == p["new_updates"] for r in rows)
        assert all(np.isfinite(r["allowed_ce"]) and np.isfinite(r["global_ce"]) for r in rows)
        assert rows[1]["route_vs_inspect_raw_margin"] is not None
    # No desired accuracy, margin sign or improvement is an acceptance criterion.


@pytest.mark.parametrize(
    "fault",
    [
        "complete",
        "hash",
        "fit",
        "cases",
        "readouts",
        "role",
        "phase",
        "clock",
        "counter",
        "binding",
        "packet",
        "dtype",
        "shape",
        "nan",
        "rounded",
        "policy",
    ],
)
def test_should_refuse_corrupt_binding_before_reference_projection(fault):
    source = saved()
    result = source["cases"][3]["result"]
    checkpoint = result["policy_checkpoints"][1]["checkpoint"]
    head = checkpoint["state"]["head_state"]["bias_output"]
    if fault == "complete":
        source["complete"] = False
    elif fault in ("hash", "fit"):
        source["protocol"]["source_sha256" if fault == "hash" else "fitted_report_sha256"] = (
            "replaced"
        )
    elif fault == "cases":
        source["cases"].pop()
    elif fault == "readouts":
        source["development_readouts"].reverse()
    elif fault == "role":
        result["training_stream"]["role"] = "final_test"
    elif fault == "phase":
        result["policy_checkpoints"][1]["policy_version"] = "v2"
    elif fault == "clock":
        checkpoint["source_updates"] = True
    elif fault == "counter":
        checkpoint["state"]["training_examples"] += 1
    elif fault == "binding":
        source["development_readouts"][2]["source_updates_by_opportunity"]["4"] = 0
    elif fault == "packet":
        result["training_stream"]["first_targets"][0][0] += 1.0
    elif fault == "dtype":
        head["dtype"] = "torch.float64"
    elif fault == "shape":
        head["shape"] = [6]
    elif fault == "nan":
        head["values"][0][0] = float("nan")
    elif fault == "rounded":
        head["values"][0][0] = 0.123456789
    else:
        checkpoint["state"]["policy"]["seed"] = 18
    work = dict(projection_calls=0, projection_rows=0, nominal_affine_terms=0)
    with pytest.raises(ValueError):
        diagnosis.diagnose_policy_study(source, work=work)
    assert work["projection_calls"] == 0


@pytest.mark.parametrize("fault", ["parameters", "scores"])
def test_should_refuse_disagreement_before_any_teacher_fit(fault, monkeypatch):
    source = saved()
    if fault == "parameters":
        source["cases"][2]["result"]["policy_checkpoints"][1]["checkpoint"]["state"]["head_state"][
            "bias_output"
        ]["values"][0][0] = 8.0
    else:
        source["development_readouts"][2]["episodes"][3]["episode"]["steps"][0]["result"]["scores"][
            0
        ]["raw_score"] += 0.1
    monkeypatch.setattr(
        diagnosis, "_policy_fit_rows", lambda *a: pytest.fail("fit before all parity")
    )
    work = dict(projection_calls=0, projection_rows=0, nominal_affine_terms=0)
    with pytest.raises(ValueError, match="parity"):
        diagnosis.diagnose_policy_study(source, work=work)
    assert work == dict(projection_calls=28, projection_rows=28, nominal_affine_terms=1092)


def test_should_keep_default_counters_strict():
    with pytest.raises(ValueError, match="counters"):
        diagnosis._parameters(saved()["cases"][0]["result"]["policy_checkpoints"][0]["checkpoint"])


def test_should_refuse_nontrain_v1_teacher_before_projection(monkeypatch):
    cases = diagnosis.training_cases(policy_version="v1")
    object.__setattr__(cases[0], "role", "probe_test")
    monkeypatch.setattr(diagnosis, "training_cases", lambda **k: cases)
    with pytest.raises(ValueError, match="TRAIN teacher"):
        diagnosis.diagnose_policy_study(saved())


def test_should_protect_cli_and_sealed_source(tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    before = SOURCE.read_bytes()
    monkeypatch.setattr(
        diagnosis, "diagnose_policy_study", lambda *a, **k: dict(complete=True, stub=True)
    )
    args = ["--policy-comparison", "--input", str(SOURCE), "--output", str(output)]
    assert diagnosis.main(args) == 0 and json.loads(output.read_bytes())["stub"]
    with pytest.raises(FileExistsError):
        diagnosis.main(args)
    bad = tmp_path / "replaced.json"
    bad.write_bytes(before + b" ")
    failed = tmp_path / "failure.json"
    assert (
        diagnosis.main(["--policy-comparison", "--input", str(bad), "--output", str(failed)]) == 1
    )
    assert "frozen original" in json.loads(failed.read_bytes())["error"]
    assert SOURCE.read_bytes() == before
    with pytest.raises(ValueError, match="bound"):
        diagnosis.main(["--input", str(SOURCE), "--output", str(tmp_path / "old.json")])
