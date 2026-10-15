"""Exercise both real adapters through the bounded delayed-label smoke demo.

Inputs are a fixed seed and the demo's four-row stream. Assertions cover scheduling,
native diagnostic definitions and detached actor behavior, without ranking methods.
Two in-process runs and one CLI run are enough to check deterministic integration.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
from typing import cast

import pytest

from src.adapters.native_stream_demo import run_demo
from src.core.backprop_mlp import NUMPY_BACKPROP_LOSS_ID
from src.core.circadian_predictive_coding import NUMPY_CIRCADIAN_ENERGY_ID

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("backprop", "circadian")
DIAGNOSTIC_DEFINITIONS = {
    "backprop": NUMPY_BACKPROP_LOSS_ID,
    "circadian": NUMPY_CIRCADIAN_ENERGY_ID,
}
ReportPair = tuple[dict[str, object], dict[str, object]]


def native_dict(value: object) -> dict[str, object]:
    assert type(value) is dict
    return cast(dict[str, object], value)


def native_list(value: object) -> list[object]:
    assert type(value) is list
    return cast(list[object], value)


def finite_predictions(value: object) -> list[list[float]]:
    rows = native_list(value)
    assert len(rows) == 4
    predictions: list[list[float]] = []
    for row in rows:
        values = native_list(row)
        assert len(values) == 1
        probability = values[0]
        assert type(probability) is float and math.isfinite(probability)
        assert 0.0 <= probability <= 1.0
        predictions.append([probability])
    return predictions


def method_report(report: dict[str, object], name: str) -> dict[str, object]:
    methods = [native_dict(item) for item in native_list(report["methods"])]
    matches = [method for method in methods if method["method"] == name]
    assert len(matches) == 1
    return matches[0]


@pytest.fixture(scope="module")
def repeated_reports() -> ReportPair:
    return run_demo(23), run_demo(23)


def test_demo_reports_both_methods_and_repeats_deterministically(
    repeated_reports: ReportPair,
) -> None:
    report, repeated = repeated_reports
    assert type(report["schema_version"]) is int and report["schema_version"] == 1
    assert report["protocol_id"] == "native_adapter_stream_v1"
    assert type(report["seed"]) is int and report["seed"] == 23
    assert report["scope"] == "integration_smoke"
    assert [native_dict(item)["method"] for item in native_list(report["methods"])] == list(METHODS)
    assert report == repeated
    # Why: a strict JSON encoder rejects NaN/Infinity hidden outside checked fields too.
    assert json.loads(json.dumps(report, allow_nan=False)) == report


@pytest.mark.parametrize("method", METHODS)
def test_labels_apply_once_after_two_ticks_and_actor_stays_frozen(
    repeated_reports: ReportPair, method: str
) -> None:
    report = method_report(repeated_reports[0], method)
    assert report["experiment_state"] == {
        "format_version": 1,
        "completed_updates": 4,
        "next_tick": 6,
    }
    assert report["model_state"] == {
        "actor_version": "actor-0",
        "candidate_version": "candidate-0",
        "snapshot_format_version": 1 if method == "backprop" else 2,
    }
    events = [native_dict(item) for item in native_list(report["events"])]
    assert len(events) == 6
    assert [event["tick"] for event in events] == list(range(6))
    assert all(type(event["tick"]) is int for event in events)
    assert [event["observed_sample_id"] for event in events] == [
        "s0",
        "s1",
        "s2",
        "s3",
        None,
        None,
    ]
    assert [event["applied_sample_ids"] for event in events] == [
        [],
        [],
        ["s0"],
        ["s1"],
        ["s2"],
        ["s3"],
    ]
    assert all(event["actor_version"] == "actor-0" for event in events)
    actor_predictions = [finite_predictions(event["actor_predictions"]) for event in events]
    assert all(prediction == actor_predictions[0] for prediction in actor_predictions)
    initial = finite_predictions(report["initial_candidate_predictions"])
    final = finite_predictions(report["final_candidate_predictions"])
    assert initial != final


@pytest.mark.parametrize("method", METHODS)
def test_diagnostics_keep_native_definitions_and_finite_values(
    repeated_reports: ReportPair, method: str
) -> None:
    report = method_report(repeated_reports[0], method)
    diagnostics = [native_dict(item) for item in native_list(report["diagnostics"])]
    assert len(diagnostics) == 4
    for diagnostic in diagnostics:
        assert diagnostic["definition"] == DIAGNOSTIC_DEFINITIONS[method]
        value = diagnostic["value"]
        assert type(value) is float and math.isfinite(value)


@pytest.mark.parametrize("seed", [True, -1, 1.5, 2**32])
def test_demo_refuses_invalid_seed_before_running(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        run_demo(cast(int, seed))


def test_cli_emits_the_same_json_report(repeated_reports: ReportPair) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.run_native_stream_demo", "--seed", "23"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert completed.stderr == ""
    assert len(completed.stdout.encode("utf-8")) <= 65536
    assert native_dict(cast(object, json.loads(completed.stdout))) == repeated_reports[0]
