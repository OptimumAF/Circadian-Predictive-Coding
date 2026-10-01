"""Closed scored metadata/worker links; all scored values are fabricated."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pytest

from test_continual_confirmation_final_observation import _observation
from test_continual_confirmation_scoring import _seal_scoring, _forbid
from test_continual_confirmation_scoring_validation import (
    _fail_endpoints,
    fabricated_scored_json as fabricated_scored_json,
)
from src.app import continual_confirmation_scoring_execution as execution
from src.app.continual_confirmation_scoring_execution import scoring_execution_request
from src.app.continual_confirmation_scoring_manifest import fixed_scoring_manifest
from src.infra.continual_confirmation_scoring_bindings import (
    PRIOR_SOURCE_SHA256,
    ADDITIONAL_SOURCE_SHA256,
    OWN_SOURCE_PATHS,
)


@pytest.fixture(scope="module")
def reference_report() -> dict[str, Any]:
    # Saved metadata only; clean tests do not require ignored original bundles.
    # This is not a replacement for either actual source-bound complete reader.
    return json.loads(
        (Path(__file__).parent / "fixtures/p67_scoring_training_references.json").read_text()
    )


@pytest.fixture
def request_bindings(reference_report: dict[str, Any]) -> dict[str, Any]:
    sources = dict(PRIOR_SOURCE_SHA256) | dict(ADDITIONAL_SOURCE_SHA256)
    sources.update({name: "1" * 64 for name in OWN_SOURCE_PATHS})
    return {
        "manifest": fixed_scoring_manifest(),
        "reference_report": deepcopy(reference_report),
        "source_sha256": sources,
        "command": [
            "fixture-python",
            "-m",
            "scripts.run_p67_confirmation_scoring",
            "--worker",
            "--request-file",
            "fixture-request.json",
            "--scope-file",
            "fixture-scope.json",
        ],
        "environment": {
            "python_version": "fixture",
            "numpy_version": "fixture",
            "platform": "fixture",
            "processor": "fixture",
        },
    }


@pytest.fixture
def scored_request(request_bindings: dict[str, Any]) -> dict[str, Any]:
    return scoring_execution_request(started_utc="2026-10-01T00:00:00+00:00", **request_bindings)


@pytest.fixture
def worker_payload(
    fabricated_scored_json: dict[str, Any], scored_request: dict[str, Any]
) -> dict[str, Any]:
    return {
        "result": deepcopy(fabricated_scored_json),
        "request_sha256": execution.encoded_identity(scored_request)["sha256"],
        "reference_report_sha256": scored_request["reference_report_sha256"],
        "source_map_sha256": scored_request["source_map_sha256"],
        "observed_updates": deepcopy(
            scored_request["reference_report"]["bundles"][0]["observed_updates"]
        ),
        "final_observation": _observation(fabricated_scored_json),
        "process_rss": {
            "pid": 1,
            "start_bytes": 100,
            "peak_bytes": 100,
            "sample_count": 2,
            "interval_seconds": 0.005,
        },
        "worker_elapsed_seconds": 1.0,
    }


@pytest.fixture(autouse=True)
def seal_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    _seal_scoring(monkeypatch)


def test_should_bind_every_original_reference_analysis_source_command_environment_and_cap(
    request_bindings: dict[str, Any],
    scored_request: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "open", _forbid)
    before = deepcopy(scored_request)
    execution.verify_scoring_execution_request(scored_request, **request_bindings)
    assert scored_request == before
    assert (
        scored_request["manifest_sha256"]
        == "76cf873e5942a661bdb76e6fd7f28fc490fc6b8001e2ae0ccc4afe87063a4223"
    )
    assert scored_request["reference_report_sha256"] == execution.REFERENCE_REPORT_SHA256
    assert (
        len(scored_request["source_sha256"]) == 97
        and scored_request["outer_selection_scored"] is False
    )
    assert scored_request["summary"]["cells"] == 560 and scored_request["summary"]["pairs"] == 580
    assert scored_request["limits"] == {
        "max_optimizer_updates": 16000,
        "wall_limit_seconds": 600,
        "max_process_rss_bytes": 536870912,
        "rss_interval_seconds": 0.005,
    }


def test_should_independently_link_all_fake_worker_events_costs_resources_and_audit_fields(
    scored_request: dict[str, Any], worker_payload: dict[str, Any]
) -> None:
    request_sha = execution.encoded_identity(scored_request)["sha256"]
    work = execution.verify_scored_worker(worker_payload, scored_request, request_sha)
    assert work == scored_request["reference_report"]["bundles"][0]["work"]
    assert work["totals"]["executed_optimizer_updates"] == 15210
    assert work["totals"]["rejected_executed_replay_updates"] == 46
    result_sha = execution.encoded_identity(worker_payload["result"])["sha256"]
    audit = execution.scoring_audit(scored_request, request_sha, result_sha, worker_payload, 2.0)
    assert audit["status"] == "completed" and audit["work"] == work
    assert audit["final_observation"]["prediction_examples"] == 67200
    assert audit["final_observation"]["by_model_kind"] == {
        "backprop": 420,
        "pc": 450,
        "circadian": 810,
    }
    assert audit["observed_updates"]["by_model_kind"] == {
        "backprop": 3708,
        "pc": 3948,
        "circadian": 7554,
    }
    assert worker_payload["result"]["external_execution_verified"] is False


@pytest.mark.parametrize("field", ["reference_report", "source_sha256", "command", "environment"])
@pytest.mark.parametrize("kind", ["empty", "unknown", "type", "late"])
def test_should_refuse_changed_current_bindings_against_the_complete_request(
    scored_request: dict[str, Any], request_bindings: dict[str, Any], field: str, kind: str
) -> None:
    bindings = deepcopy(request_bindings)
    value = bindings[field]
    if kind == "empty":
        bindings[field] = {} if isinstance(value, dict) else []
    elif kind == "unknown":
        if isinstance(value, dict):
            value["unexpected"] = "0" * 64
        else:
            value.append("--unexpected")
    elif kind == "type":
        bindings[field] = None
    elif field == "reference_report":
        value["bundles"][-1]["work"]["by_seed"][-1]["cells"] += 1
    elif field == "source_sha256":
        value[OWN_SOURCE_PATHS[-1]] = "0" * 64
    elif field == "command":
        value[-1] = "changed-scope.json"
    else:
        value["numpy_version"] = "changed"
    with pytest.raises(ValueError):
        execution.verify_scoring_execution_request(scored_request, **bindings)


_REQUEST_DRIFT = [
    (("schema_id",), "other"),
    (("protocol_id",), "other"),
    (("manifest_sha256",), "0" * 64),
    (("analysis_contract_sha256",), "0" * 64),
    (("reference_report_sha256",), "0" * 64),
    (("source_map_sha256",), "0" * 64),
    (("summary", "cells"), 559),
    (("summary", "final_calls"), 1681),
    (("summary", "final_examples"), 67160),
    (("summary", "pairs"), 579),
    (("limits", "max_optimizer_updates"), 16001),
    (("limits", "wall_limit_seconds"), 601),
    (("limits", "max_process_rss_bytes"), 536870913),
    (("limits", "rss_interval_seconds"), 0.01),
    (("limits", "wall_limit_seconds"), 600.0),
    (("outer_selection_scored",), 0),
    (("outer_selection_scored",), True),
    (("complete_independent_final_required",), False),
    (("environment", "numpy_version"), 1),
    (("command", -1), "changed-scope.json"),
    (("manifest", "analysis_contract", "train_result_sha256"), "0" * 64),
]


def _set(payload: dict[str, Any], path: tuple[Any, ...], value: Any) -> None:
    row: Any = payload
    for key in path[:-1]:
        row = row[key]
    row[path[-1]] = value


@pytest.mark.parametrize(("path", "value"), _REQUEST_DRIFT)
def test_should_require_exact_request_schema_types_values_and_original_scientific_contract(
    scored_request: dict[str, Any],
    request_bindings: dict[str, Any],
    path: tuple[Any, ...],
    value: Any,
) -> None:
    _set(scored_request, path, value)
    with pytest.raises(ValueError):
        execution.verify_scoring_execution_request(scored_request, **request_bindings)


@pytest.mark.parametrize(
    "value", [None, 0, "bad", "2026-10-01T00:00:00", "2026-10-01T00:00:00+01:00"]
)
def test_should_refuse_non_utc_or_malformed_request_times(
    request_bindings: dict[str, Any], value: Any
) -> None:
    with pytest.raises(ValueError):
        scoring_execution_request(started_utc=value, **request_bindings)


@pytest.mark.parametrize("kind", ["missing", "extra", "float", "bool", "late", "authority"])
def test_should_require_the_exact_pre_final_complete_reference_report(
    reference_report: dict[str, Any], kind: str
) -> None:
    report = deepcopy(reference_report)
    if kind == "missing":
        report["bundles"].pop()
    elif kind == "extra":
        report["unexpected"] = False
    elif kind == "float":
        report["bundles"][-1]["observed_updates"]["executed_updates"] = 15210.0
    elif kind == "bool":
        report["final_release_authorized"] = 0
    elif kind == "late":
        report["bundles"][-1]["work"]["by_seed"][-1]["cells"] += 1
    else:
        report["final_release_authorized"] = True
    with pytest.raises(ValueError):
        execution.verify_reference_report(report)


_WORKER_DRIFT = [
    (("request_sha256",), "0" * 64),
    (("reference_report_sha256",), "0" * 64),
    (("source_map_sha256",), "0" * 64),
    (("observed_updates", "executed_updates"), 15209),
    (("observed_updates", "attempted_updates"), 15211),
    (("observed_updates", "executed_updates"), 15210.0),
    (("observed_updates", "by_model_kind", "backprop"), 3707),
    (("final_observation", "prediction_attempts"), 1679),
    (("final_observation", "source_events", -1, "example_count"), 39),
    (("final_observation", "prediction_events", -1, "model_kind"), "circadian"),
    (("final_observation", "prediction_events", -1, "result", "correct_count"), 41),
    (("process_rss", "peak_bytes"), 536870913),
    (("process_rss", "interval_seconds"), 0.01),
    (("process_rss", "sample_count"), 1),
    (("process_rss", "pid"), False),
    (("worker_elapsed_seconds",), 600.0),
    (("worker_elapsed_seconds",), -1.0),
    (("worker_elapsed_seconds",), True),
    (("worker_elapsed_seconds",), float("nan")),
    (("result", "cells", -1, "accuracy", "b_after_b"), 0.1),
    (("result", "training_after_evaluation", "training_artifact", "sha256"), "0" * 64),
]


@pytest.mark.parametrize(("path", "value"), _WORKER_DRIFT)
def test_should_reject_late_fake_worker_proof_endpoint_count_cost_type_and_resource_forgeries(
    scored_request: dict[str, Any],
    worker_payload: dict[str, Any],
    path: tuple[Any, ...],
    value: Any,
) -> None:
    _set(worker_payload, path, value)
    with pytest.raises(ValueError):
        execution.verify_scored_worker(
            worker_payload, scored_request, execution.encoded_identity(scored_request)["sha256"]
        )


@pytest.mark.parametrize("kind", ["missing", "extra", "list", "updates_extra", "rss_extra"])
def test_should_refuse_open_or_malformed_worker_envelopes(
    scored_request: dict[str, Any], worker_payload: dict[str, Any], kind: str
) -> None:
    payload: Any = worker_payload
    if kind == "missing":
        del payload["final_observation"]
    elif kind == "extra":
        payload["unexpected"] = False
    elif kind == "list":
        payload = [payload]
    elif kind == "updates_extra":
        payload["observed_updates"]["unexpected"] = 0
    else:
        payload["process_rss"]["unexpected"] = 0
    with pytest.raises(ValueError):
        execution.verify_scored_worker(
            payload, scored_request, execution.encoded_identity(scored_request)["sha256"]
        )


@pytest.mark.parametrize("kind", ["nonfinite_predictions", "numerical_prediction_error"])
@pytest.mark.parametrize("indices", [[0], [1679], list(range(1680))])
def test_should_accept_every_declared_numerical_null_without_dropping_calls_or_changing_costs(
    scored_request: dict[str, Any], worker_payload: dict[str, Any], kind: str, indices: list[int]
) -> None:
    _fail_endpoints(worker_payload["result"], indices, kind)
    worker_payload["final_observation"] = _observation(worker_payload["result"])
    work = execution.verify_scored_worker(
        worker_payload, scored_request, execution.encoded_identity(scored_request)["sha256"]
    )
    assert work["totals"]["executed_optimizer_updates"] == 15210
    assert worker_payload["final_observation"]["prediction_attempts"] == 1680
    assert worker_payload["result"]["totals"]["failed_endpoints"] == len(indices)


def test_should_refuse_a_partial_scoring_manifest_before_any_execution_binding() -> None:
    manifest = fixed_scoring_manifest()
    manifest = replace(manifest, train_manifest=replace(manifest.train_manifest, families=()))
    with pytest.raises(ValueError, match="frozen complete"):
        scoring_execution_request(
            manifest=manifest,
            reference_report={},
            source_sha256={},
            command=[],
            environment={},
            started_utc="2026-10-01T00:00:00+00:00",
        )
