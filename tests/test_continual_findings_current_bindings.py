"""Whole current source/input/request/marker binding behavior."""

import json

import pytest

from src.infra import continual_findings_current_bindings as module
from findings_publication_fixtures import make_repository, write_body


def test_should_bind_every_prior_source_and_current_bundle_without_reader_dispatch(
    tmp_path, monkeypatch
):
    fixture = make_repository(tmp_path, monkeypatch)
    snapshot = module.current_findings_bindings(tmp_path, fixture["scope"])
    assert len(snapshot["source_sha256"]) == 150
    assert len(snapshot["current_upstream_bundles"]) == 4
    assert snapshot["preserved_timeout_claim_present"] is True
    assert snapshot["preserved_failed_synthesis_directory_empty"] is True


@pytest.mark.parametrize(
    "target",
    [
        "old/source-0.py",
        "saved.json",
        "pure.json",
        "matrix/confirmation-matrix.md",
        "outcome_costs/outcome-costs.audit.json",
    ],
)
def test_should_reject_drift_in_any_whole_prior_source_input_or_bundle(
    target, tmp_path, monkeypatch
):
    fixture = make_repository(tmp_path, monkeypatch)
    (tmp_path / target).write_bytes(b"changed\n")
    with pytest.raises(ValueError):
        module.current_findings_bindings(tmp_path, fixture["scope"])


@pytest.mark.parametrize(
    "marker", ["matrix/confirmation-matrix.claim", "outcome_costs/outcome-costs.failure.json"]
)
def test_should_reject_upstream_claim_or_failure_without_granting_completion(
    marker, tmp_path, monkeypatch
):
    fixture = make_repository(tmp_path, monkeypatch)
    (tmp_path / marker).write_text("incomplete", encoding="utf-8")
    with pytest.raises(ValueError, match="failure/claim"):
        module.current_findings_bindings(tmp_path, fixture["scope"])


def test_should_reject_erased_historical_failed_claim_or_populated_failed_directory(
    tmp_path, monkeypatch
):
    fixture = make_repository(tmp_path, monkeypatch)
    claim = tmp_path / "artifacts/runs/p610-outcome-costs/outcome-costs.claim"
    saved = claim.read_bytes()
    claim.unlink()
    with pytest.raises(ValueError, match="preserved timeout"):
        module.current_findings_bindings(tmp_path, fixture["scope"])
    claim.write_bytes(saved)
    (tmp_path / "artifacts/runs/p612-complete-findings-pure/result.json").write_text(
        "false success", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="failed synthesis"):
        module.current_findings_bindings(tmp_path, fixture["scope"])


@pytest.mark.parametrize(
    "timestamp", ["2026-10-01T00:00:00", "2026-10-01T00:00:00-07:00", "malformed"]
)
def test_should_reject_non_UTC_or_malformed_request_time(timestamp, tmp_path, monkeypatch):
    fixture = make_repository(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        module.findings_request(tmp_path, tmp_path / "output", fixture["scope"], timestamp)


def test_should_reject_late_own_source_and_request_environment_drift(tmp_path, monkeypatch):
    fixture = make_repository(tmp_path, monkeypatch)
    output = tmp_path / "output"
    output.mkdir()
    paths = {"request": output / "findings.request.json"}
    request = module.findings_request(
        tmp_path, output, fixture["scope"], "2026-10-01T00:00:00+00:00"
    )
    write_body(paths["request"], request)
    assert module.checked_findings_request(tmp_path, paths, fixture["scope"]) == request
    own = tmp_path / module.OWN_SOURCES[0]
    old = own.read_bytes()
    own.write_bytes(b"new source\n")
    with pytest.raises(ValueError, match="current request"):
        module.checked_findings_request(tmp_path, paths, fixture["scope"])
    own.write_bytes(old)
    changed = json.loads(paths["request"].read_bytes())
    changed["bindings"]["environment"]["python_version"] = "other"
    write_body(paths["request"], changed)
    with pytest.raises(ValueError, match="current request"):
        module.checked_findings_request(tmp_path, paths, fixture["scope"])
