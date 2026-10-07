"""Operational failure behavior on complete small fabricated raw records."""

from copy import deepcopy
import json
from typing import Any

import pytest

from src.app.continual_findings_failures import _failed_check, _timeout_finding


def _fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    identity = {"byte_count": 10, "sha256": "f" * 64}
    preservation = {
        "actual_child_reader_or_forbidden_call_counts_terminally_unobserved": True,
        "claim_remains_so_completed_audit_does_not_grant_success": True,
        "hard_timeout_seconds": 240,
        "parent_elapsed_seconds": 240.01,
        "nonterminal_audit_elapsed_seconds": 232.5,
        "preserved_partial_bundle_files": {
            "outcome-costs.audit.json": identity,
            "outcome-costs.claim": identity,
        },
        "source_snapshots": {"old.py": {"snapshot": "old-source.py", "identity": identity}},
        "repair_rationale": "remove duplicate traversal while preserving complete bindings and cap",
    }
    timeout = {
        "completed_observations": [],
        "hard_timeout_seconds": 240,
        "parent_elapsed_seconds": 240.01,
    }
    records = {
        name: {"content": json.dumps(body), "identity": deepcopy(identity)}
        for name, body in {
            "preservation.json": preservation,
            "timeout.json": timeout,
            "partial-audit.json": {
                "status": "completed",
                "derivative_validation_elapsed_seconds": 232.5,
                "forbidden_calls": {"train": 0},
            },
            "partial.claim": {"pid": 10},
            "old-source.py": {"unused": "raw text retained"},
        }.items()
    }
    group = {
        "preservation": "preservation.json",
        "timeout": "timeout.json",
        "partial_parts": {
            "outcome-costs.audit.json": "partial-audit.json",
            "outcome-costs.claim": "partial.claim",
        },
        "repaired_validation": "later-validation.json",
    }
    return records, group


def _change(records: dict[str, Any], name: str, key: str, value: Any) -> None:
    body = json.loads(records[name]["content"])
    body[key] = value
    records[name]["content"] = json.dumps(body)


def test_should_keep_killed_child_counters_unobserved_despite_zero_nonterminal_audit_guards():
    records, group = _fixture()
    finding = _timeout_finding(records, group)
    assert finding["status"] == "timed_out_partial_not_completed"
    assert finding["terminal_child_reader_counts"] is None
    assert finding["terminal_child_scientific_guard_counts"] is None
    assert finding["subsequent_repeat_or_readback_in_failed_attempt"] == "not_executed"
    assert finding["later_complete_operations"] == "later-validation.json"


@pytest.mark.parametrize(
    "key",
    [
        "actual_child_reader_or_forbidden_call_counts_terminally_unobserved",
        "claim_remains_so_completed_audit_does_not_grant_success",
    ],
)
def test_should_reject_timeout_that_drops_its_unobserved_or_partial_scope(key):
    records, group = _fixture()
    _change(records, "preservation.json", key, False)
    with pytest.raises(ValueError, match="unobserved"):
        _timeout_finding(records, group)


@pytest.mark.parametrize(
    "key,value",
    [
        ("hard_timeout_seconds", 300),
        ("parent_elapsed_seconds", 232.5),
        ("completed_observations", [{"completed": True}]),
    ],
)
def test_should_reject_changed_timeout_caps_elapsed_or_completed_observations(key, value):
    records, group = _fixture()
    _change(records, "timeout.json", key, value)
    with pytest.raises(ValueError):
        _timeout_finding(records, group)


@pytest.mark.parametrize("path", ["partial-audit.json", "partial.claim", "old-source.py"])
def test_should_reject_missing_whole_partial_or_source_snapshot(path):
    records, group = _fixture()
    del records[path]
    with pytest.raises(KeyError):
        _timeout_finding(records, group)


def test_should_reject_drifted_partial_identity_or_nonterminal_elapsed():
    records, group = _fixture()
    records["partial.claim"]["identity"]["byte_count"] = 9
    with pytest.raises(ValueError, match="partial"):
        _timeout_finding(records, group)
    records, group = _fixture()
    _change(records, "partial-audit.json", "derivative_validation_elapsed_seconds", 240.01)
    with pytest.raises(ValueError, match="nonterminal"):
        _timeout_finding(records, group)


def test_should_preserve_failure_returncodes_and_reasons_without_new_experiment():
    records = {
        "check.json": {
            "content": json.dumps(
                {"commands": [{"returncode": 0}, {"returncode": 1}], "reason": "late source drift"}
            )
        }
    }
    finding = _failed_check(records, "check.json")
    assert finding["original_returncodes"] == [0, 1]
    assert finding["reason"] == "late source drift"
    assert "not_an_independent_experiment" in finding["scope"]


def test_should_not_label_a_successful_correctness_record_as_a_failure():
    with pytest.raises(ValueError, match="lacks"):
        _failed_check(
            {"check.json": {"content": '{"commands": [{"returncode": 0}]}'}}, "check.json"
        )


def test_should_preserve_failed_complete_derivation_without_inventing_child_counters():
    finding = _failed_check(
        {
            "derive.json": {
                "content": '{"returncode": 1, "stdout": "", "stderr": "schema mismatch"}'
            }
        },
        "derive.json",
    )
    assert finding["original_returncodes"] == [1]
    assert finding["status"] == "failed_operational_check_or_saved_input_derivation_preserved"
    assert "terminal_child_guard_counts" not in finding


def test_should_require_the_original_derivative_audit_field_without_ambiguous_fallback():
    records, group = _fixture()
    audit = json.loads(records["partial-audit.json"]["content"])
    audit["elapsed_seconds"] = audit.pop("derivative_validation_elapsed_seconds")
    records["partial-audit.json"]["content"] = json.dumps(audit)
    with pytest.raises(KeyError, match="derivative_validation_elapsed_seconds"):
        _timeout_finding(records, group)
