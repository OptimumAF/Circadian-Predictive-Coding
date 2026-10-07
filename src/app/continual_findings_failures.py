"""Preserve operational failures without inventing terminal child observations.

Inputs are complete bound UTF-8 records and their fixed catalog groups. Output
labels the original timeout/partial publication and failed correctness checks.
This pure module does not validate current files, run readers or score models.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from src.app.continual_confirmation_json import require, same_json


def _record(records: dict[str, Any], path: str) -> dict[str, Any]:
    body = json.loads(records[path]["content"])
    require(type(body) is dict, "failure record must be a complete object")
    return body


def _timeout_finding(records: dict[str, Any], group: dict[str, Any]) -> dict[str, Any]:
    preservation = _record(records, group["preservation"])
    timeout = _record(records, group["timeout"])
    require(
        preservation["actual_child_reader_or_forbidden_call_counts_terminally_unobserved"] is True
        and preservation["claim_remains_so_completed_audit_does_not_grant_success"] is True,
        "timeout must retain unobserved terminal counters and partial claim",
    )
    same_json(timeout["completed_observations"], [], "timeout has no completed observations")
    same_json(timeout["hard_timeout_seconds"], preservation["hard_timeout_seconds"], "timeout cap")
    same_json(
        timeout["parent_elapsed_seconds"], preservation["parent_elapsed_seconds"], "timeout elapsed"
    )
    for name, expected in preservation["preserved_partial_bundle_files"].items():
        same_json(records[group["partial_parts"][name]]["identity"], expected, "whole partial part")
    for original, snapshot in preservation["source_snapshots"].items():
        same_json(
            records[snapshot["snapshot"].replace("\\", "/")]["identity"],
            snapshot["identity"],
            "whole timeout source snapshot " + original,
        )
    audit = _record(records, group["partial_parts"]["outcome-costs.audit.json"])
    same_json(
        audit["derivative_validation_elapsed_seconds"],
        preservation["nonterminal_audit_elapsed_seconds"],
        "nonterminal audit",
    )
    return {
        "operation": "first_P6.10_outcome_cost_publication",
        "status": "timed_out_partial_not_completed",
        "hard_timeout_seconds": timeout["hard_timeout_seconds"],
        "parent_elapsed_seconds": timeout["parent_elapsed_seconds"],
        "nonterminal_audit_elapsed_seconds": audit["derivative_validation_elapsed_seconds"],
        "claim_preserved": True,
        "terminal_child_reader_counts": None,
        "terminal_child_scientific_guard_counts": None,
        "counter_status": "unobserved_killed_child_not_zero",
        "subsequent_repeat_or_readback_in_failed_attempt": "not_executed",
        "repair_rationale": preservation["repair_rationale"],
        "original_cap_and_scientific_protocol_unchanged": True,
        "preservation_record": group["preservation"],
        "timeout_record": group["timeout"],
        "partial_parts": deepcopy(group["partial_parts"]),
        "later_complete_operations": group["repaired_validation"],
        "interpretation": "operational_timeout_not_a_new_scientific_regression_or_a_successful_publication",
    }


def _failed_check(records: dict[str, Any], path: str) -> dict[str, Any]:
    body = _record(records, path)
    commands = body.get("commands", [])
    if type(commands) is dict:
        commands = list(commands.values())
    codes = [command["returncode"] for command in commands]
    if "exit_code" in body:
        codes.append(body["exit_code"])
    if "returncode" in body:
        codes.append(body["returncode"])
    require(any(code != 0 for code in codes), "failed check lacks its original failure")
    return {
        "record": path,
        "status": "failed_operational_check_or_saved_input_derivation_preserved",
        "original_returncodes": codes,
        "reason": body.get("reason") or body.get("stderr"),
        "scope": "operational_correctness_not_an_independent_experiment",
    }


def build_failure_findings(records: dict[str, Any], groups: dict[str, Any]) -> dict[str, Any]:
    """Interpret already bound records; every original raw record remains external."""
    return {
        "timeout": _timeout_finding(records, groups["p610_timeout"]),
        "failed_checks": [_failed_check(records, path) for path in groups["failed_checks"]],
        "history_scope": "complete_fixed_six_family_inputs_and_their_reporting_failures_not_a_claim_that_all_historical_attempts_succeeded",
        "scientific_failures": "retain_original_cell_and_endpoint_failure_records_in_the_complete_primary_outcome_and_matrix_bodies",
        "terminal_counter_rule": "parent_zero_guards_and_nonterminal_audit_never_establish_killed_child_zero_counters",
    }
