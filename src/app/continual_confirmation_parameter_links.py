"""Bind original parameter hash contracts and unscored raw checkpoint links.

Inputs are schema-checked confirmation JSON and frozen family metadata.
Output is success or a link failure. Intermediate combined/parent records
contain hashes, not tensors; this module checks their chains and boundary
links, not unseen tensor values, work/controller semantics, scores or IO.
"""

from __future__ import annotations

import json
from typing import Any

from src.app.continual_combined_factor_manifest import DECISION_ARMS
from src.app.continual_confirmation_json import hash_value, object_fields, require, same_json
from src.app.continual_confirmation_manifest import ConfirmationFamily


GATING_CONTRACT = "p63-shallow-parameter-tensors-v1"
REPLAY_CONTRACT = "p63-replay-shallow-parameter-tensors-v1"
SLEEP_CONTRACT = "p63-sleep-factor-shallow-parameters-v1"
PARAMETER_CONTRACTS = (GATING_CONTRACT, REPLAY_CONTRACT, SLEEP_CONTRACT)


def _link(actual: Any, expected: Any, context: str) -> None:
    require(
        hash_value(actual, context) == hash_value(expected, context),
        f"parameter hash link differs: {context}",
    )


def _rows(value: Any, key: str, names: tuple[str, ...], context: str) -> dict[str, Any]:
    require(
        type(value) is list
        and all(type(row) is dict and key in row for row in value)
        and [row[key] for row in value] == list(names),
        f"parameter link row inventory differs: {context}",
    )
    return {row[key]: row for row in value}


def _checkpoint(row: dict[str, Any], stage: str, name: str, contract: str) -> str:
    return hash_value(row[stage][name]["parameter_sha256_by_contract"][contract], "checkpoint")


def verify_parameter_links(row: dict[str, Any], family: ConfirmationFamily) -> None:
    """Verify parameter links only; the full family work gate remains separate."""
    contract = {"gating": GATING_CONTRACT, "sleep": SLEEP_CONTRACT}.get(
        family.name, REPLAY_CONTRACT
    )
    legacy = row["legacy_train_facts"]
    methods = _rows(
        legacy["arms" if family.name == "sleep" else "methods"],
        "method" if family.name in {"gating", "replay"} else "name",
        family.arms,
        "methods",
    )
    for name, method in methods.items():
        for field, stage in (
            ("initial_parameter_sha256", "initial"),
            ("final_parameter_sha256", "after_b"),
        ):
            _link(method[field], _checkpoint(row, stage, name, contract), f"{name}/{stage}")
        if family.name in {"schedule", "combined", "parent"}:
            _link(
                method["after_a_parameter_sha256"],
                _checkpoint(row, "after_a", name, contract),
                f"{name}/after_a",
            )
    if family.name == "sleep":
        _verify_sleep_links(row, methods)
    elif family.name == "schedule":
        _verify_schedule_links(row)
    elif family.name in {"combined", "parent"}:
        _verify_epoch_links(row, family)


def _verify_sleep_links(row: dict[str, Any], methods: dict[str, Any]) -> None:
    witnesses = {item["name"]: item for item in row["supplemental_guards"]}
    for name, method in methods.items():
        _link(
            method["post_sleep_parameter_sha256"],
            _checkpoint(row, "after_a", name, SLEEP_CONTRACT),
            f"{name}/post_sleep/after_a",
        )
        witness = witnesses.get(name)
        if witness is None:
            _link(
                method["pre_sleep_parameter_sha256"],
                method["post_sleep_parameter_sha256"],
                f"{name}/no_sleep",
            )
            continue
        for endpoint, field in (
            ("before", "pre_sleep_parameter_sha256"),
            ("after", "post_sleep_parameter_sha256"),
        ):
            _link(
                method[field],
                witness[endpoint]["parameter_sha256_by_contract"][SLEEP_CONTRACT],
                f"{name}/guard/{endpoint}",
            )


def _opportunities(legacy: dict[str, Any]) -> list[dict[str, Any]]:
    rows = legacy["opportunities"]
    require(type(rows) is list and len(rows) == 24, "parameter link opportunities differ")
    for index, item in enumerate(rows, start=1):
        require(type(item) is dict, "parameter link opportunity is not an object")
        for key, expected in (
            ("phase", "a" if index <= 12 else "b"),
            ("epoch", (index - 1) % 12 + 1),
            ("global_epoch", index),
        ):
            same_json(item[key], expected, f"parameter link opportunity/{key}")
    return rows


def _verify_schedule_links(row: dict[str, Any]) -> None:
    policies = ("periodic", "adaptive", "no_sleep")
    opportunities = _opportunities(row["legacy_train_facts"])
    for index, item in enumerate(opportunities):
        decisions = _rows(item["decisions"], "policy", policies, "schedule guards")
        for policy_index, policy in enumerate(policies):
            witness = row["supplemental_guards"][3 * index + policy_index]
            decision = decisions[policy]
            for endpoint in ("before", "after"):
                _link(
                    decision[f"parameter_sha256_{endpoint}"],
                    witness[endpoint]["parameter_sha256_by_contract"][REPLAY_CONTRACT],
                    f"{witness['name']}/guard/{index + 1}/{endpoint}",
                )
            if index in (11, 23):
                stage = "after_a" if index == 11 else "after_b"
                _link(
                    decision["parameter_sha256_after"],
                    _checkpoint(row, stage, witness["name"], REPLAY_CONTRACT),
                    f"{witness['name']}/{stage}/guard",
                )


def _verify_epoch_links(row: dict[str, Any], family: ConfirmationFamily) -> None:
    manifest = json.loads(family.development_manifest_json)
    decision_names = tuple(
        arm["name"]
        for arm in manifest["arms"]
        if (
            arm.get("parent_mode") is not None
            if family.name == "parent"
            else arm["name"] in DECISION_ARMS
        )
    )
    for index, opportunity in enumerate(_opportunities(row["legacy_train_facts"]), start=1):
        wake = _rows(opportunity["wake"], "name", family.arms, "epoch wake")
        after = object_fields(
            opportunity["after_epoch_parameter_sha256"], set(family.arms), "epoch parameter links"
        )
        decisions = _rows(opportunity["decisions"], "name", decision_names, "epoch guards")
        for name in family.arms:
            hash_value(wake[name]["parameter_sha256"], "epoch wake")
            hash_value(after[name], "after epoch")
            if name in decisions:
                _verify_decision_links(decisions[name], wake[name], after[name], family.name)
            elif family.name == "parent" or not opportunity["applied_control_replay_ids"].get(name):
                _link(after[name], wake[name]["parameter_sha256"], f"{name}/unmodified wake")
            if index in (12, 24):
                stage = "after_a" if index == 12 else "after_b"
                _link(
                    after[name],
                    _checkpoint(row, stage, name, REPLAY_CONTRACT),
                    f"{name}/{stage}/epoch",
                )


def _verify_decision_links(
    decision: dict[str, Any], wake: dict[str, Any], after_epoch: str, family: str
) -> None:
    name = decision["name"]
    if family == "parent":
        before = decision["before"]["parameter_sha256"]
        proposed = hash_value(decision["proposed"]["parameter_sha256"], "parent proposal")
        after = decision["after"]["parameter_sha256"]
        outcome = decision["event"]["outcome"]
        if outcome == "accepted":
            _link(after, proposed, f"{name}/accepted proposal")
        elif outcome == "skipped":
            _link(before, proposed, f"{name}/skipped proposal")
    else:
        before = decision["parameter_sha256_before"]
        after = decision["parameter_sha256_after"]
        outcome = decision["event"]["outcome"]
    _link(before, wake["parameter_sha256"], f"{name}/guard/before/wake")
    _link(after, after_epoch, f"{name}/guard/after/epoch")
    require(
        outcome in {"accepted", "rolled_back", "skipped"}, "parameter link guard outcome differs"
    )
    if outcome != "accepted":
        _link(before, after, f"{name}/rejected or skipped guard")
