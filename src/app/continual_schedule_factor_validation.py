"""Independently rederive unscored schedule facts from a complete JSON object.

Inputs are a train-only payload and its separately fixed manifest. This module
checks all cells, decisions, clocks, replay/guard costs and capacity. It does
not train, construct sources, read evaluation arrays or write files.
"""

from __future__ import annotations

from math import isfinite
import re
from typing import Any


ROLE_COUNTS = {
    "a_train": 72,
    "a_inner_guard": 24,
    "a_outer_selection": 24,
    "b_train": 36,
    "b_inner_guard": 12,
    "b_outer_selection": 12,
}
PARAMETER_HASH_FIELDS = (
    "initial_parameter_sha256",
    "after_a_parameter_sha256",
    "final_parameter_sha256",
)
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def verify_schedule_preflight_payload(payload: dict[str, Any], manifest: dict[str, Any]) -> None:
    if (
        payload["protocol_id"] != manifest["protocol_id"]
        or payload["manifest"] != manifest
        or payload["outer_selection_scored"] is not False
        or payload["final_released"] is not False
        or [row["seed"] for row in payload["seed_results"]] != manifest["seeds"]
    ):
        raise ValueError("P6.3 schedule protocol, cells or evaluation seal differs")
    for seed in payload["seed_results"]:
        _verify_seed(seed, manifest)
    executed = sum(seed["executed_optimizer_updates"] for seed in payload["seed_results"])
    if executed > 1014 or executed > manifest["max_optimizer_updates"]:
        raise ValueError("P6.3 schedule executed optimizer budget exceeded")


def _is_hash(value: Any) -> bool:
    return isinstance(value, str) and HEX_SHA256.fullmatch(value) is not None


def _verify_seed(seed: dict[str, Any], manifest: dict[str, Any]) -> None:
    hashes = seed["role_hashes"]
    if (
        seed["final_released"] is not False
        or seed["role_counts"] != ROLE_COUNTS
        or set(hashes) != set(ROLE_COUNTS)
        or not all(_is_hash(value) for value in hashes.values())
        or [method["name"] for method in seed["methods"]] != manifest["arms"]
        or len(seed["opportunities"]) != 24
    ):
        raise ValueError("P6.3 schedule roles, methods or opportunity cells differ")
    methods = {row["name"]: row for row in seed["methods"]}
    if len({methods[name]["initial_parameter_sha256"] for name in manifest["arms"][:9]}) != 1:
        raise ValueError("P6.3 schedule width-eight initialization differs")
    if (
        methods[manifest["arms"][-1]]["initial_parameter_sha256"]
        != methods[manifest["arms"][-2]]["initial_parameter_sha256"]
    ):
        raise ValueError("P6.3 schedule planned-width initialization differs")
    counts = {
        policy: {"attempts": 0, "accepted": 0, "rejected_updates": 0}
        for policy in manifest["policies"]
    }
    last_sleep = dict.fromkeys(manifest["policies"], 0)
    windows: dict[str, list[float]] = {policy: [] for policy in manifest["policies"]}
    supply: dict[str, tuple[list[str], list[str]]] = {}
    for index, opportunity in enumerate(seed["opportunities"], start=1):
        _verify_opportunity(opportunity, index, seed, manifest, supply)
        for decision in opportunity["decisions"]:
            policy = decision["policy"]
            _verify_decision(
                decision, opportunity, manifest, last_sleep[policy], windows[policy], hashes
            )
            windows[policy] = decision["energy_window"]
            counts[policy]["attempts"] += int(decision["attempted"])
            counts[policy]["accepted"] += int(decision["outcome"] == "accepted")
            counts[policy]["rejected_updates"] += (
                decision["proposed_replay_updates"] if decision["outcome"] == "rolled_back" else 0
            )
            if decision["outcome"] == "accepted":
                last_sleep[policy] = index
            checkpoint = (
                "after_a_parameter_sha256"
                if index == 12
                else "final_parameter_sha256"
                if index == 24
                else None
            )
            if (
                checkpoint
                and decision["parameter_sha256_after"] != methods[f"neutral_{policy}"][checkpoint]
            ):
                raise ValueError("P6.3 schedule checkpoint parameter hash differs")
    for policy in manifest["policies"]:
        if any(
            methods[f"pc_{policy}"][field] != methods[f"neutral_{policy}"][field]
            for field in PARAMETER_HASH_FIELDS
        ):
            raise ValueError("P6.3 schedule ordinary PC/neutral parameter parity differs")
    for method in seed["methods"]:
        _verify_method(method, counts, manifest)
    executed = sum(
        row["wake_updates"]
        + row["applied_replay_updates"]
        + row["rejected_executed_replay_updates"]
        for row in seed["methods"]
    )
    if seed["executed_optimizer_updates"] != executed:
        raise ValueError("P6.3 schedule executed optimizer total differs")


def _verify_opportunity(
    row: dict[str, Any],
    index: int,
    seed: dict[str, Any],
    manifest: dict[str, Any],
    supply: dict[str, tuple[list[str], list[str]]],
) -> None:
    phase = "a" if index <= 12 else "b"
    epoch = (index - 1) % 12 + 1
    selected, retained = row["selected_ids"], row["retained_order_ids"]
    if (
        (row["phase"], row["epoch"], row["global_epoch"]) != (phase, epoch, index)
        or row["train_role_hash"] != seed["role_hashes"][f"{phase}_train"]
        or row["retained_examples"] != manifest["memory_examples"]
        or row["retained_bytes"] != manifest["memory_bytes"]
        or len(retained) != 8
        or len(set(retained)) != 8
        or not all(_is_hash(value) for value in retained)
        or selected != retained[-manifest["replay_updates_per_attempt"] :]
        or [decision["policy"] for decision in row["decisions"]] != manifest["policies"]
    ):
        raise ValueError("P6.3 schedule shared opportunity supply differs")
    if phase in supply and supply[phase] != (selected, retained):
        raise ValueError("P6.3 schedule same-phase shared supply changed")
    supply[phase] = (selected, retained)


def _verify_decision(
    row: dict[str, Any],
    opportunity: dict[str, Any],
    manifest: dict[str, Any],
    last_sleep: int,
    previous_window: list[float],
    role_hashes: dict[str, str],
) -> None:
    policy = row["policy"]
    config = manifest["configs"][policy]
    window = row["energy_window"]
    index = opportunity["global_epoch"]
    if (
        len(window) != min(index, config["sleep_energy_window"])
        or not all(type(value) in (int, float) and isfinite(value) for value in window)
        or previous_window
        and window[:-1] != previous_window[-(config["sleep_energy_window"] - 1) :]
        or type(row["chemical_variance"]) not in (int, float)
        or not isfinite(row["chemical_variance"])
        or row["chemical_variance"] < 0
    ):
        raise ValueError("P6.3 schedule adaptive signal history differs")
    spacing = index - last_sleep
    periodic = policy == "periodic" and opportunity["epoch"] % manifest["periodic_interval"] == 0
    adaptive = (
        config["use_adaptive_sleep_trigger"]
        and spacing >= config["min_epochs_between_sleep"]
        and len(window) == config["sleep_energy_window"]
        and window[0] - window[-1] <= config["sleep_plateau_delta"]
        and row["chemical_variance"] >= config["sleep_chemical_variance_threshold"]
    )
    attempted = periodic or adaptive
    accepted = row["outcome"] == "accepted"
    expected_ids = opportunity["selected_ids"] if accepted else []
    if (
        row["spacing_before"] != spacing
        or row["spacing_after"] != (0 if accepted else spacing)
        or row["periodic_due"] is not periodic
        or row["adaptive_due"] is not adaptive
        or row["attempted"] is not attempted
        or row["force_sleep"] is not periodic
        or row["trigger_reason"]
        != ("periodic" if periodic else "adaptive" if adaptive else "not_due")
        or row["proposed_replay_ids"] != (opportunity["selected_ids"] if attempted else [])
        or row["proposed_replay_updates"] != (2 if attempted else 0)
        or row["proposed_replay_examples"] != (2 if attempted else 0)
        or row["applied_ids_by_method"]
        != {method: expected_ids for method in ("backprop", "pc", "neutral")}
        or not _is_hash(row["parameter_sha256_before"])
        or not _is_hash(row["parameter_sha256_after"])
        or not accepted
        and row["parameter_sha256_before"] != row["parameter_sha256_after"]
    ):
        raise ValueError("P6.3 schedule decision, clocks or replay IDs differ")
    _verify_guard(row, attempted, accepted, opportunity, role_hashes)


def _verify_guard(
    row: dict[str, Any],
    attempted: bool,
    accepted: bool,
    opportunity: dict[str, Any],
    role_hashes: dict[str, str],
) -> None:
    if not attempted:
        if (
            row["outcome"] != "skipped"
            or row["reason"] != "schedule_not_due"
            or any(
                row[key] is not None
                for key in ("guard_role_hash", "guard_pre_accuracy", "guard_post_accuracy")
            )
        ):
            raise ValueError("P6.3 schedule miss has a guard or outcome")
        return
    before, after = row["guard_pre_accuracy"], row["guard_post_accuracy"]
    if (
        row["outcome"] not in {"accepted", "rolled_back"}
        or row["guard_role_hash"] != role_hashes[f"{opportunity['phase']}_inner_guard"]
        or any(
            type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1
            for value in (before, after)
        )
        or accepted != (after >= before)
        or row["reason"] != ("guard_accepted" if accepted else "guard_rejected")
    ):
        raise ValueError("P6.3 schedule guard acceptance or role differs")


def _verify_method(
    row: dict[str, Any], counts: dict[str, dict[str, int]], manifest: dict[str, Any]
) -> None:
    name = row["name"]
    reference = name in manifest["arms"][-2:]
    neutral = name.startswith("neutral_")
    pc = not name.startswith("backprop_")
    policy = next(
        (
            p
            for p in manifest["policies"]
            if name in {f"{m}_{p}" for m in ("backprop", "pc", "neutral")}
        ),
        None,
    )
    count = (
        counts[policy]
        if policy is not None
        else {"attempts": 0, "accepted": 0, "rejected_updates": 0}
    )
    width = manifest["planned_width"] if reference else manifest["width"]
    applied = 2 * count["accepted"]
    rejected = count["rejected_updates"] if neutral else 0
    expected = {
        "width_initial": width,
        "width_final": width,
        "width_peak": width,
        "parameters_initial": 4 * width + 1,
        "parameters_final": 4 * width + 1,
        "parameters_peak": 4 * width + 1,
        "wake_updates": 24,
        "wake_presentations": 1296,
        "wake_inference_loops": 48 if pc else 0,
        "wake_example_inference_iterations": 2592 if pc else 0,
        "applied_replay_updates": applied,
        "applied_replay_presentations": applied,
        "applied_replay_inference_loops": 2 * applied if pc else 0,
        "rejected_executed_replay_updates": rejected,
        "rejected_replay_inference_loops": 2 * rejected,
        "controller_accepted_events": count["accepted"],
        "sleep_attempts": count["attempts"] if neutral else 0,
        "guard_evaluations": 2 * count["attempts"] if neutral else 0,
        "retained_examples": 8 if neutral else 0,
        "retained_array_bytes": 192 if neutral else 0,
        "replay_memory_access": "none" if reference else "model_fifo" if neutral else "shared_fifo",
    }
    if any(row[key] != value for key, value in expected.items()) or not all(
        _is_hash(row[key]) for key in PARAMETER_HASH_FIELDS
    ):
        raise ValueError(f"P6.3 schedule method work/capacity differs: {name}")
