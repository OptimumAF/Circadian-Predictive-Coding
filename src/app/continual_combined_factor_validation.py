"""Independently rederive complete combined train facts from finite JSON.

Inputs are the payload and separately fixed manifest. Outputs are explicit
validation failures or success. This module performs no training, dataset
construction, score selection or file IO.
"""

from __future__ import annotations

from math import isfinite
import re
from typing import Any

from src.app.continual_combined_factor_manifest import DECISION_ARMS, PARITY_PAIRS, REPLAY_CONTROLS
from src.app.continual_schedule_factor_validation import ROLE_COUNTS
from src.core.sleep_telemetry import (
    ChemicalSummaries,
    ChemicalSummary,
    SleepBudgets,
    SleepDurations,
    SleepEventTelemetry,
    SleepGuardMetrics,
    SleepReplayUsage,
    SleepStructuralChanges,
)


HASH = re.compile(r"[0-9a-f]{64}\Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"P6.3 combined {message}")


def _hash(value: Any) -> bool:
    return isinstance(value, str) and HASH.fullmatch(value) is not None


def _telemetry(row: dict[str, Any]) -> SleepEventTelemetry:
    changes: dict[str, Any] = {
        key: tuple(tuple(pair) for pair in value) if key.endswith("split_pairs") else tuple(value)
        for key, value in row["changes"].items()
    }
    chemistry: dict[str, Any] = {
        key: ChemicalSummaries(
            **{name: ChemicalSummary(**value) for name, value in row[key].items()}
        )
        for key in ("chemistry_before", "chemistry_proposed", "chemistry_final")
    }
    simple = {
        key: value
        for key, value in row.items()
        if key
        not in {
            "changes",
            "budgets",
            "guard",
            "replay",
            *chemistry,
        }
    }
    return SleepEventTelemetry(
        **simple,
        changes=SleepStructuralChanges(**changes),
        budgets=SleepBudgets(**row["budgets"]),
        guard=SleepGuardMetrics(**row["guard"]) if row["guard"] is not None else None,
        replay=SleepReplayUsage(**row["replay"]),
        **chemistry,
        durations=SleepDurations(0.0, 0.0),
    )


def verify_combined_payload(payload: dict[str, Any], manifest: dict[str, Any]) -> None:
    _require(
        payload["protocol_id"] == manifest["protocol_id"]
        and payload["manifest"] == manifest
        and payload["outer_selection_scored"] is False
        and payload["final_released"] is False
        and [seed["seed"] for seed in payload["seed_results"]] == manifest["seeds"],
        "protocol, seeds or evaluation seal differs",
    )
    for seed in payload["seed_results"]:
        _verify_seed(seed, manifest)
    _require(
        sum(seed["executed_optimizer_updates"] for seed in payload["seed_results"])
        <= min(1548, manifest["max_optimizer_updates"]),
        "executed work exceeds cap",
    )


def _new_counts() -> dict[str, int]:
    return {
        key: 0
        for key in ("replay", "rejected", "attempts", "accepted", "guard_examples", "last_sleep")
    }


def _verify_seed(seed: dict[str, Any], manifest: dict[str, Any]) -> None:
    specs = {arm["name"]: arm for arm in manifest["arms"]}
    _require(
        seed["final_released"] is False
        and seed["role_counts"] == ROLE_COUNTS
        and set(seed["role_hashes"]) == set(ROLE_COUNTS)
        and all(_hash(value) for value in seed["role_hashes"].values())
        and [row["name"] for row in seed["methods"]] == list(specs)
        and len(seed["opportunities"]) == 24,
        "role or model cells differ",
    )
    counts = {name: _new_counts() for name in specs}
    peaks = {name: spec["width"] for name, spec in specs.items()}
    previous_width = peaks.copy()
    supply: dict[str, tuple[list[str], list[str]]] = {}
    for index, opportunity in enumerate(seed["opportunities"], start=1):
        _verify_opportunity(opportunity, index, seed, specs, previous_width, supply)
        for decision in opportunity["decisions"]:
            name = decision["name"]
            _verify_decision(decision, opportunity, seed, specs[name], counts[name])
            event = decision["event"]
            peaks[name] = max(
                peaks[name], event["before_width"] + len(event["changes"]["proposed_split_pairs"])
            )
        committed = (
            next(row for row in opportunity["decisions"] if row["name"] == "full")["event"][
                "outcome"
            ]
            == "accepted"
        )
        for name in REPLAY_CONTROLS:
            expected = opportunity["selected_ids"] if committed else []
            _require(
                opportunity["applied_control_replay_ids"][name] == expected,
                "matched replay IDs differ",
            )
            if committed:
                counts[name]["replay"] += 2
                if name == "neutral_full_replay":
                    counts[name]["accepted"] += 1
                    counts[name]["last_sleep"] = index
        previous_width = opportunity["after_epoch_widths"]
    _verify_methods(seed, specs, counts, peaks)


def _verify_opportunity(
    row: dict[str, Any],
    index: int,
    seed: dict[str, Any],
    specs: dict[str, Any],
    prior_width: dict[str, int],
    supply: dict[str, tuple[list[str], list[str]]],
) -> None:
    phase, epoch = ("a", index) if index <= 12 else ("b", index - 12)
    circadian = {name for name, spec in specs.items() if spec["model_kind"] == "circadian"}
    _require(
        (row["phase"], row["epoch"], row["global_epoch"]) == (phase, epoch, index)
        and row["train_role_hash"] == seed["role_hashes"][f"{phase}_train"]
        and row["retained_examples"] == 8
        and row["retained_bytes"] == 192
        and len(row["retained_order_ids"]) == len(set(row["retained_order_ids"])) == 8
        and all(_hash(value) for value in row["retained_order_ids"])
        and row["selected_ids"] == row["retained_order_ids"][-2:]
        and [wake["name"] for wake in row["wake"]] == list(specs)
        and [decision["name"] for decision in row["decisions"]] == list(DECISION_ARMS)
        and set(row["after_epoch_parameter_sha256"]) == set(specs)
        and set(row["after_epoch_widths"]) == set(specs)
        and set(row["after_epoch_state_sha256"]) == circadian
        and set(row["applied_control_replay_ids"]) == set(REPLAY_CONTROLS),
        "opportunity, supply or decision cells differ",
    )
    offered = (row["retained_order_ids"], row["selected_ids"])
    if phase in supply:
        _require(offered == supply[phase], "supply changed within fixed phase")
    supply[phase] = offered
    decisions = {decision["name"]: decision for decision in row["decisions"]}
    for wake in row["wake"]:
        name, spec = wake["name"], specs[wake["name"]]
        _require(
            wake["width"] == prior_width[name]
            and wake["parameters"] == 4 * wake["width"] + 1
            and spec["min_width"] <= wake["width"] <= spec["max_width"]
            and _hash(wake["parameter_sha256"])
            and _hash(row["after_epoch_parameter_sha256"][name]),
            "wake capacity or parameter identity differs",
        )
        _verify_wake_state(wake, spec)
        if name in decisions:
            decision = decisions[name]
            _require(
                decision["parameter_sha256_before"] == wake["parameter_sha256"]
                and decision["event"]["before_width"] == wake["width"]
                and decision["parameter_sha256_after"] == row["after_epoch_parameter_sha256"][name]
                and decision["state_sha256_after"] == row["after_epoch_state_sha256"][name]
                and decision["event"]["final_width"] == row["after_epoch_widths"][name],
                "decision checkpoint differs from epoch facts",
            )
        else:
            _require(
                row["after_epoch_widths"][name] == wake["width"], "fixed consumer changed width"
            )
            if name not in REPLAY_CONTROLS or not row["applied_control_replay_ids"][name]:
                _require(
                    row["after_epoch_parameter_sha256"][name] == wake["parameter_sha256"],
                    "unreplayed baseline changed parameters",
                )
        if name in circadian:
            _require(_hash(row["after_epoch_state_sha256"][name]), "state fingerprint is invalid")
    for left, right in PARITY_PAIRS:
        _require(
            row["after_epoch_parameter_sha256"][left] == row["after_epoch_parameter_sha256"][right],
            "neutral PC parity differs",
        )


def _verify_wake_state(wake: dict[str, Any], spec: dict[str, Any]) -> None:
    config = spec["config"]
    if config is None:
        _require(
            wake["minimum_plasticity"] is None and wake["reward_scale"] is None,
            "baseline has circadian state",
        )
        return
    plasticity, scale = wake["minimum_plasticity"], wake["reward_scale"]
    _require(
        type(plasticity) in (int, float)
        and isfinite(plasticity)
        and config["min_plasticity"] <= plasticity <= 1.0
        and type(scale) in (int, float)
        and isfinite(scale),
        "plasticity/reward observation is invalid",
    )
    if config["use_reward_modulated_learning"]:
        _require(
            config["reward_scale_min"] <= scale <= config["reward_scale_max"],
            "difficulty scale exceeds declared range",
        )
    else:
        _require(scale == 1.0, "disabled difficulty changed wake scale")


def _clocks(index: int, count: dict[str, int]) -> dict[str, int]:
    return {
        "wake_batches": index,
        "wake_examples": 72 * min(index, 12) + 36 * max(index - 12, 0),
        "wake_batches_since_sleep": index - count["last_sleep"],
        "replay_updates": count["replay"],
        "sleep_events": count["accepted"],
    }


def _verify_decision(
    row: dict[str, Any],
    opportunity: dict[str, Any],
    seed: dict[str, Any],
    spec: dict[str, Any],
    count: dict[str, int],
) -> None:
    event = _telemetry(row["event"])
    index = opportunity["global_epoch"]
    due = bool(spec["interval"] and opportunity["epoch"] % spec["interval"] == 0)
    _require(
        event.completed_epoch == index
        and event.wake_batches == index
        and row["clocks_before"] == _clocks(index, count)
        and all(
            _hash(row[key])
            for key in (
                "state_sha256_before",
                "state_sha256_after",
                "parameter_sha256_before",
                "parameter_sha256_after",
            )
        ),
        "decision clocks or fingerprints differ",
    )
    if due:
        guard = event.guard
        _require(
            guard is not None
            and event.outcome in {"accepted", "rolled_back"}
            and event.trigger_reason == "periodic",
            "due guard or outcome differs",
        )
        assert guard is not None
        _require(
            guard.role == "inner_guard"
            and guard.metric_name == "accuracy"
            and guard.tolerance == 0
            and guard.role_hash == seed["role_hashes"][f"{opportunity['phase']}_inner_guard"]
            and guard.examples_scored == 2 * ROLE_COUNTS[f"{opportunity['phase']}_inner_guard"],
            "guard role or exposure differs",
        )
        count["attempts"] += 1
        count["guard_examples"] += guard.examples_scored
    else:
        _require(
            event.outcome == "skipped" and event.guard is None, "not-due event performed sleep"
        )
    expected_updates = 2 if due and spec["config"]["sleep_enable_replay"] else 0
    config = spec["config"]
    progress = index / 24
    split_limit = int(
        due
        and config["sleep_enable_split"]
        and progress < config["sleep_prune_only_after_fraction"]
    )
    prune_limit = int(
        due
        and config["sleep_enable_prune"]
        and progress >= config["sleep_split_only_until_fraction"]
    )
    _require(
        event.budgets == SleepBudgets(split_limit, prune_limit, expected_updates, None),
        "resolved phase/component budget differs",
    )
    _require(
        event.replay.proposed_updates == event.replay.proposed_examples == expected_updates,
        "executed replay proposal differs",
    )
    _require(
        row["proposed_replay_ids"] == (opportunity["selected_ids"] if expected_updates else []),
        "proposed replay IDs differ",
    )
    _require(
        row["applied_replay_ids"]
        == (opportunity["selected_ids"] if event.replay.applied_updates else []),
        "applied replay IDs differ",
    )
    if event.outcome == "accepted":
        count["accepted"] += 1
        count["replay"] += event.replay.applied_updates
        count["last_sleep"] = index
    else:
        _require(
            row["state_sha256_before"] == row["state_sha256_after"]
            and row["parameter_sha256_before"] == row["parameter_sha256_after"],
            "rollback failed to restore complete state",
        )
        if event.outcome == "rolled_back":
            count["rejected"] += event.replay.proposed_updates
    _require(
        row["clocks_after"] == _clocks(index, count), "applied clocks differ from guard outcome"
    )
    _verify_lineage(row, event, spec)


def _verify_lineage(row: dict[str, Any], event: SleepEventTelemetry, spec: dict[str, Any]) -> None:
    before, after, changes = row["lineage_before"], row["lineage_after"], event.changes
    ids, parents = before["neuron_ids"], before["parent_ids"]
    proposed = changes.proposed_split_pairs
    _require(
        len(ids) == len(parents) == event.before_width
        and len(set(ids)) == len(ids)
        and all(parent in ids for parent, _ in proposed)
        and [child for _, child in proposed]
        == list(range(before["next_neuron_id"], before["next_neuron_id"] + len(proposed)))
        and set(changes.proposed_prune_ids).issubset(ids)
        and not changes.proposed_scheduled_prune_ids
        and event.before_width + len(proposed) <= spec["max_width"]
        and spec["min_width"] <= event.final_width <= spec["max_width"],
        "lineage or transient width differs",
    )
    config = spec["config"]
    if not config["sleep_enable_split"]:
        _require(not proposed, "disabled split proposed a change")
    if not config["sleep_enable_prune"]:
        _require(not changes.proposed_prune_ids, "disabled prune proposed a change")
    _require(
        len(proposed) <= 1 and len(changes.proposed_prune_ids) <= 1, "structural budget exceeded"
    )
    if event.outcome != "accepted":
        _require(after == before, "rejected/skipped lineage changed")
        return
    added = [child for _, child in proposed]
    mapping = dict(zip(ids + added, parents + [parent for parent, _ in proposed], strict=True))
    expected_ids = [
        value for value in ids + added if value not in changes.applied_removed_prune_ids
    ]
    _require(
        after["neuron_ids"] == expected_ids
        and after["parent_ids"] == [mapping[value] for value in expected_ids]
        and after["next_neuron_id"] == before["next_neuron_id"] + len(added),
        "committed lineage differs",
    )


def _verify_methods(
    seed: dict[str, Any],
    specs: dict[str, Any],
    counts: dict[str, Any],
    peaks: dict[str, int],
) -> None:
    opportunities = seed["opportunities"]
    methods = {row["name"]: row for row in seed["methods"]}
    for width in (8, 14):
        _require(
            len(
                {
                    methods[name]["initial_parameter_sha256"]
                    for name, spec in specs.items()
                    if spec["width"] == width
                }
            )
            == 1,
            "initial parity within width differs",
        )
    for name, method in methods.items():
        spec, count = specs[name], counts[name]
        is_pc = spec["model_kind"] != "backprop"
        expected = {
            "width_initial": spec["width"],
            "width_final": opportunities[-1]["after_epoch_widths"][name],
            "width_peak": peaks[name],
            "parameters_initial": 4 * spec["width"] + 1,
            "parameters_final": 4 * opportunities[-1]["after_epoch_widths"][name] + 1,
            "parameters_peak": 4 * peaks[name] + 1,
            "after_a_parameter_sha256": opportunities[11]["after_epoch_parameter_sha256"][name],
            "final_parameter_sha256": opportunities[-1]["after_epoch_parameter_sha256"][name],
            "wake_updates": 24,
            "wake_presentations": 1296,
            "wake_inference_loops": 48 if is_pc else 0,
            "wake_example_inference_iterations": 2592 if is_pc else 0,
            "applied_replay_updates": count["replay"],
            "applied_replay_presentations": count["replay"],
            "applied_replay_inference_loops": 2 * count["replay"] if is_pc else 0,
            "rejected_executed_replay_updates": count["rejected"],
            "rejected_replay_inference_loops": 2 * count["rejected"],
            "own_sleep_attempts": count["attempts"],
            "committed_sleep_events": count["accepted"],
            "guard_evaluations": 2 * count["attempts"],
            "guard_examples": count["guard_examples"],
        }
        _require(
            all(method[key] == value for key, value in expected.items())
            and _hash(method["initial_parameter_sha256"]),
            "method cost, capacity or checkpoint facts differ",
        )
        if spec["config"] is None:
            _require(
                method["final_clocks"] is None
                and method["retention"] is None
                and method["final_state_sha256"] is None,
                "baseline has circadian clocks/memory",
            )
        else:
            _require(
                method["final_clocks"] == _clocks(24, count)
                and method["final_state_sha256"]
                == opportunities[-1]["after_epoch_state_sha256"][name],
                "final model state differs",
            )
            _require(
                method["retention"]
                == {
                    "sample_ids": sorted(opportunities[-1]["retained_order_ids"]),
                    "example_count": 8,
                    "retained_bytes": 192,
                },
                "final retained memory differs",
            )
    _require(
        seed["executed_optimizer_updates"]
        == sum(
            row["wake_updates"]
            + row["applied_replay_updates"]
            + row["rejected_executed_replay_updates"]
            for row in methods.values()
        ),
        "executed optimizer total differs",
    )
