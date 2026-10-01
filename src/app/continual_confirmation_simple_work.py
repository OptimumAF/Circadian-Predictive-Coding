"""Independently derive gating, matched replay and guarded sleep work.

Inputs are schema-checked raw facts and complete checkpoint views. Original
rules are rederived in app without importing CLI validators. No training,
source construction, evaluation, source selection or IO is performed.
"""

from __future__ import annotations

from typing import Any

from src.app.continual_confirmation_json import canonical_dataclass, hash_value, require, same_json
from src.app.continual_sleep_factor_preflight import GATING_ARMS, SLEEP_ARMS


def counts(row: dict[str, Any], expected: dict[str, Any], context: str) -> None:
    for key, value in expected.items():
        same_json(row[key], value, f"{context}/{key}")


def lineage(checkpoint: dict[str, Any]) -> dict[str, Any]:
    value = canonical_dataclass(
        checkpoint["lineage"], "src.core.neuron_adaptation.NeuronLineageSnapshot"
    )
    return {
        "neuron_ids": value["neuron_ids"]["tuple"],
        "parent_ids": value["parent_ids"]["tuple"],
        "next_neuron_id": value["next_neuron_id"],
    }


def clocks(index: int, accepted: int = 0, replay: int = 0, last_sleep: int = 0) -> dict[str, int]:
    return {
        "wake_batches": index,
        "wake_examples": 72 * min(index, 12) + 36 * max(index - 12, 0),
        "wake_batches_since_sleep": index - last_sleep,
        "replay_updates": replay,
        "sleep_events": accepted,
    }


def require_parity(row: dict[str, Any], pairs: tuple[tuple[str, str], ...]) -> None:
    for stage in ("initial", "after_a", "after_b"):
        for left, right in pairs:
            same_json(
                row[stage][left]["parameter_sha256"],
                row[stage][right]["parameter_sha256"],
                f"{stage} neutral parameter parity",
            )


def verify_gating(row: dict[str, Any], manifest: dict[str, Any]) -> None:
    training = manifest["source"]["training"]
    updates = training["phase_a_epochs"] + training["phase_b_epochs"]
    presentations = 72 * training["phase_a_epochs"] + 36 * training["phase_b_epochs"]
    for method in row["legacy_train_facts"]["methods"]:
        name = method["method"]
        counts(
            method,
            {
                "wake_updates": updates,
                "train_presentations": presentations,
                "latent_iterations": updates * training["pc_inference_steps"],
                "example_inference_iterations": presentations * training["pc_inference_steps"],
                "sleep_attempts": 0,
                "replay_updates": 0,
                "hidden_width_start": 8,
                "hidden_width_end": 8,
                "parameter_count_start": 33,
                "parameter_count_end": 33,
            },
            f"gating {name} work",
        )
        minimum = method["minimum_plasticity"]
        if name == "chemical_gating":
            require(type(minimum) in (int, float) and 0.2 <= minimum < 1.0, "gating inactive")
        else:
            same_json(minimum, None if name == "ordinary_pc" else 1.0, "neutral plasticity")
        if name != "ordinary_pc":
            for stage, index in (("after_a", 12), ("after_b", 24)):
                same_json(row[stage][name]["clocks"], clocks(index), "gating state/work clocks")
    require_parity(row, (("ordinary_pc", "neutral_circadian"),))


def verify_replay(row: dict[str, Any], manifest: dict[str, Any]) -> None:
    raw = row["legacy_train_facts"]
    expected_boundaries = [(phase, epoch) for phase in ("a", "b") for epoch in (4, 8, 12)]
    require(
        [(item["phase"], item["epoch"]) for item in raw["boundaries"]] == expected_boundaries,
        "replay boundary inventory differs",
    )
    on = {"backprop_on", "pc_on", "circadian_on"}
    supply: dict[str, Any] = {}
    for boundary in raw["boundaries"]:
        ids = boundary["retained_order_ids"]
        selected = boundary["selected_ids"]
        require(len(ids) == len(set(ids)) == 8, "replay retained supply differs")
        for value in ids:
            hash_value(value, "replay retained ID")
        counts(boundary, {"retained_examples": 8, "retained_bytes": 192}, "replay memory")
        same_json(selected, ids[-2:], "replay selected FIFO IDs")
        same_json(
            boundary["applied_ids_by_method"],
            {name: selected if name in on else [] for name in manifest["arms"]},
            "matched replay applications",
        )
        phase = boundary["phase"]
        if phase in supply:
            same_json(ids, supply[phase], "fixed-phase replay supply")
        supply[phase] = ids
    for method in raw["methods"]:
        _verify_replay_method(row, method, manifest, supply)
    require_parity(row, (("pc_off", "circadian_off"), ("pc_on", "circadian_on")))


def _verify_replay_method(
    row: dict[str, Any], method: dict[str, Any], manifest: dict[str, Any], supply: dict[str, Any]
) -> None:
    name = method["method"]
    width = manifest["planned_width"] if "width12" in name else manifest["width"]
    pc = not name.startswith("backprop")
    replay = sum(
        len(item["applied_ids_by_method"][name]) for item in row["legacy_train_facts"]["boundaries"]
    )
    counts(
        method,
        {
            "wake_updates": 24,
            "wake_presentations": 1296,
            "wake_inference_loops": 48 if pc else 0,
            "wake_example_inference_iterations": 2592 if pc else 0,
            "replay_updates": replay,
            "replay_presentations": replay,
            "replay_inference_loops": 2 * replay if pc else 0,
            "sleep_attempts": 6 if name.startswith("circadian") else 0,
            "hidden_width_start": width,
            "hidden_width_end": width,
            "parameter_count_start": 4 * width + 1,
            "parameter_count_end": 4 * width + 1,
        },
        f"replay {name} work",
    )
    if not name.startswith("circadian"):
        return
    for stage, phase, index, events in (("after_a", "a", 12, 3), ("after_b", "b", 24, 6)):
        checkpoint = row[stage][name]
        applied = replay * events // 6
        same_json(
            checkpoint["clocks"], clocks(index, events, applied, index), "replay state clocks"
        )
        same_json(
            checkpoint["retention"],
            {"sample_ids": sorted(supply[phase]), "example_count": 8, "retained_bytes": 192},
            "replay checkpoint retention",
        )


def verify_sleep(row: dict[str, Any], manifest: dict[str, Any]) -> None:
    require(row["legacy_train_facts"]["final_released"] is False, "sleep legacy final seal differs")
    arms = row["legacy_train_facts"]["arms"]
    witnesses = {item["name"]: item for item in row["supplemental_guards"]}
    for arm in arms:
        name = arm["name"]
        width = 12 if name.endswith("_12") else 8
        pc = not name.startswith("backprop")
        counts(
            arm,
            {
                "wake_updates": 24,
                "wake_presentations": 1296,
                "latent_inference_loops": 48 if pc else 0,
                "example_inference_iterations": 2592 if pc else 0,
                "replay_updates": 0,
                "width_initial": width,
                "width_final": width,
                "parameters_initial": 4 * width + 1,
                "parameters_final": 4 * width + 1,
            },
            f"sleep {name} work",
        )
        minimum = arm["minimum_a_plasticity"]
        if name in GATING_ARMS:
            require(type(minimum) in (int, float) and 0.2 <= minimum < 1.0, "sleep gating inactive")
        else:
            same_json(minimum, None, "sleep plasticity observation")
        if name in SLEEP_ARMS:
            _verify_guarded_sleep(row, arm, witnesses[name])
        else:
            counts(
                arm,
                {"sleep": None, "width_peak": width, "parameters_peak": 4 * width + 1},
                "no sleep work",
            )
    require_parity(row, (("pc_8", "neutral_sham"),))
    sham, reset = (
        next(item for item in arms if item["name"] == name)
        for name in ("gating_sham", "gating_reset")
    )
    for key in ("pre_sleep_parameter_sha256", "post_sleep_parameter_sha256"):
        same_json(sham[key], reset[key], "sleep gated boundary parity")


def _verify_guarded_sleep(
    row: dict[str, Any], arm: dict[str, Any], witness: dict[str, Any]
) -> None:
    name, event = arm["name"], arm["sleep"]
    require(type(event) is dict, "sleep guarded event missing")
    accepted = event["outcome"] == "accepted"
    before, after = event["guard_pre_accuracy"], event["guard_post_accuracy"]
    require(
        event["outcome"] in {"accepted", "rolled_back"}
        and 0 <= before <= 1
        and 0 <= after <= 1
        and accepted == (after >= before),
        "sleep guard acceptance differs",
    )
    counts(
        event,
        {
            "guard_role_hash": row["roles"][0]["hashes"]["inner_guard"],
            "guard_evaluations": 2,
            "replay_updates": 0,
            "width_before": 8,
            "width_after": 8,
            "proposed_width": 8,
        },
        "sleep guard work",
    )
    same_json(witness["outcome"], event["outcome"], "sleep raw/witness outcome")
    for endpoint, field in (("before", "width_before"), ("after", "width_after")):
        same_json(witness[endpoint]["width"], event[field], "sleep raw/witness capacity")
    same_json(witness["after"], row["after_a"][name], "sleep complete held A state")
    same_json(witness["before"]["clocks"], clocks(12), "sleep before clocks")
    same_json(
        lineage(witness["before"]),
        lineage(row["initial"][name]),
        "sleep initial lineage continuity",
    )
    for stage, index in (("after_a", 12), ("after_b", 24)):
        same_json(
            row[stage][name]["clocks"],
            clocks(index, int(accepted), 0, 12 if accepted else 0),
            "sleep endpoint clocks",
        )
    _verify_sleep_lineage(arm, event, witness, accepted)
    _verify_sleep_effects(arm, event, accepted)


def _verify_sleep_effects(arm: dict[str, Any], event: dict[str, Any], accepted: bool) -> None:
    name = arm["name"]
    if name in {"neutral_sham", "gating_sham", "gating_reset"}:
        same_json(
            arm["pre_sleep_parameter_sha256"], arm["post_sleep_parameter_sha256"], "no-weight sleep"
        )
    if name == "homeostasis_only" and accepted:
        require(
            arm["pre_sleep_parameter_sha256"] != arm["post_sleep_parameter_sha256"],
            "accepted homeostasis inactive",
        )
    if name == "gating_reset" and accepted:
        require(
            0 < event["chemical_proposed_mean"] < event["chemical_before_mean"],
            "chemical reset inactive",
        )
    for key in ("chemical_before_mean", "chemical_proposed_mean", "chemical_final_mean"):
        require(event[key] >= 0, "negative sleep chemical summary")
    same_json(
        event["chemical_final_mean"],
        event["chemical_proposed_mean"] if accepted else event["chemical_before_mean"],
        "sleep chemical commit/rollback",
    )


def _verify_sleep_lineage(
    arm: dict[str, Any], event: dict[str, Any], witness: dict[str, Any], accepted: bool
) -> None:
    before = lineage(witness["before"])
    pairs, removed = event["proposed_split_pairs"], event["proposed_removed_prune_ids"]
    structural = arm["name"] == "structure_only"
    require(
        len(pairs) == len(removed) == int(structural), "sleep structural proposal count differs"
    )
    ids, parents = before["neuron_ids"], before["parent_ids"]
    for parent, child in pairs:
        require(parent in ids and child == before["next_neuron_id"], "sleep split lineage differs")
    proposed_ids = ids + [child for _, child in pairs]
    require(
        len(set(removed)) == len(removed) and set(removed) <= set(proposed_ids),
        "sleep prune lineage differs",
    )
    same_json(event["applied_split_pairs"], pairs if accepted else [], "sleep split commit")
    same_json(event["applied_removed_prune_ids"], removed if accepted else [], "sleep prune commit")
    peak = 8 + len(pairs)
    counts(event, {"transient_peak_width": peak}, "sleep transient width")
    counts(arm, {"width_peak": peak, "parameters_peak": 4 * peak + 1}, "sleep transient parameters")
    expected = before
    if accepted:
        mapping = dict(zip(proposed_ids, parents + [parent for parent, _ in pairs], strict=True))
        kept = [value for value in proposed_ids if value not in removed]
        expected = {
            "neuron_ids": kept,
            "parent_ids": [mapping[value] for value in kept],
            "next_neuron_id": before["next_neuron_id"] + len(pairs),
        }
    same_json(lineage(witness["after"]), expected, "sleep applied lineage")
