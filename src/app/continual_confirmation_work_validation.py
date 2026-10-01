"""Independently verify complete unscored family work and held state links.

Inputs are finite saved JSON and the exact P6.7a scope. Outputs are derived
work records, never scores. The public gate requires every reserved row; the
private seed seam serves declared development tests. No models, sources,
training, evaluation or IO; runtime/artifact enforcement remains separate.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from src.app import continual_combined_factor_validation as combined
from src.app import continual_parent_factor_validation as parent
from src.app import continual_schedule_factor_validation as schedule
from src.app.continual_confirmation_fact_schema import verify_raw_schema
from src.app.continual_confirmation_json import (
    canonical_dataclass,
    canonical_mapping,
    finite_json,
    require,
    same_json,
)
from src.app.continual_confirmation_manifest import ConfirmationFamily, ConfirmationManifest
from src.app.continual_confirmation_simple_work import (
    clocks,
    lineage,
    verify_gating,
    verify_replay,
    verify_sleep,
)
from src.app.continual_confirmation_validation import verify_confirmation_envelope


@dataclass(frozen=True)
class SeedWork:
    family: str
    seed: int
    cells: int
    wake_updates: int
    applied_replay_updates: int
    rejected_executed_replay_updates: int
    guarded_attempts: int
    guard_evaluations: int
    guard_examples: int
    maximum_transient_width: int
    retained_array_bytes_before_copies: int

    @property
    def executed_optimizer_updates(self) -> int:
        return (
            self.wake_updates + self.applied_replay_updates + self.rejected_executed_replay_updates
        )


def verify_confirmation_payload(
    payload: dict[str, Any], manifest: ConfirmationManifest
) -> tuple[SeedWork, ...]:
    """Require the whole scientific train gate; resources/IO/final stay sealed."""
    verify_confirmation_envelope(payload, manifest)
    try:
        families = {item.name: item for item in manifest.families}
        work = tuple(
            _verify_seed_work(row, families[row["family"]]) for row in payload["seed_results"]
        )
        for family in manifest.families:
            rows = [item for item in work if item.family == family.name]
            require(
                sum(item.wake_updates for item in rows) == family.wake_updates,
                "family wake bound differs",
            )
            require(
                sum(item.executed_optimizer_updates for item in rows)
                <= family.maximum_optimizer_updates,
                "family executed work cap exceeded",
            )
            require(
                sum(item.guarded_attempts for item in rows) <= family.maximum_guarded_attempts,
                "family guard cap exceeded",
            )
        require(
            sum(item.executed_optimizer_updates for item in work) <= manifest.max_optimizer_updates,
            "joint optimizer cap exceeded",
        )
        return work
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("confirmation JSON malformed raw work/state facts") from error


def _verify_seed_work(row: dict[str, Any], family: ConfirmationFamily) -> SeedWork:
    finite_json(row)
    raw = row["legacy_train_facts"]
    verify_raw_schema(raw, family.name)
    manifest = json.loads(family.development_manifest_json)
    simple = {"gating": verify_gating, "replay": verify_replay, "sleep": verify_sleep}
    periodic = {
        "schedule": schedule._verify_seed,
        "combined": combined._verify_seed,
        "parent": parent._verify_seed,
    }
    if family.name in simple:
        simple[family.name](row, manifest)
    else:
        periodic[family.name](raw, manifest)
        if family.name == "schedule":
            _verify_schedule_state_links(row)
        else:
            _verify_periodic_state_links(row, family)
    _verify_capacity_links(row, family)
    _verify_baseline_traffic(row, family)
    _verify_memory_links(row, family)
    observed = _summarize_work(row, family)
    same_json(
        observed.retained_array_bytes_before_copies,
        family.retained_array_bytes_per_seed_before_copies,
        "declared retained array storage",
    )
    return observed


def _verify_capacity_links(row: dict[str, Any], family: ConfirmationFamily) -> None:
    simple = family.name in {"gating", "replay"}
    methods = row["legacy_train_facts"]["arms" if family.name == "sleep" else "methods"]
    for method in methods:
        name = method["method" if simple else "name"]
        for stage, suffix in (
            ("initial", "start" if simple else "initial"),
            ("after_b", "end" if simple else "final"),
        ):
            width = method[f"hidden_width_{suffix}" if simple else f"width_{suffix}"]
            parameters = method[f"parameter_count_{suffix}" if simple else f"parameters_{suffix}"]
            same_json(width, row[stage][name]["width"], "raw/held capacity link")
            same_json(
                parameters, row[stage][name]["parameter_count"], "raw/held parameter-count link"
            )


def _owner(checkpoint: dict[str, Any]) -> dict[str, Any]:
    state = canonical_dataclass(
        checkpoint["state"], "src.core.circadian_predictive_coding.CircadianNetworkSnapshot"
    )
    return canonical_mapping(state["state"])


def _verify_baseline_traffic(row: dict[str, Any], family: ConfirmationFamily) -> None:
    # Both baseline train_epoch implementations record traffic once per applied
    # update. Prediction does not record it; these fixed baselines never reset it.
    for stage, index in (("after_a", 12), ("after_b", 24)):
        for name, checkpoint in row[stage].items():
            if checkpoint["clocks"] is not None:
                continue
            replay = _baseline_replay_updates(row, name, index, family.name)
            same_json(
                canonical_mapping(checkpoint["state"])["_traffic_steps"],
                index + replay,
                "baseline applied-work traffic link",
            )


def _baseline_replay_updates(row: dict[str, Any], name: str, index: int, family: str) -> int:
    raw = row["legacy_train_facts"]
    if family == "replay":
        return sum(
            len(item["applied_ids_by_method"][name])
            for item in raw["boundaries"]
            if item["epoch"] + (12 if item["phase"] == "b" else 0) <= index
        )
    if family == "schedule":
        model, _, policy = name.partition("_")
        return sum(
            len(decision["applied_ids_by_method"][model])
            for item in raw["opportunities"][:index]
            for decision in item["decisions"]
            if decision["policy"] == policy
        )
    if family == "combined":
        return sum(
            len(item["applied_control_replay_ids"].get(name, []))
            for item in raw["opportunities"][:index]
        )
    return 0


def _exposure_link(checkpoint: dict[str, Any], applied_ids: set[str]) -> None:
    same_json(
        _owner(checkpoint)["_replay_exposed_ids"],
        {"set": sorted(applied_ids)},
        "committed replay exposure/state link",
    )


def _verify_schedule_state_links(row: dict[str, Any]) -> None:
    accepted = dict.fromkeys(("periodic", "adaptive", "no_sleep"), 0)
    last = accepted.copy()
    exposed: dict[str, set[str]] = {name: set() for name in accepted}
    for index, opportunity in enumerate(row["legacy_train_facts"]["opportunities"], start=1):
        for offset, decision in enumerate(opportunity["decisions"]):
            policy = decision["policy"]
            name = f"neutral_{policy}"
            witness = row["supplemental_guards"][3 * (index - 1) + offset]
            same_json(witness["outcome"], decision["outcome"], "schedule raw/witness outcome")
            same_json(
                witness["before"]["clocks"],
                clocks(index, accepted[policy], 2 * accepted[policy], last[policy]),
                "schedule complete before/work clocks",
            )
            owner = _owner(witness["before"])
            window = owner["config"]["fields"]["sleep_energy_window"]
            same_json(
                owner["_energy_history"]["list"][-window:],
                decision["energy_window"],
                "schedule trigger/state energy history",
            )
            _exposure_link(witness["before"], exposed[policy])
            if decision["outcome"] == "accepted":
                accepted[policy] += 1
                last[policy] = index
                exposed[policy].update(decision["applied_ids_by_method"]["neutral"])
            same_json(
                witness["after"]["clocks"],
                clocks(index, accepted[policy], 2 * accepted[policy], last[policy]),
                "schedule complete after/work clocks",
            )
            _exposure_link(witness["after"], exposed[policy])
            if index in (12, 24):
                stage = "after_a" if index == 12 else "after_b"
                same_json(witness["after"], row[stage][name], "schedule complete held boundary")


def _verify_periodic_state_links(row: dict[str, Any], family: ConfirmationFamily) -> None:
    raw = row["legacy_train_facts"]
    methods = {item["name"]: item for item in raw["methods"]}
    previous = {
        name: lineage(checkpoint)
        for name, checkpoint in row["initial"].items()
        if checkpoint["lineage"] is not None
    }
    for index, opportunity in enumerate(raw["opportunities"], start=1):
        for decision in opportunity["decisions"]:
            name = decision["name"]
            before = (
                decision["lineage_before"]
                if family.name == "combined"
                else decision["before"]["lineage"]
            )
            after = (
                decision["lineage_after"]
                if family.name == "combined"
                else decision["after"]["lineage"]
            )
            same_json(before, previous[name], "periodic lineage continuity")
            previous[name] = after
            if index in (12, 24):
                stage = "after_a" if index == 12 else "after_b"
                _verify_decision_boundary(decision, row[stage][name], family.name, after)
        if index in (12, 24):
            _verify_epoch_state_links(
                row, opportunity, methods, "after_a" if index == 12 else "after_b", family.name
            )


def _verify_decision_boundary(
    decision: dict[str, Any], checkpoint: dict[str, Any], family: str, after: dict[str, Any]
) -> None:
    state = (
        {"state_sha256": decision["state_sha256_after"], "clocks": decision["clocks_after"]}
        if family == "combined"
        else decision["after"]
    )
    same_json(state["state_sha256"], checkpoint["state_sha256"], "decision complete held state")
    same_json(state["clocks"], checkpoint["clocks"], "decision held clock view")
    same_json(after, lineage(checkpoint), "decision held lineage view")
    if family == "parent":
        same_json(state["selector"], checkpoint["selector"], "decision held selector view")


def _verify_epoch_state_links(
    row: dict[str, Any],
    opportunity: dict[str, Any],
    methods: dict[str, Any],
    stage: str,
    family: str,
) -> None:
    for name, checkpoint in row[stage].items():
        same_json(
            opportunity["after_epoch_widths"][name], checkpoint["width"], "held epoch capacity"
        )
        if checkpoint["clocks"] is None:
            continue
        same_json(
            opportunity["after_epoch_state_sha256"][name],
            checkpoint["state_sha256"],
            "epoch complete held state",
        )
        method = methods[name]
        if stage == "after_b":
            for source, target in (
                ("final_state_sha256", "state_sha256"),
                ("final_clocks", "clocks"),
                ("retention", "retention"),
            ):
                same_json(method[source], checkpoint[target], "method complete held state/view")
            if family == "parent":
                same_json(
                    method["final_selector"], checkpoint["selector"], "method held selector view"
                )
        elif family == "parent":
            same_json(
                method["after_a_state_sha256"], checkpoint["state_sha256"], "method held A state"
            )


def _applied_ids(row: dict[str, Any], name: str, index: int) -> set[str]:
    family, raw = row["family"], row["legacy_train_facts"]
    ids: set[str] = set()
    if family == "replay":
        for item in raw["boundaries"]:
            global_epoch = item["epoch"] + (12 if item["phase"] == "b" else 0)
            if global_epoch <= index:
                ids.update(item["applied_ids_by_method"][name])
    elif family == "schedule":
        policy = name.removeprefix("neutral_")
        for item in raw["opportunities"][:index]:
            decision = next(value for value in item["decisions"] if value["policy"] == policy)
            ids.update(decision["applied_ids_by_method"]["neutral"])
    elif family == "combined":
        for item in raw["opportunities"][:index]:
            decision = next((value for value in item["decisions"] if value["name"] == name), None)
            ids.update(
                decision["applied_replay_ids"]
                if decision is not None
                else item["applied_control_replay_ids"].get(name, [])
            )
    return ids


def _verify_memory_links(row: dict[str, Any], family: ConfirmationFamily) -> None:
    if family.name not in {"replay", "schedule", "combined", "parent"}:
        return
    for stage, index, phase in (("after_a", 12, "a"), ("after_b", 24, "b")):
        raw = row["legacy_train_facts"]
        boundary = (
            next(item for item in reversed(raw["boundaries"]) if item["phase"] == phase)
            if family.name == "replay"
            else raw["opportunities"][index - 1]
        )
        for name, checkpoint in row[stage].items():
            if checkpoint["retention"] is None:
                continue
            same_json(
                checkpoint["retention"],
                {
                    "sample_ids": sorted(boundary["retained_order_ids"]),
                    "example_count": 8,
                    "retained_bytes": 192,
                },
                "held retained FIFO supply",
            )
            _exposure_link(checkpoint, _applied_ids(row, name, index))


def _summarize_work(row: dict[str, Any], family: ConfirmationFamily) -> SeedWork:
    name, raw = family.name, row["legacy_train_facts"]
    methods = raw["arms" if name == "sleep" else "methods"]
    replay_field = (
        "replay_updates" if name in {"gating", "replay", "sleep"} else "applied_replay_updates"
    )
    guards = [
        (item["phase"], decision)
        for item in raw.get("opportunities", [])
        for decision in item["decisions"]
        if (decision["attempted"] if name == "schedule" else decision["event"]["guard"] is not None)
    ]
    attempts = (
        len(guards) if name != "sleep" else sum(item["sleep"] is not None for item in methods)
    )
    examples = (
        sum(2 * (24 if phase == "a" else 12) for phase, _ in guards)
        if name != "sleep"
        else 48 * attempts
    )
    peak_field = "hidden_width_end" if name in {"gating", "replay"} else "width_peak"
    retained = sum(
        checkpoint["retention"]["retained_bytes"]
        for checkpoint in row["after_b"].values()
        if checkpoint["retention"] is not None
    )
    if name in {"replay", "schedule", "combined", "parent"}:
        retained += 192  # One shared FIFO, declared separately from live model copies.
    return SeedWork(
        name,
        row["seed"],
        len(methods),
        sum(item["wake_updates"] for item in methods),
        sum(item[replay_field] for item in methods),
        sum(item.get("rejected_executed_replay_updates", 0) for item in methods),
        attempts,
        2 * attempts,
        examples,
        max(item[peak_field] for item in methods),
        retained,
    )
