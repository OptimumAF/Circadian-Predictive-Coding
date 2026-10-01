"""Rederive parent ordering, guard outcomes, topology and work from finite facts.

Inputs are a complete unscored JSON payload and its fixed manifest. Output is
success or a useful validation error. No training, data creation, scores or IO.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Any

import numpy as np

from src.app.continual_combined_factor_validation import _hash, _telemetry
from src.app.continual_parent_factor_manifest import GROWTH_ARMS, fixed_parent_manifest
from src.app.continual_schedule_factor_validation import ROLE_COUNTS
from src.core.sleep_telemetry import SleepBudgets, SleepEventTelemetry


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"P6.3 parent {message}")


def _rng_hash(rng: np.random.Generator) -> str:
    return sha256(
        json.dumps(rng.bit_generator.state, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


@dataclass
class _Selection:
    mode: str
    seed: int
    rng: np.random.Generator
    cursor: int = 0
    calls: int = 0
    last: dict[str, Any] | None = None

    def view(self) -> dict[str, Any]:
        return {
            "settings": {"mode": self.mode, "seed": self.seed, "initial_cursor_id": 0},
            "cursor_id": self.cursor,
            "selection_calls": self.calls,
            "rng_sha256": _rng_hash(self.rng),
            "last_decision": self.last,
        }

    def choose(self, ids: list[int], scores: list[float]) -> int:
        # Independent ordering uses recorded original scores, stable IDs and
        # the declared PCG64 stream, never the model's selector implementation.
        cursor, rng_hash = self.cursor, _rng_hash(self.rng)
        if self.mode == "usage":
            parent = ids[int(np.argsort(np.array(scores))[::-1][0])]
        elif self.mode == "scheduled":
            parent = next((value for value in sorted(ids) if value >= self.cursor), min(ids))
            self.cursor = parent + 1
        else:
            parent = int(self.rng.permutation(np.array(sorted(ids), dtype=np.int64))[0])
        self.calls += 1
        self.last = {
            "mode": self.mode,
            "eligible_parent_ids": sorted(ids),
            "preferred_parent_ids": sorted(ids),
            "selected_parent_ids": [parent],
            "cursor_before": cursor,
            "cursor_after": self.cursor,
            "rng_sha256_before": rng_hash,
            "rng_sha256_after": _rng_hash(self.rng),
        }
        return parent


@dataclass
class _Counts:
    attempts: int = 0
    accepted: int = 0
    guard_examples: int = 0
    last_sleep: int = 0
    peak: int = 8


def _clocks(index: int, count: _Counts) -> dict[str, int]:
    return {
        "wake_batches": index,
        "wake_examples": 72 * min(index, 12) + 36 * max(index - 12, 0),
        "wake_batches_since_sleep": index - count.last_sleep,
        "replay_updates": 0,
        "sleep_events": count.accepted,
    }


def verify_parent_payload(payload: dict[str, Any], manifest: dict[str, Any]) -> None:
    try:
        _require(
            manifest == json.loads(json.dumps(asdict(fixed_parent_manifest()))),
            "independent manifest differs",
        )
        _require(
            set(payload)
            == {
                "protocol_id",
                "manifest",
                "seed_results",
                "outer_selection_scored",
                "final_released",
            }
            and payload["manifest"] == manifest
            and payload["protocol_id"] == manifest["protocol_id"]
            and payload["outer_selection_scored"] is False
            and payload["final_released"] is False
            and [row["seed"] for row in payload["seed_results"]] == manifest["seeds"],
            "protocol, cells or evaluation seal differs",
        )
        for seed in payload["seed_results"]:
            _verify_seed(seed, manifest)
        _require(
            sum(row["executed_optimizer_updates"] for row in payload["seed_results"])
            == 576
            <= manifest["max_optimizer_updates"],
            "executed optimizer work differs",
        )
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("P6.3 parent malformed or incomplete train facts") from error


def _verify_seed(seed: dict[str, Any], manifest: dict[str, Any]) -> None:
    specs = {arm["name"]: arm for arm in manifest["arms"]}
    _require(
        set(seed)
        == {
            "seed",
            "role_hashes",
            "role_counts",
            "methods",
            "opportunities",
            "executed_optimizer_updates",
            "final_released",
        }
        and seed["final_released"] is False
        and seed["role_counts"] == ROLE_COUNTS
        and set(seed["role_hashes"]) == set(ROLE_COUNTS)
        and all(_hash(value) for value in seed["role_hashes"].values())
        and len(set(seed["role_hashes"].values())) == 6
        and [row["name"] for row in seed["methods"]] == list(specs)
        and len(seed["opportunities"]) == 24
        and seed["executed_optimizer_updates"] == 192,
        "role, seed, method or work cells differ",
    )
    counts = {name: _Counts(peak=spec["width"]) for name, spec in specs.items()}
    selector_seed = seed["seed"] + manifest["selector_seed_offset"]
    selection = {
        name: _Selection(
            specs[name]["parent_mode"],
            selector_seed,
            np.random.Generator(np.random.PCG64(selector_seed)),
        )
        for name in GROWTH_ARMS
    }
    lineage = {
        name: {"neuron_ids": list(range(8)), "parent_ids": [None] * 8, "next_neuron_id": 8}
        for name in GROWTH_ARMS
    }
    widths = {name: spec["width"] for name, spec in specs.items()}
    supply: dict[str, list[str]] = {}
    for index, opportunity in enumerate(seed["opportunities"], 1):
        _verify_opportunity(opportunity, index, seed, specs, widths, supply)
        for row in opportunity["decisions"]:
            name = row["name"]
            lineage[name] = _verify_decision(
                row, opportunity, seed, manifest, selection[name], lineage[name], counts[name]
            )
        widths = opportunity["after_epoch_widths"].copy()
    _verify_methods(seed, specs, counts, selection)


def _verify_opportunity(
    row: dict[str, Any],
    index: int,
    seed: dict[str, Any],
    specs: dict[str, Any],
    widths: dict[str, int],
    supply: dict[str, list[str]],
) -> None:
    phase = "a" if index <= 12 else "b"
    circadian = {name for name, spec in specs.items() if spec["config"] is not None}
    _require(
        set(row)
        == {
            "phase",
            "epoch",
            "global_epoch",
            "train_role_hash",
            "wake",
            "retained_order_ids",
            "retained_examples",
            "retained_bytes",
            "decisions",
            "after_epoch_parameter_sha256",
            "after_epoch_state_sha256",
            "after_epoch_widths",
        }
        and row["phase"] == phase
        and row["epoch"] == (index - 1) % 12 + 1
        and row["global_epoch"] == index
        and row["train_role_hash"] == seed["role_hashes"][f"{phase}_train"]
        and row["retained_examples"] == len(row["retained_order_ids"]) == 8
        and row["retained_bytes"] == 192
        and len(set(row["retained_order_ids"])) == 8
        and all(_hash(value) for value in row["retained_order_ids"])
        and [wake["name"] for wake in row["wake"]] == list(specs)
        and [decision["name"] for decision in row["decisions"]] == list(GROWTH_ARMS)
        and set(row["after_epoch_parameter_sha256"]) == set(specs)
        and set(row["after_epoch_state_sha256"]) == circadian
        and set(row["after_epoch_widths"]) == set(specs),
        "opportunity/supply/model cells differ",
    )
    if phase in supply:
        _require(row["retained_order_ids"] == supply[phase], "FIFO changed within fixed phase")
    supply[phase] = row["retained_order_ids"]
    decisions = {item["name"]: item for item in row["decisions"]}
    for wake in row["wake"]:
        name, spec = wake["name"], specs[wake["name"]]
        _require(
            set(wake)
            == {
                "name",
                "width",
                "parameters",
                "parameter_sha256",
                "state_sha256",
                "minimum_plasticity",
                "reward_scale",
            }
            and wake["width"] == widths[name]
            and wake["parameters"] == 4 * wake["width"] + 1
            and spec["min_width"] <= wake["width"] <= spec["max_width"]
            and _hash(wake["parameter_sha256"])
            and (_hash(wake["state_sha256"]) if name in circadian else wake["state_sha256"] is None)
            and _hash(row["after_epoch_parameter_sha256"][name])
            and wake["minimum_plasticity"] == (1.0 if name in circadian else None)
            and wake["reward_scale"] == (1.0 if name in circadian else None),
            "wake/capacity/state differs",
        )
        if name in decisions:
            item = decisions[name]
            _require(
                item["before"]["parameter_sha256"] == wake["parameter_sha256"]
                and item["before"]["state_sha256"] == wake["state_sha256"]
                and item["event"]["before_width"] == wake["width"]
                and item["after"]["parameter_sha256"] == row["after_epoch_parameter_sha256"][name]
                and item["after"]["state_sha256"] == row["after_epoch_state_sha256"][name]
                and item["event"]["final_width"] == row["after_epoch_widths"][name],
                "epoch/decision checkpoint differs",
            )
        else:
            _require(
                row["after_epoch_widths"][name] == wake["width"]
                and row["after_epoch_parameter_sha256"][name] == wake["parameter_sha256"],
                "fixed reference changed outside wake",
            )
        if name in circadian:
            _require(_hash(row["after_epoch_state_sha256"][name]), "state fingerprint is invalid")
    _require(
        row["after_epoch_parameter_sha256"]["pc_off"]
        == row["after_epoch_parameter_sha256"]["neutral_off"],
        "neutral PC parity differs",
    )


def _verify_decision(
    row: dict[str, Any],
    opportunity: dict[str, Any],
    seed: dict[str, Any],
    manifest: dict[str, Any],
    selection: _Selection,
    lineage: dict[str, Any],
    count: _Counts,
) -> dict[str, Any]:
    index = opportunity["global_epoch"]
    due, add = index % 4 == 0, int(index in manifest["add_epochs"])
    event = _telemetry(row["event"])
    before, proposed, after = row["before"], row["proposed"], row["after"]
    _require(
        set(row)
        == {"name", "requested_add_count", "split_scores", "event", "before", "proposed", "after"}
        and row["requested_add_count"] == add
        and event.completed_epoch == index
        and event.wake_batches == index
        and before["lineage"] == lineage
        and before["clocks"] == _clocks(index, count)
        and before["selector"] == selection.view()
        and len(row["split_scores"]) == len(lineage["neuron_ids"])
        and all(type(value) in (int, float) and isfinite(value) for value in row["split_scores"]),
        "decision counts/scores/before state differ",
    )
    for state in (before, proposed, after):
        _require(
            set(state) == {"state_sha256", "parameter_sha256", "lineage", "clocks", "selector"}
            and _hash(state["state_sha256"])
            and _hash(state["parameter_sha256"]),
            "decision fingerprints differ",
        )
    _require(
        event.budgets == SleepBudgets(int(due and index / 24 < 0.85), 0, 0, None)
        and event.replay.proposed_updates
        == event.replay.applied_updates
        == event.replay.proposed_examples
        == event.replay.applied_examples
        == 0
        and not event.changes.proposed_prune_ids
        and not event.changes.proposed_scheduled_prune_ids
        and not event.changes.proposed_removed_prune_ids
        and not event.changes.applied_removed_prune_ids
        and not event.changes.applied_scheduled_prune_ids
        and event.before_width == len(lineage["neuron_ids"])
        and event.proposed_width == event.before_width + add <= 13,
        "phase/component budget, replay, pruning or width differs",
    )
    if not due:
        _require(
            event.outcome == "skipped"
            and event.guard is None
            and event.trigger_reason == "not_due"
            and before == proposed == after
            and not event.changes.proposed_split_pairs,
            "not-due decision changed state",
        )
        return lineage
    return _verify_attempt(row, opportunity, seed, selection, lineage, count, event, add)


def _verify_attempt(
    row: dict[str, Any],
    opportunity: dict[str, Any],
    seed: dict[str, Any],
    selection: _Selection,
    lineage: dict[str, Any],
    count: _Counts,
    event: SleepEventTelemetry,
    add: int,
) -> dict[str, Any]:
    index, guard = opportunity["global_epoch"], event.guard
    _require(
        guard is not None
        and event.trigger_reason == "periodic"
        and event.outcome in {"accepted", "rolled_back"},
        "due guard/outcome differs",
    )
    assert guard is not None
    _require(
        guard.pre_accuracy is not None and guard.post_accuracy is not None,
        "guard accuracy is missing",
    )
    assert guard.pre_accuracy is not None and guard.post_accuracy is not None
    _require(
        guard.role == "inner_guard"
        and guard.metric_name == "accuracy"
        and guard.tolerance == 0
        and guard.role_hash == seed["role_hashes"][f"{opportunity['phase']}_inner_guard"]
        and guard.examples_scored == 2 * ROLE_COUNTS[f"{opportunity['phase']}_inner_guard"]
        and (event.outcome == "accepted") == (guard.post_accuracy >= guard.pre_accuracy),
        "guard semantics/exposure differs",
    )
    saved = deepcopy(selection)
    pairs: tuple[tuple[int, int], ...] = ()
    expected_lineage = deepcopy(lineage)
    if add:
        parent = selection.choose(lineage["neuron_ids"], row["split_scores"])
        child = lineage["next_neuron_id"]
        pairs = ((parent, child),)
        expected_lineage["neuron_ids"].append(child)
        expected_lineage["parent_ids"].append(parent)
        expected_lineage["next_neuron_id"] += 1
    proposed_count = _Counts(accepted=count.accepted + 1, last_sleep=index)
    _require(
        row["proposed"]["selector"] == selection.view()
        and event.changes.proposed_split_pairs == pairs
        and row["proposed"]["lineage"] == expected_lineage
        and row["proposed"]["clocks"] == _clocks(index, proposed_count)
        and (bool(add) or row["proposed"]["parameter_sha256"] == row["before"]["parameter_sha256"]),
        "proposed parent/RNG/cursor/lineage/clocks differ",
    )
    count.attempts += 1
    count.guard_examples += guard.examples_scored
    count.peak = max(count.peak, event.proposed_width)
    if event.outcome == "accepted":
        count.accepted += 1
        count.last_sleep = index
        _require(
            event.changes.applied_split_pairs == pairs and row["after"] == row["proposed"],
            "commit differs from proposed state",
        )
    else:
        selection.rng = saved.rng
        selection.cursor, selection.calls, selection.last = saved.cursor, saved.calls, saved.last
        expected_lineage = lineage
        _require(
            not event.changes.applied_split_pairs and row["after"] == row["before"],
            "rollback failed to restore complete state",
        )
    _require(
        row["after"]["selector"] == selection.view()
        and row["after"]["clocks"] == _clocks(index, count)
        and event.final_width == len(expected_lineage["neuron_ids"]),
        "applied state differs",
    )
    return expected_lineage


def _verify_methods(
    seed: dict[str, Any],
    specs: dict[str, Any],
    counts: dict[str, _Counts],
    selection: dict[str, _Selection],
) -> None:
    opportunities = seed["opportunities"]
    methods = {row["name"]: row for row in seed["methods"]}
    for width in (8, 13):
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
        circadian, is_pc = spec["config"] is not None, spec["model_kind"] != "backprop"
        final_width = opportunities[-1]["after_epoch_widths"][name]
        expected = {
            "name": name,
            "initial_parameter_sha256": method["initial_parameter_sha256"],
            "after_a_parameter_sha256": opportunities[11]["after_epoch_parameter_sha256"][name],
            "final_parameter_sha256": opportunities[-1]["after_epoch_parameter_sha256"][name],
            "after_a_state_sha256": opportunities[11]["after_epoch_state_sha256"].get(name),
            "final_state_sha256": opportunities[-1]["after_epoch_state_sha256"].get(name),
            "width_initial": spec["width"],
            "width_final": final_width,
            "width_peak": count.peak,
            "parameters_initial": 4 * spec["width"] + 1,
            "parameters_final": 4 * final_width + 1,
            "parameters_peak": 4 * count.peak + 1,
            "wake_updates": 24,
            "wake_presentations": 1296,
            "wake_inference_loops": 48 if is_pc else 0,
            "wake_example_inference_iterations": 2592 if is_pc else 0,
            "applied_replay_updates": 0,
            "rejected_executed_replay_updates": 0,
            "own_sleep_attempts": count.attempts,
            "committed_sleep_events": count.accepted,
            "guard_evaluations": 2 * count.attempts,
            "guard_examples": count.guard_examples,
            "final_clocks": _clocks(24, count) if circadian else None,
            "retention": {
                "sample_ids": sorted(opportunities[-1]["retained_order_ids"]),
                "example_count": 8,
                "retained_bytes": 192,
            }
            if circadian
            else None,
            "final_selector": selection[name].view() if name in selection else None,
        }
        _require(
            method == expected and _hash(method["initial_parameter_sha256"]),
            "method cost/capacity/checkpoint/selector facts differ",
        )
