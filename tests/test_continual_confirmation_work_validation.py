"""Independent cost/state gates use fixed development fixtures, never reserved data."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json
from typing import Any

import pytest

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_confirmation_training as training
from src.app import continual_confirmation_validation as envelope
from src.app import continual_confirmation_work_validation as work
from src.app.continual_confirmation_manifest import fixed_confirmation_manifest
from src.app.continual_confirmation_validation import _verify_seed_envelope
from src.app.continual_confirmation_json import canonical_mapping, state_digest
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.controlled_parent_selection import ParentControlledCircadianNetwork
from src.core.predictive_coding import PredictiveCodingNetwork


NAMES = ("gating", "replay", "sleep", "schedule", "combined", "parent")


@pytest.fixture(scope="module")
def fixtures() -> tuple[Any, Any]:
    families = tuple(
        replace(item, seeds=(item.development_seeds[0],))
        for item in fixed_confirmation_manifest().families
    )
    rows = training._train_families(families)
    return families, {
        item.facts.family: json.loads(json.dumps(asdict(item.facts), allow_nan=False))
        for item in rows
    }


def _forbid(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("work validation constructed data/model or trained/scored")


def _seal_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    for model in (
        BackpropMLP,
        PredictiveCodingNetwork,
        CircadianPredictiveCodingNetwork,
        ParentControlledCircadianNetwork,
    ):
        monkeypatch.setattr(model, "__init__", _forbid)
        monkeypatch.setattr(model, "train_epoch", _forbid)
        monkeypatch.setattr(model, "predict_proba", _forbid)
    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "compute_accuracy", _forbid)
    monkeypatch.setattr(arrived, "_build_phase_a_roles", _forbid)
    monkeypatch.setattr(arrived, "_build_phase_b_roles", _forbid)


@pytest.mark.parametrize("name", NAMES)
def test_should_independently_verify_each_actual_development_work_body(
    name: str, fixtures: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    families, facts = fixtures
    family = next(item for item in families if item.name == name)
    _seal_validation(monkeypatch)
    _verify_seed_envelope(facts[name], family)
    observed = work._verify_seed_work(facts[name], family)
    assert observed.family == name and observed.seed == family.seeds[0]
    assert observed.cells == len(family.arms)
    assert observed.wake_updates == 24 * len(family.arms)
    assert observed.executed_optimizer_updates >= observed.wake_updates
    assert (
        observed.retained_array_bytes_before_copies
        == family.retained_array_bytes_per_seed_before_copies
    )


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize(
    "corruption", ["unknown", "float_count", "bool_count", "wake_cost", "latent_cost"]
)
def test_should_refuse_ambiguous_or_forged_raw_work(
    name: str, corruption: str, fixtures: tuple[Any, Any]
) -> None:
    families, facts = fixtures
    family = next(item for item in families if item.name == name)
    row = deepcopy(facts[name])
    method = row["legacy_train_facts"]["arms" if name == "sleep" else "methods"][0]
    if corruption == "unknown":
        method["unknown_fact"] = 0
    elif corruption == "float_count":
        method["wake_updates"] = float(method["wake_updates"])
    elif corruption == "bool_count":
        method["wake_updates"] = True
    elif corruption == "wake_cost":
        method["wake_updates"] -= 1
    else:
        key = (
            "latent_iterations"
            if name == "gating"
            else "latent_inference_loops"
            if name == "sleep"
            else "wake_inference_loops"
        )
        method[key] += 1
    with pytest.raises(ValueError):
        work._verify_seed_work(row, family)


@pytest.mark.parametrize("name", ["sleep", "schedule", "combined", "parent"])
def test_should_bind_guard_outcomes_to_full_state_even_when_parameters_match(
    name: str, fixtures: tuple[Any, Any]
) -> None:
    families, facts = fixtures
    family = next(item for item in families if item.name == name)
    row = deepcopy(facts[name])
    if name in {"sleep", "schedule"}:
        witness = row["supplemental_guards"][0]
        witness["outcome"] = "rolled_back" if witness["outcome"] == "accepted" else "accepted"
    else:
        row["legacy_train_facts"]["opportunities"][11]["after_epoch_state_sha256"][
            "neutral_off"
        ] = "0" * 64
    with pytest.raises(ValueError):
        work._verify_seed_work(row, family)


@pytest.mark.parametrize("name", ["sleep", "schedule", "combined", "parent"])
def test_should_accept_and_count_forced_rejected_work_without_scoring(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    family = next(item for item in fixed_confirmation_manifest().families if item.name == name)
    family = replace(family, seeds=(family.development_seeds[0],))
    calls: dict[CircadianPredictiveCodingNetwork, int] = {}

    def reject(model: CircadianPredictiveCodingNetwork, *args: Any) -> float:
        count = calls.get(model, 0)
        calls[model] = count + 1
        return 1.0 if count % 2 == 0 else 0.0

    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "compute_accuracy", reject)
    held = training._train_families((family,))[0]
    row = json.loads(json.dumps(asdict(held.facts), allow_nan=False))
    _seal_validation(monkeypatch)
    _verify_seed_envelope(row, family)
    observed = work._verify_seed_work(row, family)
    assert observed.applied_replay_updates == 0
    assert observed.guarded_attempts > 0
    assert observed.rejected_executed_replay_updates == (
        12 if name == "schedule" else 72 if name == "combined" else 0
    )


@pytest.mark.parametrize("name", ["replay", "schedule", "combined", "parent"])
@pytest.mark.parametrize("corruption", ["supply", "exposure", "retention", "capacity"])
def test_should_refuse_changed_memory_capacity_or_exposure_links(
    name: str, corruption: str, fixtures: tuple[Any, Any]
) -> None:
    families, facts = fixtures
    family = next(item for item in families if item.name == name)
    row = deepcopy(facts[name])
    checkpoint = next(item for item in row["after_b"].values() if item["retention"] is not None)
    if corruption == "retention":
        checkpoint["retention"]["sample_ids"][0] = "0" * 64
    elif corruption == "capacity":
        checkpoint["width"] += 1
        checkpoint["parameter_count"] += 4
    elif corruption == "exposure":
        owner = checkpoint["state"]["fields"]["state"]
        next(item for item in owner["dict"] if item[0] == "_replay_exposed_ids")[1] = {
            "set": ["0" * 64]
        }
        checkpoint["state_sha256"] = state_digest(checkpoint["state"])
    else:
        opportunity = row["legacy_train_facts"][
            "boundaries" if name == "replay" else "opportunities"
        ][-1]
        opportunity["retained_order_ids"][0] = "0" * 64
    with pytest.raises(ValueError):
        work._verify_seed_work(row, family)


@pytest.mark.parametrize("name", ["schedule", "combined", "parent"])
@pytest.mark.parametrize(
    "corruption", ["rejected_cost", "guard_cost", "transient_cost", "unknown_nested"]
)
def test_should_refuse_forged_periodic_cost_or_nested_schema(
    name: str, corruption: str, fixtures: tuple[Any, Any]
) -> None:
    families, facts = fixtures
    family = next(item for item in families if item.name == name)
    row = deepcopy(facts[name])
    raw = row["legacy_train_facts"]
    if corruption == "unknown_nested":
        decision = raw["opportunities"][0]["decisions"][0]
        target = decision if name == "schedule" else decision["event"]["budgets"]
        target["new_privilege"] = 0
    else:
        method = next(item for item in raw["methods"] if item["guard_evaluations"] > 0)
        field = {
            "rejected_cost": "rejected_executed_replay_updates",
            "guard_cost": "guard_evaluations",
            "transient_cost": "parameters_peak",
        }[corruption]
        method[field] += 1
        if corruption == "rejected_cost":
            raw["executed_optimizer_updates"] += 1
    with pytest.raises(ValueError):
        work._verify_seed_work(row, family)


@pytest.mark.parametrize("corruption", ["history", "clock", "guard", "outcome"])
def test_should_bind_schedule_trigger_and_guard_to_full_witnesses(
    corruption: str, fixtures: tuple[Any, Any]
) -> None:
    families, facts = fixtures
    family = next(item for item in families if item.name == "schedule")
    row = deepcopy(facts["schedule"])
    if corruption == "history":
        for witness in row["supplemental_guards"]:
            if witness["name"] == "neutral_no_sleep":
                owner = witness["before"]["state"]["fields"]["state"]
                history = canonical_mapping(owner)["_energy_history"]["list"]
                history[-1] += 0.125
                witness["before"]["state_sha256"] = state_digest(witness["before"]["state"])
    elif corruption == "clock":
        row["supplemental_guards"][0]["before"]["clocks"]["sleep_events"] += 1
    elif corruption == "outcome":
        row["supplemental_guards"][0]["outcome"] = "rolled_back"
    else:
        guard = row["legacy_train_facts"]["opportunities"][3]["decisions"][0]
        guard["guard_role_hash"] = row["roles"][0]["hashes"]["outer_selection"]
    with pytest.raises(ValueError):
        work._verify_seed_work(row, family)


@pytest.mark.parametrize(
    "corruption", ["proposal", "prune", "transient", "chemistry", "guard", "capacity"]
)
def test_should_refuse_forged_sleep_structural_guard_or_state_fact(
    corruption: str, fixtures: tuple[Any, Any]
) -> None:
    families, facts = fixtures
    family = next(item for item in families if item.name == "sleep")
    row = deepcopy(facts["sleep"])
    arm = next(
        item for item in row["legacy_train_facts"]["arms"] if item["name"] == "structure_only"
    )
    event = arm["sleep"]
    if corruption == "proposal":
        event["proposed_split_pairs"][0][1] += 1
    elif corruption == "prune":
        event["proposed_removed_prune_ids"][0] = 999
    elif corruption == "transient":
        event["transient_peak_width"] += 1
        arm["width_peak"] += 1
        arm["parameters_peak"] += 4
    elif corruption == "chemistry":
        event["chemical_final_mean"] += 0.5
    elif corruption == "guard":
        event["guard_role_hash"] = row["roles"][0]["hashes"]["outer_selection"]
    else:
        row["after_b"]["structure_only"]["width"] += 1
        row["after_b"]["structure_only"]["parameter_count"] += 4
    with pytest.raises(ValueError):
        work._verify_seed_work(row, family)


@pytest.mark.parametrize("corruption", ["selector", "lineage", "rng", "metric", "unknown_event"])
def test_should_refuse_forged_parent_state_or_event(
    corruption: str, fixtures: tuple[Any, Any]
) -> None:
    families, facts = fixtures
    family = next(item for item in families if item.name == "parent")
    row = deepcopy(facts["parent"])
    decision = row["legacy_train_facts"]["opportunities"][3]["decisions"][0]
    if corruption == "selector":
        decision["proposed"]["selector"]["cursor_id"] += 1
    elif corruption == "lineage":
        decision["proposed"]["lineage"]["parent_ids"][-1] = 999
    elif corruption == "rng":
        decision["proposed"]["selector"]["rng_sha256"] = "0" * 64
    elif corruption == "metric":
        decision["event"]["guard"]["metric_name"] = "cross_entropy"
    else:
        decision["event"]["chemistry_before"]["primary"]["unknown"] = 0
    with pytest.raises(ValueError):
        work._verify_seed_work(row, family)


def _metadata_envelope() -> dict[str, Any]:
    manifest = fixed_confirmation_manifest()
    return {
        "manifest": json.loads(json.dumps(asdict(manifest))),
        "protocol_id": "continual_mechanism_confirmation_train_only_v1",
        "seed_results": [
            {"family": family.name, "seed": seed}
            for family in manifest.families
            for seed in family.seeds
        ],
        "all_a_completed_before_first_b": True,
        "outer_selection_scored": False,
        "final_released": False,
    }


def test_should_refuse_nested_sleep_final_release_even_with_outer_envelope_sealed(
    fixtures: tuple[Any, Any],
) -> None:
    families, facts = fixtures
    family = next(item for item in families if item.name == "sleep")
    row = deepcopy(facts["sleep"])
    row["legacy_train_facts"]["final_released"] = True
    assert row["final_released"] is False
    with pytest.raises(ValueError, match="sleep legacy final seal"):
        work._verify_seed_work(row, family)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("stage", ["after_a", "after_b"])
def test_should_refuse_resealed_baseline_traffic_inconsistent_with_applied_work(
    name: str, stage: str, fixtures: tuple[Any, Any]
) -> None:
    families, facts = fixtures
    family = next(item for item in families if item.name == name)
    row = deepcopy(facts[name])
    checkpoint = next(item for item in row[stage].values() if item["clocks"] is None)
    owner = checkpoint["state"]
    entry = next(item for item in owner["dict"] if item[0] == "_traffic_steps")
    entry[1] += 1
    checkpoint["state_sha256"] = state_digest(owner)
    with pytest.raises(ValueError, match="baseline applied-work traffic link"):
        work._verify_seed_work(row, family)


def test_should_reject_partial_scientific_scope_before_any_work_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _metadata_envelope()
    payload["seed_results"].pop()
    monkeypatch.setattr(work, "_verify_seed_work", _forbid)
    with pytest.raises(ValueError, match="complete family/seed order"):
        work.verify_confirmation_payload(payload, fixed_confirmation_manifest())


@pytest.mark.parametrize("corruption", ["source", "final", "outer", "metric", "nonfinite"])
def test_should_seal_source_evaluation_and_finite_scope_before_any_work_body(
    corruption: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _metadata_envelope()
    if corruption == "source":
        payload["manifest"]["families"][0]["development_source_map_sha256"] = "0" * 64
    elif corruption == "final":
        payload["final_released"] = True
    elif corruption == "outer":
        payload["outer_selection_scored"] = True
    elif corruption == "metric":
        payload["development"] = {"accuracy": 1.0}
    else:
        payload["seed_results"][0]["unexpected"] = float("inf")
    _seal_validation(monkeypatch)
    monkeypatch.setattr(work, "_verify_seed_work", _forbid)
    with pytest.raises(ValueError, match="confirmation JSON"):
        work.verify_confirmation_payload(payload, fixed_confirmation_manifest())


@pytest.mark.parametrize("corruption", [None, "family_wake", "family_updates", "family_guards"])
def test_should_delegate_all_reserved_metadata_and_enforce_aggregate_bounds(
    corruption: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Metadata-only spy: no trained bodies, no reserved source or 560-cell claim.
    payload = _metadata_envelope()
    calls = []
    monkeypatch.setattr(envelope, "_verify_seed_envelope", lambda *args: None)

    def seed_work(row: dict[str, Any], family: Any) -> work.SeedWork:
        calls.append((family.name, row["seed"]))
        wake, replay, guards = 24 * len(family.arms), 0, 0
        if row["family"] == "gating" and row["seed"] == family.seeds[0]:
            if corruption == "family_wake":
                wake -= 1
            elif corruption == "family_updates":
                replay = 1
            elif corruption == "family_guards":
                guards = 1
        return work.SeedWork(
            family.name,
            row["seed"],
            len(family.arms),
            wake,
            replay,
            0,
            guards,
            2 * guards,
            48 * guards,
            8,
            family.retained_array_bytes_per_seed_before_copies,
        )

    monkeypatch.setattr(work, "_verify_seed_work", seed_work)
    if corruption is not None:
        with pytest.raises(ValueError, match="family .* (differs|exceeded)"):
            work.verify_confirmation_payload(payload, fixed_confirmation_manifest())
    else:
        result = work.verify_confirmation_payload(payload, fixed_confirmation_manifest())
        assert calls == [(row["family"], row["seed"]) for row in payload["seed_results"]]
        assert len(result) == 60 and sum(item.cells for item in result) == 560
        assert sum(item.wake_updates for item in result) == 13440
