"""Envelope/schema fixtures do not construct reserved sources or models."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json
from typing import Any

import pytest

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_confirmation_training as training
from src.app import continual_confirmation_validation as validation
from src.app.continual_confirmation_checkpoints import declarations, verify_checkpoint
from src.app.continual_confirmation_json import (
    canonical_mapping,
    canonical_value,
    finite_json,
    state_digest,
)
from src.app.continual_confirmation_manifest import fixed_confirmation_manifest
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.controlled_parent_selection import ParentControlledCircadianNetwork
from src.core.predictive_coding import PredictiveCodingNetwork


NAMES = ("gating", "replay", "sleep", "schedule", "combined", "parent")


@pytest.fixture(scope="module")
def fixtures() -> tuple[tuple[Any, ...], dict[str, dict[str, Any]]]:
    families = tuple(
        replace(family, seeds=(family.development_seeds[0],))
        for family in fixed_confirmation_manifest().families
    )
    rows = training._train_families(families)
    facts = {
        row.facts.family: json.loads(json.dumps(asdict(row.facts), allow_nan=False)) for row in rows
    }
    return families, facts


def _forbid(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("JSON validation created data/model or trained/scored")


@pytest.mark.parametrize("name", NAMES)
def test_should_verify_all_development_checkpoints_without_source_model_or_score(
    name: str,
    fixtures: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    families, facts = fixtures
    for model in (
        BackpropMLP,
        PredictiveCodingNetwork,
        CircadianPredictiveCodingNetwork,
        ParentControlledCircadianNetwork,
    ):
        monkeypatch.setattr(model, "__init__", _forbid)
        monkeypatch.setattr(model, "train_epoch", _forbid)
        monkeypatch.setattr(model, "predict_proba", _forbid)
    monkeypatch.setattr(arrived, "_build_phase_a_roles", _forbid)
    monkeypatch.setattr(arrived, "_build_phase_b_roles", _forbid)
    family = next(item for item in families if item.name == name)
    validation._verify_seed_envelope(facts[name], family)


def _owner(checkpoint: dict[str, Any]) -> dict[str, Any]:
    state = checkpoint["state"]
    if "dataclass" in state:
        state = state["fields"]["state"]
    return canonical_mapping(state)


def _set_owner(checkpoint: dict[str, Any], key: str, value: Any) -> None:
    state = checkpoint["state"]
    if "dataclass" in state:
        state = state["fields"]["state"]
    entry = next(row for row in state["dict"] if row[0] == key)
    entry[1] = value
    checkpoint["state_sha256"] = state_digest(checkpoint["state"])


@pytest.mark.parametrize(
    "corruption",
    [
        "digest",
        "unknown_member",
        "array_shape",
        "model_type",
        "clock_view",
        "self_consistent_clock",
        "retention_view",
        "config",
        "lineage_view",
        "selector_view",
        "self_consistent_cursor",
        "initial_parameters",
        "initial_rng",
        "reward_type",
        "history_type",
        "decision_type",
    ],
)
def test_should_refuse_forged_checkpoint_even_with_resealed_state_digest(
    corruption: str,
    fixtures: tuple[Any, Any],
) -> None:
    families, facts = fixtures
    family = next(item for item in families if item.name == "parent")
    seed = family.seeds[0]
    stage = "initial" if corruption.startswith("initial") else "after_b"
    name = "random_growth"
    checkpoint = deepcopy(facts["parent"][stage][name])
    declared = declarations(family, seed)[name]
    owner = _owner(checkpoint)
    if corruption == "digest":
        checkpoint["state_sha256"] = "0" * 64
    elif corruption == "unknown_member":
        checkpoint["state"]["fields"]["state"]["dict"].append(["unexpected", 0])
        checkpoint["state"]["fields"]["state"]["dict"].sort(key=lambda row: row[0])
    elif corruption == "array_shape":
        owner["weight_input_hidden"]["shape"] = [checkpoint["width"], 2]
    elif corruption == "model_type":
        checkpoint["model_type"] = "src.core.backprop_mlp.BackpropMLP"
    elif corruption == "clock_view":
        checkpoint["clocks"]["wake_examples"] += 1
    elif corruption == "self_consistent_clock":
        checkpoint["clocks"]["wake_examples"] += 1
        _set_owner(checkpoint, "_wake_examples", checkpoint["clocks"]["wake_examples"])
    elif corruption == "retention_view":
        checkpoint["retention"]["retained_bytes"] += 1
    elif corruption == "config":
        owner["config"]["fields"]["min_plasticity"] = 0.2
    elif corruption == "lineage_view":
        checkpoint["lineage"]["fields"]["next_neuron_id"] += 1
    elif corruption == "selector_view":
        checkpoint["selector"]["selection_calls"] += 1
    elif corruption == "self_consistent_cursor":
        checkpoint["selector"]["cursor_id"] = 1
        _set_owner(checkpoint, "_parent_selection_cursor", 1)
    elif corruption == "initial_parameters":
        owner["weight_input_hidden"]["bytes_sha256"] = "0" * 64
        checkpoint["parameter_sha256"] = "0" * 64
    elif corruption == "initial_rng":
        rng = canonical_mapping(owner["_rng"]["generator"])
        nested = canonical_mapping(rng["state"])
        for item in rng["state"]["dict"]:
            if item[0] == "state":
                item[1] = nested["state"] + 1
    elif corruption == "reward_type":
        _set_owner(checkpoint, "_last_reward_scale", True)
    elif corruption == "history_type":
        _set_owner(checkpoint, "_energy_history", {"list": ["not a diagnostic"]})
    else:
        owner["_last_parent_selection"]["fields"]["cursor_before"] = True
        checkpoint["selector"]["last_decision"]["cursor_before"] = True
    if corruption != "digest":
        checkpoint["state_sha256"] = state_digest(checkpoint["state"])
    with pytest.raises(ValueError, match="confirmation JSON"):
        verify_checkpoint(checkpoint, declared, stage)


@pytest.mark.parametrize(
    "corruption",
    [
        "overlap",
        "position",
        "final_id",
        "float_count",
        "bool_seed",
        "hash_link",
        "extra_role",
        "metric",
        "missing_checkpoint",
        "nonfinite",
        "witness_missing",
        "witness_rollback",
        "witness_clock",
    ],
)
def test_should_refuse_changed_role_envelope_or_full_rollback(
    corruption: str,
    fixtures: tuple[Any, Any],
) -> None:
    families, facts = fixtures
    family = next(item for item in families if item.name == "schedule")
    row = deepcopy(facts["schedule"])
    role = row["roles"][0]
    if corruption == "overlap":
        role["sample_ids"]["inner_guard"][0] = role["sample_ids"]["train"][0]
    elif corruption == "position":
        role["sample_ids"]["train"][-1] = f"phase_a/seed_{row['seed']}/development/120"
    elif corruption == "final_id":
        role["sample_ids"]["final_test"][-1] = "changed"
    elif corruption == "float_count":
        role["counts"]["train"] = 72.0
    elif corruption == "bool_seed":
        role["seed"] = True
    elif corruption == "hash_link":
        role["hashes"]["train"] = "0" * 64
    elif corruption == "extra_role":
        role["hashes"]["final_test"] = "0" * 64
    elif corruption == "metric":
        row["legacy_train_facts"]["development"] = {}
    elif corruption == "missing_checkpoint":
        del row["after_b"][family.arms[-1]]
    elif corruption == "nonfinite":
        row["legacy_train_facts"]["methods"][0]["wake_updates"] = float("inf")
    elif corruption == "witness_missing":
        row["supplemental_guards"].pop()
    elif corruption == "witness_clock":
        witness = row["supplemental_guards"][0]
        for endpoint in ("before", "after"):
            witness[endpoint]["clocks"]["wake_examples"] += 1
            _set_owner(
                witness[endpoint], "_wake_examples", witness[endpoint]["clocks"]["wake_examples"]
            )
    else:
        witness = next(item for item in row["supplemental_guards"] if item["outcome"] == "skipped")
        _set_owner(witness["after"], "_last_reward_scale", 0.9)
    with pytest.raises(ValueError, match="confirmation JSON"):
        validation._verify_seed_envelope(row, family)


def _synthetic_envelope() -> dict[str, Any]:
    manifest = fixed_confirmation_manifest()
    # Metadata-only unit fixture isolates whole-scope ordering/delegation.
    # Per-seed checkpoint/role bodies are tested separately on actual fixtures.
    return {
        "manifest": json.loads(json.dumps(asdict(manifest))),
        "seed_results": [
            {"family": family.name, "seed": seed}
            for family in manifest.families
            for seed in family.seeds
        ],
        "protocol_id": "continual_mechanism_confirmation_train_only_v1",
        "all_a_completed_before_first_b": True,
        "outer_selection_scored": False,
        "final_released": False,
    }


def test_should_delegate_every_complete_reserved_metadata_row_without_constructing_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visited = []

    def visit(row: dict[str, Any], family: Any) -> None:
        assert row["seed"] in family.seeds
        visited.append((row["family"], row["seed"]))

    monkeypatch.setattr(validation, "_verify_seed_envelope", visit)
    monkeypatch.setattr(arrived, "_build_phase_a_roles", _forbid)
    payload = _synthetic_envelope()
    validation.verify_confirmation_envelope(payload, fixed_confirmation_manifest())
    assert visited == [(row["family"], row["seed"]) for row in payload["seed_results"]]
    assert len(visited) == 60 and len({seed for _, seed in visited}) == 50


@pytest.mark.parametrize(
    "corruption", ["missing_seed", "reorder", "extra_seed", "float_budget", "seal", "extra_field"]
)
def test_should_refuse_changed_whole_scope_before_any_seed_body(
    monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    payload = _synthetic_envelope()
    if corruption == "missing_seed":
        payload["seed_results"].pop()
    elif corruption == "reorder":
        payload["seed_results"].reverse()
    elif corruption == "extra_seed":
        payload["seed_results"].append(payload["seed_results"][0])
    elif corruption == "float_budget":
        payload["manifest"]["max_optimizer_updates"] = 16000.0
    elif corruption == "seal":
        payload["final_released"] = True
    else:
        payload["accuracy"] = 1.0
    monkeypatch.setattr(validation, "_verify_seed_envelope", _forbid)
    with pytest.raises(ValueError, match="confirmation JSON"):
        validation.verify_confirmation_envelope(payload, fixed_confirmation_manifest())


@pytest.mark.parametrize(
    "node",
    [
        {"dict": [], "list": []},
        {"dict": [["x", 0], ["x", 1]]},
        {"dict": [["z", 0], ["a", 1]]},
        {"dataclass": "unknown.Type", "fields": {}},
        {"array_dtype": "<f4", "shape": [2], "bytes_sha256": "0" * 64},
        {"array_dtype": "<f8", "shape": [True], "bytes_sha256": "0" * 64},
        {"set": ["duplicate", "duplicate"]},
        {"numpy_scalar": "<f8", "value": True},
    ],
)
def test_should_refuse_ambiguous_or_changed_canonical_node(node: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="confirmation JSON"):
        canonical_value(node)


def test_should_refuse_excess_json_nesting_before_recursive_checkpoint_work() -> None:
    value: Any = None
    for _ in range(45):
        value = [value]
    with pytest.raises(ValueError, match="nesting limit"):
        finite_json(value)


@pytest.mark.parametrize("corruption", ["missing", "extra", "swap", "malformed", "uniform_link"])
def test_should_refuse_incomplete_or_changed_parameter_contract_map(
    corruption: str, fixtures: tuple[Any, Any]
) -> None:
    families, facts = fixtures
    family = next(item for item in families if item.name == "gating")
    checkpoint = deepcopy(facts["gating"]["initial"]["ordinary_pc"])
    hashes = checkpoint["parameter_sha256_by_contract"]
    gating = "p63-shallow-parameter-tensors-v1"
    replay = "p63-replay-shallow-parameter-tensors-v1"
    sleep = "p63-sleep-factor-shallow-parameters-v1"
    if corruption == "missing":
        del hashes[gating]
    elif corruption == "extra":
        hashes["unknown_contract"] = "0" * 64
    elif corruption == "swap":
        hashes[gating], hashes[sleep] = hashes[sleep], hashes[gating]
    elif corruption == "malformed":
        hashes[sleep] = "not-a-sha256"
    else:
        checkpoint["parameter_sha256"] = hashes[gating]
        assert checkpoint["parameter_sha256"] != hashes[replay]
    with pytest.raises(ValueError, match="confirmation JSON"):
        verify_checkpoint(
            checkpoint, declarations(family, family.seeds[0])["ordinary_pc"], "initial"
        )


@pytest.mark.parametrize(
    "name,contract",
    [
        ("gating", "p63-shallow-parameter-tensors-v1"),
        ("replay", "p63-replay-shallow-parameter-tensors-v1"),
        ("sleep", "p63-sleep-factor-shallow-parameters-v1"),
    ],
)
def test_should_refuse_resealed_initial_contract_and_matching_raw_endpoint(
    name: str, contract: str, fixtures: tuple[Any, Any]
) -> None:
    families, facts = fixtures
    family = next(item for item in families if item.name == name)
    row = deepcopy(facts[name])
    arm = family.arms[0]
    checkpoint = row["initial"][arm]
    checkpoint["parameter_sha256_by_contract"][contract] = "0" * 64
    if name == "replay":
        checkpoint["parameter_sha256"] = "0" * 64
    raw = row["legacy_train_facts"]["arms" if name == "sleep" else "methods"][0]
    raw["initial_parameter_sha256"] = "0" * 64
    checkpoint["state_sha256"] = state_digest(checkpoint["state"])
    with pytest.raises(ValueError, match="seeded initial parameter"):
        validation._verify_seed_envelope(row, family)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("endpoint", ["initial", "final"])
def test_should_refuse_changed_raw_parameter_endpoint(
    name: str, endpoint: str, fixtures: tuple[Any, Any]
) -> None:
    families, facts = fixtures
    family = next(item for item in families if item.name == name)
    row = deepcopy(facts[name])
    rows = row["legacy_train_facts"]["arms" if name == "sleep" else "methods"]
    rows[0][f"{endpoint}_parameter_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="parameter.*link"):
        validation._verify_seed_envelope(row, family)


@pytest.mark.parametrize("name", ["schedule", "combined", "parent"])
def test_should_refuse_changed_raw_a_checkpoint_parameter_link(
    name: str, fixtures: tuple[Any, Any]
) -> None:
    families, facts = fixtures
    family = next(item for item in families if item.name == name)
    row = deepcopy(facts[name])
    row["legacy_train_facts"]["methods"][0]["after_a_parameter_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="parameter.*link"):
        validation._verify_seed_envelope(row, family)


@pytest.mark.parametrize("name", ["sleep", "schedule"])
@pytest.mark.parametrize("endpoint", ["before", "after"])
def test_should_refuse_changed_raw_guard_parameter_link(
    name: str, endpoint: str, fixtures: tuple[Any, Any]
) -> None:
    families, facts = fixtures
    family = next(item for item in families if item.name == name)
    row = deepcopy(facts[name])
    if name == "sleep":
        arm_name = row["supplemental_guards"][0]["name"]
        arm = next(item for item in row["legacy_train_facts"]["arms"] if item["name"] == arm_name)
        arm[f"{'pre' if endpoint == 'before' else 'post'}_sleep_parameter_sha256"] = "0" * 64
    else:
        row["legacy_train_facts"]["opportunities"][0]["decisions"][0][
            f"parameter_sha256_{endpoint}"
        ] = "0" * 64
    with pytest.raises(ValueError, match="parameter.*link"):
        validation._verify_seed_envelope(row, family)


@pytest.mark.parametrize("name", ["combined", "parent"])
@pytest.mark.parametrize("endpoint", ["before", "after", "epoch_a", "epoch_b", "wake"])
def test_should_refuse_broken_raw_epoch_or_guard_parameter_chain(
    name: str, endpoint: str, fixtures: tuple[Any, Any]
) -> None:
    families, facts = fixtures
    family = next(item for item in families if item.name == name)
    row = deepcopy(facts[name])
    index = 11 if endpoint == "epoch_a" else 23 if endpoint == "epoch_b" else 0
    opportunity = row["legacy_train_facts"]["opportunities"][index]
    decision = opportunity["decisions"][0]
    if endpoint.startswith("epoch"):
        opportunity["after_epoch_parameter_sha256"][family.arms[0]] = "0" * 64
    elif endpoint == "wake":
        wake = next(item for item in opportunity["wake"] if item["name"] == decision["name"])
        wake["parameter_sha256"] = "0" * 64
    elif name == "parent":
        decision[endpoint]["parameter_sha256"] = "0" * 64
    else:
        decision[f"parameter_sha256_{endpoint}"] = "0" * 64
    with pytest.raises(ValueError, match="parameter.*link"):
        validation._verify_seed_envelope(row, family)


@pytest.mark.parametrize("name", ["sleep", "schedule", "combined", "parent"])
def test_should_verify_parameter_chains_with_all_guarded_proposals_rejected(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    family = next(item for item in fixed_confirmation_manifest().families if item.name == name)
    family = replace(family, seeds=(family.development_seeds[0],))
    calls: dict[CircadianPredictiveCodingNetwork, int] = {}

    def accuracy(model: CircadianPredictiveCodingNetwork, *args: Any) -> float:
        count = calls.get(model, 0)
        calls[model] = count + 1
        return 1.0 if count % 2 == 0 else 0.0

    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "compute_accuracy", accuracy)
    held = training._train_families((family,))[0]
    row = json.loads(json.dumps(asdict(held.facts), allow_nan=False))
    if name in {"sleep", "schedule"}:
        outcomes = [item["outcome"] for item in row["supplemental_guards"]]
    else:
        outcomes = [
            decision["event"]["outcome"]
            for item in row["legacy_train_facts"]["opportunities"]
            for decision in item["decisions"]
        ]
    assert "rolled_back" in outcomes
    assert "accepted" not in outcomes
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
    validation._verify_seed_envelope(row, family)


@pytest.mark.parametrize("name", NAMES)
def test_should_refuse_omitted_raw_method_parameter_link(
    name: str, fixtures: tuple[Any, Any]
) -> None:
    families, facts = fixtures
    family = next(item for item in families if item.name == name)
    row = deepcopy(facts[name])
    row["legacy_train_facts"]["arms" if name == "sleep" else "methods"].pop()
    with pytest.raises(ValueError, match="parameter link row inventory"):
        validation._verify_seed_envelope(row, family)
