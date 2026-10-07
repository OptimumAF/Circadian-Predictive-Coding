"""Bind live confirmation checkpoints and sealed arrived-role metadata.

Inputs are existing shallow models/roles. Outputs contain complete canonical
fingerprints, never outer/final arrays or scores. This module owns no training,
source construction, resource measurement or artifact IO.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, fields, is_dataclass
from hashlib import sha256
import json
from typing import Any

import numpy as np

from src.app.continual_combined_factor_preflight import _canonical_state
from src.app.continual_confirmation_parameter_links import PARAMETER_CONTRACTS
from src.app.continual_replay_factor_pilot import Model, _parameter_count, _parameter_hash
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.controlled_parent_selection import ParentControlledCircadianNetwork
from src.infra.continual_roles import PhaseDecisionRoles, RoleAvailability, _hash_role


@dataclass(frozen=True)
class CheckpointFacts:
    model_type: str
    parameter_sha256: str
    parameter_sha256_by_contract: dict[str, str]
    state_sha256: str
    width: int
    parameter_count: int
    state: Any
    clocks: dict[str, int] | None
    retention: dict[str, Any] | None
    lineage: Any
    selector: dict[str, Any] | None


@dataclass(frozen=True)
class RoleFacts:
    phase: str
    seed: int
    hashes: dict[str, str]
    counts: dict[str, int]
    sample_ids: dict[str, tuple[str, ...]]
    expected_final_count: int
    final_released: bool = False


@dataclass(frozen=True)
class GuardWitness:
    name: str
    phase: str
    epoch: int
    outcome: str
    before: CheckpointFacts
    after: CheckpointFacts


@dataclass(frozen=True)
class FamilySeedFacts:
    family: str
    seed: int
    roles: tuple[RoleFacts, RoleFacts]
    initial: dict[str, CheckpointFacts]
    after_a: dict[str, CheckpointFacts]
    after_b: dict[str, CheckpointFacts]
    legacy_train_facts: dict[str, Any]
    supplemental_guards: tuple[GuardWitness, ...]
    outer_selection_scored: bool = False
    final_released: bool = False


@dataclass
class HeldSeed:
    facts: FamilySeedFacts
    roles_a: PhaseDecisionRoles
    roles_b: PhaseDecisionRoles
    models_after_a: dict[str, Model]
    models_after_b: dict[str, Model]


def finite_json(value: Any) -> dict[str, Any]:
    return json.loads(
        json.dumps(
            asdict(value) if is_dataclass(value) and not isinstance(value, type) else value,
            allow_nan=False,
        )
    )


def legacy_role_facts(
    roles_a: PhaseDecisionRoles, roles_b: PhaseDecisionRoles
) -> tuple[dict[str, str], dict[str, int]]:
    rows = (capture_roles(roles_a, "a", roles_a.seed), capture_roles(roles_b, "b", roles_a.seed))
    return (
        {f"{row.phase}_{role}": digest for row in rows for role, digest in row.hashes.items()},
        {f"{row.phase}_{role}": count for row in rows for role, count in row.counts.items()},
    )


def _require_finite_state(value: Any) -> None:
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.inexact) and not np.all(np.isfinite(value)):
            raise ValueError("confirmation checkpoint contains nonfinite array state")
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _require_finite_state(getattr(value, field.name))
    elif isinstance(value, dict):
        for item in value.values():
            _require_finite_state(item)
    elif isinstance(value, (list, tuple, deque, set, frozenset)):
        for item in value:
            _require_finite_state(item)


def capture_model(model: Model) -> CheckpointFacts:
    circadian = model if isinstance(model, CircadianPredictiveCodingNetwork) else None
    raw = circadian.snapshot_state() if circadian is not None else vars(model)
    _require_finite_state(raw)
    state = _canonical_state(raw)
    encoded = json.dumps(state, sort_keys=True, allow_nan=False).encode("utf-8")
    width = int(model.weight_hidden_output.shape[0])
    count = _parameter_count(model)
    shapes = tuple(
        getattr(model, name).shape
        for name in ("weight_input_hidden", "bias_hidden", "weight_hidden_output", "bias_output")
    )
    if (
        count != 4 * width + 1
        or model.input_dim != 2
        or shapes != ((2, width), (1, width), (width, 1), (1, 1))
    ):
        raise ValueError("confirmation checkpoint requires the frozen shallow geometry")
    if not isinstance(model, CircadianPredictiveCodingNetwork) and (
        len(model._hidden_weights) != 1
        or len(model._hidden_biases) != 1
        or model.weight_input_hidden is not model._hidden_weights[0]
        or model.bias_hidden is not model._hidden_biases[0]
    ):
        # Equal array bytes do not prove equal future behavior if a baseline's
        # public parameter aliases have been detached from its forward path.
        raise ValueError("confirmation checkpoint baseline parameter aliases differ")
    return CheckpointFacts(
        f"{type(model).__module__}.{type(model).__qualname__}",
        _parameter_hash(model),
        _parameter_hashes(model),
        sha256(encoded).hexdigest(),
        width,
        count,
        state,
        asdict(circadian.get_sleep_clocks()) if circadian is not None else None,
        # Original gating/sleep configurations have zero memory and no optional
        # retention budget. Fingerprinting must not configure a new privilege.
        asdict(circadian.get_replay_retention())
        if circadian is not None and hasattr(circadian, "_replay_retention_budget")
        else None,
        _canonical_state(circadian.get_neuron_lineage()) if circadian is not None else None,
        asdict(model.get_parent_selection_state())
        if isinstance(model, ParentControlledCircadianNetwork)
        else None,
    )


def _parameter_hashes(model: Model) -> dict[str, str]:
    # Why this: old families domain-separate identical tensor bytes differently.
    # Capture their original contracts from bytes; hashes cannot be converted.
    digests = {contract: sha256(contract.encode("ascii")) for contract in PARAMETER_CONTRACTS}
    for name in ("weight_input_hidden", "bias_hidden", "weight_hidden_output", "bias_output"):
        array = np.ascontiguousarray(getattr(model, name), dtype="<f8")
        shape = np.asarray(array.shape, dtype="<i8").tobytes()
        raw = array.tobytes()
        for digest in digests.values():
            digest.update(name.encode("ascii"))
            digest.update(shape)
            digest.update(raw)
    return {contract: digest.hexdigest() for contract, digest in digests.items()}


def capture_models(models: dict[str, Model], arms: tuple[str, ...]) -> dict[str, CheckpointFacts]:
    if set(models) != set(arms) or len(arms) != len(set(arms)):
        raise ValueError("confirmation checkpoint arm inventory differs")
    return {name: capture_model(models[name]) for name in arms}


def require_checkpoints(
    models: dict[str, Model], expected: dict[str, CheckpointFacts], context: str
) -> None:
    if capture_models(models, tuple(expected)) != expected:
        raise ValueError(f"confirmation complete checkpoint changed: {context}")


def capture_roles(roles: PhaseDecisionRoles, phase: str, seed: int) -> RoleFacts:
    names = ("train", "inner_guard", "outer_selection")
    expected = (72, 24, 24) if phase == "a" else (36, 12, 12)
    if (
        phase not in {"a", "b"}
        or roles.phase != phase
        or type(roles.seed) is not int
        or roles.seed != seed
        or roles.final_released
        or roles.final_test is not None
        or roles._source is None
        or roles.expected_final_count != 40
        or set(roles.split_hashes) != set(names)
        or set(roles.sample_ids) != {*names, "final_test"}
        or roles.release_policy.get("final_test")
        != RoleAvailability("global_freeze", "global_freeze")
    ):
        raise ValueError("confirmation arrived role identity/final seal differs")
    ids = {name: tuple(roles.sample_ids[name]) for name in (*names, "final_test")}
    counts = {name: len(ids[name]) for name in names}
    if tuple(counts.values()) != expected or len(ids["final_test"]) != 40:
        raise ValueError("confirmation arrived role counts differ")
    all_ids = [item for group in ids.values() for item in group]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("confirmation arrived role IDs overlap")
    for name in ("train", "inner_guard"):
        data = getattr(roles, name)
        if (
            len(data.input) != counts[name]
            or _hash_role(phase, seed, name, ids[name], data) != roles.split_hashes[name]
        ):
            raise ValueError("confirmation arrived train/guard content differs from role hash")
    # Why this: outer identity/counts are already declared by the arrived split;
    # checking their arrays here would violate the unscored composition boundary.
    return RoleFacts(phase, seed, dict(roles.split_hashes), counts, ids, 40)


def guard_witness(
    name: str, phase: str, epoch: int, outcome: str, before: CheckpointFacts, model: Model
) -> GuardWitness:
    after = capture_model(model)
    if outcome not in {"accepted", "rolled_back", "skipped"}:
        raise ValueError("confirmation guard outcome differs")
    if outcome != "accepted" and before != after:
        raise ValueError(f"confirmation rejected/skipped guard changed complete state: {name}")
    return GuardWitness(name, phase, epoch, outcome, before, after)


def require_held_seed(item: HeldSeed) -> None:
    facts = item.facts
    if facts.outer_selection_scored or facts.final_released:
        raise ValueError("confirmation evaluation seal differs")
    if (
        capture_roles(item.roles_a, "a", facts.seed),
        capture_roles(item.roles_b, "b", facts.seed),
    ) != facts.roles:
        raise ValueError("confirmation held role metadata changed")
    require_checkpoints(item.models_after_a, facts.after_a, f"{facts.family}/{facts.seed}/a")
    require_checkpoints(item.models_after_b, facts.after_b, f"{facts.family}/{facts.seed}/b")
