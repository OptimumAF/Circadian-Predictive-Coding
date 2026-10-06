"""Full request metadata fixtures; source, array, RNG, model and IO work are forbidden."""

from copy import deepcopy
from dataclasses import fields, make_dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.app import continual_arrived_benchmark as arrived
from src.app.prospective_confirmation_design import (
    fixed_prospective_design,
    prospective_design_identity,
)
from src.app.prospective_role_requests import (
    declare_prospective_sources,
    inspect_prospective_role_request,
    prospective_role_request_identity,
    prospective_source_map_identity,
)
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.predictive_coding import PredictiveCodingNetwork
from src.core.prospective_replications import ReplicaSeedBinding, ReplicaSlot
from src.core.prospective_role_requests import ProspectiveRoleDeclaration, ProspectiveRoleRequest
from src.core.seed_stream_screening import EvidenceIdentity, confirmation_seed_streams


@pytest.fixture(autouse=True)
def no_science_or_io(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("role/source request inspection performed scientific work or IO")

    for name in ("_build_phase_a_roles", "_build_phase_b_roles", "release_final_test"):
        monkeypatch.setattr(arrived, name, forbidden)
    for model in (BackpropMLP, PredictiveCodingNetwork, CircadianPredictiveCodingNetwork):
        monkeypatch.setattr(model, "__init__", forbidden)
        monkeypatch.setattr(model, "compute_accuracy", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)


def _reseal(role_request: ProspectiveRoleRequest) -> ProspectiveRoleRequest:
    changed = replace(
        role_request, source_map_identity=prospective_source_map_identity(role_request.sources)
    )
    return replace(changed, request_identity=prospective_role_request_identity(changed))


def _unknown_record(value: Any) -> Any:
    foreign = make_dataclass("UnknownRecord", [], bases=(type(value),), frozen=True)
    return foreign(**{field.name: getattr(value, field.name) for field in fields(value)})


@pytest.fixture
def design() -> dict[str, Any]:
    return fixed_prospective_design()


@pytest.fixture
def code_identity() -> EvidenceIdentity:
    # Fabricated expected code declaration, never a physical source proof.
    return EvidenceIdentity(123, "a" * 64)


@pytest.fixture
def role_request(design: dict[str, Any], code_identity: EvidenceIdentity) -> ProspectiveRoleRequest:
    slots = tuple(ReplicaSlot(**row) for row in design["replica_slots"])
    groups = tuple(dict.fromkeys(row.source_group for row in slots))
    bindings = tuple(
        ReplicaSeedBinding(slot, 1_000_000 + 20_000 * groups.index(slot.source_group))
        for slot in slots
    )
    seeds = dict((row.slot, row.base_seed) for row in bindings)
    streams = tuple(
        stream
        for group in groups
        for stream in confirmation_seed_streams(
            next(row.base_seed for row in bindings if row.slot.source_group == group)
        )
    )
    roles = []
    for row in design["role_requirements"]:
        slot = ReplicaSlot(**row["slot"])
        phase, role, count = row["phase"], row["role"], row["expected_count"]
        start = (
            0
            if role == "final_test"
            else {"train": 0, "inner_guard": 72, "outer_selection": 96}[role]
            if phase == "a"
            else {"train": 60, "inner_guard": 96, "outer_selection": 108}[role]
        )
        namespace = "final" if role == "final_test" else "development"
        ids = tuple(
            f"phase_{phase}/seed_{seeds[slot]}/{namespace}/{index}"
            for index in range(start, start + count)
        )
        roles.append(
            ProspectiveRoleDeclaration(
                slot,
                phase,
                role,
                count,
                ids,
                row["source_available_at"],
                row["labels_available_at"],
                row["allowed_use"],
            )
        )
    sources = declare_prospective_sources(design, bindings, code_identity)
    value = ProspectiveRoleRequest(
        EvidenceIdentity(**prospective_design_identity(design)),
        bindings,
        streams,
        sources,
        tuple(roles),
        prospective_source_map_identity(sources),
        EvidenceIdentity(0, "0" * 64),
        tuple(design["required_actual_request_bindings"]),
    )
    return _reseal(value)


def test_should_inspect_every_source_role_and_stream_without_provenance_or_execution(
    design: dict[str, Any], code_identity: EvidenceIdentity, role_request: ProspectiveRoleRequest
) -> None:
    before = deepcopy((design, role_request))
    result = inspect_prospective_role_request(design, code_identity, role_request)
    assert result.sources == role_request.sources and len(result.sources) == 100
    assert result.roles == role_request.roles and len(result.roles) == 480
    assert len(role_request.streams) == 400 and len(role_request.bindings) == 60
    assert result.request_identity == role_request.request_identity
    assert result.source_map_identity == role_request.source_map_identity
    assert result.required_actual_bindings == tuple(design["required_actual_request_bindings"])
    assert result.independent_source_replications is None
    assert not result.actual_code_verified
    assert not any(
        (
            result.actual_sources_verified,
            result.actual_roles_verified,
            result.chronology_verified,
            result.fresh_roles_authorized,
            result.execution_authorized,
            result.precision_certified,
        )
    )
    assert (design, role_request) == before
    b_train = role_request.roles[4]
    assert b_train.sample_ids[0].endswith("/60") and b_train.sample_ids[-1].endswith("/95")
    assert role_request.roles[7].sample_ids[-1].endswith("/39")
    assert len(design["cell_plan"]) == 560 and len(design["ordered_primary_statements"]) == 116


@pytest.mark.parametrize("phase, source_index", (("a", 0), ("b", 1)))
def test_should_bind_the_original_generator_geometry_and_every_declared_data_argument(
    design: dict[str, Any],
    code_identity: EvidenceIdentity,
    role_request: ProspectiveRoleRequest,
    phase: str,
    source_index: int,
) -> None:
    # Independent literal oracle: no RNG or source construction proves these declarations.
    base = role_request.sources[source_index].base_seed
    body = {
        "generator": "generate_two_cluster_dataset_with_transform",
        "sample_count": 160,
        "noise_scale": 0.8 if phase == "a" else 1.0,
        "test_ratio": 0.25,
        "rotation_degrees": 0.0 if phase == "a" else 40.0,
        "translation_x": 0.0 if phase == "a" else 0.9,
        "translation_y": 0.0 if phase == "a" else -0.7,
        "source_seed": base if phase == "a" else base + 101,
        "role_split_seed": base + (17 if phase == "a" else 138),
        "exposure_seed": None if phase == "a" else base + 118,
        "exposure_fraction": 1.0 if phase == "a" else 0.5,
        "original_development_count": 120,
        "retained_development_count": 120 if phase == "a" else 60,
        "final_count": 40,
        "inner_guard_fraction": 0.2,
        "outer_selection_fraction": 0.2,
    }
    raw = (json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n").encode()
    assert role_request.sources[source_index].generator_configuration_identity == EvidenceIdentity(
        len(raw), sha256(raw).hexdigest()
    )
    assert (
        inspect_prospective_role_request(design, code_identity, role_request).sources
        == role_request.sources
    )


@pytest.mark.parametrize(
    "kind",
    (
        "missing",
        "extra",
        "reordered",
        "duplicate",
        "mutable",
        "unknown_record",
        "foreign_group",
        "foreign_seed",
        "code_pin",
        "configuration_pin",
    ),
)
def test_should_reject_resealed_source_map_drift_at_the_last_source(
    design: dict[str, Any],
    code_identity: EvidenceIdentity,
    role_request: ProspectiveRoleRequest,
    kind: str,
) -> None:
    sources: Any = role_request.sources
    if kind == "missing":
        sources = sources[:-1]
    elif kind == "extra":
        sources += (sources[-1],)
    elif kind == "reordered":
        sources = sources[:-2] + (sources[-1], sources[-2])
    elif kind == "duplicate":
        sources = sources[:-1] + (sources[-2],)
    elif kind == "mutable":
        sources = list(sources)
    elif kind == "unknown_record":
        sources = sources[:-1] + (_unknown_record(sources[-1]),)
    else:
        changes: dict[str, dict[str, Any]] = {
            "foreign_group": {"source_group": "unknown_group"},
            "foreign_seed": {"base_seed": sources[-1].base_seed + 1},
            "code_pin": {"code_identity": EvidenceIdentity(123, "b" * 64)},
            "configuration_pin": {
                "generator_configuration_identity": EvidenceIdentity(123, "c" * 64)
            },
        }
        sources = sources[:-1] + (replace(sources[-1], **changes[kind]),)
    changed = _reseal(replace(role_request, sources=sources))
    before = deepcopy(changed)
    with pytest.raises(ValueError):
        inspect_prospective_role_request(design, code_identity, changed)
    assert changed == before


@pytest.mark.parametrize(
    "kind",
    (
        "missing",
        "extra",
        "reordered",
        "duplicate",
        "mutable",
        "unknown_record",
        "foreign_slot",
        "boolean_ordinal",
        "float_count",
        "boolean_count",
        "early_source",
        "early_labels",
        "outer_use",
        "final_released",
        "release_zero",
        "array_claim",
    ),
)
def test_should_reject_late_resealed_role_schema_scope_or_availability_drift(
    design: dict[str, Any],
    code_identity: EvidenceIdentity,
    role_request: ProspectiveRoleRequest,
    kind: str,
) -> None:
    roles: Any = role_request.roles
    if kind == "missing":
        roles = roles[:-1]
    elif kind == "extra":
        roles += (roles[-1],)
    elif kind == "reordered":
        roles = roles[:-2] + (roles[-1], roles[-2])
    elif kind == "duplicate":
        roles = roles[:-1] + (roles[-2],)
    elif kind == "mutable":
        roles = list(roles)
    elif kind == "unknown_record":
        roles = roles[:-1] + (_unknown_record(roles[-1]),)
    else:
        position = 8 if kind == "boolean_ordinal" else 478 if kind == "outer_use" else -1
        last = roles[position]
        changes: dict[str, dict[str, Any]] = {
            "foreign_slot": {"slot": replace(last.slot, family="unknown")},
            "boolean_ordinal": {"slot": replace(last.slot, ordinal=True)},
            "float_count": {"expected_count": float(last.expected_count)},
            "boolean_count": {"expected_count": True},
            "early_source": {"source_available_at": "phase_b_arrival"},
            "early_labels": {"labels_available_at": "phase_b_arrival"},
            "outer_use": {"allowed_use": "select_favorable_settings"},
            "final_released": {"final_released": True},
            "release_zero": {"final_released": 0},
            "array_claim": {"array_identity": EvidenceIdentity(8, "b" * 64)},
        }
        items = list(roles)
        items[position] = replace(last, **changes[kind])
        roles = tuple(items)
    changed = _reseal(replace(role_request, roles=roles))
    before = deepcopy(changed)
    with pytest.raises(ValueError):
        inspect_prospective_role_request(design, code_identity, changed)
    assert changed == before


@pytest.mark.parametrize(
    "kind",
    (
        "foreign_phase",
        "foreign_seed",
        "foreign_namespace",
        "noncanonical_index",
        "out_of_range",
        "duplicate_id",
        "reordered_ids",
        "short_ids",
        "mutable_ids",
        "overlap",
        "shared_view",
        "final_reordered",
        "final_out_of_range",
        "string_subclass",
    ),
)
def test_should_reject_wrong_sample_identity_partitions_or_shared_views(
    design: dict[str, Any],
    code_identity: EvidenceIdentity,
    role_request: ProspectiveRoleRequest,
    kind: str,
) -> None:
    position = 84 if kind == "shared_view" else 479 if kind.startswith("final_") else 476
    role = role_request.roles[position]
    ids: Any = role.sample_ids
    prefix = ids[0].rsplit("/", 1)[0]
    if kind == "foreign_phase":
        ids = (ids[0].replace("phase_b", "phase_a"),) + ids[1:]
    elif kind == "foreign_seed":
        ids = (ids[0].replace("seed_", "seed_9", 1),) + ids[1:]
    elif kind == "foreign_namespace":
        ids = (ids[0].replace("development", "final"),) + ids[1:]
    elif kind == "noncanonical_index":
        ids = (prefix + "/060",) + ids[1:]
    elif kind in {"out_of_range", "final_out_of_range"}:
        ids = ids[:-1] + (prefix + ("/40" if kind.startswith("final_") else "/120"),)
    elif kind == "duplicate_id":
        ids = ids[:-1] + (ids[-2],)
    elif kind in {"reordered_ids", "final_reordered"}:
        ids = ids[:-2] + (ids[-1], ids[-2])
    elif kind == "short_ids":
        ids = ids[:-1]
    elif kind == "mutable_ids":
        ids = list(ids)
    elif kind == "overlap":
        ids = ids[1:] + (prefix + "/96",)
    elif kind == "string_subclass":

        class ForeignString(str):
            pass

        ids = (ForeignString(ids[0]),) + ids[1:]
    else:
        # Valid disjoint/count-correct B partition differs from its paired view.
        ids = tuple(prefix + f"/{index}" for index in range(24, 60))
    changed_roles = list(role_request.roles)
    changed_roles[position] = replace(role, sample_ids=ids)
    changed = _reseal(replace(role_request, roles=tuple(changed_roles)))
    before = deepcopy(changed)
    with pytest.raises(ValueError):
        inspect_prospective_role_request(design, code_identity, changed)
    assert changed == before


@pytest.mark.parametrize(
    "kind",
    (
        "request_pin",
        "source_map_pin",
        "design_pin",
        "execution",
        "execution_zero",
        "missing_proof",
        "reorder_proofs",
        "mutable_proofs",
        "schema",
        "late_stream",
        "unknown_record",
    ),
)
def test_should_reject_detached_envelopes_and_forged_authority_or_missing_obligations(
    design: dict[str, Any],
    code_identity: EvidenceIdentity,
    role_request: ProspectiveRoleRequest,
    kind: str,
) -> None:
    changes: dict[str, dict[str, Any]] = {
        "request_pin": {"request_identity": EvidenceIdentity(123, "b" * 64)},
        "source_map_pin": {"source_map_identity": EvidenceIdentity(123, "b" * 64)},
        "design_pin": {"design_identity": EvidenceIdentity(123, "b" * 64)},
        "execution": {"execution_requested": True},
        "execution_zero": {"execution_requested": 0},
        "missing_proof": {"required_actual_bindings": role_request.required_actual_bindings[:-1]},
        "reorder_proofs": {
            "required_actual_bindings": tuple(reversed(role_request.required_actual_bindings))
        },
        "mutable_proofs": {"required_actual_bindings": list(role_request.required_actual_bindings)},
        "schema": {"schema_id": "unknown_schema"},
        "late_stream": {
            "streams": role_request.streams[:-1]
            + (replace(role_request.streams[-1], value=role_request.streams[-1].value + 1),)
        },
        "unknown_record": {},
    }
    changed = (
        _unknown_record(role_request)
        if kind == "unknown_record"
        else replace(role_request, **changes[kind])
    )
    if kind not in {"request_pin", "source_map_pin"}:
        changed = _reseal(changed)
    before = deepcopy(changed)
    with pytest.raises(ValueError):
        inspect_prospective_role_request(design, code_identity, changed)
    assert changed == before
