"""Full recipe/realization metadata behavior with scientific and IO access forbidden."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, make_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from prospective_generation_fixtures import concrete_role_declarations, generation_inputs
from src.app import continual_arrived_benchmark as arrived
from src.app.prospective_generation_requests import (
    freeze_prospective_generation_request,
    inspect_arrived_role_declarations,
    inspect_prospective_generation_request,
    prospective_generation_request_identity,
)
from src.app.prospective_role_requests import prospective_source_map_identity
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.predictive_coding import PredictiveCodingNetwork
from src.core.prospective_generation_requests import ProspectiveGenerationRequest
from src.core.seed_stream_screening import EvidenceIdentity


@pytest.fixture(autouse=True)
def no_science_or_io(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("generation request performed source/array/RNG/model/final work or IO")

    for name in ("_build_phase_a_roles", "_build_phase_b_roles", "release_final_test"):
        monkeypatch.setattr(arrived, name, forbidden)
    for model in (BackpropMLP, PredictiveCodingNetwork, CircadianPredictiveCodingNetwork):
        monkeypatch.setattr(model, "__init__", forbidden)
        monkeypatch.setattr(model, "compute_accuracy", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    for name in ("array", "asarray", "ascontiguousarray"):
        monkeypatch.setattr(np, name, forbidden)
    monkeypatch.setattr(Path, "open", forbidden)


@pytest.fixture
def inputs() -> tuple[Any, ...]:
    return generation_inputs()


@pytest.fixture
def generation_request(inputs: tuple[Any, ...]) -> ProspectiveGenerationRequest:
    design, code, bindings, streams = inputs
    return freeze_prospective_generation_request(design, bindings, code, streams)


def _reseal(value: ProspectiveGenerationRequest) -> ProspectiveGenerationRequest:
    changed = replace(value, source_map_identity=prospective_source_map_identity(value.sources))
    return replace(changed, request_identity=prospective_generation_request_identity(changed))


def _unknown_record(value: Any) -> Any:
    foreign = make_dataclass("UnknownRecord", [], bases=(type(value),), frozen=True)
    return foreign(**{field.name: getattr(value, field.name) for field in fields(value)})


def test_should_freeze_all_recipes_before_source_and_leave_development_ids_unrealized(
    inputs: tuple[Any, ...], generation_request: ProspectiveGenerationRequest
) -> None:
    design, code, _, _ = inputs
    before = deepcopy((design, generation_request))
    result = inspect_prospective_generation_request(design, code, generation_request)
    assert len(generation_request.bindings) == 60 and len(generation_request.streams) == 400
    assert len(generation_request.sources) == 100 and len(result.role_recipes) == 480
    final = [row for row in result.role_recipes if row.role == "final_test"]
    development = [row for row in result.role_recipes if row.role != "final_test"]
    assert len(final) == 120 and len(development) == 360
    assert all(row.declared_sample_ids is None for row in development)
    assert all(len(row.declared_sample_ids or ()) == 40 for row in final)
    assert result.final_id_declarations_complete and not result.development_sample_ids_bound
    assert result.request_identity == generation_request.request_identity
    assert result.source_map_identity == generation_request.source_map_identity
    assert result.required_actual_bindings == tuple(design["required_actual_request_bindings"])
    assert len(result.required_actual_bindings) == 12
    assert result.independent_source_replications is None
    assert not any(
        (
            result.actual_sources_verified,
            result.actual_roles_verified,
            result.actual_code_verified,
            result.chronology_verified,
            result.fresh_roles_authorized,
            result.execution_authorized,
            result.precision_certified,
        )
    )
    assert len(design["cell_plan"]) == 560 and len(design["ordered_primary_statements"]) == 116
    assert (design, generation_request) == before
    with pytest.raises(FrozenInstanceError):
        setattr(result, "execution_authorized", True)


@pytest.mark.parametrize("position", range(8))
def test_should_declare_the_original_phase_geometry_and_label_dependent_assignment_rules(
    inputs: tuple[Any, ...], generation_request: ProspectiveGenerationRequest, position: int
) -> None:
    # Literal oracle for both phases and all four roles, independent of factory internals.
    design, code, _, _ = inputs
    row = generation_request.role_recipes[position]
    phase_b, final = position >= 4, position % 4 == 3
    base = generation_request.bindings[0].base_seed
    assert row.phase == ("b" if phase_b else "a")
    assert row.expected_count == ((36, 12, 12, 40) if phase_b else (72, 24, 24, 40))[position % 4]
    assert row.source_index_count == (40 if final else 120)
    assert row.retained_source_count == (40 if final else 60 if phase_b else 120)
    assert row.assignment_policy == (
        "canonical_final_source_order_v1" if final else "arrived_class_stratified_roles_v1"
    )
    assert row.assignment_seed == (None if final else base + (138 if phase_b else 17))
    assert row.exposure_policy == (
        "not_applied_to_final_test"
        if final
        else "class_balanced_original_source_positions_v1"
        if phase_b
        else "identity_original_source_positions_v1"
    )
    assert row.exposure_seed == (base + 118 if phase_b and not final else None)
    assert row.inner_guard_fraction == (None if final else 0.2)
    assert row.outer_selection_fraction == (None if final else 0.2)
    arrival = "phase_b_arrival" if phase_b else "phase_a_arrival"
    assert row.sample_ids_available_at == ("before_first_source" if final else arrival)
    assert row.source_available_at == ("complete_global_freeze" if final else arrival)
    assert row.labels_available_at == row.source_available_at
    if final:
        assert row.declared_sample_ids == tuple(f"{row.sample_id_prefix}/{i}" for i in range(40))
    else:
        assert row.declared_sample_ids is None
    assert row.array_identity is None and row.final_released is False
    assert (
        inspect_prospective_generation_request(design, code, generation_request).role_recipes[
            position
        ]
        == row
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
        "foreign_slot",
        "boolean_ordinal",
        "float_count",
        "boolean_count",
        "source_geometry",
        "float_geometry",
        "retained_geometry",
        "assignment_policy",
        "assignment_seed",
        "float_assignment_seed",
        "exposure_policy",
        "exposure_seed",
        "inner_fraction",
        "boolean_fraction",
        "outer_fraction",
        "prefix",
        "early_ids",
        "early_id_availability",
        "early_source",
        "early_labels",
        "outer_use",
        "missing_final_ids",
        "reordered_final_ids",
        "mutable_final_ids",
        "final_assignment",
        "array_claim",
        "final_released",
        "release_zero",
        "string_subclass",
    ),
)
def test_should_reject_last_resealed_recipe_drift_without_source_or_rng(
    inputs: tuple[Any, ...], generation_request: ProspectiveGenerationRequest, kind: str
) -> None:
    design, code, _, _ = inputs
    recipes: Any = generation_request.role_recipes
    if kind in {"missing", "extra", "reordered", "duplicate", "mutable", "unknown_record"}:
        recipes = {
            "missing": recipes[:-1],
            "extra": recipes + (recipes[-1],),
            "reordered": recipes[:-2] + (recipes[-1], recipes[-2]),
            "duplicate": recipes[:-1] + (recipes[-2],),
            "mutable": list(recipes),
            "unknown_record": recipes[:-1] + (_unknown_record(recipes[-1]),),
        }[kind]
    else:
        position = (
            472
            if kind == "boolean_ordinal"
            else 479
            if kind
            in {
                "missing_final_ids",
                "reordered_final_ids",
                "mutable_final_ids",
                "final_assignment",
                "array_claim",
                "final_released",
                "release_zero",
            }
            else 478
        )
        row = recipes[position]

        class ForeignString(str):
            pass

        changes: dict[str, dict[str, Any]] = {
            "foreign_slot": {"slot": replace(row.slot, family="unknown")},
            "boolean_ordinal": {"slot": replace(row.slot, ordinal=True)},
            "float_count": {"expected_count": float(row.expected_count)},
            "boolean_count": {"expected_count": True},
            "source_geometry": {"source_index_count": 119},
            "float_geometry": {"source_index_count": 120.0},
            "retained_geometry": {"retained_source_count": 120},
            "assignment_policy": {"assignment_policy": "contiguous_ids"},
            "assignment_seed": {
                "assignment_seed": row.assignment_seed + 1 if row.assignment_seed is not None else 0
            },
            "float_assignment_seed": {"assignment_seed": float(row.assignment_seed or 0)},
            "exposure_policy": {"exposure_policy": "identity_original_source_positions_v1"},
            "exposure_seed": {"exposure_seed": None},
            "inner_fraction": {"inner_guard_fraction": 0.3},
            "boolean_fraction": {"inner_guard_fraction": True},
            "outer_fraction": {"outer_selection_fraction": 0.3},
            "prefix": {"sample_id_prefix": row.sample_id_prefix + "/foreign"},
            "early_ids": {
                "declared_sample_ids": tuple(
                    f"{row.sample_id_prefix}/{i}" for i in range(row.expected_count)
                )
            },
            "early_id_availability": {"sample_ids_available_at": "before_first_source"},
            "early_source": {"source_available_at": "before_first_source"},
            "early_labels": {"labels_available_at": "before_first_source"},
            "outer_use": {"allowed_use": "select_confirmation_settings"},
            "missing_final_ids": {"declared_sample_ids": None},
            "reordered_final_ids": {
                "declared_sample_ids": tuple(reversed(row.declared_sample_ids or ()))
            },
            "mutable_final_ids": {"declared_sample_ids": list(row.declared_sample_ids or ())},
            "final_assignment": {"assignment_seed": 0},
            "array_claim": {"array_identity": EvidenceIdentity(8, "b" * 64)},
            "final_released": {"final_released": True},
            "release_zero": {"final_released": 0},
            "string_subclass": {"assignment_policy": ForeignString(row.assignment_policy)},
        }
        items = list(recipes)
        items[position] = replace(row, **changes[kind])
        recipes = tuple(items)
    changed = _reseal(replace(generation_request, role_recipes=recipes))
    before = deepcopy(changed)
    with pytest.raises(ValueError):
        inspect_prospective_generation_request(design, code, changed)
    assert changed == before


@pytest.mark.parametrize(
    "kind",
    (
        "schema",
        "unknown_request",
        "execution",
        "execution_zero",
        "missing_proof",
        "reordered_proofs",
        "mutable_proofs",
        "source_missing",
        "source_reordered",
        "source_code_pin",
        "source_configuration_pin",
        "late_stream",
        "mutable_streams",
        "missing_binding",
        "late_binding",
        "unknown_identity",
        "design_pin",
        "map_pin",
        "request_pin",
    ),
)
def test_should_reject_full_envelope_corruption_even_after_resealing(
    inputs: tuple[Any, ...], generation_request: ProspectiveGenerationRequest, kind: str
) -> None:
    design, code, _, _ = inputs
    value = generation_request
    changes: dict[str, dict[str, Any]] = {
        "schema": {"schema_id": "p67_complete_prospective_role_source_request_v1"},
        "unknown_request": {},
        "execution": {"execution_requested": True},
        "execution_zero": {"execution_requested": 0},
        "missing_proof": {"required_actual_bindings": value.required_actual_bindings[:-1]},
        "reordered_proofs": {
            "required_actual_bindings": tuple(reversed(value.required_actual_bindings))
        },
        "mutable_proofs": {"required_actual_bindings": list(value.required_actual_bindings)},
        "source_missing": {"sources": value.sources[:-1]},
        "source_reordered": {
            "sources": value.sources[:-2] + (value.sources[-1], value.sources[-2])
        },
        "source_code_pin": {
            "sources": value.sources[:-1]
            + (replace(value.sources[-1], code_identity=EvidenceIdentity(123, "b" * 64)),)
        },
        "source_configuration_pin": {
            "sources": value.sources[:-1]
            + (
                replace(
                    value.sources[-1],
                    generator_configuration_identity=EvidenceIdentity(123, "b" * 64),
                ),
            )
        },
        "late_stream": {
            "streams": value.streams[:-1]
            + (replace(value.streams[-1], value=value.streams[-1].value + 1),)
        },
        "mutable_streams": {"streams": list(value.streams)},
        "missing_binding": {"bindings": value.bindings[:-1]},
        "late_binding": {
            "bindings": value.bindings[:-1]
            + (replace(value.bindings[-1], base_seed=value.bindings[-1].base_seed + 1),)
        },
        "unknown_identity": {"design_identity": _unknown_record(value.design_identity)},
        "design_pin": {"design_identity": EvidenceIdentity(123, "b" * 64)},
        "map_pin": {"source_map_identity": EvidenceIdentity(123, "b" * 64)},
        "request_pin": {"request_identity": EvidenceIdentity(123, "b" * 64)},
    }
    changed = (
        _unknown_record(value) if kind == "unknown_request" else replace(value, **changes[kind])
    )
    if kind not in {"map_pin", "request_pin"}:
        changed = _reseal(changed)
    before = deepcopy(changed)
    with pytest.raises(ValueError):
        inspect_prospective_generation_request(design, code, changed)
    assert changed == before


def test_should_bind_complete_concrete_role_claims_to_frozen_recipes_without_arrival_proof(
    inputs: tuple[Any, ...], generation_request: ProspectiveGenerationRequest
) -> None:
    design, code, bindings, _ = inputs
    declarations = concrete_role_declarations(design, bindings)
    before = deepcopy((generation_request, declarations))
    result = inspect_arrived_role_declarations(design, code, generation_request, declarations)
    assert result.generation_request_identity == generation_request.request_identity
    assert len(result.concrete_role_inspection.roles) == 480
    assert result.concrete_role_inspection.roles == declarations
    assert (
        result.concrete_role_inspection.required_actual_bindings
        == generation_request.required_actual_bindings
    )
    assert not result.actual_arrival_verified and not result.assignment_execution_verified
    assert not result.concrete_role_inspection.actual_roles_verified
    assert result.concrete_role_inspection.independent_source_replications is None
    assert not result.concrete_role_inspection.execution_authorized
    assert (generation_request, declarations) == before
    assert all(
        row.declared_sample_ids is None
        for row in generation_request.role_recipes
        if row.role != "final_test"
    )


@pytest.mark.parametrize(
    "kind",
    (
        "missing",
        "reordered",
        "mutable",
        "unknown_record",
        "float_count",
        "overlap",
        "shared_view",
        "final_reordered",
        "final_released",
        "early_labels",
        "outer_use",
        "array_claim",
        "generation_recipe",
        "expected_code",
    ),
)
def test_should_reject_late_concrete_role_claim_or_generation_request_drift(
    inputs: tuple[Any, ...], generation_request: ProspectiveGenerationRequest, kind: str
) -> None:
    design, code, bindings, _ = inputs
    declarations: Any = concrete_role_declarations(design, bindings)
    request = generation_request
    if kind == "missing":
        declarations = declarations[:-1]
    elif kind == "reordered":
        declarations = declarations[:-2] + (declarations[-1], declarations[-2])
    elif kind == "mutable":
        declarations = list(declarations)
    elif kind == "unknown_record":
        declarations = declarations[:-1] + (_unknown_record(declarations[-1]),)
    elif kind == "generation_recipe":
        request = _reseal(
            replace(
                request,
                role_recipes=request.role_recipes[:-1]
                + (replace(request.role_recipes[-1], labels_available_at="phase_b_arrival"),),
            )
        )
    elif kind == "expected_code":
        code = EvidenceIdentity(123, "b" * 64)
    else:
        position = (
            84
            if kind == "shared_view"
            else 476
            if kind == "overlap"
            else 478
            if kind == "outer_use"
            else 479
        )
        row = declarations[position]
        prefix = row.sample_ids[0].rsplit("/", 1)[0]
        changes: dict[str, dict[str, Any]] = {
            "float_count": {"expected_count": float(row.expected_count)},
            "overlap": {"sample_ids": row.sample_ids[1:] + (prefix + "/96",)},
            "shared_view": {"sample_ids": tuple(f"{prefix}/{i}" for i in range(24, 60))},
            "final_reordered": {"sample_ids": tuple(reversed(row.sample_ids))},
            "final_released": {"final_released": True},
            "early_labels": {"labels_available_at": "phase_b_arrival"},
            "outer_use": {"allowed_use": "select_confirmation_settings"},
            "array_claim": {"array_identity": EvidenceIdentity(8, "b" * 64)},
        }
        rows = list(declarations)
        rows[position] = replace(row, **changes[kind])
        declarations = tuple(rows)
    before = deepcopy((request, declarations))
    with pytest.raises(ValueError):
        inspect_arrived_role_declarations(design, code, request, declarations)
    assert (request, declarations) == before


@pytest.mark.parametrize(
    "kind",
    (
        "foreign_row",
        "foreign_slot",
        "foreign_id",
        "foreign_array",
        "foreign_policy",
        "foreign_ids_container",
    ),
)
def test_should_reject_foreign_concrete_payload_before_copying_or_encoding_it(
    inputs: tuple[Any, ...], generation_request: ProspectiveGenerationRequest, kind: str
) -> None:
    design, code, bindings, _ = inputs
    declarations: Any = concrete_role_declarations(design, bindings)

    class CopySentinel:
        calls = 0

        def __deepcopy__(self, memo: Any) -> Any:
            self.calls += 1
            raise AssertionError("foreign concrete payload was copied before type rejection")

    payload = CopySentinel()
    row = declarations[-1]
    changes: dict[str, dict[str, Any]] = {
        "foreign_row": {},
        "foreign_slot": {"slot": payload},
        "foreign_id": {"sample_ids": row.sample_ids[:-1] + (payload,)},
        "foreign_array": {"array_identity": payload},
        "foreign_policy": {"allowed_use": payload},
        "foreign_ids_container": {"sample_ids": payload},
    }
    declarations = declarations[:-1] + (
        payload if kind == "foreign_row" else replace(row, **changes[kind]),
    )
    with pytest.raises(ValueError):
        inspect_arrived_role_declarations(design, code, generation_request, declarations)
    assert payload.calls == 0
