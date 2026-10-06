"""Complete external metadata claims with literal offsets and raising IO/science guards."""

from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.app import continual_arrived_benchmark as arrived
from src.app.prospective_confirmation_design import fixed_prospective_design
from src.app.prospective_stream_declarations import validate_prospective_stream_declarations
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.predictive_coding import PredictiveCodingNetwork
from src.core.prospective_replications import ReplicaSeedBinding, ReplicaSlot
from src.core.seed_stream_screening import DerivedSeedStream, EvidenceIdentity

OFFSETS = (
    ("phase_a_source", 0),
    ("phase_b_source", 101),
    ("phase_a_roles", 17),
    ("phase_b_roles", 138),
    ("phase_b_exposure", 118),
    ("model_initialization", 1001),
    ("parent_selection", 5001),
    ("circadian_local_rng", 11002),
)


def _identity(design: dict[str, Any]) -> EvidenceIdentity:
    raw = (
        json.dumps(design, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    return EvidenceIdentity(len(raw), sha256(raw).hexdigest())


def _claims(bindings: tuple[ReplicaSeedBinding, ...]) -> tuple[DerivedSeedStream, ...]:
    groups = dict((row.slot.source_group, row.base_seed) for row in bindings)
    return tuple(
        DerivedSeedStream(seed, name, seed + offset)
        for seed in groups.values()
        for name, offset in OFFSETS
    )


@dataclass(frozen=True)
class StreamFixture:
    design: dict[str, Any]
    bindings: tuple[ReplicaSeedBinding, ...]
    identity: EvidenceIdentity
    claims: tuple[DerivedSeedStream, ...]


@pytest.fixture(autouse=True)
def no_science_or_io(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "stream claim validation constructed scientific values or performed IO"
        )

    for name in ("_build_phase_a_roles", "_build_phase_b_roles", "release_final_test"):
        monkeypatch.setattr(arrived, name, forbidden)
    for model in (BackpropMLP, PredictiveCodingNetwork, CircadianPredictiveCodingNetwork):
        monkeypatch.setattr(model, "__init__", forbidden)
        monkeypatch.setattr(model, "compute_accuracy", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)


@pytest.fixture
def fixture() -> StreamFixture:
    design = fixed_prospective_design()
    slots = tuple(ReplicaSlot(**row) for row in design["replica_slots"])
    groups = tuple(dict.fromkeys(row.source_group for row in slots))
    # Existing full-layout synthetic values, never a scientific seed selection.
    bindings = tuple(
        ReplicaSeedBinding(slot, 1_000_000 + 20_000 * groups.index(slot.source_group))
        for slot in slots
    )
    return StreamFixture(design, bindings, _identity(design), _claims(bindings))


def _validate(fixture: StreamFixture, claims: Any = None) -> Any:
    return validate_prospective_stream_declarations(
        fixture.design,
        fixture.bindings,
        fixture.identity,
        fixture.claims if claims is None else claims,
    )


def test_should_validate_all_four_hundred_claims_without_mutation_or_authority(
    fixture: StreamFixture,
) -> None:
    before = deepcopy(fixture)
    result = _validate(fixture)
    assert len(result.bindings) == 60 and len(result.source_groups) == 50
    assert result.streams == fixture.claims and len(result.streams) == 400
    assert result.bindings == fixture.bindings
    assert result.independent_source_replications is None
    assert result.fresh_roles_authorized is False
    assert fixture == before
    assert len(fixture.design["cell_plan"]) == 560
    assert len(fixture.design["role_requirements"]) == 480
    assert len(fixture.design["ordered_primary_statements"]) == 116


@dataclass(frozen=True)
class ForeignStream(DerivedSeedStream):
    rng_domain: str | None = None


class StreamName(str):
    """A value-equal string subclass must not extend the exact stream schema."""


@pytest.mark.parametrize("field", ("base_seed", "value", "name"))
def test_should_refuse_inexact_types_that_compare_equal_to_a_required_stream(
    fixture: StreamFixture, field: str
) -> None:
    first_group = fixture.bindings[0].slot.source_group
    bindings = tuple(
        replace(row, base_seed=1) if row.slot.source_group == first_group else row
        for row in fixture.bindings
    )
    exact = _claims(bindings)
    changed_value: Any = StreamName(exact[0].name) if field == "name" else True
    altered = replace(exact[0], **{field: changed_value})
    assert altered == exact[0]
    changed = replace(fixture, bindings=bindings, claims=(altered,) + exact[1:])
    before = deepcopy(changed)
    with pytest.raises(ValueError, match="stream"):
        _validate(changed)
    assert changed == before


@pytest.mark.parametrize(
    "kind",
    (
        "missing",
        "extra",
        "reordered",
        "duplicate",
        "mutable",
        "untyped_unknown_domain",
        "subclass_unknown_domain",
        "boolean_base",
        "nonstring_name",
        "boolean_value",
        "float_value",
        "symbolic_value",
        "negative_value",
        "foreign_base",
        "foreign_name",
        "wrong_derived_value",
    ),
)
def test_should_refuse_late_partial_reordered_or_untrusted_external_claims(
    fixture: StreamFixture, kind: str
) -> None:
    claims: Any = fixture.claims
    last = claims[-1]
    if kind == "missing":
        claims = claims[:-1]
    elif kind == "extra":
        claims += (last,)
    elif kind == "reordered":
        claims = claims[:-2] + (claims[-1], claims[-2])
    elif kind == "duplicate":
        claims = claims[:-1] + (claims[-2],)
    elif kind == "mutable":
        claims = list(claims)
    elif kind == "untyped_unknown_domain":
        claims = claims[:-1] + (
            {
                "base_seed": last.base_seed,
                "name": last.name,
                "value": last.value,
                "rng_domain": None,
            },
        )
    elif kind == "subclass_unknown_domain":
        claims = claims[:-1] + (ForeignStream(last.base_seed, last.name, last.value),)
    else:
        changes: dict[str, dict[str, Any]] = {
            "boolean_base": {"base_seed": True},
            "nonstring_name": {"name": 7},
            "boolean_value": {"value": True},
            "float_value": {"value": float(last.value)},
            "symbolic_value": {"value": "base + unresolved_offset"},
            "negative_value": {"value": -1},
            "foreign_base": {"base_seed": last.base_seed + 1},
            "foreign_name": {"name": "unknown_rng_domain"},
            "wrong_derived_value": {"value": last.value + 1},
        }
        claims = claims[:-1] + (replace(last, **changes[kind]),)
    before = deepcopy((fixture, claims))
    with pytest.raises(ValueError, match="stream"):
        _validate(fixture, claims)
    assert (fixture, claims) == before


@pytest.mark.parametrize(
    "kind", ("boolean_bytes", "float_bytes", "wrong_bytes", "wrong_digest", "uppercase", "untyped")
)
def test_should_refuse_a_detached_or_inexact_whole_design_identity(
    fixture: StreamFixture, kind: str
) -> None:
    changes: dict[str, Any] = {
        "boolean_bytes": {"byte_count": True},
        "float_bytes": {"byte_count": float(fixture.identity.byte_count)},
        "wrong_bytes": {"byte_count": fixture.identity.byte_count + 1},
        "wrong_digest": {"sha256": "a" * 64},
        "uppercase": {"sha256": fixture.identity.sha256.upper()},
    }
    pin: Any = (
        {"byte_count": fixture.identity.byte_count, "sha256": fixture.identity.sha256}
        if kind == "untyped"
        else replace(fixture.identity, **changes[kind])
    )
    altered = replace(fixture, identity=pin)
    before = deepcopy(altered)
    with pytest.raises(ValueError, match="identity"):
        _validate(altered)
    assert altered == before


@pytest.mark.parametrize("kind", ("omit_cell", "omit_role", "change_contrast", "admission"))
def test_should_refuse_resealed_scope_changes_even_with_matching_streams(
    fixture: StreamFixture, kind: str
) -> None:
    design = deepcopy(fixture.design)
    if kind == "omit_cell":
        design["cell_plan"].pop()
    elif kind == "omit_role":
        design["role_requirements"].pop()
    elif kind == "change_contrast":
        design["ordered_primary_statements"][-1]["metric"] = "favorable_metric"
    else:
        design["fresh_roles_authorized"] = True
    changed = replace(fixture, design=design, identity=_identity(design))
    before = deepcopy(changed)
    with pytest.raises(ValueError):
        _validate(changed)
    assert changed == before


@pytest.mark.parametrize("kind", ("missing", "reordered", "boolean_seed", "shared_group"))
def test_should_refuse_bad_full_binding_layout_before_trusting_stream_claims(
    fixture: StreamFixture, kind: str
) -> None:
    bindings = fixture.bindings
    if kind == "missing":
        bindings = bindings[:-1]
    elif kind == "reordered":
        bindings = tuple(reversed(bindings))
    elif kind == "boolean_seed":
        bindings = bindings[:-1] + (replace(bindings[-1], base_seed=True),)
    else:
        bindings = bindings[:10] + (replace(bindings[10], base_seed=7_000_000),) + bindings[11:]
    changed = replace(fixture, bindings=bindings)
    before = deepcopy(changed)
    with pytest.raises(ValueError):
        _validate(changed)
    assert changed == before


@pytest.mark.parametrize("offset", tuple(offset for _, offset in OFFSETS))
def test_should_refuse_self_consistent_stream_claims_with_cross_group_collisions(
    fixture: StreamFixture, offset: int
) -> None:
    bindings = fixture.bindings[:-1] + (
        replace(fixture.bindings[-1], base_seed=fixture.bindings[0].base_seed + offset),
    )
    changed = replace(fixture, bindings=bindings, claims=_claims(bindings))
    before = deepcopy(changed)
    with pytest.raises(ValueError, match="stream"):
        _validate(changed)
    assert changed == before
