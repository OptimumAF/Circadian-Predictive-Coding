"""Real whole V2 file/port regressions; scientific work is forbidden, IO is intentional."""

from dataclasses import asdict, fields, make_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from prospective_generation_bundle_fixtures import (
    GenerationBundleFixture,
    make_generation_bundle,
    reseal_generation_request,
)
from prospective_request_bundle_fixtures import canonical, make_bundle, raw_identity
from src.app import continual_arrived_benchmark as arrived
from src.app.prospective_confirmation_design import fixed_prospective_design
from src.app.prospective_generation_bundles import preflight_prospective_generation_bundle
from src.app.prospective_request_bundles import preflight_prospective_request_bundle
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.predictive_coding import PredictiveCodingNetwork
from src.core.prospective_generation_bundles import GenerationBundleSnapshot
from src.core.prospective_request_bundles import ProspectiveBundleSpec
from src.core.seed_stream_screening import EvidenceIdentity
from src.infra.prospective_generation_bundles import FileProspectiveGenerationBundleReader
from src.infra.prospective_request_bundles import FileProspectiveBundleReader


@pytest.fixture(autouse=True)
def no_science(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("generation bundle touched source/array/RNG/model/final/scoring work")

    for name in ("_build_phase_a_roles", "_build_phase_b_roles", "release_final_test"):
        monkeypatch.setattr(arrived, name, forbidden)
    for model in (BackpropMLP, PredictiveCodingNetwork, CircadianPredictiveCodingNetwork):
        monkeypatch.setattr(model, "__init__", forbidden)
        monkeypatch.setattr(model, "compute_accuracy", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    for name in ("array", "asarray", "ascontiguousarray"):
        monkeypatch.setattr(np, name, forbidden)


@pytest.fixture
def bundle(tmp_path: Path) -> GenerationBundleFixture:
    return make_generation_bundle(tmp_path)


def _reader(bundle: GenerationBundleFixture) -> FileProspectiveGenerationBundleReader:
    return FileProspectiveGenerationBundleReader(
        bundle.root, clock=lambda: datetime(2026, 10, 5, 13, tzinfo=timezone.utc)
    )


def _preflight(bundle: GenerationBundleFixture) -> Any:
    return preflight_prospective_generation_bundle(
        fixed_prospective_design(), bundle.spec, _reader(bundle)
    )


def _drift(bundle: GenerationBundleFixture, kind: str) -> None:
    name = {
        "request": bundle.spec.request_path,
        "sources": bundle.spec.source_map_path,
        "manifest": bundle.spec.code_manifest_path,
        "last_code": bundle.spec.code_paths[-1],
    }[kind]
    path = bundle.root / name
    raw = path.read_bytes()
    path.write_bytes(raw[:-2] + bytes((raw[-2] ^ 1,)) + raw[-1:])


def test_should_bind_every_v2_physical_file_and_recipe_without_actual_admission(
    bundle: GenerationBundleFixture,
) -> None:
    result = _preflight(bundle)
    request = result.snapshot.generation_request
    assert result.physical_files_verified
    assert len(request.role_recipes) == 480 and len(request.sources) == 100
    assert len(request.streams) == 400 and len(request.bindings) == 60
    assert len(result.snapshot.code_files) == 5
    assert result.generation_inspection.request_identity == request.request_identity
    assert result.generation_inspection.final_id_declarations_complete
    assert not result.generation_inspection.development_sample_ids_bound
    assert all(
        row.declared_sample_ids is None for row in request.role_recipes if row.role != "final_test"
    )
    assert len([row for row in request.role_recipes if row.role == "final_test"]) == 120
    assert result.required_actual_bindings == tuple(
        fixed_prospective_design()["required_actual_request_bindings"]
    )
    assert len(result.required_actual_bindings) == 12
    assert result.independent_source_replications is None
    assert not any(
        (
            result.runtime_code_closure_verified,
            result.exclusive_owner_verified,
            result.chronology_verified,
            result.actual_sources_verified,
            result.actual_roles_verified,
            result.fresh_roles_authorized,
            result.execution_authorized,
            result.precision_certified,
        )
    )
    assert _preflight(bundle) == result


@pytest.mark.parametrize("kind", ("request", "sources", "manifest", "last_code"))
def test_should_reject_whole_equal_size_byte_drift_in_every_physical_file_kind(
    bundle: GenerationBundleFixture,
    kind: str,
) -> None:
    _drift(bundle, kind)
    with pytest.raises(ValueError):
        _preflight(bundle)


@pytest.mark.parametrize(
    "kind",
    (
        "missing_recipe",
        "reordered_recipe",
        "unknown_recipe",
        "boolean_count",
        "float_split_seed",
        "early_development_ids",
        "missing_final_ids",
        "reordered_final_ids",
        "exposure_rule",
        "early_labels",
        "outer_use",
        "array_claim",
        "final_release",
        "missing_source",
        "late_source_code",
        "late_stream",
        "missing_proof",
        "generation_schema",
        "bundle_schema",
        "execution",
        "unknown_request_field",
    ),
)
def test_should_reject_late_full_recipe_source_or_request_corruption_after_resealing(
    bundle: GenerationBundleFixture,
    kind: str,
) -> None:
    request = bundle.body["generation_request"]
    recipes = request["role_recipes"]
    if kind == "missing_recipe":
        request["role_recipes"] = recipes[:-1]
    elif kind == "reordered_recipe":
        request["role_recipes"] = recipes[:-2] + [recipes[-1], recipes[-2]]
    elif kind == "unknown_recipe":
        recipes[-1]["unknown"] = False
    elif kind == "boolean_count":
        recipes[-1]["expected_count"] = True
    elif kind == "float_split_seed":
        recipes[-2]["assignment_seed"] = float(recipes[-2]["assignment_seed"])
    elif kind == "early_development_ids":
        recipes[-2]["declared_sample_ids"] = [
            recipes[-2]["sample_id_prefix"] + f"/{i}" for i in range(12)
        ]
    elif kind == "missing_final_ids":
        recipes[-1]["declared_sample_ids"] = None
    elif kind == "reordered_final_ids":
        recipes[-1]["declared_sample_ids"].reverse()
    elif kind == "exposure_rule":
        recipes[-2]["exposure_policy"] = "identity_original_source_positions_v1"
    elif kind == "early_labels":
        recipes[-1]["labels_available_at"] = "phase_b_arrival"
    elif kind == "outer_use":
        recipes[-2]["allowed_use"] = "choose_confirmation_settings"
    elif kind == "array_claim":
        recipes[-1]["array_identity"] = {"byte_count": 8, "sha256": "b" * 64}
    elif kind == "final_release":
        recipes[-1]["final_released"] = True
    elif kind == "missing_source":
        request["sources"] = request["sources"][:-1]
    elif kind == "late_source_code":
        request["sources"][-1]["code_identity"]["sha256"] = "b" * 64
    elif kind == "late_stream":
        request["streams"][-1]["value"] += 1
    elif kind == "missing_proof":
        request["required_actual_bindings"] = request["required_actual_bindings"][:-1]
    elif kind == "generation_schema":
        request["schema_id"] = "p67_complete_prospective_role_source_request_v1"
    elif kind == "bundle_schema":
        bundle.body["schema_id"] = "p67_complete_prospective_request_bundle_v1"
    elif kind == "execution":
        request["execution_requested"] = True
    else:
        request["unknown"] = False
    reseal_generation_request(bundle.body)
    bundle.save()
    with pytest.raises(ValueError):
        _preflight(bundle)


@pytest.mark.parametrize(
    "kind",
    (
        "future_utc",
        "naive_utc",
        "noncanonical_utc",
        "short_owner",
        "uppercase_owner",
        "resource_omission",
        "resource_changed",
        "repeat_pin",
        "repeat_caps",
        "repeat_required",
        "repeat_actual_claim",
        "repeat_omission",
    ),
)
def test_should_reject_canonical_time_owner_or_full_resource_repeat_declaration_drift(
    bundle: GenerationBundleFixture,
    kind: str,
) -> None:
    if kind == "future_utc":
        bundle.body["prospective_utc"] = "2026-10-05T14:00:00+00:00"
    elif kind == "naive_utc":
        bundle.body["prospective_utc"] = "2026-10-05T12:00:00"
    elif kind == "noncanonical_utc":
        bundle.body["prospective_utc"] = "2026-10-05T12:00:00Z"
    elif kind == "short_owner":
        bundle.body["owner_id"] = "1" * 31
    elif kind == "uppercase_owner":
        bundle.body["owner_id"] = "A" * 32
    elif kind == "resource_omission":
        bundle.body["resource_envelope"].pop(next(reversed(bundle.body["resource_envelope"])))
    elif kind == "resource_changed":
        bundle.body["resource_envelope"]["unknown_cap"] = 1
    else:
        repeat = bundle.body["independent_repeat_envelope"]
        if kind == "repeat_pin":
            repeat["generation_request_identity"]["sha256"] = "b" * 64
        elif kind == "repeat_caps":
            repeat["caps_identical"] = False
        elif kind == "repeat_required":
            repeat["required"] = False
        elif kind == "repeat_actual_claim":
            repeat["actual_independent_repeat_verified"] = True
        else:
            repeat.pop("success_failure_and_every_attempt_charged")
    bundle.save()
    with pytest.raises(ValueError):
        _preflight(bundle)


@pytest.mark.parametrize(
    "kind",
    (
        "missing",
        "extra",
        "reordered",
        "duplicate",
        "unknown_field",
        "foreign_path",
        "float_identity",
    ),
)
def test_should_reject_resealed_closed_code_membership_or_identity_drift(
    bundle: GenerationBundleFixture,
    kind: str,
) -> None:
    files = bundle.manifest["files"]
    if kind == "missing":
        bundle.manifest["files"] = files[:-1]
    elif kind == "extra":
        files.append(files[-1])
    elif kind == "reordered":
        bundle.manifest["files"] = files[:-2] + [files[-1], files[-2]]
    elif kind == "duplicate":
        files[-1] = files[-2]
    elif kind == "unknown_field":
        files[-1]["unknown"] = False
    elif kind == "foreign_path":
        files[-1]["path"] = "other.py"
    else:
        files[-1]["identity"]["byte_count"] = float(files[-1]["identity"]["byte_count"])
    code_pin = asdict(raw_identity(canonical(bundle.manifest)))
    for source in bundle.body["generation_request"]["sources"]:
        source["code_identity"] = code_pin
    reseal_generation_request(bundle.body)
    bundle.save()
    with pytest.raises(ValueError):
        _preflight(bundle)


@pytest.mark.parametrize(
    "kind",
    ("escape", "absolute", "backslash", "case_alias", "mutable_members", "duplicate", "empty_pin"),
)
def test_should_reject_unsafe_specs_before_calling_the_io_port(
    bundle: GenerationBundleFixture,
    kind: str,
) -> None:
    changes: dict[str, dict[str, Any]] = {
        "escape": {"request_path": "../request.json"},
        "absolute": {"request_path": "C:/request.json"},
        "backslash": {"request_path": "dir\\request.json"},
        "case_alias": {"source_map_path": "REQUEST.JSON"},
        "mutable_members": {"code_paths": list(bundle.spec.code_paths)},
        "duplicate": {"code_paths": bundle.spec.code_paths[:-1] + (bundle.spec.code_paths[-2],)},
        "empty_pin": {"request_identity": EvidenceIdentity(0, "0" * 64)},
    }

    class ForbiddenPort:
        def read_bundle(self, spec: ProspectiveBundleSpec) -> GenerationBundleSnapshot:
            raise AssertionError("invalid spec reached IO")

        def recheck_bundle(self, snapshot: GenerationBundleSnapshot) -> None:
            raise AssertionError("invalid spec reached IO")

    with pytest.raises(ValueError):
        preflight_prospective_generation_bundle(
            fixed_prospective_design(), replace(bundle.spec, **changes[kind]), ForbiddenPort()
        )


@pytest.mark.parametrize("kind", ("claim", "failure", "hardlink"))
def test_should_reject_pending_failed_or_aliased_physical_namespace(
    bundle: GenerationBundleFixture,
    kind: str,
) -> None:
    if kind == "hardlink":
        original = bundle.root / bundle.spec.code_paths[0]
        alias = bundle.root / bundle.spec.code_paths[-1]
        alias.unlink()
        alias.hardlink_to(original)
    else:
        (bundle.root / bundle.spec.request_path).with_suffix(
            ".claim" if kind == "claim" else ".failure.json"
        ).write_bytes(b"{}\n")
    with pytest.raises(ValueError):
        _preflight(bundle)


@pytest.mark.parametrize("kind", ("request", "sources", "manifest", "last_code"))
def test_should_reject_late_physical_drift_after_full_app_inspection(
    bundle: GenerationBundleFixture,
    kind: str,
) -> None:
    reader = _reader(bundle)

    class LatePort:
        def read_bundle(self, spec: ProspectiveBundleSpec) -> GenerationBundleSnapshot:
            return reader.read_bundle(spec)

        def recheck_bundle(self, snapshot: GenerationBundleSnapshot) -> None:
            _drift(bundle, kind)
            reader.recheck_bundle(snapshot)

    with pytest.raises(ValueError):
        preflight_prospective_generation_bundle(fixed_prospective_design(), bundle.spec, LatePort())


@pytest.mark.parametrize(
    "kind",
    ("unknown_record", "mutable_code", "foreign_code_row", "detached_spec", "foreign_request"),
)
def test_should_reject_foreign_snapshot_or_code_records_from_the_inner_port(
    bundle: GenerationBundleFixture,
    kind: str,
) -> None:
    snapshot: Any = _reader(bundle).read_bundle(bundle.spec)
    if kind in {"unknown_record", "foreign_request"}:
        value = snapshot if kind == "unknown_record" else snapshot.generation_request
        foreign = make_dataclass("UnknownRecord", [], bases=(type(value),), frozen=True)
        changed = foreign(**{field.name: getattr(value, field.name) for field in fields(value)})
        snapshot = (
            changed if kind == "unknown_record" else replace(snapshot, generation_request=changed)
        )
    elif kind == "mutable_code":
        snapshot = replace(snapshot, code_files=list(snapshot.code_files))
    elif kind == "foreign_code_row":
        snapshot = replace(
            snapshot, code_files=snapshot.code_files[:-1] + (asdict(snapshot.code_files[-1]),)
        )
    else:
        snapshot = replace(
            snapshot, spec=replace(bundle.spec, request_identity=EvidenceIdentity(1, "b" * 64))
        )

    class ForeignPort:
        def read_bundle(self, spec: ProspectiveBundleSpec) -> GenerationBundleSnapshot:
            return snapshot

        def recheck_bundle(self, checked: GenerationBundleSnapshot) -> None:
            raise AssertionError("foreign snapshot reached recheck")

    with pytest.raises(ValueError):
        preflight_prospective_generation_bundle(
            fixed_prospective_design(), bundle.spec, ForeignPort()
        )


@pytest.mark.parametrize("consumer", ("v1", "v2"))
def test_should_preserve_v1_and_v2_schema_separation(
    bundle: GenerationBundleFixture,
    tmp_path: Path,
    consumer: str,
) -> None:
    with pytest.raises(ValueError):
        if consumer == "v1":
            preflight_prospective_request_bundle(
                fixed_prospective_design(), bundle.spec, FileProspectiveBundleReader(bundle.root)
            )
        else:
            other = tmp_path / "v1"
            other.mkdir()
            old = make_bundle(other)
            preflight_prospective_generation_bundle(
                fixed_prospective_design(), old.spec, FileProspectiveGenerationBundleReader(other)
            )


@pytest.mark.parametrize("version", ("v1", "v2"))
def test_should_reject_snapshot_source_numeric_alias_during_direct_whole_recheck(
    bundle: GenerationBundleFixture, tmp_path: Path, version: str
) -> None:
    reader: Any
    if version == "v1":
        other = tmp_path / "direct-v1"
        other.mkdir()
        old = make_bundle(other)
        reader = FileProspectiveBundleReader(other)
        snapshot: Any = reader.read_bundle(old.spec)
        field_name = "role_request"
    else:
        reader = _reader(bundle)
        snapshot = reader.read_bundle(bundle.spec)
        field_name = "generation_request"
    request = getattr(snapshot, field_name)
    source = request.sources[-1]
    changed_identity = replace(
        source.code_identity, **{"byte_count": float(source.code_identity.byte_count)}
    )
    changed_source = replace(source, code_identity=changed_identity)
    changed_request = replace(request, sources=request.sources[:-1] + (changed_source,))
    changed = replace(snapshot, **{field_name: changed_request})
    # Dataclass numeric equality permits this alias. Whole source-map comparison
    # must still reject it, without relying on a later app inspection.
    with pytest.raises(ValueError):
        reader.recheck_bundle(changed)
