"""Full-envelope real-file corruption tests; metadata IO is allowed, science is sealed."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.app import continual_arrived_benchmark as arrived
from src.app.prospective_confirmation_design import fixed_prospective_design
from src.app.prospective_request_bundles import preflight_prospective_request_bundle
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.predictive_coding import PredictiveCodingNetwork
from src.core.prospective_request_bundles import BundleSnapshot, ProspectiveBundleSpec
from src.core.seed_stream_screening import EvidenceIdentity
from src.infra.prospective_request_bundles import FileProspectiveBundleReader
from prospective_request_bundle_fixtures import (
    BundleFixture,
    make_bundle,
    raw_identity,
    repeat_declaration,
    reseal_role_request,
)


@pytest.fixture(autouse=True)
def no_science(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("prospective file preflight performed scientific work")

    for name in ("_build_phase_a_roles", "_build_phase_b_roles", "release_final_test"):
        monkeypatch.setattr(arrived, name, forbidden)
    for model in (BackpropMLP, PredictiveCodingNetwork, CircadianPredictiveCodingNetwork):
        monkeypatch.setattr(model, "__init__", forbidden)
        monkeypatch.setattr(model, "compute_accuracy", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)


@pytest.fixture
def bundle(tmp_path: Path) -> BundleFixture:
    return make_bundle(tmp_path)


def _reader(bundle: BundleFixture) -> FileProspectiveBundleReader:
    return FileProspectiveBundleReader(
        bundle.root, clock=lambda: datetime(2026, 10, 5, 13, tzinfo=timezone.utc)
    )


def _inspect(bundle: BundleFixture) -> Any:
    return preflight_prospective_request_bundle(
        fixed_prospective_design(), bundle.spec, _reader(bundle)
    )


def test_should_verify_all_physical_files_and_preserve_every_actual_proof_obligation(
    bundle: BundleFixture,
) -> None:
    before = deepcopy(bundle.body)
    result = _inspect(bundle)
    assert result.physical_files_verified
    assert (
        len(result.snapshot.role_request.roles) == 480
        and len(result.snapshot.role_request.sources) == 100
    )
    assert (
        len(result.snapshot.role_request.streams) == 400
        and len(result.snapshot.role_request.bindings) == 60
    )
    assert len(result.snapshot.code_files) == 5
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
            result.fresh_roles_authorized,
            result.execution_authorized,
            result.precision_certified,
        )
    )
    assert not result.role_inspection.actual_code_verified
    assert bundle.body == before


@pytest.mark.parametrize(
    "kind",
    (
        "unknown_schema",
        "unknown_field",
        "missing_field",
        "bad_utc",
        "naive_utc",
        "future_utc",
        "boolean_utc",
        "empty_owner",
        "uppercase_owner",
        "boolean_owner",
        "cap_change",
        "boolean_cap",
        "fit_claim",
        "repeat_claim",
        "repeat_missing",
        "repeat_detached",
        "repeat_unknown",
        "array_claim",
        "late_role",
        "late_stream",
        "missing_role",
        "missing_source",
        "execute",
    ),
)
def test_should_reject_resealed_request_header_scope_and_authority_corruption(
    bundle: BundleFixture, kind: str
) -> None:
    body = bundle.body
    if kind == "unknown_schema":
        body["schema_id"] = "unknown"
    elif kind == "unknown_field":
        body["execution_authorized"] = True
    elif kind == "missing_field":
        del body["owner_id"]
    elif kind in {"bad_utc", "naive_utc", "future_utc", "boolean_utc"}:
        body["prospective_utc"] = {
            "bad_utc": "yesterday",
            "naive_utc": "2026-10-05T12:00:00",
            "future_utc": "2026-10-05T14:00:00+00:00",
            "boolean_utc": True,
        }[kind]
    elif kind in {"empty_owner", "uppercase_owner", "boolean_owner"}:
        body["owner_id"] = {"empty_owner": "", "uppercase_owner": "A" * 32, "boolean_owner": True}[
            kind
        ]
    elif kind in {"cap_change", "boolean_cap", "fit_claim"}:
        envelope = body["resource_envelope"]
        envelope[
            "maximum_optimizer_updates"
            if kind != "fit_claim"
            else "future_time_and_memory_fit_measured"
        ] = 16001 if kind == "cap_change" else True
    elif kind.startswith("repeat_"):
        repeat = body["independent_repeat_envelope"]
        if kind == "repeat_claim":
            repeat["actual_independent_repeat_verified"] = True
        elif kind == "repeat_missing":
            del repeat["success_failure_and_every_attempt_charged"]
        elif kind == "repeat_detached":
            repeat["role_request_identity"]["sha256"] = "a" * 64
        else:
            repeat["additional_attempts"] = 1
    else:
        role = body["role_request"]
        if kind == "array_claim":
            role["roles"][-1]["array_identity"] = {"byte_count": 8, "sha256": "b" * 64}
        elif kind == "late_role":
            role["roles"][-1]["expected_count"] = 40.0
        elif kind == "late_stream":
            role["streams"][-1]["value"] = float(role["streams"][-1]["value"])
        elif kind == "missing_role":
            role["roles"] = role["roles"][:-1]
        elif kind == "missing_source":
            role["sources"] = role["sources"][:-1]
        else:
            role["execution_requested"] = True
        reseal_role_request(body)
        body["independent_repeat_envelope"] = repeat_declaration(body)
    bundle.save()
    before = deepcopy(body)
    with pytest.raises(ValueError):
        _inspect(bundle)
    assert body == before


@pytest.mark.parametrize("part", ("request.json", "sources.json", "code.json", "code/part_4.py"))
def test_should_reject_late_same_size_physical_drift_in_every_bundle_part(
    bundle: BundleFixture, part: str
) -> None:
    path = bundle.root / part
    raw = path.read_bytes()
    path.write_bytes(bytes((raw[0] ^ 1,)) + raw[1:])
    with pytest.raises(ValueError):
        _inspect(bundle)


@pytest.mark.parametrize(
    "kind",
    (
        "missing",
        "extra",
        "reordered",
        "duplicate",
        "unknown_field",
        "foreign_path",
        "boolean_bytes",
    ),
)
def test_should_check_complete_closed_code_membership_even_after_manifest_resealing(
    bundle: BundleFixture, kind: str
) -> None:
    rows = bundle.manifest["files"]
    if kind == "missing":
        rows.pop()
    elif kind == "extra":
        rows.append(deepcopy(rows[-1]))
    elif kind == "reordered":
        rows[-2:] = reversed(rows[-2:])
    elif kind == "duplicate":
        rows[-1] = deepcopy(rows[-2])
    elif kind == "unknown_field":
        rows[-1]["runtime_verified"] = True
    elif kind == "foreign_path":
        rows[-1]["path"] = "code/unknown.py"
    else:
        rows[-1]["identity"]["byte_count"] = True
    bundle.save()
    with pytest.raises(ValueError):
        _inspect(bundle)


@pytest.mark.parametrize("suffix", (".claim", ".failure.json"))
def test_should_reject_pending_or_failed_publication_markers(
    bundle: BundleFixture, suffix: str
) -> None:
    (bundle.root / "sources.json").with_suffix(suffix).write_bytes(b"retained marker\n")
    with pytest.raises(ValueError):
        _inspect(bundle)


@pytest.mark.parametrize(
    "kind",
    (
        "relative_escape",
        "absolute",
        "backslash",
        "double_separator",
        "empty_membership",
        "mutable_membership",
        "duplicate_membership",
        "boolean_identity",
    ),
)
def test_should_reject_bad_trusted_specs_before_calling_the_io_port(
    bundle: BundleFixture, kind: str
) -> None:
    changes: dict[str, dict[str, Any]] = {
        "relative_escape": {"request_path": "../request.json"},
        "absolute": {"request_path": "C:/request.json"},
        "backslash": {"request_path": "directory\\request.json"},
        "double_separator": {"request_path": "directory//request.json"},
        "empty_membership": {"code_paths": ()},
        "mutable_membership": {"code_paths": list(bundle.spec.code_paths)},
        "duplicate_membership": {
            "code_paths": bundle.spec.code_paths[:-1] + (bundle.spec.code_paths[-2],)
        },
        "boolean_identity": {"request_identity": EvidenceIdentity(True, "a" * 64)},
    }

    class ForbiddenPort:
        def read_bundle(self, spec: ProspectiveBundleSpec) -> BundleSnapshot:
            raise AssertionError("invalid spec reached IO")

        def recheck_bundle(self, snapshot: BundleSnapshot) -> None:
            raise AssertionError("invalid spec reached IO recheck")

    spec = replace(bundle.spec, **changes[kind])
    with pytest.raises(ValueError):
        preflight_prospective_request_bundle(fixed_prospective_design(), spec, ForbiddenPort())


def test_should_recheck_code_files_after_app_inspection_and_refuse_drift(
    bundle: BundleFixture,
) -> None:
    reader = _reader(bundle)

    class LateDriftPort:
        def read_bundle(self, spec: ProspectiveBundleSpec) -> BundleSnapshot:
            return reader.read_bundle(spec)

        def recheck_bundle(self, snapshot: BundleSnapshot) -> None:
            path = bundle.root / bundle.spec.code_paths[-1]
            path.write_bytes(path.read_bytes().replace(b"pass", b"fail", 1))
            reader.recheck_bundle(snapshot)

    with pytest.raises(ValueError):
        preflight_prospective_request_bundle(
            fixed_prospective_design(), bundle.spec, LateDriftPort()
        )


@pytest.mark.parametrize("kind", ("whitespace", "duplicate_field", "nonfinite"))
def test_should_reject_pinned_noncanonical_or_ambiguous_json_bytes(
    bundle: BundleFixture, kind: str
) -> None:
    raw = (bundle.root / "request.json").read_bytes()
    if kind == "whitespace":
        raw = b" " + raw
    elif kind == "duplicate_field":
        raw = raw.replace(
            b'{"independent_repeat_envelope":',
            b'{"owner_id":"11111111111111111111111111111111","independent_repeat_envelope":',
            1,
        )
    else:
        raw = raw.replace(
            b'"maximum_optimizer_updates":16000', b'"maximum_optimizer_updates":NaN', 1
        )
    (bundle.root / "request.json").write_bytes(raw)
    bundle.spec = replace(bundle.spec, request_identity=raw_identity(raw))
    with pytest.raises(ValueError):
        _inspect(bundle)


@pytest.mark.parametrize(
    "kind",
    (
        "foreign_snapshot",
        "mutable_code_files",
        "foreign_code_row",
        "boolean_code_identity",
        "detached_spec",
    ),
)
def test_should_reject_foreign_port_snapshots_before_recheck(
    bundle: BundleFixture, kind: str
) -> None:
    reader = _reader(bundle)
    snapshot: Any = reader.read_bundle(bundle.spec)
    changed: Any = snapshot
    if kind == "foreign_snapshot":
        changed = {"spec": snapshot.spec}
    elif kind == "mutable_code_files":
        changed = replace(snapshot, code_files=list(snapshot.code_files))
    elif kind == "foreign_code_row":
        changed = replace(
            snapshot,
            code_files=snapshot.code_files[:-1] + ({"path": snapshot.code_files[-1].path},),
        )
    elif kind == "boolean_code_identity":
        changed = replace(
            snapshot,
            code_files=snapshot.code_files[:-1]
            + (replace(snapshot.code_files[-1], identity=EvidenceIdentity(True, "a" * 64)),),
        )
    else:
        changed = replace(
            snapshot, spec=replace(snapshot.spec, request_identity=EvidenceIdentity(123, "a" * 64))
        )

    class ForeignPort:
        def read_bundle(self, spec: ProspectiveBundleSpec) -> BundleSnapshot:
            return changed

        def recheck_bundle(self, snapshot: BundleSnapshot) -> None:
            raise AssertionError("foreign port snapshot reached recheck")

    with pytest.raises(ValueError):
        preflight_prospective_request_bundle(fixed_prospective_design(), bundle.spec, ForeignPort())


def test_should_refuse_two_code_names_for_one_physical_file(bundle: BundleFixture) -> None:
    original = bundle.root / bundle.spec.code_paths[0]
    alias = bundle.root / bundle.spec.code_paths[-1]
    alias.unlink()
    os.link(original, alias)
    with pytest.raises(ValueError):
        _inspect(bundle)
