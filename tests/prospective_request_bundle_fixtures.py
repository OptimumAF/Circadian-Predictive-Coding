"""Complete fabricated request metadata backed by real temporary files; no science."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from src.app.prospective_confirmation_design import (
    fixed_prospective_design,
    prospective_design_identity,
)
from src.app.prospective_role_requests import declare_prospective_sources
from src.core.prospective_replications import ReplicaSeedBinding, ReplicaSlot
from src.core.prospective_request_bundles import ProspectiveBundleSpec
from src.core.prospective_role_requests import ProspectiveRoleDeclaration, ProspectiveRoleRequest
from src.core.seed_stream_screening import EvidenceIdentity, confirmation_seed_streams


def canonical(body: Any) -> bytes:
    return (
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def raw_identity(raw: bytes) -> EvidenceIdentity:
    return EvidenceIdentity(len(raw), sha256(raw).hexdigest())


def reseal_role_request(body: dict[str, Any]) -> None:
    role = body["role_request"]
    role["source_map_identity"] = asdict(raw_identity(canonical(role["sources"])))
    unsigned = {name: value for name, value in role.items() if name != "request_identity"}
    role["request_identity"] = asdict(raw_identity(canonical(unsigned)))


def repeat_declaration(body: dict[str, Any]) -> dict[str, Any]:
    role = body["role_request"]
    return {
        "required": True,
        "design_identity": role["design_identity"],
        "source_map_identity": role["source_map_identity"],
        "role_request_identity": role["request_identity"],
        "caps_identical": True,
        "success_failure_and_every_attempt_charged": True,
        "actual_independent_repeat_verified": False,
    }


@dataclass
class BundleFixture:
    root: Path
    spec: ProspectiveBundleSpec
    body: dict[str, Any]
    manifest: dict[str, Any]

    def save(self) -> None:
        request = canonical(self.body)
        source_map = canonical(self.body["role_request"]["sources"])
        manifest = canonical(self.manifest)
        for name, raw in (
            ("request.json", request),
            ("sources.json", source_map),
            ("code.json", manifest),
        ):
            (self.root / name).write_bytes(raw)
        self.spec = ProspectiveBundleSpec(
            "request.json",
            raw_identity(request),
            "sources.json",
            raw_identity(source_map),
            "code.json",
            raw_identity(manifest),
            self.spec.code_paths,
        )


def make_bundle(root: Path) -> BundleFixture:
    design = fixed_prospective_design()
    names = tuple(f"code/part_{index}.py" for index in range(5))
    files = []
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"pass\n# fabricated code membership; no runtime or dataset proof\n")
        files.append({"path": name, "identity": asdict(raw_identity(path.read_bytes()))})
    manifest = {"schema_id": "p67_closed_prospective_code_files_v1", "files": files}
    code_identity = raw_identity(canonical(manifest))
    slots = tuple(ReplicaSlot(**row) for row in design["replica_slots"])
    groups = tuple(dict.fromkeys(row.source_group for row in slots))
    bindings = tuple(
        ReplicaSeedBinding(slot, 1_000_000 + 20_000 * groups.index(slot.source_group))
        for slot in slots
    )
    seeds = {row.slot: row.base_seed for row in bindings}
    streams = tuple(
        stream
        for index in range(len(groups))
        for stream in confirmation_seed_streams(1_000_000 + 20_000 * index)
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
    request = ProspectiveRoleRequest(
        EvidenceIdentity(**prospective_design_identity(design)),
        bindings,
        streams,
        sources,
        tuple(roles),
        EvidenceIdentity(0, "0" * 64),
        EvidenceIdentity(0, "0" * 64),
        tuple(design["required_actual_request_bindings"]),
    )
    body = {
        "schema_id": "p67_complete_prospective_request_bundle_v1",
        "role_request": asdict(request),
        "prospective_utc": "2026-10-05T12:00:00+00:00",
        "owner_id": "1" * 32,
        "resource_envelope": design["resource_envelope"],
        "independent_repeat_envelope": {},
    }
    reseal_role_request(body)
    body["independent_repeat_envelope"] = repeat_declaration(body)
    spec = ProspectiveBundleSpec(
        "request.json",
        EvidenceIdentity(0, "0" * 64),
        "sources.json",
        EvidenceIdentity(0, "0" * 64),
        "code.json",
        code_identity,
        names,
    )
    bundle = BundleFixture(root, spec, body, manifest)
    bundle.save()
    return bundle
