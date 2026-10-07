"""Full invented V2 generation declarations backed by real temporary whole files."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from prospective_generation_fixtures import generation_inputs
from prospective_request_bundle_fixtures import canonical, raw_identity
from src.app.prospective_generation_requests import freeze_prospective_generation_request
from src.core.prospective_request_bundles import ProspectiveBundleSpec


def repeat_declaration(body: dict[str, Any]) -> dict[str, Any]:
    request = body["generation_request"]
    return {
        "required": True,
        "design_identity": dict(request["design_identity"]),
        "source_map_identity": dict(request["source_map_identity"]),
        "generation_request_identity": dict(request["request_identity"]),
        "caps_identical": True,
        "success_failure_and_every_attempt_charged": True,
        "actual_independent_repeat_verified": False,
    }


def reseal_generation_request(body: dict[str, Any]) -> None:
    request = body["generation_request"]
    request["source_map_identity"] = asdict(raw_identity(canonical(request["sources"])))
    unsigned = {name: value for name, value in request.items() if name != "request_identity"}
    request["request_identity"] = asdict(raw_identity(canonical(unsigned)))
    body["independent_repeat_envelope"] = repeat_declaration(body)


@dataclass
class GenerationBundleFixture:
    root: Path
    spec: ProspectiveBundleSpec
    body: dict[str, Any]
    manifest: dict[str, Any]

    def save(self) -> None:
        request = canonical(self.body)
        source_map = canonical(self.body["generation_request"]["sources"])
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


def make_generation_bundle(root: Path) -> GenerationBundleFixture:
    design, _, bindings, streams = generation_inputs()
    names = tuple(f"code/part_{index}.py" for index in range(5))
    files = []
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"pass\n# invented code descriptor; never import or execute\n")
        files.append({"path": name, "identity": asdict(raw_identity(path.read_bytes()))})
    manifest = {"schema_id": "p67_closed_prospective_code_files_v1", "files": files}
    request = freeze_prospective_generation_request(
        design, bindings, raw_identity(canonical(manifest)), streams
    )
    body = {
        "schema_id": "p67_complete_prospective_generation_bundle_v2",
        "generation_request": json.loads(canonical(asdict(request))),
        "prospective_utc": "2026-10-05T12:00:00+00:00",
        "owner_id": "1" * 32,
        "resource_envelope": design["resource_envelope"],
        "independent_repeat_envelope": {},
    }
    body["independent_repeat_envelope"] = repeat_declaration(body)
    bundle = GenerationBundleFixture(
        root,
        ProspectiveBundleSpec(
            "request.json",
            request.request_identity,
            "sources.json",
            request.source_map_identity,
            "code.json",
            raw_identity(canonical(manifest)),
            names,
        ),
        body,
        manifest,
    )
    bundle.save()
    return bundle
