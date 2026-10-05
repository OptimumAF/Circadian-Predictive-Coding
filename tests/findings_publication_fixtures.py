"""Small fabricated IO records; no ignored artifacts or scientific models."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from src.app.continual_confirmation_execution import digest_json
from src.app.continual_findings_readers import FindingsReaders
from src.infra import continual_findings_current_bindings as bindings


def write_body(path: Path, body: dict[str, Any]) -> dict[str, Any]:
    raw = (json.dumps(body, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {"byte_count": len(raw), "sha256": sha256(raw).hexdigest()}


def file_id(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"byte_count": len(raw), "sha256": sha256(raw).hexdigest()}


def record_call(events: list[str], name: str, result: Any) -> Any:
    events.append(name)
    return result


def make_repository(root: Path, monkeypatch: Any) -> dict[str, Any]:
    sources = {}
    for index in range(144):
        name = f"old/source-{index}.py"
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"original source {index}\n", encoding="utf-8")
        sources[name] = file_id(path)["sha256"]
    for name in (*bindings.OWN_SOURCES, *bindings.OWN_TEST_FILES):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("own file " + name + "\n", encoding="utf-8")
    scope = root / "scope.json"
    write_body(scope, {"scope": "fixed"})
    record = root / "saved.json"
    record_id = write_body(record, {"negative": -0.025, "unmeasured": None})
    pure_id = write_body(root / "pure.json", {"all_original_evidence": True})
    prior: dict[str, Any] = {
        "source_sha256": sources,
        "source_map_sha256": digest_json(sources),
        "test_files": {},
        "input_files": {"saved.json": record_id},
        "current_output_files": {"pure.json": pure_id},
    }
    prior_id = write_body(root / "prior.json", prior)
    legacy_claim = root / "artifacts/runs/p610-outcome-costs/outcome-costs.claim"
    legacy_claim.parent.mkdir(parents=True, exist_ok=True)
    legacy_claim.write_text("preserved partial publication\n", encoding="utf-8")
    (root / "artifacts/runs/p612-complete-findings-pure").mkdir(parents=True)
    bundles, returned = [], {}
    for kind, prefix in (("outcome_costs", "outcome-costs"), ("matrix", "confirmation-matrix")):
        for suffix in ("", "-repeat"):
            directory = kind + suffix
            parts: dict[str, dict[str, Any]] = {
                "request": {"kind": kind, "scope": "fixed"},
                "result": {"kind": kind, "negative": -0.025, "null": None},
                "audit": {"status": "completed", "kind": kind},
            }
            entries = {}
            for part, body in parts.items():
                name = f"{directory}/{prefix}.{part}.json"
                entries[part] = {"path": name, "identity": write_body(root / name, body)}
            name = f"{directory}/{prefix}.md"
            (root / name).write_text("whole table\n", encoding="utf-8")
            entries["markdown"] = {"path": name, "identity": file_id(root / name)}
            bundles.append(
                {"kind": kind, "directory": directory, "prefix": prefix, "parts": entries}
            )
            returned[kind] = tuple(parts[name] for name in ("request", "result", "audit"))
    catalog: dict[str, Any] = {
        "prior_b2_handoff": {"path": "prior.json", "identity": prior_id},
        "budget_seconds": bindings.BUDGETS.copy(),
        "development_input_files": {},
        "upstream_handoffs": {},
        "current_bundles": bundles,
    }
    catalog_id = write_body(root / bindings.CATALOG_FILE, catalog)
    monkeypatch.setattr(bindings, "CATALOG_ID", catalog_id)
    monkeypatch.setattr(
        bindings,
        "CURRENT_BUNDLE_CHECKERS",
        {
            kind: (builder, lambda _root, paths, _scope: json.loads(paths["request"].read_bytes()))
            for kind, (builder, _) in bindings.CURRENT_BUNDLE_CHECKERS.items()
        },
    )
    return {"catalog": catalog, "prior": prior, "scope": scope, "returned": returned}


def make_artifact_boundary(
    root: Path, monkeypatch: Any
) -> tuple[dict[str, Any], FindingsReaders, list[str]]:
    from src.infra import continual_findings_artifacts as artifacts

    body: dict[str, Any] = {
        "coverage": {"all": 1},
        "negative": -0.025,
        "unmeasured": None,
        "hypothesis_status": "unresolved",
    }
    text = "complete -0.025 None unresolved\n"
    events: list[str] = []
    state: dict[str, Any] = {"source": "original", "input": "whole", "environment": "fixed"}

    def request(_root, directory, _scope, started):
        return {
            "protocol_id": "fixed",
            "source_map_sha256": state["source"],
            "bindings": state.copy(),
            "started_utc": started,
            "output_dir": str(directory),
        }

    def checked(_root, paths, _scope):
        from src.app.continual_confirmation_json import same_json

        stored = json.loads(paths["request"].read_bytes())
        same_json(
            stored,
            request(_root, paths["request"].parent, _scope, stored["started_utc"]),
            "complete test current bindings",
        )
        return stored

    def rebuild(_root, paths, _scope, readers, _request):
        readers.outcome_costs()
        readers.matrix()
        readers.development()
        evidence = {
            "current_complete_reader_ports": {
                "outcome_costs": {"returned_complete": True},
                "matrix": {"returned_complete": True},
                "development": {"returned_complete": True},
            },
            "pure_derivation_elapsed_seconds": 0.0,
        }
        return body.copy(), text, evidence

    monkeypatch.setattr(artifacts, "findings_request", request)
    monkeypatch.setattr(artifacts, "checked_findings_request", checked)
    monkeypatch.setattr(artifacts, "rebuild_current_findings", rebuild)
    readers = FindingsReaders(
        lambda: record_call(events, "outcome_costs", ({}, {}, {})),
        lambda: record_call(events, "matrix", ({}, {}, {})),
        lambda: record_call(events, "development", {}),
    )
    return {"body": body, "text": text, "state": state}, readers, events
