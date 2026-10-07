"""Verify development evidence and inspect reserved confirmation scope locally."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from scripts import run_p63_combined_factor_development as combined
from scripts import run_p63_gating_pilot as gating
from scripts import run_p63_parent_factor_development as parent
from scripts import run_p63_replay_factor_pilot as replay
from scripts import run_p63_schedule_factor_development as schedule
from scripts import run_p63_sleep_factor_development as sleep
from scripts import run_p63_sleep_factor_preflight as artifacts
from src.app.continual_confirmation_manifest import (
    PROTOCOL_ID,
    ConfirmationFamily,
    fixed_confirmation_manifest,
    validate_confirmation_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = {
    "gating": (gating, "p63-gating-pilot"),
    "replay": (replay, "p63-replay-factor-pilot"),
    "sleep": (sleep, "p63-sleep-factor-development"),
    "schedule": (schedule, "p63-schedule-factor-development"),
    "combined": (combined, "p63-combined-factor-development"),
    "parent": (parent, "p63-parent-factor-development"),
}


def _verify_bundle(adapter: Any, directory: Path, family: ConfirmationFamily) -> dict[str, Any]:
    paths = adapter.artifact_paths(directory)
    if paths["failure"].exists() or any(
        not paths[name].is_file() for name in ("request", "result", "audit")
    ):
        raise ValueError(
            f"confirmation inventory requires complete development bundle: {directory}"
        )
    values = {
        name: artifacts.parse_finite_json(paths[name].read_text(encoding="utf-8"))
        for name in ("request", "result", "audit")
    }
    digests = {name: sha256(paths[name].read_bytes()).hexdigest() for name in values}
    if digests["result"] != family.development_result_sha256:
        raise ValueError(f"confirmation development result bytes differ: {family.name}")
    if family.name in ("gating", "replay"):
        adapter.verify_result(values["result"])
    else:
        adapter.verify_result(values["result"], adapter.read_reference())
    request, audit = values["request"], values["audit"]
    sources = adapter.check_source_hashes()
    if (
        request["manifest"] != json.loads(family.development_manifest_json)
        or request["manifest_sha256"] != family.development_manifest_sha256
        or request["source_sha256"] != sources
        or artifacts.digest_json(sources) != family.development_source_map_sha256
        or request["adapter_sha256"] != sha256(Path(adapter.__file__).read_bytes()).hexdigest()
        or audit["status"] != "completed"
        or audit["request_sha256"] != digests["request"]
        or audit["result_sha256"] != digests["result"]
        or not 0 <= audit["elapsed_seconds"] <= 120
    ):
        raise ValueError(f"confirmation development request/audit identities differ: {family.name}")
    if "process_rss" in audit:
        artifacts.verify_memory(audit["process_rss"], 256 * 1024 * 1024)
    return {
        "directory": str(directory.relative_to(REPO_ROOT)),
        "file_sha256": digests,
        "source_sha256": sources,
        "result_bytes": paths["result"].stat().st_size,
    }


def _result_seeds(payload: dict[str, Any]) -> set[int]:
    seeds: set[int] = set()
    for key in ("seed_results", "scored_seeds"):
        for row in payload.get(key, ()):
            if type(row.get("seed")) is not int:
                raise ValueError("confirmation inventory encountered a malformed seed row")
            seeds.add(row["seed"])
    if "train_facts" in payload:
        seeds.update(_result_seeds(payload["train_facts"]))
    return seeds


def _verify_unused_reservations(root: Path, reserved: set[int]) -> dict[str, Any]:
    files: list[str] = []
    observed: set[int] = set()
    for directory in sorted((root / "artifacts/runs").glob("p63-*")):
        for path in sorted(directory.glob("*.result.json")):
            payload = artifacts.parse_finite_json(path.read_text(encoding="utf-8"))
            seeds = _result_seeds(payload)
            if seeds & reserved:
                raise ValueError(
                    f"confirmation reservation already appears in a P6.3 result: {path}"
                )
            files.append(str(path.relative_to(root)))
            observed.update(seeds)
    return {"result_files_checked": files, "observed_source_seeds": sorted(observed)}


def inspect_confirmation_scope() -> dict[str, Any]:
    manifest = fixed_confirmation_manifest()
    summary = validate_confirmation_manifest(manifest)
    references = []
    for family in manifest.families:
        adapter, name = ADAPTERS[family.name]
        pair = tuple(
            _verify_bundle(adapter, REPO_ROOT / "artifacts/runs" / (name + suffix), family)
            for suffix in ("", "-repeat")
        )
        references.append({"family": family.name, "bundles": pair})
    reserved = {seed for family in manifest.families for seed in family.seeds}
    usage = _verify_unused_reservations(REPO_ROOT, reserved)
    return {
        "schema_id": "p67_confirmation_scope_inspection_v1",
        "protocol_id": PROTOCOL_ID,
        "manifest": asdict(manifest),
        "manifest_sha256": artifacts.digest_json(asdict(manifest)),
        "summary": summary,
        "development_references": references,
        "reservation_usage": usage,
        "inspection_source_sha256": {
            name: sha256((REPO_ROOT / name).read_bytes()).hexdigest()
            for name in (
                "src/app/continual_confirmation_manifest.py",
                "scripts/inspect_p67_confirmation_scope.py",
            )
        },
        "confirmation_source_constructed": False,
        "confirmation_scored": False,
        "final_released": False,
    }


def publish_scope_inspection(output_file: Path) -> dict[str, Any]:
    output_file = output_file.resolve()
    if output_file.exists():
        raise FileExistsError(f"confirmation scope output already exists: {output_file}")
    report = inspect_confirmation_scope()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    artifacts.write_exclusive(output_file, report)
    return {
        "path": str(output_file),
        "sha256": sha256(output_file.read_bytes()).hexdigest(),
        "summary": report["summary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-file", type=Path, default=Path("artifacts/runs/p67-confirmation-scope.json")
    )
    options = parser.parse_args()
    print(
        json.dumps(publish_scope_inspection(options.output_file), sort_keys=True, allow_nan=False)
    )


if __name__ == "__main__":
    main()
