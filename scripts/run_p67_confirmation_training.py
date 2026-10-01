"""Bind and run the frozen joint unscored confirmation worker locally.

The scope and every current source/request identity must match before data.
The resource/artifact worker is a separate acceptance gate from this binding.
No outer/final score or scientific setting selection is permitted here.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path
import platform
import subprocess
import sys
from time import monotonic
from typing import Any

import numpy as np

from scripts import inspect_p67_confirmation_scope as inventory
from src.app.continual_confirmation_execution import (
    AUDIT_SCHEMA,
    FAILURE_SCHEMA,
    execution_request,
    json_value,
    verify_execution_request,
    verify_observed_updates,
    verify_process_memory,
    work_summary,
)
from src.app.continual_confirmation_json import object_fields, require, same_json
from src.app.continual_confirmation_manifest import (
    fixed_confirmation_manifest,
    validate_confirmation_manifest,
)
from src.app.continual_confirmation_state import require_held_seed
from src.app.continual_confirmation_training import PROTOCOL_ID, train_confirmation
from src.app.continual_confirmation_work_validation import SeedWork, verify_confirmation_payload
from src.infra.continual_confirmation_io import (
    claim_artifacts,
    encoded_digest,
    file_digest,
    parse_json,
    read_json,
    verify_source_files,
    write_exclusive,
)
from src.infra.continual_confirmation_runtime import ExecutionObserver
from src.shared.process_memory import ProcessRssSampler


REPO_ROOT = Path(__file__).resolve().parents[1]
SCOPE_FILE = REPO_ROOT / "artifacts/runs/p67-confirmation-scope.json"
SCOPE_FILE_SHA256 = "622feead54f155521928341151c8496c23c02a54b355b1b3b1e0f68b76c5772f"
# Why this: the original selected maps omit transitive helpers/packages. Bind
# their inspected closure too; this adapter's own bytes are bound separately
# in the prelaunch request to avoid a self-referential hash constant.
EXTRA_SOURCE_SHA256 = {
    "scripts/__init__.py": "ed3888e5ac0bad9b11c74c5a8055c0292ad190c6401361bd3f1bfc700ee0c8ab",
    "scripts/run_p63_combined_factor_development.py": "edf68589d83033d0abdf39c737f78044938f84a97b49d86a25698edce42d8c40",
    "scripts/run_p63_gating_pilot.py": "919347ba9a9a38d88a83c97b5096898e13f9807f53225a2834b9927cba967c07",
    "scripts/run_p63_parent_factor_development.py": "be57cf809149af570876d75d6f924ec073ee1a35a1d96b68ae4fc7b59129e9c0",
    "scripts/run_p63_replay_factor_pilot.py": "9255ad6615b3c41953fc5f76ad4c5597389724abf57ad996f2c1945c83b8826a",
    "scripts/run_p63_schedule_factor_development.py": "94fc133317c9b0954cee47b5d5a431285bcf30da3effc9b457153b6bfe6000c1",
    "scripts/run_p63_sleep_factor_development.py": "08dedd66328f41ab24403e29ca0f8bc8c4b10030528f9e4a2c745b4f0d330634",
    "src/__init__.py": "799506c6c6c8bb02ffb41f82b11e784c9b62a575d714519ae9a319494c05f3df",
    "src/app/__init__.py": "65855836756b6c2d6669b454decefedb25833ca2b406a58fda7ca0ea34254ff5",
    "src/app/circadian_checkpoint.py": "ee3c34833108007fa652f2d728b02625c168fa7b278dc939c19231628ae9e53c",
    "src/app/comparison_scope.py": "7738c5c9931d9fb835055fa2f0e94398e58ca88fb3d586164b2d45e22e6679ef",
    "src/app/continual_arrived_checkpoint.py": "2a6502ebdb2ad7a8b8f8e29db0ce73f6bc56faaabb8391d0faf33a706affd640",
    "src/app/continual_arrived_sleep_history.py": "c47b8a31eb14056656668c1ea82b30c1d8162f6e3b664ab363fd025e10a1d6e8",
    "src/app/continual_arrived_transactions.py": "cb11cfb05ca7bfe4e073b110c3a7654a3432192f63317bf14559988fab3cb6c3",
    "src/app/continual_checkpoint.py": "4884e67839372044c49fefb4db32e427d17cea4c15e91c64c3c0f9691d9b23d6",
    "src/app/continual_confirmation_checkpoints.py": "4b00b454d5b317d2190ca942d36e93e09446b4f043956fd74213abc5d27f73ba",
    "src/app/continual_confirmation_execution.py": "dc45ad04c9381bee5cbee0ab7e3d70f7da6676304d056a18668be23affc16006",
    "src/app/continual_confirmation_fact_schema.py": "01f8437eb53eda763edb73c8695c6bd18dbbfa8cb6d8fef0fb7f874e3dc47862",
    "src/app/continual_confirmation_json.py": "307642a05dd2b73ef3af4a3714c33f85926e205c25d71d85c9dc796d6d9ff203",
    "src/app/continual_confirmation_parameter_links.py": "4c1fb55e5d414ed7fbb99392ede60d05eafd8c3d025e38e76cfa137f62130262",
    "src/app/continual_confirmation_periodic.py": "a22f23ef2b3c86682b4e68f65a4c211a664b407de00f21ecd2a2039080de9b9e",
    "src/app/continual_confirmation_simple.py": "f9f46848310e02a10a8a47caa414c21b6d36812ede3684f803282f1b3eec1073",
    "src/app/continual_confirmation_simple_work.py": "f440aa80935f49e76cb2d9876f0beb476b0df18f6ecea4e8dd56ce408065de64",
    "src/app/continual_confirmation_state.py": "2808f11b22aa78894fd322a686e423cd5052d0fea0f7508209f6f89181394cd1",
    "src/app/continual_confirmation_training.py": "372aeeba9b8661918c8137c3d4c5a4f34d26666b76cc84644c99362808c103e1",
    "src/app/continual_confirmation_validation.py": "2d220d74e26533c33b14f1a5267923abe8d6691f9478580fb4e5f67f64272733",
    "src/app/continual_confirmation_work_validation.py": "cf26b0e73c5230f033c41a25cc436e4e6faa59b76a2375f89d08d6836ec4113f",
    "src/app/continual_matched_replay_schedule.py": "aa3f14f447704a19d6de7997b1aa6daaf13f7637b2fda81cceb425a6bf4229e7",
    "src/app/numpy_checkpoint_validation.py": "885b6ad87dedc7999f9bda2b0ed7884ab0265f7c4b644bfbff60936668409f9e",
    "src/core/__init__.py": "979801ce2dd7643e3d41b160fdb72e6fd2e4c199cea580b0a3e1ea985ba9c4b5",
    "src/core/activations.py": "9fdd713deff66709d75ff41a8d6844faead78fb8e9d7bcb89d39e94935f9c297",
    "src/core/dimension_validation.py": "33fc41ae58af6bb3d000c97b6f0acd805e74306f751bedbc7b39daddb6e8e83d",
    "src/core/resnet50_variants.py": "23a4ee75f938e2cb2152458ab6586a894cc22a1e35b19693b37b83e5f011a605",
    "src/core/training_validation.py": "e49db0ed6c66eb40803a53254e6d9e2890589fe0756b749879d4c9c132429e1c",
    "src/infra/__init__.py": "7c6f6cf33e9fbf71f6df77af377b5df00ab2893be05278c308f7e9da01be6ffc",
    "src/infra/continual_confirmation_io.py": "c77b6fdc58f42c924384e28acce53e9872b279213b77179293f647c4b5b86c82",
    "src/infra/continual_confirmation_runtime.py": "faf8ea73611c179eb4238138c831d0db94c8959130c37e4b8cde6efa87ba8078",
    "src/shared/__init__.py": "f728c7cf4d519be569354d482d94d84bc2257b319d72b51625ec7365c7fc36c4",
    "src/shared/torch_runtime.py": "698074d56fdbc09e234691e780ae2ce1111c6314af9b5acb6949c3a0430455e8",
}


def read_scope_reference(path: Path) -> dict[str, Any]:
    """Revalidate pinned bytes and complete evidence without new training."""
    if not path.is_file() or file_digest(path) != SCOPE_FILE_SHA256:
        raise ValueError("confirmation saved scope bytes changed or missing")
    saved = read_json(path)
    same_json(saved, json_value(inventory.inspect_confirmation_scope()), "saved scope revalidation")
    return saved


def check_source_hashes(scope: dict[str, Any]) -> dict[str, str]:
    expected = dict(scope["inspection_source_sha256"])
    for reference in scope["development_references"]:
        for bundle in reference["bundles"]:
            for name, digest in bundle["source_sha256"].items():
                if name in expected and expected[name] != digest:
                    raise ValueError(f"confirmation source pins disagree: {name}")
                expected[name] = digest
    require(bool(EXTRA_SOURCE_SHA256), "execution source pins are not frozen")
    for name, digest in EXTRA_SOURCE_SHA256.items():
        if name in expected and expected[name] != digest:
            raise ValueError(f"confirmation source pins disagree: {name}")
        expected[name] = digest
    return verify_source_files(REPO_ROOT, expected)


def worker_command(request_file: Path, scope_file: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "scripts.run_p67_confirmation_training",
        "--worker",
        "--request-file",
        str(request_file.resolve()),
        "--scope-file",
        str(scope_file.resolve()),
    ]


def execution_bindings(request_file: Path, scope_file: Path = SCOPE_FILE) -> dict[str, Any]:
    """Resolve identities before request publication or any reserved source."""
    manifest = fixed_confirmation_manifest()
    validate_confirmation_manifest(manifest)
    scope = read_scope_reference(scope_file)
    return {
        "manifest": manifest,
        "scope_sha256": SCOPE_FILE_SHA256,
        "source_sha256": check_source_hashes(scope),
        "adapter_sha256": file_digest(Path(__file__)),
        "command": worker_command(request_file, scope_file),
        "environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
    }


def artifact_paths(output_dir: Path) -> dict[str, Path]:
    paths = {
        name: output_dir / f"confirmation-train.{name}.json"
        for name in ("request", "result", "audit", "failure")
    }
    paths["claim"] = output_dir / "confirmation-train.claim"
    return paths


def _checked_request(
    path: Path, scope_file: Path, expected_digest: str | None = None
) -> dict[str, Any]:
    if expected_digest is not None and file_digest(path) != expected_digest:
        raise ValueError("confirmation request bytes changed during execution")
    bindings = execution_bindings(path, scope_file)
    request = read_json(path)
    verify_execution_request(request, **bindings)
    return request


def _elapsed(value: Any, name: str, maximum: float | None = None) -> float:
    require(
        type(value) in (int, float) and isfinite(value) and value >= 0, f"{name} elapsed differs"
    )
    if maximum is not None:
        require(value < maximum, f"{name} wall limit exceeded")
    return float(value)


def _verify_worker(payload: dict[str, Any], request_sha256: str) -> tuple[SeedWork, ...]:
    object_fields(
        payload,
        {"result", "request_sha256", "observed_updates", "process_rss", "worker_elapsed_seconds"},
        "worker envelope",
    )
    same_json(payload["request_sha256"], request_sha256, "worker request identity")
    manifest = fixed_confirmation_manifest()
    work = verify_confirmation_payload(payload["result"], manifest)
    verify_observed_updates(payload["observed_updates"], work, payload["result"])
    verify_process_memory(payload["process_rss"], manifest)
    _elapsed(payload["worker_elapsed_seconds"], "worker", manifest.wall_limit_seconds)
    return work


def _audit(
    request: dict[str, Any],
    request_sha256: str,
    result_sha256: str,
    payload: dict[str, Any],
    work: tuple[SeedWork, ...],
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_id": AUDIT_SCHEMA,
        "status": "completed",
        "protocol_id": PROTOCOL_ID,
        "manifest_sha256": request["manifest_sha256"],
        "scope_record_sha256": request["scope_record_sha256"],
        "source_sha256": request["source_sha256"],
        "adapter_sha256": request["adapter_sha256"],
        "request_sha256": request_sha256,
        "result_sha256": result_sha256,
        "work": work_summary(work),
        "observed_updates": payload["observed_updates"],
        "process_rss": payload["process_rss"],
        "worker_elapsed_seconds": payload["worker_elapsed_seconds"],
        "elapsed_seconds": elapsed_seconds,
    }


def _record_failure(path: Path, request_file: Path, exc: BaseException, started: float) -> None:
    write_exclusive(
        path,
        {
            "schema_id": FAILURE_SCHEMA,
            "status": "failed",
            "reason": "wall_limit"
            if isinstance(exc, subprocess.TimeoutExpired)
            else "canceled"
            if isinstance(exc, (KeyboardInterrupt, SystemExit))
            else "worker_or_audit",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "request_sha256": file_digest(request_file) if request_file.is_file() else None,
            "elapsed_seconds": monotonic() - started,
        },
    )


def run_bounded_training(output_dir: Path, scope_file: Path = SCOPE_FILE) -> dict[str, Any]:
    """Claim new local artifacts and publish only a fully verified child result."""
    paths = artifact_paths(output_dir.resolve())
    occupied = [str(path) for path in paths.values() if path.exists()]
    if occupied:
        raise FileExistsError(f"confirmation output already exists: {occupied}")
    bindings = execution_bindings(paths["request"], scope_file)
    request = execution_request(started_utc=datetime.now(timezone.utc).isoformat(), **bindings)
    paths["request"].parent.mkdir(parents=True, exist_ok=True)
    with claim_artifacts(paths):
        return _run_claimed(paths, scope_file, request)


def _run_claimed(
    paths: dict[str, Path], scope_file: Path, request: dict[str, Any]
) -> dict[str, Any]:
    started = monotonic()
    request_published = False
    try:
        request_sha256 = encoded_digest(request)
        write_exclusive(paths["request"], request)
        request_published = True
        require(file_digest(paths["request"]) == request_sha256, "published request bytes differ")
        process = subprocess.run(
            request["command"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=request["limits"]["wall_limit_seconds"],
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(f"confirmation worker exited {process.returncode}: {process.stderr}")
        payload = parse_json(process.stdout)
        work = _verify_worker(payload, request_sha256)
        _checked_request(paths["request"], scope_file, request_sha256)
        result_sha256 = encoded_digest(payload["result"])
        write_exclusive(paths["result"], payload["result"])
        require(file_digest(paths["result"]) == result_sha256, "published result bytes differ")
        audit = _audit(
            request,
            request_sha256,
            result_sha256,
            payload,
            work,
            monotonic() - started,
        )
        write_exclusive(paths["audit"], audit)
    except BaseException as exc:
        if not request_published and isinstance(exc, FileExistsError):
            # A writer bypassing the cooperative claim still owns its bytes.
            # Do not add our failure marker to that writer's request.
            raise
        _record_failure(paths["failure"], paths["request"], exc, started)
        raise
    return audit


def read_completed_bundle(
    output_dir: Path, scope_file: Path = SCOPE_FILE
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Read back every identity, body, derived cost and resource fact."""
    paths = artifact_paths(output_dir.resolve())
    if (
        paths["failure"].exists()
        or paths["claim"].exists()
        or any(not paths[name].is_file() for name in ("request", "result", "audit"))
    ):
        raise ValueError("confirmation requires a complete successful bundle")
    request = _checked_request(paths["request"], scope_file)
    result, audit = read_json(paths["result"]), read_json(paths["audit"])
    elapsed = _elapsed(audit.get("elapsed_seconds"), "parent")
    payload = {
        "result": result,
        "request_sha256": file_digest(paths["request"]),
        "observed_updates": audit.get("observed_updates"),
        "process_rss": audit.get("process_rss"),
        "worker_elapsed_seconds": audit.get("worker_elapsed_seconds"),
    }
    work = _verify_worker(payload, file_digest(paths["request"]))
    expected = _audit(
        request, file_digest(paths["request"]), file_digest(paths["result"]), payload, work, elapsed
    )
    same_json(audit, expected, "complete audit")
    return request, result, audit


def _require_live(trained: Any, observer: ExecutionObserver) -> None:
    for item in trained.held:
        require_held_seed(item)
        observer.checkpoint()


def _worker_parts(request_file: Path, scope_file: Path) -> tuple[str, dict[str, Any]]:
    manifest = fixed_confirmation_manifest()
    with ProcessRssSampler(interval_seconds=manifest.rss_interval_seconds) as sampler:
        observer = ExecutionObserver(manifest, sampler)
        request = _checked_request(request_file, scope_file)
        request_sha256 = file_digest(request_file)
        with observer.observe_updates():
            trained = train_confirmation(manifest)
            _require_live(trained, observer)
            payload = parse_json(json.dumps(asdict(trained.facts), sort_keys=True, allow_nan=False))
            work = verify_confirmation_payload(payload, manifest)
            verify_observed_updates(observer.updates(), work, payload)
            observer.checkpoint()
            _require_live(trained, observer)
            _checked_request(request_file, scope_file, request_sha256)
            # Why this: sample allocations for the complete result serialization
            # while live models/copies and independent decoded facts still exist.
            result_json = json.dumps(payload, sort_keys=True, allow_nan=False)
            _require_live(trained, observer)
            observer.checkpoint()
    observer.checkpoint()
    memory = asdict(sampler.snapshot())
    verify_process_memory(memory, manifest)
    metadata = {
        "request_sha256": request_sha256,
        "observed_updates": observer.updates(),
        "process_rss": memory,
        "worker_elapsed_seconds": observer.elapsed(),
    }
    _elapsed(metadata["worker_elapsed_seconds"], "worker", request["limits"]["wall_limit_seconds"])
    return result_json, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/runs/p67-confirmation-train")
    )
    parser.add_argument("--scope-file", type=Path, default=SCOPE_FILE)
    parser.add_argument(
        "--read-only", action="store_true", help="verify an existing complete bundle"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--request-file", type=Path, help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.worker:
        try:
            if options.request_file is None:
                raise ValueError("confirmation worker requires its saved request")
            result_json, metadata = _worker_parts(options.request_file, options.scope_file)
        except Exception as exc:
            sys.stderr.write(
                json.dumps(
                    {
                        "schema_id": FAILURE_SCHEMA,
                        "reason": getattr(exc, "reason", "worker_or_audit"),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
            raise SystemExit(1) from exc
        # Framing/stdout is outside sampling, as prospectively declared; the
        # complete scientific result serialization above is inside the gate.
        sys.stdout.write(json.dumps(metadata, sort_keys=True, allow_nan=False)[:-1])
        sys.stdout.write(',"result":')
        sys.stdout.write(result_json)
        sys.stdout.write("}\n")
        return
    if options.request_file is not None:
        parser.error("--request-file belongs to the private worker boundary")
    if options.read_only:
        _, _, audit = read_completed_bundle(options.output_dir, options.scope_file)
    else:
        audit = run_bounded_training(options.output_dir, options.scope_file)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "output_dir": str(options.output_dir.resolve()),
                "request_sha256": audit["request_sha256"],
                "result_sha256": audit["result_sha256"],
                "work_totals": audit["work"]["totals"],
                "maximum_transient_width": audit["work"]["maximum_transient_width"],
                "rss_peak_bytes": audit["process_rss"]["peak_bytes"],
                "worker_elapsed_seconds": audit["worker_elapsed_seconds"],
                "elapsed_seconds": audit["elapsed_seconds"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
