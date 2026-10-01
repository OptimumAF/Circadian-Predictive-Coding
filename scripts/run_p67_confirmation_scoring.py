"""Execute or independently read the complete fixed scored confirmation bundle.

Correctness gates must pass before --execute opens any reserved final value.
The fixed source-bound request has no scientific/seed/metric/budget overrides.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from scripts import run_p67_confirmation_training as training_adapter
from src.app.continual_confirmation_scoring_execution import FAILURE_SCHEMA
from src.app.continual_confirmation_scoring_manifest import fixed_scoring_manifest
from src.infra.continual_confirmation_scoring_artifacts import (
    read_completed_scored_bundle,
    run_bounded_scoring,
)
from src.infra.continual_confirmation_scoring_worker import worker_parts
from src.infra.continual_confirmation_training_references import read_training_references


REPO_ROOT = Path(__file__).resolve().parents[1]
SCOPE_FILE = REPO_ROOT / "artifacts/runs/p67-confirmation-scope.json"


def _references(scope_file: Path) -> dict[str, Any]:
    return read_training_references(
        REPO_ROOT,
        fixed_scoring_manifest(),
        lambda directory: training_adapter.read_completed_bundle(directory, scope_file),
    )


def _worker(request_file: Path, scope_file: Path) -> None:
    try:
        result_json, metadata = worker_parts(REPO_ROOT, request_file, scope_file)
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
    # Scientific serialization is already measured with held models/views;
    # stdout framing follows the original predeclared worker contract.
    sys.stdout.write(json.dumps(metadata, sort_keys=True, allow_nan=False)[:-1])
    sys.stdout.write(',"result":')
    sys.stdout.write(result_json)
    sys.stdout.write("}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/runs/p67-confirmation-scored")
    )
    parser.add_argument("--scope-file", type=Path, default=SCOPE_FILE)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute", action="store_true", help="execute every fixed independent final cell"
    )
    mode.add_argument(
        "--read-only", action="store_true", help="independently verify a complete bundle"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--request-file", type=Path, help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.worker:
        if options.request_file is None or options.execute or options.read_only:
            parser.error("private worker requires only its saved --request-file/--scope-file")
        _worker(options.request_file, options.scope_file)
        return
    if options.request_file is not None:
        parser.error("--request-file belongs to the private worker boundary")
    if not options.execute and not options.read_only:
        parser.error("select --execute or --read-only")
    reader = lambda: _references(options.scope_file)
    if options.read_only:
        _, _, audit = read_completed_scored_bundle(
            REPO_ROOT, options.output_dir, options.scope_file, reader
        )
    else:
        audit = run_bounded_scoring(REPO_ROOT, options.output_dir, options.scope_file, reader)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "output_dir": str(options.output_dir.resolve()),
                "request_sha256": audit["request_sha256"],
                "result_sha256": audit["result_sha256"],
                "work_totals": audit["work"]["totals"],
                "final_prediction_calls": audit["final_observation"]["prediction_attempts"],
                "final_prediction_examples": audit["final_observation"]["prediction_examples"],
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
