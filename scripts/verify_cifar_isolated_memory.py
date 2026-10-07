"""Run one declared real-CIFAR isolated-memory boundary without final test."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.app.isolated_head_memory import (  # noqa: E402
    PROCESS_ISOLATED_CIFAR_MEMORY_PROTOCOL,
    run_process_isolated_fixed_width_memory,
)
from src.app.repeated_head_confirmation import restore_confirmation_manifest  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "artifacts" / "benchmark_cifar_v2_manifest_smoke.json"
OUTPUT_PATH = REPO_ROOT / "data" / "cifar-memory-seed83.json"


def main() -> None:
    manifest = restore_confirmation_manifest(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")))
    if manifest.confirmation_seeds != (83, 89, 97):
        raise ValueError("Unexpected CIFAR confirmation seed manifest")
    if any(item.config != manifest.base_config for item in manifest.selected_heads):
        raise ValueError("This one-seed probe requires the selected base settings")
    report = run_process_isolated_fixed_width_memory(
        replace(manifest.base_config, seed=manifest.confirmation_seeds[0]),
        timeout_seconds=60.0,
    )
    if report.protocol_id != PROCESS_ISOLATED_CIFAR_MEMORY_PROTOCOL:
        raise AssertionError("CIFAR memory used an unexpected protocol")
    with OUTPUT_PATH.open("x", encoding="utf-8") as stream:
        json.dump(asdict(report), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "manifest_digest": manifest.manifest_digest,
                "protocol_id": report.protocol_id,
                "seed": report.config.seed,
                "child_pids": {name: row.pid for name, row in report.reports.items()},
                "output": str(OUTPUT_PATH),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
