"""Describe or run a frozen original or separate legacy-only scaling pilot."""

import argparse
import json
import os
import sys
from contextlib import ExitStack
from dataclasses import asdict
from time import monotonic

COMMAND_STARTED = monotonic()

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from src.adapters.nonlinear_scaling_pilot import (
    RESOURCES,
    describe_protocol,
    json_default,
    run_pilot,
    unstarted_pilot_report,
)
from src.shared.process_memory import ProcessRssSampler


def _encode(report: dict) -> str:
    encoded = json.dumps(report, default=json_default, allow_nan=False, indent=2)
    if _output_bytes(encoded) > RESOURCES.storage_bytes:
        raise RuntimeError("pilot report exceeds frozen output allowance")
    return encoded


def _output_bytes(encoded: str) -> int:
    # Why: Windows text stdout expands every LF to CRLF, including JSON indentation.
    return len((encoded + "\n").replace("\n", os.linesep).encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", help="Protocol only, no model work")
    parser.add_argument(
        "--legacy-only", action="store_true", help="Separate frozen legacy-Gaussian extension"
    )
    args = parser.parse_args()
    # Why: each flag selects a separate fixed protocol; no completed cells are rerun.
    selection = {"legacy_only": True} if args.legacy_only else {}
    if args.describe:
        print(_encode(describe_protocol(**selection)))
        return
    # Why: one live sampler covers model work AND retained report/encoding buffers.
    stack = ExitStack()
    try:
        sampler = stack.enter_context(ProcessRssSampler())
    except Exception as error:
        report = unstarted_pilot_report(
            f"{type(error).__name__}: {error}", COMMAND_STARTED, **selection
        )
        print(_encode(report), flush=True)
        print(json.dumps(dict(within_resources=False, startup_error=str(error))), file=sys.stderr)
        raise SystemExit(2) from error
    with stack:
        report = run_pilot(sampler=sampler, started=COMMAND_STARTED, **selection)
        encoded = _encode(report)
        sampler.sample()
        print(encoded, flush=True)
    elapsed = monotonic() - COMMAND_STARTED
    memory = asdict(sampler.snapshot())
    within = (
        elapsed <= RESOURCES.wall_seconds and memory["peak_bytes"] <= RESOURCES.process_memory_bytes
    )
    receipt = dict(
        scope="Wall: after standard-library setup through numerical imports and sampler shutdown. RSS: sampled after imports through model work, report assembly, encoding and stdout emission; transient import RSS unobserved. External command elapsed covers full subprocess.",
        elapsed_seconds=elapsed,
        observed_process_rss=memory,
        output_bytes=_output_bytes(encoded),
        within_resources=within,
        pilot_status=report["status"],
    )
    print(json.dumps(receipt), file=sys.stderr, flush=True)
    if not within or report["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
