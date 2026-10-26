"""Preview or explicitly run the fixed saved-A32 public representation control.

This boundary reads one bounded prior JSON report, reuses its existing typed
admission, and publishes a new result exclusively. The adapter owns execution;
this script does not restore old budgets, recollect data, or select checkpoints.
"""

import argparse
import json
import os
from pathlib import Path
import sys
from time import monotonic
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.action_memory import ActionMemorySnapshot
    from src.core.demonstration_episode import DemonstrationEpisode

MAX_OUTPUT_BYTES = 32 * 1024**2


def describe_protocol() -> dict[str, Any]:
    from src.adapters.minigrid_invariant_pilot import describe_protocol as describe

    return describe()


def run_pilot(
    states: tuple["ActionMemorySnapshot", "ActionMemorySnapshot"],
    sources: tuple["DemonstrationEpisode", "DemonstrationEpisode"],
) -> dict[str, Any]:
    from src.adapters.minigrid_invariant_pilot import run_pilot as run

    return run(states, sources)


def load_inputs(
    path: Path,
) -> tuple[
    tuple["ActionMemorySnapshot", "ActionMemorySnapshot"],
    tuple["DemonstrationEpisode", "DemonstrationEpisode"],
]:
    """Read once, validate the complete prior run, and detach its fixed A32 slice."""
    from scripts.run_minigrid_memory_probe import (
        _read_source,
        _return_episodes,
        _snapshot,
        _snapshots_from_report,
    )

    report = _read_source(path)
    initial, _ = _snapshots_from_report(report)
    sources = _return_episodes(report)
    # Why: checkpoint32 is the original A prefix, not the mature128 endpoint.
    # The shared validator above still admits all five original observations.
    prefix = _snapshot(report["observations"][1]["snapshot"], 32)
    return (initial, prefix), sources


def _output_preflight(parser: argparse.ArgumentParser, path: Path | None) -> None:
    if path is not None:
        if path.exists():
            parser.error("output already exists; preserve the original attempt")
        if not path.parent.is_dir():
            parser.error("output parent directory must exist")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="execute the frozen local protocol")
    mode.add_argument("--describe", action="store_true", help="print the resource-free preview")
    parser.add_argument("--source", type=Path, help="original completed memory-live JSON artifact")
    parser.add_argument("--output", type=Path, help="publish a new JSON artifact; never overwrite")
    args = parser.parse_args(argv)
    if not args.run:
        if args.source is not None or args.output is not None:
            parser.error("source/output require --run; preview reads no artifact")
        print(json.dumps(describe_protocol(), indent=2, allow_nan=False))
        return 0
    if args.source is None:
        parser.error("--run requires --source")
    _output_preflight(parser, args.output)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        started = monotonic()
        states, sources = load_inputs(args.source)
        source_load_seconds = monotonic() - started
        report = run_pilot(states, sources)
        if type(report) is not dict or report.get("status") not in (
            "complete",
            "incomplete",
            "failed",
        ):
            raise ValueError("recipe did not return a report with a declared completion status")
        report.update(source_artifact=str(args.source), source_load_seconds=source_load_seconds)
        payload = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
        if len(payload) > MAX_OUTPUT_BYTES:
            raise ValueError("invariant pilot output exceeds 32 MiB; no result published")
        if args.output is not None:
            with args.output.open("xb") as handle:
                handle.write(payload)
    except Exception as error:
        print(f"invariant pilot failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    sys.stdout.write(payload.decode("utf-8"))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
