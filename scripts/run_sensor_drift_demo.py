"""Run a fixed small recurring-sensor qualification (no comparative claim)."""

import argparse
import json
import os
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    condition = parser.add_mutually_exclusive_group()
    condition.add_argument(
        "--contrast-inversion",
        action="store_true",
        help="reverse B camera contrast (1 - pixels), preserving semantic labels and original A return",
    )
    condition.add_argument(
        "--stationary",
        action="store_true",
        help="use twelve baseline-A batches for post-calibration false alarms; requires --monitoring",
    )
    parser.add_argument("--method", choices=("bp", "pc", "neutral_cpc", "gated_cpc"), default="bp")
    parser.add_argument(
        "--heldout",
        action="store_true",
        help="score frozen final A/B sets only after training and probes",
    )
    parser.add_argument(
        "--learnability",
        action="store_true",
        help="observe isolated reserved-task probes at fixed wake checkpoints",
    )
    control = parser.add_mutually_exclusive_group()
    control.add_argument(
        "--replay",
        action="store_true",
        help="rehearse arrived recent rows using the existing shared helper",
    )
    control.add_argument(
        "--scheduled", action="store_true", help="update BP after each four arrived batches"
    )
    parser.add_argument(
        "--monitoring",
        action="store_true",
        help="fit prefix-only descriptive shift/recovery diagnostics",
    )
    parser.add_argument(
        "--incident-replay",
        action="store_true",
        help="admit future replay boundaries from arrived feature onsets; requires --replay --monitoring",
    )
    parser.add_argument("--output", type=Path, help="new JSON path; preserve existing attempts")
    args = parser.parse_args(argv)
    if args.stationary and not args.monitoring:
        parser.error("--stationary requires --monitoring")
    if args.stationary and (args.heldout or args.learnability):
        parser.error("--stationary does not use recurring --heldout or --learnability protocols")
    if args.incident_replay and not (args.replay and args.monitoring):
        parser.error("--incident-replay requires --replay --monitoring")
    if args.method != "bp" and (args.scheduled or args.replay):
        parser.error("--scheduled and --replay require --method bp")
    if args.learnability and (args.scheduled or args.replay):
        parser.error("--learnability requires ordinary methods without --scheduled or --replay")
    if args.describe and args.output is not None:
        parser.error("--describe does not write a run artifact")
    if args.output is not None and (args.output.exists() or not args.output.parent.is_dir()):
        parser.error("output must be a new file in an existing directory")
    # Bound BLAS before importing NumPy; this affects only the local child process.
    if not args.describe:
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[name] = "1"
    from src.adapters.sensor_drift_demo import describe_protocol, run_demo

    options: dict[str, Any] = {}
    if args.contrast_inversion:
        options["contrast_inversion"] = True
    if args.stationary:
        options["stationary"] = True
    if args.monitoring:
        options["monitoring"] = True
    if args.scheduled:
        options["scheduled"] = True
    if args.replay:
        options["replay"] = True
    if args.learnability:
        options["learnability"] = True
    if args.heldout:
        options["heldout"] = True
    if args.incident_replay:
        options["incident_replay"] = True
    # Preserve the old BP call shape for existing callers and mode descriptors.
    if args.method != "bp":
        options["method"] = args.method
    if args.describe:
        report = describe_protocol(**options)
    else:
        report = run_demo(**options)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if len(payload.encode("utf8")) > 128 * 1024:
        raise ValueError("sensor demo JSON exceeds 128KiB")
    if args.output is None:
        print(payload, end="")
    else:
        with args.output.open("x", encoding="utf8", newline="\n") as handle:
            handle.write(payload)
        print(json.dumps(dict(status=report["status"], output=str(args.output))))
    return 0 if args.describe or report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
