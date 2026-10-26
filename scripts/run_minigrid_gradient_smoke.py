"""Run one fixed warm128 TRAIN cue-gradient measurement, or describe it."""

import argparse
import json
import os
from pathlib import Path
import sys

MAX_OUTPUT_BYTES = 32 * 1024**2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--describe", action="store_true", help="print the resource-free fixed protocol"
    )
    parser.add_argument("--output", type=Path, help="write a new JSON artifact; never overwrite")
    parser.add_argument(
        "--cue-update-check",
        action="store_true",
        help="also compare two isolated single-cue updates",
    )
    args = parser.parse_args(argv)
    if args.describe and args.output is not None:
        parser.error("--describe does not publish a run artifact")
    if args.output is not None:
        if args.output.exists():
            parser.error("output already exists; preserve the original attempt")
        if not args.output.parent.is_dir():
            parser.error("output parent directory must exist")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from src.adapters.minigrid_warm_gradient_pilot import describe_protocol, run_pilot

    if args.describe:
        protocol = (
            describe_protocol(cue_updates=True) if args.cue_update_check else describe_protocol()
        )
        print(json.dumps(protocol, indent=2, allow_nan=False))
        return 0
    from src.shared.torch_runtime import require_torch

    torch = require_torch()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    report = run_pilot(cue_updates=True) if args.cue_update_check else run_pilot()
    payload = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf8")
    if len(payload) > MAX_OUTPUT_BYTES:
        raise ValueError("warm-gradient output exceeds32MiB; no result published")
    if args.output is None:
        sys.stdout.write(payload.decode("utf8"))
    else:
        with args.output.open("xb") as handle:
            handle.write(payload)
        print(
            json.dumps(
                dict(status=report["status"], output=str(args.output), bytes=len(payload)),
                allow_nan=False,
            )
        )
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
