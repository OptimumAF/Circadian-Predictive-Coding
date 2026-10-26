"""Run one declared TRAIN-only MiniGrid reward route through the generic runtime.

The describe path performs no model/environment work. This is an engineering
recipe, not a comparative learning protocol or a final evaluation runner.
"""

import argparse
import json
import os
from pathlib import Path
import sys

MAX_OUTPUT_BYTES = 16 * 1024**2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", help="print the fixed protocol only")
    parser.add_argument(
        "--warm-start",
        action="store_true",
        help="recreate the fixed accepted TRAIN A128 BC prefix before reward updates",
    )
    parser.add_argument(
        "--train-readouts",
        action="store_true",
        help="read owned warm and reward checkpoints on known TRAIN contexts after all updates",
    )
    parser.add_argument(
        "--momentum-control",
        action="store_true",
        help="apply eight separate zero-loss control updates to original TRAIN episodes, then read both arms",
    )
    parser.add_argument(
        "--extended-control-readouts",
        action="store_true",
        help="select paired protocol v2: separately budgeted210s diagnostics within300s whole limit",
    )
    parser.add_argument("--output", type=Path, help="publish to a new JSON file; never overwrite")
    args = parser.parse_args(argv)
    if args.train_readouts and not args.warm_start:
        parser.error("--train-readouts requires --warm-start")
    if args.momentum_control and not (args.warm_start and args.train_readouts):
        parser.error("--momentum-control requires --warm-start and --train-readouts")
    if args.extended_control_readouts and not args.momentum_control:
        parser.error("--extended-control-readouts requires --momentum-control")
    if args.describe and args.output is not None:
        parser.error("--describe does not publish a run artifact")
    if args.output is not None:
        if args.output.exists():
            parser.error("output already exists; preserve the original attempt")
        if not args.output.parent.is_dir():
            parser.error("output parent directory must exist")

    # Set before optional native imports; all learner work remains CPU only.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from src.adapters.minigrid_reward_pilot import describe_protocol, run_pilot

    options = {"warm_start": True} if args.warm_start else {}
    if args.train_readouts:
        options["train_readouts"] = True
    if args.momentum_control:
        options["momentum_control"] = True
    if args.extended_control_readouts:
        options["extended_control_readouts"] = True
    if args.describe:
        print(json.dumps(describe_protocol(**options), indent=2, allow_nan=False))
        return 0
    from src.shared.torch_runtime import require_torch

    torch = require_torch()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    report = run_pilot(**options)
    payload = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf8")
    if len(payload) > MAX_OUTPUT_BYTES:
        raise ValueError("reward route output exceeds16MiB; no result published")
    if args.output is None:
        sys.stdout.write(payload.decode("utf8"))
    else:
        with args.output.open("xb") as handle:
            handle.write(payload)
        print(
            json.dumps(
                {"status": report["status"], "output": str(args.output), "bytes": len(payload)},
                allow_nan=False,
            )
        )
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
