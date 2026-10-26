"""Describe or run one fixed, local learnability engineering pilot."""

import argparse
import json
import os

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

from src.adapters.learnability_pilot import describe_protocol, run_pilot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--describe", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument(
        "--scenario",
        choices=("four-update", "sixteen-update"),
        default="four-update",
        help="Choose a fixed probe protocol; the original four-update protocol is the default.",
    )
    args = parser.parse_args()
    selected = describe_protocol if args.describe else run_pilot
    report = selected(extended=True) if args.scenario == "sixteen-update" else selected()
    raw = json.dumps(report, allow_nan=False, indent=2)
    if len(raw.encode("utf8")) > 32 * 1024**2:
        raise ValueError("learnability report exceeds the fixed output allowance")
    print(raw)
    return 0 if args.describe or report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
