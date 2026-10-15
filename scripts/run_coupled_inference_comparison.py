"""Run the frozen allocated-time PC pilot, or describe it without training."""

import argparse
import json
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from src.adapters.coupled_inference_comparison import describe_protocol, run_pilot

MAX_OUTPUT_BYTES = 32 * 1024**2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed allocated-time coupled inference comparison"
    )
    parser.add_argument("--describe", action="store_true", help="Print protocol without training")
    parser.add_argument(
        "--granularity-aware",
        action="store_true",
        help="Select the separately frozen 30ms window and 3ms iteration protocol",
    )
    args = parser.parse_args()
    report = (
        describe_protocol(granularity_aware=args.granularity_aware)
        if args.describe
        else run_pilot(granularity_aware=args.granularity_aware)
    )
    encoded = json.dumps(report, allow_nan=False, indent=2)
    # Charge exactly the platform's printed line endings and terminal newline.
    if len((encoded + "\n").replace("\n", os.linesep).encode("utf-8")) > MAX_OUTPUT_BYTES:
        raise RuntimeError("pilot report exceeds its fixed output allowance")
    print(encoded)
    if not args.describe and report["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
