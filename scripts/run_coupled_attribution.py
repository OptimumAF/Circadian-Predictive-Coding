"""Run the frozen local pilot or describe its protocol without training."""

import argparse
import json
import os

# Limit only this invocation, before NumPy/BLAS is imported by the adapter.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from src.adapters.coupled_attribution_pilot import describe_protocol, run_pilot


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed local coupled attribution pilot")
    parser.add_argument("--describe", action="store_true", help="Print protocol without training")
    args = parser.parse_args()
    report = describe_protocol() if args.describe else run_pilot()
    encoded = json.dumps(report, allow_nan=False, indent=2)
    if len(encoded.encode("utf-8")) > 32 * 1024**2:
        raise RuntimeError("pilot report exceeds its fixed output allowance")
    print(encoded)
    if not args.describe and report["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
