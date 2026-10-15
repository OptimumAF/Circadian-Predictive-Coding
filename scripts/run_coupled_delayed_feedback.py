"""Run fixed delayed producing-version feedback across native serving changes."""

import argparse
import json
import os
from typing import Any

from src.adapters.coupled_delayed_feedback import describe_example, run_example

MAX_OUTPUT_BYTES = 2 * 1024**2


def _print_report(report: dict[str, Any]) -> None:
    encoded = json.dumps(report, allow_nan=False, sort_keys=True, indent=2)
    if len((encoded + "\n").replace("\n", os.linesep).encode("utf-8")) > MAX_OUTPUT_BYTES:
        raise RuntimeError("delayed feedback report exceeds2MiB")
    print(encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", help="Preview without operational work")
    args = parser.parse_args()
    _print_report(describe_example() if args.describe else run_example())


if __name__ == "__main__":
    main()
