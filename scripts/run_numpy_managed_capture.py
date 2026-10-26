"""Describe or run bounded legacy, issued, restored or successive replay capture."""

import argparse
import json
import os

from src.adapters.numpy_capture_example import describe_example, run_example
from src.adapters.numpy_snapshot_capture_example import (
    describe_snapshot_example,
    run_snapshot_example,
)
from src.adapters.numpy_restored_capture_example import (
    describe_restored_example,
    run_restored_example,
    describe_successive_example,
    run_successive_example,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--run", action="store_true")
    example = parser.add_mutually_exclusive_group()
    example.add_argument(
        "--issued-snapshot",
        action="store_true",
        help="Capture one trained CPC checkpoint with its original replay witnesses.",
    )
    example.add_argument(
        "--restored-replay",
        action="store_true",
        help="Capture original current and retired CPC replay after one managed restore.",
    )
    example.add_argument(
        "--successive-replay",
        action="store_true",
        help="Capture successive original CPC updates after restore and a paid paused poll.",
    )
    args = parser.parse_args()
    if args.successive_replay:
        report = describe_successive_example() if args.describe else run_successive_example()
    elif args.restored_replay:
        report = describe_restored_example() if args.describe else run_restored_example()
    elif args.issued_snapshot:
        report = describe_snapshot_example() if args.describe else run_snapshot_example()
    else:
        report = describe_example() if args.describe else run_example()
    raw = json.dumps(report, allow_nan=False, sort_keys=True, indent=2 if args.describe else None)
    maximum = (
        131072 if args.issued_snapshot or args.restored_replay or args.successive_replay else 524288
    )
    if len((raw + "\n").replace("\n", os.linesep).encode("utf-8")) > maximum:
        kind = (
            "successive replay"
            if args.successive_replay
            else "restored replay"
            if args.restored_replay
            else "snapshot"
            if args.issued_snapshot
            else "legacy"
        )
        raise ValueError(f"{kind} capture report exceeds bounded output allowance")
    print(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
