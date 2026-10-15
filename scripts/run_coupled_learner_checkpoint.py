"""Preview or run six fixed complete coupled learner checkpoint controls."""

import argparse
import json
from pathlib import Path

from src.adapters.coupled_learner_checkpoint_example import describe_example, run_example


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.run and args.directory is None:
        parser.error("--run requires an existing private --directory")
    value = describe_example() if args.describe else run_example(args.directory)
    print(
        json.dumps(
            value,
            indent=2 if args.describe else None,
            separators=None if args.describe else (",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
