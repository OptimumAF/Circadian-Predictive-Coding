"""Compare saved original MiniGrid no-replay/replay reports without model work."""

import argparse
import json
from pathlib import Path

from src.adapters.minigrid_rehearsal_comparison import compare_reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--rehearsal", type=Path, required=True)
    parser.add_argument("--method", choices=("bp", "pc", "neutral_cpc", "gated_cpc"), required=True)
    options = parser.parse_args()
    try:
        control = json.loads(options.control.read_text(encoding="utf8"))
        rehearsal = json.loads(options.rehearsal.read_text(encoding="utf8"))
        result = compare_reports(control, rehearsal, method=options.method)
    except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
        parser.error(f"saved MiniGrid comparison refused: {error}")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
