"""Save all fixed v11 modulation/control outcomes to a new local JSON file."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from src.app.difficulty_matched_benchmark import (
    fixed_difficulty_manifest,
    run_difficulty_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    path = parser.parse_args().result
    result = run_difficulty_comparison(fixed_difficulty_manifest())
    payload = json.dumps(asdict(result), sort_keys=True, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(payload)
    print(json.dumps({"result": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
