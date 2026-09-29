"""Write every fixed v14 full-stack trigger outcome to new local JSON."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from src.app.continual_trigger_replay_outcomes import (
    TriggerReplayComparison,
    run_trigger_replay_outcomes,
)
from src.app.continual_trigger_replay_schedule import fixed_trigger_replay_manifest


def build_payload() -> str:
    """Serialize all predeclared cells and contrasts without choosing one."""
    result = run_trigger_replay_outcomes(fixed_trigger_replay_manifest())
    return serialize_outcome_comparison(result)


def serialize_outcome_comparison(result: TriggerReplayComparison) -> str:
    """Serialize one globally sealed comparison without retraining."""
    return json.dumps(asdict(result), indent=2, sort_keys=True, allow_nan=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    result_path = parser.parse_args().result
    payload = build_payload()
    with result_path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(payload)
    print(json.dumps({"result": str(result_path), "sha256": sha256(payload.encode()).hexdigest()}))


if __name__ == "__main__":
    main()
