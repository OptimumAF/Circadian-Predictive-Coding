"""Narrow retained-probe metadata checks and fail-fast child launch.

No learner, data decoding, experiment permission, budget reset or retry is
provided. Scientific callers still own their frozen protocol and all costs.
"""

import argparse
from dataclasses import asdict
import json
from math import isfinite
from pathlib import Path
import subprocess
import sys

from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.infra.minigrid_affine_example import example_affine_training


def require_source_budget(owner: ToyBudgetSession) -> None:
    """Validate configuration and paid work separately, without observing time."""
    if type(owner) is not ToyBudgetSession or type(owner.budget) is not ToyExecutionBudget:
        raise ValueError("original retained-source budget session required")
    # Why: the source's omitted replay limit is None. That does not permit any
    # completed consolidation replay in this protocol; inspect its charge too.
    expected = asdict(
        ToyExecutionBudget(
            max_training_updates=6, max_wall_seconds=90, max_process_rss_bytes=512 * 1024**2
        )
    )
    actual = asdict(owner.budget)
    if actual.keys() != expected.keys() or any(
        type(actual[k]) is not type(value) or actual[k] != value for k, value in expected.items()
    ):
        raise ValueError(
            "retained-source limits must remain6/None/90/512MiB with no hidden-width limit"
        )
    if type(owner.replay_examples_completed) is not int or owner.replay_examples_completed != 0:
        raise ValueError(
            "retained-source completed consolidation replay must remain exact integer0"
        )
    if type(owner.updates_completed) is not int or not 0 <= owner.updates_completed <= 6:
        raise ValueError("retained-source completed updates must be an integer in0..6")


def require_training_asset(path: Path) -> None:
    """Metadata only; a matching size does not qualify dataset contents."""
    if not path.is_file() or path.stat().st_size != 94705:
        raise ValueError(
            f"retained-probe packaged TRAIN asset must be a regular94705-byte file: {path}"
        )


def run_checked(
    command: list[str], *, timeout_seconds: float = 180
) -> subprocess.CompletedProcess[str]:
    """Keep the resource alive and refuse before spawning if preflight fails."""
    if (
        type(command) is not list
        or not command
        or any(type(a) is not str or not a for a in command)
    ):
        raise ValueError("an explicit nonempty child argv list is required")
    if (
        type(timeout_seconds) not in (int, float)
        or not isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("child timeout must be finite and positive")
    with example_affine_training() as asset:
        require_training_asset(asset)
        return subprocess.run(command, check=True, timeout=timeout_seconds, text=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="optional child argv after --")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        if command:
            run_checked(command)
        else:
            with example_affine_training() as asset:
                require_training_asset(asset)
                print(
                    json.dumps(
                        dict(
                            status="metadata_pass",
                            path=str(asset),
                            bytes=asset.stat().st_size,
                            child_launched=False,
                            scientific_authorization=False,
                        )
                    )
                )
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"retained-probe preflight/child failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
