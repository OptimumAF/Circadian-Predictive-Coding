"""Build a standalone recorded game, sensor, stream or memory inspector without a learner."""

import argparse
from contextlib import AbstractContextManager, contextmanager, nullcontext
from importlib import resources
import json
from pathlib import Path
from typing import Iterator

from src.infra.key_door_inspector_files import write_key_door_inspector

TICKET_METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")


@contextmanager
def _example_ticket(method: str) -> Iterator[Path]:
    """Hold a saved ticket resource open through exclusive HTML publication."""
    if method not in TICKET_METHODS:
        raise ValueError("unknown ticket example method")
    filename = f"{method}.json"
    local = Path(__file__).resolve().parents[1] / "examples/ticket_workflow" / filename
    if local.is_file():
        yield local
    else:
        with resources.as_file(
            resources.files("src.infra.ticket_workflow_data").joinpath(filename)
        ) as path:
            yield path


@contextmanager
def _example_comparison() -> Iterator[Path]:
    """Resolve the single saved example in a checkout or an installed wheel."""
    local = (
        Path(__file__).resolve().parents[1]
        / "examples/continuous_comparison/comparison-summary.json"
    )
    if local.is_file():
        yield local
    else:
        with resources.as_file(
            resources.files("src.infra.continuous_comparison_data").joinpath(
                "comparison-summary.json"
            )
        ) as path:
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--source-artifact",
        type=Path,
        help="saved game, sensor, stream, memory, comparison or detached CLM probe review JSON",
    )
    source.add_argument(
        "--example-comparison",
        action="store_true",
        help="bundled saved four-family comparison; no training",
    )
    source.add_argument(
        "--example-ticket",
        choices=TICKET_METHODS,
        help="bundled saved native ticket report; no training or model dependency",
    )
    parser.add_argument(
        "--resource-artifact",
        type=Path,
        help="separate sensor or coherent-stream cost JSON; source association remains unverified",
    )
    parser.add_argument("--output", type=Path, required=True, help="new standalone HTML file")
    args = parser.parse_args()
    try:
        context: AbstractContextManager[Path]
        if args.example_ticket:
            context = _example_ticket(args.example_ticket)
            source_label = f"bundled saved ticket {args.example_ticket}"
        elif args.example_comparison:
            context = _example_comparison()
            source_label = "bundled saved continuous comparison"
        else:
            context = nullcontext(args.source_artifact)
            source_label = str(args.source_artifact.resolve())
        with context as source_path:
            if args.resource_artifact is None:
                result = write_key_door_inspector(source_path, args.output)
            else:
                result = write_key_door_inspector(
                    source_path, args.output, resource_artifact=args.resource_artifact
                )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            dict(
                output=str(result.resolve()),
                source=source_label,
                bytes=result.stat().st_size,
            )
        )
    )


if __name__ == "__main__":
    main()
