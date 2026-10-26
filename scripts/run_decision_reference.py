"""Run one frozen six-case lexical routing reference; no training or tool actions."""

import argparse
import json
from pathlib import Path

from src.app.decision_reference_evaluation import (
    DecisionCase,
    DecisionEvaluationFailed,
    evaluate_decision_reference,
)
from src.core.typed_decision import DecisionCandidate, DecisionRequest


def example_cases() -> tuple[DecisionCase, ...]:
    catalog = (
        DecisionCandidate("billing", "invoice billing payment refund"),
        DecisionCandidate("password", "password reset login account"),
        DecisionCandidate("shipping", "shipping parcel delivery tracking"),
    )
    inputs = (
        ("invoice refund", "billing", catalog),
        ("password reset", "password", catalog),
        ("delivery tracking", "shipping", catalog),
        ("forgot credentials", "password", catalog),
        ("password reset", "password", (catalog[0], catalog[2])),
        ("refund payment", "billing", (catalog[1], catalog[2])),
    )
    return tuple(
        DecisionCase(
            DecisionRequest(f"route-{index + 1}", query, allowed), target, "support-intent-v1"
        )
        for index, (query, target, allowed) in enumerate(inputs)
    )


def _publish(report: dict, output: Path | None) -> None:
    encoded = json.dumps(report, allow_nan=False, indent=2)
    if output is None:
        print(encoded)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf8") as handle:
            handle.write(encoded + "\n")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="fresh JSON path; otherwise stdout")
    args = parser.parse_args(argv)
    try:
        if args.output is not None and args.output.exists():
            raise FileExistsError(f"output already exists: {args.output}")
        exit_code = 0
        try:
            report = evaluate_decision_reference(example_cases())
        except DecisionEvaluationFailed as error:
            report = error.report
            exit_code = 1
        _publish(report, args.output)
        if exit_code:
            raise SystemExit(exit_code)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
