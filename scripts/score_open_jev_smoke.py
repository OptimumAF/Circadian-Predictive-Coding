"""Score the fixed saved native smoke; no model imports or new inference.

Reprocesses one admitted original/reverse pair, preserving its source hash and
reported token counts. This is descriptive scoring, not a calibration study.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.adapters.open_jev_choice import build_open_jev_choice, decode_open_jev_probabilities
from src.app.decision_reference_evaluation import (
    DecisionCase,
    DecisionEvaluationFailed,
    evaluate_probability_reference,
)
from src.core.typed_decision import DecisionCandidate, DecisionRequest


def score_saved_smoke(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        body = stream.read(256 * 1024 + 1)
    if len(body) > 256 * 1024:
        raise ValueError("saved smoke exceeds256KiB")
    source = json.loads(body)
    if (
        source.get("definition") != "open-jev-native-choice-smoke-v1"
        or source.get("complete") is not True
        or len(source.get("requests", [])) != 2
    ):
        raise ValueError("expected a complete fixed native smoke pair")
    request = DecisionRequest(
        "smoke-1",
        "I forgot my password and cannot sign in.",
        (
            DecisionCandidate("password", "Password resets and login problems"),
            DecisionCandidate("billing", "Charges, invoices and refunds"),
        ),
    )
    reverse = DecisionRequest(
        request.request_id, request.query, tuple(reversed(request.allowed_candidates))
    )
    instructions = "Choose the customer's support intent."
    for row, value in zip(source["requests"], (request, reverse), strict=True):
        wire = build_open_jev_choice(value, instructions)
        if row.get("request") != wire:
            raise ValueError("saved query, candidates or instructions differ from frozen smoke")
        # Dict equality ignores order; native first-max semantics require it here.
        if list(row["request"]["questions"]["decision"]["criteria"]) != [
            item.candidate_id for item in value.allowed_candidates
        ]:
            raise ValueError("saved candidate order differs from frozen smoke")
        raw = row.get("response_json")
        if type(raw) is not str or len(raw.encode()) > 65536:
            raise ValueError("saved native response JSON missing or outside bound")
        if json.loads(raw) != row.get("response"):
            raise ValueError("saved native raw and owned responses disagree")
    calls = 0

    def predict(value):
        nonlocal calls
        if calls >= 2 or value != (request, reverse)[calls]:
            raise ValueError("saved callback request differs or observation allowance exhausted")
        raw = source["requests"][calls]["response_json"].encode()
        calls += 1
        return decode_open_jev_probabilities(value, raw)

    try:
        report = evaluate_probability_reference(
            (DecisionCase(request, "password", "saved-native-smoke", "outer_selection"),),
            predict=predict,
        )
    except DecisionEvaluationFailed as error:
        report = error.report
    report["source_artifact"] = dict(
        path=str(path.resolve()), bytes=len(body), sha256=hashlib.sha256(body).hexdigest()
    )
    report["execution"] = dict(
        mode="saved_reply_reprocessing",
        callback_observations=calls,
        new_model_loads=0,
        new_model_forwards=0,
        new_training_updates=0,
        reported_tokens="inherited from saved native execution; not newly paid compute",
    )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; preserve previous evidence")
    with args.output.open("x", encoding="utf8") as stream:
        stream.write("{}\n")
    try:
        report = score_saved_smoke(args.smoke_report)
    except Exception as error:
        report = dict(
            complete=False,
            error=f"{type(error).__name__}: {error}",
            execution=dict(new_model_loads=0, new_model_forwards=0),
        )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
