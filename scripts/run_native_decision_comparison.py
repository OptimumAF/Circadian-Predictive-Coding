"""Fixed descriptive six-case native/linear/lexical comparison, without training.

Uses saved lexical decisions and the final saved linear state. Native inference
is optional until the outer CLI's lazy loader; pure comparison accepts a fake.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from scripts.run_decision_reference import example_cases
from scripts.run_open_jev_native_smoke import _load, _validate_metadata
from src.adapters.open_jev_choice import build_open_jev_choice, decode_open_jev_probabilities
from src.adapters.perceptron_decision import PerceptronDecisionPredictor
from src.app.decision_reference_evaluation import (
    DecisionEvaluationFailed,
    _metrics,
    _score_case,
    evaluate_decision_reference,
    evaluate_probability_reference,
)
from src.core.categorical_perceptron import PerceptronPolicy, PerceptronSnapshot
from src.core.decision_features import DecisionFeatureSpec
from src.core.typed_decision import (
    CandidateScore,
    DecisionCandidate,
    DecisionResult,
    validate_decision_result,
)


def _controls(lexical, linear):
    cases = example_cases()
    if (
        lexical.get("complete") is not True
        or lexical.get("definition") != "typed_decision_retrieval_reference_v1"
        or len(lexical.get("cases", [])) != 6
    ):
        raise ValueError("expected complete six-case saved lexical report")
    checked = []
    for row, case in zip(lexical["cases"], cases, strict=True):
        if row.get("request") != json.loads(json.dumps(asdict(case.request))) or any(
            row.get(name) != getattr(case, name) for name in ("target_id", "source_group", "role")
        ):
            raise ValueError("saved lexical cases differ from fixed protocol")
        results = []
        for key in ("result", "reversed_result"):
            raw = row[key]
            result = DecisionResult(
                raw["request_id"],
                raw["selected_id"],
                tuple(CandidateScore(**item) for item in raw["scores"]),
                raw["token_visits"],
                raw["candidate_comparisons"],
            )
            validate_decision_result(case.request, result)
            results.append(result)
        if results[0] != results[1]:
            raise ValueError("saved lexical order control differs")
        checked.append(_score_case(case, *results))
    if _metrics(checked) != lexical["metrics"]:
        raise ValueError("saved lexical metrics disagree with observed decisions")
    if linear.get("complete") is not True or linear.get("definition") != "typed_linear_routing_v1":
        raise ValueError("expected complete saved linear report")
    spec = linear["feature_spec"]
    if spec["vocabulary"] != ["credentials", "invoice", "parcel", "password", "refund", "tracking"]:
        raise ValueError("saved linear vocabulary differs from frozen control")
    features = DecisionFeatureSpec(
        tuple(spec["vocabulary"]), tuple(DecisionCandidate(**x) for x in spec["catalog"])
    )
    snapshots = [x for x in linear["snapshots"] if x["update"] == 6]
    if len(snapshots) != 1:
        raise ValueError("expected unique final linear snapshot6")
    state = snapshots[0]["state"]
    if (state["completed_updates"], state["training_examples"], state["mistake_updates"]) != (
        6,
        18,
        6,
    ):
        raise ValueError("saved final linear counters differ")
    predictor = PerceptronDecisionPredictor(
        features,
        PerceptronSnapshot(
            PerceptronPolicy(**state["policy"]),
            tuple(tuple(row) for row in state["coefficients"]),
            6,
            18,
            6,
        ),
    )
    for case in cases:
        if any(item not in features.catalog for item in case.request.allowed_candidates):
            raise ValueError("linear candidate IDs/descriptions differ from fixed cases")
    return deepcopy(lexical), predictor


def _contrasts(native, control):
    result = {
        name: native[name] - control[name]
        for name in ("accuracy", "candidate_recall", "brier_mean", "ece_5bin")
    }
    finite = native["nll_status"] == control["nll_status"] == "finite"
    result["nll_mean"] = native["nll_mean"] - control["nll_mean"] if finite else None
    result["nll_status"] = "finite_difference" if finite else "nonfinite_comparison_unavailable"
    return result


def compare_decisions(predict, lexical: dict, linear: dict, progress=None) -> dict[str, Any]:
    report: dict[str, Any] = dict(
        definition="native_decision_comparison_v1",
        complete=False,
        results={},
        native_returns=[],
        qualification="six descriptive development cases; unequal pretraining/capacity/compute; not calibration, adaptation or CPC benefit",
    )
    try:
        lexical, predictor = _controls(deepcopy(lexical), deepcopy(linear))
        report["results"]["lexical"] = lexical
        report["results"]["linear"] = evaluate_decision_reference(
            example_cases(), predict=predictor
        )
        if progress:
            progress(
                dict(
                    event="comparison_linear_complete",
                    selected_update=6,
                    linear_report=deepcopy(report["results"]["linear"]),
                )
            )

        def native(request):
            wire = build_open_jev_choice(request, "Choose the customer's support intent.")
            row: dict[str, Any] = {"request": deepcopy(wire)}
            report["native_returns"].append(row)
            response = predict(wire)
            try:
                raw = json.dumps(response, allow_nan=True, separators=(",", ":"))
                if len(raw.encode()) > 65536:
                    raise ValueError("native reply exceeds64KiB")
                row["response_json"] = raw
            except (TypeError, ValueError, OverflowError, RecursionError) as error:
                row["response_unavailable"] = type(error).__name__
                raise ValueError("native reply cannot be captured") from error
            if progress:
                progress(dict(event="comparison_native_reply", **deepcopy(row)))
            if wire != row["request"] or list(wire["questions"]["decision"]["criteria"]) != [
                x.candidate_id for x in request.allowed_candidates
            ]:
                raise ValueError("native predictor changed frozen wire request")
            body = json.dumps(response, allow_nan=False, separators=(",", ":")).encode()
            _validate_metadata(response, candidate_count=len(request.allowed_candidates))
            return decode_open_jev_probabilities(request, body)

        report["results"]["native"] = evaluate_probability_reference(
            example_cases(), predict=native
        )
        metrics = report["results"]["native"]["metrics"]
        report["contrasts"] = {
            "native_minus_" + name: _contrasts(metrics, report["results"][name]["metrics"])
            for name in ("lexical", "linear")
        }
        report["directions"] = dict(
            accuracy="higher",
            candidate_recall="higher",
            brier_mean="lower",
            nll_mean="lower",
            ece_5bin="lower",
        )
        report["complete"] = True
    except DecisionEvaluationFailed as error:
        report["results"]["native" if report["native_returns"] else "linear"] = error.report
        report["error"] = str(error)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    return report


def _read(path):
    with path.open("rb") as stream:
        body = stream.read(128 * 1024 + 1)
    if len(body) > 128 * 1024:
        raise ValueError("saved control exceeds128KiB")
    return json.loads(body), dict(
        path=str(path.resolve()), bytes=len(body), sha256=hashlib.sha256(body).hexdigest()
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lexical-report", "linear-report", "source", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output exists; preserve prior evidence")
    with args.output.open("x", encoding="utf8") as stream:
        stream.write("{}\n")
    report: dict[str, Any] = dict(complete=False, events=[], model_load_attempts=0)

    def progress(event):
        report["events"].append(event)
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )

    try:
        lexical, lexical_source = _read(args.lexical_report)
        linear, linear_source = _read(args.linear_report)
        report["control_sources"] = dict(lexical=lexical_source, linear=linear_source)
        loaded: list[Any] = []

        def predict(wire):
            if not loaded:
                report["model_load_attempts"] += 1
                loaded.append(
                    _load(
                        args.source,
                        args.checkpoint,
                        progress,
                        prediction_limit=12,
                        forward_limit=32,
                        token_limit=8192,
                    )
                )
            return loaded[0](wire)

        report.update(compare_decisions(predict, lexical, linear, progress))
        report["current_work"] = dict(
            lexical_predictions=0,
            linear_readouts=report.get("results", {}).get("linear", {}).get("work", {}),
            training_updates=0,
            training_rows=0,
            model_load_attempts=report["model_load_attempts"],
            inherited_linear_training=dict(updates=6, rows=18, mistakes=6),
            compute_matched=False,
        )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        sys.setprofile(None)
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
