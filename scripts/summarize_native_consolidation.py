"""Describe frozen native calibration/rank evidence without loading a learner.

Inputs are four accepted development reports, not final data or model weights.
Outputs retain original failures, costs and intervention-history qualifications.
No fitting, temperature selection, predictions, training or rank search occurs.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from math import isclose, isfinite, sqrt, fsum
from pathlib import Path
from typing import Any

from src.adapters.open_jev_choice import decode_open_jev_probabilities
from src.app.decision_reference_evaluation import DecisionCase, _metrics, _probability_row
from src.core.native_choice_sequence import sequence_examples
from src.shared.process_memory import read_process_rss_bytes

MAX_REPORT_BYTES = 1024 * 1024
MAX_OUTPUT_BYTES = 256 * 1024
SOURCES = {
    "warm": dict(
        path="native-choice-recurring-20261022/actual.json",
        sha256="3f680ee3bbc93ff0dfc144f7abe322317a034290460b8c546d198b8f1646ad58",
        definition="native_choice_recurring_v1",
        phase="after_B",
        updates=18,
    ),
    "reset": dict(
        path="native-optimizer-reset-20261022/actual.json",
        sha256="04c2b8d5b387c5d0617a68e37516246a5560961b5f04a1bea7a44bb99db18b45",
        definition="native_optimizer_reset_control_v1",
        phase="after_optimizer_reset_B",
        updates=12,
    ),
    "projection": dict(
        path="native-projection-stream-20261022/actual.json",
        sha256="f3527d7a3fc56f27fbd1ec6acb95d5767b72b7af31dc7957dd77d4cfd52f5bda",
        definition="native_adapter_projection_control_v1",
        phase="after_adapter_projection_B",
        updates=12,
    ),
    "consolidation": dict(
        path="native-consolidation-stream-20261022/actual.json",
        sha256="d33a5e7e40f431bb282f57c21a86c9b9e2d12f960d57bd7ab1e8c7752b6159ca",
        definition="native_adapter_consolidation_control_v1",
        phase="after_adapter_consolidation_B",
        updates=12,
    ),
}


def _score_rows(source, phase):
    examples = {e.example_id: e for e in sequence_examples() if e.role == "outer_selection"}
    rows = source.get("readouts")
    if type(rows) is not list or not 6 <= len(rows) <= 24:
        raise ValueError("expected bounded saved development readouts")
    selected = [row for row in rows if type(row) is dict and row.get("phase") == phase]
    if len(selected) != 6 or {r.get("example_id") for r in selected} != examples.keys():
        raise ValueError("expected exactly six distinct declared endpoint readouts")
    scored = []
    for row in selected:
        example = examples[row["example_id"]]
        if any(
            row.get(key) != value
            for key, value in dict(
                context=example.context,
                role="outer_selection",
                source_group=example.source_group,
                target=example.target_id,
                request=json.loads(json.dumps(asdict(example.request))),
            ).items()
        ):
            raise ValueError("saved development identity/role/target/request changed")
        raw = row.get("response_json")
        if type(raw) is not str or json.loads(raw) != row.get("response"):
            raise ValueError("saved raw response copies disagree")
        result = decode_open_jev_probabilities(example.request, raw.encode("utf8"))
        if json.loads(json.dumps(asdict(result))) != row.get("result"):
            raise ValueError("saved result differs from captured native probabilities")
        case = DecisionCase(example.request, example.target_id, example.source_group)
        score = _probability_row(case, result.selected_id, dict(result.probabilities), {}, {})
        expected = _metrics([score])
        infinite = expected["nll_mean"] is None
        if (
            type(row.get("correct")) is not bool
            or row["correct"] != score["correct"]
            or row.get("nll_status") != ("infinite" if infinite else "finite")
            or (infinite and row.get("nll") is not None)
            or (
                not infinite
                and (
                    type(row.get("nll")) not in (int, float)
                    or not isclose(row["nll"], expected["nll_mean"], rel_tol=1e-12, abs_tol=1e-12)
                )
            )
        ):
            raise ValueError("saved correctness/NLL differs from captured probabilities")
        score.update(
            request_id=example.example_id,
            context=example.context,
            selected_id=result.selected_id,
            probabilities=dict(result.probabilities),
        )
        scored.append(score)
    return scored


def _arm(rows):
    groups = {c: _metrics([r for r in rows if r["context"] == c]) for c in ("A", "B", "anchor")}
    keys = (
        "request_id",
        "context",
        "target_id",
        "selected_id",
        "probabilities",
        "target_probability",
        "candidate_recalled",
        "correct",
        "brier",
        "confidence",
    )
    return dict(
        overall=_metrics(rows),
        by_context=groups,
        cases=[{key: row[key] for key in keys} for row in rows],
    )


def _difference(after, before):
    result = {
        key: after[key] - before[key]
        for key in ("accuracy", "candidate_recall", "brier_mean", "ece_5bin")
    }
    finite = after["nll_mean"] is not None and before["nll_mean"] is not None
    result.update(
        nll_mean=after["nll_mean"] - before["nll_mean"] if finite else None,
        nll_status="finite_difference" if finite else "nonfinite_comparison_unavailable",
    )
    return result


def _contrast(after, before):
    return dict(
        overall=_difference(after["overall"], before["overall"]),
        by_context={
            c: _difference(after["by_context"][c], before["by_context"][c])
            for c in ("A", "B", "anchor")
        },
    )


def _rank_characterization(receipt, rank):
    if (
        type(receipt) is not dict
        or type(receipt.get("effective_rank")) is not int
        or receipt["effective_rank"] != rank
    ):
        raise ValueError("rank receipt differs from the declared intervention")
    modules = receipt.get("modules")
    if type(modules) is not list or not 1 <= len(modules) <= 256:
        raise ValueError("expected1..256 bounded module spectra")
    result = []
    seen = set()
    for row in modules:
        if (
            type(row) is not dict
            or type(row.get("module")) is not str
            or not row["module"]
            or row["module"] in seen
        ):
            raise ValueError("module spectrum identity missing or duplicated")
        seen.add(row["module"])
        values = row.get("singular_values")
        if (
            type(values) is not list
            or not 1 <= len(values) <= 64
            or any(type(v) not in (int, float) or not isfinite(v) or v < 0 for v in values)
            or values != sorted(values, reverse=True)
        ):
            raise ValueError("expected finite descending nonnegative bounded spectrum")
        numerical = row.get("numerical_rank")
        if (
            type(row.get("allocated_rank")) is not int
            or row["allocated_rank"] != 8
            or type(row.get("effective_rank")) is not int
            or row["effective_rank"] != rank
            or type(numerical) is not int
            or not 0 <= numerical <= len(values)
            or type(row.get("scaling")) not in (int, float)
            or not isfinite(row["scaling"])
            or row["scaling"] <= 0
            or type(row.get("discarded_frobenius")) not in (int, float)
            or not isfinite(row["discarded_frobenius"])
            or row["discarded_frobenius"] < 0
            or not isclose(
                row["discarded_frobenius"],
                sqrt(fsum(v * v for v in values[rank:])),
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("rank metadata or discarded spectral tail differs")
        result.append(
            {
                key: deepcopy(row[key])
                for key in (
                    "module",
                    "scaling",
                    "allocated_rank",
                    "effective_rank",
                    "numerical_rank",
                    "singular_values",
                    "discarded_frobenius",
                )
            }
        )
    return dict(
        allocated_rank=8,
        requested_projection_rank=rank,
        modules=result,
        numerical_rank_scope="reported_float64_operator_before_truncation",
        installed_float32_numerical_rank="unmeasured",
        qualification="fixed allocation; spectra do not establish task performance at untested ranks or memory savings",
    )


def _historical_work(source):
    keys = (
        "app_committed_updates",
        "native_work",
        "optimizer_reset_work",
        "adapter_projection_work",
        "adapter_consolidation_work",
        "outer_seconds",
    )
    work = {key: deepcopy(source[key]) for key in keys if key in source}
    events = source.get("events", [])
    if type(events) is not list or len(events) > 512 or any(type(e) is not dict for e in events):
        raise ValueError("expected bounded historical work observations")
    observed = next((e["observed_work"] for e in reversed(events) if "observed_work" in e), None)
    work["observed_work"] = deepcopy(observed)
    peaks = [e[k] for e in events for k in ("peak_rss_bytes", "outer_peak_rss_bytes") if k in e]
    work["peak_rss_bytes"] = max(peaks, default=None)
    work["counter_scope"] = "source_reported_total; absent_observations_unknown_not_zero"
    return work


def summarize_native_reports(reports: dict[str, Any]) -> dict[str, Any]:
    """Recompute descriptive scores from captured replies; inputs stay owned."""
    if type(reports) is not dict or reports.keys() != SOURCES.keys():
        raise ValueError("expected exactly warm/reset/projection/consolidation reports")
    corpus = json.loads(json.dumps([asdict(e) for e in sequence_examples()]))
    for name, spec in SOURCES.items():
        source = reports[name]
        flag = "sequence_complete" if name == "warm" else "control_complete"
        if (
            type(source) is not dict
            or source.get("definition") != spec["definition"]
            or source.get("complete") is not True
            or source.get(flag) is not True
            or type(source.get("app_committed_updates")) is not int
            or source["app_committed_updates"] != spec["updates"]
            or source.get("examples") != corpus
        ):
            raise ValueError("expected completed fixed native study/corpus")
    arms = {"unchanged": _arm(_score_rows(reports["warm"], "baseline"))}
    arms.update(
        {name: _arm(_score_rows(reports[name], spec["phase"])) for name, spec in SOURCES.items()}
    )
    baseline = arms["unchanged"]["by_context"]["anchor"]
    for arm in arms.values():
        groups = arm["by_context"]
        anchor = groups["anchor"]
        drift = (
            None
            if anchor["nll_mean"] is None or baseline["nll_mean"] is None
            else anchor["nll_mean"] - baseline["nll_mean"]
        )
        arm["criteria"] = dict(
            B_after_phase_success=groups["B"]["accuracy"] == 1,
            A_retained=groups["A"]["accuracy"] == 1,
            anchors_preserved=anchor["accuracy"] == 1 and drift is not None and drift <= 0.1,
        )
    contrasts = {
        name + "_minus_unchanged": _contrast(arms[name], arms["unchanged"]) for name in SOURCES
    }
    for after, before in (("consolidation", "warm"), ("projection", "reset"), ("reset", "warm")):
        contrasts[after + "_minus_" + before] = _contrast(arms[after], arms[before])
    return dict(
        definition="native_consolidation_calibration_summary_v1",
        complete=True,
        arms=arms,
        contrasts=contrasts,
        rank_characterization={
            name: _rank_characterization(reports[name].get(name + "_receipt"), rank)
            for name, rank in (("projection", 4), ("consolidation", 8))
        },
        historical_source_work={name: _historical_work(source) for name, source in reports.items()},
        derived_work=dict(
            model_initializations=0,
            training_updates=0,
            prediction_calls=0,
            random_draws=0,
            scored_rows=30,
            spectrum_modules=sum(
                len(reports[n][n + "_receipt"]["modules"]) for n in ("projection", "consolidation")
            ),
        ),
        comparison_scope="historical_nonconcurrent_controls; warm_source_totals_include_return_beyond_selected_B12; additional_interventions_not_matched_compute",
        confidence_scope="max_probability_five_fixed_bins; provider_margin_confidence_not_used",
        calibration_status="inherited_unrecalibrated; tiny_development_samples_descriptive_only",
        rank_scope="projection_after_A_before_B_vs_consolidation_after_B; correlated_complete_operators_current_head; installed_rank_unmeasured",
        metric_directions=dict(
            accuracy="higher",
            candidate_recall="higher",
            brier_mean="lower",
            nll_mean="lower",
            ece_5bin="lower",
        ),
        recovery="not_tested",
        final_examples=0,
    )


def _reject_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("output must be fresh")
    reports, provenance = {}, {}
    for name, spec in SOURCES.items():
        path = args.runs_dir / spec["path"]
        with path.open("rb") as stream:
            raw = stream.read(MAX_REPORT_BYTES + 1)
        digest = hashlib.sha256(raw).hexdigest()
        if len(raw) > MAX_REPORT_BYTES or digest != spec["sha256"]:
            raise ValueError("saved " + name + " report differs from accepted evidence")
        reports[name] = json.loads(raw, parse_constant=_reject_constant)
        provenance[name] = dict(path=str(path), sha256=digest, bytes=len(raw))
    report = summarize_native_reports(reports)
    report["source_reports"] = provenance
    report["derivation_boundary_rss_bytes"] = read_process_rss_bytes()
    if (
        report["derivation_boundary_rss_bytes"] is not None
        and report["derivation_boundary_rss_bytes"] > 512 * 1024**2
    ):
        raise ValueError("summary process exceeds512MiB RSS boundary")
    encoded = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf8")
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise ValueError("summary exceeds bounded output")
    with args.output.open("xb") as stream:
        stream.write(encoded)
    print(
        json.dumps(
            dict(complete=True, output_bytes=len(encoded), derived_work=report["derived_work"])
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
