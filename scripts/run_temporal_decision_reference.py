"""Fixed learned temporal-tool workflow through the existing delayed stream.

One conventional perceptron, observation-bound rules, frozen controls, no tuning.
Typed tool choices are scored locally; no tool or external action is executed.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import monotonic
from typing import Any

from src.adapters.perceptron_decision import (
    PerceptronDecisionPredictor,
    build_decision_training_batch,
)
from src.adapters.perceptron_learner import PerceptronLearner
from src.app.coupled_attribution import AttributionStream
from src.app.decision_reference_evaluation import (
    DecisionCase,
    DecisionEvaluationFailed,
    _metrics,
    evaluate_decision_reference,
)
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.categorical_perceptron import PerceptronPolicy, PerceptronSnapshot
from src.core.decision_features import DecisionFeatureSpec
from src.core.temporal_decision_sequence import (
    CATALOG,
    CHECKPOINTS,
    PHASES,
    VOCABULARY,
    policy_for_phase,
    temporal_examples,
    validate_temporal_examples,
)
from src.shared.process_memory import ProcessRssSampler

MAX_UPDATES = 12


def _cases(examples):
    validate_temporal_examples(examples)
    return tuple(DecisionCase(e.request, e.target_id, e.source_group, e.role) for e in examples)


def example_data():
    spec = DecisionFeatureSpec(VOCABULARY, CATALOG)
    training = tuple(
        build_decision_training_batch(
            _cases(temporal_examples(phase, "train", exposure=exposure)),
            spec,
            phase,
            allow_subsets=True,
        )
        for phase in PHASES
        for exposure in range(4)
    )
    development = {phase: temporal_examples(phase, "outer_selection") for _, phase in CHECKPOINTS}
    return spec, training, development


def _annotate(evaluation, examples):
    for row, example in zip(evaluation["cases"], examples):
        selected = row["result"]["selected_id"]
        row.update(
            phase=example.phase,
            policy_version=example.policy_version,
            domain=example.domain,
            concept=example.concept,
            obsolete_target_id=example.obsolete_target_id,
            obsolete_target_available=example.obsolete_target_id
            in {c.candidate_id for c in example.request.allowed_candidates},
            obsolete_rule_error=example.obsolete_target_id is not None
            and selected == example.obsolete_target_id,
        )
    rows = evaluation["cases"]
    evaluation["temporal"] = dict(
        policy_version=examples[0].policy_version,
        obsolete_rule_errors=sum(r["obsolete_rule_error"] for r in rows),
        obsolete_rule_opportunities=sum(r["obsolete_target_available"] for r in rows),
        scored_cases=len(rows),
        qualification="retired-tool mask differs from learning a still-allowed revised workflow",
    )
    evaluation["by_domain"] = {
        domain: _metrics([r for r in rows if r["domain"] == domain])
        for domain in ("A", "B")
        if any(r["domain"] == domain for r in rows)
    }
    return evaluation


def _evaluate(spec, state, examples):
    cases = _cases(examples)
    predictor = PerceptronDecisionPredictor(spec, state)
    try:
        evaluation = evaluate_decision_reference(cases, predict=predictor)
    except DecisionEvaluationFailed as error:
        _annotate(error.report, examples)
        raise
    return _annotate(evaluation, examples)


def _accuracy(evaluation, concepts):
    rows = [r for r in evaluation["cases"] if r["concept"] in concepts]
    return sum(r["correct"] for r in rows) / len(rows)


def _metric_difference(after, before):
    result = {key: after[key] - before[key] for key in ("accuracy", "brier_mean", "ece_5bin")}
    finite = after["nll_mean"] is not None and before["nll_mean"] is not None
    result.update(
        nll_mean=after["nll_mean"] - before["nll_mean"] if finite else None,
        nll_status="finite_difference" if finite else "nonfinite_comparison_unavailable",
    )
    return result


def _contrasts(checkpoints):
    points = {r["phase"]: r for r in checkpoints}
    comparisons: dict[str, Any] = {}
    for row in checkpoints:
        learned = row["learned"]
        comparisons[row["phase"]] = {}
        for control in ("unchanged", "frozen_A"):
            if control not in row:
                continue
            other = row[control]
            comparisons[row["phase"]]["learned_minus_" + control] = dict(
                **_metric_difference(learned["metrics"], other["metrics"]),
                current_old_domain_accuracy=_accuracy(learned, ("invoice", "legacy", "password"))
                - _accuracy(other, ("invoice", "legacy", "password")),
                new_domain_accuracy=_accuracy(learned, ("parcel",)) - _accuracy(other, ("parcel",)),
                obsolete_rule_error_count=learned["temporal"]["obsolete_rule_errors"]
                - other["temporal"]["obsolete_rule_errors"],
            )
    a, b, returned = (points[p]["learned"] for p in PHASES)
    password = tuple(_accuracy(e, ("password",)) for e in (a, b, returned))
    return dict(
        by_phase=comparisons,
        phase_differences={
            later + "_minus_" + earlier: dict(
                **_metric_difference(
                    points[later]["learned"]["metrics"], points[earlier]["learned"]["metrics"]
                ),
                same_policy_truth=policy_for_phase(later)["version"]
                == policy_for_phase(earlier)["version"],
                qualification="changed-policy targets across A-to-B are not an old-rule forgetting metric",
            )
            for earlier, later in (
                ("baseline", "after_A"),
                ("after_A", "after_B"),
                ("after_B", "after_return"),
            )
        },
        stable_password_accuracy=dict(zip(PHASES, password, strict=True)),
        stable_password_recovery_demonstrated=password[0] == 1
        and password[1] < 1
        and password[2] == 1,
        current_rule_old_domain_return_success=_accuracy(
            returned, ("invoice", "legacy", "password")
        )
        == 1,
        direction="accuracy higher; NLL/Brier/ECE and obsolete error count lower; no favorable endpoint selection",
        qualification="return uses v2 truth; obsolete v1 retention is error, return success alone is not recovery",
    )


def run_example() -> dict[str, Any]:
    """Execute the complete fixed dose, then score every frozen checkpoint."""
    started = monotonic()
    states: dict[int, PerceptronSnapshot] = {}
    report: dict[str, Any] = dict(
        definition="temporal_linear_decision_reference_v1",
        complete=False,
        checkpoints=[],
        final_examples=0,
        probe_updates=0,
        replay_examples=0,
        qualification="known union tools/vocabulary; synthetic concept-overlapping templates; no generalization/modern-model/CPC or external action claim",
    )
    budget = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=MAX_UPDATES,
            max_wall_seconds=30,
            max_process_rss_bytes=512 * 1024**2,
        ),
        clock=monotonic,
    )
    sampler = ProcessRssSampler()
    try:
        sampler.sample()
        budget.attach_memory(sampler)
        spec, training, development = example_data()
        report["feature_spec"] = asdict(spec)
        report["training_cases"] = [
            [asdict(e) for e in temporal_examples(phase, "train", exposure=exposure)]
            for phase in PHASES
            for exposure in range(4)
        ]
        report["development_cases"] = {
            p: [asdict(e) for e in examples] for p, examples in development.items()
        }

        def capture(update, state, batch):
            states[update] = state

        learner = PerceptronLearner(PerceptronPolicy(len(VOCABULARY), len(CATALOG)))
        stream = AttributionStream(
            learner,
            training,
            (),
            checkpoint_updates=tuple(n for n, _ in CHECKPOINTS),
            label_delay=1,
            budget=budget,
            observe_checkpoint=capture,
            training_only=True,
        )
        stream.run_until(MAX_UPDATES)
        report["stream"] = stream.finish()
        # Why this: no development result can select the TRAIN dose or next rule.
        for update, phase in CHECKPOINTS:
            row: dict[str, Any] = dict(update=update, phase=phase, policy=policy_for_phase(phase))
            report["checkpoints"].append(row)
            sources = dict(learned=update, unchanged=0)
            if phase in ("after_B", "after_return"):
                sources["frozen_A"] = 4
            row["source_updates"] = sources
            for name, source in sources.items():
                budget.before_final()
                try:
                    row[name] = _evaluate(spec, states[source], development[phase])
                except DecisionEvaluationFailed as error:
                    row[name] = error.report
                    raise
                budget.before_final()
        report["contrasts"] = _contrasts(report["checkpoints"])
        report["complete"] = True
    except Exception as error:
        report["error"] = type(error).__name__ + ": " + str(error)[:512]
        if "stream" not in report:
            report["training_failure_evidence"] = (
                "unfinished stream receipts unavailable; original budget counts and captured native prefix retained; unobserved attempted work unknown"
            )
    finally:
        report["snapshots"] = [dict(update=n, state=asdict(s)) for n, s in states.items()]
        evaluations = [
            row[name]
            for row in report["checkpoints"]
            for name in ("learned", "unchanged", "frozen_A")
            if name in row
        ]
        counts = {
            k: sum(e["work"][k] for e in evaluations)
            for k in (
                "prediction_calls_attempted",
                "prediction_calls_returned",
                "prediction_calls_validated",
            )
        }
        last = states[max(states)] if states else None
        report["work"] = dict(
            updates_committed=budget.updates_completed,
            training_rows_committed=3 * budget.updates_completed,
            **counts,
            model_random_draws=0,
            external_actions=0,
            replay_examples=budget.replay_examples_completed,
            latest_observed_native_updates=None if last is None else last.completed_updates,
            latest_observed_mistakes=None if last is None else last.mistake_updates,
            retained_coefficient_bytes=160 * len(states),
            coefficient_bytes_per_snapshot=160,
            committed_training_affine_multiply_terms=60 * budget.updates_completed,
            validated_readout_affine_multiply_terms=15 * counts["prediction_calls_validated"],
            coefficient_update_additions_observed=None
            if last is None
            else 10 * last.mistake_updates,
            elapsed_seconds=monotonic() - started,
            cost_scope="logical D4/K4 TRAIN and D4/K3 masked readout affine terms; nominal script snapshots only; stream/actor/copies/validation/softmax/Python additional; unobserved attempted work unknown on failure",
        )
        if sampler.start_bytes is not None:
            sampler.sample()
            report["work"]["process_rss_observation"] = asdict(sampler.snapshot())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("output must be fresh")
    report = run_example()
    encoded = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    if len(encoded) > 256 * 1024:
        raise ValueError("temporal report exceeds256KiB")
    with args.output.open("xb") as stream:
        stream.write(encoded)
    if not report["complete"]:
        raise SystemExit(1)
    print(json.dumps(dict(complete=True, output_bytes=len(encoded), work=report["work"])))


if __name__ == "__main__":
    main()
