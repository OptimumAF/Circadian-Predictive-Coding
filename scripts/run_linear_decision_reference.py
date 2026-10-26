"""Frozen six-update linear routing example through the existing TRAIN stream."""

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
    evaluate_decision_reference,
    _metrics,
)
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.categorical_perceptron import PerceptronPolicy, PerceptronSnapshot
from src.core.decision_features import DecisionFeatureSpec
from src.core.typed_decision import DecisionCandidate, DecisionRequest
from src.shared.process_memory import ProcessRssSampler


def example_data():
    catalog = (
        DecisionCandidate("billing", "invoice billing payment refund"),
        DecisionCandidate("password", "password reset login account"),
        DecisionCandidate("shipping", "shipping parcel delivery tracking"),
    )
    spec = DecisionFeatureSpec(
        ("credentials", "invoice", "parcel", "password", "refund", "tracking"), catalog
    )
    words = {"A": ("invoice", "password", "parcel"), "B": ("refund", "credentials", "tracking")}
    training = []
    for index, context in enumerate(("A", "A", "B", "B", "A", "A")):
        cases = tuple(
            DecisionCase(
                DecisionRequest(f"train-{index}-{item.candidate_id}", word, catalog),
                item.candidate_id,
                f"train-{context}",
                "train",
            )
            for item, word in zip(catalog, words[context], strict=True)
        )
        training.append(build_decision_training_batch(cases, spec, context))
    development = tuple(
        DecisionCase(
            DecisionRequest(f"dev-{context}-{item.candidate_id}", word + " please", catalog),
            item.candidate_id,
            f"dev-{context}",
        )
        for context in ("A", "B")
        for item, word in zip(catalog, words[context], strict=True)
    )
    return spec, tuple(training), development


def _contrasts(checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    by_update = {row["update"]: row["evaluation"] for row in checkpoints}

    def accuracy(update, group):
        rows = [row for row in by_update[update]["cases"] if row["source_group"] == group]
        return sum(row["correct"] for row in rows) / len(rows)

    return dict(
        definition="signed accuracy difference; higher is favorable; all fixed checkpoints retained",
        new_task_gain=accuracy(4, "dev-B") - accuracy(2, "dev-B"),
        old_task_retention=accuracy(4, "dev-A") - accuracy(2, "dev-A"),
        return_recovery=accuracy(6, "dev-A") - accuracy(4, "dev-A"),
    )


def run_example() -> dict[str, Any]:
    """Run once; preserve ordinary failures and observed work without retrying."""
    started = monotonic()
    states: dict[int, PerceptronSnapshot] = {}
    report: dict[str, Any] = dict(
        definition="typed_linear_routing_v1",
        complete=False,
        checkpoints=[],
        qualification="fixed overlapping-vocabulary development control; not generalization, calibration, modern-model or CPC advantage; no final data",
    )
    sampler = ProcessRssSampler()
    budget = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=6, max_wall_seconds=30, max_process_rss_bytes=512 * 1024**2
        ),
        clock=monotonic,
    )
    try:
        sampler.sample()
        budget.attach_memory(sampler)
        spec, training, development = example_data()
        report["feature_spec"] = asdict(spec)

        def capture(update, state, batch):
            states[update] = state

        learner = PerceptronLearner(PerceptronPolicy(len(spec.vocabulary), len(spec.catalog)))
        stream = AttributionStream(
            learner,
            training,
            (),
            checkpoint_updates=(0, 2, 4, 6),
            label_delay=1,
            budget=budget,
            observe_checkpoint=capture,
            training_only=True,
        )
        stream.run_until(6)
        report["stream"] = stream.finish()
        # Why this: development labels are first scored after every TRAIN update.
        for update, state in states.items():
            budget.before_final()
            row: dict[str, Any] = dict(update=update)
            report["checkpoints"].append(row)
            try:
                row["evaluation"] = evaluate_decision_reference(
                    development, predict=PerceptronDecisionPredictor(spec, state)
                )
                row["by_group"] = {
                    group: _metrics(
                        [
                            item
                            for item in row["evaluation"]["cases"]
                            if item["source_group"] == group
                        ]
                    )
                    for group in ("dev-A", "dev-B")
                }
            except DecisionEvaluationFailed as error:
                row["evaluation"] = error.report
                raise
            budget.before_final()
        report["accuracy_contrasts"] = _contrasts(report["checkpoints"])
        report["complete"] = True
    except Exception as error:
        report["error"] = type(error).__name__ + ": " + str(error)[:512]
        if "stream" not in report:
            report["training_failure_evidence"] = dict(
                receipts_and_diagnostics="unavailable: TRAIN stream did not finish; committed budget counts and selected native checkpoint prefix retained",
                qualification="latest checkpoint mistakes may omit later committed or failed-attempt work",
            )
    finally:
        report["snapshots"] = [
            dict(update=point, state=asdict(state)) for point, state in states.items()
        ]
        report["work"] = dict(
            updates_committed=budget.updates_completed,
            training_rows_committed=3 * budget.updates_completed,
            attempted_uncommitted_training_work="unknown on failure; entry meter is external",
            replay_examples=budget.replay_examples_completed,
            coefficient_bytes_per_snapshot=168,  # six fixed features plus native bias, three classes
            retained_coefficient_bytes=168 * len(states),
            memory_scope="nominal float64 coefficients only; excludes Python/array/runtime overhead",
            elapsed_seconds=monotonic() - started,
        )
        observed_point = max(states, default=None)
        mistakes = None if observed_point is None else states[observed_point].mistake_updates
        validated = sum(
            row.get("evaluation", {}).get("work", {}).get("prediction_calls_validated", 0)
            for row in report["checkpoints"]
        )
        report["work"].update(
            latest_observed_mistakes=mistakes,
            latest_observed_mistake_update=observed_point,
            coefficient_update_additions_observed=None if mistakes is None else 14 * mistakes,
            committed_training_affine_multiply_terms=21 * 3 * budget.updates_completed,
            validated_readout_affine_multiply_terms=21 * validated,
            arithmetic_scope="fixed D6/K3 affine coefficient multiply terms include bias; two column additions per mistaken row; excludes comparisons, exp, normalization, copies, validation and metrics; failed/unvalidated work may be unknown",
        )
        if sampler.start_bytes is not None:
            sampler.sample()
            report["work"]["process_rss_observation"] = asdict(sampler.snapshot())
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="fresh JSON path; otherwise stdout")
    args = parser.parse_args(argv)
    try:
        if args.output is not None and args.output.exists():
            raise FileExistsError(f"output already exists: {args.output}")
        report = run_example()
        encoded = json.dumps(report, allow_nan=False, indent=2)
        if args.output is None:
            print(encoded)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf8") as handle:
                handle.write(encoded + "\n")
        if not report["complete"]:
            raise SystemExit(1)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
