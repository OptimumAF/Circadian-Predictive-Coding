"""Run one bounded TRAIN-only BP diagnostic from saved TRAIN source episodes."""

import argparse
import json
from pathlib import Path
from time import monotonic

from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeaturePolicy,
)
from src.adapters.torch_bp_update_diagnostics import diagnose_bp_updates
from src.adapters.minigrid_train_catalogue import _validate_sources
from src.adapters.torch_feature_learner import TorchFeaturePolicy
from src.adapters.torch_recurrent_gradient_diagnostics import GradientDiagnosticFailed
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    episode_arrays,
)
from src.shared.torch_runtime import require_torch


def _training_sources(path: Path) -> dict[str, DemonstrationEpisode]:
    if path.stat().st_size > 8 * 1024**2:
        raise ValueError("saved TRAIN source report exceeds8MiB")
    records = json.loads(path.read_text(encoding="utf8"))["training_sources"]
    expected = [
        f"minigrid-source-{context}-{cue}"
        for context in ("A", "B", "C_train")
        for cue in ("blue", "red")
    ]
    if type(records) is not list or [record["episode_id"] for record in records] != expected:
        raise ValueError("expected original six TRAIN source records in order")
    result = {}
    for record in records:
        if record["role"] != "train":
            raise ValueError("saved diagnostic sources must remain TRAIN")
        initial = tuple(record["transitions"][0]["observation"])
        if len(initial) != 148:
            raise ValueError("expected raw148 source frames before history construction")
        result[record["episode_id"]] = DemonstrationEpisode(
            record["episode_id"],
            record["context"],
            record["role"],
            tuple(
                DemonstrationTransition(
                    row["step"],
                    initial + tuple(row["observation"]),
                    row["action"],
                    row["reward"],
                    initial + tuple(row["next_observation"]),
                    row["terminal"],
                    row["num_classes"],
                )
                for row in record["transitions"]
            ),
            record["num_classes"],
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-report", type=Path)
    parser.add_argument("--describe", action="store_true")
    options = parser.parse_args()
    if options.describe:
        print(
            json.dumps(
                dict(
                    classification="engineering_TRAIN_counterfactual_not_scientific_F_replay_explanation",
                    recipe="original invariant BP296/K7/H8/seed17/lr.03/momentum.9",
                    warmup="one A-blue TRAIN update on a new disposable source; not historical joint128 state",
                    cycle=["B-blue TRAIN wake", "A-blue TRAIN replay", "A-red TRAIN replay"],
                    native_calls=4,
                    native_training_rows=36,
                    replay_rows=18,
                    published_updates=0,
                    extra_diagnostic_forwards=15,
                    extra_diagnostic_forward_rows=135,
                    extra_diagnostic_backwards=9,
                    extra_diagnostic_backward_rows=81,
                    final_data_used=False,
                    probe_data_used=False,
                ),
                indent=2,
            )
        )
        return
    if options.train_report is None:
        parser.error("--train-report is required for a bounded native diagnostic")
    prefix = dict(
        warmup_work=dict(
            native_train_calls_attempted=0,
            native_train_calls_returned=0,
            native_training_rows_attempted=0,
            native_training_rows_returned=0,
            charged_updates=0,
            failure_work_may_be_unobserved=False,
        ),
        source_admission_work=dict(
            validation_projection_rows=0, validation_projected_array_equivalent_bytes=0
        ),
    )
    try:
        sources = _training_sources(options.train_report)
        _validate_sources(tuple(sources.values()), 1)
        prefix["source_admission_work"].update(
            validation_projection_rows=108, validation_projected_array_equivalent_bytes=255744
        )
        old_a = tuple(sources[f"minigrid-source-A-{cue}"] for cue in ("blue", "red"))
        wake = sources["minigrid-source-B-blue"]
        torch = require_torch()
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        source = InvariantTorchFeatureLearner(
            InvariantTorchFeaturePolicy(TorchFeaturePolicy("bp", 296, 8, 17, num_classes=7))
        )
        warmup = ToyBudgetSession(
            ToyExecutionBudget(max_training_updates=1, max_wall_seconds=30), monotonic
        )
        warmup.before_update()
        warmup.record_update()
        prefix["warmup_work"].update(
            charged_updates=warmup.updates_completed,
            native_train_calls_attempted=1,
            native_training_rows_attempted=9,
        )
        source.train_batch(*episode_arrays(old_a[0]))
        prefix["warmup_work"].update(native_train_calls_returned=1, native_training_rows_returned=9)
        original = source.snapshot_state()
        budget = ToyBudgetSession(
            ToyExecutionBudget(max_training_updates=3, max_replay_examples=18, max_wall_seconds=30),
            monotonic,
        )
        report = diagnose_bp_updates(original, wake, old_a, budget=budget)
        report.update(prefix)
        report["scope"] = (
            "engineering warmup and TRAIN cycle only; no joint128 replay, probe/final use, acquisition or causal scientific claim"
        )
    except GradientDiagnosticFailed as error:
        error.report.update(prefix)
        print(json.dumps(error.report, indent=2, allow_nan=False))
        raise SystemExit(2) from error
    except Exception as error:
        if prefix["warmup_work"]["native_train_calls_attempted"]:
            prefix["warmup_work"]["failure_work_may_be_unobserved"] = True
            print(
                json.dumps(
                    dict(status="incomplete", reason=f"{type(error).__name__}: {error}", **prefix),
                    indent=2,
                    allow_nan=False,
                )
            )
            raise SystemExit(2) from error
        parser.error(f"BP diagnostic refused: {error}")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
