"""Cold-start B control through existing native sequence/inbox helpers.

Reuses an accepted development baseline after initial-state binding. No prior A,
final/probe labels, replay, outcome-based tuning, actor fork or saved checkpoint.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import sys
from time import monotonic
from typing import Any

from scripts.run_open_jev_native_smoke import TEMPERATURE, _load
from scripts.run_open_jev_native_update import _weights, _weight_summary, record_readout
from scripts.run_open_jev_sequence import _deliver_phase, _difference, _group_metrics
from src.adapters.open_jev_training import OpenJevTrainingLearner, OpenJevTrainingPolicy
from src.app.experience_inbox import ExperienceInbox
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.experience import LogicalClock
from src.core.native_choice_sequence import (
    INSTRUCTIONS,
    sequence_examples,
    validate_sequence_examples,
)

REFERENCE_SHA256 = "3f680ee3bbc93ff0dfc144f7abe322317a034290460b8c546d198b8f1646ad58"
MAX_REFERENCE_BYTES = 1024 * 1024
MAX_UPDATES = 6


def load_reference(
    path: Path, *, include_after_A: bool = False, include_after_B: bool = False
) -> dict[str, Any]:
    """Bind one accepted report, not an arbitrary or recursively loaded archive."""
    with path.open("rb") as stream:
        body = stream.read(MAX_REFERENCE_BYTES + 1)
    digest = hashlib.sha256(body).hexdigest()
    if len(body) > MAX_REFERENCE_BYTES or digest != REFERENCE_SHA256:
        raise ValueError("reference differs from the accepted recurring study")
    source = json.loads(body)
    if (
        type(source) is not dict
        or source.get("definition") != "native_choice_recurring_v1"
        or source.get("complete") is not True
        or source.get("sequence_complete") is not True
        or source.get("app_committed_updates") != 18
        or source.get("examples")
        != json.loads(json.dumps([asdict(e) for e in sequence_examples()]))
    ):
        raise ValueError("reference is not the complete fixed recurring study")
    baseline = [r for r in source["readouts"] if r["phase"] == "baseline"]
    _validate_readouts(baseline, "baseline")
    initial = source["weight_checkpoints"][0]
    if initial["after_updates"] != 0 or initial["model_training"] is not False:
        raise ValueError("reference lacks a cold evaluation state")
    rng = next(
        e["rng_state_sha256"] for e in source["events"] if e["event"] == "prediction_attempt"
    )
    reference = dict(
        source_sha256=digest,
        baseline_readouts=baseline,
        baseline_summary=source["summary"]["by_phase"]["baseline"],
        sequential_summary=source["summary"]["by_phase"]["after_B"],
        sequential_B_acquisition=source["summary"]["contrasts"]["phase_deltas"]["B_acquisition"],
        baseline_weights=initial["weights"],
        initial_rng_sha256=rng,
    )
    if include_after_A:
        after = next(c for c in source["weight_checkpoints"] if c["after_updates"] == 6)
        if after["model_training"] is not False:
            raise ValueError("reference lacks the after-A evaluation state")
        reference.update(
            after_A_weights=after["weights"],
            after_A_summary=source["summary"]["by_phase"]["after_A"],
        )
    if include_after_B:
        after_B = [c for c in source["weight_checkpoints"] if c["after_updates"] == 12]
        if len(after_B) != 1 or after_B[0]["model_training"] is not False:
            raise ValueError("reference lacks the unique after-B evaluation state")
        reference["after_B_weights"] = after_B[0]["weights"]
    return deepcopy(reference)


def _validate_readouts(readouts, phase):
    expected = {e.example_id: e for e in sequence_examples() if e.role == "outer_selection"}
    if (
        type(readouts) is not list
        or len(readouts) != 6
        or {r.get("example_id") for r in readouts} != expected.keys()
    ):
        raise ValueError("control requires exactly the six declared development readouts")
    for row in readouts:
        example = expected[row["example_id"]]
        # Saved reports have JSON lists; current dataclass projections retain
        # tuples. Round-trip both without losing values or candidate order.
        if json.loads(json.dumps(row.get("request"), allow_nan=False)) != json.loads(
            json.dumps(asdict(example.request), allow_nan=False)
        ):
            raise ValueError("control request differs from the declared development case")
        if any(
            row.get(key) != value
            for key, value in dict(
                phase=phase,
                context=example.context,
                role="outer_selection",
                target=example.target_id,
                source_group=example.source_group,
            ).items()
        ):
            raise ValueError("control readout differs from the declared development case")


def validate_initial_weights(observed_summary, reference) -> None:
    expected = reference["baseline_weights"]
    if (
        type(observed_summary) is not dict
        or not expected
        or observed_summary.keys() != expected.keys()
    ):
        raise ValueError("initial trainable parameter keys differ from baseline")
    for name, row in observed_summary.items():
        origin = expected[name]
        if not (
            row["before_sha256"]
            == row["after_sha256"]
            == origin["before_sha256"]
            == origin["after_sha256"]
            and row["elements"] == origin["elements"]
            and row["max_absolute_delta"] == row["l2_delta"] == 0
        ):
            raise ValueError("initial trainable weights differ from accepted baseline")


def summarize_b_control(readouts, reference) -> dict[str, Any]:
    _validate_readouts(readouts, "after_cold_B")
    groups = {
        context: _group_metrics([r for r in readouts if r["context"] == context])
        for context in ("A", "B", "anchor")
    }
    baseline, sequential = reference["baseline_summary"], reference["sequential_summary"]
    anchor, original = groups["anchor"], baseline["anchor"]
    return dict(
        by_context=groups,
        baseline_reused=deepcopy(baseline),
        sequential_after_B=deepcopy(sequential),
        contrasts=dict(
            cold_minus_baseline={c: _difference(groups[c], baseline[c]) for c in groups},
            cold_minus_sequential_after_B={
                c: _difference(groups[c], sequential[c]) for c in groups
            },
            sequential_B_acquisition=deepcopy(reference["sequential_B_acquisition"]),
        ),
        criteria=dict(
            B_after_phase_success=groups["B"]["accuracy"] == 1,
            anchors_preserved=(
                anchor["accuracy"] == 1
                and anchor["nll_status"] == original["nll_status"] == "finite"
                and anchor["nll_mean"] - original["nll_mean"] <= 0.1
            ),
        ),
        A_interpretation="transfer_without_A_training_not_retention",
        comparison_scope="same_B_exposures_different_prior_history_and_total_compute",
    )


def run_b_control(learner, predict, reference, report: dict[str, Any], progress) -> None:
    examples = sequence_examples()
    validate_sequence_examples(examples)
    reference = deepcopy(reference)
    _validate_readouts(reference["baseline_readouts"], "baseline")
    if (
        set(learner.work)
        != {
            "training_attempts",
            "forward_calls",
            "backward_calls",
            "optimizer_steps",
            "committed_updates",
        }
        or any(type(value) is not int or value != 0 for value in learner.work.values())
        or learner.last_update is not None
    ):
        raise ValueError("cold control requires fresh zero-work native learner")
    report.update(
        reference_sha256=reference["source_sha256"],
        reused_baseline=reference["baseline_readouts"],
        examples=[asdict(e) for e in examples],
        readouts=[],
        delivery=[],
        control_complete=False,
    )
    clock = LogicalClock()
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=MAX_UPDATES, max_wall_seconds=600), monotonic
    )
    inbox = ExperienceInbox(
        learner,
        clock=clock,
        budget=budget,
        learner_version="native-rank8",
        max_experiences=MAX_UPDATES,
    )
    try:
        _deliver_phase(learner, inbox, clock, examples, "cold_B", "B", report, progress)
        if budget.updates_completed != MAX_UPDATES:
            raise RuntimeError("cold control did not complete exactly six B updates")
        for example in (e for e in examples if e.role == "outer_selection"):
            row = record_readout(
                example.request,
                predict,
                report["readouts"],
                progress,
                phase="after_cold_B",
                context=example.context,
                target=example.target_id,
            )
            row.update(example_id=example.example_id, source_group=example.source_group)
        report["summary"] = summarize_b_control(report["readouts"], reference)
        report["control_complete"] = True
        progress("control_validated")
    finally:
        report["native_work"] = learner.work
        report["app_committed_updates"] = budget.updates_completed
        report["inbox_stopped"] = inbox.stopped


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("reference", "source", "checkpoint", "output"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output exists; preserve previous evidence")
    report: dict[str, Any] = dict(
        definition="native_cold_B_control_v1",
        complete=False,
        model_load_attempts=0,
        events=[],
        seed=17,
        prior_A_updates=0,
        reused_baseline_prediction_calls=0,
        training_temperature=1,
        inference_temperature=TEMPERATURE,
        calibration_status="inherited_unrecalibrated",
        training_random_draws="unknown",
        checkpoint_recomputation_work="unmeasured_internal_layers",
        final_examples=0,
        probe_updates=0,
        replay_examples=0,
        checkpoint_fingerprint_scope="original_load_birth_provenance",
    )
    with args.output.open("x", encoding="utf8") as stream:
        stream.write("{}\n")
    owner: dict[str, Any] = {}
    started = monotonic()

    def progress(event):
        import resource

        event = dict(event=event) if type(event) is str else event
        event["outer_seconds"] = monotonic() - started
        event["outer_peak_rss_bytes"] = (
            getattr(resource, "getrusage")(getattr(resource, "RUSAGE_SELF")).ru_maxrss * 1024
        )
        report["events"].append(event)
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )
        if event["outer_seconds"] > 600 or event["outer_peak_rss_bytes"] > 20 * 1024**3:
            raise RuntimeError("cold control resource allowance exceeded")

    def observe_model(model):
        api = importlib.import_module("jev.api")
        learner = OpenJevTrainingLearner(
            model,
            api.compile_request,
            instructions=INSTRUCTIONS,
            policy=OpenJevTrainingPolicy(MAX_UPDATES, 8),
        )
        owner["learner"], owner["before"] = learner, _weights(model)
        observed = _weight_summary(model, owner["before"])
        validate_initial_weights(observed, owner["reference"])
        rng = hashlib.sha256(learner._rng.numpy().tobytes()).hexdigest()
        if (
            model.training
            or learner._optimizer.state_dict()["state"]
            or rng != owner["reference"]["initial_rng_sha256"]
        ):
            raise ValueError("cold optimizer/mode/RNG differs from accepted initial state")
        report["initial_binding"] = dict(
            weights=observed, optimizer_empty=True, model_training=False, private_rng_sha256=rng
        )
        progress("initial_state_bound")

    try:
        owner["reference"] = load_reference(args.reference)
        validate_sequence_examples(sequence_examples())
        report["reference_sha256"] = owner["reference"]["source_sha256"]
        report["model_load_attempts"] = 1
        predict = _load(
            args.source,
            args.checkpoint,
            progress,
            prediction_limit=6,
            forward_limit=24,
            token_limit=9216,
            model_observer=observe_model,
        )
        run_b_control(owner["learner"], predict, owner["reference"], report, progress)
        learner = owner["learner"]
        learner._guard()
        report["final_weights"] = _weight_summary(learner.model, owner["before"])
        report["model_training_after"] = learner.model.training
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        if "learner" in owner:
            report["native_work"] = owner["learner"].work
    finally:
        sys.setprofile(None)
        report["outer_seconds"] = monotonic() - started
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
