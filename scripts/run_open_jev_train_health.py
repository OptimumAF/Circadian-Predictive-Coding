"""Observe seven fixed TRAIN updates, with no inference or adaptation selection.

Composes the existing inbox/delivery and native owner. Reports arrived-TRAIN
group magnitudes; does not implement a probe, learner fork or comparison runner.
"""

import argparse
import hashlib
import importlib
import json
from math import isfinite, sqrt
from pathlib import Path
import sys
from time import monotonic
from typing import Any

from scripts.run_open_jev_b_control import load_reference, validate_initial_weights
from scripts.run_open_jev_native_smoke import _load
from scripts.run_open_jev_native_update import _weights, _weight_summary
from scripts.run_open_jev_optimizer_control import validate_after_A_weights
from scripts.run_open_jev_sequence import _deliver_exposure, _deliver_phase
from src.adapters.open_jev_training import OpenJevTrainingLearner, OpenJevTrainingPolicy
from src.adapters.open_jev_update_health import DEFINITION, GROUPS
from src.app.experience_inbox import ExperienceInbox
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.experience import LogicalClock
from src.core.native_choice_sequence import (
    INSTRUCTIONS,
    sequence_examples,
    validate_sequence_examples,
)

MAX_UPDATES = 7


def _nonnegative(value):
    if type(value) not in (int, float) or not isfinite(value) or value < 0:
        raise ValueError("diagnosis requires finite nonnegative measurements")
    return value


def summarize_update_health(delivery) -> list[dict[str, Any]]:
    """Scalar report only; never choose subsequent work from a measured norm."""
    if type(delivery) is not list or len(delivery) != MAX_UPDATES:
        raise ValueError("diagnosis requires exactly seven paid TRAIN rows")
    result = []
    for number, row in enumerate(delivery, 1):
        if (
            type(row["receipt"]["update_number"]) is not int
            or row["receipt"]["update_number"] != number
        ):
            raise ValueError("diagnostic receipt update sequence changed")
        update = row["training"]
        loss = _nonnegative(update["loss"])
        norm = _nonnegative(update["gradient_norm_before_clip"])
        health = update["update_health"]
        if health["definition"] != DEFINITION or set(health["groups"]) != set(GROUPS):
            raise ValueError("diagnosis requires the native group-health definition")
        groups = {}
        for group, values in health["groups"].items():
            counts = (
                "parameter_tensors",
                "parameter_elements",
                "gradient_tensors",
                "missing_gradient_tensors",
            )
            if any(type(values[key]) is not int or values[key] < 0 for key in counts):
                raise ValueError("diagnostic group counts must be nonnegative integers")
            elements, tensors = values["parameter_elements"], values["parameter_tensors"]
            if (
                elements < tensors
                or tensors < 1
                or values["gradient_tensors"] + values["missing_gradient_tensors"] != tensors
            ):
                raise ValueError("diagnostic group counts disagree")
            gradient = _nonnegative(values["gradient_l2_before_clip"])
            step = _nonnegative(values["parameter_step_l2"])
            groups[group] = dict(
                values,
                # Missing gradients contribute no measured gradient; retain
                # missing counts and normalize over all owned coordinates.
                gradient_rms_before_clip=gradient / sqrt(elements),
                parameter_step_rms=step / sqrt(elements),
            )
        result.append(
            dict(
                update_number=number,
                source_example_id=row["source_example_id"],
                loss=loss,
                gradient_norm_before_clip=norm,
                clipping_scale_from_reported_norm=min(1.0, 1.0 / (norm + 1e-6)),
                groups=groups,
            )
        )
    return result


def run_train_health(learner, report, progress) -> None:
    """A6, required binding callback, B1; all labels use the ordinary inbox."""
    examples = sequence_examples()
    validate_sequence_examples(examples)
    if (
        set(learner.work)
        != {
            "training_attempts",
            "forward_calls",
            "backward_calls",
            "optimizer_steps",
            "committed_updates",
        }
        or any(type(v) is not int or v != 0 for v in learner.work.values())
        or learner.last_update is not None
        or learner.optimizer_reset_work
        != dict(optimizer_reset_attempts=0, optimizer_resets_completed=0)
        or any(type(v) is not int or v != 0 for v in learner.update_health_work.values())
    ):
        raise ValueError("TRAIN diagnosis requires a fresh zero-work native learner")
    report.update(delivery=[], diagnostic_complete=False)
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(MAX_UPDATES, 600), clock=monotonic)
    inbox = ExperienceInbox(
        learner,
        clock=clock,
        budget=budget,
        learner_version="native-rank8",
        max_experiences=MAX_UPDATES,
    )
    try:
        _deliver_phase(learner, inbox, clock, examples, "after_A", "A", report, progress)
        progress("before_B_update")
        first_b = next(e for e in examples if e.role == "train" and e.context == "B")
        _deliver_exposure(
            learner,
            inbox,
            clock,
            first_b,
            "first_B-train-B-0",
            "first_B",
            report,
            progress,
        )
        if budget.updates_completed != MAX_UPDATES:
            raise RuntimeError("diagnosis did not complete seven TRAIN updates")
        report["update_diagnostics"] = summarize_update_health(report["delivery"])
        report["diagnostic_complete"] = True
        progress("diagnostic_validated")
    finally:
        report["native_work"] = learner.work
        report["optimizer_reset_work"] = learner.optimizer_reset_work
        report["update_health_work"] = learner.update_health_work
        report["app_committed_updates"] = budget.updates_completed
        report["inbox_stopped"] = inbox.stopped


def validate_completed_work(report, counts) -> None:
    """Check the fixed observation's paid work before claiming completion."""
    if (
        set(report["native_work"].values()) != {7}
        or report["optimizer_reset_work"]
        != dict(optimizer_reset_attempts=0, optimizer_resets_completed=0)
        or report["app_committed_updates"] != 7
        or report["inbox_stopped"]
    ):
        raise ValueError("TRAIN diagnosis native work does not match seven updates")
    health = report["update_health_work"]
    elements = sum(v["elements"] for v in report["initial_binding"]["weights"].values())
    expected = dict(
        update_health_attempts=7,
        parameter_copy_elements_attempted=7 * elements,
        gradient_reduction_elements_attempted=7 * elements,
        step_reduction_elements_attempted=7 * elements,
        update_health_reports_completed=7,
    )
    if not 1 <= elements <= 5_000_000 or health != expected:
        raise ValueError("TRAIN diagnosis health work does not match owned parameters")
    lengths = counts["candidate_sequence_lengths"]
    if (
        counts["decision_forward_calls"] != 7
        or counts["scorer_calls"] != 0
        or counts["native_prediction_calls_attempted"] != 0
        or counts["native_prediction_calls_returned"] != 0
        or counts["initializer_cpu_elements"] > 1
        or counts["initializer_meta_elements"] > 511147008
        or len(lengths) != 21
        or any(type(n) is not int or not 1 <= n <= 256 for n in lengths)
        or sum(lengths) > 5376
    ):
        raise ValueError("TRAIN diagnosis loader observations exceed the fixed protocol")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("reference", "source", "checkpoint", "output"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output exists; preserve previous evidence")
    with args.output.open("x", encoding="utf8") as stream:
        stream.write("{}\n")
    report: dict[str, Any] = dict(
        definition="native_train_update_health_v1",
        complete=False,
        model_load_attempts=0,
        events=[],
        seed=17,
        training_temperature=1,
        prediction_calls=0,
        replay_examples=0,
        probe_updates=0,
        final_examples=0,
        checkpoint_fingerprint_scope="original_load_birth_provenance",
        training_random_draws="unknown",
        checkpoint_recomputation_work="unmeasured_internal_layers",
        clipping_scale_arithmetic="float64_reconstruction_not_bit_exact_tensor_coefficient",
    )
    owner: dict[str, Any] = {}
    started = monotonic()

    def progress(event):
        import resource

        event = dict(event=event) if type(event) is str else event
        event.update(
            outer_seconds=monotonic() - started,
            outer_peak_rss_bytes=getattr(resource, "getrusage")(
                getattr(resource, "RUSAGE_SELF")
            ).ru_maxrss
            * 1024,
        )
        report["events"].append(event)
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )
        if event["outer_seconds"] > 600 or event["outer_peak_rss_bytes"] > 20 * 1024**3:
            raise RuntimeError("TRAIN diagnosis resource allowance exceeded")
        if event["event"] == "before_B_update":
            learner = owner["learner"]
            learner._guard()
            weights = _weight_summary(learner.model, owner["before"])
            validate_after_A_weights(weights, owner["reference"])
            if learner.model.training or set(learner.work.values()) != {6}:
                raise ValueError("after-A diagnostic work/mode differs from fixed prefix")
            report["after_A_binding"] = dict(weights=weights, native_work=learner.work)

    def observe_model(model):
        api = importlib.import_module("jev.api")
        learner = OpenJevTrainingLearner(
            model,
            api.compile_request,
            instructions=INSTRUCTIONS,
            policy=OpenJevTrainingPolicy(MAX_UPDATES, 8, record_update_health=True),
        )
        owner["learner"], owner["before"] = learner, _weights(model)
        weights = _weight_summary(model, owner["before"])
        validate_initial_weights(weights, owner["reference"])
        rng = hashlib.sha256(learner._rng.numpy().tobytes()).hexdigest()
        if (
            model.training
            or learner._optimizer.state_dict()["state"]
            or rng != owner["reference"]["initial_rng_sha256"]
        ):
            raise ValueError("diagnostic initial optimizer/mode/RNG differs from reference")
        report["initial_binding"] = dict(
            weights=weights,
            optimizer_empty=True,
            model_training=False,
            private_rng_sha256=rng,
        )
        progress("initial_state_bound")

    try:
        reference = load_reference(args.reference, include_after_A=True)
        # Why: reuse existing bounded admission, then expose only state bindings
        # to this runner. Development endpoints never choose its fixed dose.
        owner["reference"] = {
            k: reference[k]
            for k in (
                "source_sha256",
                "baseline_weights",
                "after_A_weights",
                "initial_rng_sha256",
            )
        }
        report["reference_sha256"] = reference["source_sha256"]
        report["model_load_attempts"] = 1
        # Positive API minimum reserves one unused prediction; no callable is
        # given to the TRAIN runner. The actual allowance is zero predictions.
        _load(
            args.source,
            args.checkpoint,
            progress,
            prediction_limit=1,
            forward_limit=7,
            token_limit=5376,
            model_observer=observe_model,
            work_reporter=lambda reporter: owner.update(observe_work=reporter),
        )
        run_train_health(owner["learner"], report, progress)
        learner = owner["learner"]
        learner._guard()
        owner["observe_work"]()
        observed = [e["observed_work"] for e in report["events"] if "observed_work" in e]
        if not observed or any(
            e["native_prediction_calls_attempted"] or e["scorer_calls"] for e in observed
        ):
            raise RuntimeError("TRAIN diagnosis performed prediction work")
        validate_completed_work(report, observed[-1])
        report["final_weights"] = _weight_summary(learner.model, owner["before"])
        report["model_training_after"] = learner.model.training
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if "learner" in owner:
            learner = owner["learner"]
            report.update(
                native_work=learner.work,
                optimizer_reset_work=learner.optimizer_reset_work,
                update_health_work=learner.update_health_work,
            )
        if "observe_work" in owner and not report["complete"]:
            try:
                owner["observe_work"]()
            except Exception as error:
                report["final_observation_error"] = f"{type(error).__name__}: {error}"
        sys.setprofile(None)
        report["outer_seconds"] = monotonic() - started
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
