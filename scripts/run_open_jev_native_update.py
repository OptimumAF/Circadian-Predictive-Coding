"""One prospectively bounded native TRAIN update with old/new readouts.

Offline CPU only. Reuses the native loader, compiler, learner port and inbox.
No tuning, replay, final/probe data, actor promotion or portable training resume.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib
import json
from math import log
from pathlib import Path
import sys
from time import monotonic
from typing import Any

from scripts.run_open_jev_native_smoke import TEMPERATURE, _load, _validate_metadata
from src.adapters.open_jev_choice import (
    MAX_REPLY_BYTES,
    build_open_jev_choice,
    decode_open_jev_probabilities,
)
from src.adapters.open_jev_training import OpenJevTrainingLearner
from src.app.experience_inbox import ExperienceInbox
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.typed_decision import DecisionCandidate, DecisionRequest

INSTRUCTIONS = "Choose the customer's support intent."
CATALOG = (
    DecisionCandidate("billing", "invoice billing payment refund"),
    DecisionCandidate("password", "password reset login account"),
    DecisionCandidate("shipping", "shipping parcel delivery tracking"),
)


def record_readout(
    request, predict, readouts, progress, *, phase, context, target, instructions=INSTRUCTIONS
):
    """Capture a paid native reply before validation; supply no target on wire."""
    wire = build_open_jev_choice(request, instructions)
    birth_wire = deepcopy(wire)
    row: dict[str, Any] = dict(
        phase=phase, context=context, role="outer_selection", request=asdict(request)
    )
    readouts.append(row)
    progress("readout_attempt")
    reply = predict(wire)
    try:
        raw = json.dumps(reply, allow_nan=True, separators=(",", ":"))
        if len(raw.encode()) > MAX_REPLY_BYTES:
            raise ValueError("native response exceeds capture bound")
        row["response_json"] = raw
        body = json.dumps(reply, allow_nan=False, separators=(",", ":")).encode()
        row["response"] = json.loads(body)
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        row["response_unavailable"] = type(error).__name__
        raise ValueError("native return is not bounded strict JSON") from error
    progress("readout_return")
    if wire != birth_wire or list(wire["questions"]["decision"]["criteria"]) != list(
        birth_wire["questions"]["decision"]["criteria"]
    ):
        raise ValueError("predictor changed frozen request")
    result = decode_open_jev_probabilities(request, body)
    _validate_metadata(reply, candidate_count=3)
    probability = dict(result.probabilities)[target]
    row.update(
        result=asdict(result),
        target=target,
        nll=None if probability == 0 else -log(probability),
        nll_status="infinite" if probability == 0 else "finite",
        correct=result.selected_id == target,
    )
    progress("readout_validated")
    return row


def run_update(learner, predict, report: dict[str, Any], progress) -> None:
    """Ordinary delayed-label inbox sequence; labels never reach readout calls."""
    clock = LogicalClock()
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=1, max_wall_seconds=600), monotonic
    )
    inbox = ExperienceInbox(
        learner, clock=clock, budget=budget, learner_version="native-rank8", max_experiences=1
    )
    report["readouts"] = []
    report["delivery"] = []
    for phase in ("before", "after"):
        for name, query, target in (
            ("old", "invoice refund", "billing"),
            ("new", "Case marker: zorbex; please assist.", "shipping"),
        ):
            request = DecisionRequest(f"{phase}-{name}", query, CATALOG)
            record_readout(
                request,
                predict,
                report["readouts"],
                progress,
                phase=phase,
                context=name,
                target=target,
            )
        if phase == "before":
            source = Experience(
                "train-zorbex",
                "native-update",
                0,
                "native-rank8",
                DecisionRequest("train-zorbex", "Case marker: zorbex", CATALOG),
                "train",
                ExperiencePermissions(training=True),
            )
            inbox.record_experience(source)
            before = inbox.drain(max_updates=1)
            if before:
                raise RuntimeError("unlabeled experience unexpectedly trained")
            report["delivery"].append(dict(event="pre_label_poll", updates=len(before), tick=0))
            clock.advance_to(1)
            inbox.record_label(
                LabelArrival(
                    "label-zorbex", "train-zorbex", "native-update", 1, "native-rank8", "shipping"
                )
            )
            progress("training_attempt")
            try:
                applied = inbox.drain(max_updates=1)
            finally:
                report["native_work"] = learner.work
                report["inbox_stopped"] = inbox.stopped
                report["app_committed_updates"] = budget.updates_completed
                progress("training_finished")
            if len(applied) != 1:
                raise RuntimeError("expected one delayed-label update")
            report["delivery"].append(
                dict(event="label_applied", tick=1, receipt=asdict(applied[0]))
            )
            report["training"] = learner.last_update
    report["sequence_complete"] = True


def _weights(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def _weight_summary(model, before):
    rows = {}
    for name, p in model.named_parameters():
        if name in before:
            value = p.detach()
            delta = value - before[name]
            rows[name] = dict(
                elements=p.numel(),
                before_sha256=hashlib.sha256(before[name].numpy().tobytes()).hexdigest(),
                after_sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest(),
                max_absolute_delta=float(delta.abs().max().item()),
                l2_delta=float(delta.norm().item()),
            )
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-checkpoint", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.save_checkpoint.exists():
        parser.error("outputs already exist; preserve previous evidence")
    report: dict[str, Any] = dict(
        definition="native_open_jev_one_train_update_v1",
        complete=False,
        model_load_attempts=0,
        events=[],
        seed=17,
        training_temperature=1,
        inference_temperature=TEMPERATURE,
        calibration_status="inherited_unrecalibrated",
        training_random_draws="unknown",
        checkpoint_recomputation_work="unmeasured_internal_layers",
        final_examples=0,
        probe_updates=0,
        replay_examples=0,
    )
    with args.output.open("x", encoding="utf8") as stream:
        stream.write("{}\n")
    started = monotonic()
    owner: dict[str, Any] = {}

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
            raise RuntimeError("native update resource allowance exceeded")

    def observe_model(model):
        api = importlib.import_module("jev.api")
        owner["learner"] = OpenJevTrainingLearner(
            model, api.compile_request, instructions=INSTRUCTIONS
        )
        owner["before"] = _weights(model)

    try:
        report["model_load_attempts"] = 1
        predict = _load(
            args.source,
            args.checkpoint,
            progress,
            prediction_limit=4,
            forward_limit=13,
            token_limit=3840,
            model_observer=observe_model,
        )
        learner = owner["learner"]
        run_update(learner, predict, report, progress)
        learner._guard()
        report["weights"] = _weight_summary(learner.model, owner["before"])
        report["model_training_after"] = learner.model.training
        report["native_work"] = learner.work
        progress("save_attempt")
        # A fresh native inference package, not a portable optimizer checkpoint.
        # The inherited temperature is deliberately not recalibrated on readouts.
        learner.model.save(args.save_checkpoint)
        (args.save_checkpoint / "temperature.json").write_text(
            json.dumps(
                dict(
                    temperature=TEMPERATURE,
                    calibration_status="inherited_unrecalibrated_after_training",
                )
            )
            + "\n",
            encoding="utf8",
        )
        saved = list(args.save_checkpoint.rglob("*"))
        report["saved_checkpoint_bytes"] = sum(p.stat().st_size for p in saved if p.is_file())
        if report["saved_checkpoint_bytes"] > 16 * 1024**2:
            raise RuntimeError("native checkpoint storage allowance exceeded")
        progress("saved")
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
