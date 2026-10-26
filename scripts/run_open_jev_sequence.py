"""Fixed A→B→A exploratory native Choice stream, using the existing inbox.

No final/probe labels, replay, tuning, early stopping, actor fork or model reload.
The unchanged control reuses baseline readouts; it is not a separately run actor.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
import importlib
import json
from math import fsum, isfinite
from pathlib import Path
import sys
from time import monotonic
from typing import Any

from scripts.run_open_jev_native_smoke import TEMPERATURE, _load
from scripts.run_open_jev_native_update import _weights, _weight_summary, record_readout
from src.adapters.open_jev_training import OpenJevTrainingLearner, OpenJevTrainingPolicy
from src.app.experience_inbox import ExperienceInbox
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.experience import require_identifier
from src.core.native_choice_sequence import (
    INSTRUCTIONS,
    PASSES,
    PHASES,
    _validate_example,
    sequence_examples,
    validate_sequence_examples,
)

CHECKPOINTS = ("baseline", *(name for name, _ in PHASES))
MAX_UPDATES = 18


def _group_metrics(rows):
    for row in rows:
        if type(row.get("correct")) is not bool:
            raise ValueError("summary requires validated boolean decisions")
        nll, status = row.get("nll"), row.get("nll_status")
        if not (
            (status == "infinite" and nll is None)
            or (status == "finite" and type(nll) in (int, float) and isfinite(nll) and nll >= 0)
        ):
            raise ValueError("summary requires finite nonnegative or explicitly infinite NLL")
    infinite = any(row["nll_status"] == "infinite" for row in rows)
    return dict(
        examples=len(rows),
        accuracy=sum(row["correct"] for row in rows) / len(rows),
        nll_mean=None if infinite else fsum(row["nll"] for row in rows) / len(rows),
        nll_status="infinite" if infinite else "finite",
    )


def _difference(after, before):
    finite = after["nll_status"] == before["nll_status"] == "finite"
    return dict(
        accuracy=after["accuracy"] - before["accuracy"],
        nll_mean=after["nll_mean"] - before["nll_mean"] if finite else None,
        nll_status="finite_difference" if finite else "nonfinite_comparison_unavailable",
    )


def summarize_sequence(readouts) -> dict[str, Any]:
    """Score all fixed development rows after training; never choose later work."""
    expected = {e.example_id: e.context for e in sequence_examples() if e.role == "outer_selection"}
    if type(readouts) is not list or len(readouts) != 24:
        raise ValueError("summary requires all 24 declared readouts")
    groups: dict[str, Any] = {}
    for phase in CHECKPOINTS:
        selected = [r for r in readouts if r.get("phase") == phase]
        if (
            len(selected) != 6
            or {r.get("example_id") for r in selected} != expected.keys()
            or any(r.get("context") != expected[r["example_id"]] for r in selected)
        ):
            raise ValueError("summary requires each declared readout exactly once per phase")
        groups[phase] = {
            context: _group_metrics([r for r in selected if r["context"] == context])
            for context in ("A", "B", "anchor")
        }
    baseline, a, b, returned = (groups[name] for name in CHECKPOINTS)
    contrasts = dict(
        trained_minus_unchanged={
            phase: {
                context: _difference(groups[phase][context], baseline[context])
                for context in ("A", "B", "anchor")
            }
            for phase in CHECKPOINTS
        },
        phase_deltas=dict(
            A_acquisition=_difference(a["A"], baseline["A"]),
            B_acquisition=_difference(b["B"], a["B"]),
            A_retention=_difference(b["A"], a["A"]),
            A_return=_difference(returned["A"], b["A"]),
        ),
    )
    criteria = dict(
        A_after_phase_success=a["A"]["accuracy"] == 1,
        B_after_phase_success=b["B"]["accuracy"] == 1,
        A_retained_if_acquired=a["A"]["accuracy"] == b["A"]["accuracy"] == 1,
        A_return_success=returned["A"]["accuracy"] == 1,
        A_recovery_demonstrated=(
            a["A"]["accuracy"] == 1 and b["A"]["accuracy"] < 1 and returned["A"]["accuracy"] == 1
        ),
        anchors_preserved=all(
            group["anchor"]["accuracy"] == 1
            and group["anchor"]["nll_status"] == baseline["anchor"]["nll_status"] == "finite"
            and group["anchor"]["nll_mean"] - baseline["anchor"]["nll_mean"] <= 0.1
            for group in groups.values()
        ),
    )
    return dict(
        by_phase=groups,
        unchanged_control={phase: deepcopy(baseline) for phase in CHECKPOINTS},
        unchanged_control_definition="baseline_reused_no_additional_calls",
        contrasts=contrasts,
        criteria=criteria,
    )


def _deliver_exposure(
    learner,
    inbox,
    clock,
    example,
    identity,
    phase,
    report,
    progress,
    *,
    replay_permission=False,
    replay_source_exposure_id=None,
    validate_example=None,
    observation_policy=None,
):
    """Deliver one TRAIN label; replay is explicit and pays the same native work."""
    validator = _validate_example if validate_example is None else validate_example
    if not callable(validator):
        raise ValueError("example validator must be callable")
    validator(example)
    if observation_policy is not None:
        require_identifier(observation_policy, "observation_policy")
    if example.role != "train" or type(replay_permission) is not bool:
        raise ValueError("only declared TRAIN exposures with boolean consent may train")
    if replay_source_exposure_id is not None and not replay_permission:
        raise ValueError("rehearsal requires explicit replay permission")
    observed = clock.now()
    permissions = ExperiencePermissions(training=True, replay=replay_permission)
    inbox.record_experience(
        Experience(
            identity,
            "native-recurring",
            observed,
            "native-rank8",
            example.request,
            "train",
            permissions,
        )
    )
    if inbox.drain(max_updates=1):
        raise RuntimeError("unlabeled sequence exposure unexpectedly trained")
    row = dict(
        phase=phase,
        exposure_id=identity,
        source_example_id=example.request.request_id,
        source_group=example.source_group,
        observed_at=observed,
        prelabel_updates=0,
    )
    if observation_policy is not None:
        row["policy_version"] = observation_policy
    if replay_permission:
        row["permissions"] = asdict(permissions)
    if replay_source_exposure_id is not None:
        row.update(exposure_kind="replay", replay_source_exposure_id=replay_source_exposure_id)
    report["delivery"].append(row)
    clock.advance_to(observed + 1)
    inbox.record_label(
        LabelArrival(
            f"label-{identity}",
            identity,
            "native-recurring",
            clock.now(),
            "native-rank8",
            example.target_id,
        )
    )
    progress("training_attempt")
    try:
        applied = inbox.drain(max_updates=1)
        if len(applied) != 1:
            raise RuntimeError("expected one sequence label update")
        # Preserve the paid receipt before a resource callback can stop work.
        row["receipt"] = asdict(applied[0])
        row["training"] = deepcopy(learner.last_update)
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["native_work"] = learner.work
        report["inbox_stopped"] = inbox.stopped
        progress("training_finished")
    clock.advance_to(clock.now() + 1)


def _rehearsal_sources(learner, inbox, examples, phase, context, report, retained):
    """Only the two previously paid and prospectively authorized A TRAIN cases."""
    validate_sequence_examples(examples)
    expected = tuple(e for e in examples if e.role == "train" and e.context == "A")
    history = report["delivery"]
    paid = inbox.applied_updates
    if (
        context != "B"
        or phase != "after_B_rehearsal"
        or type(retained) is not tuple
        or retained != expected
        or len(history) != 6
        or len(paid) != 6
        or learner.work["committed_updates"] != 6
    ):
        raise ValueError("rehearsal requires exactly two acquired A TRAIN sources before B")
    for index, row in enumerate(history):
        example = expected[index % 2]
        if (
            row.get("phase") != "after_A"
            or row.get("source_example_id") != example.example_id
            or row.get("source_group") != example.source_group
            or row.get("exposure_id") != f"after_A-pass{index // 2}-{example.example_id}"
            or row.get("permissions") != dict(training=True, replay=True, evaluation=False)
            or row.get("receipt", {}).get("update_number") != index + 1
            or row.get("receipt") != asdict(paid[index])
        ):
            raise ValueError("rehearsal source lacks the original successful replay consent")
    return {example.example_id: history[i]["exposure_id"] for i, example in enumerate(expected)}


def _deliver_phase(
    learner,
    inbox,
    clock,
    examples,
    phase,
    context,
    report,
    progress,
    *,
    replay_permission=False,
    rehearsal_examples=(),
):
    if type(replay_permission) is not bool or type(rehearsal_examples) is not tuple:
        raise ValueError("phase replay options require a boolean and tuple")
    if replay_permission:
        validate_sequence_examples(examples)
        if phase != "after_A" or context != "A":
            raise ValueError("only initial A TRAIN sources may grant retained replay consent")
    sources = (
        _rehearsal_sources(learner, inbox, examples, phase, context, report, rehearsal_examples)
        if rehearsal_examples
        else {}
    )
    for repetition in range(PASSES):
        for index, example in enumerate(
            e for e in examples if e.role == "train" and e.context == context
        ):
            identity = f"{phase}-pass{repetition}-{example.example_id}"
            _deliver_exposure(
                learner,
                inbox,
                clock,
                example,
                identity,
                phase,
                report,
                progress,
                replay_permission=replay_permission,
            )
            if rehearsal_examples:
                retained = rehearsal_examples[index]
                _deliver_exposure(
                    learner,
                    inbox,
                    clock,
                    retained,
                    f"{phase}-pass{repetition}-replay-{retained.example_id}",
                    phase,
                    report,
                    progress,
                    replay_permission=True,
                    replay_source_exposure_id=sources[retained.example_id],
                )


def run_sequence(learner, predict, report: dict[str, Any], progress) -> None:
    examples = sequence_examples()
    validate_sequence_examples(examples)
    report.update(
        examples=[asdict(e) for e in examples], readouts=[], delivery=[], sequence_complete=False
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
        for phase, context in (("baseline", None), *PHASES):
            if context is not None:
                _deliver_phase(learner, inbox, clock, examples, phase, context, report, progress)
            for example in (e for e in examples if e.role == "outer_selection"):
                row = record_readout(
                    example.request,
                    predict,
                    report["readouts"],
                    progress,
                    phase=phase,
                    context=example.context,
                    target=example.target_id,
                )
                row.update(example_id=example.example_id, source_group=example.source_group)
            progress("checkpoint_validated")
        if budget.updates_completed != MAX_UPDATES:
            raise RuntimeError("sequence did not execute exactly 18 updates")
        report["summary"] = summarize_sequence(report["readouts"])
        report["sequence_complete"] = True
    finally:
        report["native_work"] = learner.work
        report["app_committed_updates"] = budget.updates_completed
        report["inbox_stopped"] = inbox.stopped


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output exists; preserve previous evidence")
    report: dict[str, Any] = dict(
        definition="native_choice_recurring_v1",
        complete=False,
        events=[],
        model_load_attempts=0,
        seed=17,
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
        if event["event"] == "checkpoint_validated":
            learner = owner["learner"]
            learner._guard()
            report.setdefault("weight_checkpoints", []).append(
                dict(
                    after_updates=learner.work["committed_updates"],
                    model_training=learner.model.training,
                    weights=_weight_summary(learner.model, owner["before"]),
                )
            )
        report["events"].append(event)
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )
        if event["outer_seconds"] > 600 or event["outer_peak_rss_bytes"] > 20 * 1024**3:
            raise RuntimeError("native sequence resource allowance exceeded")

    def observe_model(model):
        api = importlib.import_module("jev.api")
        owner["learner"] = OpenJevTrainingLearner(
            model,
            api.compile_request,
            instructions=INSTRUCTIONS,
            policy=OpenJevTrainingPolicy(MAX_UPDATES, 8),
        )
        owner["before"] = _weights(model)

    try:
        validate_sequence_examples(sequence_examples())
        report["model_load_attempts"] = 1
        predict = _load(
            args.source,
            args.checkpoint,
            progress,
            prediction_limit=24,
            forward_limit=90,
            token_limit=32256,
            model_observer=observe_model,
        )
        run_sequence(owner["learner"], predict, report, progress)
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
