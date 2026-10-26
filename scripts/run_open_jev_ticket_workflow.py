"""Bounded native TRAIN-to-ticket effects using existing inbox and typed ports.

Consumes twelve local TRAIN demonstrations twice, then executes six fixed
development episodes. No final/probe/tuning, new adapter or external tool handler.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import importlib
import json
from pathlib import Path
import sys
from time import monotonic
from typing import Any

from scripts.run_open_jev_native_smoke import TEMPERATURE, _load
from scripts.run_open_jev_sequence import _deliver_exposure
from scripts.run_open_jev_temporal_sequence import _tensor_bytes
from scripts.run_ticket_workflow import training_cases
from src.adapters.open_jev_training import OpenJevTrainingLearner, OpenJevTrainingPolicy
from src.app.decision_reference_evaluation import DecisionCase, _observed_probability_return
from src.app.experience_inbox import ExperienceInbox
from src.app.ticket_workflow import run_workflow_episode
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.experience import ExperiencePermissions, LogicalClock
from src.core.temporal_decision_sequence import VOCABULARY, policy_for_phase
from src.core.ticket_workflow import ACTION_IDS, CATALOG, WorkflowCapabilities, WorkflowState
from src.core.typed_decision import (
    DecisionRequest,
    validate_probability_decision_result,
)

INSTRUCTIONS = "Choose the ticket workflow tool."
MAX_UPDATES = 24
MAX_PREDICTIONS = 36
SNAPSHOT_BYTES = 160 * 1024**2
DEVELOPMENT = (("ordinary", "invoice"), ("stale_once", "parcel"), ("revoke_commit", "invoice"))
TRAIN_COUNTERS = ("training_attempts", "backward_calls", "optimizer_steps", "committed_updates")
ACTIVE_IDS = ("commit", "defer", "inspect", "open", "route")


def _validate_train(example):
    if type(example) is not DecisionCase:
        raise ValueError("expected exact workflow TRAIN case")
    example.__post_init__()
    if example.role != "train" or example.source_group != "train-workflow-traces":
        raise ValueError("only declared workflow TRAIN examples may update")


def _validate_fixed(case, index, prefix=""):
    _validate_train(case)
    concept = VOCABULARY[index // 3]
    step = index % 3
    target = ("open", policy_for_phase("after_return")["rules"][concept], "commit")[step]
    query = (
        "sealed feedback initial train",
        f"ready {concept} feedback applied train",
        "staged feedback applied train",
    )[step]
    if (
        case.request.request_id != f"{prefix}train-{index // 3}-{step}"
        or case.request.query != query
        or case.target_id != target
        or case.request.allowed_candidates
        != tuple(c for c in CATALOG if c.candidate_id in ACTIVE_IDS)
    ):
        raise ValueError("workflow TRAIN demonstration differs from fixed protocol")


def _training():
    cases = training_cases()
    if type(cases) is not tuple or len(cases) != 12:
        raise ValueError("workflow requires twelve fixed TRAIN demonstrations")
    for index, case in enumerate(cases):
        _validate_fixed(case, index)
    return tuple(
        replace(
            case, request=replace(case.request, request_id=f"pass{p}-{case.request.request_id}")
        )
        for p in range(2)
        for case in cases
    )


def _prefix(learner, inbox, budget, report, examples, completed, evidence_reuse=False):
    work = learner.work
    predictions = report["prediction_work"]["attempted"]
    if (
        len(report["delivery"]) != completed
        or len(inbox.applied_updates) != completed
        or budget.updates_completed != completed
        or set(work) != {*TRAIN_COUNTERS, "forward_calls"}
        or any(type(work[k]) is not int or work[k] != completed for k in TRAIN_COUNTERS)
        or type(work["forward_calls"]) is not int
        or work["forward_calls"] != completed + predictions
    ):
        raise ValueError("workflow prior paid work differs")
    for example, row, receipt in zip(examples, report["delivery"], inbox.applied_updates):
        if (
            json.dumps(row.get("receipt"), sort_keys=True, allow_nan=False)
            != json.dumps(asdict(receipt), sort_keys=True, allow_nan=False)
            or row.get("exposure_id") != example.request.request_id
            or row.get("source_example_id") != example.request.request_id
            or row.get("source_group") != example.source_group
            or row.get("policy_version") != "v2"
            or row.get("observed_at") != 2 * (receipt.update_number - 1)
            or type(row.get("observed_at")) is not int
            or row.get("prelabel_updates") != 0
            or type(row.get("prelabel_updates")) is not int
            or json.dumps(row.get("permissions"), sort_keys=True, allow_nan=False)
            != json.dumps(
                (
                    asdict(ExperiencePermissions(training=True, replay=True))
                    if evidence_reuse
                    else None
                ),
                sort_keys=True,
                allow_nan=False,
            )
        ):
            raise ValueError("workflow preceding receipt differs")
    if completed and report["delivery"][-1].get("training") != learner.last_update:
        raise ValueError("workflow preceding successful update evidence differs")


class BoundedWorkflowPredictor:
    """Bound the existing typed native port; retain returns before postchecks.

    Why this: its permanent forward count includes both TRAIN and batched
    readouts. Serving's independent-candidate counter semantics do not apply.
    """

    def __init__(self, learner, report, progress, *, max_predictions=MAX_PREDICTIONS):
        if type(max_predictions) is not int or not 1 <= max_predictions <= 128:
            raise ValueError("max_predictions must be an integer in1..128")
        # Older reports had only the default36 allowance. Missing metadata must
        # not enlarge a spent history; explicit larger caps bind only at birth.
        if "prediction_limit" not in report and max_predictions != MAX_PREDICTIONS:
            prior = report.get("prediction_work", dict(attempted=0, returned=0, validated=0))
            if (
                type(prior) is not dict
                or set(prior) != {"attempted", "returned", "validated"}
                or any(type(value) is not int or value != 0 for value in prior.values())
                or report.get("native_predictions", []) != []
            ):
                raise ValueError("missing prediction allowance cannot enlarge prior history")
        bound = report.get("prediction_limit", max_predictions)
        if type(bound) is not int or bound != max_predictions:
            raise ValueError("aggregate prediction allowance cannot be replaced")
        report.setdefault("prediction_limit", max_predictions)
        self._max_predictions = max_predictions
        self.learner, self.report, self.progress = learner, report, progress
        report.setdefault("prediction_work", dict(attempted=0, returned=0, validated=0))
        report.setdefault("native_predictions", [])
        self.failed = False

    def __call__(self, request):
        learner, report = self.learner, self.report
        if (
            type(report.get("prediction_limit")) is not int
            or report["prediction_limit"] != self._max_predictions
        ):
            raise ValueError("aggregate prediction allowance changed")
        work = report["prediction_work"]
        if self.failed or not work["attempted"] == work["returned"] == work["validated"]:
            raise ValueError("workflow prediction stopped after uncertain prior work")
        if work["attempted"] >= self._max_predictions:
            raise ValueError("workflow prediction allowance exhausted")
        if type(request) is not DecisionRequest:
            raise ValueError("expected exact workflow DecisionRequest")
        request.__post_init__()
        if tuple(c.candidate_id for c in request.allowed_candidates) != ACTIVE_IDS:
            raise ValueError("native workflow requires exactly five active candidates")
        learner._guard()
        before = learner.work
        if any(before[k] != MAX_UPDATES for k in TRAIN_COUNTERS) or before["forward_calls"] != (
            MAX_UPDATES + work["attempted"]
        ):
            raise ValueError("native workflow training/readout work differs")
        versions = learner._trainable_versions()
        mode = learner.model.training
        torch = learner._torch
        caller_rng, private_rng = torch.get_rng_state().clone(), learner._rng.clone()
        supplied = deepcopy(request)
        row: dict[str, Any] = dict(request=asdict(request))
        report["native_predictions"].append(row)
        work["attempted"] += 1
        try:
            self.progress("typed_prediction_attempt")
            if (
                learner.work != before
                or versions != learner._trainable_versions()
                or mode != learner.model.training
                or not torch.equal(caller_rng, torch.get_rng_state())
                or not torch.equal(private_rng, learner._rng)
            ):
                raise ValueError("workflow observation changed native state before prediction")
            result = learner.predict(supplied)
            work["returned"] += 1
            row["returned"] = _observed_probability_return(result)
            validate_probability_decision_result(request, result)
            row["result"] = asdict(result)
            learner._guard()
            after = learner.work
            if (
                supplied != request
                or set(after) != set(before)
                or any(after[k] != before[k] for k in TRAIN_COUNTERS)
                or after["forward_calls"] != before["forward_calls"] + 1
                or versions != learner._trainable_versions()
                or mode != learner.model.training
                or not torch.equal(caller_rng, torch.get_rng_state())
                or not torch.equal(private_rng, learner._rng)
            ):
                raise ValueError("native prediction changed owned state, RNG, mode or counters")
            self.progress("typed_prediction_validated")
            learner._guard()
            if (
                after != learner.work
                or versions != learner._trainable_versions()
                or mode != learner.model.training
                or not torch.equal(caller_rng, torch.get_rng_state())
                or not torch.equal(private_rng, learner._rng)
            ):
                raise ValueError("workflow observation changed native state after prediction")
            work["validated"] += 1
            return result
        except Exception as error:
            self.failed = True
            row["error"] = type(error).__name__ + ": " + str(error)[:512]
            raise


def _capture(learner, states, report, completed):
    if len(states) >= 2 or completed not in (0, MAX_UPDATES) or completed in states:
        raise ValueError("workflow snapshot allowance exhausted")
    trainable = getattr(learner, "_trainable", None)
    if type(trainable) is dict:
        prospective = (
            _tensor_bytes(trainable)
            + _tensor_bytes([p.grad for p in trainable.values()])
            + _tensor_bytes(learner._optimizer.state_dict())
            + _tensor_bytes(learner._rng)
        )
        retained = sum(_tensor_bytes(vars(s)) for s in states.values())
        if retained + prospective > SNAPSHOT_BYTES:
            raise ValueError("workflow snapshot bytes exceeded before copy")
    work = report["snapshot_work"]
    work["capture_attempts"] += 1
    states[completed] = learner.snapshot_state()
    work["captures_completed"] += 1
    sizes = [_tensor_bytes(vars(s)) if hasattr(s, "__dict__") else 0 for s in states.values()]
    work["retained_numeric_bytes"] = sum(sizes) if all(sizes) else None
    if sum(sizes) > SNAPSHOT_BYTES:
        raise ValueError("workflow snapshot bytes exceeded")


def _evaluate_default(learner, states, report, progress, verify):
    for control, source in (("unchanged", 0), ("learned", MAX_UPDATES)):
        verify()
        report["snapshot_work"]["restore_attempts"] += 1
        learner.restore_state(states[source])
        report["snapshot_work"]["restores_completed"] += 1
        report["selected_source_updates"] = source
        verify()
        progress("snapshot_restored")
        predict = BoundedWorkflowPredictor(learner, report, progress)
        for index, (scenario, concept) in enumerate(DEVELOPMENT):
            episode = run_workflow_episode(
                WorkflowState(f"development-ticket-{index}", concept),
                predict,
                capabilities=WorkflowCapabilities(ACTION_IDS),
                max_steps=6,
                scenario=scenario,
            )
            report["episodes"].append(
                dict(
                    control=control,
                    source_updates=source,
                    role="outer_selection",
                    source_group="development-workflow-traces",
                    episode=episode,
                )
            )
            progress("episode_finished")
            if not episode["complete"]:
                raise RuntimeError("native workflow episode contains uncertain paid work")
            verify()
    report["summary"] = {
        control: dict(
            episodes=len(rows),
            successes=sum(e["success"] for e in rows),
            recoveries=sum(e["error_recovery_demonstrated"] for e in rows),
            permission_denials=sum(e["stopped_reason"] == "permission_denied" for e in rows),
        )
        for control in ("unchanged", "learned")
        for rows in ([r["episode"] for r in report["episodes"] if r["control"] == control],)
    }


def run_sequence(
    learner,
    report: dict[str, Any],
    progress,
    *,
    training_examples=None,
    evaluate_sources=None,
    evidence_reuse=False,
) -> None:
    if evaluate_sources is not None and not callable(evaluate_sources):
        raise ValueError("evaluate_sources must be callable")
    if type(evidence_reuse) is not bool:
        raise ValueError("evidence_reuse must be an explicit boolean")
    evaluate = _evaluate_default if evaluate_sources is None else evaluate_sources
    # The CLI admits these once before loading; direct callers generate them here.
    examples = _training() if training_examples is None else training_examples
    if type(examples) is not tuple or len(examples) != MAX_UPDATES:
        raise ValueError("workflow requires twenty-four fixed TRAIN exposures")
    for index, example in enumerate(examples):
        _validate_fixed(example, index % 12, f"pass{index // 12}-")
    report.update(
        delivery=[],
        episodes=[],
        native_predictions=[],
        sequence_complete=False,
        prediction_work=dict(attempted=0, returned=0, validated=0),
        snapshot_work=dict(
            capture_attempts=0,
            captures_completed=0,
            restore_attempts=0,
            restores_completed=0,
            retained_numeric_bytes=0,
        ),
        training_cases=[asdict(e) for e in examples],
        training_case_generation=dict(
            demonstration_transitions=12,
            candidate_constructions=12,
            catalog_entries_considered=72,
            candidates_returned=60,
        ),
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
    states: dict[int, Any] = {}
    try:
        _prefix(learner, inbox, budget, report, examples, 0, evidence_reuse)
        _capture(learner, states, report, 0)
        progress("snapshot_captured")
        for completed, example in enumerate(examples, 1):
            _prefix(learner, inbox, budget, report, examples, completed - 1, evidence_reuse)
            _deliver_exposure(
                learner,
                inbox,
                clock,
                example,
                example.request.request_id,
                "workflow",
                report,
                progress,
                validate_example=_validate_train,
                observation_policy="v2",
                replay_permission=evidence_reuse,
            )
            _prefix(learner, inbox, budget, report, examples, completed, evidence_reuse)
        _capture(learner, states, report, MAX_UPDATES)
        progress("snapshot_captured")
        evaluate(
            learner,
            states,
            report,
            progress,
            lambda: _prefix(learner, inbox, budget, report, examples, MAX_UPDATES, evidence_reuse),
        )
        _prefix(learner, inbox, budget, report, examples, MAX_UPDATES, evidence_reuse)
        report["sequence_complete"] = True
    finally:
        report["native_work"] = learner.work
        report["app_committed_updates"] = budget.updates_completed
        report["logical_tick"] = clock.now()
        report["inbox_stopped"] = inbox.stopped
        report["snapshot_work"]["released_snapshots"] = len(states)
        states.clear()
        works = [r["episode"]["work"] for r in report["episodes"]]
        report["episode_work"] = {k: sum(w[k] for w in works) for k in works[0]} if works else {}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output exists; preserve previous evidence")
    report: dict[str, Any] = dict(
        definition="native_ticket_workflow_v1",
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
        external_actions=0,
        prediction_scope="typed five-candidate batch; no independent-serving parity; usage absent",
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
        encoded = json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n"
        if len(encoded.encode()) > 2 * 1024**2:
            raise RuntimeError("native workflow report exceeds2MiB")
        args.output.write_text(encoded, encoding="utf8")
        if event["outer_seconds"] > 600 or event["outer_peak_rss_bytes"] > 20 * 1024**3:
            raise RuntimeError("native workflow resource allowance exceeded")

    def observe_model(model):
        api = importlib.import_module("jev.api")
        owner["learner"] = OpenJevTrainingLearner(
            model,
            api.compile_request,
            instructions=INSTRUCTIONS,
            policy=OpenJevTrainingPolicy(MAX_UPDATES, 8),
        )

    try:
        examples = _training()
        report["model_load_attempts"] = 1
        _load(
            args.source,
            args.checkpoint,
            progress,
            prediction_limit=MAX_PREDICTIONS,
            forward_limit=MAX_UPDATES + MAX_PREDICTIONS,
            token_limit=76800,
            model_observer=observe_model,
            work_reporter=lambda observe: owner.update(observe=observe),
        )
        run_sequence(owner["learner"], report, progress, training_examples=examples)
        owner["observe"]()
        owner["learner"]._guard()
        report["model_training_after"] = owner["learner"].model.training
        report["observed_work_terminal_status"] = "final_observation_after_all_work"
        report["complete"] = True
    except Exception as error:
        report["error"] = type(error).__name__ + ": " + str(error)[:512]
        report["observed_work_terminal_status"] = (
            "unknown_after_failure; latest observation is a lower bound"
        )
        if "learner" in owner:
            report["native_work"] = owner["learner"].work
    finally:
        sys.setprofile(None)
        report["outer_seconds"] = monotonic() - started
        args.output.write_text(
            json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf8"
        )
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
