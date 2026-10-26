"""Native observation-bound temporal workflow through existing inbox and scorer.

Consumes the fixed core corpus; emits local TRAIN receipts and development metrics.
No final/probe selection, tool execution, tuning, new owner or checkpoint format.
"""

import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import sys
from time import monotonic
from typing import Any

from scripts.run_open_jev_native_smoke import TEMPERATURE, _load
from scripts.run_open_jev_native_update import record_readout
from scripts.run_open_jev_sequence import _deliver_exposure
from scripts.run_temporal_decision_reference import _annotate, _cases, _contrasts
from src.adapters.open_jev_training import OpenJevTrainingLearner, OpenJevTrainingPolicy
from src.app.decision_reference_evaluation import (
    DecisionEvaluationFailed,
    evaluate_probability_reference,
)
from src.app.experience_inbox import ExperienceInbox
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.experience import LogicalClock
from src.core.temporal_decision_sequence import (
    CHECKPOINTS,
    PHASES,
    TemporalDecisionExample,
    temporal_examples,
    validate_temporal_examples,
)
from src.core.typed_decision import ProbabilityDecisionResult

INSTRUCTIONS = "Choose the ticket workflow tool."
MAX_UPDATES = 36
SNAPSHOT_BYTES = 400 * 1024**2


def _validate_train(example):
    if type(example) is not TemporalDecisionExample:
        raise ValueError("expected exact temporal TRAIN example")
    example.__post_init__()
    if example.role != "train":
        raise ValueError("only declared temporal TRAIN examples may update")


def _training():
    packets = tuple(
        temporal_examples(phase, "train", exposure=exposure)
        for phase in PHASES
        for exposure in range(4)
    )
    for packet in packets:
        validate_temporal_examples(packet)
        for example in packet:
            _validate_train(example)
    return tuple(example for packet in packets for example in packet)


def _prefix(learner, inbox, budget, report, examples, completed):
    """A fixed36-row history: verify preceding success before more paid work."""
    if (
        budget.updates_completed != completed
        or len(inbox.applied_updates) != completed
        or len(report["delivery"]) != completed
        or set(learner.work)
        != {
            "training_attempts",
            "forward_calls",
            "backward_calls",
            "optimizer_steps",
            "committed_updates",
        }
        or any(type(value) is not int or value != completed for value in learner.work.values())
    ):
        raise ValueError("temporal paid update counts differ")
    for row, receipt, example in zip(report["delivery"], inbox.applied_updates, examples):
        if (
            row.get("receipt") != asdict(receipt)
            or row.get("source_example_id") != example.request.request_id
            or row.get("source_group") != example.source_group
            or row.get("policy_version") != example.policy_version
            or row.get("phase") != example.phase
            or row.get("exposure_id") != example.request.request_id
            or row.get("observed_at") != 2 * (receipt.update_number - 1)
            or row.get("prelabel_updates") != 0
        ):
            raise ValueError("temporal prior receipt or observation policy differs")
    if completed and report["delivery"][-1].get("training") != learner.last_update:
        raise ValueError("temporal preceding successful training evidence differs")


def _tensor_bytes(value):
    """Count existing snapshot tensor storage; fake snapshots report unknown."""
    if isinstance(value, dict):
        return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(v) for v in value)
    if callable(getattr(value, "numel", None)) and callable(getattr(value, "element_size", None)):
        return value.numel() * value.element_size()
    return 0


def _capture(learner, states, report, completed, progress):
    work = report["snapshot_work"]
    if len(states) >= 4 or completed in states:
        raise ValueError("temporal snapshot count allowance exhausted")
    # Why this: inspect the already-owned tensors before making another copy.
    # Scalar recording fakes have no tensors and report an unknown byte count.
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
            raise ValueError("temporal snapshot storage allowance exceeded before copy")
    work["capture_attempts"] += 1
    snapshot = learner.snapshot_state()
    states[completed] = snapshot
    work["captures_completed"] += 1
    sizes = [_tensor_bytes(vars(s)) if hasattr(s, "__dict__") else 0 for s in states.values()]
    work["retained_numeric_bytes"] = sum(sizes) if all(sizes) else None
    work["peak_snapshots"] = max(work["peak_snapshots"], len(states))
    if len(states) > 4 or sum(sizes) > SNAPSHOT_BYTES:
        raise ValueError("temporal snapshot storage allowance exceeded")
    progress("snapshot_captured")


def _restore(learner, snapshot, source, report, progress):
    work = report["snapshot_work"]
    if work["restore_attempts"] >= 11:
        raise ValueError("temporal restore allowance exhausted")
    work["restore_attempts"] += 1
    learner.restore_state(snapshot)
    work["restores_completed"] += 1
    report["selected_source_updates"] = source
    progress("snapshot_restored")


def _evaluate_native(predict, report, progress, examples, control, source, verify):
    validate_temporal_examples(examples)
    by_id = {e.request.request_id: e for e in examples}

    def typed_predict(request):
        example = by_id[request.request_id]

        def capture_progress(event):
            report["readouts"][-1].update(
                policy_version=example.policy_version, control=control, source_updates=source
            )
            progress(event)
            verify()

        row = record_readout(
            request,
            predict,
            report["readouts"],
            capture_progress,
            phase=example.phase,
            context=example.domain,
            target=example.target_id,
            instructions=INSTRUCTIONS,
        )
        row.update(policy_version=example.policy_version, control=control, source_updates=source)
        result = dict(row["result"])
        result["probabilities"] = tuple(tuple(p) for p in result["probabilities"])
        return ProbabilityDecisionResult(**result)

    try:
        evaluation = evaluate_probability_reference(_cases(examples), predict=typed_predict)
    except DecisionEvaluationFailed as error:
        _annotate(error.report, examples)
        raise
    return _annotate(evaluation, examples)


def run_sequence(learner, predict, report: dict[str, Any], progress) -> None:
    """Complete fixed TRAIN before scoring; snapshots never renew native work."""
    examples = _training()
    development = {p: temporal_examples(p, "outer_selection") for _, p in CHECKPOINTS}
    for packet in development.values():
        validate_temporal_examples(packet)
    report.update(
        delivery=[],
        readouts=[],
        checkpoints=[],
        sequence_complete=False,
        snapshot_work=dict(
            capture_attempts=0,
            captures_completed=0,
            restore_attempts=0,
            restores_completed=0,
            peak_snapshots=0,
            retained_numeric_bytes=0,
        ),
        training_cases=[asdict(e) for e in examples],
        development_cases={p: [asdict(e) for e in es] for p, es in development.items()},
    )
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=36, max_wall_seconds=600), monotonic
    )
    clock = LogicalClock()
    inbox = ExperienceInbox(
        learner, clock=clock, budget=budget, learner_version="native-rank8", max_experiences=36
    )
    states: dict[int, Any] = {}
    try:
        _prefix(learner, inbox, budget, report, examples, 0)
        _capture(learner, states, report, 0, progress)
        for completed, example in enumerate(examples, 1):
            _prefix(learner, inbox, budget, report, examples, completed - 1)
            _deliver_exposure(
                learner,
                inbox,
                clock,
                example,
                example.request.request_id,
                example.phase,
                report,
                progress,
                validate_example=_validate_train,
                observation_policy=example.policy_version,
            )
            _prefix(learner, inbox, budget, report, examples, completed)
            if completed in (12, 24, 36):
                _capture(learner, states, report, completed, progress)
        for conventional_update, phase in CHECKPOINTS:
            sources = dict(learned=conventional_update * 3, unchanged=0)
            if phase in ("after_B", "after_return"):
                sources["frozen_A"] = 12
            row = dict(phase=phase, source_updates=sources)
            report["checkpoints"].append(row)
            for control, source in sources.items():
                _prefix(learner, inbox, budget, report, examples, 36)
                _restore(learner, states[source], source, report, progress)
                _prefix(learner, inbox, budget, report, examples, 36)
                try:
                    row[control] = _evaluate_native(
                        predict,
                        report,
                        progress,
                        development[phase],
                        control,
                        source,
                        lambda: _prefix(learner, inbox, budget, report, examples, 36),
                    )
                except DecisionEvaluationFailed as error:
                    row[control] = error.report
                    raise
        _prefix(learner, inbox, budget, report, examples, 36)
        _restore(learner, states[36], 36, report, progress)
        _prefix(learner, inbox, budget, report, examples, 36)
        report["contrasts"] = _contrasts(report["checkpoints"])
        report["sequence_complete"] = True
    finally:
        report["native_work"] = learner.work
        report["app_committed_updates"] = budget.updates_completed
        report["inbox_stopped"] = inbox.stopped
        report["logical_tick"] = clock.now()
        report["snapshot_work"]["released_snapshots"] = len(states)
        states.clear()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output exists; preserve previous evidence")
    report: dict[str, Any] = dict(
        definition="native_temporal_workflow_v1",
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
        checkpoint_fingerprint_scope="original_load_birth_provenance",
        snapshot_scope="in_memory_owner_state; permanent work and last_update not restored",
        comparison_scope="same36row exposures as linear reference; different36vs12updates/math/compute",
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
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if len(encoded.encode()) > 4 * 1024**2:
            raise RuntimeError("native temporal report exceeds4MiB")
        args.output.write_text(encoded, encoding="utf8")
        if event["outer_seconds"] > 600 or event["outer_peak_rss_bytes"] > 20 * 1024**3:
            raise RuntimeError("native temporal resource allowance exceeded")

    def observe_model(model):
        api = importlib.import_module("jev.api")
        owner["learner"] = OpenJevTrainingLearner(
            model,
            api.compile_request,
            instructions=INSTRUCTIONS,
            policy=OpenJevTrainingPolicy(MAX_UPDATES, 8),
        )

    try:
        _training()
        report["model_load_attempts"] = 1
        predict = _load(
            args.source,
            args.checkpoint,
            progress,
            prediction_limit=80,
            forward_limit=276,
            token_limit=89088,
            model_observer=observe_model,
            work_reporter=lambda observe: owner.update(observe=observe),
        )
        run_sequence(owner["learner"], predict, report, progress)
        owner["observe"]()
        report["observed_work_terminal_status"] = "final_observation_after_all_work"
        owner["learner"]._guard()
        report["model_training_after"] = owner["learner"].model.training
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["observed_work_terminal_status"] = (
            "unknown_after_failure; latest observed_work event is a lower bound, not a final total"
        )
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
