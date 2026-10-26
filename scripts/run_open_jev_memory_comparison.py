"""Four fixed native/evidence arms through the existing ticket TRAIN runner.

One shared fit; the same causal TRAIN labels feed bounded evidence. Context
returns remain native choices. No final/probe/tuning or new learner framework.
"""

import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import sys
from time import monotonic
from typing import Any

from scripts import run_open_jev_ticket_workflow as native
from scripts.run_open_jev_native_smoke import TEMPERATURE, _load
from src.adapters.open_jev_training import OpenJevTrainingLearner, OpenJevTrainingPolicy
from src.app.decision_evidence import EvidenceDecisionPredictor
from src.app.ticket_workflow import run_workflow_episode
from src.core.decision_evidence import DecisionEvidence, DecisionEvidenceStore, EvidencePolicy
from src.core.experience import ExperiencePermissions
from src.core.ticket_workflow import ACTION_IDS, WorkflowCapabilities, WorkflowState

ARMS = (
    ("frozen_context", 0, "context"),
    ("retrieval_only", None, "retrieval"),
    ("adapter_only", 24, None),
    ("hybrid", 24, "context"),
)
MAX_PREDICTIONS = 54


def build_evidence(examples, delivery, *, store=None):
    if type(examples) is not tuple or len(examples) != 24 or len(delivery) != 24:
        raise ValueError("memory comparison requires all24 causal TRAIN receipts")
    records = []
    for index, (example, row) in enumerate(zip(examples, delivery, strict=True)):
        native._validate_fixed(example, index % 12, f"pass{index // 12}-")
        receipt = row.get("receipt", {})
        if (
            any(
                type(receipt.get(k)) is not int
                for k in ("update_number", "arrived_at", "applied_at")
            )
            or row.get("source_example_id") != example.request.request_id
            or row.get("exposure_id") != example.request.request_id
            or row.get("source_group") != example.source_group
            or row.get("policy_version") != "v2"
            or row.get("observed_at") != 2 * index
            or receipt.get("update_number") != index + 1
            or receipt.get("arrived_at") != 2 * index + 1
            or receipt.get("applied_at") != 2 * index + 1
            or row.get("permissions") != asdict(ExperiencePermissions(training=True, replay=True))
        ):
            raise ValueError("evidence differs from its consented observation/paid receipt")
        records.append(
            DecisionEvidence(
                row["exposure_id"],
                row["observed_at"],
                receipt["arrived_at"],
                receipt["applied_at"],
                "v2",
                row["source_group"],
                example.request,
                example.target_id,
                role=example.role,
                permissions=ExperiencePermissions(**row["permissions"]),
            )
        )
    # Admit the complete receipt packet before any paid add. Callers that own
    # the store can retain its prefix even if a later capacity/write step fails.
    store = DecisionEvidenceStore(EvidencePolicy(32, 32768, 768, 1)) if store is None else store
    for record in records:
        store.add(record)
    return store


def _delta(after, before):
    return {key: after[key] - before[key] for key in before}


def evaluate_arms(learner, states, report, progress, verify, *, examples):
    verify()
    store = DecisionEvidenceStore(EvidencePolicy(32, 32768, 768, 1))
    report["evidence_cutoff"] = 48
    report["arms"] = []
    try:
        build_evidence(examples, report["delivery"], store=store)
        report["evidence_events"] = [asdict(e) for e in store.events]
        report["evidence_storage_bytes"] = store.storage_bytes
        report["evidence_build_work"] = store.work
        for name, source, mode in ARMS:
            verify()
            row: dict[str, Any] = dict(
                name=name,
                source_updates=source,
                mode=mode,
                optimization_updates=0 if source is None else source,
                training_rows_available=24,
                context_additions_available=24 if mode is not None else 0,
                context_edits=0,
                episode_indices=[],
            )
            report["arms"].append(row)
            if source is not None:
                work = report["snapshot_work"]
                if work["restore_attempts"] >= 3:
                    raise ValueError("comparison restore allowance exhausted")
                work["restore_attempts"] += 1
                learner.restore_state(states[source])
                work["restores_completed"] += 1
                report["selected_source_updates"] = source
                progress("comparison_source_restored")
                verify()
            native_before, evidence_before = learner.work, store.work
            started = monotonic()
            guarded = (
                None
                if source is None
                else native.BoundedWorkflowPredictor(
                    learner, report, progress, max_predictions=MAX_PREDICTIONS
                )
            )
            predictor: Any = (
                guarded
                if mode is None
                else EvidenceDecisionPredictor(
                    store, cutoff=48, policy_version="v2", mode=mode, predict=guarded, max_calls=18
                )
            )
            try:
                for index, (scenario, concept) in enumerate(native.DEVELOPMENT):
                    episode = run_workflow_episode(
                        WorkflowState(f"development-ticket-{index}", concept),
                        predictor,
                        capabilities=WorkflowCapabilities(ACTION_IDS),
                        max_steps=6,
                        scenario=scenario,
                    )
                    row["episode_indices"].append(len(report["episodes"]))
                    report["episodes"].append(
                        dict(
                            control=name,
                            source_updates=source,
                            role="outer_selection",
                            source_group="development-workflow-traces",
                            episode=episode,
                        )
                    )
                    if mode is not None:
                        row["evidence_predictor"] = predictor.report
                    progress("comparison_episode_finished")
                    if not episode["complete"]:
                        raise RuntimeError("memory comparison contains uncertain paid work")
                    verify()
            finally:
                row["native_work_delta"] = _delta(learner.work, native_before)
                row["evidence_work_delta"] = _delta(store.work, evidence_before)
                row["elapsed_seconds"] = monotonic() - started
                if mode is not None:
                    row["evidence_predictor"] = predictor.report
        report["summary"] = {
            name: dict(
                episodes=len(rows),
                successes=sum(e["success"] for e in rows),
                successful_recoveries=sum(e["error_recovery_demonstrated"] for e in rows),
                permission_denials=sum(e["stopped_reason"] == "permission_denied" for e in rows),
                stops=[e["stopped_reason"] for e in rows],
            )
            for name, _, _ in ARMS
            for rows in ([r["episode"] for r in report["episodes"] if r["control"] == name],)
        }
    finally:
        report["evidence_work"] = store.work
        report["evidence_events"] = [asdict(e) for e in store.events]
        report["evidence_storage_bytes"] = store.storage_bytes


def run_comparison(learner, report, progress, *, training_examples=None):
    examples = native._training() if training_examples is None else training_examples

    def evaluate(owner, states, observed, emit, verify):
        evaluate_arms(owner, states, observed, emit, verify, examples=examples)

    native.run_sequence(
        learner,
        report,
        progress,
        training_examples=examples,
        evaluate_sources=evaluate,
        evidence_reuse=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report: dict[str, Any] = dict(
        definition="native_memory_four_arms_v1",
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
        comparison_scope="same task TRAIN information; unequal optimization/storage/compute; known concept overlap",
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
        if len(encoded.encode()) > 4 * 1024**2:
            raise RuntimeError("native comparison report exceeds4MiB")
        args.output.write_text(encoded, encoding="utf8")
        if event["outer_seconds"] > 600 or event["outer_peak_rss_bytes"] > 20 * 1024**3:
            raise RuntimeError("native comparison resource allowance exceeded")

    def observe_model(model):
        api = importlib.import_module("jev.api")
        owner["learner"] = OpenJevTrainingLearner(
            model,
            api.compile_request,
            instructions=native.INSTRUCTIONS,
            policy=OpenJevTrainingPolicy(24, 8),
        )

    try:
        examples = native._training()
        report["model_load_attempts"] = 1
        _load(
            args.source,
            args.checkpoint,
            progress,
            prediction_limit=54,
            forward_limit=78,
            token_limit=99840,
            model_observer=observe_model,
            work_reporter=lambda observe: owner.update(observe=observe),
        )
        run_comparison(owner["learner"], report, progress, training_examples=examples)
        owner["observe"]()
        owner["learner"]._guard()
        report["model_training_after"] = owner["learner"].model.training
        report["observed_work_terminal_status"] = "final_observation_after_all_work"
        report["complete"] = True
        progress("comparison_complete")
    except Exception as error:
        report["complete"] = False
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
