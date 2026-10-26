"""Controlled TRAIN-evidence correction with a reused frozen linear predictor.

Runs fixed retrieval episodes and separate context callbacks. No model fitting,
native model, final/probe access, target-based output repair or comparative study.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from time import monotonic
from typing import Any

from scripts.run_ticket_workflow import training_cases
from src.adapters.perceptron_decision import PerceptronDecisionPredictor
from src.app.decision_evidence import EvidenceDecisionPredictor
from src.app.ticket_workflow import run_workflow_episode
from src.core.categorical_perceptron import PerceptronPolicy, PerceptronSnapshot
from src.core.decision_evidence import DecisionEvidence, DecisionEvidenceStore, EvidencePolicy
from src.core.decision_features import DecisionFeatureSpec
from src.core.experience import ExperiencePermissions
from src.core.ticket_workflow import (
    ACTION_IDS,
    WorkflowCapabilities,
    WorkflowState,
    workflow_request,
)
from src.core.typed_decision import DecisionCandidate, validate_decision_result
from src.shared.process_memory import ProcessRssSampler

REFERENCE_SHA = "9a7c6325cc76b42688f2bbb8ddb6b9449a2d9505a6486f21fd0b1a99bf1d6f62"


def _reference(path):
    if path.stat().st_size > 512 * 1024:
        raise ValueError("reference exceeds512KiB")
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != REFERENCE_SHA:
        raise ValueError("reference is not the accepted frozen local case")
    saved = json.loads(body)
    spec = DecisionFeatureSpec(
        tuple(saved["feature_spec"]["vocabulary"]),
        tuple(DecisionCandidate(**c) for c in saved["feature_spec"]["catalog"]),
    )
    state = next(row["state"] for row in saved["snapshots"] if row["update"] == 8)
    state = PerceptronSnapshot(
        PerceptronPolicy(**state["policy"]),
        tuple(tuple(row) for row in state["coefficients"]),
        state["completed_updates"],
        state["training_examples"],
        state["mistake_updates"],
    )
    return PerceptronDecisionPredictor(spec, state)


def run_example(reference, progress=lambda report: None):
    started = monotonic()
    report: dict[str, Any] = dict(
        definition="editable_decision_evidence_v1",
        complete=False,
        reference_sha256=REFERENCE_SHA,
        evidence_receipts=[],
        episodes=[],
        context_calls=[],
        frozen_calls=[],
        new_training_updates=0,
        final_examples=0,
        probe_updates=0,
        replay_optimization_rows=0,
        external_actions=0,
        training_case_generation=dict(
            demonstration_transitions=0,
            candidate_constructions=0,
            catalog_entries_considered=0,
            candidates_returned=0,
        ),
        interpretation="controlled declared TRAIN-label correction, not a matched study or native parcel repair",
    )
    sampler = ProcessRssSampler()
    store = DecisionEvidenceStore(EvidencePolicy(16, 16384, 768, 1))
    predictor = None
    frozen_before = None
    case_finished = False

    def check_resources():
        sampler.sample()
        report["elapsed_seconds"] = monotonic() - started
        if report["elapsed_seconds"] > 30 or sampler.snapshot().peak_bytes > 512 * 1024**2:
            raise RuntimeError("evidence functional time/RSS allowance exceeded")

    def checkpoint():
        check_resources()
        report["store_work"] = store.work
        report["storage_bytes"] = store.storage_bytes
        progress(report)
        check_resources()

    def frozen(request):
        if predictor is None:
            raise ValueError("frozen predictor is not admitted")
        if len(report["frozen_calls"]) >= 8:
            raise ValueError("frozen readout allowance exhausted")
        row: dict[str, Any] = dict(request=asdict(request))
        report["frozen_calls"].append(row)
        if asdict(predictor.state) != frozen_before:
            raise ValueError("frozen model changed before readout")
        result = predictor(request)
        row["returned"] = asdict(result)
        validate_decision_result(request, result)
        if asdict(predictor.state) != frozen_before:
            raise ValueError("frozen model changed after readout")
        row["validated"] = True
        return result

    try:
        sampler.sample()
        predictor = _reference(reference)
        frozen_before = asdict(predictor.state)
        report["frozen_state_before"] = frozen_before
        cases = training_cases()
        report["training_case_generation"] = dict(
            demonstration_transitions=12,
            candidate_constructions=12,
            catalog_entries_considered=72,
            candidates_returned=60,
        )
        for index, case in enumerate(cases):
            target = "inspect" if index == 7 else case.target_id
            receipt = store.add(
                DecisionEvidence(
                    f"evidence-{index}",
                    0,
                    1,
                    10,
                    "v2",
                    "train-evidence-case",
                    case.request,
                    target,
                    permissions=ExperiencePermissions(training=True, replay=True),
                )
            )
            report["evidence_receipts"].append(asdict(receipt))
        report["controlled_error"] = dict(
            evidence_id="evidence-7",
            target_before="inspect",
            declared_train_correction="defer",
            provenance="new controlled evidence-case records; original TRAIN receipts unchanged",
        )
        checkpoint()
        request = workflow_request(
            WorkflowState("development-ready", "parcel", revision=1, phase="ready"),
            WorkflowCapabilities(ACTION_IDS),
            "development-parcel",
        )
        request = replace(request, query=request.query + " please")
        report["baseline"] = [
            asdict(frozen(query))
            for query in (
                request,
                replace(request, allowed_candidates=tuple(reversed(request.allowed_candidates))),
            )
        ]
        for cutoff in (10, 20):
            if cutoff == 20:
                report["edit_receipt"] = asdict(
                    store.edit(
                        "evidence-7", "defer", available_at=20, recorded_at=20, expected_revision=0
                    )
                )
                checkpoint()
            retrieval = EvidenceDecisionPredictor(
                store, cutoff=cutoff, policy_version="v2", max_calls=6
            )
            episode = run_workflow_episode(
                WorkflowState(f"development-{cutoff}", "parcel"),
                retrieval,
                capabilities=WorkflowCapabilities(ACTION_IDS),
                max_steps=6,
                scenario="stale_once",
            )
            report["episodes"].append(
                dict(
                    cutoff=cutoff,
                    role="outer_selection",
                    source_group="development-evidence-please",
                    episode=episode,
                    evidence_predictor=retrieval.report,
                )
            )
            checkpoint()
            if not episode["complete"]:
                raise RuntimeError("evidence episode has uncertain paid work")
            context = EvidenceDecisionPredictor(
                store,
                cutoff=cutoff,
                policy_version="v2",
                mode="context",
                predict=frozen,
                max_calls=2,
            )
            try:
                for query in (
                    request,
                    replace(
                        request, allowed_candidates=tuple(reversed(request.allowed_candidates))
                    ),
                ):
                    context(query)
            finally:
                report["context_calls"].append(context.report)
                checkpoint()
        past = store.retrieve(request, cutoff=10, policy_version="v2")
        unavailable = store.retrieve(request, cutoff=9, policy_version="v2")
        report["past_view_after_edit"] = asdict(past)
        report["pre_recording_view"] = asdict(unavailable)
        report["summary"] = dict(
            retrieval_episode_success=[r["episode"]["success"] for r in report["episodes"]],
            retrieval_successful_recovery=[
                r["episode"]["error_recovery_demonstrated"] for r in report["episodes"]
            ],
            context_selected=[
                [c["result"]["selected_id"] for c in row["calls"]]
                for row in report["context_calls"]
            ],
            original_baseline_selected=[r["selected_id"] for r in report["baseline"]],
            past_target=past.items[0].evidence.target_id,
            future_records_visible_early=len(unavailable.items),
        )
        case_finished = True
    except Exception as error:
        report["error"] = type(error).__name__ + ": " + str(error)[:512]
    finally:
        if predictor is not None:
            report["frozen_state_after"] = asdict(predictor.state)
            report["weights_and_history_unchanged"] = frozen_before == report["frozen_state_after"]
        report["store_events"] = [asdict(e) for e in store.events]
        report["store_work"] = store.work
        report["storage_bytes"] = store.storage_bytes
        report["elapsed_seconds"] = monotonic() - started
        report["frozen_readouts_attempted"] = len(report["frozen_calls"])
        report["validated_affine_terms"] = 40 * sum(
            c.get("validated", False) for c in report["frozen_calls"]
        )
        try:
            check_resources()
            report["rss_observation"] = asdict(sampler.snapshot())
            report["complete"] = case_finished
            progress(report)
            check_resources()
        except Exception as error:
            report["complete"] = False
            report["error"] = type(error).__name__ + ": " + str(error)[:512]
            progress(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    with args.output.open("x", encoding="utf8") as stream:
        stream.write("{}\n")

    def progress(report):
        encoded = json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n"
        if len(encoded.encode()) > 512 * 1024:
            raise ValueError("evidence report exceeds512KiB")
        args.output.write_text(encoded, encoding="utf8")

    report = run_example(args.reference, progress)
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
