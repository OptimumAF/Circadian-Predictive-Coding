"""Causal TRAIN evidence beside the existing fixed delayed temporal learner.

Readouts use live prefixes; scoring starts only after all TRAIN updates finish.
No final/probe data, native2B graph, permission grant, tuning or promotion.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from scripts.run_temporal_decision_reference import _annotate, _cases
from src.adapters.perceptron_decision import (
    PerceptronDecisionPredictor,
    build_decision_training_batch,
)
from src.adapters.perceptron_learner import PerceptronLearner
from src.app.coupled_attribution import AttributionStream
from src.app.decision_evidence import apply_decision_feedback, decision_from_evidence
from src.app.decision_reference_evaluation import (
    DecisionCase,
    _observed_probability_return,
    _observed_return,
    evaluate_decision_reference,
    evaluate_probability_reference,
)
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.categorical_perceptron import PerceptronPolicy
from src.core.decision_evidence import DecisionEvidenceStore, EvidencePolicy
from src.core.decision_features import DecisionFeatureSpec
from src.core.experience import Experience, ExperiencePermissions, LabelArrival
from src.core.temporal_decision_sequence import (
    CATALOG,
    CHECKPOINTS,
    PHASES,
    VOCABULARY,
    policy_for_phase,
    temporal_examples,
    validate_temporal_examples,
)
from src.core.typed_decision import validate_decision_result, validate_probability_decision_result
from src.shared.process_memory import ProcessRssSampler

MAX_UPDATES = 12


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


def _training_data():
    spec = DecisionFeatureSpec(VOCABULARY, CATALOG)
    packets = tuple(
        temporal_examples(phase, "train", exposure=exposure)
        for phase in PHASES
        for exposure in range(4)
    )
    for packet in packets:
        validate_temporal_examples(packet)
    training = tuple(
        build_decision_training_batch(_cases(packet), spec, packet[0].phase, allow_subsets=True)
        for packet in packets
    )
    return spec, packets, training


def _read_weight_view(spec, state, phase, name, report, budget):
    row: dict[str, Any] = dict(
        control=name, phase=phase, source_updates=state.completed_updates, pairs=[]
    )
    report["weight_views"].append(row)
    predictor = PerceptronDecisionPredictor(spec, state)
    for example in temporal_examples(phase, "outer_selection"):
        pair: dict[str, Any] = dict(request_id=example.request.request_id, calls=[])
        row["pairs"].append(pair)
        for request in (
            example.request,
            replace(
                example.request,
                allowed_candidates=tuple(reversed(example.request.allowed_candidates)),
            ),
        ):
            budget.before_final()
            call: dict[str, Any] = dict(request=asdict(request))
            pair["calls"].append(call)
            if report["readout_work"]["weight_attempted"] >= 80:
                raise ValueError("online weight readout allowance exhausted")
            report["readout_work"]["weight_attempted"] += 1
            supplied = deepcopy(request)
            result = predictor(supplied)
            report["readout_work"]["weight_returned"] += 1
            call["returned"] = _observed_return(result)
            if supplied != request:
                raise ValueError("online weight predictor changed its request")
            validate_decision_result(request, result)
            call["result"] = asdict(result)
            call["_result"] = deepcopy(result)
            report["readout_work"]["weight_validated"] += 1
            budget.before_final()


def _read_memory_view(store, phase, name, cutoff, report, budget):
    row: dict[str, Any] = dict(
        control=name,
        phase=phase,
        cutoff=cutoff,
        policy_version=policy_for_phase(phase)["version"],
        pairs=[],
    )
    report["memory_views"].append(row)
    for example in temporal_examples(phase, "outer_selection"):
        pair: dict[str, Any] = dict(request_id=example.request.request_id, calls=[])
        row["pairs"].append(pair)
        for request in (
            example.request,
            replace(
                example.request,
                allowed_candidates=tuple(reversed(example.request.allowed_candidates)),
            ),
        ):
            budget.before_final()
            if report["readout_work"]["memory_attempted"] >= 48:
                raise ValueError("online memory readout allowance exhausted")
            call: dict[str, Any] = dict(request=asdict(request))
            pair["calls"].append(call)
            report["readout_work"]["memory_attempted"] += 1
            selected = store.retrieve(request, cutoff=cutoff, policy_version=row["policy_version"])
            call["selection"] = asdict(selected)
            result = decision_from_evidence(request, selected)
            if result is None:
                if selected.items:
                    raise ValueError("nonempty evidence produced an unobserved decision")
                call["status"] = "unavailable"
                report["readout_work"]["memory_unavailable"] += 1
            else:
                call["returned"] = _observed_probability_return(result)
                validate_probability_decision_result(request, result)
                call.update(status="answered", result=asdict(result), _result=deepcopy(result))
                report["readout_work"]["memory_validated"] += 1
            report["readout_work"]["memory_completed"] += 1
            budget.before_final()


def _score_view(row, *, memory):
    examples = temporal_examples(row["phase"], "outer_selection")
    by_id = {pair["request_id"]: pair for pair in row["pairs"]}
    answered = tuple(
        e for e in examples if all("_result" in c for c in by_id[e.request.request_id]["calls"])
    )
    cached = [c for e in answered for c in by_id[e.request.request_id]["calls"]]
    position = 0

    def predict(request):
        nonlocal position
        call = cached[position]
        position += 1
        if not _same(asdict(request), call["request"]):
            raise ValueError("scoring request differs from its paid online readout")
        return deepcopy(call["_result"])

    evaluation = None
    if answered:
        evaluator = evaluate_probability_reference if memory else evaluate_decision_reference
        cases = tuple(
            DecisionCase(e.request, e.target_id, e.source_group, e.role) for e in answered
        )
        evaluation = _annotate(evaluator(cases, predict=predict), answered)
        if position != len(cached):
            raise ValueError("scoring did not consume its exact paid readout cache")
    # An abstention has no probability vector. Never invent calibration for it.
    count = len(examples)
    correct = 0 if evaluation is None else sum(c["correct"] for c in evaluation["cases"])
    row["score"] = dict(
        cases=count,
        answered=len(answered),
        correct=correct,
        accuracy_all_cases=correct / count,
        coverage=len(answered) / count,
        answered_only_evaluation=evaluation,
        proper_score_denominator=len(answered),
        qualification="conditional proper scores are not full-coverage comparative scores",
    )
    row["outcomes"] = []
    for example in examples:
        call = by_id[example.request.request_id]["calls"][0]
        selected = call.get("result", {}).get("selected_id")
        row["outcomes"].append(
            dict(
                concept=example.concept,
                domain=example.domain,
                policy_version=example.policy_version,
                target_id=example.target_id,
                answered=selected is not None,
                selected_id=selected,
                correct=selected == example.target_id,
                obsolete_rule_error=example.obsolete_target_id is not None
                and selected == example.obsolete_target_id,
            )
        )
    for pair in row["pairs"]:
        statuses = ["_result" in c for c in pair["calls"]]
        if statuses[0] != statuses[1]:
            raise ValueError("candidate order changed answer availability")


def _summarize(report):
    current = {r["phase"]: r for r in report["memory_views"] if r["control"] == "adaptive_memory"}
    learned = {r["phase"]: r for r in report["weight_views"] if r["control"] == "learned"}
    facts = {phase: {r["concept"]: r for r in view["outcomes"]} for phase, view in current.items()}
    a, b, returned = (facts[p] for p in PHASES)
    return dict(
        by_phase={
            p: dict(
                memory_minus_learned_accuracy=current[p]["score"]["accuracy_all_cases"]
                - learned[p]["score"]["accuracy_all_cases"],
                memory_coverage=current[p]["score"]["coverage"],
                obsolete_rule_errors=sum(r["obsolete_rule_error"] for r in facts[p].values()),
            )
            for _, p in CHECKPOINTS
        },
        parcel_acquisition_after_B=not a["parcel"]["correct"] and b["parcel"]["correct"],
        parcel_retained_at_return=b["parcel"]["correct"] and returned["parcel"]["correct"],
        password_answered=[facts[p]["password"]["answered"] for p in PHASES],
        password_correct=[facts[p]["password"]["correct"] for p in PHASES],
        current_rule_old_domain_return_success=all(
            returned[c]["correct"] for c in ("invoice", "legacy", "password")
        ),
        qualification="policy v1-to-v2 filtering may remove stable-password evidence; availability loss is not ordinary forgetting or demonstrated recovery",
        direction="full accuracy and coverage higher; obsolete errors lower; conditional proper scores require their stated denominator",
    )


def _verify(stream, store, report, expected):
    receipts = [asdict(r) for r in stream.applied_updates]
    if len(receipts) != expected or not _same(receipts, report["receipts"]):
        raise ValueError("online preceding committed receipt evidence differs")
    published = [r["event"] for r in report["feedback"] if "event" in r]
    if not _same(published, [asdict(e) for e in store.events]):
        raise ValueError("online preceding feedback evidence differs")
    for i, receipt in enumerate(receipts):
        if (
            receipt["sample_id"] != f"batch-{i}"
            or receipt["episode_id"] != "attribution"
            or receipt["event_id"] != f"label-{i}"
            or any(
                type(receipt[k]) is not int
                for k in ("observed_at", "arrived_at", "applied_at", "update_number")
            )
            or (
                receipt["observed_at"],
                receipt["arrived_at"],
                receipt["applied_at"],
                receipt["update_number"],
            )
            != (i, i + 1, i + 1, i + 1)
        ):
            raise ValueError("online feedback clocks differ from the fixed delayed stream")


def run_example(*, progress=None):
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable or None")
    started = monotonic()
    report: dict[str, Any] = dict(
        definition="online_temporal_feedback_v1",
        complete=False,
        observations=[],
        receipts=[],
        feedback=[],
        weight_views=[],
        memory_views=[],
        events=[],
        readout_work=dict(
            weight_attempted=0,
            weight_returned=0,
            weight_validated=0,
            memory_attempted=0,
            memory_completed=0,
            memory_validated=0,
            memory_unavailable=0,
        ),
        final_examples=0,
        probe_updates=0,
        replay_examples=0,
        external_actions=0,
        memory_source_scope="new explicitly consented raw TRAIN view; projected array TRAIN consent is unchanged",
        qualification="fixed concept-overlapping development templates; no CPC/generalization/native2B/promotion claim",
    )
    store = DecisionEvidenceStore(EvidencePolicy(64, 65536, 768, 1))
    budget = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=MAX_UPDATES,
            max_wall_seconds=60,
            max_process_rss_bytes=512 * 1024**2,
        ),
        clock=monotonic,
    )
    sampler = ProcessRssSampler()
    states: dict[int, Any] = {}
    sources: dict[str, Any] = {}
    learner = None

    def emit(event):
        budget.before_final()
        report["events"].append(
            dict(event=event, updates=budget.updates_completed, evidence_events=len(store.events))
        )
        if progress is not None:
            progress(event)
        budget.before_final()

    try:
        sampler.sample()
        budget.attach_memory(sampler)
        spec, packets, training = _training_data()
        report["training_cases"] = [[asdict(e) for e in packet] for packet in packets]

        def observe(observation, _candidate):
            index = observation.tick
            packet = packets[index]
            if observation.row_ids != training[index].row_ids or not np.array_equal(
                observation.features, training[index].features
            ):
                raise ValueError("raw memory observations differ from admitted stream features")
            for example in packet:
                identity = example.request.request_id
                if identity in sources:
                    raise ValueError("duplicate raw TRAIN memory observation")
                source = Experience(
                    identity,
                    "attribution",
                    index,
                    "a0",
                    deepcopy(example.request),
                    "train",
                    ExperiencePermissions(training=True, replay=True),
                    tuple(c.candidate_id for c in example.request.allowed_candidates),
                )
                sources[identity] = source
                report["observations"].append(
                    dict(parent_batch_id=f"batch-{index}", source=asdict(source))
                )
            emit("memory_observed_before_label")

        def capture(update, state, _batch):
            states[update] = state

        learner = PerceptronLearner(PerceptronPolicy(len(VOCABULARY), len(CATALOG)))
        stream = AttributionStream(
            learner,
            training,
            (),
            checkpoint_updates=tuple(n for n, _ in CHECKPOINTS),
            label_delay=1,
            budget=budget,
            observe_checkpoint=capture,
            observe_features=observe,
            training_only=True,
        )
        boundaries = dict(CHECKPOINTS)

        def read_boundary(update):
            phase = boundaries[update]
            controls = dict(learned=update, unchanged=0)
            if update >= 8:
                controls["frozen_A"] = 4
            for name, source in controls.items():
                _read_weight_view(spec, states[source], phase, name, report, budget)
            _read_memory_view(store, phase, "adaptive_memory", update, report, budget)
            if update >= 8:
                _read_memory_view(store, phase, "frozen_A_memory", 4, report, budget)
            emit("live_prefix_readouts_finished")

        read_boundary(0)
        for completed in range(1, MAX_UPDATES + 1):
            _verify(stream, store, report, completed - 1)
            stream.run_until(completed)
            receipts = stream.applied_updates
            receipt = receipts[-1]
            report["receipts"].append(asdict(receipt))
            if receipt.update_number != completed or len(receipts) != completed:
                raise ValueError("feedback requires the exact new committed batch")
            packet = packets[completed - 1]
            for example in packet:
                source = sources[example.request.request_id]
                original = next(
                    r["source"]
                    for r in report["observations"]
                    if r["source"]["sample_id"] == source.sample_id
                )
                if not _same(original, asdict(source)) or source.observed_at != receipt.observed_at:
                    raise ValueError("raw consented observation changed before feedback")
                label = LabelArrival(
                    f"{receipt.event_id}-{source.sample_id}",
                    source.sample_id,
                    source.episode_id,
                    receipt.arrived_at,
                    receipt.actor_version,
                    example.target_id,
                )
                row: dict[str, Any] = dict(
                    parent_update=completed,
                    source_id=source.sample_id,
                    label=asdict(label),
                    recorded_at=receipt.applied_at,
                )
                report["feedback"].append(row)
                event = apply_decision_feedback(
                    store,
                    source,
                    label,
                    policy_version=example.policy_version,
                    source_group=example.source_group,
                    recorded_at=receipt.applied_at,
                )
                row["event"] = asdict(event)
            emit("feedback_published_after_commit")
            _verify(stream, store, report, completed)
            if completed in boundaries:
                read_boundary(completed)
        report["stream"] = stream.finish()
        emit("training_finished_before_scoring")
        for row in report["weight_views"]:
            _score_view(row, memory=False)
        for row in report["memory_views"]:
            _score_view(row, memory=True)
        report["summary"] = _summarize(report)
        report["complete"] = True
        emit("online_case_complete")
    except Exception as error:
        report["complete"] = False
        report["error"] = type(error).__name__ + ": " + str(error)[:512]
        report["failure_scope"] = (
            "retained paid prefixes are lower bounds; no continuation after failure"
        )
    finally:
        # Retain paid stores/returns even if the last write or score loses its return.
        for row in (*report["weight_views"], *report["memory_views"]):
            for pair in row["pairs"]:
                for call in pair["calls"]:
                    call.pop("_result", None)
        report["evidence_events"] = [asdict(e) for e in store.events]
        report["evidence_work"] = store.work
        report["evidence_storage_bytes"] = store.storage_bytes
        report["snapshots"] = [dict(update=n, state=asdict(s)) for n, s in states.items()]
        original_state = None
        if learner is not None:
            try:
                original_state = asdict(learner.snapshot_state())
            except Exception as error:
                original_state = dict(
                    status="unavailable", error=type(error).__name__ + ": " + str(error)[:256]
                )
        report["work"] = dict(
            committed_weight_updates=budget.updates_completed,
            committed_training_rows=3 * budget.updates_completed,
            latest_observed_candidate_state=None if not states else asdict(states[max(states)]),
            original_source_state=original_state,
            state_observation_scope="latest fixed candidate checkpoint; lower bound between checkpoints on failure",
            retained_coefficient_bytes=160 * len(states),
            model_random_draws=0,
            elapsed_seconds=monotonic() - started,
            cost_scope="logical coefficients/readouts plus separate measured execution; stream copies/validation/softmax/Python additional",
        )
        if sampler.start_bytes is not None:
            sampler.sample()
            report["work"]["process_rss_observation"] = asdict(sampler.snapshot())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    # Refuse replacement before any model, store or runtime is created.
    with args.output.open("x", encoding="utf8") as target:
        target.write("{}\n")
    report = run_example()
    encoded = (json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n").encode("utf8")
    if len(encoded) > 512 * 1024:
        raise ValueError("online temporal report exceeds512KiB")
    args.output.write_bytes(encoded)
    print(
        json.dumps(
            dict(
                complete=report["complete"], output_bytes=len(encoded), work=report["readout_work"]
            )
        )
    )
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
