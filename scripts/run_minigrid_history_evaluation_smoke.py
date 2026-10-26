"""Exercise trained public-history policies through the existing native evaluator.

One fixed local engineering case, not a learning comparison or final evaluation.
Existing streams own updates; the evaluator owns target-free physical episodes.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
import pickle
from math import fsum, isfinite
from pathlib import Path
from time import monotonic
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from src.adapters import minigrid_learning_pilot as shared
from src.adapters.minigrid_learnability import _history_packets, _history_learner, _score_checkpoint
from src.adapters.minigrid_policy_evaluation import evaluate_checkpoint, MiniGridEvaluationFailed
from src.core.minigrid_key_door import predefined_layouts
from src.shared.process_memory import ProcessRssSampler
from src.shared.torch_runtime import require_torch

BODY_SECONDS = 90.0
RSS_LIMIT = 768 * 1024**2
CANONICAL_NATIVE_METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")
CANONICAL_METHODS = CANONICAL_NATIVE_METHODS + tuple(
    f"public_invariant_{method}" for method in CANONICAL_NATIVE_METHODS
)


def describe_protocol(
    *, fixed_a_acquisition: bool = False, canonical_neural_comparison: bool = False
) -> dict[str, Any]:
    if type(fixed_a_acquisition) is not bool or type(canonical_neural_comparison) is not bool:
        raise ValueError("recipe mode flags must be exact bools")
    if fixed_a_acquisition and canonical_neural_comparison:
        raise ValueError("fixed acquisition and canonical comparison are mutually exclusive")
    if canonical_neural_comparison:
        return _canonical_protocol()
    protocol = dict(
        protocol_id="minigrid_trained_history_live_boundary_v1",
        classification="engineering_trained_target_free_boundary_not_acquisition_or_comparison",
        methods=list(shared.METHODS),
        seed=17,
        hidden_dim=8,
        num_classes=7,
        feature_dim_by_method={m: 148 if m == "recurrent_bc" else 296 for m in shared.METHODS},
        learning_rate=0.03,
        momentum=0.9,
        inference_steps=2,
        inference_learning_rate=0.04,
        cpc_configs=shared.describe_protocol()["cpc_configs"],
        training="A32 alternating original teacher cues through existing delayed AttributionStream; all five streams before any readout",
        checkpoints=[0, 32],
        label_delay=1,
        role="train",
        final_episodes=0,
        probe_updates=0,
        readout="all six known TRAIN packets plus six target-free live episodes per learner; B/C not trained",
        action_rule="all7 native actions, first argmax, no mask/repair/retry",
        history="each FF live episode starts with its own first public148+current148; RNN148 fresh session",
        planned_work=dict(
            native_training_updates=160,
            training_rows=1440,
            raw_score_forwards=30,
            raw_score_rows=270,
            live_episodes=30,
            max_live_predictions=480,
            max_live_physics=480,
            teacher_episodes=6,
            expected_teacher_physics=54,
            max_teacher_physics=96,
            max_total_physics=576,
            max_environment_resets=36,
            recurrent_training_rows=288,
            recurrent_raw_rows=54,
            max_recurrent_live_steps=96,
        ),
        limits=dict(
            body_seconds=90,
            parent_seconds=120,
            process_rss_bytes=RSS_LIMIT,
            output_bytes=16 * 1024**2,
            cpu_threads=1,
            interop_threads=1,
            gpu=0,
        ),
        limitations="one fixed seed/engineering dose; raw fit and live success distinct; no final/probe/replay/RL/selection/promotion/tuning; no acquisition/retention/recovery/generalization/CPC benefit established",
    )
    if fixed_a_acquisition:
        protocol.update(
            protocol_id="minigrid_fixed_a128_acquisition_v1",
            classification="exploratory_single_seed_fixed_total_dose_acquisition_control",
            training="A128 alternating original TRAIN cues with unique packet IDs through existing delayed AttributionStream; all five streams before readout",
            checkpoints=list(shared.POINTS),
            readout="only A: raw0/32/64/96/128, target-free live0/128 both cues; original six-source collection charged including unused B/C",
            acquisition="at128 both raw exact sequences AND both native goals; report each condition separately; missing/incomplete is null; all five arms run regardless",
            metric_direction="CE lower; raw accuracy/exact, native success/return higher; gain=mean CE0-CE128; active-minus-PC/neutral gains; live success-count change and mean-return gain",
            dose_rationale="same TOTAL128 dose as original four-phase stream, not extension until acquisition; no automatic dose/seed/protocol adjustment",
            planned_work=dict(
                native_training_updates=640,
                training_rows=5760,
                raw_score_forwards=50,
                raw_score_rows=450,
                live_episodes=20,
                max_live_predictions=320,
                max_live_physics=320,
                teacher_episodes=6,
                expected_teacher_physics=54,
                max_teacher_physics=96,
                max_total_physics=416,
                max_environment_resets=26,
                recurrent_training_rows=1152,
                recurrent_raw_rows=90,
                max_recurrent_live_steps=64,
            ),
            limits=dict(
                body_seconds=180,
                parent_seconds=210,
                process_rss_bytes=RSS_LIMIT,
                output_bytes=16 * 1024**2,
                cpu_threads=1,
                interop_threads=1,
                gpu=0,
            ),
            limitations="one fixed seed/dose; 128 A episodes versus64 in original four-phase128 does not isolate interference; no retention/recovery/generalization/equivalence/CPC advantage; no final/probe/replay/RL/selection/promotion/tuning; preserve failed acquisition",
        )
    return protocol


def _canonical_protocol() -> dict[str, Any]:
    from src.core.minigrid_invariant_encoding import INVARIANT_HISTORY_ENCODING_ID

    return dict(
        protocol_id="minigrid_matched_raw_canonical_A32_v1",
        classification="exploratory_known_A_previously_inspected_E_nuisance_transfer",
        methods=list(CANONICAL_METHODS),
        encoding_id=INVARIANT_HISTORY_ENCODING_ID,
        feature_dim=296,
        hidden_dim=8,
        num_classes=7,
        seed=17,
        learning_rate=0.03,
        momentum=0.9,
        inference_steps=2,
        inference_learning_rate=0.04,
        cpc_configs=shared.describe_protocol()["cpc_configs"],
        training="32 alternating original nine-row TRAIN A-blue/red packets per arm; all eight delayed streams before any readout",
        checkpoints=[0, 32],
        label_delay=1,
        training_only=True,
        heldout=[],
        readout_order=[[method, point] for method in CANONICAL_METHODS for point in (0, 32)],
        slots=[
            ["A", "blue", "train"],
            ["A", "red", "train"],
            ["E-blue", "blue", "probe_test"],
            ["E-red", "red", "probe_test"],
        ],
        max_steps=16,
        action_rule="raw seven-action first argmax; no mask/repair/retry",
        planned_work=dict(
            native_training_updates=256,
            training_rows=2304,
            raw_score_forwards=0,
            teacher_episodes=0,
            live_episodes=64,
            max_live_predictions=1024,
            max_live_physics=1024,
            max_environment_resets=64,
        ),
        limits=dict(
            body_seconds=300,
            parent_seconds=330,
            shared_cleanup_seconds=340,
            process_rss_bytes=RSS_LIMIT,
            output_bytes=32 * 1024**2,
            cpu_threads=1,
            interop_threads=1,
            gpu=0,
        ),
        metric_direction="successes/return higher; steps descriptive; end-minus-initial, canonical-minus-raw endpoints/gains, active-minus-PC/neutral gains within each representation",
        limitations="known A/previously inspected E exploratory; same dose/shape is not equal compute or total memory; no final D/replay/promotion/tuning, independent generalization, retention/return learning, useful RL or CPC advantage claim",
    )


def _comparison_inputs(sources):
    from src.core.demonstration_episode import copy_episode

    if type(sources) is not tuple or len(sources) != 2:
        raise ValueError("canonical comparison requires the original two TRAIN A episodes")
    owned = tuple(copy_episode(episode) for episode in sources)
    for cue, episode in zip(("blue", "red"), owned, strict=True):
        if (
            (episode.episode_id, episode.context, episode.role, episode.num_classes)
            != (f"minigrid-source-A-{cue}", "A", "train", 7)
            or len(episode.transitions) != 9
            or len(episode.transitions[0].observation) != 148
        ):
            raise ValueError("canonical comparison requires original nine-row TRAIN A-blue/red")
    raw_sources = tuple(shared._batch(episode) for episode in owned)
    packets = tuple(
        shared._batch(replace(owned[i % 2], episode_id=f"minigrid-training-{i}")) for i in range(32)
    )
    history = _history_packets(packets)
    # Admit discrete public coordinates now, before any model/resource creation.
    from src.core.minigrid_invariant_encoding import canonicalize_public_history

    for packet in history:
        for row in packet.features:
            canonicalize_public_history(tuple(float(value) for value in row))
    return owned, raw_sources, packets, history


def _canonical_learner(method):
    from src.adapters.invariant_torch_feature_learner import (
        InvariantTorchFeatureLearner,
        InvariantTorchFeaturePolicy,
    )
    from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy

    bound = method.startswith("public_invariant_")
    native = method.removeprefix("public_invariant_")
    if method not in CANONICAL_METHODS:
        raise ValueError("undeclared canonical comparison method")
    policy = TorchFeaturePolicy(native, 296, 8, 17, num_classes=7)
    return (
        InvariantTorchFeatureLearner(InvariantTorchFeaturePolicy(policy))
        if bound
        else TorchFeatureLearner(policy)
    )


def _validate_representation_readout(record) -> None:
    """Validate observed metrics before admitting another paid readout."""
    episodes = record["episodes"]
    if (
        len(episodes) != 4
        or any(slot["status"] != "complete" for slot in episodes)
        or [(slot["context"], slot["cue"], slot["source_role"]) for slot in episodes]
        != [tuple(slot) for slot in _canonical_protocol()["slots"]]
    ):
        raise ValueError("canonical comparison live identities/evidence changed")
    for slot in episodes:
        value, steps = slot["return"], slot["environment"]["environment_steps"]
        if type(slot["success"]) is not bool:
            raise ValueError("canonical comparison success must be observed bool")
        if type(value) not in (int, float) or not isfinite(value):
            raise ValueError("canonical comparison return must be finite native numeric evidence")
        if type(steps) is not int or not 1 <= steps <= 16:
            raise ValueError("canonical comparison steps must be observed native int in1..16")


def _failure_numbers(value):
    """Keep invalid numbers visible in failed JSON without treating them as metrics.

    Only failure publication uses these explicit markers. Original private paid
    captures remain untouched; no learner state or finite charge is repaired.
    """
    if type(value) is float and not isfinite(value):
        return {"invalid_native_float": repr(value)}
    if type(value) is dict:
        return {key: _failure_numbers(item) for key, item in value.items()}
    if type(value) is list:
        return [_failure_numbers(item) for item in value]
    if type(value) is tuple:
        return tuple(_failure_numbers(item) for item in value)
    return value


def _representation_outcomes(evaluations, *, complete):
    metrics = ("successes", "mean_return", "mean_steps")
    outcomes = []
    for method in CANONICAL_METHODS:
        records = [record for record in evaluations if record["method"] == method]
        by_point = {record["updates"]: record for record in records}
        available = (
            complete
            and len(records) == 2
            and set(by_point) == {0, 32}
            and all(record["status"] == "complete" for record in records)
        )
        row: dict[str, Any] = dict(
            method=method, status="complete" if available else "unavailable", A=None, E=None
        )
        if available:
            for context, positions in (("A", (0, 1)), ("E", (2, 3))):
                endpoints = []
                for point in (0, 32):
                    _validate_representation_readout(by_point[point])
                    episodes = by_point[point]["episodes"]
                    if (
                        len(episodes) != 4
                        or any(slot["status"] != "complete" for slot in episodes)
                        or [
                            (slot["context"], slot["cue"], slot["source_role"]) for slot in episodes
                        ]
                        != [tuple(slot) for slot in _canonical_protocol()["slots"]]
                    ):
                        raise ValueError("canonical comparison live identities/evidence changed")
                    slots = [episodes[index] for index in positions]
                    if any(type(slot["success"]) is not bool for slot in slots):
                        raise ValueError("canonical comparison success must be observed bool")
                    endpoints.append(
                        dict(
                            successes=sum(slot["success"] for slot in slots),
                            mean_return=fsum(slot["return"] for slot in slots) / 2,
                            mean_steps=fsum(
                                slot["environment"]["environment_steps"] for slot in slots
                            )
                            / 2,
                        )
                    )
                row[context] = dict(
                    initial=endpoints[0],
                    end=endpoints[1],
                    gain={
                        metric: endpoints[1][metric] - endpoints[0][metric] for metric in metrics
                    },
                )
        outcomes.append(row)
    by_method = {row["method"]: row for row in outcomes}

    def difference(left, right, points):
        return {
            context: {
                point: {
                    metric: (
                        left[context][point][metric] - right[context][point][metric]
                        if left["status"] == right["status"] == "complete"
                        else None
                    )
                    for metric in metrics
                }
                for point in points
            }
            for context in ("A", "E")
        }

    canonical_minus_raw = {
        method: difference(
            by_method[f"public_invariant_{method}"], by_method[method], ("end", "gain")
        )
        for method in CANONICAL_NATIVE_METHODS
    }
    active_minus_controls = {
        name: {
            control: difference(
                by_method[prefix + "gated_cpc"], by_method[prefix + control], ("gain",)
            )
            for control in ("pc", "neutral_cpc")
        }
        for name, prefix in (("raw", ""), ("canonical", "public_invariant_"))
    }
    return dict(
        methods=outcomes,
        canonical_minus_raw=canonical_minus_raw,
        active_minus_controls=active_minus_controls,
    )


def _fixed_a_packets(sources):
    # Reuse original collection admission before making a distinct fixed-dose stream.
    initial = shared._packets(sources)[:32]
    return tuple(
        shared._batch(
            replace(shared._episode(initial[i % 32]), episode_id=f"minigrid-fixed-a-training-{i}")
        )
        for i in range(128)
    )


def _acquisition_outcomes(evaluations, *, complete: bool = True) -> dict[str, Any]:
    outcomes: list[dict[str, Any]] = []
    for method in shared.METHODS:
        records = [r for r in evaluations if r["method"] == method]
        by_point = {r["updates"]: r for r in records}
        available = (
            complete
            and len(records) == 5
            and tuple(sorted(by_point)) == shared.POINTS
            and all(r["status"] == "complete" for r in records)
        )
        row: dict[str, Any] = dict(
            method=method,
            status="complete" if available else "unavailable",
            raw_both_cues_exact=None,
            live_both_cues_goals=None,
            joint_acquisition=None,
            ce_gain=None,
            live_success_count_change=None,
            mean_return_gain=None,
        )
        if available:
            initial, final = by_point[0], by_point[128]
            for record in records:
                if [(s["context"], s["cue"]) for s in record["scores"]] != [
                    ("A", "blue"),
                    ("A", "red"),
                ]:
                    raise ValueError("fixed-A raw readout identities changed")
            for record in (initial, final):
                if len(record["episodes"]) != 2 or any(
                    e["status"] != "complete" for e in record["episodes"]
                ):
                    raise ValueError("fixed-A endpoint live evidence incomplete")
            raw = all(s["exact_sequence_accuracy"] == 1 for s in final["scores"])
            live = all(e["success"] is True for e in final["episodes"])
            row.update(
                raw_both_cues_exact=raw,
                live_both_cues_goals=live,
                joint_acquisition=raw and live,
                ce_gain=fsum(s["ce"] for s in initial["scores"]) / 2
                - fsum(s["ce"] for s in final["scores"]) / 2,
                live_success_count_change=sum(e["success"] for e in final["episodes"])
                - sum(e["success"] for e in initial["episodes"]),
                mean_return_gain=fsum(e["return"] for e in final["episodes"]) / 2
                - fsum(e["return"] for e in initial["episodes"]) / 2,
            )
        outcomes.append(row)
    active = outcomes[3]
    comparisons = {}
    for control in (outcomes[1], outcomes[2]):
        comparisons[f"gated_cpc-minus-{control['method']}"] = {
            metric: (
                active[metric] - control[metric]
                if active["status"] == control["status"] == "complete"
                else None
            )
            for metric in ("ce_gain", "live_success_count_change", "mean_return_gain")
        }
    return dict(methods=outcomes, signed_comparisons=comparisons)


def run_smoke(
    *,
    fixed_a_acquisition: bool = False,
    canonical_neural_comparison: bool = False,
    sources=None,
) -> dict[str, Any]:
    if type(fixed_a_acquisition) is not bool or type(canonical_neural_comparison) is not bool:
        raise ValueError("recipe mode flags must be exact bools")
    if fixed_a_acquisition and canonical_neural_comparison:
        raise ValueError("fixed acquisition and canonical comparison are mutually exclusive")
    started = monotonic()
    protocol = (
        describe_protocol(canonical_neural_comparison=True)
        if canonical_neural_comparison
        else describe_protocol(fixed_a_acquisition=True)
        if fixed_a_acquisition
        else describe_protocol()
    )
    if not canonical_neural_comparison and sources is not None:
        raise ValueError("original source arguments are only admitted by canonical comparison")
    comparison_inputs = _comparison_inputs(sources) if canonical_neural_comparison else None
    supplied_sources = sources
    supplied_binding = () if sources is None else tuple(id(value) for value in sources)
    supplied_bytes = None if sources is None else pickle.dumps(sources, protocol=5)
    methods = CANONICAL_METHODS if canonical_neural_comparison else shared.METHODS
    report: dict[str, Any] = dict(
        protocol=protocol,
        status="incomplete",
        reason=None,
        methods=[shared._row(method) for method in methods],
        evaluations=[],
        source_reports=[],
        training_sources=[],
        observed_process_rss=None,
    )
    body_seconds = report["protocol"]["limits"]["body_seconds"]
    checkpoints = tuple(report["protocol"]["checkpoints"])
    arms: list[dict[str, Any]] = []
    original_rows = tuple(report["methods"])
    original_methods, original_evaluations = report["methods"], report["evaluations"]
    completed_evaluations: list[tuple[Any, bytes]] = []
    paid_training: dict[int, bytes] = {}
    evidence = {"protocol": (report["protocol"], pickle.dumps(report["protocol"], protocol=5))}
    readout_budget = None
    readout_budget_guard = None
    active_evaluation = None
    private_planned = []
    finished = False
    active_training = None
    expected_reason = None
    arm_bindings: list[Any] = []
    initial_rows = {id(row): pickle.dumps(row, protocol=5) for row in original_rows}
    initial_signatures = (
        []
        if comparison_inputs is None
        else [(values, shared._signature(values)) for values in comparison_inputs[1:]]
    )

    def bind(key, value):
        report[key] = value
        evidence[key] = (value, pickle.dumps(value, protocol=5))

    def failure(error):
        nonlocal expected_reason
        reason = f"{type(error).__name__}: {error}"
        if canonical_neural_comparison and expected_reason is not None:
            if reason != expected_reason:
                report.setdefault("additional_failures", []).append(reason)
        else:
            expected_reason = reason
        report["reason"] = expected_reason
        if canonical_neural_comparison:
            report["status"] = "incomplete"

    if canonical_neural_comparison:
        private_planned = [
            dict(method=method, updates=point, status="not_run", reason="not reached")
            for method in methods
            for point in (0, 32)
        ]
        bind("planned_evaluations", deepcopy(private_planned))
        bind("initial_pair_parity", [])

    def restore_publication():
        report["methods"] = [
            pickle.loads(paid_training[id(row)]) if id(row) in paid_training else row
            for row in original_rows
        ]
        report["evaluations"] = [pickle.loads(payload) for _, payload in completed_evaluations]
        for key, (_, payload) in evidence.items():
            report[key] = pickle.loads(payload)

    sampler = None
    try:
        torch = require_torch()
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        with ProcessRssSampler() as sampler:
            signatures: list[Any] = list(initial_signatures)

            def common():
                sampler.sample()
                if (
                    monotonic() - started > body_seconds
                    or sampler.snapshot().peak_bytes > RSS_LIMIT
                ):
                    raise ValueError("original whole-case resource limit exhausted")
                if any(
                    not shared._check_signature(batches, expected)
                    for batches, expected in signatures
                ):
                    raise ValueError("original source or training inputs changed")
                if canonical_neural_comparison and (
                    type(supplied_sources) is not tuple
                    or tuple(id(value) for value in supplied_sources) != supplied_binding
                    or pickle.dumps(supplied_sources, protocol=5) != supplied_bytes
                ):
                    raise ValueError("original saved TRAIN episode inputs changed")
                if canonical_neural_comparison:
                    if (
                        type(report["status"]) is not str
                        or report["status"] != "incomplete"
                        or report["reason"] != expected_reason
                    ):
                        raise ValueError("comparison completion status changed before publication")
                    for row in original_rows:
                        if id(row) != active_training and pickle.dumps(
                            row, protocol=5
                        ) != paid_training.get(id(row), initial_rows[id(row)]):
                            raise ValueError("preceding or planned training report changed")
                    if len(arms) != len(arm_bindings) or any(
                        arm is not original or any(arm[key] is not value for key, value in bindings)
                        for arm, (original, bindings) in zip(arms, arm_bindings, strict=True)
                    ):
                        raise ValueError("original training source/state/budget bindings changed")
                if (
                    report["methods"] is not original_methods
                    or len(original_methods) != len(methods)
                    or any(
                        current is not expected
                        for current, expected in zip(original_methods, original_rows, strict=True)
                    )
                    or tuple(row["method"] for row in original_rows) != methods
                ):
                    raise ValueError("original method records changed")
                if any(
                    report[key] is not original or pickle.dumps(report[key], protocol=5) != payload
                    for key, (original, payload) in evidence.items()
                ):
                    raise ValueError("original collection/protocol evidence changed")
                if (
                    report["evaluations"] is not original_evaluations
                    or len(completed_evaluations)
                    > (16 if canonical_neural_comparison else (25 if fixed_a_acquisition else 5))
                    or len(original_evaluations) != len(completed_evaluations)
                    or any(
                        current is not original or pickle.dumps(current, protocol=5) != payload
                        for current, (original, payload) in zip(
                            original_evaluations, completed_evaluations, strict=True
                        )
                    )
                ):
                    raise ValueError("preceding paid evaluation evidence changed")
                if readout_budget_guard is not None:
                    readout_budget_guard()

            def completed():
                common()
                for arm in arms:
                    arm["guard"]()
                common()

            try:
                if canonical_neural_comparison:
                    assert comparison_inputs is not None
                    _, sources, packets, history = comparison_inputs
                else:
                    sources = shared._collect_sources(common, report["source_reports"])
                bind("source_reports", report["source_reports"])
                if not canonical_neural_comparison:
                    packets = (
                        _fixed_a_packets(sources)
                        if fixed_a_acquisition
                        else shared._packets(sources)[:32]
                    )
                    history = _history_packets(packets)
                raw_sources = sources[:2] if fixed_a_acquisition else sources
                history_sources = (
                    _history_packets(raw_sources)
                    if fixed_a_acquisition or canonical_neural_comparison
                    else ()
                )
                signatures.extend((b, shared._signature(b)) for b in (sources, packets, history))
                if fixed_a_acquisition or canonical_neural_comparison:
                    signatures.extend(
                        (b, shared._signature(b)) for b in (raw_sources, history_sources)
                    )
                    bind(
                        "retained_input_array_bytes",
                        {
                            name: sum(p.features.nbytes + p.targets.nbytes for p in values)
                            for name, values in (
                                ("original_sources", sources),
                                ("training_raw", packets),
                                ("training_history", history),
                                ("raw_history_a", history_sources),
                            )
                        },
                    )
                bind("training_sources", [asdict(shared._episode(p)) for p in sources])
                for method, row in zip(methods, original_rows, strict=True):
                    completed()
                    # Capture only earlier peers: guards must not recursively call themselves.
                    prior = tuple(a["guard"] for a in arms)

                    def before(prior=prior):
                        common()
                        for guard in prior:
                            guard()
                        if canonical_neural_comparison:
                            common()

                    if canonical_neural_comparison:
                        active_training = id(row)
                    arms.append(
                        shared._train_arm(
                            row,
                            packets if method == "recurrent_bc" else history,
                            sampler,
                            started + body_seconds,
                            before,
                            learner_factory=_canonical_learner
                            if canonical_neural_comparison
                            else _history_learner,
                            checkpoint_updates=checkpoints,
                        )
                    )
                    if canonical_neural_comparison:
                        arm_bindings.append(
                            (
                                arms[-1],
                                tuple(
                                    (key, arms[-1][key])
                                    for key in ("source", "states", "budget", "guard")
                                ),
                            )
                        )
                        # Preserve this returned paid training report even if its
                        # first completed guard later discovers callback damage.
                        paid_training[id(row)] = pickle.dumps(row, protocol=5)
                        active_training = None
                    arms[-1]["guard"]()
                    paid_training[id(row)] = pickle.dumps(row, protocol=5)
                completed()
                if canonical_neural_comparison:
                    for arm in arms:
                        if (
                            type(arm["states"]) not in (list, tuple)
                            or len(arm["states"]) != 2
                            or any(
                                type(point) is not int or point != expected
                                for (point, _), expected in zip(arm["states"], (0, 32), strict=True)
                            )
                        ):
                            raise ValueError(
                                "comparison requires both original native checkpoints0/32"
                            )
                    for index, method in enumerate(CANONICAL_NATIVE_METHODS):
                        completed()
                        raw = arms[index]["states"][0][1]
                        bound = arms[index + 4]["states"][0][1].native_state
                        if not shared._same_state(raw, bound):
                            raise ValueError("paired raw/canonical initial native states differ")
                        completed()
                        bind(
                            "initial_pair_parity",
                            [*report["initial_pair_parity"], dict(method=method, state_equal=True)],
                        )
                    from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
                    from src.adapters.minigrid_policy_evaluation import _return_budget_guard

                    completed()
                    readout_budget = ToyBudgetSession(
                        ToyExecutionBudget(
                            max_training_updates=0,
                            max_wall_seconds=max(0.0, started + body_seconds - monotonic()),
                            max_process_rss_bytes=RSS_LIMIT,
                        ),
                        monotonic,
                    )
                    readout_budget.attach_memory(sampler)
                    readout_budget_guard = _return_budget_guard(readout_budget)
                for method, arm in zip(methods, arms, strict=True):
                    states = (
                        arm["states"]
                        if fixed_a_acquisition or canonical_neural_comparison
                        else arm["states"][-1:]
                    )
                    for point, state in states:
                        completed()
                        if canonical_neural_comparison:
                            active_evaluation = len(completed_evaluations)
                        try:
                            if canonical_neural_comparison:
                                from src.adapters.minigrid_policy_evaluation import (
                                    evaluate_raw_neural_representation_checkpoint,
                                    evaluate_neural_representation_checkpoint,
                                )

                                assert readout_budget is not None
                                evaluator = (
                                    evaluate_raw_neural_representation_checkpoint
                                    if method in CANONICAL_NATIVE_METHODS
                                    else evaluate_neural_representation_checkpoint
                                )
                                result = evaluator(
                                    arm["source"],
                                    state,
                                    updates=point,
                                    budget=readout_budget,
                                    check_boundary=completed,
                                )
                            elif fixed_a_acquisition and point not in (0, 128):
                                result = _score_checkpoint(
                                    arm["source"],
                                    state,
                                    method=method,
                                    updates=point,
                                    sources=raw_sources
                                    if method == "recurrent_bc"
                                    else history_sources,
                                    budget=arm["budget"],
                                    check_boundary=completed,
                                )
                            else:
                                result = evaluate_checkpoint(
                                    arm["source"],
                                    state,
                                    method=method,
                                    updates=point,
                                    layouts=predefined_layouts()[: 1 if fixed_a_acquisition else 3],
                                    role="train",
                                    sources=raw_sources,
                                    budget=arm["budget"],
                                    check_boundary=completed,
                                    public_history=True,
                                )
                        except MiniGridEvaluationFailed as error:
                            original_evaluations.append(error.report)
                            completed_evaluations.append(
                                (error.report, pickle.dumps(error.report, protocol=5))
                            )
                            raise
                        original_evaluations.append(result)
                        completed_evaluations.append((result, pickle.dumps(result, protocol=5)))
                        if canonical_neural_comparison:
                            completed()
                            if (
                                type(result.get("method")) is not str
                                or result["method"] != method
                                or type(result.get("updates")) is not int
                                or result["updates"] != point
                            ):
                                raise ValueError(
                                    "delegated readout changed its original method/native clock"
                                )
                            if result["status"] == "complete":
                                _validate_representation_readout(result)
                            assert active_evaluation is not None
                            private_planned[active_evaluation].update(
                                status=result["status"], reason=result.get("reason")
                            )
                            bind("planned_evaluations", deepcopy(private_planned))
                            active_evaluation = None
                        if result["status"] != "complete":
                            raise ValueError(
                                "fixed-A raw readout failed; preserve paid partial evidence"
                            )
                completed()
                if canonical_neural_comparison and len(completed_evaluations) != 16:
                    raise ValueError("canonical comparison requires all sixteen paid readouts")
                if not canonical_neural_comparison:
                    report["status"] = "complete"
                else:
                    finished = True
            except Exception as error:
                failure(error)
            if canonical_neural_comparison:
                try:
                    report["observed_process_rss"] = asdict(sampler.snapshot())
                    completed()
                except Exception as error:
                    finished = False
                    failure(error)
            else:
                restore_publication()
                if fixed_a_acquisition:
                    report["acquisition"] = _acquisition_outcomes(
                        report["evaluations"], complete=report["status"] == "complete"
                    )
                report["observed_process_rss"] = asdict(sampler.snapshot())
    except Exception as error:
        if not canonical_neural_comparison:
            raise
        finished = False
        failure(error)
    if canonical_neural_comparison:
        try:
            # Sampler exit and its last readback are external boundaries too.
            if sampler is None:
                raise ValueError("whole-case sampler did not start")
            completed()
            final_rss = asdict(sampler.snapshot())
            completed()
            if finished:
                report["status"] = "complete"
            report["observed_process_rss"] = final_rss
        except Exception as error:
            failure(error)
        if active_evaluation is not None:
            captured = len(completed_evaluations) > active_evaluation
            private_planned[active_evaluation].update(status="failed", reason=report["reason"])
            if not captured:
                private_planned[active_evaluation]["delegated_report_unavailable"] = True
                private_planned[active_evaluation]["failure_work_may_be_unobserved"] = True
                private_planned[active_evaluation]["work"] = dict(
                    live_prediction_calls_attempted=None,
                    environment_resets_attempted=None,
                    environment_steps_attempted=None,
                    failure_work_may_be_unobserved=True,
                )
        restore_publication()
        report["planned_evaluations"] = private_planned
        report["readout_training_updates"] = (
            None if readout_budget is None else readout_budget.updates_completed
        )
        try:
            report["representation_comparison"] = _representation_outcomes(
                report["evaluations"], complete=report["status"] == "complete"
            )
        except Exception as error:
            failure(error)
            report["representation_comparison"] = _representation_outcomes([], complete=False)
        if report["status"] != "complete":
            report["evaluations"] = _failure_numbers(report["evaluations"])
        report["training_projection_accounting"] = dict(
            methods=[
                dict(
                    method=row["method"],
                    returned_native_rows=row.get("native_training_examples"),
                    projected_array_bytes_returned=(
                        None
                        if row.get("native_training_examples") is None
                        else row["native_training_examples"] * 2368
                        if row["method"].startswith("public_invariant_")
                        else 0
                    ),
                )
                for row in report["methods"]
            ],
            scope="additional296-float64 feature arrays for returned committed bound rows only; raw adds zero; validation/Python/copies/failed-attempt allocations unknown",
        )
        report["scientific_advantage_verified"] = False
    report["seconds"] = monotonic() - started
    return report


def load_comparison_sources(path: Path):
    from scripts.run_minigrid_invariant_pilot import load_inputs

    return load_inputs(path)[1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--fixed-a-acquisition", action="store_true")
    modes.add_argument("--canonical-neural-comparison", action="store_true")
    parser.add_argument("--source-artifact", type=Path)
    args = parser.parse_args(argv)
    if args.source_artifact is not None and (args.describe or not args.canonical_neural_comparison):
        parser.error("source artifacts are only read by a canonical comparison run")
    if args.canonical_neural_comparison:
        if args.describe:
            report = describe_protocol(canonical_neural_comparison=True)
        else:
            if args.source_artifact is None:
                parser.error("canonical comparison requires --source-artifact")
            loaded_at = monotonic()
            sources = load_comparison_sources(args.source_artifact)
            source_load_seconds = monotonic() - loaded_at
            report = run_smoke(canonical_neural_comparison=True, sources=sources)
            report.update(
                source_artifact=str(args.source_artifact), source_load_seconds=source_load_seconds
            )
    elif args.fixed_a_acquisition:
        report = (
            describe_protocol(fixed_a_acquisition=True)
            if args.describe
            else run_smoke(fixed_a_acquisition=True)
        )
    else:
        report = describe_protocol() if args.describe else run_smoke()
    encoded = json.dumps(report, indent=2, allow_nan=False)
    maximum_bytes = (32 if args.canonical_neural_comparison else 16) * 1024**2
    if len(encoded.encode("utf8")) > maximum_bytes:
        raise RuntimeError("fixed history-live report allowance exceeded")
    print(encoded)
    if not args.describe and report["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
