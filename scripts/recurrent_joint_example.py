"""Ordinary joint TRAIN recurrent BC using the existing stream and native games."""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
from math import isfinite
from pathlib import Path
import pickle
import random
from time import monotonic
from typing import Any, Callable

import numpy as np

from scripts.recurrent_readout_example import _episodes, _budget_report, _check_paid_evidence
from src.adapters.minigrid_public_cue_recurrent_pilot import _typed_sources
from src.adapters.minigrid_public_cue_recurrent_learner import (
    PublicCueRecurrentLearner,
    PublicCueRecurrentPolicy,
    copy_snapshot,
)
from src.adapters.torch_train_weighted_learner import TorchTrainWeightedLearner
from src.adapters.minigrid_learning_pilot import _batch, _episode, _signature, _check_signature
from src.adapters.minigrid_policy_evaluation import (
    evaluate_public_cue_recurrent_checkpoint,
    _return_budget_guard,
)
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.app.coupled_attribution import AttributionStream
from src.app.probe_return_phase import paid_report_guard
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.learner_ports import TrainingUpdateFailed
from src.infra.minigrid_memory import _require_minigrid
from src.shared.process_memory import ProcessRssSampler
from src.shared.torch_runtime import require_torch

RSS = 768 * 1024**2


def describe(
    *,
    fresh_task_probe=False,
    rehearsal_comparison=False,
    cold_fresh_task=False,
    class_balanced_joint=False,
    reward_training=False,
):
    if (
        type(fresh_task_probe) is not bool
        or type(rehearsal_comparison) is not bool
        or type(cold_fresh_task) is not bool
        or type(class_balanced_joint) is not bool
        or type(reward_training) is not bool
        or sum(
            (
                fresh_task_probe,
                rehearsal_comparison,
                cold_fresh_task,
                class_balanced_joint,
                reward_training,
            )
        )
        > 1
    ):
        raise ValueError("choose one exact Boolean isolated probe or rehearsal comparison")
    result = dict(
        definition="joint_train_recurrent_bc128_v1",
        source="original six TRAIN episodes released upfront",
        schedule="128 presentations of index i%6: Ablue/Ared/Bblue/Bred/C_trainblue/C_trainred",
        native="raw148/H8/K7/seed17/lr.03/momentum.9/meanCE, episode reset",
        presentations=dict(A=44, B=42, C_train=42),
        unique_train_rows=54,
        row_exposures=1152,
        label_delay=1,
        updates_per_cue=64,
        readout_points=[0, 128],
        native_game_points=[128],
        games="six known TRAIN tasks/all7firstargmax/seed17/max16",
        limits=dict(body_seconds=240, training_seconds=180, readout_seconds=200, rss_bytes=RSS),
        probe_available=False,
        final_available=False,
        limitation="joint upfront TRAIN, not causal retention/recovery, unseen transfer or CPC advantage",
    )
    if fresh_task_probe:
        from scripts.recurrent_joint_probe import describe as describe_probe

        result.update(
            definition="joint_train_recurrent_bc128_F16_A16_v1",
            isolated_probe=describe_probe(),
            probe_available=True,
        )
    if rehearsal_comparison:
        from scripts.recurrent_joint_rehearsal import describe as describe_comparison

        result.update(
            definition="joint128_F48_vs_F16_ABC32_A16_v1",
            isolated_probe=describe_comparison(),
            probe_available=True,
            limits=dict(body_seconds=480, training_seconds=180, readout_seconds=420, rss_bytes=RSS),
        )
    if cold_fresh_task:
        result.update(
            definition="cold0_F48_A16_v1",
            source="cold seed17 full state; original TRAIN episodes only for monitors and later A acquisition",
            schedule="ZERO benchmark TRAIN; disposable F48 then first A16 acquisition",
            presentations=dict(A=0, B=0, C_train=0),
            row_exposures=0,
            updates_per_cue=0,
            readout_points=[0],
            native_game_points=[],
            probe_available=True,
            isolated_probe=dict(
                source_updates=0,
                F_updates=48,
                A_updates=16,
                native_endpoints=[48, 64],
                native_head_endpoints=[24, 32],
                replay=False,
                phase_seconds=300,
                endpoint_seconds=120.0,
            ),
            limitation="cold F learnability control; A first acquisition, not recovery; not matched total work or CPC benefit",
        )
    if class_balanced_joint:
        result.update(
            definition="joint_train_recurrent_bc128_class_balanced_v1",
            native="raw148/H8/K7/seed17/lr.03/momentum.9/existing class-weightedCE, episode reset",
            class_balanced_training=True,
            objective="existing current complete TRAIN episode n/(present_classes*count), absent1",
            limitation="joint TRAIN conventional weighted control, not retention/generalization/CPC benefit; no probes",
        )
    if reward_training:
        result.update(
            definition="joint128_sampled_reward8_momentum8_v1",
            reward_training=True,
            readout_points=[],
            native_game_points=[],
            games="after all updates: frozen jointBC128, reward136, zero-loss136; six known TRAIN tasks each",
            adaptation="existing delayed reward pilot; A/B/C_train/A both cues; sampling29/seed17/max16/gamma1",
            limits=dict(body_seconds=480, training_seconds=180, readout_seconds=420, rss_bytes=RSS),
            limitation="TRAIN-only joint warm reward/conditional zero-loss comparison; no final/probe, selection or matched-compute/CPC claim",
        )
    return result


def joint_packets(episodes):
    sources = _typed_sources(episodes)
    packets = tuple(
        _batch(replace(sources[i % 6].episode, episode_id=f"joint-rnn-arrival-{i}"))
        for i in range(128)
    )
    return sources, packets


def _native_owner(learner, weighted):
    """Read the existing exact wrapper without adopting a new native owner."""
    if weighted:
        if type(learner) is not TorchTrainWeightedLearner:
            raise ValueError("joint weighted control requires its original exact schedule owner")
        learner.snapshot_state()  # Existing schedule/full-state/failure checks.
        learner = learner._learner
    if type(learner) is not PublicCueRecurrentLearner:
        raise ValueError("joint control requires its exact public-cue native owner")
    return learner


def _qualify(point, result):
    roster = [(c, cue) for c in ("A", "B", "C_train") for cue in ("blue", "red")]
    if (
        result.get("status") != "complete"
        or result.get("state_unchanged") is not True
        or result.get("method") != "recurrent_public_cue"
        or result.get("role") != "train"
        or type(result.get("updates")) is not int
        or result["updates"] != point
    ):
        raise ValueError("joint readout requires complete original-state proof")
    if [(r["context"], r["cue"]) for r in result["scores"]] != roster:
        raise ValueError("joint readout lost a declared TRAIN score")
    for row in result["scores"]:
        for key, maximum in (("ce", None), ("accuracy", 1), ("exact_sequence_accuracy", 1)):
            value = row.get(key)
            if (
                type(value) not in (int, float)
                or not isfinite(value)
                or value < 0
                or (maximum is not None and value > maximum)
            ):
                raise ValueError("joint TRAIN score lacks a finite bounded metric")
        if row["exact_sequence_accuracy"] not in (0, 1):
            raise ValueError("joint TRAIN exact sequence score must be binary")
    expected = roster if point == 128 else []
    if [(r["context"], r["cue"]) for r in result["episodes"]] != expected:
        raise ValueError("joint readout lost a declared native episode")
    if any(
        r.get("status") != "complete"
        or type(r.get("success")) is not bool
        or r.get("state_unchanged") is not True
        or r.get("environment", {}).get("closed") is not True
        or r.get("source_role") != "train"
        or type(r.get("return")) not in (int, float)
        or not isfinite(r["return"])
        for r in result["episodes"]
    ):
        raise ValueError("joint native outcomes lack completion/preservation/cleanup")


def run_example(
    episodes=None,
    *,
    fresh_task_probe=False,
    rehearsal_comparison=False,
    cold_fresh_task=False,
    class_balanced_joint=False,
    reward_training=False,
):
    started = monotonic()
    report: dict[str, Any] = dict(
        protocol=describe(
            fresh_task_probe=fresh_task_probe,
            rehearsal_comparison=rehearsal_comparison,
            cold_fresh_task=cold_fresh_task,
            class_balanced_joint=class_balanced_joint,
            reward_training=reward_training,
        ),
        status="incomplete",
        reason=None,
        training=None,
        readouts=[],
        failed_readout=None,
        failure_work=None,
        preservation=None,
        training_budget=None,
        readout_budget=None,
        readback_error=None,
    )
    protocol = report["protocol"]
    protocol_check, _ = paid_report_guard(protocol)
    records = report["readouts"]
    prior: list[tuple[dict[str, Any], Callable[[], None]]] = []
    training_evidence = None
    adaptation_evidence = None
    paid_receipts: list[dict[str, Any]] = []
    expected_updates = 0
    stream = train_budget = readout_budget = None
    whole_seconds = 480 if rehearsal_comparison or reward_training else 240
    with ProcessRssSampler() as memory:
        try:
            sources, packets = joint_packets(_episodes() if episodes is None else episodes)
            signatures = [(v, _signature(v)) for v in (sources, packets)]
            _require_minigrid()
            torch = require_torch()
            rng = (
                torch.get_rng_state().clone(),
                pickle.dumps(np.random.get_state()),
                random.getstate(),
            )
            source: Any = (
                TorchTrainWeightedLearner(
                    PublicCueRecurrentPolicy(), tuple(_episode(p) for p in packets)
                )
                if class_balanced_joint
                else PublicCueRecurrentLearner()
            )
            source_native = _native_owner(source, class_balanced_joint)
            source_owner, source_policy = source_native._state, source_native._policy
            initial = source.snapshot_state()
            train_limits = ToyExecutionBudget(
                max_training_updates=0 if cold_fresh_task else 128,
                max_wall_seconds=180,
                max_process_rss_bytes=RSS,
            )
            train_budget = ToyBudgetSession(train_limits, monotonic)
            train_values = asdict(train_limits)
            train_birth = train_budget.started_at
            train_budget.attach_memory(memory)
            stream = AttributionStream(
                source,
                packets,
                (),
                checkpoint_updates=(0, 128),
                label_delay=1,
                budget=train_budget,
                training_only=True,
            )
            actor_wrapper = stream._runtime.actor._learner
            actor_owner = _native_owner(actor_wrapper, class_balanced_joint)
            actor_state, actor_policy = actor_owner._state, actor_owner._policy
            actor_initial = deepcopy(stream._runtime.actor.snapshot_state())
            candidate_wrapper = stream._runtime._candidate
            candidate_native = _native_owner(candidate_wrapper, class_balanced_joint)
            read_limits = ToyExecutionBudget(
                max_training_updates=0,
                max_replay_examples=0,
                max_wall_seconds=420 if rehearsal_comparison or reward_training else 200,
                max_process_rss_bytes=RSS,
            )
            readout_budget = ToyBudgetSession(read_limits, monotonic)
            read_values = asdict(read_limits)
            read_birth = readout_budget.started_at
            readout_budget.attach_memory(memory)
            readout_check = _return_budget_guard(readout_budget)

            def check_evidence():
                _check_paid_evidence(
                    report["protocol"], protocol, protocol_check, "joint protocol changed"
                )
                if report["readouts"] is not records or len(records) != len(prior):
                    raise ValueError("preceding joint readout evidence changed or was replaced")
                for record, (owner, check) in zip(records, prior, strict=True):
                    _check_paid_evidence(record, owner, check, "preceding joint readout changed")
                if training_evidence is not None:
                    _check_paid_evidence(
                        report["training"],
                        *training_evidence,
                        "completed joint TRAIN evidence changed",
                    )
                if adaptation_evidence is not None:
                    _check_paid_evidence(
                        report["adaptation"],
                        *adaptation_evidence,
                        "completed joint probe evidence changed",
                    )

            def check_pure():
                check_evidence()
                if class_balanced_joint and (
                    _native_owner(source, True) is not source_native
                    or stream._runtime._candidate is not candidate_wrapper
                    or _native_owner(candidate_wrapper, True) is not candidate_native
                    or _native_owner(actor_wrapper, True) is not actor_owner
                ):
                    raise ValueError("joint weighted schedule or native owner replaced")
                # Inspect pinned originals directly after snapshot/clock callbacks;
                # a newly returned public snapshot cannot attest to later edits.
                if (
                    source_native._state is not source_owner
                    or source_native._policy is not source_policy
                    or stream._runtime.actor._learner is not actor_wrapper
                    or actor_owner._state is not actor_state
                    or actor_owner._policy is not actor_policy
                    or stream._runtime.actor._version != actor_initial.actor_version
                    or not _same_state(initial, source_owner)
                    or not _same_state(actor_initial.state, actor_state)
                    or not bool(torch.equal(rng[0], torch.get_rng_state()))
                    or rng[1] != pickle.dumps(np.random.get_state())
                    or rng[2] != random.getstate()
                ):
                    raise ValueError("original joint source/actor/RNG changed")
                if (
                    not all(_check_signature(v, sig) for v, sig in signatures)
                    or train_budget.budget is not train_limits
                    or readout_budget.budget is not read_limits
                    or train_budget.started_at != train_birth
                    or readout_budget.started_at != read_birth
                    or train_budget.clock is not monotonic
                    or readout_budget.clock is not monotonic
                    or asdict(train_budget.budget) != train_values
                    or asdict(readout_budget.budget) != read_values
                    or type(train_budget.updates_completed) is not int
                    or train_budget.updates_completed != expected_updates
                    or type(train_budget.replay_examples_completed) is not int
                    or train_budget.replay_examples_completed != 0
                    or type(readout_budget.updates_completed) is not int
                    or readout_budget.updates_completed != 0
                    or type(readout_budget.replay_examples_completed) is not int
                    or readout_budget.replay_examples_completed != 0
                ):
                    raise ValueError("original joint input, budget or permanent charge changed")
                actual = [asdict(r) for r in stream.applied_updates]
                if len(actual) != expected_updates or not _same_state(
                    actual[: len(paid_receipts)], paid_receipts
                ):
                    raise ValueError("preceding joint TRAIN receipts changed")
                # Why: a later snapshot/clock callback must not replace an
                # earlier validated leaf and let its equal values bless it.
                # These final identity reads invoke no wrapper snapshot ports.
                if class_balanced_joint and (
                    type(source) is not TorchTrainWeightedLearner
                    or type(actor_wrapper) is not TorchTrainWeightedLearner
                    or type(candidate_wrapper) is not TorchTrainWeightedLearner
                    or source._learner is not source_native
                    or source._native_owner is not source_native
                    or actor_wrapper._learner is not actor_owner
                    or actor_wrapper._native_owner is not actor_owner
                    or stream._runtime.actor._learner is not actor_wrapper
                    or stream._runtime._candidate is not candidate_wrapper
                    or candidate_wrapper._learner is not candidate_native
                    or candidate_wrapper._native_owner is not candidate_native
                ):
                    raise ValueError("joint weighted owner changed during final state callbacks")

            def guard():
                check_pure()
                if (
                    monotonic() - started >= whole_seconds
                    or memory.peak_bytes is None
                    or memory.peak_bytes > RSS
                ):
                    raise ValueError("joint whole-case time/RSS exhausted")
                stream._check_original()
                actual = [asdict(r) for r in stream.applied_updates]
                if len(actual) != expected_updates or not _same_state(
                    actual[: len(paid_receipts)], paid_receipts
                ):
                    raise ValueError("preceding joint TRAIN receipts changed")
                if (
                    source_native._state is not source_owner
                    or source_native._policy is not source_policy
                    or stream._runtime.actor._learner is not actor_wrapper
                    or actor_owner._state is not actor_state
                    or actor_owner._policy is not actor_policy
                    or not _same_state(initial, source.snapshot_state())
                    or not _same_state(actor_initial, stream._runtime.actor.snapshot_state())
                    or not bool(torch.equal(rng[0], torch.get_rng_state()))
                    or rng[1] != pickle.dumps(np.random.get_state())
                    or rng[2] != random.getstate()
                ):
                    raise ValueError("original joint source/actor/RNG changed")
                readout_check()
                readout_budget.before_final()
                check_pure()  # Snapshot/clock callbacks cannot forgive earlier evidence/charges.

            for point in (0,) if cold_fresh_task else (0, 128):
                guard()
                if point:
                    for update in range(1, 129):
                        guard()  # Validate the preceding readout before EACH additional update.
                        stream.run_until(update)
                        expected_updates = update
                        guard()
                        paid_receipts = [asdict(r) for r in stream.applied_updates]
                    endpoint = copy_snapshot(stream.candidate_snapshot())
                    if (endpoint.completed_updates, endpoint.training_examples) != (
                        128,
                        1152,
                    ) or any(
                        (h.completed_updates, h.training_examples) != (64, 576)
                        for h in endpoint.heads
                    ):
                        raise ValueError("joint endpoint lost its fixed per-cue update/row clocks")
                    report["training"] = stream.finish()
                    seal, _ = paid_report_guard(report["training"])
                    training_evidence = (report["training"], seal)
                    guard()
                else:
                    endpoint = initial
                guard()
                if reward_training:
                    # The fixed BC/reward/control readouts all follow learning;
                    # do not repeat this original prefix's scores or games.
                    continue
                # The evaluator owns a private scorer restored to endpoint;
                # its source must stay cold. The flag selects six TRAIN tasks.
                current = evaluate_public_cue_recurrent_checkpoint(
                    source_native,
                    endpoint,
                    updates=point,
                    sources=sources,
                    budget=readout_budget,
                    check_boundary=guard,
                    recurring_training=True,
                    **({"joint_comparison": True} if rehearsal_comparison else {}),
                    **({"class_balanced_training": True} if class_balanced_joint else {}),
                )
                record = dict(updates=point, games=current)
                records.append(record)  # Preserve returned paid work before outer qualification.
                seal, _ = paid_report_guard(record)
                prior.append((record, seal))
                _qualify(point, current)
                guard()
            if cold_fresh_task:
                report["training"] = dict(
                    status="complete", completed_updates=0, applied_updates=[]
                )
                seal, _ = paid_report_guard(report["training"])
                training_evidence = (report["training"], seal)
                guard()
            if fresh_task_probe or rehearsal_comparison or cold_fresh_task or reward_training:
                probe_runner: Callable[..., Any]
                if reward_training:
                    from src.adapters.minigrid_joint_warm import run_joint_reward

                    probe_runner = run_joint_reward
                elif rehearsal_comparison or cold_fresh_task:
                    from scripts.recurrent_joint_rehearsal import run_comparison

                    probe_runner = run_comparison
                else:
                    from scripts.recurrent_joint_probe import run_probe

                    probe_runner = run_probe

                candidate = stream._runtime._candidate
                if type(candidate) is not PublicCueRecurrentLearner:
                    raise ValueError("joint probe requires the original exact benchmark learner")
                candidate_owner, candidate_policy = candidate._state, candidate._policy
                benchmark = copy_snapshot(endpoint)

                def probe_guard():
                    guard()
                    if (
                        stream._runtime._candidate is not candidate
                        or candidate._state is not candidate_owner
                        or candidate._policy is not candidate_policy
                        or not _same_state(candidate_owner, benchmark)
                        or not _same_state(endpoint, benchmark)
                    ):
                        raise ValueError("original joint benchmark/checkpoint changed during probe")

                report["adaptation"] = probe_runner(
                    source,
                    endpoint,
                    sources,
                    check_boundary=probe_guard,
                    memory=memory,
                    **({"old_training_consent": True} if rehearsal_comparison else {}),
                    **({"cold_source": True} if cold_fresh_task else {}),
                )
                seal, _ = paid_report_guard(report["adaptation"])
                adaptation_evidence = (report["adaptation"], seal)
                probe_guard()
                if report["adaptation"].get("status") != "complete":
                    raise ValueError(report["adaptation"].get("reason"))
            report["preservation"] = dict(
                source=True, actor=True, torch_rng=True, numpy_rng=True, python_rng=True
            )
            if fresh_task_probe or rehearsal_comparison or cold_fresh_task or reward_training:
                report["preservation"]["benchmark"] = True
            report["status"] = "complete"
        except Exception as error:
            report["reason"] = f"{type(error).__name__}: {error}"
            if isinstance(error, TrainingUpdateFailed):
                report["failure_work"] = dict(error.work)
            if hasattr(error, "report"):
                report["failed_readout"] = deepcopy(error.report)
            if stream is not None and report["training"] is None:
                try:
                    report["training"] = dict(
                        status="incomplete",
                        applied_updates=[asdict(r) for r in stream.applied_updates],
                    )
                except Exception as readback:
                    report["readback_error"] = f"{type(readback).__name__}: {readback}"
        finally:
            report["training_budget"] = _budget_report(train_budget)
            report["readout_budget"] = _budget_report(readout_budget)
            report["seconds"], report["peak_bytes"] = monotonic() - started, memory.peak_bytes
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    selected = parser.add_mutually_exclusive_group()
    selected.add_argument("--fresh-task-probe", action="store_true")
    selected.add_argument("--compare-rehearsal", action="store_true")
    selected.add_argument("--cold-fresh-task-probe", action="store_true")
    selected.add_argument("--class-balanced-joint", action="store_true")
    selected.add_argument("--reward-training", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.describe:
        print(
            json.dumps(
                describe(
                    fresh_task_probe=args.fresh_task_probe,
                    rehearsal_comparison=args.compare_rehearsal,
                    cold_fresh_task=args.cold_fresh_task_probe,
                    class_balanced_joint=args.class_balanced_joint,
                    reward_training=args.reward_training,
                ),
                allow_nan=False,
                indent=2,
            )
        )
        return
    if args.output is None:
        parser.error("--output is required")
    with args.output.open("x", encoding="utf8") as output:
        result = run_example(
            **({"fresh_task_probe": True} if args.fresh_task_probe else {}),
            **({"rehearsal_comparison": True} if args.compare_rehearsal else {}),
            **({"cold_fresh_task": True} if args.cold_fresh_task_probe else {}),
            **({"class_balanced_joint": True} if args.class_balanced_joint else {}),
            **({"reward_training": True} if args.reward_training else {}),
        )
        encoded = json.dumps(result, allow_nan=False, separators=(",", ":"))
        if (
            len(encoded.encode("utf8"))
            > (8 if args.reward_training else 4 if args.compare_rehearsal else 2) * 1024**2
        ):
            raise ValueError("joint report exceeds its declared output limit")
        output.write(encoded + "\n")
    print(json.dumps(dict(status=result["status"], reason=result["reason"])))
    raise SystemExit(0 if result["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
