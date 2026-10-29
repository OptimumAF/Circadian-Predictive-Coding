"""Publish an arrived affine TRAIN candidate, then play through its actual actor.

Composes existing stream, guard/controller and MiniGrid serving ports. Inputs
are bundled TRAIN examples; output is a finite report, never native handles or
weights. No new learner, scheduler, evaluator, teacher or final/probe authority.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
import pickle
from time import monotonic
from typing import Any

import numpy as np

from src.adapters.minigrid_context_linear_learner import (
    ContextLinearLearner,
    copy_context_linear_training_snapshot,
)
from src.adapters.minigrid_policy_evaluation import MiniGridEvaluationFailed
from src.adapters.minigrid_serving import run_promotable_minigrid_episode
from src.app.coupled_attribution import AttributionBatch, AttributionStream, _same_value
from src.app.probe_return_phase import budget_binding, paid_report_guard
from src.app.promotion_guard_evaluation import PromotionGuardEvaluator
from src.app.serving_promotion import PromotableActor, ServingPromotionController, PromotionRejected
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.categorical_probability_scores import probability_scores, validate_class_probabilities
from src.core.demonstration_episode import episode_arrays
from src.core.promotion_guard import GuardBatch, PromotionPolicy
from src.core.serving_ports import ServingConfiguration
from src.infra.minigrid_affine_example import example_affine_training
from src.infra.minigrid_memory import _require_minigrid
from src.shared.process_memory import ProcessRssSampler

TRAIN_SHA = "5a368cfa8d4d7326d5d3ab4d6663a0e8e35e1e660968a68212e627ba9203d508"


def describe_published_workflow():
    return dict(
        definition="portable_published_affine_train_and_play_v1",
        source="bundled six nine-row TRAIN episodes; A blue/red fit, B blue/red inner guards, C unused",
        source_sha256=TRAIN_SHA,
        fit_calls=2,
        fit_rows=18,
        label_delay=1,
        checkpoints=[0, 1, 2],
        games="actual published actor; known A3 TRAIN blue/red; seed17/max16; raw K7 firstargmax",
        guard=dict(
            metric="raw_first_argmax_accuracy_K7",
            min_new_utility=0.0,
            min_new_gain=0.0,
            max_old_drop=0.0,
            max_prediction_seconds=2.0,
            coefficient_proxy_bytes=2 * 1024**2,
        ),
        train_wall_seconds=90,
        readout_wall_seconds=150,
        work_deadline_seconds=140,
        max_process_rss_bytes=512 * 1024**2,
        readout_updates=0,
        replay=0,
        teacher=0,
        probe=0,
        final=0,
        qualification="known TRAIN publication behavior; B is not a learned old task; no CPC/generalization/efficiency/release claim",
    )


def _digest(value):
    # Existing promotion ports require a complete state/feature SHA256 callback.
    return sha256(pickle.dumps(value)).hexdigest()


def _training_batches(episodes):
    expected = tuple(
        f"minigrid-source-{context}-{cue}"
        for context in ("A", "B", "C_train")
        for cue in ("blue", "red")
    )
    if (
        type(episodes) is not tuple
        or tuple(e.episode_id for e in episodes) != expected
        or any(e.role != "train" or e.num_classes != 7 or len(e.transitions) != 9 for e in episodes)
    ):
        raise ValueError("published workflow requires the exact bundled six TRAIN episodes")
    batches = []
    for episode in episodes[:2]:
        features, targets = episode_arrays(episode)
        batches.append(
            AttributionBatch(
                "train",
                episode.context,
                tuple(f"{episode.episode_id}:step:{t.step}" for t in episode.transitions),
                features,
                targets,
                episode,
                7,
            )
        )
    return tuple(batches)


def _controller(actor, work):
    def builder(state):
        learner = ContextLinearLearner()
        learner.restore_state(state)
        return learner

    def valid_prediction(prediction):
        # These are observed returned guard predictions, not inferred attempts.
        work["guard_predictions_returned"] += 1
        if type(prediction) is np.ndarray and prediction.ndim == 2:
            work["guard_rows_returned"] += len(prediction)
        else:
            work["guard_rows_may_be_unknown"] = True
        validate_class_probabilities(prediction, 7)
        return True

    policy = PromotionPolicy(
        "raw_first_argmax_accuracy_K7",
        "coefficient_float64_proxy_v1",
        0.0,
        0.0,
        0.0,
        2.0,
        2 * 1024**2,
        tuple(str(i) for i in range(7)),
    )
    evaluator = PromotionGuardEvaluator(
        build_learner=builder,
        utility=lambda p, y: probability_scores(p, y, 7)["accuracy"],
        actions=lambda p: tuple(str(int(a)) for a in np.argmax(p, axis=1)),
        state_valid=lambda state: copy_context_linear_training_snapshot(state).failed_work is None,
        prediction_valid=valid_prediction,
        resource_bytes=lambda state: sum(0 if h is None else 297 * 7 * 8 for h in state.heads),
        state_digest=_digest,
        clock=monotonic,
    )
    return ServingPromotionController(
        actor, policy=policy, evaluator=evaluator, build_learner=builder, state_digest=_digest
    )


def _inner_guards(episodes):
    guards = []
    for episode in episodes[2:4]:
        features, targets = episode_arrays(episode)
        guards.append(
            GuardBatch(
                episode.episode_id,
                tuple(
                    (episode.episode_id, f"{episode.episode_id}:step:{t.step}")
                    for t in episode.transitions
                ),
                2,
                features,
                targets,
                "inner_guard",
            )
        )
    return tuple(guards)


def _budget_report(owner, binding, limits):
    return dict(
        initial_limits=asdict(limits),
        limits=asdict(owner.budget),
        original_controls=budget_binding(owner) == binding,
        started_at=binding[2],
        observed_started_at=owner.started_at,
        last_clock=owner.last_clock,
        updates=owner.updates_completed,
        replay=owner.replay_examples_completed,
    )


def run_published_workflow():
    protocol = describe_published_workflow()
    report: dict[str, Any] = dict(
        definition=protocol["definition"],
        protocol=protocol,
        status="running",
        phase="preflight",
        reason=None,
        training=None,
        guards=None,
        publication=None,
        games=[],
        failed_game=None,
        budgets=None,
        candidate=None,
        committed_receipts=None,
        preservation=None,
        summary=None,
        work=dict(
            guard_predictions_returned=0,
            guard_rows_returned=0,
            guard_rows_may_be_unknown=False,
            guard_work_may_be_unobserved=False,
            train_work_may_be_unobserved=False,
        ),
    )
    try:
        # Why: a missing optional runtime must cost no source/model/budget work.
        _require_minigrid()
    except Exception as error:
        report.update(status="failed", reason=f"{type(error).__name__}: {error}"[:512])
        return report
    try:
        return _run_ready_workflow(report)
    except Exception as error:
        report.update(
            status="failed",
            preservation=False,
            summary=None,
            capture_error=f"{type(error).__name__}: {error}"[:512],
        )
        if report["reason"] is None:
            report["reason"] = report["capture_error"]
        return report


def _run_ready_workflow(report):
    from scripts.run_minigrid_linear_fit_control import read_training_sources

    protocol = report["protocol"]
    started = monotonic()
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        readout = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=0,
                max_replay_examples=0,
                max_wall_seconds=150,
                max_process_rss_bytes=512 * 1024**2,
            ),
            monotonic,
        )
        training_budget = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=2, max_wall_seconds=90, max_process_rss_bytes=512 * 1024**2
            ),
            monotonic,
        )
        readout.attach_memory(memory)
        training_budget.attach_memory(memory)
        read_pin, train_pin = budget_binding(readout), budget_binding(training_budget)
        read_limits, train_limits = replace(readout.budget), replace(training_budget.budget)
        proofs: list[tuple[Any, Any, Any]] = []
        paid_values = []
        stream: Any = None
        candidate_pin = None
        source = actor = None
        source_pin = actor_pin = None
        episodes = episode_pin = None
        try:
            report["phase"] = "source"
            with example_affine_training() as path:
                episodes = read_training_sources(path, expected_sha256=TRAIN_SHA)
            batches = _training_batches(episodes)
            episode_pin = deepcopy(episodes)
            source = ContextLinearLearner()
            source_pin = source.snapshot_state()
            actor = PromotableActor(
                source,
                version="a0",
                configuration=ServingConfiguration(3, 2),
                feature_digest=_digest,
                metadata={"workflow": "published_affine", "phase": "cold"},
            )
            actor_pin = actor.serving_snapshot()
            protocol_check, _ = paid_report_guard(protocol)

            def boundary():
                assert source is not None and actor is not None
                if monotonic() - started >= 140:
                    raise TimeoutError(
                        "published workflow deadline; original cleanup time reserved"
                    )
                if (
                    budget_binding(readout) != read_pin
                    or budget_binding(training_budget) != train_pin
                    or readout.budget != read_limits
                    or training_budget.budget != train_limits
                    or readout.updates_completed != 0
                    or readout.replay_examples_completed != 0
                    or not _same_value(episodes, episode_pin)
                    or not _same_value(source.snapshot_state(), source_pin)
                    or not _same_value(actor.serving_snapshot(), actor_pin)
                    or report["protocol"] is not protocol
                ):
                    raise ValueError(
                        "published workflow source, actor or original controls changed"
                    )
                readout.before_final()
                protocol_check()
                for key, original, check in proofs:
                    observed = report["games"][key] if type(key) is int else report[key]
                    if observed is not original:
                        raise ValueError("preceding paid publication evidence was replaced")
                    check()
                if candidate_pin is not None and (
                    training_budget.updates_completed != 2
                    or not _same_value(stream._runtime.candidate_snapshot().state, candidate_pin)
                ):
                    raise ValueError("paid TRAIN candidate or charge changed")

            def retain(key, value):
                check, detached = paid_report_guard(value)
                proofs.append((key, value, check))
                paid_values.append((key, detached))

            boundary()
            report["phase"] = "train"
            stream = AttributionStream(
                source,
                batches,
                (),
                training_only=True,
                checkpoint_updates=(0, 1, 2),
                label_delay=1,
                budget=training_budget,
                actor=actor,
            )
            stream.run_until(2)
            report["training"] = stream.finish()
            candidate_pin = stream._runtime.candidate_snapshot().state
            if (
                candidate_pin.completed_updates,
                candidate_pin.training_examples,
                candidate_pin.failed_work,
            ) != (2, 18, None):
                raise ValueError("publication requires complete native two-fit eighteen-row TRAIN")
            retain("training", report["training"])
            boundary()
            report["phase"] = "guard"
            controller = _controller(actor, report["work"])
            train_ids = frozenset(
                (episode.episode_id, f"{episode.episode_id}:step:{t.step}")
                for episode in episodes[:2]
                for t in episode.transitions
            )
            try:
                ticket = controller.prepare(
                    stream._runtime,
                    *_inner_guards(episodes),
                    now=2,
                    training_ids=train_ids,
                    metadata={"workflow": "published_affine", "phase": "published"},
                )
            except PromotionRejected as error:
                report["guards"] = asdict(error.report)
                raise
            except Exception:
                report["work"]["guard_work_may_be_unobserved"] = True
                raise
            report["guards"] = asdict(ticket.report)
            retain("guards", report["guards"])
            boundary()
            report["phase"] = "publish"
            report["publication"] = asdict(controller.commit(stream._runtime, ticket))
            actor_pin = actor.serving_snapshot()
            if (actor_pin.actor_version, actor_pin.generation) != ("c0", 1) or not _same_value(
                actor_pin.model_state, candidate_pin
            ):
                raise ValueError("published actor differs from accepted TRAIN candidate")
            retain("publication", report["publication"])
            boundary()
            for cue in ("blue", "red"):
                report["phase"] = "game-" + cue
                boundary()
                try:
                    game = run_promotable_minigrid_episode(
                        actor,
                        cue=cue,
                        episode_id="published-affine-" + cue,
                        budget=readout,
                        check_boundary=boundary,
                    )
                except MiniGridEvaluationFailed as error:
                    report["failed_game"] = dict(cue=cue, report=error.report)
                    raise
                report["games"].append(game)
                retain(len(report["games"]) - 1, game)
                boundary()
            report.update(
                status="complete",
                preservation=True,
                summary=dict(
                    episodes=2, successes=sum(g["episodes"][0]["success"] for g in report["games"])
                ),
            )
        except Exception as error:
            if report["phase"] == "train":
                report["work"]["train_work_may_be_unobserved"] = True
            report.update(status="failed", reason=f"{type(error).__name__}: {error}"[:512])
        finally:
            if report["status"] == "complete":
                try:
                    boundary()
                except Exception as error:
                    report.update(
                        status="failed",
                        preservation=False,
                        summary=None,
                        capture_error=f"{type(error).__name__}: {error}"[:512],
                    )
            # Rebuild the known paid prefix without trusting a corrupted list.
            report["games"] = [detached for key, detached in paid_values if type(key) is int]
            for key, detached in paid_values:
                if type(key) is int:
                    continue
                else:
                    report[key] = detached
            if stream is not None:
                try:
                    report["committed_receipts"] = [
                        asdict(r) for r in stream._runtime.applied_updates
                    ]
                except Exception as error:
                    report["receipt_capture_error"] = f"{type(error).__name__}: {error}"[:512]
                    if report["status"] == "complete":
                        report.update(
                            status="failed",
                            preservation=False,
                            summary=None,
                            capture_error=report["receipt_capture_error"],
                        )
                try:
                    candidate = stream._runtime.candidate_snapshot().state
                    report["candidate"] = dict(
                        native_updates=candidate.completed_updates,
                        native_rows=candidate.training_examples,
                        failed_work=candidate.failed_work,
                        receipts=[asdict(r) for r in stream._runtime.applied_updates],
                    )
                except Exception as error:
                    report["candidate"] = dict(error=f"{type(error).__name__}: {error}"[:512])
                    if report["status"] == "complete":
                        report.update(
                            status="failed",
                            preservation=False,
                            summary=None,
                            capture_error=report["candidate"]["error"],
                        )
            report["budgets"] = dict(
                readout=_budget_report(readout, read_pin, read_limits),
                train=_budget_report(training_budget, train_pin, train_limits),
                elapsed_seconds=monotonic() - started,
                sampled_peak_rss_bytes=memory.peak_bytes,
            )
    return report


def handle_cli(args, parser):
    deploy = getattr(args, "example_affine_deploy_a4", False)
    live = getattr(args, "example_affine_live_a4", False) or deploy
    allowed = {
        "example_affine_published",
        "example_affine_live_a4",
        "example_affine_deploy_a4",
        "describe",
        "output",
    }
    if any(v is not None and v is not False for k, v in vars(args).items() if k not in allowed):
        parser.error("published affine workflow accepts only --describe or --output")
    if args.describe:
        if args.output is not None:
            parser.error("published describe accepts no output")
        if live:
            from scripts.minigrid_live_adaptation import describe_live_adaptation

            description = (
                describe_live_adaptation(deploy=True) if deploy else describe_live_adaptation()
            )
        else:
            description = describe_published_workflow()
        print(json.dumps(description, indent=2, allow_nan=False))
        return
    if args.output is not None and args.output.exists():
        raise FileExistsError("published affine output already exists")
    if live:
        from scripts.minigrid_live_adaptation import run_live_adaptation

        result = run_live_adaptation(deploy=True) if deploy else run_live_adaptation()
    else:
        result = run_published_workflow()
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output_limit = (8 if live else 4) * 1024**2
    if len(payload.encode("utf-8")) > output_limit:
        raise ValueError(
            f"published affine report exceeds{8 if live else 4}MiB; paid work must not retry"
        )
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as output:
            output.write(payload)
    print(payload)
    if result["status"] != "complete":
        raise SystemExit(2)
