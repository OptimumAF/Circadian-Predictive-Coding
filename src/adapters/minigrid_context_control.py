"""One explicit conventional TRAIN recurrence recipe through existing ports.

Owns fixed input/stream/readout composition and paid failure reporting. No final
or probe access, tuning, publication, alternate scheduler or teacher collection.
"""

from copy import deepcopy
from dataclasses import asdict
from time import monotonic
from typing import Any, Callable

import numpy as np

from src.adapters.minigrid_context_perceptron import (
    MiniGridContextPerceptronLearner,
    MiniGridContextSnapshot,
    copy_snapshot,
    project_features,
)
from src.adapters.minigrid_train_catalogue import _validate_sources
from src.app.coupled_attribution import AttributionBatch, AttributionStream
from src.app.probe_return_phase import paid_report_guard
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.categorical_probability_scores import probability_scores
from src.core.demonstration_episode import episode_arrays
from src.core.minigrid_context_encoding import CONTEXT_HISTORY_ENCODING_ID
from src.shared.process_memory import ProcessRssSampler

POINTS = (0, 8, 16, 24, 32)
ORDER = tuple(phase * 2 + i % 2 for phase in (0, 1, 2, 0) for i in range(8))


def describe_protocol():
    return dict(
        definition="minigrid_public_cue_context_control_v1",
        encoding_id=CONTEXT_HISTORY_ENCODING_ID,
        schedule=list(ORDER),
        checkpoint_updates=list(POINTS),
        phase_order=["A8", "B8", "C8", "A8"],
        label_delay=1,
        updates=32,
        training_rows=288,
        zero_initialized_heads=2,
        coefficient_elements=4158,
        numeric_coefficient_bytes=33264,
        readouts="all six known TRAIN fits and fixed known-A blue/red games at every checkpoint, after all32updates; no final/probe/replay/publication",
        directions="per-task success/return and TRAIN accuracy/exact higher; mistakes, labels, time/memory lower; no scalar utility or early stopping",
        limits=dict(seconds=120, process_rss_bytes=768 * 1024**2),
        limitation="two public-cue heads add capacity; different exposure from old pooled BP/FIFO controls; no matched-compute, neural, CPC or generalization claim",
    )


def prepare_training(episodes):
    witness, rows = _validate_sources(episodes, 32)
    expected = tuple(
        (context, f"minigrid-source-{context}-{cue}")
        for context in ("A", "B", "C_train")
        for cue in ("blue", "red")
    )
    if (
        rows != 54
        or len(episodes) != 6
        or tuple((e.context, e.episode_id) for e in episodes) != expected
        or any(e.role != "train" or len(e.transitions) != 9 for e in episodes)
    ):
        raise ValueError("context control requires the six original nine-row TRAIN sources")
    arrays = tuple(episode_arrays(e) for e in episodes)
    for features, _ in arrays:
        projected = project_features(features)
        if not (projected[:, -2:] == projected[0, -2:]).all():
            raise ValueError("each context TRAIN episode must have one initial public cue")
    training = tuple(
        AttributionBatch(
            "train",
            episodes[index].context,
            tuple(f"context:{wake}:{episodes[index].episode_id}:{row}" for row in range(9)),
            arrays[index][0].copy(),
            arrays[index][1].copy(),
            num_classes=7,
        )
        for wake, index in enumerate(ORDER)
    )
    witness.check(episodes)
    return witness, training


def _record_failure(report, error):
    """Preserve paid prefixes and make a stopped checkpoint's status explicit."""
    from src.adapters.minigrid_policy_evaluation import MiniGridEvaluationFailed

    reason = f"{type(error).__name__}: {error}"[:512]
    if isinstance(error, MiniGridEvaluationFailed):
        game_work = error.report.get("work")
        if (
            type(game_work) is not dict
            or game_work.get("failure_work_may_be_unobserved") is not False
        ):
            report["work"]["unreturned_work_unknown"] = True
    if report["checkpoints"] and report["checkpoints"][-1]["status"] == "running":
        current = report["checkpoints"][-1]
        if isinstance(error, MiniGridEvaluationFailed):
            current["games"] = error.report
        current.update(status="failed", reason=reason)
    report.update(status="failed", reason=reason)


def run_control(episodes):
    started = monotonic()
    # Full input/role/cue admission precedes construction or game work.
    witness, training = prepare_training(episodes)
    protocol = describe_protocol()
    protocol_pin = deepcopy(protocol)
    report: dict[str, Any] = dict(
        status="running",
        protocol=protocol,
        checkpoints=[],
        work=dict(
            committed_updates=0,
            committed_rows=0,
            score_calls=0,
            score_rows=0,
            unreturned_work_unknown=True,
        ),
    )
    rng = np.random.get_state()
    assert type(rng) is tuple
    source = initial = stream = actor = final = budget = None
    states: list[tuple[int, MiniGridContextSnapshot]] = []
    paid_guards: list[Callable[[], None]] = []
    limits = ToyExecutionBudget(
        max_training_updates=32, max_wall_seconds=120, max_process_rss_bytes=768 * 1024**2
    )
    try:
        with ProcessRssSampler() as sampler:
            budget = ToyBudgetSession(limits, monotonic)
            budget.attach_memory(sampler)
            clock, born_start = budget.clock, budget.started_at

            def guard():
                budget.before_final()
                if monotonic() - started > 120:
                    raise TimeoutError("context control lifetime includes input admission/readouts")
                if (
                    budget.budget is not limits
                    or budget.clock is not clock
                    or budget.started_at != born_start
                    or budget.process_rss_sampler is not sampler
                    or budget.replay_examples_completed != 0
                    or protocol != protocol_pin
                ):
                    raise ValueError("original context control budget/recipe changed")
                witness.check(episodes)
                for check in paid_guards:
                    check()
                if source is not None and source.snapshot_state() != initial:
                    raise ValueError("context control source changed")
                if actor is not None and actor.snapshot_state().state != initial:
                    raise ValueError("context control actor changed")
                if final is not None and stream._runtime._candidate.snapshot_state() != final:
                    raise ValueError("completed context candidate changed during readout")
                witness.check(episodes)
                if (
                    budget.budget is not limits
                    or budget.clock is not clock
                    or budget.started_at != born_start
                    or budget.process_rss_sampler is not sampler
                ):
                    raise ValueError("context budget binding changed after snapshot reads")
                for check in paid_guards:
                    check()

            source = MiniGridContextPerceptronLearner()
            initial = source.snapshot_state()

            def observe(point, state, batch):
                guard()
                owned = copy_snapshot(state)
                if (
                    point != POINTS[len(states)]
                    or owned.completed_updates != point
                    or owned.training_examples != 9 * point
                ):
                    raise ValueError("context checkpoint differs from original receipt clock")
                states.append((point, owned))

            stream = AttributionStream(
                source,
                training,
                (),
                checkpoint_updates=POINTS,
                label_delay=1,
                budget=budget,
                observe_checkpoint=observe,
                training_only=True,
            )
            actor = stream._runtime.actor
            guard()
            stream.run_until(32)
            final = copy_snapshot(stream.candidate_snapshot())
            report["stream"] = stream.finish()
            guard()
            if (
                final.completed_updates != 32
                or final.training_examples != 288
                or budget.updates_completed != 32
                or len(stream._runtime.applied_updates) != 32
            ):
                raise ValueError("incomplete context work cannot admit frozen readouts")
            report["work"].update(
                committed_updates=32, committed_rows=288, unreturned_work_unknown=False
            )
            # Outcomes never select future work: all prescribed updates are done.
            from src.adapters.minigrid_policy_evaluation import evaluate_context_checkpoint

            for point, state in states:
                guard()
                record = dict(updates=point, status="running", state=asdict(state), scores=[])
                report["checkpoints"].append(record)
                reader = source.fork()
                reader.restore_state(state)
                for episode in episodes:
                    guard()
                    features, targets = episode_arrays(episode)
                    report["work"]["score_calls"] += 1
                    report["work"]["score_rows"] += len(features)
                    report["work"]["unreturned_work_unknown"] = True
                    scores = reader.predict(features)
                    report["work"]["unreturned_work_unknown"] = False
                    record["scores"].append(
                        dict(
                            episode_id=episode.episode_id,
                            **probability_scores(scores, targets, 7),
                            probabilities=scores.tolist(),
                        )
                    )
                    if reader.snapshot_state() != state:
                        raise ValueError("context TRAIN readout changed its private checkpoint")
                guard()
                record["games"] = evaluate_context_checkpoint(
                    source,
                    state,
                    updates=point,
                    budget=budget,
                    check_boundary=guard,
                )
                if record["games"].get("status") != "complete":
                    report["work"]["unreturned_work_unknown"] = True
                    raise ValueError("incomplete game readout cannot admit another checkpoint")
                guard()
                record["status"] = "complete"
                check, _ = paid_report_guard(record)
                paid_guards.append(check)
            guard()
            report.update(status="complete", source_unchanged=True, actor_unchanged=True)
    except Exception as error:
        _record_failure(report, error)
    finally:
        if stream is not None:
            report["applied_updates"] = [asdict(u) for u in stream._runtime.applied_updates]
            report["work"]["committed_updates"] = len(report["applied_updates"])
            try:
                report["native_state"] = asdict(stream._runtime._candidate.snapshot_state())
                report["work"]["committed_rows"] = report["native_state"]["native_state"][
                    "training_examples"
                ]
            except Exception as error:
                report.update(
                    status="failed", capture_error=f"{type(error).__name__}: {error}"[:512]
                )
                report["native_state"] = None
        if budget is not None:
            report["budget"] = dict(
                limits=asdict(limits),
                updates_completed=budget.updates_completed,
                started_at=budget.started_at,
                last_clock=budget.last_clock,
            )
        current = np.random.get_state()
        assert type(current) is tuple
        report["numpy_rng_unchanged"] = bool(
            rng[0] == current[0] and np.array_equal(rng[1], current[1]) and rng[2:] == current[2:]
        )
        if not report["numpy_rng_unchanged"]:
            report.update(status="failed", reason="caller NumPy RNG changed")
        report["elapsed_seconds"] = monotonic() - started
        report["sampled_process_rss"] = asdict(sampler.snapshot())
    return report


def describe_neural_a_protocol():
    """Fixed conventional baseline gate, separate from the old Perceptron recipe."""
    return dict(
        definition="minigrid_public_cue_bp_known_a_v1",
        encoding_id=CONTEXT_HISTORY_ENCODING_ID,
        method="bp",
        heads=2,
        hidden_dim_per_head=32,
        seed=17,
        learning_rate=0.03,
        momentum=0.9,
        updates=128,
        training_rows=1152,
        updates_per_head=64,
        checkpoint_updates=(0, 32, 64, 96, 128),
        label_delay=1,
        source="original packaged TRAIN A blue/red only; eighteen rows, repeated equally",
        scores="ten detached TRAIN calls/ninety rows AFTER all128updates; native games only final",
        games="known Ablue/Ared; originalbranch3/seed17/max16/all7firstargmax",
        whole_seconds=300,
        work_deadline_seconds=290,
        training_seconds=180,
        readout_seconds=150,
        max_process_rss_bytes=768 * 1024**2,
        qualification="known TRAIN conventional BP gate; extra two-head capacity, no comparison/generalization/CPC claim",
    )


def _neural_summary(state):
    """JSON metadata only; the observer receives complete owned native tensors."""
    return dict(
        policy=asdict(state.policy),
        completed_updates=state.completed_updates,
        training_examples=state.training_examples,
        failed_work=state.failed_work,
        heads=[
            dict(completed_updates=h.completed_updates, training_examples=h.training_examples)
            for h in state.heads
        ],
    )


def _neural_receipts(receipts):
    """Own the complete causal prefix before exporting paid data, not authority."""
    records = [asdict(receipt) for receipt in receipts]
    fields = ("observed_at", "arrived_at", "applied_at", "update_number")
    if len(records) != 128 or any(
        any(type(record[name]) is not int for name in fields)
        or tuple(record[name] for name in fields) != (i, i + 1, i + 1, i + 1)
        for i, record in enumerate(records)
    ):
        raise ValueError("native BP endpoint requires the exact128causal receipt prefix")
    return records


def run_neural_a_control(episodes, *, endpoint_observer=None):
    """Use the original delayed stream and native evaluator for one fixed BP gate."""
    from src.adapters.minigrid_context_torch_feature_learner import (
        ContextTorchFeatureLearner,
        ContextTorchFeaturePolicy,
        copy_context_torch_feature_snapshot,
    )
    from src.adapters.torch_feature_learner import TorchFeaturePolicy
    from src.adapters.minigrid_policy_evaluation import (
        evaluate_context_neural_bp_checkpoint,
        MiniGridEvaluationFailed,
    )
    from src.app.coupled_attribution import _same_value
    from src.app.probe_return_phase import budget_binding
    from src.core.learner_ports import TrainingUpdateFailed
    from src.infra.minigrid_memory import _require_minigrid

    started = monotonic()
    if endpoint_observer is not None and not callable(endpoint_observer):
        raise ValueError("native endpoint observer must be callable")
    if type(episodes) is not tuple or len(episodes) != 2:
        raise ValueError("context neural baseline requires only original A blue/red TRAIN pair")
    witness, rows = _validate_sources(episodes, 128)
    if (
        rows != 18
        or tuple(e.episode_id for e in episodes)
        != ("minigrid-source-A-blue", "minigrid-source-A-red")
        or any(len(e.transitions) != 9 for e in episodes)
    ):
        raise ValueError("context neural baseline requires ordered original nine-row A sources")
    if tuple(int(project_features(episode_arrays(e)[0])[0, 297]) for e in episodes) != (0, 1):
        raise ValueError("context neural A pair requires original public blue/red cue order")
    protocol = describe_neural_a_protocol()
    report: dict[str, Any] = dict(
        definition=protocol["definition"],
        protocol=protocol,
        status="running",
        reason=None,
        stream=None,
        checkpoints=[],
        games=None,
        committed_receipts=None,
        failure_work=None,
        native_endpoint=None,
        paid_prefix=None,
        finalization="not_started",
        source_unchanged=None,
        actor_unchanged=None,
    )
    source = initial = stream = actor = final = budget = runtime = None
    states: list[tuple[int, Any]] = []
    checks: list[Callable[[], None]] = []
    paid_owners: list[tuple[str, Any]] = []
    paid_detached = None
    receipt_prefix = None
    finalization = "not_started"
    monitors = report["checkpoints"]
    monitor_proofs: list[tuple[dict, dict]] = []
    protocol_check, original_protocol = paid_report_guard(protocol)
    limits = ToyExecutionBudget(
        max_training_updates=128,
        # ActorShadow rejects any replay ceiling, including zero: it has no
        # replay-budget enforcement port. This fixed stream never consolidates;
        # guard() separately requires the original replay charge to remain zero.
        max_wall_seconds=180,
        max_process_rss_bytes=768 * 1024**2,
    )
    original_limits = deepcopy(limits)
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        try:
            # No TRAIN work is spent when the optional native game runtime is missing.
            _require_minigrid()
            budget = ToyBudgetSession(limits, monotonic)
            budget.attach_memory(memory)
            binding = budget_binding(budget)
            policy = ContextTorchFeaturePolicy(TorchFeaturePolicy("bp", 296, 32, 17, num_classes=7))
            source = ContextTorchFeatureLearner(policy)
            initial = source.snapshot_state()
            arrays = tuple(episode_arrays(e) for e in episodes)
            training = tuple(
                AttributionBatch(
                    "train",
                    episodes[i % 2].context,
                    tuple(
                        f"{episodes[i % 2].episode_id}:neural-wake:{i}:step:{t.step}"
                        for t in episodes[i % 2].transitions
                    ),
                    arrays[i % 2][0].copy(),
                    arrays[i % 2][1].copy(),
                    num_classes=7,
                )
                for i in range(128)
            )

            def guard():
                if monotonic() - started >= 290:
                    raise TimeoutError(
                        "context neural baseline work deadline reached; capture reserved"
                    )
                if (
                    memory.sample() is None
                    or memory.peak_bytes is None
                    or memory.peak_bytes > 768 * 1024**2
                ):
                    raise RuntimeError("context neural baseline RSS unavailable or exceeds768MiB")
                if (
                    budget_binding(budget) != binding
                    or not _same_value(vars(budget.budget), vars(original_limits))
                    or type(budget.updates_completed) is not int
                    or not 0 <= budget.updates_completed <= 128
                    or budget.replay_examples_completed != 0
                ):
                    raise ValueError("original context neural TRAIN clock or budget changed")
                witness.check(episodes)
                if report["protocol"] is not protocol:
                    raise ValueError("original context neural protocol was replaced")
                protocol_check()
                if report["finalization"] != finalization:
                    raise ValueError("original context neural finalization evidence changed")
                if receipt_prefix is not None and not _same_value(
                    _neural_receipts(stream.applied_updates), receipt_prefix
                ):
                    raise ValueError("original context neural paid receipt prefix changed")
                if any(report[name] is not owner for name, owner in paid_owners):
                    raise ValueError("preceding context neural training evidence was replaced")
                if (
                    report["checkpoints"] is not monitors
                    or not len(monitor_proofs) <= len(monitors) <= 5
                ):
                    raise ValueError("preceding context neural monitors changed or were replaced")
                if any(
                    current is not owner for current, (owner, _) in zip(monitors, monitor_proofs)
                ):
                    raise ValueError("preceding context neural monitor record was replaced")
                if final is not None and budget.updates_completed != 128:
                    raise ValueError("completed context neural TRAIN charges changed")
                if not _same_value(source.snapshot_state(), initial):
                    raise ValueError("context neural baseline source changed")
                if actor is not None and not _same_value(actor.snapshot_state().state, initial):
                    raise ValueError("context neural baseline actor changed")
                if runtime is not None and stream._runtime is not runtime:
                    raise ValueError("original context neural candidate runtime was replaced")
                # finish seals the stream transport, while its original runtime
                # retains the public read-only candidate snapshot port. Keep
                # validating full state there; never reopen the sealed stream.
                if final is not None and not _same_value(runtime.candidate_snapshot().state, final):
                    raise ValueError("completed context neural candidate changed during readout")
                for check in checks:
                    check()
                if budget_binding(budget) != binding or not _same_value(
                    vars(budget.budget), vars(original_limits)
                ):
                    raise ValueError("context neural TRAIN authority changed during snapshot reads")
                protocol_check()
                witness.check(episodes)

            def observe(n, state, batch):
                guard()
                owned = copy_context_torch_feature_snapshot(state)
                if (
                    n != protocol["checkpoint_updates"][len(states)]
                    or (
                        owned.completed_updates,
                        owned.training_examples,
                    )
                    != (n, n * 9)
                    or budget.updates_completed != n
                ):
                    raise ValueError(
                        "context neural checkpoint differs from original receipt prefix"
                    )
                states.append((n, owned))

            stream = AttributionStream(
                source,
                training,
                (),
                checkpoint_updates=protocol["checkpoint_updates"],
                label_delay=1,
                budget=budget,
                training_only=True,
                observe_checkpoint=observe,
            )
            runtime = stream._runtime
            actor = stream._runtime.actor
            for stop in range(1, 129):
                guard()
                stream.run_until(stop)
                guard()
            final = copy_context_torch_feature_snapshot(stream.candidate_snapshot())
            report["committed_receipts"] = _neural_receipts(stream.applied_updates)
            if (
                budget.updates_completed != 128
                or (final.completed_updates, final.training_examples) != (128, 1152)
                or any((h.completed_updates, h.training_examples) != (64, 576) for h in final.heads)
                or final.policy != policy
                or final.failed_work is not None
            ):
                raise ValueError(
                    "native BP endpoint lacks complete original128update/1152row receipts"
                )
            guard()
            receipt_prefix = deepcopy(report["committed_receipts"])
            report["paid_prefix"] = dict(status="unqualified", **_neural_summary(final))
            paid_owners = [(name, report[name]) for name in ("committed_receipts", "paid_prefix")]
            check, paid_detached = paid_report_guard(dict(paid_owners))
            checks.append(check)
            if endpoint_observer is not None:
                # Why: a fallible finish must not discard already-paid owned data.
                # This callback conveys an unqualified prefix, never game authority.
                endpoint_observer(copy_context_torch_feature_snapshot(final))
            guard()
            finalization = report["finalization"] = "running"
            report["stream"] = stream.finish()
            guard()
            if (
                report["stream"].get("completed_updates") != 128
                or report["stream"].get("training_examples") != 1152
                or report["stream"].get("evaluation_scope") != "training_only"
                or report["stream"].get("score_forward_calls") != 0
            ):
                raise ValueError("native BP finalization report lacks its complete TRAIN prefix")
            finalization = report["finalization"] = "complete"
            report["native_endpoint"] = _neural_summary(final)
            paid_owners.extend((name, report[name]) for name in ("stream", "native_endpoint"))
            check, paid_detached = paid_report_guard(dict(paid_owners))
            checks.append(check)
            guard()
            # Monitoring cannot select any subsequent training or its dose.
            for n, state in states:
                guard()
                reader = source.fork()
                reader.restore_state(state)
                record = dict(updates=n, status="running", state=_neural_summary(state), scores=[])
                report["checkpoints"].append(record)
                for episode, (x, y) in zip(episodes, arrays):
                    guard()
                    p = reader.predict(x)
                    record["scores"].append(
                        dict(
                            episode_id=episode.episode_id,
                            **probability_scores(p, y, 7),
                            actions=p.argmax(axis=1).tolist(),
                            expected=y[:, 0].astype(int).tolist(),
                        )
                    )
                    if not _same_value(reader.snapshot_state(), state):
                        raise ValueError("context neural TRAIN scoring changed checkpoint")
                guard()
                record["status"] = "complete"
                check, detached = paid_report_guard(record)
                checks.append(check)
                monitor_proofs.append((record, detached))
                guard()
            budget.before_final()
            report["training_budget"] = dict(
                initial_limits=asdict(original_limits),
                limits=asdict(budget.budget),
                updates_completed=budget.updates_completed,
                replay_examples_completed=budget.replay_examples_completed,
                started_at=budget.started_at,
                ended_at=monotonic(),
                scope="TRAIN updates and detached TRAIN monitors; no native games",
            )
            readout = ToyBudgetSession(
                ToyExecutionBudget(
                    max_training_updates=0,
                    max_replay_examples=0,
                    max_wall_seconds=150,
                    max_process_rss_bytes=768 * 1024**2,
                ),
                monotonic,
            )
            readout.attach_memory(memory)
            readout_binding = budget_binding(readout)

            def readout_guard():
                guard()
                if (
                    budget_binding(readout) != readout_binding
                    or readout.updates_completed != 0
                    or readout.replay_examples_completed != 0
                ):
                    raise ValueError("context neural readout clock/charges changed")

            try:
                report["games"] = evaluate_context_neural_bp_checkpoint(
                    source, final, budget=readout, check_boundary=readout_guard
                )
                readout_guard()
                games = report["games"]
                if (
                    games.get("status") != "complete"
                    or games.get("state_unchanged") is not True
                    or len(games.get("episodes", ())) != 2
                    or any(
                        e.get("status") != "complete"
                        or e.get("environment", {}).get("closed") is not True
                        for e in games["episodes"]
                    )
                ):
                    raise ValueError("context neural known-A game report is incomplete")
                report.update(status="complete", source_unchanged=True, actor_unchanged=True)
            except MiniGridEvaluationFailed as error:
                report["games"] = error.report
                raise
            finally:
                report["readout_budget"] = dict(
                    limits=asdict(readout.budget),
                    updates_completed=readout.updates_completed,
                    replay_examples_completed=readout.replay_examples_completed,
                    started_at=readout_binding[2],
                    observed_budget_started_at=readout.started_at,
                    ended_at=monotonic(),
                )
        except Exception as error:
            report.update(status="failed", reason=f"{type(error).__name__}: {error}"[:512])
            if finalization == "running":
                finalization = "failed"
            if monitors and monitors[-1].get("status") == "running":
                monitors[-1].update(status="failed", reason=report["reason"])
            if isinstance(error, TrainingUpdateFailed):
                report["failure_work"] = dict(definition=error.definition, work=error.work)
        finally:
            report["protocol"] = original_protocol
            report["finalization"] = finalization
            if paid_detached is not None:
                report.update(paid_detached)
            prefix = [detached for _, detached in monitor_proofs]
            report["checkpoints"] = prefix + monitors[len(prefix) : 5]
            report["checkpoint_capture_truncated"] = len(monitors) > 5
            if stream is not None and report["committed_receipts"] is None:
                try:
                    report["committed_receipts"] = [asdict(r) for r in stream.applied_updates]
                except Exception as error:
                    report["receipt_capture_error"] = f"{type(error).__name__}: {error}"[:512]
            if stream is not None:
                try:
                    state = (
                        runtime.candidate_snapshot().state
                        if runtime is not None
                        else stream.candidate_snapshot()
                    )
                    report["last_native_state"] = _neural_summary(state)
                except Exception as error:
                    report["state_capture_error"] = f"{type(error).__name__}: {error}"[:512]
            report["unscored_checkpoints"] = [
                dict(updates=n, state=_neural_summary(s))
                for n, s in states
                if n
                not in {
                    r["updates"] for r in report["checkpoints"] if r.get("status") == "complete"
                }
            ]
            if budget is not None and "training_budget" not in report:
                report["training_budget"] = dict(
                    initial_limits=asdict(original_limits),
                    limits=asdict(budget.budget),
                    updates_completed=budget.updates_completed,
                    replay_examples_completed=budget.replay_examples_completed,
                    started_at=budget.started_at,
                    ended_at=monotonic(),
                )
            report["workflow_budget"] = dict(
                max_wall_seconds=300,
                work_deadline_seconds=290,
                max_process_rss_bytes=768 * 1024**2,
                started_at=started,
                ended_at=monotonic(),
                sampled_peak_rss_bytes=memory.peak_bytes,
            )
    return report
