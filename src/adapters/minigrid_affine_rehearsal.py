"""Compare private joint ABC/F fitting with equal-row F-only fitting.

Inputs are an owned after-C state, original TRAIN episodes and the four selected
raw F sources from the original validated release. Outputs are probe diagnostics, full
private endpoints and explicit attempted/committed exposure costs. No collection,
benchmark replay bank, deployed learning, task selection or native games.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import pickle
from time import monotonic
from typing import Any

import numpy as np

from src.adapters import minigrid_affine_probe as original
from src.adapters.minigrid_context_linear_learner import (
    ContextLinearLearner,
    copy_context_linear_training_snapshot,
)
from src.app.coupled_attribution import AttributionBatch, _same_value
from src.app.fresh_task_probe import _validate_batches, run_fresh_task_probe
from src.app.probe_return_phase import budget_binding, paid_report_guard
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.demonstration_episode import DemonstrationEpisode
from src.core.minigrid_context_encoding import context_public_history
from src.shared.process_memory import ProcessRssSampler

ARMS = ("current_only", "joint_ABC_F")


def describe_protocol(*, fresh_reference=False):
    if type(fresh_reference) is not bool:
        raise ValueError("fresh_reference must be an exact bool")
    protocol = dict(
        definition="minigrid_private_affine_rehearsal_comparison_v1",
        arms=ARMS,
        updates_per_arm=2,
        fit_rows_per_update=36,
        total_new_fits=4,
        total_new_fit_rows=144,
        current_only_F_exposures=72,
        current_only_extra_F_exposures=54,
        joint_F_exposures=18,
        joint_old_TRAIN_reuse_exposures=54,
        source_native_fits=6,
        source_native_rows=108,
        upstream_paid_fits=12,
        upstream_paid_rows=198,
        checkpoint_updates=(0, 1, 2),
        wall_seconds=90,
        peak_rss_bytes=512 * 1024**2,
        qualification=(
            "private fixed F task; same update/row dimensions, not assumed equal compute; "
            "old TRAIN diagnostics include reused rows, not heldout retention; no games/final/CPC claim"
        ),
    )

    if fresh_reference:
        # Why: a from-birth reference didn't spend the historical study arms.
        # Provenance changes here grant no additional fitting authority.
        protocol.update(
            upstream_paid_fits=6,
            upstream_paid_rows=108,
            reference_origin="current_workflow_actual_TRAIN_endpoint",
        )
    return protocol


def prepare_selected_probe(episodes):
    """Decode no release: admit only the four already qualified selected sources."""
    if type(episodes) is not tuple or len(episodes) != 4:
        raise ValueError("selected F probe requires exactly two train and two test sources")
    for index, episode in enumerate(episodes):
        if (
            type(episode) is not DemonstrationEpisode
            or episode.role != ("probe_train" if index < 2 else "probe_test")
            or episode.context != ("F-blue" if index % 2 == 0 else "F-red")
            or episode.num_classes != 7
            or len(episode.transitions) != 9
            or any(len(t.observation) != 148 for t in episode.transitions)
        ):
            raise ValueError("selected F sources must preserve original roles/order/nine raw rows")
    raw = tuple(original.reserved._packet(e) for e in episodes)
    _validate_batches(raw[:2], raw[2:])
    history = original.reserved._history_packets(raw)
    for index, batch in enumerate(history):
        if int(context_public_history(tuple(map(float, batch.features[0])))[297]) != index % 2:
            raise ValueError("selected F context differs from initial public cue")
    detached = tuple(replace(batch, episode=None) for batch in history)
    return detached[:2], detached[2:]


def compose_training(training, monitors, spans, arm):
    """Build two owned matrices; row spans retain their original data roles."""
    if arm not in ARMS:
        raise ValueError("private rehearsal requires its fixed current-only or joint arm")
    if (
        type(training) is not tuple
        or type(monitors) is not tuple
        or type(spans) is not tuple
        or len(training) != 2
        or len(monitors) != 2
        or len(spans) != 2
        or any(
            type(b) is not AttributionBatch
            or b.role != "probe_train"
            or b.features.shape != (9, 296)
            or b.num_classes != 7
            for b in training
        )
        or any(
            type(b) is not AttributionBatch
            or b.role != "train"
            or b.features.shape != (27, 296)
            or b.num_classes != 7
            for b in monitors
        )
        or any([(s.start, s.stop) for s in group] != [(0, 9), (9, 18), (18, 27)] for group in spans)
    ):
        raise ValueError(
            "joint fitting requires two original F probe_train packets and ABC TRAIN monitors"
        )
    batches, wakes = [], []
    for cue, current in enumerate(training):
        if arm == "current_only":
            segments = [(current, 0, 9, "probe_train", current.context, n) for n in range(4)]
        else:
            segments = [
                (monitors[cue], span.start, span.stop, "train", span.episode_id, 0)
                for span in spans[cue]
            ] + [(current, 0, 9, "probe_train", current.context, 0)]
        inputs, targets, origins, offset = [], [], [], 0
        ids: list[str] = []
        for batch, start, stop, role, identity, repeat in segments:
            inputs.append(batch.features[start:stop])
            targets.append(batch.targets[start:stop])
            ids.extend(f"{arm}:{cue}:{repeat}:{key}" for key in batch.row_ids[start:stop])
            origins.append(
                dict(
                    start=offset,
                    stop=offset + stop - start,
                    role=role,
                    source=identity,
                    repeat=repeat,
                    original_row_ids=list(batch.row_ids[start:stop]),
                )
            )
            offset += stop - start
        batches.append(
            AttributionBatch(
                "probe_train",
                current.context,
                tuple(ids),
                np.concatenate(inputs),
                np.concatenate(targets),
                None,
                7,
            )
        )
        wakes.append(dict(cue=cue, fit_rows=offset, spans=origins))
    return tuple(batches), dict(
        arm=arm,
        wakes=wakes,
        input_array_bytes=sum(b.features.nbytes + b.targets.nbytes for b in batches),
        old_TRAIN_reuse_limit=54 if arm == "joint_ABC_F" else 0,
        extra_F_exposure_limit=54 if arm == "current_only" else 0,
        storage_scope="owned fit matrices; original inputs/helper copies/monitors/native/reports extra in RSS",
    )


class _ChargedProbeLearner(ContextLinearLearner):
    def __init__(self, admit_update):
        super().__init__()
        self._admit_update = admit_update

    def train_batch(self, features, targets):
        # Why: attempted data reuse is permanently charged before a physical
        # solve, including failed solves. Native commits are reported separately.
        self._admit_update(features, targets)
        return super().train_batch(features, targets)


class _PrivateSource(original._AffineProbeSource):
    def __init__(self, state, admit_update):
        super().__init__(state)
        self._admit_update = admit_update

    def fork(self):
        self.fork_calls += 1
        if self.fork_calls != 1:
            raise ValueError("private rehearsal permits one disposable fork per arm")
        self.child = _ChargedProbeLearner(self._admit_update)
        self.child.restore_state(self.snapshot_state())
        return self.child


def _ledger(budget, initial, memory, started_at):
    return dict(
        initial_limits=asdict(initial),
        limits=asdict(budget.budget),
        updates_completed=budget.updates_completed,
        replay_examples_completed=budget.replay_examples_completed,
        started_at=started_at,
        observed_budget_started_at=budget.started_at,
        ended_at=monotonic(),
        sampled_peak_rss_bytes=memory.peak_bytes,
    )


def _require_completed_probe(probe):
    if (
        probe.get("status") != "complete"
        or probe.get("completed_updates") != 2
        or probe.get("completed_training_examples") != 72
        or [p.get("updates") for p in probe.get("checkpoints", ())] != [0, 1, 2]
        or any(set(p.get("scores", {})) != {"F-blue", "F-red"} for p in probe["checkpoints"])
        or [p.get("updates") for p in probe.get("training_readouts", {}).get("checkpoints", ())]
        != [0, 1, 2]
        or any(
            p.get("status") != "complete" or len(p.get("scores", {})) != 2
            for p in probe["training_readouts"]["checkpoints"]
        )
    ):
        raise ValueError("complete fixed2fit72row F/old diagnostic checkpoints required")


def _run_arm(
    saved,
    training,
    evaluation,
    monitors,
    spans,
    plan,
    budget,
    memory,
    check_boundary,
    record_commit,
):
    started = monotonic()
    report: dict[str, Any] = dict(
        arm=plan["arm"],
        status="running",
        reason=None,
        plan=plan,
        attempts=[],
        probe=None,
        old_task_readouts=[],
    )
    attempts = report["attempts"]
    proofs: list[tuple[dict, Any, dict]] = []
    plan_check, original_plan = paid_report_guard(plan)
    expected, binding = replace(budget.budget), budget_binding(budget)
    signature = original.shared._signature((*training, *evaluation, *monitors))
    observed = 0
    native_expected = saved
    floor = budget.last_clock

    def guard():
        nonlocal observed, floor, native_expected
        check_boundary()
        if (
            budget_binding(budget) != binding
            or not _same_value(vars(budget.budget), vars(expected))
            or type(budget.updates_completed) is not int
            or type(budget.replay_examples_completed) is not int
            or budget.replay_examples_completed
            != observed * (27 if plan["arm"] == "joint_ABC_F" else 0)
            or not 0 <= budget.updates_completed <= 2
            or not floor <= budget.last_clock
        ):
            raise ValueError("original private arm budget/clock changed")
        floor = budget.last_clock
        if (
            report["plan"] is not plan
            or report["attempts"] is not attempts
            or len(attempts) != len(proofs)
        ):
            raise ValueError("private arm exposure evidence was replaced")
        plan_check()
        for attempt, (owner, proof, _) in zip(attempts, proofs):
            if attempt is not owner:
                raise ValueError("private arm paid exposure record was replaced")
            proof()
        if not original.shared._check_signature((*training, *evaluation, *monitors), signature):
            raise ValueError("private arm role-separated arrays changed")
        if source.snapshot_state() != saved or source.fork_calls not in (0, 1):
            raise ValueError("private arm source or fork ownership changed")
        current = saved if source.child is None else source.child.snapshot_state()
        updates = current.completed_updates - 6
        if (
            current.failed_work is not None
            or current.policy != saved.policy
            or current.training_examples != 108 + 36 * updates
            or updates != budget.updates_completed
            or not observed <= updates <= min(observed + 1, len(attempts), 2)
        ):
            raise ValueError("private native commits differ from charged update/exposure prefix")
        if updates > observed:
            if (
                current.heads[1 - observed] != native_expected.heads[1 - observed]
                or current.heads[observed] is None
                or current.heads[observed].fit_rows != 36
            ):
                raise ValueError("private update changed the wrong cue head or fit geometry")
            replay_rows = 27 if plan["arm"] == "joint_ABC_F" else 0
            budget.record_replay(replay_rows)
            record_commit(replay_rows)
            observed = updates
            native_expected = current
        if current != native_expected or (
            source.child is not None and source.child.snapshot_state() != native_expected
        ):
            raise ValueError("private native parameters changed without an observed update")
        budget.before_final()

    def admit_update(features, targets):
        guard()
        index = len(attempts)
        if index != observed or index >= 2:
            raise ValueError("private fit lacks preceding completed update or exceeds recipe")
        batch = training[index]
        if not np.array_equal(features, batch.features) or not np.array_equal(
            targets, batch.targets
        ):
            raise ValueError("private fit differs from its fixed36-row recipe")
        entry = dict(
            update=index + 1,
            fit_rows_attempted=36,
            old_TRAIN_reuse_charged=27 if plan["arm"] == "joint_ABC_F" else 0,
            extra_F_exposures_charged=27 if plan["arm"] == "current_only" else 0,
        )
        proof, detached = paid_report_guard(entry)
        attempts.append(entry)
        proofs.append((entry, proof, detached))
        if (
            sum(x["old_TRAIN_reuse_charged"] for x in attempts) > plan["old_TRAIN_reuse_limit"]
            or sum(x["extra_F_exposures_charged"] for x in attempts)
            > plan["extra_F_exposure_limit"]
        ):
            raise ValueError("private fit exceeds original nonrefundable exposure allowance")

    source = _PrivateSource(saved, admit_update)
    try:
        guard()
        report["probe"] = run_fresh_task_probe(
            source,
            saved,
            training,
            evaluation,
            budget=budget,
            checkpoint_updates=(0, 1, 2),
            decision_diagnostics=True,
            training_readouts=monitors,
            check_boundary=guard,
        )
        guard()
        _require_completed_probe(report["probe"])
        report["old_task_readouts"] = original._old_task_rows(report["probe"], spans)
        if report["probe"]["status"] != "complete" or observed != 2:
            raise ValueError("private arm probe is incomplete")
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", reason=f"{type(error).__name__}: {error}"[:512])
    finally:
        report["plan"] = original_plan
        report["attempts"] = [detached for _, _, detached in proofs]
        report["source_fork_calls"] = source.fork_calls
        try:
            report["source_native_state_unchanged"] = source.snapshot_state() == saved
            report["native_endpoint"] = (
                None if source.child is None else asdict(source.child.snapshot_state())
            )
        except Exception as error:
            report.update(
                status="failed", endpoint_capture_error=f"{type(error).__name__}: {error}"[:512]
            )
        endpoint = report.get("native_endpoint")
        native_commits = None if endpoint is None else endpoint["completed_updates"] - 6
        report["updates_accounted_by_study"] = observed
        report["committed_old_TRAIN_reuse_rows"] = (
            None
            if native_commits is None
            else native_commits * (27 if plan["arm"] == "joint_ABC_F" else 0)
        )
        report["committed_extra_F_rows"] = (
            None
            if native_commits is None
            else native_commits * (27 if plan["arm"] == "current_only" else 0)
        )
        report["budget"] = _ledger(budget, expected, memory, binding[2])
        report["elapsed_seconds"] = monotonic() - started
    return report


def compare_private_rehearsal(
    episodes, state, probe_episodes, *, check_boundary=lambda: None, fresh_reference=False
):
    protocol = describe_protocol(fresh_reference=fresh_reference)
    report: dict[str, Any] = dict(
        definition=protocol["definition"],
        protocol=protocol,
        status="running",
        reason=None,
        arms=[],
    )
    limits = ToyExecutionBudget(
        max_training_updates=4,
        max_replay_examples=54,
        max_wall_seconds=90,
        max_process_rss_bytes=512 * 1024**2,
    )
    arms = report["arms"]
    preceding: list[tuple[dict, Any, dict]] = []
    pending = None
    started = monotonic()
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        study = ToyBudgetSession(limits, monotonic)
        study.attach_memory(memory)
        binding, expected = budget_binding(study), replace(limits)
        sessions = []
        for name in ARMS:
            session = ToyBudgetSession(
                replace(
                    limits,
                    max_training_updates=2,
                    max_replay_examples=54 if name == "joint_ABC_F" else 0,
                ),
                monotonic,
            )
            session.started_at = (
                study.started_at
            )  # Both finite allowances share the original lifetime.
            session.attach_memory(memory)
            sessions.append(session)
        arm_bindings = tuple(budget_binding(s) for s in sessions)
        arm_limits = tuple(replace(s.budget) for s in sessions)
        arm_floors = [s.last_clock for s in sessions]
        active_index = -1
        paid = 0
        paid_replay = 0
        floor = study.last_clock
        try:
            saved = copy_context_linear_training_snapshot(state)
            if (
                (saved.completed_updates, saved.training_examples) != (6, 108)
                or saved.failed_work is not None
                or any(h is None or h.fit_rows != 27 for h in saved.heads)
            ):
                raise ValueError("private comparison requires actual after-C6/108/head27")
            witness, monitors, spans = original._prepare_monitors(episodes)
            training, evaluation = prepare_selected_probe(probe_episodes)
            probe_signature = pickle.dumps(probe_episodes, protocol=5)

            def guard():
                nonlocal floor
                if monotonic() - started >= 80:
                    raise TimeoutError("private rehearsal work deadline reached; capture reserved")
                check_boundary()
                if (
                    budget_binding(study) != binding
                    or not _same_value(vars(study.budget), vars(expected))
                    or type(study.updates_completed) is not int
                    or study.updates_completed != paid
                    or type(study.replay_examples_completed) is not int
                    or study.replay_examples_completed != paid_replay
                    or not floor <= study.last_clock
                ):
                    raise ValueError("original comparison lifetime budget/clock/charges changed")
                floor = study.last_clock
                for index, (session, authority, config) in enumerate(
                    zip(sessions, arm_bindings, arm_limits)
                ):
                    if (
                        budget_binding(session) != authority
                        or not _same_value(vars(session.budget), vars(config))
                        or type(session.updates_completed) is not int
                        or not 0 <= session.updates_completed <= (2 if index <= active_index else 0)
                        or type(session.replay_examples_completed) is not int
                        or not 0
                        <= session.replay_examples_completed
                        <= (54 if index == 1 and index <= active_index else 0)
                        or not arm_floors[index] <= session.last_clock
                    ):
                        raise ValueError("original preallocated private arm authority changed")
                    arm_floors[index] = session.last_clock
                study.before_final()
                witness.check(episodes)
                if pickle.dumps(probe_episodes, protocol=5) != probe_signature:
                    raise ValueError("original selected F episode evidence changed")
                if report["arms"] is not arms or len(arms) != len(preceding):
                    raise ValueError("preceding private arm prefix changed or was replaced")
                for arm, (owner, proof, _) in zip(arms, preceding):
                    if arm is not owner:
                        raise ValueError("preceding private arm was replaced")
                    proof()

            def record_commit(replay_rows):
                nonlocal paid, paid_replay
                guard()
                study.before_update()
                study.record_update()
                study.record_replay(replay_rows)
                paid += 1
                paid_replay += replay_rows

            for active_index, (name, session) in enumerate(zip(ARMS, sessions)):
                guard()
                batches, plan = compose_training(training, monitors, spans, name)
                _validate_batches(batches, evaluation, monitors)
                result = _run_arm(
                    saved,
                    batches,
                    evaluation,
                    monitors,
                    spans,
                    plan,
                    session,
                    memory,
                    guard,
                    record_commit,
                )
                pending = result
                guard()
                if result["status"] != "complete":
                    raise ValueError("preceding private arm incomplete; no later fitting admitted")
                proof, detached = paid_report_guard(result)
                arms.append(result)
                preceding.append((result, proof, detached))
                pending = None
            guard()
            report["status"] = "complete"
        except Exception as error:
            report.update(status="failed", reason=f"{type(error).__name__}: {error}"[:512])
            report["arms"] = [detached for _, _, detached in preceding] + (
                [] if pending is None else [deepcopy(pending)]
            )
        finally:
            report["budget"] = _ledger(study, expected, memory, binding[2])
            report["elapsed_seconds"] = monotonic() - started
    return report
