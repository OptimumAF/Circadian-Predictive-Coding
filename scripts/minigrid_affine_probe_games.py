"""Execute two saved F probe endpoints through the existing native evaluator.

Inputs are the qualified111/113 reports; outputs are bounded game evidence and
costs. No labels, training, teacher, selection or promotion belong here.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
from time import monotonic
from typing import Any

from src.adapters.minigrid_context_linear_learner import ContextLinearLearner
from src.adapters.minigrid_policy_evaluation import (
    MiniGridEvaluationFailed,
    evaluate_context_linear_probe_checkpoint,
    evaluate_context_linear_rehearsal_checkpoint,
)
from src.app.coupled_attribution import _same_value
from src.app.probe_return_phase import budget_binding, paid_report_guard
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.shared.process_memory import ProcessRssSampler

QUALIFIED_PROBE_SHA = "7d01ab5d6ff3d7b4bc84248463e4d7d912e6bfa75c25d63e7d8cefd4f962ae3b"
QUALIFIED_REHEARSAL_SHA = "99fcfb765a909653e51d749811bda450f934fae077317c571fefbb41cdfa6fed"


def describe_rehearsal_protocol():
    return dict(
        definition="minigrid_private_rehearsal_F_then_A_games_v1",
        phases=("current_only", "joint_ABC_F"),
        source_native_fit_counts=(8, 8),
        source_native_row_counts=(180, 180),
        source_head_rows=(36, 36),
        upstream_paid_fit_calls=16,
        upstream_paid_fit_rows=342,
        new_fit_calls=0,
        new_replay_rows=0,
        games_per_phase=4,
        contexts=("F-blue", "F-red", "A", "A"),
        roles=("probe_test", "probe_test", "train", "train"),
        seed=17,
        max_steps_per_game=16,
        max_steps_per_phase=64,
        actions="raw K7 first argmax, all native actions legal",
        wall_seconds=150,
        peak_rss_bytes=512 * 1024**2,
        qualification="fixed known F and returned A; retention, not relearning/generalization/CPC or equal-compute claim",
    )


def _qualified_rehearsal_bytes(path):
    if path.stat().st_size > 1024**2:
        raise ValueError("qualified118 private rehearsal exceeds1MiB")
    raw = path.read_bytes()
    if len(raw) != 656386 or sha256(raw).hexdigest() != QUALIFIED_REHEARSAL_SHA:
        raise ValueError("qualified118 private rehearsal bytes changed")
    return raw


def _rehearsal_states(report):
    from scripts.run_minigrid_linear_fit_control import _context_linear_training_snapshot

    limits = dict(
        max_training_updates=4,
        max_wall_seconds=90,
        max_replay_examples=54,
        max_hidden_width=None,
        max_process_rss_bytes=512 * 1024**2,
    )
    budget = report.get("budget", {})
    arms = report.get("arms", ())
    if (
        report.get("definition") != "minigrid_private_affine_rehearsal_comparison_v1"
        or report.get("status") != "complete"
        or len(arms) != 2
        or [a.get("arm") for a in arms] != ["current_only", "joint_ABC_F"]
        or not _same_value(budget.get("initial_limits"), limits)
        or not _same_value(budget.get("limits"), limits)
        or not _same_value(
            (budget.get("updates_completed"), budget.get("replay_examples_completed")), (4, 54)
        )
        or budget.get("started_at") != budget.get("observed_budget_started_at")
        or budget.get("ended_at", -1) < budget.get("started_at", 0)
        or not 0 <= report.get("elapsed_seconds", -1) <= 90
    ):
        raise ValueError("complete original118 matched arms/lifetime/charges required")
    states = []
    for index, arm in enumerate(arms):
        b, state = arm.get("budget", {}), arm.get("native_endpoint", {})
        replay = 54 if index else 0
        if (
            arm.get("status") != "complete"
            or arm.get("source_native_state_unchanged") is not True
            or type(arm.get("source_fork_calls")) is not int
            or arm["source_fork_calls"] != 1
            or not _same_value(
                b.get("initial_limits"),
                limits | dict(max_training_updates=2, max_replay_examples=replay),
            )
            or not _same_value(b.get("limits"), b.get("initial_limits"))
            or b.get("started_at") != budget["started_at"]
            or b.get("observed_budget_started_at") != budget["started_at"]
            or not _same_value(
                (b.get("updates_completed"), b.get("replay_examples_completed")), (2, replay)
            )
            or b.get("ended_at", -1) < b.get("started_at", 0)
            or b.get("ended_at", 0) > budget["ended_at"]
            or arm.get("updates_accounted_by_study") != 2
            or arm.get("committed_old_TRAIN_reuse_rows") != replay
            or arm.get("committed_extra_F_rows") != 54 - replay
            or arm.get("attempts")
            != [
                dict(
                    update=n,
                    fit_rows_attempted=36,
                    old_TRAIN_reuse_charged=replay // 2,
                    extra_F_exposures_charged=(54 - replay) // 2,
                )
                for n in (1, 2)
            ]
            or arm.get("probe", {}).get("completed_updates") != 2
            or arm.get("probe", {}).get("completed_training_examples") != 72
            or [p.get("updates") for p in arm.get("old_task_readouts", ())] != [0, 1, 2]
            or (state.get("completed_updates"), state.get("training_examples")) != (8, 180)
            or state.get("failed_work") is not None
            or any(type(h) is not dict for h in state.get("heads", ()))
            or [h.get("fit_rows") for h in state.get("heads", ())] != [36, 36]
        ):
            raise ValueError(
                "complete paid118 arm source/native8/180/head36/exposure evidence required"
            )
        states.append(state)
    return tuple(_context_linear_training_snapshot(state) for state in states)


def describe_protocol():
    return dict(
        definition="minigrid_affine_F_before_after_games_v1",
        phases=("before_probe", "after_probe"),
        source_native_fit_counts=(6, 8),
        source_native_row_counts=(108, 126),
        source_head_rows=(27, 9),
        upstream_paid_fit_calls=12,
        upstream_paid_fit_rows=198,
        new_fit_calls=0,
        new_replay_rows=0,
        games_per_phase=2,
        contexts=("F-blue", "F-red"),
        role="probe_test",
        seed=17,
        max_steps_per_game=16,
        actions="raw K7 first argmax, all native actions legal",
        wall_seconds=150,
        peak_rss_bytes=512 * 1024**2,
        qualification=(
            "same fixed F geometry/seed as original private probe; execution, not independent "
            "generalization; no old-task native retention, return, CPC or efficiency claim"
        ),
    )


def _qualified_probe_bytes(path):
    if path.stat().st_size > 1024**2:
        raise ValueError("qualified113 F probe exceeds1MiB")
    raw = path.read_bytes()
    if len(raw) != 930239 or sha256(raw).hexdigest() != QUALIFIED_PROBE_SHA:
        raise ValueError("qualified113 F probe bytes changed")
    return raw


def _after_probe_state(report):
    from scripts.run_minigrid_linear_fit_control import (
        QUALIFIED_AFTER_C_SHA,
        QUALIFIED_CAPACITY_TRAIN_SHA,
        _context_linear_training_snapshot,
    )

    limits = dict(
        max_training_updates=2,
        max_wall_seconds=90,
        max_replay_examples=0,
        max_hidden_width=None,
        max_process_rss_bytes=512 * 1024**2,
    )
    ledger = report.get("budget", {})
    probe = report.get("probe", {})
    if (
        report.get("definition") != "minigrid_after_c_affine_fresh_F_probe_v1"
        or report.get("status") != "complete"
        or report.get("source_native_state_unchanged") is not True
        or type(report.get("source_fork_calls")) is not int
        or report["source_fork_calls"] != 1
        or probe.get("status") != "complete"
        or probe.get("completed_updates") != 2
        or probe.get("completed_training_examples") != 18
        or ledger.get("updates_completed") != 2
        or ledger.get("replay_examples_completed") != 0
        or not _same_value(report.get("initial_probe_limits"), limits)
        or not _same_value(ledger.get("limits"), limits)
        or not 0 <= ledger.get("elapsed_seconds", -1) <= 90
        or ledger.get("ended_at", -1) < ledger.get("started_at", 0)
        or report.get("source_bindings")
        != dict(
            after_c_sha256=QUALIFIED_AFTER_C_SHA,
            after_c_bytes=377836,
            training_sha256=QUALIFIED_CAPACITY_TRAIN_SHA,
            training_bytes=2346812,
        )
    ):
        raise ValueError("completed paid113 probe/source/budget evidence required")
    state = _context_linear_training_snapshot(report["disposable_state"])
    if (
        (state.completed_updates, state.training_examples) != (8, 126)
        or state.failed_work is not None
        or any(head is None or head.fit_rows != 9 for head in state.heads)
    ):
        raise ValueError("paid113 endpoint must retain native8fits126rows and two9-row heads")
    return state


def run_saved_affine_probe_games(probe_path, before_path=None, *, rehearsal=False):
    from scripts.run_minigrid_linear_fit_control import (
        QUALIFIED_AFTER_C_SHA,
        _after_c_affine_state,
        _qualified_after_c_bytes,
    )

    if type(rehearsal) is not bool or (rehearsal and before_path is not None):
        raise ValueError("exact rehearsal flag and one qualified118 source required")
    protocol = describe_rehearsal_protocol() if rehearsal else describe_protocol()
    limits = ToyExecutionBudget(
        max_training_updates=0,
        max_replay_examples=0,
        max_wall_seconds=150,
        max_process_rss_bytes=512 * 1024**2,
    )
    report: dict[str, Any] = dict(
        definition=protocol["definition"],
        protocol=protocol,
        status="running",
        reason=None,
        phases=[],
        source_unchanged=None,
        initial_readout_limits=asdict(limits),
    )
    phases = report["phases"]
    preceding: list[tuple[dict, Any, dict]] = []
    pending = None
    started = monotonic()
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        budget = ToyBudgetSession(limits, monotonic)
        budget.attach_memory(memory)
        binding, expected_limits = budget_binding(budget), replace(limits)
        try:
            before_raw, probe_raw = (
                (b"", _qualified_rehearsal_bytes(probe_path))
                if rehearsal
                else (
                    _qualified_after_c_bytes(before_path),
                    _qualified_probe_bytes(probe_path),
                )
            )
            probe_report = json.loads(probe_raw)
            if rehearsal:
                states = _rehearsal_states(probe_report)
                report.update(
                    source_bindings=dict(
                        rehearsal_sha256=QUALIFIED_REHEARSAL_SHA, rehearsal_bytes=len(probe_raw)
                    ),
                    original_study_budget=deepcopy(probe_report["budget"]),
                    original_arm_budgets=[deepcopy(a["budget"]) for a in probe_report["arms"]],
                    original_arm_fork_calls=[a["source_fork_calls"] for a in probe_report["arms"]],
                    original_probe_budget=deepcopy(probe_report["original_probe_budget"]),
                    original_probe_source_fork_calls=probe_report[
                        "original_probe_source_fork_calls"
                    ],
                )
            else:
                states = (
                    _after_c_affine_state(json.loads(before_raw)),
                    _after_probe_state(probe_report),
                )
                report["source_bindings"] = dict(
                    before_sha256=QUALIFIED_AFTER_C_SHA,
                    before_bytes=len(before_raw),
                    after_sha256=QUALIFIED_PROBE_SHA,
                    after_bytes=len(probe_raw),
                )
                # Retain clocks/charges as evidence, never as new spending authority.
                report["original_probe_budget"] = deepcopy(probe_report["budget"])
                report["original_probe_source_fork_calls"] = probe_report["source_fork_calls"]
            provenance = {
                key: report[key]
                for key in (
                    "source_bindings",
                    "original_probe_budget",
                    "original_probe_source_fork_calls",
                    "initial_readout_limits",
                    "protocol",
                )
                + (
                    ("original_study_budget", "original_arm_budgets", "original_arm_fork_calls")
                    if rehearsal
                    else ()
                )
            }
            check_provenance, original_provenance = paid_report_guard(provenance)
            source = ContextLinearLearner()
            initial = source.snapshot_state()

            def guard():
                if monotonic() - started >= 140:
                    raise TimeoutError("F game work deadline reached; cleanup time reserved")
                if (
                    budget_binding(budget) != binding
                    or not _same_value(vars(budget.budget), vars(expected_limits))
                    or not _same_value(
                        (budget.updates_completed, budget.replay_examples_completed), (0, 0)
                    )
                ):
                    raise ValueError("original F game lifetime budget changed")
                if any(report[key] is not value for key, value in provenance.items()):
                    raise ValueError("original F source/charge evidence was replaced")
                check_provenance()
                if (
                    (_qualified_rehearsal_bytes(probe_path) != probe_raw)
                    if rehearsal
                    else (
                        _qualified_after_c_bytes(before_path) != before_raw
                        or _qualified_probe_bytes(probe_path) != probe_raw
                    )
                ):
                    raise ValueError("preceding qualified F model evidence changed")
                if report["phases"] is not phases or len(phases) != len(preceding):
                    raise ValueError("preceding F game prefix changed or was replaced")
                for current, (original, check, _) in zip(phases, preceding):
                    if current is not original:
                        raise ValueError("preceding F game phase was replaced")
                    check()
                if source.snapshot_state() != initial:
                    raise ValueError("F game untrained source changed")

            for name, state in zip(protocol["phases"], states):
                guard()
                try:
                    phase = (
                        evaluate_context_linear_rehearsal_checkpoint(
                            source, state, arm=name, budget=budget, check_boundary=guard
                        )
                        if rehearsal
                        else evaluate_context_linear_probe_checkpoint(
                            source, state, phase=name, budget=budget, check_boundary=guard
                        )
                    )
                except MiniGridEvaluationFailed as error:
                    pending = error.report
                    raise
                pending = phase
                if (
                    phase.get("status") != "complete"
                    or phase.get("state_unchanged") is not True
                    or len(phase.get("episodes", ())) != protocol["games_per_phase"]
                    or any(
                        e.get("status") != "complete"
                        or e.get("environment", {}).get("closed") is not True
                        for e in phase["episodes"]
                    )
                ):
                    raise ValueError("preceding F game phase is incomplete")
                guard()
                check, detached = paid_report_guard(phase)
                phases.append(phase)
                preceding.append((phase, check, detached))
                pending = None
            guard()
            budget.before_final()
            report.update(status="complete", source_unchanged=True)
        except Exception as error:
            report.update(status="failed", reason=f"{type(error).__name__}: {error}"[:512])
            report["phases"] = [detached for _, _, detached in preceding] + (
                [] if pending is None else [deepcopy(pending)]
            )
            if "original_provenance" in locals():
                report.update(original_provenance)
        finally:
            report["budget"] = dict(
                limits=asdict(budget.budget),
                updates_completed=budget.updates_completed,
                replay_examples_completed=budget.replay_examples_completed,
                started_at=binding[2],
                observed_budget_started_at=budget.started_at,
                ended_at=monotonic(),
                elapsed_seconds=monotonic() - started,
                sampled_peak_rss_bytes=memory.peak_bytes,
            )
    return report
