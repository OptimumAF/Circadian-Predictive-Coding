"""Compose the existing disposable F16/A16 probe at a live joint128 endpoint.

No training policy, new runner, model export, final data or promotion authority.
The caller owns benchmark preservation; existing generic probe owns its fork.
"""

from dataclasses import asdict
from time import monotonic
from typing import Any

from scripts.recurrent_readout_example import _budget_report, _check_paid_evidence
from src.adapters.minigrid_learning_pilot import _signature, _check_signature
from src.adapters.minigrid_public_cue_recurrent_learner import (
    PublicCueRecurrentLearner,
    PublicCueRecurrentPolicy,
    copy_snapshot,
)
from src.adapters.minigrid_learnability import _collect_probe_sources
from src.adapters.minigrid_probe_rollout import ProbeRollouts
from src.adapters.minigrid_return_fit import return_training_sources
from src.app.fresh_task_probe import run_fresh_task_probe
from src.app.probe_return_phase import paid_report_guard
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget


def describe():
    return dict(
        definition="joint128_disposable_F16_A16_v1",
        source_updates=128,
        probe_updates=16,
        return_updates=16,
        teacher_points=[0, 4, 8, 16],
        monitors="two original complete A TRAIN episodes; no concatenated recurrent episodes",
        native_endpoints=[144, 160],
        native_games="F blue/red probe_test plus all ABC blue/red TRAIN after EACH phase",
        endpoint_seconds=80,
        update_seconds=180,
        replay_updates=0,
        final_available=False,
        limitation="disposable diagnostic, not deployed adaptation/generalization/CPC advantage",
    )


def run_probe(source, state, old_sources, *, check_boundary, memory):
    report: dict[str, Any] = dict(
        status="incomplete",
        reason=None,
        collection=dict(reports=[]),
        probe=None,
        endpoints=[],
        probe_budget=None,
        return_budget=None,
    )
    probe_budget = return_budget = None
    try:
        owned = copy_snapshot(state)
        if (
            type(source) is not PublicCueRecurrentLearner
            or owned.policy != PublicCueRecurrentPolicy()
            or owned.failed_work is not None
            or (owned.completed_updates, owned.training_examples) != (128, 1152)
            or any((h.completed_updates, h.training_examples) != (64, 576) for h in owned.heads)
        ):
            raise ValueError("joint probe requires the complete original native128 endpoint")
        games = ProbeRollouts(128, public_cue=True, joint_public_cue=True)
        report["endpoints"] = games.records
        collection = report["collection"]
        collected = collection["reports"]
        collection_check = None
        signatures: list[tuple[Any, Any]] = []

        def guard():
            check_boundary()
            games.check()
            if report["endpoints"] is not games.records:
                raise ValueError("joint probe endpoint list replaced")
            if collection_check is not None:
                _check_paid_evidence(
                    report["collection"],
                    collection,
                    collection_check,
                    "joint probe collection evidence changed",
                )
            if not all(_check_signature(data, sig) for data, sig in signatures):
                raise ValueError("joint probe original inputs changed")

        training, evaluation = _collect_probe_sources(guard, collected, distinct_probe=True)
        if len(collected) != 4 or any(
            row.get("closed") is not True
            or row.get("status") != "complete"
            or row.get("environment_steps") != 9
            or row.get("role") != role
            or row.get("cue") != cue
            for row, (role, cue) in zip(
                collected,
                (
                    ("probe_train", "blue"),
                    ("probe_train", "red"),
                    ("probe_test", "blue"),
                    ("probe_test", "red"),
                ),
                strict=True,
            )
        ):
            raise ValueError("joint probe requires four complete closed original F sources")
        collection_check, _ = paid_report_guard(collection)
        monitors = old_sources[:2]
        templates = return_training_sources(monitors)
        signatures = [
            (data, _signature(data)) for data in (training, evaluation, monitors, templates)
        ]
        limits = ToyExecutionBudget(
            max_training_updates=16,
            max_replay_examples=0,
            max_wall_seconds=180,
            max_process_rss_bytes=768 * 1024**2,
        )
        probe_budget = ToyBudgetSession(limits, monotonic)
        return_budget = ToyBudgetSession(ToyExecutionBudget(**asdict(limits)), monotonic)
        probe_budget.attach_memory(memory)
        return_budget.attach_memory(memory)
        guard()
        report["probe"] = run_fresh_task_probe(
            source,
            state,
            training,
            evaluation,
            budget=probe_budget,
            checkpoint_updates=(0, 4, 8, 16),
            decision_diagnostics=True,
            training_readouts=monitors,
            return_training=templates,
            return_budget=return_budget,
            endpoint_observer=games.observe_qualified,
            check_boundary=guard,
        )
        guard()
        paid = report["probe"]
        if (
            paid.get("status") != "complete"
            or paid.get("completed_updates") != 16
            or paid.get("return_phase", {}).get("status") != "complete"
            or paid["return_phase"].get("completed_updates") != 16
            or (probe_budget.updates_completed, return_budget.updates_completed) != (16, 16)
            or len(games.records) != 2
            or any(r["status"] != "complete" for r in games.records)
        ):
            raise ValueError("joint disposable probe or return did not complete")
        report["status"] = "complete"
    except Exception as error:
        report["reason"] = f"{type(error).__name__}: {error}"
    finally:
        report["probe_budget"] = _budget_report(probe_budget)
        report["return_budget"] = _budget_report(return_budget)
    return report
