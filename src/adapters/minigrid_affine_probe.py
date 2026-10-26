"""Bind the saved affine control to existing isolated MiniGrid F diagnostics.

Inputs: admitted after-C state and original TRAIN episodes for readonly monitors.
Output: original probe report, row diagnostics, collection/partial costs and one
disposable native endpoint. No benchmark/deployed updates, return or promotion.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import pickle
from time import monotonic
from typing import Any

from src.adapters import minigrid_learnability as reserved
from src.adapters import minigrid_learning_pilot as shared
from src.adapters.minigrid_context_linear_learner import (
    ContextLinearLearner,
    copy_context_linear_training_snapshot,
)
from src.adapters.minigrid_train_catalogue import _validate_sources, compose_catalogue_training
from src.app.coupled_attribution import _same_value
from src.app.fresh_task_probe import run_fresh_task_probe, _validate_batches
from src.app.probe_return_phase import budget_binding
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.minigrid_context_encoding import context_public_history
from src.shared.process_memory import ProcessRssSampler


def describe_protocol():
    return dict(
        definition="minigrid_after_c_affine_fresh_F_probe_v1",
        selected_native_updates=6,
        selected_native_rows=108,
        paid_source_fits=10,
        paid_source_rows=180,
        probe_updates=2,
        probe_rows=18,
        checkpoint_updates=(0, 1, 2),
        probe_task="existing distinct Fblue/Fred, branch2, fixedseed17, same train/test layout and seed",
        collector="original16train/2test release; validate all, select fixedfirst2train only",
        transport="independent raw296 feedforward rows, original episodes retained as evidence",
        monitors="two27row TRAIN batches, A/B/C grouped by public cue; original source spans",
        score_calls=12,
        score_rows=216,
        decision_diagnostic_calls=12,
        collection_episodes=4,
        max_collection_steps=64,
        wall_seconds=90,
        peak_rss_bytes=512 * 1024**2,
        qualification="isolated fixed F fit/interference diagnostic; not heldout/native-probe/general plasticity/CPC benefit",
    )


class _AffineProbeSource(ContextLinearLearner):
    """Existing one-child source pattern; full native state stays unchanged."""

    def __init__(self, saved):
        super().__init__()
        self.restore_state(saved)
        self.fork_calls = 0
        self.child: ContextLinearLearner | None = None

    def fork(self):
        self.fork_calls += 1
        if self.fork_calls != 1:
            raise ValueError("affine probe permits exactly one disposable fork")
        self.child = super().fork()
        return self.child


def _prepare_monitors(episodes):
    expected = tuple(
        f"minigrid-source-{context}-{cue}"
        for context in ("A", "B", "C_train")
        for cue in ("blue", "red")
    )
    if type(episodes) is not tuple or tuple(e.episode_id for e in episodes) != expected:
        raise ValueError("affine probe requires ordered original A/B/C_train sources")
    witness, _ = _validate_sources(episodes, 1)
    if any(
        len(e.transitions) != 9
        or int(context_public_history(e.transitions[0].observation)[297]) != n % 2
        for n, e in enumerate(episodes)
    ):
        raise ValueError("affine monitor sources require original nine-row public blue/red pairs")
    plans = tuple(compose_catalogue_training(episodes[cue::2], 1) for cue in (0, 1))
    return (
        witness,
        tuple(plan.training[0] for plan in plans),
        tuple(plan.source_spans for plan in plans),
    )


def _prepare_probe(raw_training, raw_evaluation):
    # Why: refuse role contradictions BEFORE reconstruction from episode metadata.
    _validate_batches(raw_training, raw_evaluation)
    if len(raw_training) != 16 or len(raw_evaluation) != 2:
        raise ValueError("original Fcollector16train/2test envelope required")
    selected = raw_training[:2]
    for packets in (selected, raw_evaluation):
        if tuple(b.context for b in packets) != ("F-blue", "F-red"):
            raise ValueError("fixed reserved Fblue/Fred order required")
    history = tuple(reserved._history_packets(packets) for packets in (selected, raw_evaluation))
    # These are feedforward independent rows. Keep complete original episodes in
    # the caller report, without inventing a 27-step monitor trajectory.
    transport = tuple(tuple(replace(b, episode=None) for b in packets) for packets in history)
    return transport, [asdict(shared._episode(b)) for b in (*selected, *raw_evaluation)]


def _old_task_rows(probe, spans):
    out = []
    for point in probe.get("training_readouts", {}).get("checkpoints", []):
        if point["status"] != "complete":
            continue
        row = dict(updates=point["updates"], contexts={})
        for cue, source_spans in enumerate(spans):
            detail = point["decision_diagnostics"][f"batch-{cue}"]
            for span in source_spans:
                decisions = detail["decisions"][span.start : span.stop]
                row["contexts"][span.episode_id] = dict(
                    rows=len(decisions),
                    correct=sum(d["correct"] for d in decisions),
                    ce=sum(d["loss"] for d in decisions) / len(decisions),
                    actions=[d["predicted_class"] for d in decisions],
                    expected=[d["target_class"] for d in decisions],
                )
        out.append(row)
    return out


def run_affine_fresh_probe(episodes, state, *, check_boundary=lambda: None):
    saved = copy_context_linear_training_snapshot(state)
    if (
        saved.completed_updates != 6
        or saved.training_examples != 108
        or saved.failed_work is not None
        or any(h is None or h.fit_rows != 27 for h in saved.heads)
    ):
        raise ValueError("affine probe requires genuine after-C6/108/two27-row heads")
    witness, monitors, spans = _prepare_monitors(episodes)
    report: dict[str, Any] = dict(
        definition=describe_protocol()["definition"],
        protocol=describe_protocol(),
        status="running",
        reason=None,
        collection=[],
        probe_data=[],
        probe=None,
        old_task_readouts=[],
        monitor_source_spans=[[asdict(s) for s in group] for group in spans],
        source_native_state_unchanged=None,
        source_fork_calls=0,
        disposable_state=None,
    )
    limits = ToyExecutionBudget(
        max_training_updates=2,
        max_replay_examples=0,
        max_wall_seconds=90,
        max_process_rss_bytes=512 * 1024**2,
    )
    report["initial_probe_limits"] = asdict(limits)
    started = monotonic()
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        budget = ToyBudgetSession(limits, monotonic)
        budget.attach_memory(memory)
        binding, expected_limits, original_clock = (
            budget_binding(budget),
            replace(limits),
            budget.clock,
        )
        source = _AffineProbeSource(saved)
        sealed = None
        collection_owner = report["collection"]
        known_collection: list[dict[str, Any]] = []
        signatures: list[tuple[Any, Any]] = []
        observed_updates = 0
        observed_clock = budget.last_clock

        def guard():
            nonlocal observed_updates, observed_clock
            if (
                budget_binding(budget) != binding
                or not _same_value(vars(budget.budget), vars(expected_limits))
                or type(budget.updates_completed) is not int
                or not observed_updates <= budget.updates_completed <= 2
                or type(budget.replay_examples_completed) is not int
                or budget.replay_examples_completed != 0
            ):
                raise ValueError("original isolated affine probe budget/charges changed")
            if (
                type(budget.last_clock) not in (int, float)
                or not observed_clock <= budget.last_clock
            ):
                raise ValueError("original isolated affine probe clock regressed")
            observed_updates = budget.updates_completed
            budget.before_final()
            observed_clock = budget.last_clock
            check_boundary()
            witness.check(episodes)
            if source.snapshot_state() != saved or source.fork_calls not in (0, 1):
                raise ValueError("original affine source or one-fork ownership changed")
            # The existing collector appends closed records. Keep only its
            # bounded four-record prefix before admitting another source or fit.
            if (
                report["collection"] is not collection_owner
                or len(collection_owner) > 4
                or pickle.dumps(collection_owner[: len(known_collection)], protocol=5)
                != pickle.dumps(known_collection, protocol=5)
            ):
                raise ValueError("preceding affine collection evidence changed")
            for item in collection_owner[len(known_collection) :]:
                if item.get("closed") is not True or item.get("status") != "complete":
                    raise ValueError("preceding affine collection is incomplete")
                known_collection.append(deepcopy(item))
            current = None if source.child is None else source.child.snapshot_state()
            native_updates = 0 if current is None else current.completed_updates - 6
            if budget.updates_completed != native_updates or (
                current is not None and current.training_examples != 108 + 9 * native_updates
            ):
                raise ValueError("isolated affine charges differ from observed native commits")
            if (
                sealed is not None
                and pickle.dumps((report["collection"], report["probe_data"]), protocol=5) != sealed
            ):
                raise ValueError("preceding affine probe source evidence changed")
            if any(not shared._check_signature(data, signature) for data, signature in signatures):
                raise ValueError("affine probe original role-separated arrays changed")

        try:
            guard()
            raw_train, raw_test = reserved._collect_probe_sources(
                guard, report["collection"], distinct_probe=True
            )
            (training, evaluation), report["probe_data"] = _prepare_probe(raw_train, raw_test)
            _validate_batches(training, evaluation, monitors)
            cohorts = (raw_train, raw_test, training, evaluation, monitors)
            signatures = [(data, shared._signature(data)) for data in cohorts]
            sealed = pickle.dumps((report["collection"], report["probe_data"]), protocol=5)
            report["retained_input_array_bytes"] = sum(
                b.features.nbytes + b.targets.nbytes for data in cohorts for b in data
            )
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
            probe = report["probe"]
            report["old_task_readouts"] = _old_task_rows(probe, spans)
            if probe["status"] != "complete":
                raise ValueError(f"isolated affine probe incomplete: {probe['reason']}")
            if source.child is None or source.fork_calls != 1:
                raise ValueError("affine probe did not retain its one disposable learner")
            final = source.child.snapshot_state()
            if (
                final.completed_updates != 6 + probe["completed_updates"]
                or final.training_examples != 108 + probe["completed_training_examples"]
                or final.failed_work is not None
                or budget.updates_completed != 2
            ):
                raise ValueError("affine native source-plus-probe counters disagree")
            guard()
            report.update(status="complete", source_native_state_unchanged=True)
        except Exception as error:
            report.update(status="failed", reason=f"{type(error).__name__}: {error}"[:512])
            report["known_collection_prefix"] = deepcopy(known_collection)
        finally:
            report["source_fork_calls"] = source.fork_calls
            if source.child is not None:
                try:
                    report["disposable_state"] = asdict(source.child.snapshot_state())
                except Exception as error:
                    report.update(
                        status="failed",
                        disposable_state_error=f"{type(error).__name__}: {error}"[:512],
                    )
            try:
                report["source_native_state_unchanged"] = source.snapshot_state() == saved
            except Exception as error:
                report["source_snapshot_error"] = f"{type(error).__name__}: {error}"[:512]
            report["budget"] = dict(
                limits=asdict(budget.budget),
                updates_completed=budget.updates_completed,
                replay_examples_completed=budget.replay_examples_completed,
                started_at=budget.started_at,
                ended_at=original_clock(),
                elapsed_seconds=monotonic() - started,
                sampled_peak_rss_bytes=memory.peak_bytes,
            )
    return report
