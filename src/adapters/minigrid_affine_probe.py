"""Bind the saved affine control to existing isolated MiniGrid F diagnostics.

Inputs: admitted after-C state and original TRAIN episodes for readonly monitors.
Output: original probe report, row diagnostics, collection/partial costs and one
disposable native endpoint. Explicit retained return_old reuses that learner for
two old-TRAIN updates with a separate original owner and retained paid F endpoint.
No benchmark/deployed updates or promotion; fit readouts are not native recovery.
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
from src.app.coupled_attribution import AttributionBatch, _same_value
from src.app.fresh_task_probe import run_fresh_task_probe, _validate_batches
from src.app.probe_return_phase import budget_binding, paid_report_guard
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.minigrid_context_encoding import context_public_history
from src.shared.process_memory import ProcessRssSampler


def describe_protocol(*, retained_a4=False, return_old=False, native_games=False):
    if type(native_games) is not bool or (native_games and return_old is not True):
        raise ValueError("native_games requires an exact bool and return_old=True")
    if type(return_old) is not bool or (return_old and retained_a4 is not True):
        raise ValueError("return_old requires an exact bool and retained_a4=True")
    if type(retained_a4) is not bool:
        raise ValueError("retained_a4 must be an exact bool")
    protocol = dict(
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
    if retained_a4:
        protocol.update(
            definition="minigrid_retained_a4_affine_fresh_F_probe_v1",
            selected_native_updates=4,
            selected_native_rows=54,
            paid_source_fits=None,
            paid_source_rows=None,
            source_provenance="caller in-memory endpoint; upstream fitting/preparation costs external and unknown here, not zero",
            monitors="two18row TRAIN batches, A3/A4 grouped by public cue; four original source spans",
            score_rows=162,
        )
    if return_old:
        protocol.update(
            definition="minigrid_retained_a4_affine_F_then_old_return_v1",
            return_updates=2,
            return_rows=36,
            return_checkpoints=(0, 1, 2),
            return_recipe="two original18row old TRAIN templates once each on same F2 learner; separate original90s owner born before F",
            total_score_calls=24,
            total_score_rows=324,
        )
    if native_games:
        protocol.update(
            native_game_phases=("after_probe", "after_return"),
            native_games_per_phase=6,
            native_game_order="Fblue/Fred probe_test then A3blue/red/A4blue/red TRAIN; seed17/max16/rawK7argmax",
            native_game_budget="caller original150s512MiB zero-update/replay owner born before source preparation; no renewal",
        )
    return protocol


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


def _prepare_monitors(episodes, *, retained_a4=False):
    describe_protocol(retained_a4=retained_a4)
    expected = tuple(
        f"minigrid-source-{context}-{cue}"
        for context in (("A", "A4-train") if retained_a4 else ("A", "B", "C_train"))
        for cue in ("blue", "red")
    )
    if type(episodes) is not tuple or tuple(e.episode_id for e in episodes) != expected:
        raise ValueError(
            "affine probe requires ordered original A3/A4 TRAIN sources"
            if retained_a4
            else "affine probe requires ordered original A/B/C_train sources"
        )
    witness, _ = _validate_sources(episodes, 1)
    if any(
        len(e.transitions) != 9
        or int(context_public_history(e.transitions[0].observation)[297]) != n % 2
        or (retained_a4 and e.context != "A")
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


def run_affine_fresh_probe(
    episodes,
    state,
    *,
    check_boundary=lambda: None,
    retained_a4=False,
    return_old=False,
    native_games=False,
    readout_budget=None,
):
    protocol = describe_protocol(
        retained_a4=retained_a4,
        **(dict(return_old=return_old) if return_old is not False else {}),
        **(dict(native_games=native_games) if native_games is not False else {}),
    )
    readout_check = None
    if native_games:
        from src.adapters import minigrid_policy_evaluation as games

        readout_check = games._return_budget_guard(readout_budget)
        expected = ToyExecutionBudget(
            max_training_updates=0,
            max_replay_examples=0,
            max_wall_seconds=150,
            max_process_rss_bytes=512 * 1024**2,
        )
        if not _same_value(readout_budget.budget, expected):
            raise ValueError("native games require the original150s512MiB zero-work owner")
    elif readout_budget is not None:
        raise ValueError("readout_budget requires native_games=True")
    preparation_started = monotonic() if retained_a4 else None
    source_updates, source_rows, head_rows = (4, 54, 18) if retained_a4 else (6, 108, 27)
    saved = copy_context_linear_training_snapshot(state)
    if (
        saved.completed_updates != source_updates
        or saved.training_examples != source_rows
        or saved.failed_work is not None
        or any(h is None or h.fit_rows != head_rows for h in saved.heads)
    ):
        raise ValueError(
            "affine probe requires genuine retained-A3/A4 4/54/two18-row heads"
            if retained_a4
            else "affine probe requires genuine after-C6/108/two27-row heads"
        )
    witness, monitors, spans = _prepare_monitors(episodes, retained_a4=retained_a4)
    report: dict[str, Any] = dict(
        definition=protocol["definition"],
        protocol=protocol,
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
    native_attempts = native_returns = 0
    native_records: list[dict[str, Any]] = []
    native_proofs: list[Any] = []
    native_work = dict(attempts=0, returned=0, unreturned_work_unknown=False)
    if native_games:
        report.update(native_games=native_records, native_game_work=native_work)
    if preparation_started is not None:
        report["preparation_seconds"] = monotonic() - preparation_started
        report["preparation_scope"] = (
            "input validation/copy and monitors before original probe budget; RSS not sampled here; "
            "caller must include preflight in whole-case limits"
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
        return_budget = None
        return_limits = return_binding = return_clock = None
        return_templates: tuple[AttributionBatch, ...] = ()
        endpoint_owner = endpoint_check = detached_endpoint = None
        if return_old:
            return_limits = ToyExecutionBudget(
                max_training_updates=2,
                max_replay_examples=0,
                max_wall_seconds=90,
                max_process_rss_bytes=512 * 1024**2,
            )
            return_budget = ToyBudgetSession(return_limits, monotonic)
            return_budget.attach_memory(memory)
            return_binding, return_clock = budget_binding(return_budget), return_budget.clock
            return_limits = replace(return_limits)
            report.update(probe_endpoint=None, return_old_task_readouts=[])
            return_templates = tuple(
                replace(
                    b,
                    features=b.features.copy(),
                    targets=b.targets.copy(),
                    row_ids=tuple(f"old-return-{cue}:{i}" for i in range(len(b.features))),
                )
                for cue, b in enumerate(monitors)
            )
        sealed = None
        collection_owner = report["collection"]
        known_collection: list[dict[str, Any]] = []
        signatures: list[tuple[Any, Any]] = []
        observed_updates = 0
        observed_clock = budget.last_clock
        collection_checks: list[tuple[dict, Any]] = []
        probe_data_owner = probe_owner = None
        data_check = probe_check = protocol_check = trusted_probe = None
        if retained_a4:
            protocol_check, _ = paid_report_guard(protocol)

        def check_native_prefix():
            assert readout_check is not None
            readout_check()
            if (
                report.get("native_games") is not native_records
                or len(native_records) != len(native_proofs)
                or len(native_records) > 2
            ):
                raise ValueError("original bounded native game prefix changed")
            if report.get("native_game_work") is not native_work or not _same_value(
                native_work,
                dict(
                    attempts=native_attempts,
                    returned=native_returns,
                    unreturned_work_unknown=native_attempts != native_returns,
                ),
            ):
                raise ValueError("original native game-call costs changed")
            for current, (owner, check, _) in zip(native_records, native_proofs):
                if current is not owner:
                    raise ValueError("preceding paid native game record was replaced")
                check()

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
            # Why: the short return's existing observer owns both clock/RSS
            # observations. Its parent guard proves bindings without advancing
            # either watched owner behind the observer's resource witnesses.
            if not return_old:
                budget.before_final()
            observed_clock = budget.last_clock
            check_boundary()
            witness.check(episodes)
            if native_games:
                check_native_prefix()
            if return_budget is not None:
                if (
                    budget_binding(return_budget) != return_binding
                    or not _same_value(return_budget.budget, return_limits)
                    or type(return_budget.updates_completed) is not int
                    or not 0 <= return_budget.updates_completed <= 2
                    or type(return_budget.replay_examples_completed) is not int
                    or return_budget.replay_examples_completed != 0
                ):
                    raise ValueError("original old TRAIN return owner or charges changed")
                if endpoint_check is not None:
                    if report["probe_endpoint"] is not endpoint_owner:
                        raise ValueError("paid F endpoint was replaced during old TRAIN return")
                    endpoint_check()
            if retained_a4:
                if report["protocol"] is not protocol:
                    raise ValueError("original retained probe protocol was replaced")
                if protocol_check is not None:
                    protocol_check()
                if not _same_value(state, saved):
                    raise ValueError("caller retained source snapshot changed")
                for index, (owner, check) in enumerate(collection_checks):
                    if index >= len(collection_owner) or collection_owner[index] is not owner:
                        raise ValueError("preceding retained collection record was replaced")
                    check()
                if data_check is not None:
                    if report["probe_data"] is not probe_data_owner:
                        raise ValueError("retained probe data evidence was replaced")
                    data_check()
                if probe_check is not None:
                    if report["probe"] is not probe_owner:
                        raise ValueError("paid retained probe evidence was replaced")
                    probe_check()
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
                if retained_a4:
                    check, _ = paid_report_guard(item)
                    collection_checks.append((item, check))
            current = None if source.child is None else source.child.snapshot_state()
            native_updates = 0 if current is None else current.completed_updates - source_updates
            return_count = 0 if return_budget is None else return_budget.updates_completed
            if budget.updates_completed + return_count != native_updates or (
                current is not None
                and current.training_examples
                != source_rows + 9 * budget.updates_completed + 18 * return_count
            ):
                raise ValueError("isolated affine charges differ from observed native commits")
            if (
                sealed is not None
                and pickle.dumps((report["collection"], report["probe_data"]), protocol=5) != sealed
            ):
                raise ValueError("preceding affine probe source evidence changed")
            if any(not shared._check_signature(data, signature) for data, signature in signatures):
                raise ValueError("affine probe original role-separated arrays changed")
            if native_games:
                # Snapshot ports above may invoke callbacks after the first proof.
                check_native_prefix()

        def observe(child, phase, boundary):
            nonlocal \
                endpoint_owner, \
                endpoint_check, \
                detached_endpoint, \
                native_attempts, \
                native_returns
            observed = []

            def read():
                current = child.snapshot_state()
                expected = (
                    (source_updates + 2, source_rows + 18, 9)
                    if phase == "after_probe"
                    else (source_updates + 4, source_rows + 54, 18)
                )
                if (
                    current.failed_work is not None
                    or (current.completed_updates, current.training_examples) != expected[:2]
                    or any(h is None or h.fit_rows != expected[2] for h in current.heads)
                ):
                    raise ValueError("actual short return endpoint differs from paid native recipe")
                observed.append(current)

            def recheck():
                if not _same_value(child.snapshot_state(), observed[0]):
                    raise ValueError("native endpoint changed during its protected observation")

            boundary(read, after_clocks=recheck)
            if phase == "after_probe":
                endpoint_owner = report["probe_endpoint"] = asdict(observed[0])
                endpoint_check, detached_endpoint = paid_report_guard(endpoint_owner)
            guard()
            if native_games:
                if (
                    len(native_records) >= 2
                    or phase != ("after_probe", "after_return")[len(native_records)]
                ):
                    raise ValueError("native game phases must follow the original F/return order")
                native_attempts += 1
                native_work.update(attempts=native_attempts, unreturned_work_unknown=True)
                row = dict(phase=phase, games=None)

                def retain():
                    check, detached = paid_report_guard(row)
                    native_records.append(row)
                    native_proofs.append((row, check, detached))

                def qualify_readout(read_ports=None, *, after_clocks=None):
                    def read_all():
                        recheck()
                        if read_ports is not None:
                            read_ports()
                        recheck()

                    def read_final():
                        recheck()
                        if after_clocks is not None:
                            after_clocks()
                        recheck()

                    boundary(read_all, after_clocks=read_final)

                try:
                    row["games"] = games.evaluate_context_linear_rehearsal_checkpoint(
                        ContextLinearLearner(),
                        observed[0],
                        arm=None,
                        budget=readout_budget,
                        check_boundary=guard,
                        short_return_phase=phase,
                        qualified_boundary=qualify_readout,
                    )
                except games.MiniGridEvaluationFailed as error:
                    row["games"] = error.report
                    retain()
                    raise
                native_returns += 1
                native_work.update(returned=native_returns, unreturned_work_unknown=False)
                retain()
                guard()
                if (
                    row["games"].get("status") != "complete"
                    or row["games"].get("state_unchanged") is not True
                ):
                    raise ValueError("native game readout is incomplete or changed its source")
                boundary(recheck, after_clocks=recheck)

        try:
            guard()
            raw_train, raw_test = reserved._collect_probe_sources(
                guard, report["collection"], distinct_probe=True
            )
            (training, evaluation), report["probe_data"] = _prepare_probe(raw_train, raw_test)
            if retained_a4:
                probe_data_owner = report["probe_data"]
                data_check, _ = paid_report_guard(probe_data_owner)
            _validate_batches(training, evaluation, monitors)
            cohorts: tuple[Any, ...] = (raw_train, raw_test, training, evaluation, monitors)
            if return_old:
                cohorts = (*cohorts, return_templates)
            signatures = [(data, shared._signature(data)) for data in cohorts]
            sealed = pickle.dumps((report["collection"], report["probe_data"]), protocol=5)
            report["retained_input_array_bytes"] = sum(
                b.features.nbytes + b.targets.nbytes for data in cohorts for b in data
            )
            guard()
            return_options: dict[str, Any] = (
                dict(
                    short_affine_return=True,
                    return_training=return_templates,
                    return_budget=return_budget,
                    endpoint_observer=observe,
                )
                if return_old
                else {}
            )
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
                **return_options,
            )
            if retained_a4:
                probe_owner = report["probe"]
                probe_check, trusted_probe = paid_report_guard(probe_owner)
            guard()
            probe = report["probe"]
            report["old_task_readouts"] = _old_task_rows(probe, spans)
            return_updates = return_rows = 0
            if return_old:
                returned = probe.get("return_phase") or {}
                report["return_old_task_readouts"] = _old_task_rows(returned, spans)
                return_updates = returned.get("completed_updates", 0)
                return_rows = returned.get("completed_training_examples", 0)
                if returned.get("status") != "complete" or (return_updates, return_rows) != (2, 36):
                    raise ValueError(
                        "old TRAIN return incomplete; paid F and partial return retained"
                    )
            if probe["status"] != "complete":
                raise ValueError(f"isolated affine probe incomplete: {probe['reason']}")
            if source.child is None or source.fork_calls != 1:
                raise ValueError("affine probe did not retain its one disposable learner")
            final = source.child.snapshot_state()
            if (
                final.completed_updates
                != source_updates + probe["completed_updates"] + return_updates
                or final.training_examples
                != source_rows + probe["completed_training_examples"] + return_rows
                or final.failed_work is not None
                or budget.updates_completed != 2
            ):
                raise ValueError("affine native source-plus-probe counters disagree")
            guard()
            report.update(status="complete", source_native_state_unchanged=True)
        except Exception as error:
            report.update(status="failed", reason=f"{type(error).__name__}: {error}"[:512])
            report["known_collection_prefix"] = deepcopy(known_collection)
            if trusted_probe is not None:
                report["probe"] = trusted_probe
        finally:
            if return_old:
                assert return_budget is not None and return_limits is not None
                assert return_binding is not None and return_clock is not None
                report["return_budget"] = dict(
                    initial_limits=asdict(return_limits),
                    limits=asdict(return_budget.budget),
                    updates_completed=return_budget.updates_completed,
                    replay_examples_completed=return_budget.replay_examples_completed,
                    started_at=return_binding[2],
                    ended_at=return_clock(),
                )
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
            if preparation_started is not None:
                report["whole_diagnostic_seconds"] = monotonic() - preparation_started
            if native_games:
                # Why: capture invokes snapshot and clock ports too. Prove the
                # paid prefixes after those callbacks, before detaching them.
                try:
                    guard()
                except Exception as error:
                    report.update(
                        status="failed",
                        native_capture_error=f"{type(error).__name__}: {error}"[:512],
                    )
                report["native_games"] = [p[2] for p in native_proofs]
                report["native_game_work"] = dict(
                    attempts=native_attempts,
                    returned=native_returns,
                    unreturned_work_unknown=native_attempts != native_returns,
                )
            if return_old:
                report["probe_endpoint"] = detached_endpoint
    return report
