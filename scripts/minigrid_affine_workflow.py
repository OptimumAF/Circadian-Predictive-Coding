"""Compose existing TRAIN stream and known-task games from one fresh endpoint.

Inputs are bundled TRAIN examples; output includes causal training receipts and
target-free game reports. This module owns no optimizer, evaluator or source
admission protocol. Historical scientific readback commands remain separate.
"""

from dataclasses import asdict, replace
from time import monotonic
from typing import Any

from src.adapters.minigrid_context_linear_learner import ContextLinearLearner
from src.adapters.minigrid_policy_evaluation import (
    MiniGridEvaluationFailed,
    evaluate_context_linear_stream_checkpoint,
)
from src.app.coupled_attribution import _same_value
from src.app.probe_return_phase import budget_binding, paid_report_guard
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.infra.minigrid_affine_example import example_affine_training
from src.infra.minigrid_memory import _require_minigrid
from src.shared.process_memory import ProcessRssSampler


def describe_workflow(*, private_adaptation=False, recurring_training=False, a4_adaptation=False):
    from scripts.run_minigrid_linear_fit_control import describe_context_linear_abc

    if type(private_adaptation) is not bool:
        raise ValueError("private_adaptation must be an exact bool")
    if type(recurring_training) is not bool or (recurring_training and private_adaptation):
        raise ValueError("recurring_training must be an exact bool and a distinct mode")
    if type(a4_adaptation) is not bool or (
        a4_adaptation and (private_adaptation or recurring_training)
    ):
        raise ValueError("a4_adaptation must be an exact bool and a distinct mode")
    if a4_adaptation:
        from scripts.minigrid_a4_adaptation import describe_fresh_adaptation

        return describe_fresh_adaptation()
    if recurring_training:
        from scripts.run_minigrid_linear_fit_control import describe_context_linear_recurring
        from src.core.minigrid_key_door import predefined_recurring_train_layout, predefined_layouts

        return dict(
            definition="portable_affine_recurring_train_v1",
            training=describe_context_linear_recurring(),
            new_layout=asdict(predefined_recurring_train_layout()),
            returned_layout=asdict(predefined_layouts()[0]),
            collection="two original G_train teacher episodes; blue/red; seed17/max16; nine rows each",
            games="known G_train blue/red then returned A blue/red; seed17/max16; K7 firstargmax",
            new_fit_calls=10,
            new_fit_rows=252,
            historical_model_reads=0,
            readout_training_updates=0,
            readout_replay_examples=0,
            max_workflow_seconds=300,
            work_deadline_seconds=290,
            max_process_rss_bytes=512 * 1024**2,
            qualification="ordinary known TRAIN acquisition/retention/return; no private promotion/final/probe/generalization/CPC claim",
        )
    if private_adaptation:
        from scripts.minigrid_affine_probe_games import describe_rehearsal_protocol
        from src.adapters.minigrid_affine_rehearsal import describe_protocol
        from src.core.minigrid_key_door import predefined_layouts

        return dict(
            definition="portable_affine_private_adaptation_v1",
            training=describe_context_linear_abc(),
            collection="original F16+2 release from four fresh closed teacher episodes; fixed first2train",
            private=describe_protocol(fresh_reference=True),
            games=describe_rehearsal_protocol()
            | dict(upstream_paid_fit_calls=10, upstream_paid_fit_rows=252),
            returned_layout=asdict(predefined_layouts()[0]),
            new_fit_calls=10,
            new_fit_rows=252,
            historical_source_reads=0,
            max_workflow_seconds=180,
            work_deadline_seconds=170,
            max_process_rss_bytes=512 * 1024**2,
            qualification="current-run private comparison, frozen TRAIN reference; known tasks, no promotion/generalization/CPC claim",
        )
    return dict(
        definition="portable_affine_train_and_play_v1",
        training=describe_context_linear_abc(),
        games="known TRAIN B blue/red, C_train blue/red; seed17; max16 each; K7 raw firstargmax",
        readout_training_updates=0,
        readout_replay_examples=0,
        readout_wall_seconds=150,
        qualification="fresh deterministic conventional fit; known tasks only; no CPC/generalization/release claim",
    )


def _require_training(training, states, *, recurring_training=False):
    if type(recurring_training) is not bool:
        raise ValueError("recurring_training must be an exact bool")
    updates, rows, head_rows = (10, 252, 36) if recurring_training else (6, 108, 27)
    if (
        training.get("status") != "complete"
        or training.get("source_unchanged") is not True
        or training.get("actor_unchanged") is not True
        or training.get("receipt_capture_error") is not None
        or len(training.get("committed_receipts") or ()) != updates
        or training.get("budget", {}).get("updates_completed") != updates
        or len(states) != 1
    ):
        raise ValueError(
            "known games require a complete observed TRAIN stream"
            if recurring_training
            else "known games require a complete observed six-fit TRAIN stream"
        )
    state = states[0]
    if (
        state.completed_updates != updates
        or state.training_examples != rows
        or state.failed_work is not None
        or any(head is None or head.fit_rows != head_rows for head in state.heads)
    ):
        raise ValueError(
            "fresh recurring TRAIN endpoint requires actual10/252/head36"
            if recurring_training
            else "fresh TRAIN endpoint must contain actual six-fit108-row capacity2 state"
        )
    return state


def run_workflow(*, private_adaptation=False, recurring_training=False, a4_adaptation=False):
    if type(private_adaptation) is not bool:
        raise ValueError("private_adaptation must be an exact bool")
    if type(recurring_training) is not bool or (recurring_training and private_adaptation):
        raise ValueError("recurring_training must be an exact bool and a distinct mode")
    if type(a4_adaptation) is not bool or (
        a4_adaptation and (private_adaptation or recurring_training)
    ):
        raise ValueError("a4_adaptation must be an exact bool and a distinct mode")
    if a4_adaptation:
        from scripts.minigrid_a4_adaptation import run_fresh_adaptation

        return run_fresh_adaptation()
    if recurring_training:
        return _run_recurring_training()
    if private_adaptation:
        return _run_private_adaptation()
    from scripts.run_minigrid_linear_fit_control import (
        read_training_sources,
        run_context_linear_stream,
    )

    report: dict[str, Any] = dict(
        definition="portable_affine_train_and_play_v1",
        protocol=describe_workflow(),
        status="running",
        reason=None,
        training=None,
        games=None,
    )
    states: list[Any] = []
    try:
        # Why: a play command must know its optional runtime is usable before
        # spending TRAIN work. Reuse the native boundary's exact version check.
        _require_minigrid()
        with example_affine_training() as path:
            episodes = read_training_sources(path)
        # Why: this is a new first-party run, not restoration of old experiment
        # authority. Only an actually completed native stream hands off a state.
        training = run_context_linear_stream(
            episodes, retain_arrived=True, abc_catalogue=True, endpoint_observer=states.append
        )
        report["training"] = training
        state = _require_training(training, states)
        check_training, preceding = paid_report_guard(training)
        limits = ToyExecutionBudget(
            max_training_updates=0,
            max_replay_examples=0,
            max_wall_seconds=150,
            max_process_rss_bytes=512 * 1024**2,
        )
        started = monotonic()
        with ProcessRssSampler(interval_seconds=0.02) as memory:
            budget = ToyBudgetSession(limits, monotonic)
            budget.attach_memory(memory)
            binding, expected = budget_binding(budget), replace(limits)
            source = ContextLinearLearner()
            initial = source.snapshot_state()

            def guard():
                if monotonic() - started >= 140:
                    raise TimeoutError("known game work deadline reached; cleanup time reserved")
                try:
                    check_training()
                except ValueError as error:
                    raise ValueError("preceding TRAIN evidence changed or was replaced") from error
                if (
                    report["training"] is not training
                    or source.snapshot_state() != initial
                    or budget_binding(budget) != binding
                    or not _same_value(vars(budget.budget), vars(expected))
                    or (budget.updates_completed, budget.replay_examples_completed) != (0, 0)
                ):
                    raise ValueError("preceding TRAIN evidence/source/readout authority changed")

            try:
                guard()
                report["games"] = evaluate_context_linear_stream_checkpoint(
                    source, state, budget=budget, check_boundary=guard, after_c_capacity=True
                )
                guard()
                games = report["games"]
                if (
                    games.get("status") != "complete"
                    or games.get("state_unchanged") is not True
                    or len(games.get("episodes", ())) != 4
                    or any(
                        row.get("status") != "complete"
                        or row.get("environment", {}).get("closed") is not True
                        for row in games["episodes"]
                    )
                ):
                    raise ValueError("known game report is incomplete")
                budget.before_final()
                report["status"] = "complete"
            except MiniGridEvaluationFailed as error:
                report["games"] = error.report
                raise
            finally:
                # Detached known training remains authoritative even if a later
                # callback corrupts the mutable report or game work fails.
                report["training"] = preceding
                report["readout_budget"] = dict(
                    initial_limits=asdict(expected),
                    limits=asdict(budget.budget),
                    updates_completed=budget.updates_completed,
                    replay_examples_completed=budget.replay_examples_completed,
                    started_at=budget.started_at,
                    ended_at=budget.clock(),
                    elapsed_seconds=monotonic() - started,
                    sampled_peak_rss_bytes=memory.peak_bytes,
                )
    except Exception as error:
        report.update(status="failed", reason=f"{type(error).__name__}: {error}"[:512])
    return report


def _collect_recurring_train_sources(check_boundary, reports, *, layout=None):
    # Why: retain the existing workflow port while sharing the bounded collector
    # with validation; the adapter has no script-orchestrator dependencies.
    from src.adapters.minigrid_train_collection import collect_train_pair

    return collect_train_pair(check_boundary, reports, layout=layout)


def _run_recurring_training():
    from scripts.run_minigrid_linear_fit_control import (
        read_training_sources,
        run_context_linear_stream,
    )
    from src.adapters.minigrid_train_catalogue import _validate_sources

    protocol = describe_workflow(recurring_training=True)
    protocol_check, original_protocol = paid_report_guard(protocol)
    report: dict[str, Any] = dict(
        definition=protocol["definition"],
        protocol=protocol,
        status="running",
        reason=None,
        collection=[],
        training=None,
        games=None,
    )
    started, attempted = monotonic(), []
    states: list[Any] = []
    collection = report["collection"]
    collection_proofs: list[tuple[dict, Any, dict]] = []
    checks: list[tuple[str, dict, Any, dict]] = []
    source_witness = None
    with ProcessRssSampler(interval_seconds=0.02) as memory:

        def guard():
            if monotonic() - started >= 290:
                raise TimeoutError(
                    "recurring TRAIN workflow deadline reached; cleanup/capture reserved"
                )
            sampled = memory.sample()
            if sampled is None or memory.peak_bytes is None or memory.peak_bytes > 512 * 1024**2:
                raise RuntimeError("recurring TRAIN workflow RSS unavailable or exceeds512MiB")
            if report["protocol"] is not protocol:
                raise ValueError("original recurring protocol was replaced")
            protocol_check()
            if source_witness is not None:
                source_witness.check(episodes)
            for name, owner, proof, _ in checks:
                if report[name] is not owner:
                    raise ValueError(f"preceding {name} evidence was replaced")
                proof()
            if (
                report["collection"] is not collection
                or not len(collection_proofs) <= len(collection) <= 2
            ):
                raise ValueError("preceding original TRAIN collection changed or was replaced")
            for current, (owner, proof, _) in zip(collection, collection_proofs):
                if current is not owner:
                    raise ValueError("preceding original TRAIN collection record was replaced")
                proof()
            for current in collection[len(collection_proofs) :]:
                if (
                    current.get("status") != "complete"
                    or current.get("closed") is not True
                    or current.get("primary_error")
                    or current.get("cleanup_error")
                ):
                    raise ValueError("preceding original TRAIN collection is incomplete")
                proof, detached = paid_report_guard(current)
                collection_proofs.append((current, proof, detached))

        try:
            _require_minigrid()
            guard()
            with example_affine_training() as path:
                episodes = read_training_sources(path)
            source_witness, _ = _validate_sources(episodes, 1)
            attempted.append("collection")
            new_sources = _collect_recurring_train_sources(guard, collection)
            guard()
            if len(new_sources) != 2 or len(collection_proofs) != 2:
                raise ValueError("recurring TRAIN requires two complete original G sources")
            episodes = (*episodes, *new_sources)
            source_witness, _ = _validate_sources(episodes, 10)
            attempted.append("training")
            report["training"] = run_context_linear_stream(
                episodes,
                retain_arrived=True,
                recurring_catalogue=True,
                endpoint_observer=states.append,
                check_boundary=guard,
            )
            state = _require_training(report["training"], states, recurring_training=True)
            proof, detached = paid_report_guard(report["training"])
            checks.append(("training", report["training"], proof, detached))
            guard()
            limits = ToyExecutionBudget(
                max_training_updates=0,
                max_replay_examples=0,
                max_wall_seconds=150,
                max_process_rss_bytes=512 * 1024**2,
            )
            budget = ToyBudgetSession(limits, monotonic)
            budget.attach_memory(memory)
            binding, expected = budget_binding(budget), replace(limits)
            source = ContextLinearLearner()
            initial = source.snapshot_state()

            def readout_guard():
                guard()
                if (
                    budget_binding(budget) != binding
                    or not _same_value(vars(budget.budget), vars(expected))
                    or (budget.updates_completed, budget.replay_examples_completed) != (0, 0)
                    or source.snapshot_state() != initial
                ):
                    raise ValueError("recurring TRAIN readout source/clock/authority changed")

            attempted.append("games")
            try:
                readout_guard()
                report["games"] = evaluate_context_linear_stream_checkpoint(
                    source,
                    state,
                    budget=budget,
                    check_boundary=readout_guard,
                    recurring_catalogue=True,
                )
                readout_guard()
                games = report["games"]
                if (
                    games.get("status") != "complete"
                    or games.get("state_unchanged") is not True
                    or len(games.get("episodes", ())) != 4
                    or any(
                        e.get("status") != "complete"
                        or e.get("environment", {}).get("closed") is not True
                        for e in games["episodes"]
                    )
                ):
                    raise ValueError("recurring known G/returned A game report is incomplete")
                budget.before_final()
                report["status"] = "complete"
            except MiniGridEvaluationFailed as error:
                report["games"] = error.report
                raise
            finally:
                report["readout_budget"] = dict(
                    initial_limits=asdict(expected),
                    limits=asdict(budget.budget),
                    updates_completed=budget.updates_completed,
                    replay_examples_completed=budget.replay_examples_completed,
                    started_at=binding[2],
                    observed_budget_started_at=budget.started_at,
                    ended_at=monotonic(),
                    sampled_peak_rss_bytes=memory.peak_bytes,
                )
        except Exception as error:
            report.update(status="failed", reason=f"{type(error).__name__}: {error}"[:512])
        finally:
            report["protocol"] = original_protocol
            for name, _, _, detached in checks:
                report[name] = detached
            prefix = [detached for _, _, detached in collection_proofs]
            report["collection"] = prefix + collection[len(prefix) : 2]
            report["collection_capture_truncated"] = len(collection) > 2
            report["attempted_phases"] = attempted
            report["workflow_budget"] = dict(
                max_wall_seconds=300,
                work_deadline_seconds=290,
                max_process_rss_bytes=512 * 1024**2,
                started_at=started,
                ended_at=monotonic(),
                elapsed_seconds=monotonic() - started,
                sampled_peak_rss_bytes=memory.peak_bytes,
                scope="whole workflow supervision; original TRAIN/readout lifetime authority separate",
            )
    return report


def _play_private_endpoints(states, report, check_boundary, memory):
    from src.adapters.minigrid_policy_evaluation import evaluate_context_linear_rehearsal_checkpoint

    limits = ToyExecutionBudget(
        max_training_updates=0,
        max_replay_examples=0,
        max_wall_seconds=150,
        max_process_rss_bytes=512 * 1024**2,
    )
    budget = ToyBudgetSession(limits, monotonic)
    budget.attach_memory(memory)
    binding, expected = budget_binding(budget), replace(limits)
    source = ContextLinearLearner()
    initial = source.snapshot_state()
    games: list[dict[str, Any]] = []
    report["games"] = games
    proofs: list[tuple[dict, Any, dict]] = []
    pending = None

    def guard():
        check_boundary()
        if (
            budget_binding(budget) != binding
            or not _same_value(vars(budget.budget), vars(expected))
            or not _same_value((budget.updates_completed, budget.replay_examples_completed), (0, 0))
            or source.snapshot_state() != initial
        ):
            raise ValueError("private game source/readout clock or authority changed")
        if report["games"] is not games or len(games) != len(proofs):
            raise ValueError("preceding private game prefix changed or was replaced")
        for phase, (owner, proof, _) in zip(games, proofs):
            if phase is not owner:
                raise ValueError("preceding private game phase was replaced")
            proof()

    try:
        for arm, state in zip(("current_only", "joint_ABC_F"), states, strict=True):
            guard()
            try:
                phase = evaluate_context_linear_rehearsal_checkpoint(
                    source, state, arm=arm, budget=budget, check_boundary=guard
                )
            except MiniGridEvaluationFailed as error:
                pending = error.report
                raise
            pending = phase
            guard()
            if (
                phase.get("status") != "complete"
                or phase.get("state_unchanged") is not True
                or phase.get("rehearsal_arm") != arm
                or len(phase.get("episodes", ())) != 4
                or any(
                    e.get("status") != "complete"
                    or e.get("environment", {}).get("closed") is not True
                    for e in phase["episodes"]
                )
            ):
                raise ValueError("private F/returned-A game phase is incomplete")
            proof, detached = paid_report_guard(phase)
            games.append(phase)
            proofs.append((phase, proof, detached))
            pending = None
        guard()
        budget.before_final()
    except Exception:
        report["games"] = [detached for _, _, detached in proofs] + (
            [] if pending is None else [pending]
        )
        raise
    finally:
        report["readout_budget"] = dict(
            initial_limits=asdict(expected),
            limits=asdict(budget.budget),
            updates_completed=budget.updates_completed,
            replay_examples_completed=budget.replay_examples_completed,
            started_at=binding[2],
            observed_budget_started_at=budget.started_at,
            ended_at=monotonic(),
            sampled_peak_rss_bytes=memory.peak_bytes,
        )


def _run_private_adaptation():
    from scripts.run_minigrid_linear_fit_control import (
        read_training_sources,
        run_context_linear_stream,
    )
    from scripts.minigrid_affine_probe_games import _rehearsal_states
    from src.adapters import minigrid_affine_probe as original
    from src.adapters.minigrid_affine_rehearsal import compare_private_rehearsal
    from src.adapters.minigrid_context_linear_learner import copy_context_linear_training_snapshot
    from src.adapters.minigrid_train_catalogue import _validate_sources

    report: dict[str, Any] = dict(
        definition="portable_affine_private_adaptation_v1",
        protocol=describe_workflow(private_adaptation=True),
        status="running",
        reason=None,
        training=None,
        collection=[],
        probe_sources=None,
        private_comparison=None,
        games=None,
    )
    started = monotonic()
    states: list[Any] = []
    reference = expected_reference = None
    source_witness = None
    checks: list[tuple[str, dict, Any, dict]] = []
    attempted: list[str] = []
    collection = report["collection"]
    collection_proofs: list[tuple[dict, Any, dict]] = []
    protocol_check, original_protocol = paid_report_guard(report["protocol"])
    with ProcessRssSampler(interval_seconds=0.02) as memory:

        def guard():
            if monotonic() - started >= 170:
                raise TimeoutError("private adaptation workflow deadline reached; capture reserved")
            sampled = memory.sample()
            peak = memory.peak_bytes
            if sampled is None or peak is None or peak > 512 * 1024**2:
                raise RuntimeError("private adaptation workflow RSS unavailable or exceeds512MiB")
            if report["protocol"] is not protocol:
                raise ValueError("original workflow protocol was replaced")
            protocol_check()
            if source_witness is not None:
                source_witness.check(episodes)
            for name, owner, proof, _ in checks:
                if report[name] is not owner:
                    raise ValueError(f"preceding {name} evidence was replaced")
                proof()
            if reference is not None and reference != expected_reference:
                raise ValueError("current-run TRAIN reference changed during private work")
            if (
                report["collection"] is not collection
                or len(collection) > 4
                or len(collection) < len(collection_proofs)
            ):
                raise ValueError("preceding private collection changed or was replaced")
            for current, (owner, proof, _) in zip(collection, collection_proofs):
                if current is not owner:
                    raise ValueError("preceding closed collection record was replaced")
                proof()
            for current in collection[len(collection_proofs) :]:
                if current.get("status") != "complete" or current.get("closed") is not True:
                    raise ValueError("preceding private collection is incomplete")
                proof, detached = paid_report_guard(current)
                collection_proofs.append((current, proof, detached))

        def seal(name):
            owner = report[name]
            proof, detached = paid_report_guard(owner)
            checks.append((name, owner, proof, detached))

        protocol = report["protocol"]
        try:
            # The optional native runtime is checked before any TRAIN cost.
            _require_minigrid()
            guard()
            with example_affine_training() as path:
                episodes = read_training_sources(path)
            source_witness, _ = _validate_sources(episodes, 6)
            guard()
            attempted.append("training")
            report["training"] = run_context_linear_stream(
                episodes,
                retain_arrived=True,
                abc_catalogue=True,
                endpoint_observer=states.append,
                check_boundary=guard,
            )
            reference = _require_training(report["training"], states)
            expected_reference = copy_context_linear_training_snapshot(reference)
            seal("training")
            guard()
            attempted.append("collection")
            raw_train, raw_test = original.reserved._collect_probe_sources(
                guard, collection, distinct_probe=True
            )
            guard()
            if len(collection_proofs) != 4:
                raise ValueError("private adaptation requires four complete fresh teacher sources")
            # Validate the ORIGINAL16+2 release before fixed first2 selection.
            _, evidence = original._prepare_probe(raw_train, raw_test)
            report["probe_sources"] = dict(
                episodes=evidence,
                released_train_packets=16,
                released_test_packets=2,
                selected_train_packets=2,
                selected_test_packets=2,
            )
            selected = tuple(original.shared._episode(b) for b in (*raw_train[:2], *raw_test))
            seal("probe_sources")
            guard()
            attempted.append("private_comparison")
            report["private_comparison"] = compare_private_rehearsal(
                episodes, reference, selected, check_boundary=guard, fresh_reference=True
            )
            guard()
            # Ordinary conversion of this run's complete captured native states;
            # no historical source file, parameter reconstruction or refit.
            endpoints = _rehearsal_states(report["private_comparison"])
            seal("private_comparison")
            guard()
            attempted.append("games")
            _play_private_endpoints(endpoints, report, guard, memory)
            guard()
            report.update(status="complete", TRAIN_reference_unchanged=True)
        except Exception as error:
            report.update(status="failed", reason=f"{type(error).__name__}: {error}"[:512])
        finally:
            report["protocol"] = original_protocol
            for name, _, _, detached in checks:
                report[name] = detached
            # A failed/unreturned cleanup record is retained as an observed
            # suffix, never admitted as a preceding successful source.
            prefix = [detached for _, _, detached in collection_proofs]
            report["collection"] = prefix + collection[len(prefix) : 4]
            report["collection_capture_truncated"] = len(collection) > 4
            report["attempted_phases"] = attempted
            report["TRAIN_reference_unchanged"] = (
                None if expected_reference is None else reference == expected_reference
            )
            report["workflow_budget"] = dict(
                max_wall_seconds=180,
                work_deadline_seconds=170,
                max_process_rss_bytes=512 * 1024**2,
                started_at=started,
                ended_at=monotonic(),
                elapsed_seconds=monotonic() - started,
                sampled_peak_rss_bytes=memory.peak_bytes,
                scope="whole workflow supervision; existing TRAIN/private/readout authority remains separate",
            )
    return report
