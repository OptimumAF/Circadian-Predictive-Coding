"""Compose existing TRAIN stream and known-task games from one fresh endpoint.

Inputs are bundled TRAIN examples; output includes causal training receipts and
target-free game reports. This module owns no optimizer, evaluator or source
admission protocol. Historical scientific readback commands remain separate.
"""

from dataclasses import asdict, replace
from math import isfinite
from copy import deepcopy
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


def describe_workflow(
    *,
    private_adaptation=False,
    recurring_training=False,
    recurring_current_only=False,
    a4_adaptation=False,
    train_update=None,
    probe_update=None,
    all_old_tasks=False,
    joint_heldout=False,
):
    from scripts.run_minigrid_linear_fit_control import _train_update_protocol

    _train_update_protocol({}, train_update)
    _require_recurring_current_only(recurring_current_only, recurring_training, train_update)
    _require_joint_heldout(
        joint_heldout,
        private_adaptation,
        (
            recurring_training,
            a4_adaptation,
            train_update is not None,
            probe_update is not None,
            all_old_tasks,
        ),
    )
    _require_all_old_tasks(
        all_old_tasks,
        private_adaptation,
        recurring_training,
        a4_adaptation,
        train_update,
        probe_update,
    )
    _require_probe_update(
        probe_update, private_adaptation, recurring_training, a4_adaptation, train_update
    )
    if train_update is not None and (private_adaptation or a4_adaptation):
        raise ValueError("minimum-change TRAIN requires the ordinary fresh or recurring workflow")
    report = _describe_workflow(
        private_adaptation=private_adaptation,
        recurring_training=recurring_training,
        a4_adaptation=a4_adaptation,
        **(dict(recurring_current_only=True) if recurring_current_only else {}),
    )
    if probe_update is not None:
        from src.adapters.minigrid_affine_probe import describe_protocol

        # Why: F was observed to interfere with B/C; freeze all old tasks, not
        # only the retained A context, before any game outcomes are available.
        report.update(
            definition="portable_affine_isolated_adaptation_v1",
            probe_update=probe_update,
            private=describe_protocol(probe_update=probe_update),
            games="F blue/red probe_test then A/B/C_train blue/red TRAIN; seed17/max16; K7 firstargmax",
            new_fit_calls=8,
            new_fit_rows=126,
            qualification="disposable F child only; all old known tasks; no promotion/recovery/generalization/CPC claim",
        )
        report.pop("returned_layout")
    if all_old_tasks:
        report.update(definition="portable_affine_matched_all_old_v1", all_old_tasks=True)
        report["games"].update(
            definition="minigrid_private_rehearsal_F_then_all_ABC_games_v1",
            games_per_phase=8,
            contexts=("F-blue", "F-red", "A", "A", "B", "B", "C_train", "C_train"),
            roles=("probe_test", "probe_test", *(["train"] * 6)),
            max_steps_per_phase=128,
            qualification="both original matched cold arms; F and all old ABC known tasks; no relearning/generalization/CPC/equal-compute claim",
        )
        report.pop("returned_layout")
    if train_update is not None:
        report["training"] = _train_update_protocol(report["training"], train_update)
    if joint_heldout:
        from src.adapters.minigrid_affine_rehearsal import (
            describe_protocol as describe_joint_protocol,
        )

        report.update(
            definition="portable_affine_joint_heldout_v1",
            joint_heldout=True,
            private=describe_joint_protocol(fresh_reference=True, joint_only=True),
            new_fit_calls=8,
            new_fit_rows=180,
            games=dict(
                definition="joint8_180_terminal_D_v1",
                contexts=("D", "D"),
                cues=("blue", "red"),
                roles=("final_test", "final_test"),
                games_per_phase=2,
                seed=17,
                max_steps_per_game=16,
            ),
            qualification="fixed joint arm terminal D only; globally seen family, no tuning/promotion/statistical generalization/CPC claim",
        )
        report.pop("returned_layout", None)
    return _train_update_protocol(report, train_update)


def _require_joint_heldout(value, private, other_modes):
    if type(value) is not bool or value and (private is not True or any(other_modes)):
        raise ValueError("joint_heldout requires only ordinary private adaptation")


def _require_recurring_current_only(value, recurring, train_update=None):
    if type(value) is not bool or value and (recurring is not True or train_update is not None):
        raise ValueError("recurring_current_only requires only the cold recurring TRAIN workflow")


def _require_probe_update(value, private, recurring, a4, train):
    if value is None:
        return
    if type(value) is not str or value != "minimum_change":
        raise ValueError("probe_update must be omitted or exactly minimum_change")
    if private is not True or recurring is not False or a4 is not False or train is not None:
        raise ValueError("probe_update requires only the ordinary private adaptation workflow")


def _require_all_old_tasks(value, private, recurring, a4, train, probe):
    if type(value) is not bool:
        raise ValueError("all_old_tasks must be an exact bool")
    if value and (
        private is not True
        or recurring is not False
        or a4 is not False
        or train is not None
        or probe is not None
    ):
        raise ValueError("all_old_tasks requires only the ordinary cold private comparison")


def _describe_workflow(
    *,
    private_adaptation=False,
    recurring_training=False,
    a4_adaptation=False,
    recurring_current_only=False,
):
    from scripts.run_minigrid_linear_fit_control import describe_context_linear_abc

    _require_recurring_current_only(recurring_current_only, recurring_training)

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
        from scripts.run_minigrid_linear_fit_control import (
            describe_context_linear_recurring,
            describe_context_linear_recurring_current_only,
        )
        from src.core.minigrid_key_door import predefined_recurring_train_layout, predefined_layouts

        return dict(
            definition=(
                "portable_affine_recurring_current_only_v1"
                if recurring_current_only
                else "portable_affine_recurring_train_v1"
            ),
            training=(
                describe_context_linear_recurring_current_only()
                if recurring_current_only
                else describe_context_linear_recurring()
            ),
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


def _require_training(training, states, *, recurring_training=False, recurring_current_only=False):
    _require_recurring_current_only(recurring_current_only, recurring_training)
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
    if recurring_current_only:
        _require_current_repetition(training)
    return state


def _require_current_repetition(training):
    """Bind new mode to paid current-row reuse, never a same-shaped joint report."""
    from scripts.run_minigrid_linear_fit_control import (
        describe_context_linear_recurring_current_only,
    )

    expected = describe_context_linear_recurring_current_only()
    costs = training.get("current_repetition", {})
    counts = dict(
        arrival_row_exposures=90,
        repeated_current_row_exposures=162,
        fit_row_exposures=252,
        prior_packet_replay_rows=0,
        retained_bank_rows=0,
    )
    rows = dict(arrivals=90, repeated_current=162, native=252)
    if (
        training.get("definition") != expected["definition"]
        or not _same_value(training.get("protocol"), expected)
        or "catalogue" in training
        or type(costs) is not dict
        or any(type(costs.get(k)) is not int or costs[k] != v for k, v in counts.items())
        or not _same_value(costs.get("committed_rows"), rows)
        or not _same_value(costs.get("verified_rows"), rows)
        or not _same_value(costs.get("verified_receipts"), training["committed_receipts"])
        or costs.get("failure_work_may_be_unobserved") is not False
    ):
        raise ValueError(
            "recurring current-only games require original paid current-repeat evidence"
        )


def run_workflow(
    *,
    private_adaptation=False,
    recurring_training=False,
    recurring_current_only=False,
    a4_adaptation=False,
    train_update=None,
    probe_update=None,
    all_old_tasks=False,
    joint_heldout=False,
):
    from scripts.run_minigrid_linear_fit_control import _train_update_protocol

    _train_update_protocol({}, train_update)
    _require_recurring_current_only(recurring_current_only, recurring_training, train_update)
    _require_joint_heldout(
        joint_heldout,
        private_adaptation,
        (
            recurring_training,
            a4_adaptation,
            train_update is not None,
            probe_update is not None,
            all_old_tasks,
        ),
    )
    _require_all_old_tasks(
        all_old_tasks,
        private_adaptation,
        recurring_training,
        a4_adaptation,
        train_update,
        probe_update,
    )
    _require_probe_update(
        probe_update, private_adaptation, recurring_training, a4_adaptation, train_update
    )
    if train_update is not None and (private_adaptation or a4_adaptation):
        raise ValueError("minimum-change TRAIN requires the ordinary fresh or recurring workflow")
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
        return _run_recurring_training(
            **(dict(train_update=train_update) if train_update is not None else {}),
            **(dict(recurring_current_only=True) if recurring_current_only else {}),
        )
    if private_adaptation:
        return _run_private_adaptation(
            **(dict(probe_update=probe_update) if probe_update is not None else {}),
            **(dict(all_old_tasks=True) if all_old_tasks else {}),
            **(dict(joint_heldout=True) if joint_heldout else {}),
        )
    from scripts.run_minigrid_linear_fit_control import (
        read_training_sources,
        run_context_linear_stream,
    )

    report: dict[str, Any] = dict(
        definition="portable_affine_train_and_play_v1",
        protocol=describe_workflow(
            **(dict(train_update=train_update) if train_update is not None else {})
        ),
        status="running",
        reason=None,
        training=None,
        games=None,
    )
    protocol_pin = deepcopy(report["protocol"])
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
            episodes,
            retain_arrived=True,
            abc_catalogue=True,
            endpoint_observer=states.append,
            **(dict(train_update=train_update) if train_update is not None else {}),
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
                if train_update is not None and not _same_value(report["protocol"], protocol_pin):
                    raise ValueError("original workflow TRAIN update protocol changed")
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


def _run_recurring_training(*, train_update=None, recurring_current_only=False):
    from scripts.run_minigrid_linear_fit_control import (
        read_training_sources,
        run_context_linear_stream,
    )
    from src.adapters.minigrid_train_catalogue import _validate_sources

    protocol = describe_workflow(
        recurring_training=True,
        **(dict(recurring_current_only=True) if recurring_current_only else {}),
        **(dict(train_update=train_update) if train_update is not None else {}),
    )
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
            control_options: dict[str, Any] = (
                dict(recurring_current_only=True) if recurring_current_only else {}
            )
            report["training"] = run_context_linear_stream(
                episodes,
                retain_arrived=not recurring_current_only,
                recurring_catalogue=True,
                **control_options,
                endpoint_observer=states.append,
                check_boundary=guard,
                **(dict(train_update=train_update) if train_update is not None else {}),
            )
            state = _require_training(
                report["training"],
                states,
                recurring_training=True,
                **(dict(recurring_current_only=True) if recurring_current_only else {}),
            )
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


def _play_private_endpoints(
    states,
    report,
    check_boundary,
    memory,
    *,
    probe_update=None,
    all_old_tasks=False,
    joint_heldout=False,
):
    from src.adapters.minigrid_policy_evaluation import (
        evaluate_context_linear_rehearsal_checkpoint,
        evaluate_context_linear_probe_checkpoint,
        evaluate_heldout_context_linear_checkpoint,
    )

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
    pending_binding = None

    def guard():
        if pending_binding is not None:
            if pending is not pending_binding[0]:
                raise ValueError("returned final phase was replaced before qualification")
            pending_binding[1]()
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
        if pending_binding is not None:
            pending_binding[1]()

    try:
        arms = (
            ("joint_ABC_F",)
            if joint_heldout
            else ("minimum_change",)
            if probe_update is not None
            else ("current_only", "joint_ABC_F")
        )
        for arm, state in zip(arms, states, strict=True):
            guard()
            try:
                if joint_heldout:
                    phase = evaluate_heldout_context_linear_checkpoint(
                        source,
                        state,
                        budget=budget,
                        check_boundary=guard,
                        joint_private_endpoint=True,
                    )
                elif probe_update is not None:
                    phase = evaluate_context_linear_probe_checkpoint(
                        source,
                        state,
                        phase="after_probe",
                        include_old_tasks=True,
                        budget=budget,
                        check_boundary=guard,
                    )
                else:
                    phase = evaluate_context_linear_rehearsal_checkpoint(
                        source,
                        state,
                        arm=arm,
                        budget=budget,
                        check_boundary=guard,
                        **(dict(include_old_tasks=True) if all_old_tasks else {}),
                    )
            except MiniGridEvaluationFailed as error:
                pending = error.report
                raise
            pending = phase
            if joint_heldout:
                # Freeze the returned paid observation before boundary ports
                # can change a Boolean outcome while leaving its type valid.
                pending_proof, pending_copy = paid_report_guard(phase)
                pending_binding = (phase, pending_proof, pending_copy)
            guard()
            if joint_heldout:
                valid_mode = phase.get("joint_private_endpoint") is True
            elif probe_update is not None:
                valid_mode = (
                    phase.get("probe_phase") == "after_probe"
                    and phase.get("include_old_tasks") is True
                )
            else:
                valid_mode = phase.get("rehearsal_arm") == arm
            if (
                phase.get("status") != "complete"
                or phase.get("state_unchanged") is not True
                or not valid_mode
                or (all_old_tasks and phase.get("include_old_tasks") is not True)
                or len(phase.get("episodes", ()))
                != (2 if joint_heldout else 8 if probe_update is not None or all_old_tasks else 4)
                or any(
                    e.get("status") != "complete"
                    or e.get("environment", {}).get("closed") is not True
                    for e in phase["episodes"]
                )
            ):
                raise ValueError("private F/returned-A game phase is incomplete")
            if joint_heldout and (
                [(e.get("context"), e.get("cue"), e.get("source_role")) for e in phase["episodes"]]
                != [("D", cue, "final_test") for cue in ("blue", "red")]
                or phase.get("source_fit_calls") != 8
                or phase.get("source_fit_rows") != 180
                or any(
                    e.get("state_unchanged") is not True
                    or type(e.get("success")) is not bool
                    or type(e.get("return")) not in (int, float)
                    or not isfinite(e["return"])
                    for e in phase["episodes"]
                )
            ):
                raise ValueError(
                    "terminal joint D readout lost its paid mode, roster or outcome proof"
                )
            if (probe_update is not None or all_old_tasks) and [
                (e.get("context"), e.get("cue"), e.get("source_role")) for e in phase["episodes"]
            ] != [(f"F-{cue}", cue, "probe_test") for cue in ("blue", "red")] + [
                (context, cue, "train")
                for context in ("A", "B", "C_train")
                for cue in ("blue", "red")
            ]:
                raise ValueError("isolated F/all-old-ABC game roster changed")
            if joint_heldout:
                assert pending_binding is not None
                proof, detached = pending_binding[1], pending_binding[2]
            else:
                proof, detached = paid_report_guard(phase)
            games.append(phase)
            proofs.append((phase, proof, detached))
            pending = None
            pending_binding = None
        guard()
        budget.before_final()
    except Exception:
        if joint_heldout and (pending_binding is not None or proofs):
            report["games"] = [detached for _, _, detached in proofs] + (
                [pending_binding[2]] if pending_binding is not None else []
            )
            report["final_boundary_refused"] = True
        else:
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
        if joint_heldout and proofs:
            # Last clock capture above is also a boundary. Never publish a
            # changed paid outcome merely because it kept valid scalar types.
            try:
                for owner, proof, _ in proofs:
                    proof()
            except Exception:
                report["games"] = [detached for _, _, detached in proofs]
                report["final_boundary_refused"] = True
                raise


def _isolated_probe_state(report):
    """Admit this invocation's paid child; no historical source authority."""
    from scripts.run_minigrid_linear_fit_control import _context_linear_training_snapshot
    from src.adapters.minigrid_affine_probe import describe_protocol

    expected = asdict(
        ToyExecutionBudget(
            max_training_updates=2,
            max_replay_examples=0,
            max_wall_seconds=90,
            max_process_rss_bytes=512 * 1024**2,
        )
    )
    budget, probe = report.get("budget", {}), report.get("probe", {})
    if (
        report.get("status") != "complete"
        or report.get("protocol") != describe_protocol(probe_update="minimum_change")
        or report.get("probe_update") != "minimum_change"
        or report.get("source_native_state_unchanged") is not True
        or type(report.get("source_fork_calls")) is not int
        or report["source_fork_calls"] != 1
        or probe.get("status") != "complete"
        or (probe.get("completed_updates"), probe.get("completed_training_examples")) != (2, 18)
        or (budget.get("updates_completed"), budget.get("replay_examples_completed")) != (2, 0)
        or not _same_value(report.get("initial_probe_limits"), expected)
        or not _same_value(budget.get("limits"), expected)
        or not 0 <= budget.get("elapsed_seconds", -1) <= 90
        or budget.get("ended_at", -1) < budget.get("started_at", 0)
    ):
        raise ValueError("isolated games require this run's complete paid two-update F probe")
    state = _context_linear_training_snapshot(report["disposable_state"])
    if (
        (state.completed_updates, state.training_examples) != (8, 126)
        or state.failed_work is not None
        or any(head is None or head.fit_rows != 9 for head in state.heads)
    ):
        raise ValueError("isolated F endpoint requires actual8/126/two9-row heads")
    return state


def _run_private_adaptation(*, probe_update=None, all_old_tasks=False, joint_heldout=False):
    from scripts.run_minigrid_linear_fit_control import (
        read_training_sources,
        run_context_linear_stream,
    )
    from scripts.minigrid_affine_probe_games import _rehearsal_states
    from src.adapters import minigrid_affine_probe as original
    from src.adapters.minigrid_affine_rehearsal import compare_private_rehearsal
    from src.adapters.minigrid_context_linear_learner import copy_context_linear_training_snapshot
    from src.adapters.minigrid_train_catalogue import _validate_sources

    protocol = describe_workflow(
        private_adaptation=True,
        **(dict(probe_update=probe_update) if probe_update is not None else {}),
        **(dict(all_old_tasks=True) if all_old_tasks else {}),
        **(dict(joint_heldout=True) if joint_heldout else {}),
    )
    report: dict[str, Any] = dict(
        definition=protocol["definition"],
        protocol=protocol,
        status="running",
        reason=None,
        training=None,
        collection=[],
        probe_sources=None,
        private_comparison=None,
        games=None,
    )
    if probe_update is not None:
        report["private_probe"] = None
    started = monotonic()
    states: list[Any] = []
    reference = expected_reference = None
    heldout_binding = None
    source_witness = None
    checks: list[tuple[str, dict, Any, dict]] = []
    attempted: list[str] = []
    collection = report["collection"]
    collection_proofs: list[tuple[dict, Any, dict]] = []
    protocol_check, original_protocol = paid_report_guard(report["protocol"])
    with ProcessRssSampler(interval_seconds=0.02) as memory:

        def guard():
            if heldout_binding is not None and not _same_value(
                asdict(heldout_binding[0]), heldout_binding[1]
            ):
                raise ValueError(
                    "original paid joint endpoint changed before terminal final readout"
                )
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
            if heldout_binding is not None and not _same_value(
                asdict(heldout_binding[0]), heldout_binding[1]
            ):
                raise ValueError("paid joint endpoint changed during final boundary callbacks")

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
            if probe_update is not None:
                attempted.append("private_probe")
                report["private_probe"] = original.run_affine_fresh_probe(
                    episodes, reference, check_boundary=guard, probe_update=probe_update
                )
                # Seal even a paid failed probe before refusing its endpoint.
                seal("private_probe")
                guard()
                endpoints = (_isolated_probe_state(report["private_probe"]),)
            else:
                attempted.append("collection")
                raw_train, raw_test = original.reserved._collect_probe_sources(
                    guard, collection, distinct_probe=True
                )
                guard()
                if len(collection_proofs) != 4:
                    raise ValueError(
                        "private adaptation requires four complete fresh teacher sources"
                    )
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
                    episodes,
                    reference,
                    selected,
                    check_boundary=guard,
                    fresh_reference=True,
                    **(dict(joint_only=True) if joint_heldout else {}),
                )
                guard()
                # Convert actual current-run endpoints, without historical files/refit.
                if joint_heldout:
                    seal("private_comparison")
                endpoints = _rehearsal_states(
                    report["private_comparison"], **(dict(joint_only=True) if joint_heldout else {})
                )
                if joint_heldout:
                    heldout_binding = (
                        endpoints[0],
                        deepcopy(report["private_comparison"]["arms"][0]["native_endpoint"]),
                    )
                else:
                    seal("private_comparison")
            guard()
            attempted.append("games")
            _play_private_endpoints(
                endpoints,
                report,
                guard,
                memory,
                **(dict(probe_update=probe_update) if probe_update is not None else {}),
                **(dict(all_old_tasks=True) if all_old_tasks else {}),
                **(dict(joint_heldout=True) if joint_heldout else {}),
            )
            if joint_heldout:
                seal("games")
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
