"""Matched private context-head controls composed from the existing probe stream.

Certifies TRAIN-derived demonstrations and runs extra-F versus old-TRAIN replay.
No final data, benchmark updates, promotion, tuning, or new orchestration engine.
"""

from contextlib import nullcontext
from copy import deepcopy
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from scripts import context_head_probe as source
from scripts import probe_ticket_policy_learning as probe
from scripts.ticket_workflow_rehearsal import _ticket_trace
from src.adapters.context_perceptron import ContextPerceptronLearner
from src.app.coupled_attribution import _same_value, _validate_batch
from src.app.fresh_task_probe import run_fresh_task_probe
from src.app.probe_return_phase import paid_report_guard
from src.app.stream_rehearsal import RecentStreamReplayPolicy, StreamRehearsal
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    EpisodeReplaySchema,
)
from src.shared.process_memory import ProcessRssSampler

CONTROLS = ("extra_f", "old_train_replay")
SCHEMA = EpisodeReplaySchema(9, 3, 6)


def prepare_cohorts(spec):
    """Keep original roles/values, certifying complete tickets and old queues."""
    cohorts = source.build_head_cohorts(spec, head_task_transfer=True)
    tasks = (*probe.TASKS, *source.TRANSFER_TASKS)
    traces = {task: _ticket_trace(spec, *task) for task in tasks}
    class_ids = tuple(c.candidate_id for c in spec.catalog)

    def certify(batch, selected, identifier):
        observations = [o for task in selected for o in traces[task][0]]
        actions = [class_ids.index(a) for task in selected for a in traces[task][1]]
        if (
            batch.episode is not None
            or batch.num_classes != 6
            or not np.array_equal(batch.features, np.array(observations))
            or not np.array_equal(batch.targets, np.array(actions).reshape(-1, 1))
        ):
            raise ValueError("context probe packet differs from its original teacher")
        steps = len(observations)
        transitions = tuple(
            DemonstrationTransition(
                i,
                observations[i],
                actions[i],
                float(i % 3 == 2),
                observations[i + 1] if i < steps - 1 else (0.0,) * 9,
                i == steps - 1,
                6,
            )
            for i in range(steps)
        )
        episode = DemonstrationEpisode(identifier, batch.context, batch.role, transitions, 6)
        return replace(
            batch, episode=episode, row_ids=tuple(f"{identifier}:step:{i}" for i in range(steps))
        )

    prepared = []
    seen: set[str] = set()
    for group_index, group in enumerate(cohorts):
        rows = []
        for i, batch in enumerate(group):
            expected_role = ("probe_train", "probe_test", "train", "train")[group_index]
            _validate_batch(batch, expected_role, seen)
            expected_context = (
                f"{source.TRANSFER_TASKS[i % 2][1]}-{source.TRANSFER_TASKS[i % 2][0]}"
                if group_index < 2
                else f"old-v{i + 1}"
            )
            if batch.context != expected_context:
                raise ValueError("context probe packet has a foreign task context")
            selected = (
                (source.TRANSFER_TASKS[i % 2],)
                if group_index < 2
                else probe.TASKS[i * 2 : i * 2 + 2]
            )
            rows.append(certify(batch, selected, f"matched-{group_index}-{i}"))
        prepared.append(tuple(rows))
    bank = []
    for i, task in enumerate(probe.TASKS):
        parent = cohorts[2][i // 2]
        start = 3 * (i % 2)
        packet = replace(
            parent,
            context=f"{task[1]}-{task[0]}",
            features=parent.features[start : start + 3].copy(),
            targets=parent.targets[start : start + 3].copy(),
        )
        bank.append(certify(packet, (task,), f"fixed-old-{i}").episode)
    return tuple(prepared), tuple(bank)


def _limits(updates=None, replay=None):
    return ToyExecutionBudget(
        max_training_updates=updates,
        max_replay_examples=replay,
        max_wall_seconds=30,
        max_process_rss_bytes=512 * 1024**2,
    )


def joint_probe_factory(cohorts, bank, control, replay, guard):
    """Bind four original TRAIN episodes; only the current probe packet is transient."""
    from src.core.context_joint_replay import JointContextReplayPolicy
    from src.core.demonstration_episode import episode_arrays

    if type(control) is not str or control not in ("joint", "current_only"):
        raise ValueError("unknown isolated joint control")
    if type(bank) is not tuple or len(bank) != 4:
        raise ValueError("joint probe requires its four original TRAIN episodes")
    for i, (episode, task) in enumerate(zip(bank, probe.TASKS, strict=True)):
        if type(episode) is not DemonstrationEpisode:
            raise ValueError("joint probe requires original TRAIN episode records")
        x, y = episode_arrays(episode)
        original = cohorts[2][i // 2]
        start = 3 * (i % 2)
        if (
            episode.role != "train"
            or episode.context != f"{task[1]}-{task[0]}"
            or not np.array_equal(x, original.features[start : start + 3])
            or not np.array_equal(y, original.targets[start : start + 3])
        ):
            raise ValueError("joint bank differs from admitted old TRAIN rows")
    pinned, owners = deepcopy(bank), tuple(id(e) for e in bank)

    def boundary():
        guard()
        if tuple(id(e) for e in bank) != owners or not _same_value(bank, pinned):
            raise ValueError("original fixed TRAIN bank changed during joint probe")

    def factory(learner, budget, count):
        boundary()
        return StreamRehearsal(
            learner,
            budget,
            RecentStreamReplayPolicy(1, 12, ReplayRetentionBudget(12, 8192)),
            replay_budget=replay,
            total_updates=count,
            input_dim=9,
            episode_schema=SCHEMA,
            fixed_training_episodes=bank,
            wake_role="probe_train",
            joint_context=JointContextReplayPolicy(9, 6, (4, 5), control),
            check_boundary=boundary,
        )

    boundary()
    return boundary, factory


def require_joint_probe(row, control):
    """Reconcile physical joint work rather than interpreting logical F as native fits."""
    if type(control) is not str or control not in ("joint", "current_only"):
        raise ValueError("unknown isolated joint control")
    session = row["joint_replay_session"]
    helper = row["probe"]["rehearsal"]
    if (
        row.get("joint_rehearsal_control") != control
        or not _same_value(session["limits"], asdict(_limits(replay=144)))
        or type(session["updates"]) is not int
        or session["updates"] != 0
        or type(session["replay"]) is not int
        or session["replay"] != 144
        or type(session["started_at"]) is not float
        or type(session["last_clock"]) is not float
        or not 0 <= session["last_clock"] - session["started_at"] <= 30
        or helper["fixed_bank"]["native_transition_count"] != 12
        or helper["joint_context"]["policy"]["control"] != control
        or any(
            type(helper[k]) is not int or helper[k] != n
            for k, n in (
                ("attempted_updates", 16),
                ("returned_updates", 16),
                ("committed_updates", 16),
                ("charged_exposures", 144),
                ("replay_budget_charged_exposures", 144),
            )
        )
        or len(helper["events"]) != 16
    ):
        raise ValueError("joint probe paid sessions/native history differs")
    for event in helper["events"]:
        detail = event["joint_context"]
        if (
            event["status"] != "complete"
            or any(
                type(event[k]) is not int or event[k] != n
                for k, n in (
                    ("attempted_updates", 1),
                    ("returned_updates", 1),
                    ("committed_updates", 1),
                    ("charged_exposures", 9),
                )
            )
            or detail["control"] != control
            or detail["current_role"] != "probe_train"
            or detail["retained_role"] != "train"
            or detail["current_packet_rows"] != 3
            or detail["retained_rows_available"] != 6
            or detail["total_fit_rows"] != 9
            or detail["retained_train_rows_used"] != (6 if control == "joint" else 0)
            or detail["current_rows_used"] != (3 if control == "joint" else 9)
            or len(event["selected_episodes"]) != 2
            or any(e["role"] != "train" for e in event["selected_episodes"])
        ):
            raise ValueError("joint probe event roles/exposures differ")


def run_arm(seed, state, cohorts, bank, spec, guard, memory, control, row):
    """Pay original sessions once; admit no arm after failed preceding evidence."""
    if type(control) is not str or control not in CONTROLS:
        raise ValueError("unknown fixed matched context control")
    extra = control == "extra_f"
    sessions = [
        ToyBudgetSession(_limits(48 if extra else 16), monotonic),
        ToyBudgetSession(_limits(16), monotonic),
    ]
    replay = None if extra else ToyBudgetSession(_limits(replay=96), monotonic)
    if replay is not None:
        sessions.append(replay)
    if memory is not None:
        for session in sessions:
            session.attach_memory(memory)
    row.update(
        mode="warm",
        control=control,
        prior_native_updates=state.completed_updates,
        prior_native_rows=state.training_examples,
        episodes=[],
        readout_work=[],
    )

    def factory(learner, budget, count):
        if replay is None:
            raise ValueError("extra F cannot construct a replay helper")
        return StreamRehearsal(
            learner,
            budget,
            RecentStreamReplayPolicy(2, 12, ReplayRetentionBudget(12, 8192)),
            replay_budget=replay,
            total_updates=count,
            input_dim=9,
            episode_schema=SCHEMA,
            fixed_training_episodes=bank,
            wake_role="probe_train",
            check_boundary=guard,
        )

    try:
        probe._observe_initial_tasks(seed, sessions, row, spec, guard, probe.TRANSFER_HEAD_TASKS)
        training, evaluation, monitors, returning = cohorts
        row["probe"] = run_fresh_task_probe(
            seed,
            state,
            training,
            evaluation,
            budget=sessions[0],
            checkpoint_updates=(0, 12, 24, 48) if extra else probe.CHECKPOINTS,
            training_readouts=monitors,
            return_training=returning,
            return_budget=sessions[1],
            decision_diagnostics=True,
            input_diagnostics=True,
            check_boundary=guard,
            **(dict(extra_f_exposure=True) if extra else dict(rehearsal_factory=factory)),
            endpoint_observer=lambda learner, phase, boundary: probe._observe_tasks(
                learner,
                phase,
                boundary,
                row,
                spec,
                guard,
                separate_context_heads=True,
                head_tasks=probe.TRANSFER_HEAD_TASKS,
                matched_head_exposure=True,
            ),
        )
    finally:
        row["sessions"] = [
            dict(
                limits=asdict(s.budget),
                updates=s.updates_completed,
                replay=s.replay_examples_completed,
                started_at=s.started_at,
                last_clock=s.last_clock,
            )
            for s in sessions
        ]
    guard()
    require_arm(row)


def require_arm(row):
    """Validate paid clocks, replay charges and all fixed ordinary task slots."""
    control = row["control"]
    if type(control) is not str or control not in CONTROLS or row["mode"] != "warm":
        raise ValueError("original matched control differs")
    extra = control == "extra_f"
    f = row["probe"]
    for phase, result in (("F", f), ("return", f.get("return_phase", {}))):
        if result.get("status") != "complete":
            raise ValueError(f"matched {phase} incomplete: {result.get('reason')}")
    if any(
        type(row[k]) is not int or row[k] != n
        for k, n in (("prior_native_updates", 12), ("prior_native_rows", 72))
    ):
        raise ValueError("original warm source clocks differ")
    expected_sessions: list[tuple[int | None, int | None, int, int]] = [
        (48 if extra else 16, None, 48 if extra else 16, 0),
        (16, None, 16, 0),
    ]
    if not extra:
        expected_sessions.append((None, 96, 0, 96))
    if len(row["sessions"]) != len(expected_sessions):
        raise ValueError("original matched sessions missing")
    for s, (updates, replay, paid_updates, paid_replay) in zip(
        row["sessions"], expected_sessions, strict=True
    ):
        if (
            not _same_value(s["limits"], asdict(_limits(updates, replay)))
            or type(s["updates"]) is not int
            or s["updates"] != paid_updates
            or type(s["replay"]) is not int
            or s["replay"] != paid_replay
            or type(s["started_at"]) is not float
            or type(s["last_clock"]) is not float
            or not 0 <= s["last_clock"] - s["started_at"] <= 30
        ):
            raise ValueError("matched paid session limits/clocks differ")
    for result, count, rows, points in (
        (f, 48 if extra else 16, 144 if extra else 48, (0, 12, 24, 48) if extra else (0, 4, 8, 16)),
        (f.get("return_phase", {}), 16, 96, (0, 4, 8, 16)),
    ):
        if (
            result.get("status") != "complete"
            or any(
                type(result.get(k)) is not int or result[k] != v
                for k, v in (
                    ("attempted_updates", count),
                    ("completed_updates", count),
                    ("attempted_training_examples", rows),
                    ("completed_training_examples", rows),
                )
            )
            or [p["updates"] for p in result["checkpoints"]] != list(points)
            or result["training_readouts"]["status"] != "complete"
            or len(result["training_readouts"]["checkpoints"]) != 4
        ):
            raise ValueError("matched paid probe/return differs")
    if extra:
        if "rehearsal" in f:
            raise ValueError("extra F control received replay")
    else:
        replay = f["rehearsal"]
        for k, n in (
            ("charged_exposures", 96),
            ("attempted_updates", 32),
            ("returned_updates", 32),
            ("committed_updates", 32),
        ):
            if type(replay[k]) is not int or replay[k] != n:
                raise ValueError("matched TRAIN replay charges differ")
        if len(replay["events"]) != 8 or replay["wake_role"] != "probe_train":
            raise ValueError("matched replay schedule differs")
    phases = ("before_probe", "after_probe", "after_return")
    if [(e["phase"], e["policy_version"], e["concept"]) for e in row["episodes"]] != [
        (phase, v, c) for phase in phases for v, c in probe.TRANSFER_HEAD_TASKS
    ] or any(
        e["role"] != "probe_test" or e["episode"]["complete"] is not True for e in row["episodes"]
    ):
        raise ValueError("matched ordinary task slots missing/unknown")
    if len(row["readout_work"]) != 3:
        raise ValueError("matched endpoint work missing")
    for reading, phase, updates, rows in zip(
        row["readout_work"], phases, (12, 60, 76), (72, 216, 312), strict=True
    ):
        work = reading["work"]
        if (
            reading["phase"] != phase
            or any(
                type(reading[k]) is not int or reading[k] != n
                for k, n in (("native_updates", updates), ("native_rows", rows))
            )
            or any(
                type(work[k]) is not int
                for k in ("attempted_calls", "returned_calls", "validated_calls")
            )
            or not 6
            <= work["attempted_calls"]
            == work["returned_calls"]
            == work["validated_calls"]
            <= 36
        ):
            raise ValueError("matched native endpoint clocks/work differ")


def _pair_report(definition):
    return dict(
        definition=definition,
        complete=False,
        cases=[],
        training_work=dict(
            attempted_updates=0, attempted_rows=0, returned_updates=0, unreturned_work_unknown=False
        ),
        final_examples=0,
        promotions=0,
        source_prior_training=dict(updates=12, rows=72, mistakes=16),
    )


def _run_pair(state, spec, report, started, check_source, memory=None):
    """Shared original pair, with origin-specific source checks at every boundary."""
    try:
        check_source()
        cohorts, bank = prepare_cohorts(spec)
        pin = deepcopy((state, cohorts, bank, np.random.get_state()))
        seeds: list[Any] = []
        paid: list[Any] = []
        cases, work = report["cases"], report["training_work"]

        def guard():
            if monotonic() - started > 120:
                raise ValueError("matched case deadline exhausted")
            check_source()
            if not _same_value((state, cohorts, bank, np.random.get_state()), pin):
                raise ValueError("matched source/cohorts/bank/RNG changed")
            if any(not _same_value(seed.snapshot_state(), state) for seed in seeds):
                raise ValueError("matched original seed changed")
            if report["cases"] is not cases:
                report["cases"] = cases
                raise ValueError("matched paid cases replaced")
            for i, (row, paid_pin, check) in enumerate(paid):
                changed = cases[i] is not row
                try:
                    check()
                except ValueError:
                    changed = True
                if changed:
                    row.clear()
                    row.update(deepcopy(paid_pin))
                    cases[i] = row
                    raise ValueError("preceding matched evidence changed")

        class CountedLearner(ContextPerceptronLearner):
            def fork(self):
                child = CountedLearner(self.policy)
                child.restore_state(self.snapshot_state())
                return child

            def train_batch(self, features, targets):
                guard()
                work["attempted_updates"] += 1
                work["attempted_rows"] += len(features)
                if work["attempted_updates"] > 128 or work["attempted_rows"] > 480:
                    raise ValueError("matched native allowance exhausted")
                result = super().train_batch(features, targets)
                work["returned_updates"] += 1
                return result

        with (
            ProcessRssSampler(interval_seconds=0.02) if memory is None else nullcontext(memory)
        ) as memory:
            for control in CONTROLS:
                guard()
                seed = CountedLearner(state.policy)
                seed.restore_state(state)
                seeds.append(seed)
                row: dict[str, Any] = {}
                cases.append(row)
                run_arm(seed, state, cohorts, bank, spec, guard, memory, control, row)
                require_arm(row)
                guard()
                check, paid_pin = paid_report_guard(row)
                paid.append((row, paid_pin, check))
            guard()
            report["sampled_process_rss"] = asdict(memory.snapshot())
        report.update(
            complete=True,
            source_owners_unchanged=True,
            numpy_rng_unchanged=True,
            model_coefficients=96,
            model_coefficient_bytes=768,
            fixed_bank_array_bytes=4 * 8 * (SCHEMA.encoded_dim + 1),
        )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"[:1024]
    report["training_work"]["unreturned_work_unknown"] = (
        report["training_work"]["attempted_updates"] != report["training_work"]["returned_updates"]
    )
    report["elapsed_seconds"] = monotonic() - started
    return report


def run_case(input_path):
    """Keep original historical source bytes/SHA and finite pair authority."""
    started = monotonic()
    report = _pair_report("matched_context_head_rehearsal_v1")
    report["source_sha256"] = source.HEAD_SOURCE_SHA
    try:
        data = source._source_bytes(input_path)
        if hashlib.sha256(data).hexdigest() != source.HEAD_SOURCE_SHA:
            raise ValueError("qualified contextual source replaced")
        state = source.head_states(json.loads(data))[0]
        spec = source.DecisionFeatureSpec(source.CONTEXT_FEATURES, source.CATALOG)

        def check_source():
            if source._source_bytes(input_path) != data:
                raise ValueError("matched source bytes changed")

        _run_pair(state, spec, report, started, check_source)
        if report["complete"]:
            report["source_bytes_unchanged"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"[:1024]
        report["elapsed_seconds"] = monotonic() - started
    return report


def prepare_owned_source(
    state,
    spec,
    source_report,
    source_budget,
    *,
    check_boundary,
    training_key="private_source_training",
    linear_fit=False,
    catalog_study=False,
    reference_validation=False,
    retained_probe=False,
):
    """Share the existing live source qualification; grant no new training authority."""
    from threading import RLock
    from src.app.probe_endpoint_observer import _BudgetWitness
    from src.app.probe_return_phase import budget_binding
    from src.core.context_perceptron import copy_context_snapshot, ContextPerceptronPolicy

    if (
        type(catalog_study) is not bool
        or type(retained_probe) is not bool
        or retained_probe
        and (not linear_fit or catalog_study or reference_validation)
        or type(reference_validation) is not bool
        or reference_validation
        and (not linear_fit or catalog_study)
        or catalog_study
        and not linear_fit
        or type(linear_fit) is not bool
        or linear_fit
        and training_key != "serving_source_training"
        or not callable(check_boundary)
        or type(source_budget) is not ToyBudgetSession
        or training_key not in ("private_source_training", "serving_source_training")
        or type(spec) is not source.DecisionFeatureSpec
        or spec != source.DecisionFeatureSpec(source.CONTEXT_FEATURES, source.CATALOG)
    ):
        raise ValueError("owned source requires its original schema, guard and budget")
    definition = "separate_context_head_ticket_workflow_v1"
    if linear_fit:
        from src.core.context_linear_state import ContextLinearFitPolicy, copy_context_linear_state

        state = copy_context_linear_state(state)
        definition = "separate_context_linear_ticket_workflow_v1"
        valid = state.policy == ContextLinearFitPolicy(9, 6, (4, 5)) and (
            state.completed_updates,
            state.training_examples,
        ) == (12, 72)
    else:
        state = copy_context_snapshot(state)
        valid = state.policy == ContextPerceptronPolicy(9, 6, (4, 5)) and (
            state.completed_updates,
            state.training_examples,
            state.mistake_updates,
        ) == (12, 72, 16)
    if not valid or [(h.completed_updates, h.training_examples) for h in state.heads] != [
        (8, 48),
        (4, 24),
    ]:
        raise ValueError("owned source requires original paired TRAIN checkpoint12")
    limits = _limits(12)
    if catalog_study or reference_validation:
        from dataclasses import replace

        # Why: born once for the declared study/validation, never extend an
        # already running source or change private-probe/fixed-trial limits.
        limits = replace(limits, max_wall_seconds=240)
    elif retained_probe:
        from dataclasses import replace

        # Why: this explicitly born combined workflow includes retained learning
        # then the original probe/return clocks. Never extend an existing source.
        limits = replace(limits, max_wall_seconds=120)
    witness = _BudgetWitness(source_budget, budget_binding(source_budget), limits, 12)
    source_pin = deepcopy((source_report.get(training_key), source_report.get("summary")))
    keys = (training_key, "summary", "checkpoints", "episodes")
    owners = tuple(source_report.get(k) for k in keys)
    row_owners = tuple(tuple(id(row) for row in source_report[k]) for k in keys[-2:])
    lock = RLock()

    def check_source():
        # The serving worker and owner observe the SAME original source session.
        # Serialize witness reads/updates; never bless an outside clock change.
        with lock:
            if any(
                source_report.get(k) is not owner for k, owner in zip(keys, owners, strict=True)
            ) or any(
                tuple(id(row) for row in source_report[k]) != rows
                for k, rows in zip(keys[-2:], row_owners, strict=True)
            ):
                raise ValueError("original owned source evidence was replaced")
            witness.observe()
            if check_boundary() is not None:
                raise ValueError("source guard must return None")
            witness.check()
            if (
                source_report.get("definition") != definition
                or source_report.get("source_unchanged") is not True
                or source_report.get("actor_unchanged") is not True
                or not _same_value(
                    (source_report.get(training_key), source_report.get("summary")),
                    source_pin,
                )
                or source_report[training_key]
                != dict(completed_updates=12, applied_update_count=12)
                or [p["source_updates"] for p in source_report["checkpoints"]] != [0, 4, 8, 12]
                or not _same_value(source_report["checkpoints"][-1]["state"], asdict(state))
            ):
                raise ValueError("current source qualification changed or incomplete")
            ordinary = [
                e
                for e in source_report["episodes"]
                if e["source_updates"] == 12 and e["episode"]["scenario"] == "ordinary"
            ]
            if [(e["policy_version"], e["concept"]) for e in ordinary] != list(probe.TASKS) or any(
                e["role"] != "outer_selection"
                or e["episode"].get("complete") is not True
                or e["episode"].get("success") is not True
                for e in ordinary
            ):
                raise ValueError("owned adaptation requires all four qualified source tasks")
            witness.check()

    check_source()
    return state, check_source


def run_owned_pair(state, spec, source_report, source_budget, *, check_boundary, memory):
    """Use the current workflow's qualified state; no historical file or SHA claim."""
    started = monotonic()
    report = _pair_report("fresh_context_head_private_adaptation_v1")
    report["source_provenance"] = "current_workflow_train_checkpoint12"
    try:
        state, check_source = prepare_owned_source(
            state,
            spec,
            source_report,
            source_budget,
            check_boundary=check_boundary,
        )
        _run_pair(state, spec, report, started, check_source, memory)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"[:1024]
        report["elapsed_seconds"] = monotonic() - started
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        raise ValueError("qualified source must stay sealed")
    if args.output.exists():
        raise FileExistsError("matched output must be new")
    if args.preview:
        spec = source.DecisionFeatureSpec(source.CONTEXT_FEATURES, source.CATALOG)
        cohorts, bank = prepare_cohorts(spec)
        report = dict(
            definition="matched_context_head_rehearsal_preview_v1",
            complete=True,
            source_read=False,
            model_constructed=False,
            source_sha256=source.HEAD_SOURCE_SHA,
            controls=CONTROLS,
            tasks=probe.TRANSFER_HEAD_TASKS,
            probe_packets=len(cohorts[0]),
            fixed_train_episodes=len(bank),
            bank_array_bytes=4 * 8 * (SCHEMA.encoded_dim + 1),
            per_arm_native_updates=64,
            per_arm_rows=240,
            f_native_checkpoints=(0, 12, 24, 48),
            f_rows=(0, 36, 72, 144),
            scope="same-template TRAIN-derived diagnostics; native passes/rows/head exposure/capacity match, not all-in compute or a CPC/heldout claim",
        )
    else:
        report = run_case(args.input)
    from scripts.context_catalog_learning import write_catalog_report

    return (
        0
        if write_catalog_report(
            args.output,
            report,
            observed_work=report.get("training_work", {}),
            max_bytes=4 * 1024**2,
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
