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


def run_owned_pair(state, spec, source_report, source_budget, *, check_boundary, memory):
    """Use the current workflow's qualified state; no historical file or SHA claim."""
    from src.app.probe_endpoint_observer import _BudgetWitness
    from src.app.probe_return_phase import budget_binding
    from src.core.context_perceptron import copy_context_snapshot, ContextPerceptronPolicy

    started = monotonic()
    report = _pair_report("fresh_context_head_private_adaptation_v1")
    report["source_provenance"] = "current_workflow_train_checkpoint12"
    try:
        if not callable(check_boundary) or type(source_budget) is not ToyBudgetSession:
            raise ValueError("owned source requires its original guard and budget")
        state = copy_context_snapshot(state)
        if (
            state.policy != ContextPerceptronPolicy(9, 6, (4, 5))
            or (state.completed_updates, state.training_examples, state.mistake_updates)
            != (12, 72, 16)
            or [(h.completed_updates, h.training_examples) for h in state.heads]
            != [(8, 48), (4, 24)]
        ):
            raise ValueError("owned source requires original paired TRAIN checkpoint12")
        witness = _BudgetWitness(source_budget, budget_binding(source_budget), _limits(12), 12)
        source_pin = deepcopy(
            (source_report.get("private_source_training"), source_report.get("summary"))
        )

        def check_source():
            witness.observe()
            if check_boundary() is not None:
                raise ValueError("source guard must return None")
            witness.check()
            if (
                source_report.get("definition") != "separate_context_head_ticket_workflow_v1"
                or source_report.get("source_unchanged") is not True
                or source_report.get("actor_unchanged") is not True
                or not _same_value(
                    (source_report.get("private_source_training"), source_report.get("summary")),
                    source_pin,
                )
                or source_report["private_source_training"]
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
                raise ValueError("private adaptation requires all four qualified source tasks")
            witness.check()

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
