"""Fixed TRAIN-derived warm/fresh diagnostic on disposable native BP forks.

Composes the existing probe, old-TRAIN return and typed sandbox readouts. This
recipe grants no benchmark updates, promotion, final-data or unseen-task claim.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from scripts import ticket_workflow_native as native
from scripts.diagnose_ticket_training import _teacher, POLICY
from scripts.run_bp_new_contexts import _source, SOURCE_SHA
from scripts.run_ticket_workflow import FEATURES, _encode_report
from src.adapters.torch_bp_record import bp_snapshot_from_record
from src.adapters.context_perceptron import ContextPerceptronDecisionPredictor
from src.app.coupled_attribution import AttributionBatch, _same_value
from src.app.fresh_task_probe import run_fresh_task_probe
from src.app.ticket_workflow import run_workflow_episode
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.decision_features import DecisionFeatureSpec
from src.core.ticket_workflow import ACTION_IDS, CATALOG, WorkflowCapabilities, WorkflowState
from src.shared.process_memory import ProcessRssSampler

CHECKPOINTS = (0, 4, 8, 16)
TASKS = (("v1", "invoice"), ("v1", "password"), ("v2", "invoice"), ("v2", "parcel"))
LIMITS = ToyExecutionBudget(
    max_training_updates=16, max_wall_seconds=120, max_process_rss_bytes=512 * 1024**2
)
HEAD_LIMITS = ToyExecutionBudget(
    max_training_updates=16, max_wall_seconds=30, max_process_rss_bytes=512 * 1024**2
)
HEAD_TASKS = (*TASKS, ("v1", "legacy"), ("v2", "legacy"))


def build_cohorts(spec):
    """Detached diagnostic copies of known TRAIN templates, not held-out data."""
    _, fresh = _teacher(spec, policy_change=True)
    _, old = _teacher(spec)

    def batch(base, index, role, context, identifier):
        start = 3 * index
        return AttributionBatch(
            role,
            context,
            tuple(f"{identifier}-{i}" for i in range(3)),
            base.features[start : start + 3].copy(),
            base.targets[start : start + 3].copy(),
            num_classes=6,
        )

    training = tuple(
        batch(fresh, i % 2, "probe_train", ("new-invoice", "new-password")[i % 2], f"probe-{i}")
        for i in range(16)
    )
    evaluation = tuple(
        batch(fresh, i, "probe_test", context, f"read-new-{i}")
        for i, context in enumerate(("new-invoice", "new-password"))
    )
    monitors = tuple(
        batch(old, i, "train", context, f"read-old-{i}")
        for i, context in enumerate(("old-invoice", "old-parcel"))
    )
    returning = tuple(
        batch(old, i, "train", context, f"return-old-{i}")
        for i, context in enumerate(("old-invoice", "old-parcel"))
    )
    return training, evaluation, monitors, returning


def _require_source_records(source):
    points = source["checkpoints"]
    if len(points) != 2 or [r["source_updates"] for r in points] != [0, 96]:
        raise ValueError("original qualified source endpoints0/96 required")
    for point in points:
        n, state = point["source_updates"], point["state"]
        if (
            type(n) is not int
            or state["policy"] != POLICY
            or any(type(state["policy"][k]) is not type(v) for k, v in POLICY.items())
            or type(state["completed_updates"]) is not int
            or state["completed_updates"] != n
            or type(state["training_examples"]) is not int
            or state["training_examples"] != 6 * n
            or point["source_rows"] != 6 * n
        ):
            raise ValueError("original qualified native policy/clocks required")


def _require_arm(row, *, separate_context_heads=False):
    """Do not buy another arm after incomplete or replaced preceding evidence."""
    if len(row["sessions"]) != 2 or any(
        s["limits"] != asdict(HEAD_LIMITS if separate_context_heads else LIMITS)
        or type(s["updates"]) is not int
        or s["updates"] != 16
        or type(s["started_at"]) is not float
        or type(s["last_clock"]) is not float
        or s["last_clock"] < s["started_at"]
        for s in row["sessions"]
    ):
        raise ValueError("original paid session limits/clocks differ")
    probe = row["probe"]
    returning = probe.get("return_phase", {})
    if probe.get("status") != "complete" or returning.get("status") != "complete":
        raise ValueError("preceding probe/return incomplete; further arms closed")
    for index, result in enumerate((probe, returning)):
        rows = 96 if separate_context_heads and index == 1 else 48
        if any(
            type(result.get(k)) is not int or result[k] != v
            for k, v in (
                ("attempted_updates", 16),
                ("completed_updates", 16),
                ("attempted_training_examples", rows),
                ("completed_training_examples", rows),
            )
        ) or [p["updates"] for p in result["checkpoints"]] != list(CHECKPOINTS):
            raise ValueError("preceding paid probe history differs")
        monitors = result["training_readouts"]
        if monitors["status"] != "complete" or len(monitors["checkpoints"]) != 4:
            raise ValueError("preceding old TRAIN readouts incomplete")
    expected = [
        (phase, version, concept)
        for phase in ("after_probe", "after_return")
        for version, concept in (HEAD_TASKS if separate_context_heads else TASKS)
    ]
    if [(e["phase"], e["policy_version"], e["concept"]) for e in row["episodes"]] != expected:
        raise ValueError("preceding endpoint tasks missing/mixed")
    if any(
        e["role"] != "probe_test" or e["episode"]["complete"] is not True for e in row["episodes"]
    ):
        raise ValueError("preceding endpoint task work unknown")


def _observe_tasks(learner, phase, boundary, row, spec, guard, *, separate_context_heads=False):
    boundary()
    state = learner.snapshot_state()
    pinned = deepcopy(state)
    expected = 16 if phase == "after_probe" else 32
    rows = (48 if phase == "after_probe" else 144) if separate_context_heads else 3 * expected
    if (
        state.completed_updates != row["prior_native_updates"] + expected
        or state.training_examples != row["prior_native_rows"] + rows
    ):
        raise ValueError("endpoint native clocks differ from paid probe history")
    reading: dict[str, Any] = dict(
        phase=phase,
        native_updates=state.completed_updates,
        native_rows=state.training_examples,
        work=None,
    )
    row["readout_work"].append(reading)
    if separate_context_heads:
        reading.update(
            state=asdict(state), work=dict(attempted_calls=0, returned_calls=0, validated_calls=0)
        )

    def unchanged():
        guard()
        if not (_same_value if separate_context_heads else native.same_state)(
            learner.snapshot_state(), pinned
        ):
            raise ValueError("endpoint readout changed the disposable learner")

    holder = []

    def construct():
        unchanged()
        holder.append(
            ContextPerceptronDecisionPredictor(spec, state)
            if separate_context_heads
            else native.readout(spec, state)
        )
        unchanged()

    boundary(construct, after_clocks=unchanged)
    reader: Any = holder[0]
    reader_pin = deepcopy(reader) if separate_context_heads else None

    def predict(request):
        unchanged()
        result = None

        def read():
            nonlocal result
            if separate_context_heads:
                if not _same_value(reader, reader_pin):
                    raise ValueError("frozen contextual probe readout changed")
                reading["work"]["attempted_calls"] += 1
            try:
                result = reader(request)
            finally:
                if not separate_context_heads:
                    reading["work"] = asdict(reader.work)
            if separate_context_heads:
                reading["work"]["returned_calls"] += 1
                from src.core.typed_decision import validate_decision_result

                validate_decision_result(request, result)
                reading["work"]["validated_calls"] += 1
                if not _same_value(reader, reader_pin):
                    raise ValueError("frozen contextual probe readout changed")
            else:
                reader.assert_unchanged()
            unchanged()

        boundary(read, after_clocks=unchanged)
        return result

    for version, concept in HEAD_TASKS if separate_context_heads else TASKS:
        boundary()
        episode = run_workflow_episode(
            WorkflowState(
                f"probe-{row['mode']}-{phase}-{version}-{concept}", concept, policy_version=version
            ),
            predict,
            capabilities=WorkflowCapabilities(ACTION_IDS),
            max_steps=6,
            **({"include_policy_context": True} if separate_context_heads else {}),
        )
        row["episodes"].append(
            dict(
                phase=phase,
                policy_version=version,
                concept=concept,
                role="probe_test",
                episode=episode,
            )
        )
        boundary()
        unchanged()
        if not episode["complete"]:
            raise ValueError("endpoint episode has unknown paid work")


def run_arm(
    seed, state, cohorts, spec, guard, memory, name, *, row=None, separate_context_heads=False
):
    """One original probe owner and two finite sessions, never budget renewal."""
    training, evaluation, monitors, returning = cohorts
    sessions = [
        ToyBudgetSession(HEAD_LIMITS if separate_context_heads else LIMITS, monotonic)
        for _ in range(2)
    ]
    if memory is not None:
        for session in sessions:
            session.attach_memory(memory)
    if row is None:
        row = {}
    row.update(
        mode=name,
        prior_native_updates=state.completed_updates,
        prior_native_rows=state.training_examples,
        episodes=[],
        readout_work=[],
    )
    try:
        row["probe"] = run_fresh_task_probe(
            seed,
            state,
            training,
            evaluation,
            budget=sessions[0],
            checkpoint_updates=CHECKPOINTS,
            training_readouts=monitors,
            return_training=returning,
            return_budget=sessions[1],
            decision_diagnostics=True,
            check_boundary=guard,
            endpoint_observer=lambda learner, phase, boundary: _observe_tasks(
                learner,
                phase,
                boundary,
                row,
                spec,
                guard,
                **({"separate_context_heads": True} if separate_context_heads else {}),
            ),
        )
    finally:
        row["sessions"] = [
            dict(
                limits=asdict(s.budget),
                updates=s.updates_completed,
                started_at=s.started_at,
                last_clock=s.last_clock,
            )
            for s in sessions
        ]
    guard()
    _require_arm(row, **({"separate_context_heads": True} if separate_context_heads else {}))
    return row


def run_case(input_path, *, separate_context_heads=False):
    if type(separate_context_heads) is not bool:
        raise ValueError("separate_context_heads must be an exact boolean")
    if separate_context_heads:
        from scripts.context_head_probe import run_context_case

        return run_context_case(input_path, run_arm=run_arm, require_arm=_require_arm)
    started = monotonic()
    report: dict[str, Any] = dict(
        definition="isolated_ticket_policy_probe_v1",
        complete=False,
        source_sha256=SOURCE_SHA,
        cases=[],
        training_work=dict(attempted_updates=0, attempted_rows=0, returned_updates=0),
        final_examples=0,
        replay_examples=0,
        promotions=0,
        scope="TRAIN-derived same-template probe; global row fit and permitted-mask sandbox tasks separate; no generalization/plasticity-loss/CPC claim",
    )
    try:
        data, source = _source(input_path)
        _require_source_records(source)
        torch = native.require_torch()
        numpy_before, torch_before = deepcopy(np.random.get_state()), torch.get_rng_state().clone()
        spec = DecisionFeatureSpec(FEATURES, CATALOG)
        cohorts = build_cohorts(spec)
        cohort_pin = deepcopy(cohorts)
        states = [bp_snapshot_from_record(source["checkpoints"][i]["state"]) for i in (1, 0)]
        state_pin = deepcopy(states)
        seeds: list[tuple[Any, Any]] = []
        paid: list[tuple[dict[str, Any], dict[str, Any]]] = []
        work = report["training_work"]

        def before_update(features):
            guard()
            work["attempted_updates"] += 1
            work["attempted_rows"] += len(features)
            if work["attempted_updates"] > 64 or work["attempted_rows"] > 192:
                raise ValueError("original aggregate probe allowance exhausted")

        def after_update():
            work["returned_updates"] += 1

        def guard():
            if monotonic() - started > 180:
                raise ValueError("original total probe deadline exhausted")
            if input_path.read_bytes() != data or not _same_value(cohorts, cohort_pin):
                raise ValueError("probe source bytes or original data changed")
            if not native.same_state(states, state_pin):
                raise ValueError("original probe input states changed")
            if not _same_value(np.random.get_state(), numpy_before) or not torch.equal(
                torch.get_rng_state(), torch_before
            ):
                raise ValueError("probe changed global NumPy/Torch RNG")
            for seed, state in seeds:
                if not native.same_state(seed.snapshot_state(), state):
                    raise ValueError("probe changed original source owner")
            for i, (row, pin) in enumerate(paid):
                if report["cases"][i] is not row or not _same_value(row, pin):
                    # Keep the original paid result while refusing further work.
                    row.clear()
                    row.update(deepcopy(pin))
                    report["cases"][i] = row
                    raise ValueError("preceding paid probe evidence changed")

        with ProcessRssSampler() as memory:
            for name, state in zip(("warm", "fresh"), states, strict=True):
                guard()
                seed = native.counted_learner("bp", 7, 6, before_update, after_update)
                seed.restore_state(state)
                seeds.append((seed, deepcopy(state)))
                guard()
                # Keep a failed paid row even when its completion check rejects it.
                row: dict[str, Any] = {}
                report["cases"].append(row)
                run_arm(seed, state, cohorts, spec, guard, memory, name, row=row)
                _require_arm(row)
                guard()
                paid.append((row, deepcopy(row)))
            report["sampled_process_rss"] = asdict(memory.snapshot())
        report.update(
            complete=True,
            source_owners_unchanged=True,
            source_bytes_unchanged=True,
            numpy_rng_unchanged=True,
            torch_rng_unchanged=True,
            input_array_bytes=sum(
                a.nbytes for group in cohorts for b in group for a in (b.features, b.targets)
            ),
            source_prior_training=dict(updates=96, rows=576),
        )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"[:1024]
    report["training_work"]["unreturned_work_unknown"] = (
        report["training_work"]["attempted_updates"] != report["training_work"]["returned_updates"]
    )
    report["elapsed_seconds"] = monotonic() - started
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--separate-context-heads", action="store_true")
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        raise ValueError("qualified source must stay sealed")
    with args.output.open("xb") as target:
        target.write(b"{}\n")
    report = (
        run_case(args.input, separate_context_heads=True)
        if args.separate_context_heads
        else run_case(args.input)
    )
    args.output.write_bytes(_encode_report(report))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
