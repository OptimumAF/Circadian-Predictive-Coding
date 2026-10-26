"""Fixed disposable BP joint-TRAIN fit through existing stream and task ports.

Qualifications use both concepts from outset, not chronological retention. No
new learning rule, promotion, final/probe/replay data or adaptive stopping.
"""

import argparse
from dataclasses import asdict, replace
from copy import deepcopy
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from scripts.diagnose_ticket_training import _summarize_logits, _teacher
from scripts.run_ticket_workflow import (
    FEATURES,
    CONTEXT_FEATURES,
    CONTEXT_TASKS,
    training_cases,
    build_joint_context_batch,
    _encode_report,
)
from scripts import ticket_workflow_native as native
from src.app.coupled_attribution import AttributionStream, _same_value
from src.app.ticket_workflow import run_workflow_episode
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.decision_features import DecisionFeatureSpec
from src.core.ticket_workflow import ACTION_IDS, CATALOG, WorkflowCapabilities, WorkflowState
from src.shared.process_memory import ProcessRssSampler

UPDATES = 96


def _context_teacher(spec):
    """Admit only aligned known TRAIN context templates before model construction."""
    if spec != DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG):
        raise ValueError("context qualification requires the exact D9 schema")
    teachers = {
        v: training_cases(policy_version=v, include_policy_context=True) for v in ("v1", "v2")
    }
    return build_joint_context_batch(spec, teachers)


def _teacher_fit(logits, cases, batch, *, tasks=None):
    _, ce, margins, summary = _summarize_logits(logits, batch.targets)
    ids = tuple(c.candidate_id for c in CATALOG)
    rows = []
    for i, case in enumerate(cases):
        allowed = sorted(c.candidate_id for c in case.request.allowed_candidates)
        winner = min(allowed, key=lambda key: (-float(logits[i, ids.index(key)]), key))
        target_value = float(logits[i, ids.index(case.target_id)])
        alternative = max(
            float(logits[i, ids.index(key)]) for key in allowed if key != case.target_id
        )
        rows.append(
            dict(
                concept=("invoice", "parcel")[i // 3] if tasks is None else tasks[i // 3][1],
                phase=("sealed", "ready", "staged")[i % 3],
                target_id=case.target_id,
                selected_id=winner,
                allowed_correct=winner == case.target_id,
                allowed_raw_margin=target_value - alternative,
                global_raw_margin=float(margins[i]),
            )
        )
        if tasks is not None:
            rows[-1]["policy_version"] = tasks[i // 3][0]
    return dict(
        global_mean_ce=float(np.mean(ce)),
        global_probability_margins=summary,
        allowed_correct_rows=sum(r["allowed_correct"] for r in rows),
        rows=rows,
    )


def run_qualification(*, policy_context=False):
    if type(policy_context) is not bool:
        raise ValueError("policy_context must be an exact bool")
    started = monotonic()
    width = len(CONTEXT_FEATURES if policy_context else FEATURES)
    rows_per_update = 12 if policy_context else 6
    tasks = CONTEXT_TASKS if policy_context else (("v2", "invoice"), ("v2", "parcel"))
    context_options = {"include_policy_context": True} if policy_context else {}
    report: dict[str, Any] = dict(
        definition="isolated_bp_ticket_joint_fit_v1",
        complete=False,
        role="disposable_train_fit_qualification",
        planned_updates=UPDATES,
        rows_per_update=rows_per_update,
        checkpoints=[],
        episodes=[],
        native_readout_work=[],
        teacher_reads=dict(attempted=0, returned=0, rows_attempted=0),
        final_examples=0,
        reserved_probe_updates=0,
        replay_examples=0,
        external_actions=0,
        interpretation="joint known TRAIN states, fixed dose; no retention/return/generalization/CPC/release claim",
    )
    if policy_context:
        report.update(
            definition="isolated_bp_context_joint_fit_v1",
            role="disposable_context_train_fit_qualification",
            policy_context_in_query=True,
        )
    work = dict(attempted_updates=0, attempted_rows=0, returned_updates=0)
    seed = stream = budget = None
    states: dict[int, Any] = {}
    torch = native.require_torch()
    torch_rng = torch.get_rng_state().clone()
    numpy_rng = np.random.get_state()
    assert type(numpy_rng) is tuple
    context_guard = lambda: None

    def before_update(features):
        if policy_context:
            context_guard()
        work["attempted_updates"] += 1
        work["attempted_rows"] += len(features)

    def after_update():
        work["returned_updates"] += 1

    def capture(point, state, batch):
        states[point] = state

    with ProcessRssSampler(interval_seconds=0.02) as sampler:
        try:
            spec = DecisionFeatureSpec(CONTEXT_FEATURES if policy_context else FEATURES, CATALOG)
            cases, batch = _context_teacher(spec) if policy_context else _teacher(spec)
            input_pin = deepcopy((spec, cases, batch)) if policy_context else None
            initial_evidence = None
            batches = tuple(
                replace(
                    batch,
                    context="joint-train",
                    row_ids=tuple(f"fit-{repeat}-{i}" for i in range(rows_per_update)),
                )
                for repeat in range(UPDATES)
            )
            report["training_data"] = dict(
                role="train",
                concepts=[c for _, c in tasks],
                all_available_at_outset=True,
                rows=[dict(query=c.request.query, target_id=c.target_id) for c in cases],
                features=batch.features.tolist(),
                targets=batch.targets.tolist(),
            )
            if policy_context:
                report["training_data"].update(
                    policy_versions=[v for v, _ in tasks],
                    observation_schema="public_policy_context_d9",
                )
            budget = ToyBudgetSession(
                ToyExecutionBudget(
                    max_training_updates=UPDATES,
                    max_wall_seconds=60,
                    max_process_rss_bytes=512 * 1024**2,
                ),
                monotonic,
            )
            budget.attach_memory(sampler)
            budget.before_final()
            seed = native.counted_learner("bp", width, len(CATALOG), before_update, after_update)
            initial = seed.snapshot_state()
            stream = AttributionStream(
                seed,
                batches,
                (),
                checkpoint_updates=(0, UPDATES),
                label_delay=1,
                budget=budget,
                observe_checkpoint=capture,
                training_only=True,
            )
            actor = stream._runtime.actor
            original_actor = actor.snapshot_state().state
            for point in (0, UPDATES):
                if point:
                    stream.run_until(point)
                state = stream.candidate_snapshot()
                if (
                    not native.same_state(state, states[point])
                    or state.completed_updates != point
                    or state.training_examples != rows_per_update * point
                    or budget.updates_completed != point
                ):
                    raise ValueError("qualification native/checkpoint/budget counters differ")
                checkpoint = native.checkpoint(point, state)
                report["checkpoints"].append(checkpoint)
                predict = native.readout(spec, state)
                readout_work = dict(source_updates=point, work=asdict(predict.work))
                report["native_readout_work"].append(readout_work)

                def require_source():
                    budget.before_final()
                    if (
                        not native.same_state(stream.candidate_snapshot(), state)
                        or not native.same_state(seed.snapshot_state(), initial)
                        or not native.same_state(actor.snapshot_state().state, original_actor)
                        or budget.updates_completed != point
                    ):
                        raise ValueError("qualification readout changed source/actor/seed/budget")
                    predict.assert_unchanged()
                    if policy_context:
                        context_guard()

                require_source()
                teacher = report["teacher_reads"]
                teacher["attempted"] += 1
                teacher["rows_attempted"] += rows_per_update
                # The separate owned typed readout also supplies the complete K
                # logits for TRAIN diagnosis; count this additional forward.
                logits = predict._learner.predict_logits(batch.features)
                teacher["returned"] += 1
                require_source()
                checkpoint["teacher_fit"] = (
                    _teacher_fit(logits, cases, batch, tasks=tasks)
                    if policy_context
                    else _teacher_fit(logits, cases, batch)
                )

                def guarded_predict(request):
                    require_source()
                    try:
                        result = predict(request)
                    finally:
                        readout_work["work"] = asdict(predict.work)
                    require_source()
                    return result

                for version, concept in tasks:
                    for scenario in ("ordinary", "stale_once", "revoke_commit"):
                        require_source()
                        episode = run_workflow_episode(
                            WorkflowState(
                                f"qual-{point}-{concept}", concept, policy_version=version
                            ),
                            guarded_predict,
                            capabilities=WorkflowCapabilities(ACTION_IDS),
                            max_steps=6,
                            scenario=scenario,
                            **context_options,
                        )
                        report["episodes"].append(
                            dict(
                                source_updates=point,
                                concept=concept,
                                role="train_task_readout",
                                episode=episode,
                            )
                        )
                        if policy_context:
                            report["episodes"][-1]["policy_version"] = version
                        require_source()
                        if not episode["complete"]:
                            raise RuntimeError("qualification task has uncertain paid work")
                if policy_context and point == 0:
                    initial_evidence = deepcopy(
                        (
                            report["checkpoints"][0],
                            report["episodes"][:12],
                            report["native_readout_work"][0],
                        )
                    )

                    def guard_context_inputs():
                        if not _same_value((spec, cases, batch), input_pin):
                            raise ValueError("original contextual TRAIN inputs changed")
                        if not _same_value(
                            (
                                report["checkpoints"][0],
                                report["episodes"][:12],
                                report["native_readout_work"][0],
                            ),
                            initial_evidence,
                        ):
                            report["checkpoints"][0].clear()
                            report["checkpoints"][0].update(deepcopy(initial_evidence[0]))
                            report["episodes"][:12] = deepcopy(initial_evidence[1])
                            report["native_readout_work"][0].clear()
                            report["native_readout_work"][0].update(deepcopy(initial_evidence[2]))
                            raise ValueError("preceding contextual paid evidence changed")
                        if not native.same_state(
                            seed.snapshot_state(), initial
                        ) or not native.same_state(actor.snapshot_state().state, original_actor):
                            raise ValueError("context qualification source owner changed")

                    context_guard = guard_context_inputs
                    context_guard()
            report["stream"] = stream.finish()
            report["source_seed_unchanged"] = native.same_state(seed.snapshot_state(), initial)
            report["deployed_actor_unchanged"] = native.same_state(
                actor.snapshot_state().state, original_actor
            )
            report["complete"] = (
                report["source_seed_unchanged"] and report["deployed_actor_unchanged"]
            )
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"[:512]
        finally:
            report["training_work"] = dict(
                **work,
                unreturned_work_unknown=work["attempted_updates"] != work["returned_updates"],
            )
            if budget is not None:
                report["budget"] = dict(
                    limits=asdict(budget.budget),
                    started_at=budget.started_at,
                    last_clock=budget.last_clock,
                    updates_completed=budget.updates_completed,
                )
            if stream is not None:
                try:
                    report["applied_updates"] = [asdict(r) for r in stream.applied_updates]
                except Exception as error:
                    report.update(
                        complete=False, receipt_prefix_unknown=True, receipt_error=str(error)[:512]
                    )
            current_rng = np.random.get_state()
            assert type(current_rng) is tuple
            report["numpy_rng_unchanged"] = (
                numpy_rng[0] == current_rng[0]
                and np.array_equal(numpy_rng[1], current_rng[1])
                and numpy_rng[2:] == current_rng[2:]
            )
            report["torch_rng_unchanged"] = bool(torch.equal(torch_rng, torch.get_rng_state()))
            if not report["numpy_rng_unchanged"] or not report["torch_rng_unchanged"]:
                report.update(complete=False, error="qualification changed caller RNG")
            report["costs"] = dict(
                elapsed_seconds=monotonic() - started,
                sampled_process_rss=asdict(sampler.snapshot()),
                attempted_training_nominal_affine_terms=work["attempted_rows"]
                * (width * 3 + 3 * 6),
                teacher_forward_rows=report["teacher_reads"]["rows_attempted"],
                checkpoint_tensor_bytes=sum(p["tensor_bytes"] for p in report["checkpoints"]),
                qualification="nominal full-class affine work; initialization/gradients/backward/copies/RNG/validation/Python additional; externally metered physical attempts and finite failures",
            )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--policy-context",
        action="store_true",
        help="qualify the fixed joint D9 public-context TRAIN recipe",
    )
    args = parser.parse_args(argv)
    with args.output.open("xb") as target:
        target.write(b"{}\n")
    result = run_qualification(policy_context=True) if args.policy_context else run_qualification()
    args.output.write_bytes(_encode_report(result))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
