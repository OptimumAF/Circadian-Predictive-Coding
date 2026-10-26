"""Fixed new-context acquisition/old return using saved qualified native BP.

Full BP payload readback only, not old application authority. Existing stream
owns new delayed TRAIN work; frozen copies own diagnostic/task readouts.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from scripts import ticket_workflow_native as native
from scripts.diagnose_ticket_training import _summarize_logits
from scripts.run_ticket_workflow import FEATURES, training_cases, _encode_report
from src.adapters.perceptron_decision import build_decision_training_batch
from src.adapters.torch_bp_record import bp_snapshot_from_record
from src.app.coupled_attribution import AttributionStream
from src.app.ticket_workflow import run_workflow_episode
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.decision_features import DecisionFeatureSpec
from src.core.temporal_decision_sequence import VOCABULARY
from src.core.ticket_workflow import ACTION_IDS, CATALOG, WorkflowCapabilities, WorkflowState
from src.shared.process_memory import ProcessRssSampler

SOURCE_SHA = "ee40be1162f38d22730b3050726687e063c5a54cde6238ade158f9fc1efc6b27"
BATCHES_PER_PHASE = 8
OLD = ("invoice", "parcel")
NEW = ("legacy", "password")
CONCEPTS = (*OLD, *NEW)


def _source(path):
    if path.stat().st_size > 512 * 1024:
        raise ValueError("qualified source exceeds original report bound")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA:
        raise ValueError("qualified source bytes changed or replaced")
    record = json.loads(data)
    if (
        record["complete"] is not True
        or record["role"] != "disposable_train_fit_qualification"
        or record["training_work"]
        != dict(
            attempted_updates=96,
            attempted_rows=576,
            returned_updates=96,
            unreturned_work_unknown=False,
        )
    ):
        raise ValueError("original complete qualified source required")
    return data, record


def _teacher_batches(spec):
    cases = training_cases()
    if len(cases) != 12 or any(
        c.role != "train" or c.source_group != "train-workflow-traces" for c in cases
    ):
        raise ValueError("only original TRAIN teacher allowed")

    def select(concepts):
        return tuple(cases[3 * VOCABULARY.index(c) + i] for c in concepts for i in range(3))

    full = select(CONCEPTS)
    all_batch = build_decision_training_batch(full, spec, "all-teacher", allow_subsets=True)
    batches = []
    for phase, concepts in (("new-contexts", NEW), ("old-return", OLD)):
        batch = build_decision_training_batch(select(concepts), spec, phase, allow_subsets=True)
        for repeat in range(BATCHES_PER_PHASE):
            batches.append(replace(batch, row_ids=tuple(f"{phase}-{repeat}-{i}" for i in range(6))))
    return full, all_batch, tuple(batches)


def _slots(point):
    slots = [("ordinary", c) for c in CONCEPTS]
    special = NEW if point == BATCHES_PER_PHASE else OLD if point == 2 * BATCHES_PER_PHASE else ()
    return tuple(slots + [(s, c) for c in special for s in ("stale_once", "revoke_commit")])


def _summary(episodes):
    ordinary = {
        p: {
            r["concept"]: r["episode"]["success"]
            for r in episodes
            if r["source_updates"] == p and r["episode"]["scenario"] == "ordinary"
        }
        for p in (0, BATCHES_PER_PHASE, 2 * BATCHES_PER_PHASE)
    }
    count = lambda p, concepts: sum(int(ordinary[p][c]) for c in concepts)
    return dict(
        ordinary=ordinary,
        new_context_gain=count(BATCHES_PER_PHASE, NEW) - count(0, NEW),
        old_forgetting=count(0, OLD) - count(BATCHES_PER_PHASE, OLD),
        old_return_change=count(2 * BATCHES_PER_PHASE, OLD) - count(BATCHES_PER_PHASE, OLD),
        new_change_during_return=count(2 * BATCHES_PER_PHASE, NEW) - count(BATCHES_PER_PHASE, NEW),
    )


def run_case(input_path: Path):
    started = monotonic()
    report: dict[str, Any] = dict(
        definition="qualified_bp_new_contexts_v1",
        complete=False,
        source_sha256=SOURCE_SHA,
        phases=[
            dict(context="new-contexts", concepts=list(NEW)),
            dict(context="old-return", concepts=list(OLD)),
        ],
        batches_per_phase=BATCHES_PER_PHASE,
        checkpoints=[],
        episodes=[],
        readout_work=[],
        teacher_reads=dict(attempted=0, returned=0, rows_attempted=0),
        final_examples=0,
        reserved_probe_updates=0,
        replay_examples=0,
        external_actions=0,
    )
    work = dict(attempted_updates=0, attempted_rows=0, returned_updates=0)
    states: dict[int, Any] = {}
    budget = seed = stream = torch = torch_rng = None
    rng = np.random.get_state()
    assert type(rng) is tuple
    with ProcessRssSampler(interval_seconds=0.02) as sampler:
        try:
            data, source = _source(input_path)
            report["prior_training_work"] = deepcopy(source["training_work"])
            report["prior_costs"] = deepcopy(source["costs"])
            torch = native.require_torch()
            torch_rng = torch.get_rng_state().clone()
            initial = bp_snapshot_from_record(source["checkpoints"][-1]["state"])
            if initial.completed_updates != 96 or initial.training_examples != 576:
                raise ValueError("qualified native clock differs")
            spec = DecisionFeatureSpec(FEATURES, CATALOG)
            cases, teacher, batches = _teacher_batches(spec)

            def require_input():
                if input_path.stat().st_size != len(data) or input_path.read_bytes() != data:
                    raise ValueError("original qualified source bytes changed")

            def before_update(features):
                work["attempted_updates"] += 1
                work["attempted_rows"] += len(features)
                require_input()

            def after_update():
                work["returned_updates"] += 1

            budget = ToyBudgetSession(
                ToyExecutionBudget(
                    max_training_updates=2 * BATCHES_PER_PHASE,
                    max_wall_seconds=60,
                    max_process_rss_bytes=512 * 1024**2,
                ),
                monotonic,
            )
            budget.attach_memory(sampler)
            budget.before_final()
            seed = native.counted_learner("bp", 7, 6, before_update, after_update)
            seed.restore_state(initial)

            def capture(point, state, batch):
                states[point] = state

            stream = AttributionStream(
                seed,
                batches,
                (),
                checkpoint_updates=(0, BATCHES_PER_PHASE, 2 * BATCHES_PER_PHASE),
                label_delay=1,
                budget=budget,
                observe_checkpoint=capture,
                training_only=True,
            )
            actor = stream._runtime.actor
            for point in (0, BATCHES_PER_PHASE, 2 * BATCHES_PER_PHASE):
                require_input()
                if point:
                    stream.run_until(point)
                state = stream.candidate_snapshot()
                if (
                    not native.same_state(state, states[point])
                    or state.completed_updates != 96 + point
                    or state.training_examples != 576 + 6 * point
                    or budget.updates_completed != point
                ):
                    raise ValueError("qualified native/source/new-budget offset mismatch")
                checkpoint = native.checkpoint(point, state)
                report["checkpoints"].append(checkpoint)
                predict = native.readout(spec, state)
                paid = dict(source_updates=point, work=asdict(predict.work))
                report["readout_work"].append(paid)

                def require_source():
                    budget.before_final()
                    require_input()
                    if (
                        stream._runtime.actor is not actor
                        or not native.same_state(stream.candidate_snapshot(), state)
                        or not native.same_state(seed.snapshot_state(), initial)
                        or not native.same_state(actor.snapshot_state().state, initial)
                        or budget.updates_completed != point
                    ):
                        raise ValueError("frozen readout changed source/seed/actor/new budget")
                    predict.assert_unchanged()

                require_source()
                report["teacher_reads"]["attempted"] += 1
                report["teacher_reads"]["rows_attempted"] += 12
                logits = predict._learner.predict_logits(teacher.features)
                report["teacher_reads"]["returned"] += 1
                require_source()
                _, ce, margins, probability = _summarize_logits(logits, teacher.targets)
                checkpoint["teacher_fit"] = dict(
                    global_mean_ce=float(np.mean(ce)),
                    probability_margins=probability,
                    rows=[
                        dict(
                            concept=CONCEPTS[i // 3],
                            phase=("sealed", "ready", "staged")[i % 3],
                            target_id=c.target_id,
                            raw_margin=float(margins[i]),
                            historical_train_exposures=(
                                96 + max(0, point - BATCHES_PER_PHASE)
                                if CONCEPTS[i // 3] in OLD
                                else min(point, BATCHES_PER_PHASE)
                            ),
                        )
                        for i, c in enumerate(cases)
                    ],
                )

                def guarded_predict(request):
                    require_source()
                    try:
                        result = predict(request)
                    finally:
                        paid["work"] = asdict(predict.work)
                    require_source()
                    return result

                for scenario, concept in _slots(point):
                    require_source()
                    episode = run_workflow_episode(
                        WorkflowState(f"new-{point}-{concept}", concept),
                        guarded_predict,
                        capabilities=WorkflowCapabilities(ACTION_IDS),
                        scenario=scenario,
                        max_steps=6,
                    )
                    report["episodes"].append(
                        dict(
                            source_updates=point,
                            concept=concept,
                            role="outer_selection",
                            episode=episode,
                        )
                    )
                    require_source()
                    if not episode["complete"]:
                        raise RuntimeError("new-context episode has uncertain paid work")
            report["stream"] = stream.finish()
            report["summary"] = _summary(report["episodes"])
            report["source_seed_actor_unchanged"] = native.same_state(
                seed.snapshot_state(), initial
            ) and native.same_state(actor.snapshot_state().state, initial)
            require_input()
            report["complete"] = report["source_seed_actor_unchanged"]
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"[:512]
        finally:
            report["training_work"] = dict(
                **work,
                unreturned_work_unknown=work["attempted_updates"] != work["returned_updates"],
            )
            if budget is not None:
                report["new_budget"] = dict(
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
            current = np.random.get_state()
            assert type(current) is tuple
            report["numpy_rng_unchanged"] = (
                rng[0] == current[0]
                and np.array_equal(rng[1], current[1])
                and rng[2:] == current[2:]
            )
            report["torch_rng_unchanged"] = (
                None if torch is None else bool(torch.equal(torch_rng, torch.get_rng_state()))
            )
            if not report["numpy_rng_unchanged"] or report["torch_rng_unchanged"] is False:
                report.update(complete=False, error="caller RNG changed")
            report["costs"] = dict(
                elapsed_seconds=monotonic() - started,
                sampled_process_rss=asdict(sampler.snapshot()),
                nominal_new_training_affine_terms=39 * work["attempted_rows"],
                checkpoint_tensor_bytes=sum(p["tensor_bytes"] for p in report["checkpoints"]),
                qualification="new work only; prior96fit/576rows/costs preserved separately; native initialization/backward/copies/teacher/typed/IO/validation/Python additional",
            )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        raise ValueError("qualified input must stay sealed")
    with args.output.open("xb") as target:
        target.write(b"{}\n")
    report = run_case(args.input)
    args.output.write_bytes(_encode_report(report))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
