"""Native peer controls or an explicit fresh BP demo through the existing stream.

Peer modes reuse BP evidence; --fresh-bp trains a new candidate without a saved
checkpoint. Equal exposure is not equal computation or equal optimization.
See docs/native-ticket-controls.md and docs/fresh-bp-ticket-demo.md.
"""

import argparse
from dataclasses import asdict, replace
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from scripts import ticket_workflow_native as native
from scripts import run_bp_new_contexts as context
from scripts.diagnose_ticket_training import _teacher
from scripts.diagnose_ticket_training import _summarize_logits
from scripts.run_ticket_workflow import FEATURES, _encode_report
from src.app.coupled_attribution import AttributionStream
from src.app.ticket_workflow import run_workflow_episode
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.decision_features import DecisionFeatureSpec
from src.core.ticket_workflow import ACTION_IDS, CATALOG, WorkflowCapabilities, WorkflowState
from src.shared.process_memory import ProcessRssSampler

METHODS = ("pc", "neutral_cpc", "gated_cpc")
WARMUP = 96
PARAMETERS = ("weight_feature_hidden", "bias_hidden", "weight_hidden_output", "bias_output")


def parameters(state):
    """Observe the four common tensors, excluding method-specific adaptive state."""
    return {key: state["head_state"][key] for key in PARAMETERS}


def training_sequence(spec):
    cases, teacher, adaptation = context._teacher_batches(spec)
    _, old = _teacher(spec)
    warm = tuple(
        replace(old, context="joint-train", row_ids=tuple(f"fit-{r}-{i}" for i in range(6)))
        for r in range(WARMUP)
    )
    templates = (old, adaptation[0], adaptation[-1])
    data: dict[str, Any] = {
        key: dict(features=b.features.tolist(), targets=b.targets.tolist())
        for key, b in zip(("warm", "new", "return"), templates, strict=True)
    }
    data.update(role="train", warmup=WARMUP, batches_per_phase=context.BATCHES_PER_PHASE)
    return cases, teacher, (*warm, *adaptation), data


def run_case(method, *, fresh_bp=False):
    if type(fresh_bp) is not bool or (fresh_bp and method != "bp"):
        raise ValueError("fresh BP mode requires explicit BP selection")
    if not fresh_bp and method not in METHODS:
        raise ValueError("explicit PC/CPC peer method required")
    started = monotonic()
    report: dict[str, Any] = dict(
        definition="fresh_bp_ticket_demo_v1" if fresh_bp else "matched_native_ticket_controls_v1",
        method=method,
        warmup_updates=WARMUP,
        complete=False,
        phases=[
            dict(context="new-contexts", concepts=list(context.NEW)),
            dict(context="old-return", concepts=list(context.OLD)),
        ],
        batches_per_phase=context.BATCHES_PER_PHASE,
        checkpoints=[],
        episodes=[],
        readout_work=[],
        teacher_reads=dict(attempted=0, returned=0, rows_attempted=0),
        final_examples=0,
        reserved_probe_updates=0,
        replay_examples=0,
        external_actions=0,
    )
    if fresh_bp:
        report["role"] = "fresh_train_demo"
    work = dict(attempted_updates=0, attempted_rows=0, returned_updates=0)
    states: dict[int, Any] = {}
    budget = seed = stream = torch = torch_rng = None
    rng = np.random.get_state()
    assert type(rng) is tuple
    with ProcessRssSampler(interval_seconds=0.02) as sampler:
        try:
            spec = DecisionFeatureSpec(FEATURES, CATALOG)
            cases, teacher, batches, data = training_sequence(spec)
            report["training_data"] = data
            torch = native.require_torch()
            torch_rng = torch.get_rng_state().clone()

            def before_update(features):
                work["attempted_updates"] += 1
                work["attempted_rows"] += len(features)

            def after_update():
                work["returned_updates"] += 1

            budget = ToyBudgetSession(
                ToyExecutionBudget(
                    max_training_updates=WARMUP + 2 * context.BATCHES_PER_PHASE,
                    max_wall_seconds=60,
                    max_process_rss_bytes=512 * 1024**2,
                ),
                monotonic,
            )
            budget.attach_memory(sampler)
            budget.before_final()
            seed = native.counted_learner(method, 7, 6, before_update, after_update)
            initial = seed.snapshot_state()
            report["initial_parameters"] = parameters(native.snapshot_record(initial))

            def capture(point, state, batch):
                states[point] = state

            stream = AttributionStream(
                seed,
                batches,
                (),
                checkpoint_updates=(
                    0,
                    WARMUP,
                    WARMUP + context.BATCHES_PER_PHASE,
                    WARMUP + 2 * context.BATCHES_PER_PHASE,
                ),
                label_delay=1,
                budget=budget,
                observe_checkpoint=capture,
                training_only=True,
            )
            actor = stream._runtime.actor
            for point in (0, context.BATCHES_PER_PHASE, 2 * context.BATCHES_PER_PHASE):
                native_point = WARMUP + point
                stream.run_until(native_point)
                state = stream.candidate_snapshot()
                if (
                    not native.same_state(state, states[native_point])
                    or state.completed_updates != native_point
                    or state.training_examples != 6 * native_point
                    or budget.updates_completed != native_point
                ):
                    raise ValueError("peer native/source/budget offset mismatch")
                checkpoint = native.checkpoint(point, state)
                report["checkpoints"].append(checkpoint)
                predict = native.readout(spec, state)
                paid = dict(source_updates=point, work=asdict(predict.work))
                report["readout_work"].append(paid)

                def require_source():
                    budget.before_final()
                    if (
                        stream._runtime.actor is not actor
                        or not native.same_state(stream.candidate_snapshot(), state)
                        or not native.same_state(seed.snapshot_state(), initial)
                        or not native.same_state(actor.snapshot_state().state, initial)
                        or budget.updates_completed != native_point
                    ):
                        raise ValueError("frozen readout changed source/seed/actor/new budget")
                    predict.assert_unchanged()

                require_source()
                report["teacher_reads"]["attempted"] += 1
                report["teacher_reads"]["rows_attempted"] += 12
                logits = predict._learner.predict_logits(teacher.features)
                report["teacher_reads"]["returned"] += 1
                require_source()
                checkpoint["teacher_logits"] = logits.tolist()
                _, ce, margins, probability = _summarize_logits(logits, teacher.targets)
                checkpoint["teacher_fit"] = dict(
                    global_mean_ce=float(np.mean(ce)),
                    probability_margins=probability,
                    rows=[
                        dict(
                            concept=context.CONCEPTS[i // 3],
                            phase=("sealed", "ready", "staged")[i % 3],
                            target_id=c.target_id,
                            raw_margin=float(margins[i]),
                            historical_train_exposures=(
                                WARMUP + max(0, point - context.BATCHES_PER_PHASE)
                                if context.CONCEPTS[i // 3] in context.OLD
                                else min(point, context.BATCHES_PER_PHASE)
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

                for scenario, concept in context._slots(point):
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
            report["summary"] = context._summary(report["episodes"])
            report["source_seed_actor_unchanged"] = native.same_state(
                seed.snapshot_state(), initial
            ) and native.same_state(actor.snapshot_state().state, initial)
            report["complete"] = report["source_seed_actor_unchanged"]
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
                nominal_training_affine_terms=39 * work["attempted_rows"],
                checkpoint_tensor_bytes=sum(p["tensor_bytes"] for p in report["checkpoints"]),
                qualification=(
                    "fresh native BP TRAIN stream; all training charged here; gradients/backward/initialization/copies/teacher/typed/IO/validation/Python additional"
                    if fresh_bp
                    else "full112update TRAIN stream; native PC latent/initialization/copies/teacher/typed/IO/validation/Python additional; BP prior work reused, not repeated"
                ),
            )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--method", choices=METHODS)
    mode.add_argument(
        "--fresh-bp",
        action="store_true",
        help="train a fresh CPU BP candidate through 112 updates; requires Torch, no saved input",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    with args.output.open("xb") as target:
        target.write(b"{}\n")
    report = run_case("bp", fresh_bp=True) if args.fresh_bp else run_case(args.method)
    args.output.write_bytes(_encode_report(report))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
