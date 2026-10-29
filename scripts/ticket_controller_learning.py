"""Finite fit/application recipe for the existing serving CLI, not a runner.

All four disposable trials and two applications reuse run_case's delayed
TRAIN stream/admission/task ports. No final/probe source is read here.
"""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from scripts import run_bp_ticket_serving as demo
from src.app.actor_shadow import StableActor
from src.app.probe_return_phase import paid_report_guard
from src.app.counterfactual_controller import CounterfactualController, CounterfactualReturn
from src.app.counterfactual_controller import (
    CatalogCounterfactualRequest,
    CatalogCounterfactualReturn,
)
from src.core.catalog_contextual_controller import ContextualArmCatalog
from scripts.ticket_catalog_control import case_work
from src.core.contextual_controller import ContextualChoice
from src.core.controller_objective import ControllerObjective
from src.core.reference_controller import ArrivedTrainingFeedback, ReferenceObservation


def run_study(input_path: Path, *, catalog: ContextualArmCatalog | None = None) -> dict[str, Any]:
    if catalog is not None:
        if type(catalog) is not ContextualArmCatalog:
            raise ValueError("catalog study requires an exact original catalog")
        catalog.__post_init__()
        n = demo.context.BATCHES_PER_PHASE
        if (
            catalog.envelope.max_decisions != n + 1
            or catalog.envelope.max_wake_updates != n
            or catalog.envelope.actions[2].replay_examples != 3
            or not 3 <= catalog.envelope.max_replay_examples <= 3 * n
            or catalog.envelope.max_replay_examples % 3
        ):
            raise ValueError(
                "catalog study requires the original TRAIN horizon and whole bank exposure"
            )
    started = monotonic()
    objective = ControllerObjective()
    clock = demo.LogicalClock()
    report: dict[str, Any] = dict(
        definition="learned_contextual_controller_v1",
        complete=False,
        objective=asdict(objective),
        declared_at=clock.now(),
        clock_scope="counterfactual_training_outcomes_v1; application learner clocks restart at0",
        trials=[],
        applications={},
        context_monitoring=[],
        final_source_reads=0,
        probe_updates=0,
        max_total_training_updates=32,
    )
    if catalog is not None:
        report.update(
            definition="catalog_ticket_controller_learning_v1",
            catalog=asdict(catalog),
            max_total_training_updates=8
            * (demo.context.BATCHES_PER_PHASE + catalog.envelope.max_replay_examples // 3),
            qualification="six controller-TRAIN trials plus two frozen private development applications; not final/probe/CPC/generalization",
        )
    torch = demo.native.require_torch()
    rng, torch_rng = np.random.get_state(), torch.get_rng_state().clone()
    assert type(rng) is tuple
    trainer = None
    paid_cases = []

    def require_paid_cases():
        for current, original, guard in paid_cases:
            if current() is not original:
                raise ValueError("preceding paid catalog case was replaced")
            try:
                guard()
            except ValueError as error:
                raise ValueError("preceding paid catalog case changed") from error

    with demo.ProcessRssSampler(interval_seconds=0.02) as sampler:
        budget = demo.ToyBudgetSession(
            demo.ToyExecutionBudget(max_wall_seconds=180, max_process_rss_bytes=512 * 1024**2),
            monotonic,
        )
        budget.attach_memory(sampler)
        try:
            _, prior = demo.context._source(input_path)
            initial = demo.context.bp_snapshot_from_record(prior["checkpoints"][-1]["state"])
            source = demo.native.counted_learner("bp", 7, 6, lambda f: None, lambda: None)
            source.restore_state(initial)
            actor = StableActor(source, version="qualified-96")
            trainer = CounterfactualController(
                actor.snapshot_state,
                demo.snapshot_digest,
                objective,
                clock,
                **({} if catalog is None else {"catalog": catalog}),
            )
            spec = demo.DecisionFeatureSpec(demo.FEATURES, demo.CATALOG)
            _, _, batches = demo.context._teacher_batches(spec)
            report["prior_training_work"] = prior["training_work"]
            report["source_sha256"] = demo.context.SOURCE_SHA
            for kind, batch in zip(
                ("new_contexts", "old_return"),
                (batches[0], batches[demo.context.BATCHES_PER_PHASE]),
                strict=True,
            ):
                budget.before_final()
                observed = clock.now()
                prediction = actor.predict(batch.features)
                clock.advance_to(observed + 1)
                feedback = ArrivedTrainingFeedback(
                    "train",
                    "context-" + kind,
                    prediction.actor_version,
                    observed,
                    clock.now(),
                    int(np.sum(np.argmax(prediction.prediction, axis=1) != batch.targets[:, 0])),
                    len(batch.features),
                )
                report["context_monitoring"].append(asdict(feedback))
                for _ in range(2 if catalog is None else len(catalog.arms)):
                    observation = ReferenceObservation(
                        clock.now(), 1, 1, 1, actor.version, feedback
                    )

                    def trial(request, snapshot):
                        budget.before_final()
                        if catalog is not None:
                            require_paid_cases()
                        row = dict(request=asdict(request), result=None)
                        report["trials"].append(row)
                        then = monotonic()
                        result = demo.run_case(
                            input_path,
                            training_context=kind,
                            base_snapshot=snapshot,
                            **(
                                dict(
                                    contextual_policy=ContextualChoice(
                                        request.action_id, request.action_id
                                    ),
                                    counterfactual=True,
                                )
                                if catalog is None
                                else dict(catalog_trial=request)
                            ),
                        )
                        seconds = monotonic() - then
                        row.update(result=result, execution_seconds=seconds)
                        budget.before_final()
                        clock.advance_to(request.started_at + 1)
                        native_work = None if catalog is None else case_work(result)
                        if native_work is not None:
                            row["native_work"] = asdict(native_work)
                        measures = demo._case_measures(result, seconds)
                        row["measures"] = asdict(measures)
                        clock.advance_to(request.started_at + 2)
                        if catalog is not None:
                            if type(request) is not CatalogCounterfactualRequest:
                                raise ValueError("catalog study lost its original request")
                            assert native_work is not None
                            guard, _ = paid_report_guard(row)
                            index = len(report["trials"]) - 1
                            paid_cases.append((lambda i=index: report["trials"][i], row, guard))
                            return CatalogCounterfactualReturn(
                                request, request.started_at + 1, clock.now(), measures, native_work
                            )
                        return CounterfactualReturn(
                            request, request.started_at + 1, clock.now(), measures
                        )

                    trainer.run(observation, trial)
            fitted = trainer.freeze()
            pin = deepcopy(fitted)
            report["fitted_policy"] = asdict(fitted)
            report["fitted_at"] = clock.now()
            # Fresh application owners receive the completed immutable fit only.
            # These shared-template development checks cannot refit/select it.
            for kind in ("new_contexts", "old_return"):
                budget.before_final()
                if catalog is not None:
                    require_paid_cases()
                result = demo.run_case(
                    input_path,
                    training_context=kind,
                    **(
                        dict(contextual_policy=fitted)
                        if catalog is None
                        else dict(catalog_policy=fitted)
                    ),
                )
                report["applications"][kind] = result
                if not result["complete"] or fitted != pin:
                    raise ValueError("application failed or fitted policy changed")
                if catalog is not None:
                    case_work(result)
                    guard, _ = paid_report_guard(result)
                    paid_cases.append((lambda key=kind: report["applications"][key], result, guard))
                trainer._require_current()
            if not demo.native.same_state(source.snapshot_state(), initial):
                raise ValueError("counterfactual source changed")
            report["source_unchanged"] = True
            report["total_training_work"] = {
                key: sum(
                    r["training_work"][key]
                    for r in [
                        *(row["result"] for row in report["trials"]),
                        *report["applications"].values(),
                    ]
                )
                for key in ("attempted_updates", "attempted_rows", "returned_updates")
            }
            if (
                report["total_training_work"]["attempted_updates"]
                > report["max_total_training_updates"]
            ):
                raise ValueError("total native training envelope exceeded")
            if catalog is not None:
                require_paid_cases()
                results = [
                    *(row["result"] for row in report["trials"]),
                    *report["applications"].values(),
                ]
                report["total_replay_examples"] = sum(r["replay_examples"] for r in results)
                report["total_catalog_work"] = {
                    key: sum(asdict(case_work(r))[key] for r in results)
                    for key in (
                        "attempted_updates",
                        "returned_updates",
                        "attempted_rows",
                        "returned_rows",
                        "replay_examples_charged",
                        "prediction_calls",
                        "prediction_rows",
                    )
                }
            budget.complete_memory()
            budget.before_final()
            report["complete"] = True
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"[:512]
        finally:
            if trainer is not None:
                report["training_state"] = {
                    k: (
                        [asdict(e) for e in v]
                        if k == "rows"
                        else asdict(v)
                        if k == "fitted" and v is not None
                        else asdict(v)
                        if catalog is not None and k in ("catalog", "failed_work") and v is not None
                        else v
                    )
                    for k, v in trainer.snapshot().items()
                }
            current = np.random.get_state()
            assert type(current) is tuple
            report["numpy_rng_unchanged"] = bool(
                rng[0] == current[0]
                and np.array_equal(rng[1], current[1])
                and rng[2:] == current[2:]
            )
            report["torch_rng_unchanged"] = bool(torch.equal(torch_rng, torch.get_rng_state()))
            if not report["numpy_rng_unchanged"] or not report["torch_rng_unchanged"]:
                report.update(complete=False, error="caller RNG changed")
            report["costs"] = dict(
                elapsed_seconds=monotonic() - started,
                sampled_process_rss=asdict(sampler.snapshot()),
            )
    return report
