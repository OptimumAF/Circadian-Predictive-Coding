"""Disposable joint TRAIN LS qualification through existing teacher/task ports.

One fixed solve, original TRAIN diagnosis and actual bounded sandbox episodes.
No native learner, actor, online schedule, selection, final/probe data or IO.
"""

from copy import deepcopy
from dataclasses import asdict
import random
from time import monotonic
from typing import Any

import numpy as np

from scripts.run_ticket_workflow import CONTEXT_FEATURES, CONTEXT_TASKS
from scripts.ticket_train_fit import teacher_fit
from src.adapters.linear_fit_decision import LinearFitDecisionPredictor
from src.adapters.perceptron_decision import build_decision_training_batch
from src.app.coupled_attribution import _same_value
from src.app.ticket_workflow import run_workflow_episode
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.categorical_linear_fit import CategoricalLinearFit, DEFINITION, fit_categorical_linear
from src.core.decision_features import DecisionFeatureSpec
from src.core.ticket_workflow import ACTION_IDS, CATALOG, WorkflowCapabilities, WorkflowState
from src.shared.process_memory import ProcessRssSampler


def _admit_teacher(spec, cases, batch):
    if (
        type(cases) is not tuple
        or len(cases) != 12
        or batch.role != "train"
        or any(case.role != "train" for case in cases)
    ):
        raise ValueError("linear reference requires twelve original TRAIN cases")
    rebuilt = build_decision_training_batch(cases, spec, "context-joint-train", allow_subsets=True)
    if not _same_value(rebuilt, batch):
        raise ValueError("linear reference TRAIN packet differs from its cases")
    if batch.features.shape != (12, 9) or batch.targets.shape != (12, 1):
        raise ValueError("linear reference requires the original D9/K6 packet")


def run_reference(teacher_factory):
    started = monotonic()
    report: dict[str, Any] = dict(
        definition="disposable_context_linear_reference_v1",
        complete=False,
        role="train_fit_reference",
        objective=DEFINITION,
        episodes=[],
        teacher_fit=None,
        final_examples=0,
        probe_updates=0,
        replay_examples=0,
        external_actions=0,
        native_updates=0,
        deployed_updates=0,
        benchmark_updates=0,
        scope="one joint known TRAIN reference; no chronological acquisition/retention/recovery, matched compute, native/CPC/generalization/release claim",
    )
    attempts = returns = solver_rows = 0
    task_attempts = task_returns = 0
    reader = budget = None
    paid_pins: list[Any] = []
    fit_pin = model_record_pin = None
    numpy_rng = deepcopy(np.random.get_state())
    python_rng = random.getstate()
    with ProcessRssSampler(interval_seconds=0.02) as sampler:
        try:
            spec = DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG)
            cases, batch = teacher_factory(spec)
            _admit_teacher(spec, cases, batch)
            input_pin = deepcopy((spec, cases, batch))
            input_owners = tuple(map(id, (spec, cases, batch, batch.features, batch.targets)))
            budget = ToyBudgetSession(
                ToyExecutionBudget(
                    max_training_updates=1, max_wall_seconds=30, max_process_rss_bytes=512 * 1024**2
                ),
                monotonic,
            )
            budget.attach_memory(sampler)
            budget_pin = (
                id(budget.budget),
                budget.budget,
                budget.started_at,
                id(budget.clock),
                id(budget.progress),
                id(budget.process_rss_sampler),
            )
            paid = report["episodes"]
            model = model_pin = None

            def require_source():
                budget.before_final()
                if (
                    input_owners
                    != tuple(map(id, (spec, cases, batch, batch.features, batch.targets)))
                    or not _same_value((spec, cases, batch), input_pin)
                    or not _same_value(np.random.get_state(), numpy_rng)
                    or random.getstate() != python_rng
                    or budget.updates_completed != returns
                    or budget_pin
                    != (
                        id(budget.budget),
                        budget.budget,
                        budget.started_at,
                        id(budget.clock),
                        id(budget.progress),
                        id(budget.process_rss_sampler),
                    )
                ):
                    raise ValueError("linear reference TRAIN/RNG/budget source changed")
                if model is not None and model != model_pin:
                    raise ValueError("linear reference fitted model changed")
                if model_record_pin is not None and (
                    report["model"] is not model_record_pin[0]
                    or not _same_value(report["model"], model_record_pin[1])
                ):
                    raise ValueError("linear reference paid model changed")
                if fit_pin is not None and (
                    report["teacher_fit"] is not fit_pin[0]
                    or not _same_value(report["teacher_fit"], fit_pin[1])
                ):
                    raise ValueError("linear reference paid TRAIN fit changed")
                if (
                    report["episodes"] is not paid
                    or len(paid) != len(paid_pins)
                    or any(
                        row is not original or not _same_value(row, value)
                        for row, (original, value) in zip(paid, paid_pins, strict=True)
                    )
                ):
                    raise ValueError("linear reference paid episodes changed")
                if reader is not None:
                    reader.assert_unchanged()

            require_source()
            budget.before_update()
            attempts += 1
            solver_rows += len(batch.features)
            model = fit_categorical_linear(batch.features, batch.targets, 6)
            returns += 1
            budget.record_update()
            if type(model) is not CategoricalLinearFit or model.fit_rows != 12:
                raise ValueError("linear fit returned the wrong reference model")
            model_pin = deepcopy(model)
            require_source()
            report["model"] = asdict(model)
            model_record_pin = (report["model"], deepcopy(report["model"]))
            reader = LinearFitDecisionPredictor(spec, model)
            logits = reader.predict_logits(batch.features)
            require_source()
            report["teacher_fit"] = teacher_fit(logits, cases, batch, tasks=CONTEXT_TASKS)
            fit_pin = (report["teacher_fit"], deepcopy(report["teacher_fit"]))

            def predict(request):
                require_source()
                result = reader(request)
                require_source()
                return result

            for version, concept in CONTEXT_TASKS:
                for scenario in ("ordinary", "stale_once", "revoke_commit"):
                    require_source()
                    row: dict[str, Any] = dict(
                        policy_version=version,
                        concept=concept,
                        scenario=scenario,
                        role="train_task_readout",
                        attempted_tasks=1,
                        returned_tasks=0,
                        accepted=False,
                        unreturned_work_unknown=True,
                        episode=None,
                    )
                    pending_pin = deepcopy(row)
                    paid.append(row)
                    paid_pins.append((row, deepcopy(row)))
                    task_attempts += 1
                    episode = run_workflow_episode(
                        WorkflowState(
                            f"linear-{version}-{concept}", concept, policy_version=version
                        ),
                        predict,
                        capabilities=WorkflowCapabilities(ACTION_IDS),
                        max_steps=6,
                        scenario=scenario,
                        include_policy_context=True,
                    )
                    task_returns += 1
                    returned = dict(
                        **pending_pin,
                    )
                    returned.update(
                        returned_tasks=1, unreturned_work_unknown=False, episode=deepcopy(episode)
                    )
                    # Capture paid return BEFORE a post-task guard can fail. A
                    # changed pending envelope never becomes accepted evidence.
                    paid_pins[-1] = (row, deepcopy(returned))
                    if (
                        report["episodes"] is not paid
                        or paid[-1] is not row
                        or not _same_value(row, pending_pin)
                    ):
                        raise ValueError("pending linear task metadata changed")
                    row.update(returned)
                    require_source()
                    if not episode["complete"]:
                        raise RuntimeError("linear task has uncertain paid work")
                    row["accepted"] = True
                    paid_pins[-1] = (row, deepcopy(row))
            require_source()
            report["complete"] = True
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"[:512]
        finally:
            if not report["complete"]:
                report["preserved_paid_evidence"] = dict(
                    episodes=[deepcopy(value) for _, value in paid_pins],
                    teacher_fit=None if fit_pin is None else deepcopy(fit_pin[1]),
                    model=None if model_record_pin is None else deepcopy(model_record_pin[1]),
                )
            report["fit_work"] = dict(
                attempted_fits=attempts,
                returned_fits=returns,
                rows_attempted=solver_rows,
                unreturned_work_unknown=attempts != returns,
            )
            report["task_work"] = dict(
                attempted_tasks=task_attempts,
                returned_tasks=task_returns,
                unreturned_work_unknown=task_attempts != task_returns,
            )
            if reader is not None:
                report["readout_work"] = asdict(reader.work)
                report["readout_work_unknown"] = (
                    reader.work.affine_attempts != reader.work.affine_returns
                    or reader.work.decision_attempts != reader.work.decision_returns
                )
            if budget is not None:
                report["budget"] = dict(
                    limits=asdict(budget.budget),
                    started_at=budget.started_at,
                    last_clock=budget.last_clock,
                    committed_fits=budget.updates_completed,
                )
            report["numpy_rng_unchanged"] = _same_value(np.random.get_state(), numpy_rng)
            report["python_rng_unchanged"] = random.getstate() == python_rng
            if not report["numpy_rng_unchanged"] or not report["python_rng_unchanged"]:
                report.update(complete=False, error="linear qualification changed caller RNG")
            report["costs"] = dict(
                elapsed_seconds=monotonic() - started,
                sampled_process_rss=asdict(sampler.snapshot()),
                fit_design_shape=[12, 10],
                targets_shape=[12, 6],
                coefficient_bytes=480,
                qualification="one minimum-norm LAPACK solve; coefficients/read nominal terms are not FLOPs or matched native compute",
            )
    return report
