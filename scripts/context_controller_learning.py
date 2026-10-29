"""Four TRAIN-only disposable trials and one frozen contextual application.

Compose existing counterfactual/controller/stream ports. Classifier fit readouts
are not task utility; the later live application cannot fit or select policies.
No final/probe data, new controller algorithm or persistent runtime belongs here.
"""

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from time import monotonic
from types import SimpleNamespace
from typing import Any

import numpy as np

from scripts import context_head_live as live
from scripts.context_head_probe import HEAD_SOURCE_SHA, _source_bytes, head_states
from scripts.context_head_serving import prepare_data, _episodes, TASKS
from scripts.run_bp_ticket_serving import _reference_schedule, _violation_count
from scripts.run_ticket_workflow import CONTEXT_FEATURES
from src.adapters.context_perceptron import ContextPerceptronLearner
from src.adapters.context_serving import snapshot_digest, ContextServingPredictor
from src.app.actor_shadow import StableActor
from src.app.counterfactual_controller import CounterfactualController, CounterfactualReturn
from src.app.coupled_attribution import _same_value
from src.app.probe_return_phase import paid_report_guard
from src.app.serving_promotion import PromotableActor
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.contextual_controller import ContextualChoice
from src.core.controller_objective import ControllerMeasures, ControllerObjective
from src.core.decision_features import DecisionFeatureSpec
from src.core.experience import LogicalClock
from src.core.reference_controller import ArrivedTrainingFeedback, ReferenceObservation
from src.core.serving_ports import ServingConfiguration
from src.core.ticket_workflow import CATALOG
from src.shared.process_memory import ProcessRssSampler


def _correct(learner, guards):
    return [
        int(np.sum(np.argmax(learner.predict(g.features), axis=1) == g.targets[:, 0]))
        for g in guards
    ]


def _task_counts(rows, phase, native_updates):
    """Admit exact private development tasks before reading outcome metrics."""
    if type(rows) is not list or len(rows) != 6 or phase not in ("before", "after"):
        raise ValueError("all six original development task outcomes required")
    for row, (version, concept) in zip(rows, TASKS, strict=True):
        if (
            row.get("role") != "controller_train"
            or row.get("phase") != phase
            or row.get("policy_version") != version
            or row.get("concept") != concept
            or row.get("complete") is not True
        ):
            raise ValueError("changed/incomplete development role or task schedule")
        episode = row["episode"]
        if (
            episode.get("complete") is not True
            or type(episode.get("success")) is not bool
            or episode.get("scenario") != "ordinary"
            or episode["initial_state"].get("policy_version") != version
            or episode["initial_state"].get("concept") != concept
            or episode.get("training_updates") != 0
            or episode.get("external_actions") != 0
            or episode.get("unreturned_prediction_work_unknown") is not False
            or episode.get("unvalidated_return_work_unknown") is not False
            or any(
                episode["work"][name] != len(row["reads"])
                for name in (
                    "prediction_calls_attempted",
                    "prediction_calls_returned",
                    "prediction_calls_validated",
                )
            )
            or not row["reads"]
            or any(
                read.get("generation") != 0
                or type(read.get("generation")) is not int
                or read.get("actor_version") != f"controller-{phase}"
                or read.get("completed_updates") != native_updates
                for read in row["reads"]
            )
        ):
            raise ValueError("unknown/changed private task readout has no utility")
    return [
        sum(r["episode"]["success"] for r in rows[4:]),
        sum(r["episode"]["success"] for r in rows[:4]),
    ]


def _task_readout(state, phase, require, report):
    """Existing typed workflows on a private frozen actor, never a deployment."""
    rows = report["task_outcomes"][phase]
    learner = ContextPerceptronLearner(state.policy)
    learner.restore_state(state)
    actor = PromotableActor(
        learner,
        version=f"controller-{phase}",
        configuration=ServingConfiguration(3, 2),
        feature_digest=lambda x: sha256(x.tobytes()).hexdigest(),
        metadata={"role": "controller_train"},
    )
    predictor = ContextServingPredictor(actor, DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG))
    report["task_readout_work"][phase] = predictor.work
    pin = deepcopy(state)

    def check():
        require()
        if state != pin or actor.serving_snapshot().model_state != pin:
            raise ValueError("private controller task state changed")

    def serve(request):
        if sum(w["attempts"] for w in report["task_readout_work"].values()) >= 64:
            raise ValueError("original per-trial typed development allowance exhausted")
        return predictor.serve(request)

    _episodes(SimpleNamespace(serve=serve), phase, check, rows, role="controller_train")
    return _task_counts(rows, phase, state.completed_updates)


def _task_measures(result):
    if (
        result.get("complete") is not True
        or result.get("role") != "controller_train"
        or any(result.get(k) != 0 for k in ("final_examples", "probe_updates", "replay_examples"))
        or type(result.get("task_outcomes")) is not dict
        or set(result["task_outcomes"]) != {"before", "after"}
        or type(result.get("task_readout_work")) is not dict
        or set(result["task_readout_work"]) != {"before", "after"}
    ):
        raise ValueError("incomplete task trial has no utility")
    n = result["training_work"]["returned_updates"]
    if (
        type(n) is not int
        or n not in (0, 1)
        or result["training_work"]["attempted_updates"] != n
        or result["budget"]["updates"] != n
        or len(result["native_calls"]) != n
        or any(c.get("complete") is not True for c in result["native_calls"])
        or result.get("source_unchanged") is not True
        or result.get("actor_unchanged") is not True
    ):
        raise ValueError("unknown or incomplete task trial model work")
    before = _task_counts(result["task_outcomes"]["before"], "before", 12)
    after = _task_counts(result["task_outcomes"]["after"], "after", 12 + n)
    for phase in ("before", "after"):
        work = result["task_readout_work"][phase]
        count = sum(len(r["reads"]) for r in result["task_outcomes"][phase])
        if any(work[k] != count for k in ("attempts", "returns", "readouts", "typed_attempts")):
            raise ValueError("unknown or incomplete task trial readout work")
    if before != result["before"] or after != result["after"]:
        raise ValueError("task counts differ from complete development episodes")
    episodes = [
        r["episode"] for phase in ("before", "after") for r in result["task_outcomes"][phase]
    ]
    return ControllerMeasures(
        before[0],
        after[0],
        2,
        before[1],
        after[1],
        4,
        result["elapsed_seconds"],
        result["memory"]["peak_bytes"],
        _violation_count(episodes),
        True,
    )


def _trial(initial, batch, guards, action, require, *, task_outcomes=False):
    """One original causal TRAIN stream; retain attempted work on every failure."""
    if type(task_outcomes) is not bool:
        raise ValueError("task trial opt-in must be an exact boolean")
    if task_outcomes and (batch.role != "train" or initial.completed_updates != 12):
        raise ValueError("task trial requires ordinary TRAIN and original qualified source")
    started = monotonic()
    report: dict[str, Any] = dict(
        complete=False,
        native_calls=[],
        training_work=dict(attempted_updates=0, attempted_rows=0, returned_updates=0),
    )
    if task_outcomes:
        report.update(
            task_outcomes=dict(before=[], after=[]),
            task_readout_work={},
            role="controller_train",
            final_examples=0,
            probe_updates=0,
            replay_examples=0,
        )
    runtime = budget = None
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        try:
            require()

            class Counted(ContextPerceptronLearner):
                def fork(self):
                    child = Counted(self.policy)
                    child.restore_state(self.snapshot_state())
                    return child

                def train_batch(self, features, targets):
                    require()
                    call: dict[str, Any] = dict(rows=len(features), complete=False)
                    report["native_calls"].append(call)
                    report["training_work"]["attempted_updates"] += 1
                    report["training_work"]["attempted_rows"] += len(features)
                    result = super().train_batch(features, targets)
                    call.update(complete=True, state=asdict(self.snapshot_state()))
                    report["training_work"]["returned_updates"] += 1
                    return result

            source = Counted(initial.policy)
            source.restore_state(initial)
            actor = StableActor(source, version="qualified-12")
            budget = ToyBudgetSession(
                ToyExecutionBudget(
                    max_training_updates=1, max_wall_seconds=30, max_process_rss_bytes=512 * 1024**2
                ),
                monotonic,
            )
            budget.attach_memory(memory)
            budget_pin = (id(budget.budget), budget.budget, budget.started_at, id(budget.clock))

            def check():
                require()
                budget.before_final()
                if source.snapshot_state() != initial or actor.snapshot_state().state != initial:
                    raise ValueError("disposable trial changed original source/actor")
                if budget_pin != (
                    id(budget.budget),
                    budget.budget,
                    budget.started_at,
                    id(budget.clock),
                ):
                    raise ValueError("original trial budget changed")

            before = (
                _task_readout(initial, "before", check, report)
                if task_outcomes
                else _correct(source, guards)
            )
            runtime, _ = _reference_schedule(
                source,
                actor,
                (batch,),
                budget,
                "periodic",
                report,
                check,
                lambda: None,
                contextual_policy=ContextualChoice(action, action),
            )
            state = runtime.candidate_snapshot().state
            if task_outcomes:
                after = _task_readout(state, "after", check, report)
            else:
                candidate = ContextPerceptronLearner(initial.policy)
                candidate.restore_state(state)
                after = _correct(candidate, guards)
            check()
            expected = int(action == "wake1")
            if (
                budget.updates_completed != expected
                or report["training_work"]["returned_updates"] != expected
            ):
                raise ValueError("disposable trial did not finish its declared finite action")
            report.update(
                before=before,
                after=after,
                candidate_state=asdict(state),
                source_unchanged=True,
                actor_unchanged=True,
                complete=True,
            )
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"[:512]
        finally:
            if runtime is not None:
                report["applied_updates"] = [asdict(u) for u in runtime.applied_updates]
            if budget is not None:
                report["budget"] = dict(
                    limits=asdict(budget.budget),
                    updates=budget.updates_completed,
                    started_at=budget.started_at,
                    last_clock=budget.last_clock,
                )
            report.update(elapsed_seconds=monotonic() - started, memory=asdict(memory.snapshot()))
    return report


def run_study(input_path, *, task_outcomes=False, catalog=False):
    """Freeze four TRAIN estimates before application, never refit from its tasks."""
    if type(task_outcomes) is not bool:
        raise ValueError("task study opt-in must be an exact boolean")
    if type(catalog) is not bool:
        raise ValueError("catalog study opt-in must be an exact boolean")
    if catalog:
        if task_outcomes:
            raise ValueError("catalog mode has its own fixed task readout protocol")
        from scripts.context_catalog_learning import run_catalog_study

        return run_catalog_study(input_path)
    started = monotonic()
    report: dict[str, Any] = dict(
        definition="context_controller_learning_v1",
        complete=False,
        source_sha256=HEAD_SOURCE_SHA,
        prior_training=dict(updates=12, rows=72, mistakes=16),
        objective=asdict(ControllerObjective()),
        trials=[],
        monitoring=[],
        final_examples=0,
        probe_updates=0,
        replay_examples=0,
        fit_metric="TRAIN-derived classifier accuracy new6/old12; not external task utility",
        application_role="development application only; never refit/selection",
    )
    if task_outcomes:
        report.update(
            definition="context_controller_task_trials_v1",
            fit_metric="controller_train closed-loop ordinary tasks new2/old4; known templates, not heldout",
            application_role="not invoked; separate prospective validation required",
            task_protocol=dict(
                role="controller_train", tasks=TASKS, scenario="ordinary", new_count=2, old_count=4
            ),
        )
    trainer = None
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        try:
            raw = _source_bytes(input_path)
            if sha256(raw).hexdigest() != HEAD_SOURCE_SHA:
                raise ValueError("original qualified contextual source replaced")
            initial = head_states(json.loads(raw))[0]
            source = ContextPerceptronLearner(initial.policy)
            source.restore_state(initial)
            actor = StableActor(source, version="qualified-12")
            data = prepare_data(DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG))
            data_pin, rng = deepcopy(data), deepcopy(np.random.get_state())
            clock = LogicalClock()
            trainer = CounterfactualController(
                actor.snapshot_state, snapshot_digest, ControllerObjective(), clock
            )
            rows = report["trials"]
            paid: list[Any] = []

            def check():
                if (
                    monotonic() - started > (180 if task_outcomes else 240)
                    or memory.snapshot().peak_bytes > 512 * 1024**2
                ):
                    raise ValueError("original fit/application allowance exhausted")
                if _source_bytes(input_path) != raw or not _same_value(data, data_pin):
                    raise ValueError("original source/TRAIN data changed")
                if source.snapshot_state() != initial or not _same_value(
                    np.random.get_state(), rng
                ):
                    raise ValueError("original source owner or RNG changed")
                trainer._require_current()
                if report["trials"] is not rows or len(rows) != len(paid):
                    report["trials"] = rows
                    rows[:] = [r for r, _, _ in paid]
                    raise ValueError("preceding paid trial container replaced")
                for i, (row, expected, guard) in enumerate(paid):
                    if rows[i] is not row:
                        rows[:] = [r for r, _, _ in paid]
                        raise ValueError("preceding paid trial replaced")
                    try:
                        guard()
                    except ValueError:
                        row.clear()
                        row.update(deepcopy(expected))
                        raise ValueError("preceding paid trial corrupted") from None

            for batch in data[0]:
                check()
                observed = clock.now()
                prediction = actor.predict(batch.features)
                clock.advance_to(observed + 1)
                feedback = ArrivedTrainingFeedback(
                    "train",
                    batch.context,
                    prediction.actor_version,
                    observed,
                    clock.now(),
                    int(np.sum(np.argmax(prediction.prediction, axis=1) != batch.targets[:, 0])),
                    len(batch.features),
                )
                report["monitoring"].append(asdict(feedback))
                for _ in range(2):
                    check()
                    observation = ReferenceObservation(
                        clock.now(), 1, 1, 1, actor.version, feedback
                    )

                    def trial(request, state):
                        check()
                        if snapshot_digest(state) != snapshot_digest(initial):
                            raise ValueError(
                                "counterfactual trial requires complete original source"
                            )
                        result = _trial(
                            state,
                            batch,
                            data[1],
                            request.action_id,
                            check,
                            **(dict(task_outcomes=True) if task_outcomes else {}),
                        )
                        row = dict(request=asdict(request), result=result)
                        rows.append(row)
                        if result.get("complete") is True:
                            if task_outcomes:
                                measures = _task_measures(result)
                            else:
                                before, after = result["before"], result["after"]
                                measures = ControllerMeasures(
                                    before[0],
                                    after[0],
                                    6,
                                    before[1],
                                    after[1],
                                    12,
                                    result["elapsed_seconds"],
                                    result["memory"]["peak_bytes"],
                                    0,
                                    True,
                                )
                            clock.advance_to(request.started_at + 1)
                            row.update(
                                measures=asdict(measures),
                                labels_arrived_at=clock.now(),
                                completed_at=clock.now(),
                            )
                        # Preserve a failed paid result before refusing any later trial.
                        guard, expected = paid_report_guard(row)
                        paid.append((row, expected, guard))
                        if result.get("complete") is not True:
                            raise ValueError(
                                "paid trial incomplete; no following trial/application"
                            )
                        return CounterfactualReturn(request, clock.now(), clock.now(), measures)

                    trainer.run(observation, trial)
            check()
            fitted = trainer.freeze()
            fitted_pin = deepcopy(fitted)
            report.update(fitted_policy=asdict(fitted), fitted_at=clock.now())
            check()
            if not task_outcomes:
                application = live.run_case(input_path, contextual_policy=fitted)
                report["application"] = application
                check()
            if fitted != fitted_pin:
                raise ValueError("original frozen fit changed during application")
            if not task_outcomes and application.get("complete") is not True:
                raise ValueError("fitted application incomplete; no further fitting permitted")
            report.update(source_unchanged=True, complete=True)
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"[:512]
        finally:
            work = [t["result"].get("training_work", {}) for t in report["trials"]]
            if "application" in report:
                work.append(report["application"].get("training_work", {}))
            report["training_work"] = {
                k: sum(w.get(k, 0) for w in work)
                for k in ("attempted_updates", "attempted_rows", "returned_updates")
            }
            if trainer is not None:
                snapshot = trainer.snapshot()
                report["training_state"] = {
                    k: [asdict(e) for e in v]
                    if k == "rows"
                    else asdict(v)
                    if k == "fitted" and v is not None
                    else v
                    for k, v in snapshot.items()
                }
            report.update(elapsed_seconds=monotonic() - started, memory=asdict(memory.snapshot()))
    return report
