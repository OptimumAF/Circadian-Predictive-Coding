"""Ordinary known-TRAIN task execution with a separately paid frozen CE readout.

The existing stream owns arrival, source training and actor authority. This
composition never installs fitted weights or supplies probe/final feedback.
"""

from copy import deepcopy
from dataclasses import asdict
from time import monotonic

from scripts import ticket_workflow_native as native
from scripts.ticket_ce_probe import _receipt_guard
from scripts.run_ticket_workflow import CONTEXT_FEATURES, CONTEXT_TASKS, training_cases
from scripts.ticket_train_fit import JOINT_TRAIN_TASKS, joint_old_fresh_teacher
from src.adapters.torch_categorical_ce_readout import (
    CategoricalCEFitFailed,
    CategoricalCEReadoutPredictor,
    fit_categorical_ce_readout,
)
from src.adapters.torch_categorical_readout_fit import _unchanged_owner
from src.app.probe_return_phase import paid_report_guard
from src.app.ticket_workflow import run_workflow_episode
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.decision_features import DecisionFeatureSpec
from src.core.ticket_workflow import ACTION_IDS, CATALOG, WorkflowCapabilities, WorkflowState

LIMITS = ToyExecutionBudget(max_wall_seconds=15, max_process_rss_bytes=512 * 1024**2)


def _finished_snapshot(stream, finished, report):
    """Seal one observed CE-source finish for read-only final validation.

    Why: candidate_snapshot deliberately refuses finished streams. Keep that
    public boundary intact and use the same runtime snapshot under its lock,
    bound to the successful finish and retained source owner guards.
    """
    runtime, candidate = stream._runtime, stream._runtime._candidate
    guard, _ = paid_report_guard(finished)

    def unchanged():
        guard()
        if (
            report.get("stream") is not finished
            or stream._runtime is not runtime
            or runtime._candidate is not candidate
            or stream._finished is not True
            or stream._failed is not False
        ):
            raise ValueError("CE finished source binding changed")

    def snapshot():
        unchanged()
        with stream._exclusive():
            unchanged()
            stream._check_original()
            state = runtime.candidate_snapshot().state
            stream._check_original()
            unchanged()
        return state

    snapshot()
    return snapshot


def _admit(stream, state, spec, report, budget, memory):
    if (
        report.get("train_readout_requested") != "ce"
        or report.get("definition") != "native_context_return_ticket_workflow_v1"
        or report.get("method") != "bp"
        or report.get("source_unchanged") is not True
        or report.get("actor_unchanged") is not True
        or type(report.get("final_examples")) is not int
        or report["final_examples"] != 0
        or type(report.get("probe_updates")) is not int
        or report["probe_updates"] != 0
        or type(report.get("replay_examples")) is not int
        or report["replay_examples"] != 0
        or report.get("train_readout_source_training")
        != dict(completed_updates=12, applied_update_count=12)
        or state.policy.method != "bp"
        or (state.policy.feature_dim, state.policy.hidden_dim, state.policy.num_classes)
        != (9, 3, 6)
        or (state.completed_updates, state.training_examples) != (12, 72)
        or type(spec) is not DecisionFeatureSpec
        or spec != DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG)
        or type(budget) is not ToyBudgetSession
        or stream._budget is not budget
        or budget.updates_completed != 12
        or budget.replay_examples_completed != 0
        or budget.process_rss_sampler is not memory
        or memory is None
        or stream.completed_updates != 12
        or len(stream.applied_updates) != 12
    ):
        raise ValueError("CE tasks require the original completed ordinary BP checkpoint12")


def run_train_readout(
    stream, state, spec, report, budget, *, check_boundary, memory, row, cohort=None
):
    """Fit once after source success, then run all fixed tasks without promotion."""
    if cohort is not None and (type(cohort) is not str or cohort != "joint"):
        raise ValueError("joint TRAIN CE readout requires an exact joint cohort")
    if (
        report.get("train_readout_cohort_requested") != cohort
        or type(report.get("train_readout_cohort_requested")) is not type(cohort)
        or ("train_readout_cohort_requested" in report) != (cohort is not None)
    ):
        raise ValueError("TRAIN CE readout cohort intent differs")
    if type(row) is not dict or row or not callable(check_boundary):
        raise ValueError("CE tasks require a fresh report and source boundary")
    check_boundary()
    _admit(stream, state, spec, report, budget, memory)
    source = stream._runtime._candidate
    runtime = stream._runtime
    receipt_guard = _receipt_guard(stream)
    original = (budget.budget, budget.clock, budget.progress, memory)
    limits, birth = deepcopy(budget.budget), budget.started_at
    source_tag = report["train_readout_source_training"]
    tag_pin = deepcopy(source_tag)
    state_pin, spec_pin = deepcopy(state), deepcopy(spec)
    aux = ToyBudgetSession(LIMITS, monotonic)
    aux.attach_memory(memory)
    aux_binding = (aux.budget, aux.clock, aux.progress, aux.process_rss_sampler)
    aux_limits, aux_birth = deepcopy(aux.budget), aux.started_at
    row.update(
        definition="ordinary_train_ce_tasks_v1",
        complete=False,
        source_updates=12,
        source_rows=72,
        final_examples=0,
        probe_updates=0,
        work={},
        episodes=[],
        scope="fixed known TRAIN tasks; separate auxiliary fitting, no actor promotion, generalization or matched computation claim",
    )
    task_order = JOINT_TRAIN_TASKS if cohort == "joint" else CONTEXT_TASKS
    if cohort == "joint":
        # Why: bind the expanded TRAIN input and task order to the same paid
        # evidence guard; defaults retain the original omitted metadata.
        row.update(
            training_cohort="joint",
            training_case_count=18,
            declared_tasks=[list(task) for task in task_order],
        )
    guard, paid = paid_report_guard(row)
    reader: CategoricalCEReadoutPredictor | None = None
    fit = fit_pin = fit_binding = fit_work = None
    budget_guard = lambda: None
    budget_record = None
    paid_budget = None

    def remember():
        nonlocal guard, paid
        guard, paid = paid_report_guard(row)

    def charge(key):
        guard()
        row["work"][key] = row["work"].get(key, 0) + 1
        remember()

    def update(**values):
        guard()
        row.update(values)
        remember()

    with _unchanged_owner(source) as (actual, owner_unchanged):
        if not native.same_state(actual, state_pin):
            raise ValueError("CE task checkpoint differs from actual source")

        def require(condition, name):
            if not condition:
                raise ValueError("CE tasks changed: " + name)

        def unchanged():
            guard()
            budget_guard()
            owner_unchanged()
            receipt_guard()
            # Why: diagnose the first failed original predicate without dropping
            # checks or evaluating later predicates after a refusal.
            require(stream._runtime is runtime, "runtime owner")
            require(runtime._candidate is source, "candidate owner")
            require(stream._budget is budget, "source budget owner")
            require(report.get("train_readout") is row, "task report owner")
            if budget_record is not None:
                require(
                    report.get("train_readout_budget") is budget_record,
                    "auxiliary budget report owner",
                )
            require(report.get("train_readout_requested") == "ce", "readout intent")
            require(
                report.get("train_readout_cohort_requested") == cohort
                and type(report.get("train_readout_cohort_requested")) is type(cohort)
                and ("train_readout_cohort_requested" in report) == (cohort is not None),
                "readout cohort intent",
            )
            require(report.get("source_unchanged") is True, "source preservation flag")
            require(report.get("actor_unchanged") is True, "actor preservation flag")
            for key in ("final_examples", "probe_updates", "replay_examples"):
                require(type(report.get(key)) is int and report[key] == 0, key)
            require(
                report.get("train_readout_source_training") is source_tag,
                "source training record owner",
            )
            require(native.same_state(source_tag, tag_pin), "source training record value")
            require(native.same_state(state, state_pin), "source checkpoint value")
            require(spec == spec_pin, "feature schema")
            if fit is not None:
                require(native.same_state(fit, fit_pin), "fit value")
                require(native.same_state(fit.source, state_pin), "fit source value")
                require(fit.spec == spec_pin, "fit feature schema")
                for name, actual_owner, expected_owner in zip(
                    ("source", "schema", "coefficients", "work"),
                    (fit.source, fit.spec, fit.coefficients, fit.work),
                    fit_binding,
                    strict=True,
                ):
                    require(actual_owner is expected_owner, "fit " + name + " owner")
            require(budget.budget == limits, "source budget limits")
            require(budget.started_at == birth, "source budget birth")
            require(budget.updates_completed == 12, "source completed updates")
            require(budget.replay_examples_completed == 0, "source replay charge")
            for name, actual_owner, expected_owner in zip(
                ("limits", "clock", "progress", "memory"),
                (budget.budget, budget.clock, budget.progress, budget.process_rss_sampler),
                original,
                strict=True,
            ):
                require(actual_owner is expected_owner, "source budget " + name + " owner")
            require(aux.budget == aux_limits, "auxiliary budget limits")
            require(aux.started_at == aux_birth, "auxiliary budget birth")
            require(aux.updates_completed == 0, "auxiliary update charge")
            require(aux.replay_examples_completed == 0, "auxiliary replay charge")
            for name, actual_owner, expected_owner in zip(
                ("limits", "clock", "progress", "memory"),
                (aux.budget, aux.clock, aux.progress, aux.process_rss_sampler),
                aux_binding,
                strict=True,
            ):
                require(actual_owner is expected_owner, "auxiliary budget " + name + " owner")
            if reader is not None:
                reader.assert_unchanged()

        def boundary():
            unchanged()
            check_boundary()
            unchanged()
            budget.before_final()
            unchanged()
            aux.before_final()
            unchanged()

        def predict(request):
            assert reader is not None
            boundary()
            try:
                return reader(request)
            finally:
                update(readout_work=asdict(reader.work))
                boundary()

        try:
            boundary()
            cases, _ = (
                joint_old_fresh_teacher(spec, training_cases)
                if cohort == "joint"
                else native.prepare_train_fit(spec)
            )
            boundary()
            charge("fit_attempts")
            try:
                fit = fit_categorical_ce_readout(source, cases, spec, check_boundary=boundary)
            except CategoricalCEFitFailed as error:
                fit_work = deepcopy(error.work)
                update(fit_work=fit_work)
                raise
            fit_binding = (fit.source, fit.spec, fit.coefficients, fit.work)
            fit_pin = deepcopy(fit)
            fit_work = deepcopy(fit.work)
            update(fit_work=fit_work, final_train_ce=fit.final_loss)
            charge("fit_returns")
            boundary()
            charge("reader_attempts")
            reader = CategoricalCEReadoutPredictor(fit)
            charge("reader_returns")
            update(readout_work=asdict(reader.work))
            boundary()
            for version, concept in task_order:
                boundary()
                charge("task_attempts")
                episode = run_workflow_episode(
                    WorkflowState(f"train-ce-{version}-{concept}", concept, policy_version=version),
                    predict,
                    capabilities=WorkflowCapabilities(ACTION_IDS),
                    max_steps=6,
                    include_policy_context=True,
                )
                guard()
                row["episodes"].append(
                    dict(
                        policy_version=version,
                        concept=concept,
                        role="controller_train",
                        episode=episode,
                    )
                )
                remember()
                charge("task_returns")
                boundary()
                if episode.get("complete") is not True:
                    raise ValueError("CE task episode has uncertain paid work")
            boundary()
            update(complete=True, source_unchanged=True)
            boundary()
        except Exception as error:
            preserved = deepcopy(paid)
            preserved.update(complete=False, error=f"{type(error).__name__}: {error}"[:512])
            if fit_work is not None:
                preserved["fit_work"] = deepcopy(fit_work)
            if reader is not None:
                preserved["readout_work"] = asdict(reader.work)
            report["preserved_train_readout"] = preserved
            raise
        finally:
            # The original source clock includes this elapsed time; never add them.
            budget_record = report["train_readout_budget"] = dict(
                limits=asdict(aux_limits),
                started_at=aux_birth,
                elapsed_seconds=monotonic() - aux_birth,
            )
            budget_guard, paid_budget = paid_report_guard(budget_record)

    def retain():
        """Keep paid evidence guarded through source finish and publication."""
        try:
            unchanged()
            check_boundary()
            unchanged()
        except Exception as error:
            preserved = deepcopy(paid)
            preserved.update(complete=False, error=f"{type(error).__name__}: {error}"[:512])
            report["preserved_train_readout"] = preserved
            report["preserved_train_readout_budget"] = deepcopy(paid_budget)
            raise

    retain()
    return row, retain
